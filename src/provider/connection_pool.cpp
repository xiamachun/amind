#include "connection_pool.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <netdb.h>
#include <spdlog/spdlog.h>
#include <sys/socket.h>
#include <unistd.h>

namespace amind {

HttpConnectionPool::HttpConnectionPool() : HttpConnectionPool(Config{}) {}

HttpConnectionPool::HttpConnectionPool(Config config) : config_(std::move(config)) {}

HttpConnectionPool::~HttpConnectionPool() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& conn : pool_) {
        ::close(conn.fd);
    }
    pool_.clear();
}

int HttpConnectionPool::acquire(const std::string& host, int port) {
    // Step 1: Wait for active connection slot (prevents overwhelming Ollama)
    {
        std::unique_lock<std::mutex> lock(active_mutex_);
        if (active_count_.load() >= config_.maxActiveConnections) {
            total_waited_.fetch_add(1);
            auto timeout = std::chrono::milliseconds(config_.activeWaitTimeoutMs);
            bool got_slot = active_cv_.wait_for(lock, timeout, [this] {
                return active_count_.load() < config_.maxActiveConnections;
            });
            if (!got_slot) {
                spdlog::warn("ConnectionPool: active slot timeout after {}ms, active={}/{}",
                            config_.activeWaitTimeoutMs, active_count_.load(), config_.maxActiveConnections);
                return -1;  // timeout, no slot available
            }
        }
        active_count_.fetch_add(1);
    }

    total_acquired_.fetch_add(1);

    // Step 2: Try to reuse pooled connection
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = pool_.begin(); it != pool_.end(); ++it) {
            if (it->host == host && it->port == port) {
                int fd = it->fd;
                pool_.erase(it);
                total_reused_.fetch_add(1);
                spdlog::debug("ConnectionPool: reused fd={} (active={})", fd, active_count_.load());
                return fd;
            }
        }
    }

    // Step 3: Create new connection
    int fd = createConnection(host, port);
    if (fd < 0) {
        // Connection failed, release the active slot
        {
            std::lock_guard<std::mutex> lock(active_mutex_);
            active_count_.fetch_sub(1);
        }
        active_cv_.notify_one();
        return -1;
    }
    spdlog::debug("ConnectionPool: new fd={} (active={})", fd, active_count_.load());
    return fd;
}

void HttpConnectionPool::release(int fd, const std::string& host, int port) {
    // Release active slot first (allows waiting threads to proceed)
    {
        std::lock_guard<std::mutex> lock(active_mutex_);
        active_count_.fetch_sub(1);
    }
    active_cv_.notify_one();

    std::lock_guard<std::mutex> lock(mutex_);
    if (pool_.size() >= config_.maxConnections) {
        ::close(fd);
        return;
    }
    pool_.push_back({fd, host, port, std::chrono::steady_clock::now()});
}

void HttpConnectionPool::discard(int fd) {
    // Release active slot first (allows waiting threads to proceed)
    {
        std::lock_guard<std::mutex> lock(active_mutex_);
        active_count_.fetch_sub(1);
    }
    active_cv_.notify_one();

    ::close(fd);
}

void HttpConnectionPool::recordSuccess() {
    consecutive_failures_.store(0);
    if (circuit_open_.load()) {
        circuit_open_.store(false);
        spdlog::info("ConnectionPool: circuit breaker closed");
    }
}

void HttpConnectionPool::recordFailure() {
    auto failures = consecutive_failures_.fetch_add(1) + 1;
    if (failures >= config_.circuitBreakerThreshold && !circuit_open_.load()) {
        circuit_open_.store(true);
        std::lock_guard<std::mutex> lock(mutex_);
        circuit_opened_at_ = std::chrono::steady_clock::now();
        spdlog::warn("ConnectionPool: circuit breaker OPEN after {} failures", failures);
    }
}

bool HttpConnectionPool::isCircuitOpen() const {
    if (!circuit_open_.load()) return false;
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - circuit_opened_at_).count();
    if (static_cast<size_t>(elapsed) >= config_.circuitBreakerCooldownMs) {
        return false;  // cooldown expired, allow probe
    }
    return true;
}

size_t HttpConnectionPool::evictIdle() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::steady_clock::now();
    size_t evicted = 0;

    pool_.erase(
        std::remove_if(pool_.begin(), pool_.end(),
            [&](const PooledConnection& conn) {
                auto idle = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - conn.last_used).count();
                if (static_cast<size_t>(idle) > config_.idleTimeoutMs) {
                    ::close(conn.fd);
                    evicted++;
                    return true;
                }
                return false;
            }),
        pool_.end());

    if (evicted > 0) spdlog::debug("ConnectionPool: evicted {} idle", evicted);
    return evicted;
}

size_t HttpConnectionPool::pooledCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pool_.size();
}

int HttpConnectionPool::createConnection(const std::string& host, int port) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        spdlog::error("ConnectionPool: socket() failed: {}", strerror(errno));
        return -1;
    }

    int opt = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_KEEPALIVE, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));

    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
        struct addrinfo hints{}, *res;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0) {
            ::close(sockfd);
            return -1;
        }
        std::memcpy(&addr, res->ai_addr, res->ai_addrlen);
        freeaddrinfo(res);
    }

    if (connect(sockfd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(sockfd);
        spdlog::error("ConnectionPool: connect {}:{} failed", host, port);
        return -1;
    }

    spdlog::debug("ConnectionPool: new fd={} to {}:{}", sockfd, host, port);
    return sockfd;
}

}  // namespace amind
