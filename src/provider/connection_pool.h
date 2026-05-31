#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace amind {

/// HTTP connection pool with keep-alive support, idle eviction, and circuit breaker.
/// Reuses TCP connections to reduce connect/close overhead for Ollama API calls.
class HttpConnectionPool {
public:
    struct Config {
        size_t maxConnections = 8;          // max pooled connections
        size_t idleTimeoutMs = 30000;       // idle connection timeout
        size_t circuitBreakerThreshold = 5; // consecutive failures to trip breaker
        size_t circuitBreakerCooldownMs = 10000; // cooldown before retry
    };

    HttpConnectionPool();
    explicit HttpConnectionPool(Config config);
    ~HttpConnectionPool();

    /// Acquire a connection to host:port. Returns socket fd, or -1 on failure.
    /// If a pooled idle connection exists, reuses it. Otherwise creates new.
    int acquire(const std::string& host, int port);

    /// Release a connection back to the pool for reuse.
    /// If the pool is full, the connection is closed instead.
    void release(int fd, const std::string& host, int port);

    /// Close a connection (on error). Does NOT return to pool.
    void discard(int fd);

    /// Record a successful request (resets circuit breaker).
    void recordSuccess();

    /// Record a failed request (increments circuit breaker counter).
    void recordFailure();

    /// Check if circuit breaker is open (too many consecutive failures).
    bool isCircuitOpen() const;

    /// Evict idle connections that exceeded timeout.
    size_t evictIdle();

    /// Dynamic configuration setters.
    void setIdleTimeout(size_t ms) { std::lock_guard<std::mutex> lk(mutex_); config_.idleTimeoutMs = ms; }
    void setCircuitThreshold(size_t n) { std::lock_guard<std::mutex> lk(mutex_); config_.circuitBreakerThreshold = n; }
    void setCircuitCooldown(size_t ms) { std::lock_guard<std::mutex> lk(mutex_); config_.circuitBreakerCooldownMs = ms; }


    /// Get pool statistics.
    size_t pooledCount() const;
    size_t totalAcquired() const { return total_acquired_.load(); }
    size_t totalReused() const { return total_reused_.load(); }

private:
    struct PooledConnection {
        int fd;
        std::string host;
        int port;
        std::chrono::steady_clock::time_point last_used;
    };

    /// Create a new TCP connection.
    int createConnection(const std::string& host, int port);

    Config config_;
    std::vector<PooledConnection> pool_;
    mutable std::mutex mutex_;

    // Circuit breaker state
    std::atomic<size_t> consecutive_failures_{0};
    std::chrono::steady_clock::time_point circuit_opened_at_;
    std::atomic<bool> circuit_open_{false};

    // Statistics
    std::atomic<size_t> total_acquired_{0};
    std::atomic<size_t> total_reused_{0};
};

}  // namespace amind
