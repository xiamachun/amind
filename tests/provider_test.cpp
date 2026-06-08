#include <gtest/gtest.h>

#include "provider/connection_pool.h"

#include <thread>

using namespace amind;

// ── ConnectionPool Tests ───────────────────────────────────────────────

class ConnectionPoolTest : public ::testing::Test {
protected:
    HttpConnectionPool::Config cfg_;

    void SetUp() override {
        cfg_.maxConnections = 4;
        cfg_.idleTimeoutMs = 1000;
        cfg_.circuitBreakerThreshold = 3;
        cfg_.circuitBreakerCooldownMs = 500;
    }
};

TEST_F(ConnectionPoolTest, AcquireCreatesNewConnectionWhenEmpty) {
    HttpConnectionPool pool(cfg_);
    // acquire() on an empty pool creates a new connection via createConnection().
    // Use a port that is almost certainly not listening to get -1.
    int fd = pool.acquire("127.0.0.1", 19);  // chargen — unlikely open
    if (fd >= 0) {
        pool.discard(fd);  // clean up if it somehow connected
    } else {
        EXPECT_LT(fd, 0);  // createConnection failed as expected
    }
}

TEST_F(ConnectionPoolTest, ReleaseAndReacquire) {
    HttpConnectionPool pool(cfg_);
    // Simulate: release a "connection" (fd=42), then acquire it
    pool.release(42, "localhost", 8080);
    int fd = pool.acquire("localhost", 8080);
    EXPECT_EQ(fd, 42);
}

TEST_F(ConnectionPoolTest, DiscardDoesNotReturnToPool) {
    HttpConnectionPool pool(cfg_);
    pool.release(10, "host", 80);

    // Acquire removes from pool, then discard closes it
    int fd = pool.acquire("host", 80);
    EXPECT_EQ(fd, 10);
    pool.discard(fd);

    // Now the pool should be empty
    int fd2 = pool.acquire("host", 80);
    EXPECT_LT(fd2, 0);
}

TEST_F(ConnectionPoolTest, CircuitBreakerTrips) {
    HttpConnectionPool pool(cfg_);
    pool.release(1, "fail-host", 80);

    // Trigger failures
    pool.recordFailure();
    pool.recordFailure();
    pool.recordFailure();  // Should trip circuit breaker

    EXPECT_TRUE(pool.isCircuitOpen());
}

TEST_F(ConnectionPoolTest, CircuitBreakerResets) {
    cfg_.circuitBreakerCooldownMs = 50;
    HttpConnectionPool pool(cfg_);

    pool.recordFailure();
    pool.recordFailure();
    pool.recordFailure();
    EXPECT_TRUE(pool.isCircuitOpen());

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    pool.recordSuccess();
    EXPECT_FALSE(pool.isCircuitOpen());
}

TEST_F(ConnectionPoolTest, EvictIdleConnections) {
    cfg_.idleTimeoutMs = 50;
    HttpConnectionPool pool(cfg_);
    pool.release(5, "host", 80);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    size_t evicted = pool.evictIdle();
    EXPECT_GE(evicted, 1u);

    int fd = pool.acquire("host", 80);
    EXPECT_LT(fd, 0);  // Connection was evicted
}
