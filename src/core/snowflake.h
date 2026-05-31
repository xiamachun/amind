#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>

namespace amind {

/// Snowflake ID generator producing time-ordered, globally unique 64-bit IDs.
///
/// Layout (64 bits total):
///   - Bit 63:     unused (sign bit, always 0)
///   - Bits 62-22: timestamp in milliseconds since custom epoch (41 bits, ~69 years)
///   - Bits 21-12: machine/worker ID (10 bits, 0-1023)
///   - Bits 11-0:  sequence number within the same millisecond (12 bits, 0-4095)
///
/// Custom epoch: 2025-01-01T00:00:00Z (Unix ms = 1735689600000)
class SnowflakeGenerator {
public:
    static constexpr int64_t CUSTOM_EPOCH_MS = 1735689600000LL;  // 2025-01-01 UTC
    static constexpr int TIMESTAMP_BITS = 41;
    static constexpr int WORKER_ID_BITS = 10;
    static constexpr int SEQUENCE_BITS = 12;
    static constexpr int64_t MAX_WORKER_ID = (1LL << WORKER_ID_BITS) - 1;  // 1023
    static constexpr int64_t MAX_SEQUENCE = (1LL << SEQUENCE_BITS) - 1;    // 4095

    /// Construct a generator with the given worker ID (0-1023).
    /// Throws std::invalid_argument if worker_id is out of range.
    explicit SnowflakeGenerator(int64_t worker_id);

    /// Generate the next unique ID. Thread-safe.
    uint64_t nextId();

    /// Extract the timestamp (ms since Unix epoch) from a snowflake ID.
    static int64_t extractTimestamp(uint64_t snowflake_id);

    /// Extract the worker ID from a snowflake ID.
    static int64_t extractWorkerId(uint64_t snowflake_id);

    /// Extract the sequence number from a snowflake ID.
    static int64_t extractSequence(uint64_t snowflake_id);

private:
    int64_t currentTimeMs() const;

    int64_t worker_id_;
    std::mutex mutex_;
    int64_t last_timestamp_ms_{-1};
    int64_t sequence_{0};
};

}  // namespace amind
