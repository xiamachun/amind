#include "snowflake.h"

#include <stdexcept>
#include <thread>

namespace amind {

SnowflakeGenerator::SnowflakeGenerator(int64_t worker_id)
    : worker_id_(worker_id) {
    if (worker_id < 0 || worker_id > MAX_WORKER_ID) {
        throw std::invalid_argument(
            "Worker ID must be between 0 and " + std::to_string(MAX_WORKER_ID) +
            ", got " + std::to_string(worker_id));
    }
}

uint64_t SnowflakeGenerator::nextId() {
    std::lock_guard<std::mutex> lock(mutex_);

    int64_t timestamp = currentTimeMs();

    if (timestamp < last_timestamp_ms_) {
        // Clock moved backwards — wait until it catches up, but cap the wait
        // to avoid infinite blocking when the clock skew is large (e.g. NTP jump).
        constexpr int64_t kMaxClockWaitMs = 5000;  // 5 seconds maximum
        int64_t waitDeadlineMs = last_timestamp_ms_ + kMaxClockWaitMs;
        while (timestamp <= last_timestamp_ms_) {
            if (currentTimeMs() > waitDeadlineMs) {
                throw std::runtime_error(
                    "Snowflake: clock moved backwards by more than " +
                    std::to_string(kMaxClockWaitMs) +
                    "ms (last=" + std::to_string(last_timestamp_ms_) +
                    ", now=" + std::to_string(timestamp) + ")");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            timestamp = currentTimeMs();
        }
    }

    if (timestamp == last_timestamp_ms_) {
        sequence_ = (sequence_ + 1) & MAX_SEQUENCE;
        if (sequence_ == 0) {
            // Sequence exhausted in this millisecond — wait for next ms
            while (timestamp <= last_timestamp_ms_) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                timestamp = currentTimeMs();
            }
        }
    } else {
        sequence_ = 0;
    }

    last_timestamp_ms_ = timestamp;

    int64_t elapsed = timestamp - CUSTOM_EPOCH_MS;

    return (static_cast<uint64_t>(elapsed) << (WORKER_ID_BITS + SEQUENCE_BITS))
         | (static_cast<uint64_t>(worker_id_) << SEQUENCE_BITS)
         | static_cast<uint64_t>(sequence_);
}

int64_t SnowflakeGenerator::extractTimestamp(uint64_t snowflake_id) {
    int64_t elapsed = static_cast<int64_t>(
        snowflake_id >> (WORKER_ID_BITS + SEQUENCE_BITS));
    return elapsed + CUSTOM_EPOCH_MS;
}

int64_t SnowflakeGenerator::extractWorkerId(uint64_t snowflake_id) {
    return static_cast<int64_t>(
        (snowflake_id >> SEQUENCE_BITS) & MAX_WORKER_ID);
}

int64_t SnowflakeGenerator::extractSequence(uint64_t snowflake_id) {
    return static_cast<int64_t>(snowflake_id & MAX_SEQUENCE);
}

int64_t SnowflakeGenerator::currentTimeMs() const {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

}  // namespace amind
