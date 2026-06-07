#pragma once

#include "core/result.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>

namespace amind {

/// A task to be executed asynchronously (Stage 2 processing).
struct AsyncTask {
    uint64_t memory_id{0};
    /// Distributed-tracing trace identifier inherited from the synchronous
    /// caller (e.g. Stage 1 fastStore generates a trace_id and threads it
    /// through to Stage 2 here). Used by observability to group all events
    /// of a single pipeline run. 0 = no trace context (legacy / standalone).
    uint64_t trace_id{0};
    std::string task_type;  // "extract", "dedup", "conflict", "graph_link"
    std::function<void()> work;
};

/// Thread-safe bounded task queue for producer-consumer pattern.
/// Used by the two-stage async pipeline to decouple fast writes from slow LLM processing.
class TaskQueue {
public:
    explicit TaskQueue(size_t capacity = 10000);
    ~TaskQueue();

    /// Push a task. Returns error if queue is full or stopped.
    [[nodiscard]] Result<void, Error> push(AsyncTask task);

    /// Pop a task (blocks until available or stopped).
    /// Returns nullopt if stopped and empty.
    [[nodiscard]] std::optional<AsyncTask> pop();

    /// Signal all waiting threads to stop.
    void stop();

    /// Check if stopped.
    [[nodiscard]] bool stopped() const;

    /// Current queue size.
    [[nodiscard]] size_t size() const;

    /// Total tasks processed (popped).
    [[nodiscard]] uint64_t totalProcessed() const;

private:
    size_t capacity_;
    std::queue<AsyncTask> queue_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::atomic<bool> stopped_{false};
    std::atomic<uint64_t> total_processed_{0};
};

}  // namespace amind
