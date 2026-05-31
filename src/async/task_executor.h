#pragma once

#include "task_queue.h"

#include <atomic>
#include <thread>
#include <vector>

namespace amind {

/// Thread pool executor for async Stage 2 tasks.
/// Spawns N worker threads that drain the TaskQueue.
class TaskExecutor {
public:
    explicit TaskExecutor(TaskQueue& queue, size_t num_threads = 4);
    ~TaskExecutor();

    /// Start worker threads.
    void start();

    /// Stop all workers (blocks until all threads join).
    void shutdown();

    [[nodiscard]] bool running() const;
    [[nodiscard]] size_t numThreads() const { return num_threads_; }
    [[nodiscard]] uint64_t tasksCompleted() const;
    [[nodiscard]] uint64_t tasksFailed() const;

private:
    void workerLoop(size_t thread_id);

    TaskQueue& queue_;
    size_t num_threads_;
    std::vector<std::thread> workers_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> tasks_completed_{0};
    std::atomic<uint64_t> tasks_failed_{0};
};

}  // namespace amind
