#include "task_executor.h"

#include <spdlog/spdlog.h>

namespace amind {

TaskExecutor::TaskExecutor(TaskQueue& queue, size_t num_threads)
    : queue_(queue), num_threads_(num_threads) {}

TaskExecutor::~TaskExecutor() {
    shutdown();
}

void TaskExecutor::start() {
    if (running_) return;
    running_ = true;
    workers_.reserve(num_threads_);
    for (size_t i = 0; i < num_threads_; ++i) {
        workers_.emplace_back(&TaskExecutor::workerLoop, this, i);
    }
    spdlog::info("TaskExecutor started with {} workers", num_threads_);
}

void TaskExecutor::shutdown() {
    if (!running_) return;
    running_ = false;
    queue_.stop();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
    workers_.clear();
    spdlog::info("TaskExecutor shutdown complete. Completed: {}, Failed: {}",
                 tasks_completed_.load(), tasks_failed_.load());
}

bool TaskExecutor::running() const {
    return running_.load(std::memory_order_acquire);
}

uint64_t TaskExecutor::tasksCompleted() const {
    return tasks_completed_.load(std::memory_order_relaxed);
}

uint64_t TaskExecutor::tasksFailed() const {
    return tasks_failed_.load(std::memory_order_relaxed);
}

void TaskExecutor::workerLoop(size_t thread_id) {
    spdlog::debug("Worker {} started", thread_id);
    while (running_) {
        auto task = queue_.pop();
        if (!task.has_value()) {
            break;  // queue stopped and empty
        }
        try {
            task->work();
            tasks_completed_.fetch_add(1, std::memory_order_relaxed);
        } catch (const std::exception& e) {
            tasks_failed_.fetch_add(1, std::memory_order_relaxed);
            spdlog::error("Worker {} task failed (memory_id={}): {}",
                          thread_id, task->memory_id, e.what());
        }
    }
    spdlog::debug("Worker {} stopped", thread_id);
}

}  // namespace amind
