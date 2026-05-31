#include "task_queue.h"

#include <spdlog/spdlog.h>

namespace amind {

TaskQueue::TaskQueue(size_t capacity) : capacity_(capacity) {}

TaskQueue::~TaskQueue() {
    stop();
}

Result<void, Error> TaskQueue::push(AsyncTask task) {
    {
        std::unique_lock lock(mutex_);
        if (stopped_) {
            return makeError(Error::QueueFull, "queue is stopped");
        }
        // Wait for space (with timeout to check stopped_)
        if (!not_full_.wait_for(lock, std::chrono::milliseconds(100),
                                [this] { return queue_.size() < capacity_ || stopped_; })) {
            return makeError(Error::QueueFull, "queue at capacity (" + std::to_string(capacity_) + ")");
        }
        if (stopped_) {
            return makeError(Error::QueueFull, "queue is stopped");
        }
        queue_.push(std::move(task));
    }
    not_empty_.notify_one();
    return Result<void, Error>();
}

std::optional<AsyncTask> TaskQueue::pop() {
    std::unique_lock lock(mutex_);
    not_empty_.wait(lock, [this] { return !queue_.empty() || stopped_; });

    if (queue_.empty()) {
        return std::nullopt;  // stopped and empty
    }

    auto task = std::move(queue_.front());
    queue_.pop();
    total_processed_.fetch_add(1, std::memory_order_relaxed);

    lock.unlock();
    not_full_.notify_one();
    return task;
}

void TaskQueue::stop() {
    stopped_ = true;
    not_empty_.notify_all();
    not_full_.notify_all();
}

bool TaskQueue::stopped() const {
    return stopped_.load(std::memory_order_acquire);
}

size_t TaskQueue::size() const {
    std::lock_guard lock(mutex_);
    return queue_.size();
}

uint64_t TaskQueue::totalProcessed() const {
    return total_processed_.load(std::memory_order_relaxed);
}

}  // namespace amind
