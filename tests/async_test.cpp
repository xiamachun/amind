#include <gtest/gtest.h>

#include "async/task_queue.h"
#include "async/task_executor.h"

#include <atomic>
#include <chrono>
#include <thread>

using namespace amind;

// ── TaskQueue Tests ────────────────────────────────────────────────────

static AsyncTask makeTask(std::function<void()> fn) {
    AsyncTask t;
    t.memory_id = 0;
    t.task_type = "test";
    t.work = std::move(fn);
    return t;
}

TEST(TaskQueueTest, PushAndPop) {
    TaskQueue queue(10);
    std::atomic<int> counter{0};

    (void)queue.push(makeTask([&counter]() { counter++; }));
    (void)queue.push(makeTask([&counter]() { counter += 2; }));

    auto task1 = queue.pop();
    auto task2 = queue.pop();
    ASSERT_TRUE(task1.has_value());
    ASSERT_TRUE(task2.has_value());

    task1->work();
    task2->work();
    EXPECT_EQ(counter.load(), 3);
}

TEST(TaskQueueTest, BoundedCapacity) {
    TaskQueue queue(2);
    (void)queue.push(makeTask([]() {}));
    (void)queue.push(makeTask([]() {}));

    EXPECT_EQ(queue.size(), 2u);
}

TEST(TaskQueueTest, StopWakesWaiters) {
    TaskQueue queue(10);
    queue.stop();

    auto task = queue.pop();
    EXPECT_FALSE(task.has_value());
}

TEST(TaskQueueTest, SizeTracking) {
    TaskQueue queue(100);
    EXPECT_EQ(queue.size(), 0u);
    (void)queue.push(makeTask([]() {}));
    (void)queue.push(makeTask([]() {}));
    EXPECT_EQ(queue.size(), 2u);
    (void)queue.pop();
    EXPECT_EQ(queue.size(), 1u);
}

// ── TaskExecutor Tests ─────────────────────────────────────────────────

TEST(TaskExecutorTest, ConcurrentExecution) {
    TaskQueue queue(100);
    TaskExecutor executor(queue, 4);
    executor.start();

    std::atomic<int> counter{0};
    for (int i = 0; i < 100; ++i) {
        (void)queue.push(makeTask([&counter]() { counter++; }));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    executor.shutdown();

    EXPECT_EQ(counter.load(), 100);
}

TEST(TaskExecutorTest, ShutdownGraceful) {
    TaskQueue queue(10);
    TaskExecutor executor(queue, 2);
    executor.start();

    std::atomic<int> counter{0};
    (void)queue.push(makeTask([&counter]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        counter++;
    }));

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    executor.shutdown();

    EXPECT_EQ(counter.load(), 1);
}

TEST(TaskExecutorTest, MultipleWorkersProcess) {
    TaskQueue queue(1000);
    TaskExecutor executor(queue, 8);
    executor.start();

    std::atomic<int> counter{0};
    for (int i = 0; i < 1000; ++i) {
        (void)queue.push(makeTask([&counter]() { counter.fetch_add(1); }));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    executor.shutdown();

    EXPECT_EQ(counter.load(), 1000);
}
