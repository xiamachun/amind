#include "observability/memory_event_log.h"

#include <gtest/gtest.h>
#include <atomic>
#include <filesystem>
#include <set>

using namespace amind;

namespace {

std::string testDir() {
    static std::atomic<int> seq{0};
    auto dir = std::filesystem::temp_directory_path()
             / ("amind_memevtlog_test_" + std::to_string(::getpid())
                + "_" + std::to_string(++seq));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir.string();
}

MemoryEvent makeEvent(EventKind kind, uint64_t memory_id,
                      uint64_t trace_id, uint64_t ts_ms = 0,
                      std::string ns = "test-ns") {
    MemoryEvent e;
    e.memory_id    = memory_id;
    e.trace_id     = trace_id;
    e.kind         = kind;
    e.status       = EventStatus::Ok;
    e.agent_id     = std::move(ns);
    e.summary      = "synthetic";
    e.timestamp_ms = ts_ms;  // 0 → MemoryEventLog::append will set nowMs()
    return e;
}

}  // namespace

// 1. append: events are persisted + indexed; event_id is auto-assigned.
TEST(MemoryEventLog, AppendAssignsIdsAndIndexes) {
    MemoryEventLog log(testDir());
    ASSERT_TRUE(log.open());

    log.append(makeEvent(EventKind::Store, /*mid=*/100, /*trace=*/1));
    log.append(makeEvent(EventKind::Gate,  /*mid=*/100, /*trace=*/1));
    log.append(makeEvent(EventKind::Reconcile, /*mid=*/100, /*trace=*/1));

    EXPECT_EQ(log.memorySize(), 3u);

    auto by_mem = log.memoryHistory(100);
    EXPECT_EQ(by_mem.size(), 3u);
    // event_id should be auto-assigned via Snowflake (non-zero, distinct).
    std::set<uint64_t> seen;
    for (const auto& e : by_mem) {
        EXPECT_GT(e.event_id, 0u);
        EXPECT_TRUE(seen.insert(e.event_id).second) << "event_id collision";
    }
}

// 2. replay: events written by one instance are visible to a fresh one.
TEST(MemoryEventLog, ReplayRestoresEvents) {
    auto dir = testDir();
    {
        MemoryEventLog log(dir);
        ASSERT_TRUE(log.open());
        log.append(makeEvent(EventKind::Store, 200, 2));
        log.append(makeEvent(EventKind::GcDecay, 200, 3));
    }

    MemoryEventLog log2(dir);
    ASSERT_TRUE(log2.open());
    log2.replay();
    EXPECT_EQ(log2.memorySize(), 2u);

    auto trace2 = log2.trace(2);
    ASSERT_EQ(trace2.size(), 1u);
    EXPECT_EQ(trace2[0].kind, EventKind::Store);
}

// 3. query by memory_id.
TEST(MemoryEventLog, QueryByMemoryId) {
    MemoryEventLog log(testDir());
    log.open();
    log.append(makeEvent(EventKind::Store, 1, 1));
    log.append(makeEvent(EventKind::Store, 2, 2));
    log.append(makeEvent(EventKind::Gate, 1, 1));

    MemoryEventLog::Filter f;
    f.memory_id = 1;
    auto rows = log.query(f);
    EXPECT_EQ(rows.size(), 2u);
    for (const auto& e : rows) EXPECT_EQ(e.memory_id, 1u);
}

// 4. query by trace_id returns chronological tree input.
TEST(MemoryEventLog, TraceReturnsChronological) {
    MemoryEventLog log(testDir());
    log.open();
    // Out-of-order inserts; trace() must reorder by timestamp ASC.
    log.append(makeEvent(EventKind::Reconcile, 10, 42, /*ts=*/3000));
    log.append(makeEvent(EventKind::Store,     10, 42, /*ts=*/1000));
    log.append(makeEvent(EventKind::Gate,      10, 42, /*ts=*/2000));

    auto t = log.trace(42);
    ASSERT_EQ(t.size(), 3u);
    EXPECT_EQ(t[0].kind, EventKind::Store);
    EXPECT_EQ(t[1].kind, EventKind::Gate);
    EXPECT_EQ(t[2].kind, EventKind::Reconcile);
}

// 5. query by kind filter.
TEST(MemoryEventLog, QueryByKind) {
    MemoryEventLog log(testDir());
    log.open();
    log.append(makeEvent(EventKind::Gate,    1, 1));
    log.append(makeEvent(EventKind::GcDecay, 2, 2));
    log.append(makeEvent(EventKind::GcDecay, 3, 3));
    log.append(makeEvent(EventKind::Recall,  4, 4));

    MemoryEventLog::Filter f;
    f.kind = EventKind::GcDecay;
    auto rows = log.query(f);
    EXPECT_EQ(rows.size(), 2u);
    for (const auto& e : rows) EXPECT_EQ(e.kind, EventKind::GcDecay);
}

// 6. since_ms time range filter.
TEST(MemoryEventLog, QueryByTimeRange) {
    MemoryEventLog log(testDir());
    log.open();
    log.append(makeEvent(EventKind::Store, 1, 1, /*ts=*/100));
    log.append(makeEvent(EventKind::Store, 2, 2, /*ts=*/200));
    log.append(makeEvent(EventKind::Store, 3, 3, /*ts=*/300));

    MemoryEventLog::Filter f;
    f.since_ms = 200;
    auto rows = log.query(f);
    EXPECT_EQ(rows.size(), 2u);  // ts=200 and ts=300
}

// 7. ring eviction: oldest events drop when capacity exceeded.
TEST(MemoryEventLog, RingEvictsOldest) {
    MemoryEventLogConfig cfg;
    cfg.ring_capacity = 5;
    cfg.flush_per_append = false;
    MemoryEventLog log(testDir(), cfg);
    log.open();
    for (int i = 0; i < 10; ++i) {
        log.append(makeEvent(EventKind::Store, /*mid=*/static_cast<uint64_t>(i + 1), 1));
    }
    EXPECT_EQ(log.memorySize(), 5u);
    // The first 5 (mid=1..5) should have been evicted.
    MemoryEventLog::Filter f;
    f.memory_id = 1;
    EXPECT_TRUE(log.query(f).empty());
    f.memory_id = 10;
    EXPECT_EQ(log.query(f).size(), 1u);
}

// 8. file rotation: small max_file_bytes triggers .1 rename.
TEST(MemoryEventLog, RotatesFileWhenSizeExceeded) {
    auto dir = testDir();
    MemoryEventLogConfig cfg;
    cfg.max_file_bytes = 512;  // tiny, forces rotation
    cfg.max_rotated_files = 3;
    cfg.flush_per_append = true;
    MemoryEventLog log(dir, cfg);
    log.open();

    // Each event JSONL line is ~200B; ~5 events trigger rotation.
    for (int i = 0; i < 20; ++i) {
        auto e = makeEvent(EventKind::Store, /*mid=*/static_cast<uint64_t>(i + 1), 1);
        e.summary = std::string(80, 'x');  // pad to ensure size
        log.append(std::move(e));
    }
    EXPECT_TRUE(std::filesystem::exists(dir + "/events.log"));
    EXPECT_TRUE(std::filesystem::exists(dir + "/events.log.1"));
}
