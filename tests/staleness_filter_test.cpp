#include "retrieval/staleness_filter.h"
#include "core/memory_record.h"

#include <gtest/gtest.h>
#include <atomic>
#include <filesystem>

using namespace amind;

// testDir() helper removed in Phase 4 — no test in this file uses it any more
// (it used to create per-test directories for the deleted StaleLog).

TEST(StalenessFilter, FiltersAggregateWithMissingNewerId) {
    AggregateStalenessFilter::Config cfg;
    cfg.enabled = true;
    AggregateStalenessFilter filter(cfg);

    MemoryRecord agg;
    agg.memory_id = 1;
    agg.content = "已执行工单清单：\n- TKT001 创建用户表\n- TKT002 修改密码\n- TKT003 添加索引";
    agg.created_at = 1000;

    MemoryRecord newer_atomic;
    newer_atomic.memory_id = 2;
    newer_atomic.content = "工单TKT099已执行成功";
    newer_atomic.created_at = 2000;

    MemoryRecord older_unrelated;
    older_unrelated.memory_id = 3;
    older_unrelated.content = "工单TKT001是 DDL 变更";
    older_unrelated.created_at = 500;

    std::vector<AggregateStalenessFilter::ScoredCandidate> cands = {
        {agg.memory_id, 0.9f, &agg},
        {newer_atomic.memory_id, 0.6f, &newer_atomic},
        {older_unrelated.memory_id, 0.5f, &older_unrelated},
    };

    // Filter logic is what we test here; the audit-log destination is
    // exercised by tests/memory_event_log_test.cpp + integration tests now.
    size_t dropped = filter.apply(cands, "哪些工单", "test-ns", nullptr);

    EXPECT_EQ(dropped, 1u);
    EXPECT_EQ(cands.size(), 2u);
    for (const auto& c : cands) {
        EXPECT_NE(c.memory_id, agg.memory_id);
    }
}

TEST(StalenessFilter, KeepsAggregateWhenAllNewerIdsAlreadyEnumerated) {
    AggregateStalenessFilter filter;

    MemoryRecord agg;
    agg.memory_id = 1;
    agg.content = "工单清单：TKT001, TKT002, TKT003";
    agg.created_at = 1000;

    MemoryRecord newer_redundant;
    newer_redundant.memory_id = 2;
    newer_redundant.content = "工单TKT002 详情";  // already in aggregate
    newer_redundant.created_at = 2000;

    std::vector<AggregateStalenessFilter::ScoredCandidate> cands = {
        {agg.memory_id, 0.9f, &agg},
        {newer_redundant.memory_id, 0.6f, &newer_redundant},
    };

    size_t dropped = filter.apply(cands, "工单", "ns", nullptr);
    EXPECT_EQ(dropped, 0u);
    EXPECT_EQ(cands.size(), 2u);
}

TEST(StalenessFilter, IgnoresMemoriesWithoutEnoughIds) {
    AggregateStalenessFilter filter;

    // Single ID, not an aggregate.
    MemoryRecord single;
    single.memory_id = 1;
    single.content = "工单TKT001 已执行";
    single.created_at = 1000;

    MemoryRecord newer_other;
    newer_other.memory_id = 2;
    newer_other.content = "工单TKT099 已执行";
    newer_other.created_at = 2000;

    std::vector<AggregateStalenessFilter::ScoredCandidate> cands = {
        {single.memory_id, 0.9f, &single},
        {newer_other.memory_id, 0.6f, &newer_other},
    };

    EXPECT_EQ(filter.apply(cands, "q", "ns", nullptr), 0u);
}

TEST(StalenessFilter, IdExtractionPatterns) {
    auto ids = AggregateStalenessFilter::extractIds(
        "TKT001 在 ai.users 表新增 phone(VARCHAR(20))，A-1001 工号张三 申请");
    bool found_prefixed = false;
    for (const auto& [pname, list] : ids) {
        if (pname == "prefixed_id") {
            found_prefixed = true;
            EXPECT_GE(list.size(), 2u);
        }
    }
    EXPECT_TRUE(found_prefixed);
}

// StaleLog.AppendAndReplay removed in Phase 4 — equivalent persistence
// coverage now lives in tests/memory_event_log_test.cpp (ReplayRestoresEvents).
