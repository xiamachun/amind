
#include "lineage/lineage_index.h"
#include "lineage/propagator.h"

#include <gtest/gtest.h>
#include <vector>

using namespace amind;

// ═══════════════════════════════════════════════════════════════════════════
// LineageIndex — basic operations
// ═══════════════════════════════════════════════════════════════════════════

TEST(LineageIndexTest, RecordAndGetParents) {
    LineageIndex idx;
    idx.recordLineage(100, {1, 2, 3}, LineageOp::Summarize);

    auto parents = idx.getParents(100);
    ASSERT_EQ(parents.size(), 3u);
    EXPECT_EQ(parents[0], 1u);
    EXPECT_EQ(parents[1], 2u);
    EXPECT_EQ(parents[2], 3u);
}

TEST(LineageIndexTest, GetParentsOfNonExistentReturnsEmpty) {
    LineageIndex idx;
    auto parents = idx.getParents(999);
    EXPECT_TRUE(parents.empty());
}

TEST(LineageIndexTest, ReverseIndexGetChildren) {
    LineageIndex idx;
    idx.recordLineage(100, {1}, LineageOp::Infer);
    idx.recordLineage(101, {1}, LineageOp::Summarize);
    idx.recordLineage(102, {1, 2}, LineageOp::Aggregate);

    auto children = idx.getChildren(1);
    EXPECT_EQ(children.size(), 3u);

    auto children2 = idx.getChildren(2);
    EXPECT_EQ(children2.size(), 1u);
}

TEST(LineageIndexTest, GetLineageOp) {
    LineageIndex idx;
    idx.recordLineage(100, {1}, LineageOp::Summarize);
    idx.recordLineage(101, {2}, LineageOp::Infer);

    EXPECT_EQ(idx.getLineageOp(100), LineageOp::Summarize);
    EXPECT_EQ(idx.getLineageOp(101), LineageOp::Infer);
    EXPECT_EQ(idx.getLineageOp(999), LineageOp::None);
}

TEST(LineageIndexTest, RemoveNodeAsChild) {
    LineageIndex idx;
    idx.recordLineage(100, {1, 2}, LineageOp::Aggregate);

    EXPECT_EQ(idx.childCount(), 1u);
    idx.removeNode(100);
    EXPECT_EQ(idx.childCount(), 0u);

    // Parent reverse index should also be cleaned up
    EXPECT_TRUE(idx.getChildren(1).empty());
    EXPECT_TRUE(idx.getChildren(2).empty());
}

TEST(LineageIndexTest, RemoveNodeAsParent) {
    LineageIndex idx;
    idx.recordLineage(100, {1}, LineageOp::Infer);
    idx.recordLineage(101, {1}, LineageOp::Summarize);

    EXPECT_EQ(idx.getChildren(1).size(), 2u);
    idx.removeNode(1);
    // Reverse index entry removed
    EXPECT_TRUE(idx.getChildren(1).empty());
    // But children's forward records still exist (they remember their parents)
    EXPECT_EQ(idx.childCount(), 2u);
}

TEST(LineageIndexTest, HasLiveParents) {
    LineageIndex idx;
    idx.recordLineage(100, {1, 2}, LineageOp::Aggregate);

    EXPECT_TRUE(idx.hasLiveParents(100));

    idx.removeNode(1);
    EXPECT_TRUE(idx.hasLiveParents(100));  // parent 2 still alive

    idx.removeNode(2);
    EXPECT_FALSE(idx.hasLiveParents(100));  // no live parents
}

TEST(LineageIndexTest, Stats) {
    LineageIndex idx;
    EXPECT_EQ(idx.childCount(), 0u);
    EXPECT_EQ(idx.parentCount(), 0u);
    EXPECT_EQ(idx.totalRecords(), 0u);

    idx.recordLineage(100, {1, 2}, LineageOp::Summarize);
    idx.recordLineage(101, {2, 3}, LineageOp::Aggregate);

    EXPECT_EQ(idx.childCount(), 2u);
    EXPECT_EQ(idx.parentCount(), 3u);  // parents 1, 2, 3
    EXPECT_EQ(idx.totalRecords(), 2u);
}

// ═══════════════════════════════════════════════════════════════════════════
// LineageIndex — transitive descendants
// ═══════════════════════════════════════════════════════════════════════════

TEST(LineageIndexTest, TransitiveDescendantsBFS) {
    LineageIndex idx;
    // Tree: 1 → 10 → 100, 101
    //       1 → 11
    idx.recordLineage(10, {1}, LineageOp::Infer);
    idx.recordLineage(11, {1}, LineageOp::Summarize);
    idx.recordLineage(100, {10}, LineageOp::Aggregate);
    idx.recordLineage(101, {10}, LineageOp::Infer);

    auto descendants = idx.getTransitiveDescendants(1, 5);
    EXPECT_EQ(descendants.size(), 4u);  // 10, 11, 100, 101
}

TEST(LineageIndexTest, TransitiveDescendantsDepthLimit) {
    LineageIndex idx;
    // Chain: 1 → 2 → 3 → 4 → 5
    idx.recordLineage(2, {1}, LineageOp::Infer);
    idx.recordLineage(3, {2}, LineageOp::Infer);
    idx.recordLineage(4, {3}, LineageOp::Infer);
    idx.recordLineage(5, {4}, LineageOp::Infer);

    // Depth 2: should find 2, 3 but not 4, 5
    auto descendants = idx.getTransitiveDescendants(1, 2);
    EXPECT_EQ(descendants.size(), 2u);
}

// ═══════════════════════════════════════════════════════════════════════════
// LineagePropagator
// ═══════════════════════════════════════════════════════════════════════════

TEST(PropagatorTest, PropagateInvalidatesSingleChild) {
    LineageIndex idx;
    idx.recordLineage(100, {1}, LineageOp::Infer);

    LineagePropagator propagator(idx);
    std::vector<uint64_t> invalidated_ids;

    auto results = propagator.propagate(1, [&](uint64_t id) {
        invalidated_ids.push_back(id);
    });

    ASSERT_EQ(results.size(), 1u);
    EXPECT_TRUE(results[0].invalidated);
    EXPECT_EQ(results[0].memory_id, 100u);
    EXPECT_EQ(invalidated_ids.size(), 1u);
    EXPECT_EQ(invalidated_ids[0], 100u);
}

TEST(PropagatorTest, PropagateSkipsChildWithLiveParents) {
    LineageIndex idx;
    idx.recordLineage(100, {1, 2}, LineageOp::Aggregate);

    LineagePropagator propagator(idx);
    std::vector<uint64_t> invalidated_ids;

    auto results = propagator.propagate(1, [&](uint64_t id) {
        invalidated_ids.push_back(id);
    });

    ASSERT_EQ(results.size(), 1u);
    EXPECT_FALSE(results[0].invalidated);  // parent 2 still alive
    EXPECT_TRUE(invalidated_ids.empty());
}

TEST(PropagatorTest, PropagateCascadesDownChain) {
    LineageIndex idx;
    // Chain: 1 → 10 → 100
    idx.recordLineage(10, {1}, LineageOp::Infer);
    idx.recordLineage(100, {10}, LineageOp::Summarize);

    LineagePropagator propagator(idx);
    std::vector<uint64_t> invalidated_ids;

    auto results = propagator.propagate(1, [&](uint64_t id) {
        invalidated_ids.push_back(id);
    });

    // Both 10 and 100 should be invalidated (cascade)
    EXPECT_EQ(invalidated_ids.size(), 2u);
    EXPECT_EQ(invalidated_ids[0], 10u);
    EXPECT_EQ(invalidated_ids[1], 100u);
}

TEST(PropagatorTest, PropagateWithMultipleChildren) {
    LineageIndex idx;
    // Parent 1 has 3 children, all sole-parent
    idx.recordLineage(10, {1}, LineageOp::Infer);
    idx.recordLineage(11, {1}, LineageOp::Summarize);
    idx.recordLineage(12, {1}, LineageOp::Aggregate);

    LineagePropagator propagator(idx);
    std::vector<uint64_t> invalidated_ids;

    auto results = propagator.propagate(1, [&](uint64_t id) {
        invalidated_ids.push_back(id);
    });

    EXPECT_EQ(invalidated_ids.size(), 3u);
}

TEST(PropagatorTest, MaxDepthLimitsConfig) {
    LineagePropagator propagator(*new LineageIndex(), 3);
    EXPECT_EQ(propagator.maxDepth(), 3);
    propagator.setMaxDepth(7);
    EXPECT_EQ(propagator.maxDepth(), 7);
}
