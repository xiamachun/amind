
#include <gtest/gtest.h>

#include "coordinator/remove_coordinator.h"
#include "forget/forget_engine.h"
#include "lineage/lineage_index.h"

using namespace amind;

class RemoveCoordinatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        lineage_ = std::make_unique<LineageIndex>();
        forget_ = std::make_unique<ForgetEngine>();
        coordinator_ = std::make_unique<RemoveCoordinator>(*lineage_, *forget_);
    }

    // In-memory record store.
    std::unordered_map<uint64_t, MemoryRecord> records_;

    void addRecord(uint64_t id, MemoryPhase phase = MemoryPhase::Active) {
        MemoryRecord record;
        record.memory_id = id;
        record.phase = phase;
        record.content = "Memory " + std::to_string(id);
        records_[id] = std::move(record);
    }

    GetRecordFunc getFunc() {
        return [this](uint64_t id) -> MemoryRecord* {
            auto it = records_.find(id);
            return (it != records_.end()) ? &it->second : nullptr;
        };
    }

    PersistFunc persistFunc() {
        return [](uint64_t, const MemoryRecord&) { /* no-op in test */ };
    }

    std::vector<uint64_t> hnsw_deleted_ids_;
    HnswSoftDeleteFunc hnswFunc() {
        return [this](uint64_t id) { hnsw_deleted_ids_.push_back(id); };
    }

    std::unique_ptr<LineageIndex> lineage_;
    std::unique_ptr<ForgetEngine> forget_;
    std::unique_ptr<RemoveCoordinator> coordinator_;
};

// ── Basic Remove ─────────────────────────────────────────────────────────

TEST_F(RemoveCoordinatorTest, RemoveNonExistentReturnsError) {
    auto result = coordinator_->remove(999, RemoveReason::UserDelete,
                                        getFunc(), persistFunc(), hnswFunc());
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.error_message.find("not found") != std::string::npos);
}

TEST_F(RemoveCoordinatorTest, RemoveSingleRecordNoChildren) {
    addRecord(1);

    auto result = coordinator_->remove(1, RemoveReason::UserDelete,
                                        getFunc(), persistFunc(), hnswFunc());

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.primary_id, 1u);
    EXPECT_EQ(result.reason, RemoveReason::UserDelete);
    EXPECT_TRUE(result.invalidated_descendants.empty());
    EXPECT_TRUE(result.recomputed_descendants.empty());

    // Record should be tombstoned
    EXPECT_EQ(records_[1].phase, MemoryPhase::Tombstone);

    // HNSW should have been notified
    ASSERT_EQ(hnsw_deleted_ids_.size(), 1u);
    EXPECT_EQ(hnsw_deleted_ids_[0], 1u);
}

// ── Cascade Invalidation ─────────────────────────────────────────────────

TEST_F(RemoveCoordinatorTest, CascadeInvalidatesSingleChild) {
    addRecord(1);   // parent
    addRecord(10);  // child (sole parent = 1)

    lineage_->recordLineage(10, {1}, LineageOp::Infer);

    auto result = coordinator_->remove(1, RemoveReason::UserDelete,
                                        getFunc(), persistFunc(), hnswFunc());

    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.invalidated_descendants.size(), 1u);
    EXPECT_EQ(result.invalidated_descendants[0], 10u);

    // Child should be Invalidated
    EXPECT_EQ(records_[10].phase, MemoryPhase::Invalidated);

    // HNSW: primary + invalidated child
    EXPECT_EQ(hnsw_deleted_ids_.size(), 2u);
}

TEST_F(RemoveCoordinatorTest, CascadeSkipsChildWithLiveParent) {
    addRecord(1);   // parent to delete
    addRecord(2);   // another parent (live)
    addRecord(10);  // child with two parents

    lineage_->recordLineage(10, {1, 2}, LineageOp::Aggregate);

    auto result = coordinator_->remove(1, RemoveReason::UserDelete,
                                        getFunc(), persistFunc(), hnswFunc());

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.invalidated_descendants.empty());
    ASSERT_EQ(result.recomputed_descendants.size(), 1u);
    EXPECT_EQ(result.recomputed_descendants[0], 10u);

    // Child should remain Active
    EXPECT_EQ(records_[10].phase, MemoryPhase::Active);
}

TEST_F(RemoveCoordinatorTest, CascadePropagatesChain) {
    addRecord(1);    // root
    addRecord(10);   // child of 1
    addRecord(100);  // child of 10

    lineage_->recordLineage(10, {1}, LineageOp::Infer);
    lineage_->recordLineage(100, {10}, LineageOp::Summarize);

    auto result = coordinator_->remove(1, RemoveReason::UserDelete,
                                        getFunc(), persistFunc(), hnswFunc());

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.invalidated_descendants.size(), 2u);
    EXPECT_EQ(records_[10].phase, MemoryPhase::Invalidated);
    EXPECT_EQ(records_[100].phase, MemoryPhase::Invalidated);
}

// ForgetLog tests removed in Phase 4 — RemoveCoordinator no longer writes
// to ForgetEngine.getLog(). Cascade invalidation correctness is still
// covered by RemovesCascadeIntoInvalidated above; the audit-log emission
// will move to MemoryEventLog in a follow-up.

// ── RemoveReason ─────────────────────────────────────────────────────────

TEST_F(RemoveCoordinatorTest, SupportsAllRemoveReasons) {
    addRecord(1);
    addRecord(2);
    addRecord(3);

    auto r1 = coordinator_->remove(1, RemoveReason::UserDelete,
                                    getFunc(), persistFunc(), hnswFunc());
    EXPECT_EQ(r1.reason, RemoveReason::UserDelete);

    auto r2 = coordinator_->remove(2, RemoveReason::GcTombstone,
                                    getFunc(), persistFunc(), hnswFunc());
    EXPECT_EQ(r2.reason, RemoveReason::GcTombstone);

    auto r3 = coordinator_->remove(3, RemoveReason::ConflictLoser,
                                    getFunc(), persistFunc(), hnswFunc());
    EXPECT_EQ(r3.reason, RemoveReason::ConflictLoser);
}

// ── Stats ────────────────────────────────────────────────────────────────

TEST_F(RemoveCoordinatorTest, StatsTrackRemovals) {
    addRecord(1);
    addRecord(10);
    lineage_->recordLineage(10, {1}, LineageOp::Infer);

    coordinator_->remove(1, RemoveReason::UserDelete,
                          getFunc(), persistFunc(), hnswFunc());

    auto stats = coordinator_->stats();
    EXPECT_EQ(stats.total_removes, 1u);
    EXPECT_EQ(stats.total_invalidated, 1u);
}

TEST_F(RemoveCoordinatorTest, StatsResettable) {
    addRecord(1);
    coordinator_->remove(1, RemoveReason::UserDelete,
                          getFunc(), persistFunc(), hnswFunc());

    EXPECT_GT(coordinator_->stats().total_removes, 0u);
    coordinator_->resetStats();
    EXPECT_EQ(coordinator_->stats().total_removes, 0u);
}
