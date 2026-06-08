
#include "core/memory_record.h"
#include "core/types.h"

#include <gtest/gtest.h>

using namespace amind;

// ═══════════════════════════════════════════════════════════════════════════
// RecordHeader binary compatibility
// ═══════════════════════════════════════════════════════════════════════════

TEST(RecordHeaderTest, SizeIs64Bytes) {
    EXPECT_EQ(sizeof(RecordHeader), 64u);
    EXPECT_EQ(sizeof(RecordHeader), WireFormat::HEADER_SIZE);
}

// ═══════════════════════════════════════════════════════════════════════════
// MemoryRecord V2 field defaults
// ═══════════════════════════════════════════════════════════════════════════

TEST(MemoryRecordTest, V2FieldsHaveCorrectDefaults) {
    MemoryRecord record;

    // Layer defaults
    EXPECT_EQ(record.layer, MemoryLayer::Raw);
    EXPECT_TRUE(record.lineage_parents.empty());
    EXPECT_EQ(record.lineage_op, LineageOp::None);

    // Write gate defaults
    EXPECT_EQ(record.source_tier, SourceTier::Inference);
    EXPECT_EQ(record.gate_decision, GateDecision::Accepted);
    EXPECT_FLOAT_EQ(record.marginal_value, 0.0f);

    // Forget engine defaults
    EXPECT_FLOAT_EQ(record.forget_score, 0.0f);
    EXPECT_EQ(record.last_gc_visit, 0u);
    EXPECT_EQ(record.resurrection_count, 0);
}

TEST(MemoryRecordTest, V1FieldsUnchanged) {
    MemoryRecord record;

    EXPECT_EQ(record.memory_id, 0u);
    EXPECT_EQ(record.scope, MemoryScope::Private);
    EXPECT_EQ(record.memory_type, MemoryType::Ephemeral);
    EXPECT_EQ(record.tier, MemoryTier::Working);
    EXPECT_EQ(record.user_id, "");
    EXPECT_EQ(record.agent_id, "");
    EXPECT_EQ(record.phase, MemoryPhase::Active);
    EXPECT_EQ(record.confidence_level, Confidence::Inferred);
    EXPECT_EQ(record.flags, RecordFlags::ALIVE);
    EXPECT_FLOAT_EQ(record.importance, 0.5f);
    EXPECT_EQ(record.access_count, 0u);
    EXPECT_EQ(record.mem_version, 1u);
    EXPECT_EQ(record.parent_id, 0u);
}

// ═══════════════════════════════════════════════════════════════════════════
// V2 field population
// ═══════════════════════════════════════════════════════════════════════════

TEST(MemoryRecordTest, CanSetDerivedLayerWithLineage) {
    MemoryRecord record;
    record.layer = MemoryLayer::Derived;
    record.lineage_parents = {100, 200, 300};
    record.lineage_op = LineageOp::Summarize;

    EXPECT_EQ(record.layer, MemoryLayer::Derived);
    EXPECT_EQ(record.lineage_parents.size(), 3u);
    EXPECT_EQ(record.lineage_parents[0], 100u);
    EXPECT_EQ(record.lineage_parents[1], 200u);
    EXPECT_EQ(record.lineage_parents[2], 300u);
    EXPECT_EQ(record.lineage_op, LineageOp::Summarize);
}

TEST(MemoryRecordTest, CanSetWriteGateMetadata) {
    MemoryRecord record;
    record.source_tier = SourceTier::Behavioral;
    record.gate_decision = GateDecision::Accepted;
    record.marginal_value = 0.85f;

    EXPECT_EQ(record.source_tier, SourceTier::Behavioral);
    EXPECT_EQ(record.gate_decision, GateDecision::Accepted);
    EXPECT_FLOAT_EQ(record.marginal_value, 0.85f);
}

TEST(MemoryRecordTest, CanSetForgetEngineMetadata) {
    MemoryRecord record;
    record.forget_score = 0.72f;
    record.last_gc_visit = 1700000000;
    record.resurrection_count = 3;

    EXPECT_FLOAT_EQ(record.forget_score, 0.72f);
    EXPECT_EQ(record.last_gc_visit, 1700000000u);
    EXPECT_EQ(record.resurrection_count, 3);
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase transitions (V2 additions)
// ═══════════════════════════════════════════════════════════════════════════

TEST(MemoryRecordTest, V1PhaseTransitionsStillWork) {
    MemoryRecord record;
    EXPECT_TRUE(record.isActive());
    EXPECT_TRUE(record.isAlive());

    record.markVersioned();
    EXPECT_EQ(record.phase, MemoryPhase::Versioned);

    MemoryRecord record2;
    record2.markArchived();
    EXPECT_EQ(record2.phase, MemoryPhase::Archived);

    MemoryRecord record3;
    record3.markTombstone();
    EXPECT_EQ(record3.phase, MemoryPhase::Tombstone);
    EXPECT_FALSE(record3.isAlive());
}

TEST(MemoryRecordTest, MarkInvalidatedSetsCorrectState) {
    MemoryRecord record;
    record.layer = MemoryLayer::Derived;
    record.lineage_parents = {42};

    record.markInvalidated();

    EXPECT_EQ(record.phase, MemoryPhase::Invalidated);
    EXPECT_TRUE(record.flags & RecordFlags::LINEAGE_ORPHAN);
    EXPECT_FALSE(record.isActive());
    EXPECT_FALSE(record.isAlive());
}
