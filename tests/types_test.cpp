
#include "core/types.h"

#include <gtest/gtest.h>

using namespace amind;

// ═══════════════════════════════════════════════════════════════════════════
// V1 Enum Tests — ensure existing behavior unchanged
// ═══════════════════════════════════════════════════════════════════════════

TEST(MemoryPhaseTest, V1PhasesToString) {
    EXPECT_EQ(phaseToString(MemoryPhase::Active), "Active");
    EXPECT_EQ(phaseToString(MemoryPhase::Versioned), "Versioned");
    EXPECT_EQ(phaseToString(MemoryPhase::Archived), "Archived");
    EXPECT_EQ(phaseToString(MemoryPhase::Tombstone), "Tombstone");
}

TEST(MemoryPhaseTest, V2InvalidatedPhase) {
    EXPECT_EQ(phaseToString(MemoryPhase::Invalidated), "Invalidated");
    EXPECT_EQ(static_cast<uint8_t>(MemoryPhase::Invalidated), 4);
}

TEST(MemoryPhaseTest, EnumValuesDoNotOverlap) {
    EXPECT_NE(static_cast<uint8_t>(MemoryPhase::Active),
              static_cast<uint8_t>(MemoryPhase::Invalidated));
    EXPECT_NE(static_cast<uint8_t>(MemoryPhase::Tombstone),
              static_cast<uint8_t>(MemoryPhase::Invalidated));
}

TEST(ConfidenceTest, AllValuesToString) {
    EXPECT_EQ(confidenceToString(Confidence::Verified), "Verified");
    EXPECT_EQ(confidenceToString(Confidence::Inferred), "Inferred");
    EXPECT_EQ(confidenceToString(Confidence::Stale), "Stale");
    EXPECT_EQ(confidenceToString(Confidence::Conflicted), "Conflicted");
}

TEST(EdgeTypeTest, AllValuesToString) {
    EXPECT_EQ(edgeTypeToString(EdgeType::Related), "Related");
    EXPECT_EQ(edgeTypeToString(EdgeType::DerivedFrom), "DerivedFrom");
    EXPECT_EQ(edgeTypeToString(EdgeType::Supersedes), "Supersedes");
    EXPECT_EQ(edgeTypeToString(EdgeType::ConflictsWith), "ConflictsWith");
}

// ═══════════════════════════════════════════════════════════════════════════
// V1 RecordFlags Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(RecordFlagsTest, V1FlagBitsNoOverlap) {
    EXPECT_EQ(RecordFlags::ALIVE, 1 << 0);
    EXPECT_EQ(RecordFlags::VERSIONED, 1 << 1);
    EXPECT_EQ(RecordFlags::ARCHIVED, 1 << 2);
    EXPECT_EQ(RecordFlags::TOMBSTONE, 1 << 3);
    EXPECT_EQ(RecordFlags::HAS_EMBEDDING, 1 << 4);
    EXPECT_EQ(RecordFlags::LLM_REFINED, 1 << 5);
}

TEST(RecordFlagsTest, V2NewFlagBitsNoOverlap) {
    EXPECT_EQ(RecordFlags::HNSW_EVICTED, 1 << 6);
    EXPECT_EQ(RecordFlags::LINEAGE_ORPHAN, 1 << 7);

    // Ensure no overlap with V1 flags
    uint16_t all_v1 = RecordFlags::ALIVE | RecordFlags::VERSIONED |
                      RecordFlags::ARCHIVED | RecordFlags::TOMBSTONE |
                      RecordFlags::HAS_EMBEDDING | RecordFlags::LLM_REFINED;
    EXPECT_EQ(all_v1 & RecordFlags::HNSW_EVICTED, 0);
    EXPECT_EQ(all_v1 & RecordFlags::LINEAGE_ORPHAN, 0);
}

TEST(RecordFlagsTest, AllFlagsFitInUint16) {
    uint16_t all_flags = RecordFlags::ALIVE | RecordFlags::VERSIONED |
                         RecordFlags::ARCHIVED | RecordFlags::TOMBSTONE |
                         RecordFlags::HAS_EMBEDDING | RecordFlags::LLM_REFINED |
                         RecordFlags::HNSW_EVICTED | RecordFlags::LINEAGE_ORPHAN;
    EXPECT_LE(all_flags, 0xFF);  // All fit in 8 bits, plenty of room in uint16
}

// ═══════════════════════════════════════════════════════════════════════════
// V2 New Enums
// ═══════════════════════════════════════════════════════════════════════════

TEST(MemoryLayerTest, AllValuesToString) {
    EXPECT_EQ(layerToString(MemoryLayer::Raw), "Raw");
    EXPECT_EQ(layerToString(MemoryLayer::Derived), "Derived");
}

TEST(MemoryLayerTest, EnumValues) {
    EXPECT_EQ(static_cast<uint8_t>(MemoryLayer::Raw), 0);
    EXPECT_EQ(static_cast<uint8_t>(MemoryLayer::Derived), 1);
}

TEST(LineageOpTest, AllValuesToString) {
    EXPECT_EQ(lineageOpToString(LineageOp::None), "None");
    EXPECT_EQ(lineageOpToString(LineageOp::Summarize), "Summarize");
    EXPECT_EQ(lineageOpToString(LineageOp::Aggregate), "Aggregate");
    EXPECT_EQ(lineageOpToString(LineageOp::Infer), "Infer");
    EXPECT_EQ(lineageOpToString(LineageOp::Distill), "Distill");
}

TEST(SourceTierTest, AllValuesToString) {
    EXPECT_EQ(sourceTierToString(SourceTier::Assertion), "Assertion");
    EXPECT_EQ(sourceTierToString(SourceTier::Behavioral), "Behavioral");
    EXPECT_EQ(sourceTierToString(SourceTier::Inference), "Inference");
}

TEST(SourceTierTest, EnumOrdering) {
    // Behavioral (1) > Assertion (0) > Inference (2) in trust ranking
    // but enum values don't encode that — just ensure they're distinct
    EXPECT_NE(static_cast<uint8_t>(SourceTier::Assertion),
              static_cast<uint8_t>(SourceTier::Behavioral));
    EXPECT_NE(static_cast<uint8_t>(SourceTier::Behavioral),
              static_cast<uint8_t>(SourceTier::Inference));
}

TEST(GateDecisionTest, AllValuesToString) {
    EXPECT_EQ(gateDecisionToString(GateDecision::Accepted), "Accepted");
    EXPECT_EQ(gateDecisionToString(GateDecision::Rejected), "Rejected");
    EXPECT_EQ(gateDecisionToString(GateDecision::Deferred), "Deferred");
}

TEST(RemoveReasonTest, AllValuesToString) {
    EXPECT_EQ(removeReasonToString(RemoveReason::UserDelete), "UserDelete");
    EXPECT_EQ(removeReasonToString(RemoveReason::LineageOrphan), "LineageOrphan");
    EXPECT_EQ(removeReasonToString(RemoveReason::GcArchive), "GcArchive");
    EXPECT_EQ(removeReasonToString(RemoveReason::GcTombstone), "GcTombstone");
    EXPECT_EQ(removeReasonToString(RemoveReason::ConflictLoser), "ConflictLoser");
}

TEST(RemoveReasonTest, EnumValuesContiguous) {
    EXPECT_EQ(static_cast<uint8_t>(RemoveReason::UserDelete), 0);
    EXPECT_EQ(static_cast<uint8_t>(RemoveReason::LineageOrphan), 1);
    EXPECT_EQ(static_cast<uint8_t>(RemoveReason::GcArchive), 2);
    EXPECT_EQ(static_cast<uint8_t>(RemoveReason::GcTombstone), 3);
    EXPECT_EQ(static_cast<uint8_t>(RemoveReason::ConflictLoser), 4);
}
