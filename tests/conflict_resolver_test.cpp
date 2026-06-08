
#include <gtest/gtest.h>

#include "conflict/conflict_resolver.h"

using namespace amind;

class ConflictResolverTest : public ::testing::Test {
protected:
    ConflictResolver resolver_;

    // Helper to build a minimal MemoryRecord for conflict tests.
    MemoryRecord makeRecord(uint64_t id, Confidence conf = Confidence::Inferred,
                            uint32_t created = 100,
                            SourceTier tier = SourceTier::Assertion) {
        MemoryRecord record;
        record.memory_id = id;
        record.confidence_level = conf;
        record.created_at = created;
        record.source_tier = tier;
        record.content = "Memory " + std::to_string(id);
        record.importance = 0.5f;
        return record;
    }
};

// ── L1: Version Ancestor ─────────────────────────────────────────────────

TEST_F(ConflictResolverTest, L1_BIsNewerVersionOfA) {
    auto a = makeRecord(1);
    auto b = makeRecord(2);
    b.parent_id = 1;  // B is child of A → A is ancestor → A loses

    auto result = resolver_.resolve(a, b);
    EXPECT_EQ(result.level, ResolutionLevel::VersionAncestor);
    EXPECT_EQ(result.winner_id, 2u);
    EXPECT_EQ(result.loser_id, 1u);
    EXPECT_FALSE(result.coexist);
}

TEST_F(ConflictResolverTest, L1_AIsNewerVersionOfB) {
    auto a = makeRecord(1);
    a.parent_id = 2;  // A is child of B
    auto b = makeRecord(2);

    auto result = resolver_.resolve(a, b);
    EXPECT_EQ(result.level, ResolutionLevel::VersionAncestor);
    EXPECT_EQ(result.winner_id, 1u);
    EXPECT_EQ(result.loser_id, 2u);
}

// ── L2: Confidence ───────────────────────────────────────────────────────

TEST_F(ConflictResolverTest, L2_VerifiedBeatsInferred) {
    auto a = makeRecord(1, Confidence::Verified, 100);
    auto b = makeRecord(2, Confidence::Inferred, 100);

    auto result = resolver_.resolve(a, b);
    EXPECT_EQ(result.level, ResolutionLevel::Confidence);
    EXPECT_EQ(result.winner_id, 1u);
}

TEST_F(ConflictResolverTest, L2_InferredBeatsStale) {
    auto a = makeRecord(1, Confidence::Stale, 100);
    auto b = makeRecord(2, Confidence::Inferred, 100);

    auto result = resolver_.resolve(a, b);
    EXPECT_EQ(result.level, ResolutionLevel::Confidence);
    EXPECT_EQ(result.winner_id, 2u);
}

// ── L3: Recency ──────────────────────────────────────────────────────────

TEST_F(ConflictResolverTest, L3_NewerWins) {
    auto a = makeRecord(1, Confidence::Inferred, 200);  // newer
    auto b = makeRecord(2, Confidence::Inferred, 100);  // older

    auto result = resolver_.resolve(a, b);
    EXPECT_EQ(result.level, ResolutionLevel::Recency);
    EXPECT_EQ(result.winner_id, 1u);
}

// ── L4: SourceTier ───────────────────────────────────────────────────────

TEST_F(ConflictResolverTest, L4_BehavioralBeatsAssertion) {
    auto a = makeRecord(1, Confidence::Inferred, 100, SourceTier::Behavioral);
    auto b = makeRecord(2, Confidence::Inferred, 100, SourceTier::Assertion);

    auto result = resolver_.resolve(a, b);
    EXPECT_EQ(result.level, ResolutionLevel::SourceTier);
    EXPECT_EQ(result.winner_id, 1u);
}

TEST_F(ConflictResolverTest, L4_AssertionBeatsInference) {
    auto a = makeRecord(1, Confidence::Inferred, 100, SourceTier::Assertion);
    auto b = makeRecord(2, Confidence::Inferred, 100, SourceTier::Inference);

    auto result = resolver_.resolve(a, b);
    EXPECT_EQ(result.level, ResolutionLevel::SourceTier);
    EXPECT_EQ(result.winner_id, 1u);
}

// ── L5: LLM Judge ────────────────────────────────────────────────────────

TEST_F(ConflictResolverTest, L5_LlmJudgePicksWinner) {
    ConflictResolverConfig config;
    config.enable_llm_judge = true;
    ConflictResolver resolver_with_llm(config);

    // LLM always picks the first ID
    resolver_with_llm.setLlmJudge(
        [](uint64_t id_a, const std::string&, uint64_t, const std::string&)
            -> std::optional<uint64_t> {
            return id_a;
        });

    // All same properties → L1-L4 all tie
    auto a = makeRecord(1, Confidence::Inferred, 100, SourceTier::Assertion);
    auto b = makeRecord(2, Confidence::Inferred, 100, SourceTier::Assertion);

    auto result = resolver_with_llm.resolve(a, b);
    EXPECT_EQ(result.level, ResolutionLevel::LlmJudge);
    EXPECT_EQ(result.winner_id, 1u);
}

TEST_F(ConflictResolverTest, L5_LlmUndecidedFallsToL6) {
    ConflictResolverConfig config;
    config.enable_llm_judge = true;
    ConflictResolver resolver_with_llm(config);

    // LLM is undecided
    resolver_with_llm.setLlmJudge(
        [](uint64_t, const std::string&, uint64_t, const std::string&)
            -> std::optional<uint64_t> {
            return std::nullopt;
        });

    auto a = makeRecord(1, Confidence::Inferred, 100, SourceTier::Assertion);
    auto b = makeRecord(2, Confidence::Inferred, 100, SourceTier::Assertion);

    auto result = resolver_with_llm.resolve(a, b);
    EXPECT_EQ(result.level, ResolutionLevel::Coexist);
    EXPECT_TRUE(result.coexist);
}

// ── L6: Coexist ──────────────────────────────────────────────────────────

TEST_F(ConflictResolverTest, L6_AllTiedCoexist) {
    // Same confidence, same timestamp, same tier → all levels tie → L6
    auto a = makeRecord(1, Confidence::Inferred, 100, SourceTier::Assertion);
    auto b = makeRecord(2, Confidence::Inferred, 100, SourceTier::Assertion);

    auto result = resolver_.resolve(a, b);
    EXPECT_EQ(result.level, ResolutionLevel::Coexist);
    EXPECT_TRUE(result.coexist);
}

// ── Stats ────────────────────────────────────────────────────────────────

TEST_F(ConflictResolverTest, StatsTrackResolutions) {
    auto a = makeRecord(1, Confidence::Verified, 100);
    auto b = makeRecord(2, Confidence::Inferred, 100);
    resolver_.resolve(a, b);  // L2

    auto c = makeRecord(3, Confidence::Inferred, 200);
    auto d = makeRecord(4, Confidence::Inferred, 100);
    resolver_.resolve(c, d);  // L3

    auto stats = resolver_.stats();
    EXPECT_EQ(stats.total_resolved, 2u);
    EXPECT_EQ(stats.by_confidence, 1u);
    EXPECT_EQ(stats.by_recency, 1u);
}

TEST_F(ConflictResolverTest, StatsResettable) {
    auto a = makeRecord(1, Confidence::Verified, 100);
    auto b = makeRecord(2, Confidence::Inferred, 100);
    resolver_.resolve(a, b);

    EXPECT_GT(resolver_.stats().total_resolved, 0u);
    resolver_.resetStats();
    EXPECT_EQ(resolver_.stats().total_resolved, 0u);
}

// ── resolutionLevelToString ──────────────────────────────────────────────

TEST_F(ConflictResolverTest, ResolutionLevelToStringCoversAll) {
    EXPECT_EQ(resolutionLevelToString(ResolutionLevel::VersionAncestor), "VersionAncestor");
    EXPECT_EQ(resolutionLevelToString(ResolutionLevel::Confidence), "Confidence");
    EXPECT_EQ(resolutionLevelToString(ResolutionLevel::Recency), "Recency");
    EXPECT_EQ(resolutionLevelToString(ResolutionLevel::SourceTier), "SourceTier");
    EXPECT_EQ(resolutionLevelToString(ResolutionLevel::LlmJudge), "LlmJudge");
    EXPECT_EQ(resolutionLevelToString(ResolutionLevel::Coexist), "Coexist");
}
