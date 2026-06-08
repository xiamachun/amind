
#include "gate/write_gate.h"

#include <gtest/gtest.h>
#include <vector>

using namespace amind;

namespace {

/// Helper: create a simple ProposedMemory with content and embedding.
ProposedMemory makeCandidate(const std::string& content,
                              float similarity_target = 0.0f,
                              SourceTier tier = SourceTier::Inference) {
    ProposedMemory candidate;
    candidate.content = content;
    candidate.source_tier = tier;
    candidate.importance = 0.5f;
    // Simple embedding: just a few floats
    candidate.embedding = {0.1f, 0.2f, 0.3f, 0.4f};
    return candidate;
}

/// Helper: search function that returns no neighbors.
SimilaritySearchFunc noNeighbors() {
    return [](const std::vector<float>&, size_t) {
        return std::vector<std::pair<uint64_t, float>>{};
    };
}

/// Helper: search function that returns one neighbor with given similarity.
SimilaritySearchFunc oneNeighbor(uint64_t id, float similarity) {
    return [id, similarity](const std::vector<float>&, size_t) {
        return std::vector<std::pair<uint64_t, float>>{{id, similarity}};
    };
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// Shadow Mode (default ON)
// ═══════════════════════════════════════════════════════════════════════════

TEST(WriteGateTest, DefaultIsShadowMode) {
    WriteGate gate;
    EXPECT_TRUE(gate.isShadowMode());
}

TEST(WriteGateTest, ShadowModeAlwaysAccepts) {
    WriteGateConfig config;
    config.shadow_mode = true;
    WriteGate gate(config);

    // Even a near-duplicate should be accepted in shadow mode
    auto verdict = gate.evaluate(
        makeCandidate("some content"),
        oneNeighbor(42, 0.99f)  // Very high similarity → would be duplicate
    );

    EXPECT_EQ(verdict.decision, GateDecision::Accepted);
    EXPECT_TRUE(verdict.reason.find("shadow") != std::string::npos);
}

TEST(WriteGateTest, ShadowModeCountsOverrides) {
    WriteGateConfig config;
    config.shadow_mode = true;
    WriteGate gate(config);

    gate.evaluate(makeCandidate("duplicate content"), oneNeighbor(1, 0.99f));
    gate.evaluate(makeCandidate("another duplicate"), oneNeighbor(2, 0.97f));

    auto stats = gate.stats();
    EXPECT_EQ(stats.total_evaluated, 2u);
    EXPECT_EQ(stats.shadow_overrides, 2u);
}

// ═══════════════════════════════════════════════════════════════════════════
// Hard Mode (shadow OFF)
// ═══════════════════════════════════════════════════════════════════════════

TEST(WriteGateTest, HardModeRejectsDuplicate) {
    WriteGateConfig config;
    config.shadow_mode = false;
    config.duplicate_threshold = 0.95f;
    WriteGate gate(config);

    auto verdict = gate.evaluate(
        makeCandidate("user likes dark mode"),
        oneNeighbor(42, 0.98f)
    );

    EXPECT_EQ(verdict.decision, GateDecision::Rejected);
    EXPECT_TRUE(verdict.reason.find("duplicate") != std::string::npos ||
                verdict.reason.find("Near-duplicate") != std::string::npos);
}

TEST(WriteGateTest, HardModeAcceptsNewContent) {
    WriteGateConfig config;
    config.shadow_mode = false;
    WriteGate gate(config);

    auto verdict = gate.evaluate(
        makeCandidate("user prefers TypeScript over JavaScript"),
        noNeighbors()
    );

    EXPECT_EQ(verdict.decision, GateDecision::Accepted);
    EXPECT_GT(verdict.marginal_value, 0.5f);
}

TEST(WriteGateTest, HardModeAcceptsMediumSimilarity) {
    WriteGateConfig config;
    config.shadow_mode = false;
    config.duplicate_threshold = 0.95f;
    WriteGate gate(config);

    // 0.6 similarity → not a duplicate, reasonable marginal value
    auto verdict = gate.evaluate(
        makeCandidate("user likes TypeScript"),
        oneNeighbor(10, 0.6f)
    );

    EXPECT_EQ(verdict.decision, GateDecision::Accepted);
    EXPECT_GT(verdict.marginal_value, 0.3f);
}

// ═══════════════════════════════════════════════════════════════════════════
// Transient Content Detection
// ═══════════════════════════════════════════════════════════════════════════

TEST(WriteGateTest, RejectsTransientContent) {
    WriteGateConfig config;
    config.shadow_mode = false;
    WriteGate gate(config);

    // Test various transient messages
    for (const auto& msg : {"hi", "ok", "thanks!", "goodbye", "sure", "yes"}) {
        auto verdict = gate.evaluate(makeCandidate(msg), noNeighbors());
        EXPECT_EQ(verdict.decision, GateDecision::Rejected)
            << "Should reject transient: " << msg;
    }
}

TEST(WriteGateTest, AcceptsSubstantiveContent) {
    WriteGateConfig config;
    config.shadow_mode = false;
    WriteGate gate(config);

    auto verdict = gate.evaluate(
        makeCandidate("User always uses vim keybindings in their IDE"),
        noNeighbors()
    );

    EXPECT_EQ(verdict.decision, GateDecision::Accepted);
}

// ═══════════════════════════════════════════════════════════════════════════
// Source Tier Classification
// ═══════════════════════════════════════════════════════════════════════════

TEST(WriteGateTest, ClassifiesSourceTierFromContent) {
    WriteGateConfig config;
    config.shadow_mode = false;
    WriteGate gate(config);

    // Behavioral pattern
    auto verdict1 = gate.evaluate(
        makeCandidate("User always chooses dark mode every time"),
        noNeighbors()
    );
    EXPECT_EQ(verdict1.tier, SourceTier::Behavioral);

    // Assertion pattern
    auto verdict2 = gate.evaluate(
        makeCandidate("I prefer using React over Vue"),
        noNeighbors()
    );
    EXPECT_EQ(verdict2.tier, SourceTier::Assertion);
}

TEST(WriteGateTest, PreservesExplicitSourceTier) {
    WriteGateConfig config;
    config.shadow_mode = false;
    WriteGate gate(config);

    auto verdict = gate.evaluate(
        makeCandidate("some generic content", 0.0f, SourceTier::Behavioral),
        noNeighbors()
    );
    EXPECT_EQ(verdict.tier, SourceTier::Behavioral);
}

// ═══════════════════════════════════════════════════════════════════════════
// Marginal Value Computation
// ═══════════════════════════════════════════════════════════════════════════

TEST(WriteGateTest, MarginalValueHighForNewContent) {
    WriteGateConfig config;
    config.shadow_mode = false;
    WriteGate gate(config);

    auto verdict = gate.evaluate(
        makeCandidate("completely unique new information"),
        noNeighbors()
    );

    EXPECT_GE(verdict.marginal_value, 0.9f);
}

TEST(WriteGateTest, MarginalValueLowForNearDuplicate) {
    WriteGateConfig config;
    config.shadow_mode = false;
    config.duplicate_threshold = 0.99f;  // Very strict duplicate → won't be rejected as dup
    WriteGate gate(config);

    // High similarity but below dup threshold → low marginal value
    auto verdict = gate.evaluate(
        makeCandidate("very similar content"),
        oneNeighbor(1, 0.92f)
    );

    EXPECT_LT(verdict.marginal_value, 0.3f);
}

// ═══════════════════════════════════════════════════════════════════════════
// Stats
// ═══════════════════════════════════════════════════════════════════════════

TEST(WriteGateTest, StatsTracking) {
    WriteGateConfig config;
    config.shadow_mode = false;
    WriteGate gate(config);

    gate.evaluate(makeCandidate("good content"), noNeighbors());
    gate.evaluate(makeCandidate("hi"), noNeighbors());

    auto stats = gate.stats();
    EXPECT_EQ(stats.total_evaluated, 2u);
    EXPECT_EQ(stats.accepted, 1u);
    EXPECT_EQ(stats.rejected, 1u);
}

TEST(WriteGateTest, StatsReset) {
    WriteGateConfig config;
    config.shadow_mode = false;
    WriteGate gate(config);

    gate.evaluate(makeCandidate("something"), noNeighbors());
    gate.resetStats();

    auto stats = gate.stats();
    EXPECT_EQ(stats.total_evaluated, 0u);
}

// ═══════════════════════════════════════════════════════════════════════════
// Config
// ═══════════════════════════════════════════════════════════════════════════

TEST(WriteGateTest, ConfigToggle) {
    WriteGate gate;
    EXPECT_TRUE(gate.isShadowMode());

    gate.setShadowMode(false);
    EXPECT_FALSE(gate.isShadowMode());

    WriteGateConfig newConfig;
    newConfig.duplicate_threshold = 0.85f;
    newConfig.shadow_mode = true;
    gate.setConfig(newConfig);
    EXPECT_TRUE(gate.isShadowMode());
    EXPECT_FLOAT_EQ(gate.config().duplicate_threshold, 0.85f);
}
