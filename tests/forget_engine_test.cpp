
#include <gtest/gtest.h>

#include "forget/forget_engine.h"

using namespace amind;

class ForgetEngineTest : public ::testing::Test {
protected:
    ForgetEngine engine_;
};

// ── forget_score computation ─────────────────────────────────────────────

TEST_F(ForgetEngineTest, ZeroSignalsYieldZeroScore) {
    ForgetSignals signals;
    EXPECT_FLOAT_EQ(engine_.computeForgetScore(signals), 0.0f);
}

TEST_F(ForgetEngineTest, MaxStalenessContribution) {
    ForgetSignals signals;
    signals.staleness = 1.0f;
    float score = engine_.computeForgetScore(signals);
    EXPECT_NEAR(score, 0.25f, 0.01f);  // weight_staleness=0.25
}

TEST_F(ForgetEngineTest, VerifiedBonusReducesScore) {
    ForgetSignals signals;
    signals.staleness = 0.5f;
    signals.low_access = 0.5f;
    float base = engine_.computeForgetScore(signals);

    ForgetSignals verified_signals = signals;
    verified_signals.verified_bonus = 1.0f;
    float with_bonus = engine_.computeForgetScore(verified_signals);

    EXPECT_LT(with_bonus, base);
}

TEST_F(ForgetEngineTest, ScoreClampedTo01) {
    // All positive signals maxed out
    ForgetSignals signals;
    signals.staleness = 1.0f;
    signals.low_access = 1.0f;
    signals.low_importance = 1.0f;
    signals.conflict_penalty = 1.0f;
    signals.redundancy = 1.0f;
    float score = engine_.computeForgetScore(signals);
    EXPECT_LE(score, 1.0f);

    // All negative signals maxed out
    ForgetSignals negative_signals;
    negative_signals.verified_bonus = 1.0f;
    negative_signals.graph_centrality = 1.0f;
    float neg_score = engine_.computeForgetScore(negative_signals);
    EXPECT_GE(neg_score, 0.0f);
}

TEST_F(ForgetEngineTest, CombinedSignalsAddUp) {
    ForgetSignals signals;
    signals.staleness = 1.0f;      // +0.25
    signals.low_access = 1.0f;     // +0.20
    signals.low_importance = 1.0f; // +0.20
    float score = engine_.computeForgetScore(signals);
    EXPECT_NEAR(score, 0.65f, 0.01f);
}

// ── evaluate() ───────────────────────────────────────────────────────────

TEST_F(ForgetEngineTest, EvaluateTombstone) {
    ForgetSignals signals;
    signals.staleness = 1.0f;
    signals.low_access = 1.0f;
    signals.low_importance = 1.0f;
    signals.conflict_penalty = 1.0f;
    signals.redundancy = 1.0f;

    auto eval = engine_.evaluate(42, signals);
    EXPECT_EQ(eval.memory_id, 42u);
    EXPECT_GE(eval.forget_score, 0.85f);
    EXPECT_EQ(eval.recommended_action, ForgetLogEntry::Decision::Tombstone);
}

TEST_F(ForgetEngineTest, EvaluateArchive) {
    ForgetSignals signals;
    signals.staleness = 1.0f;  // +0.25
    signals.low_access = 1.0f; // +0.20
    signals.low_importance = 1.0f; // +0.20
    // Total ~0.65, above archive_threshold(0.6) but below tombstone(0.85)

    auto eval = engine_.evaluate(42, signals);
    EXPECT_EQ(eval.recommended_action, ForgetLogEntry::Decision::Archive);
}

TEST_F(ForgetEngineTest, EvaluateDecay) {
    ForgetSignals signals;
    signals.staleness = 1.0f;  // +0.25
    signals.low_access = 0.5f; // +0.10
    // Total ~0.35, above decay_threshold(0.3) but below archive(0.6)

    auto eval = engine_.evaluate(42, signals);
    EXPECT_EQ(eval.recommended_action, ForgetLogEntry::Decision::Decay);
}

TEST_F(ForgetEngineTest, EvaluateBelowThresholds) {
    ForgetSignals signals;
    signals.staleness = 0.1f;

    auto eval = engine_.evaluate(42, signals);
    EXPECT_LT(eval.forget_score, 0.3f);
    EXPECT_EQ(eval.recommended_action, ForgetLogEntry::Decision::Decay);
    EXPECT_TRUE(eval.reason.find("below all thresholds") != std::string::npos);
}

// ── runCycle() ───────────────────────────────────────────────────────────

TEST_F(ForgetEngineTest, RunCycleSortsByScoreDescending) {
    std::vector<std::pair<uint64_t, ForgetSignals>> batch;

    ForgetSignals low;
    low.staleness = 0.1f;
    batch.push_back({1, low});

    ForgetSignals high;
    high.staleness = 1.0f;
    high.low_access = 1.0f;
    batch.push_back({2, high});

    ForgetSignals mid;
    mid.staleness = 0.5f;
    batch.push_back({3, mid});

    auto results = engine_.runCycle(batch);
    ASSERT_EQ(results.size(), 3u);
    EXPECT_GT(results[0].forget_score, results[1].forget_score);
    EXPECT_GT(results[1].forget_score, results[2].forget_score);
    EXPECT_EQ(results[0].memory_id, 2u);  // highest score
}

TEST_F(ForgetEngineTest, RunCycleEmptyBatch) {
    auto results = engine_.runCycle({});
    EXPECT_TRUE(results.empty());
}

// LogEntryAndRetrieve / ClearLog tests removed in Phase 4 — the legacy
// in-process ForgetLog API was deleted in favor of MemoryEventLog. Equivalent
// coverage now lives in tests/memory_event_log_test.cpp.

// ── Shadow Mode ──────────────────────────────────────────────────────────

TEST_F(ForgetEngineTest, ShadowModeDefaultOn) {
    EXPECT_TRUE(engine_.isShadowMode());
}

TEST_F(ForgetEngineTest, ShadowModeToggle) {
    engine_.setShadowMode(false);
    EXPECT_FALSE(engine_.isShadowMode());
    engine_.setShadowMode(true);
    EXPECT_TRUE(engine_.isShadowMode());
}

// ── decisionToString ─────────────────────────────────────────────────────

TEST_F(ForgetEngineTest, DecisionToStringCoversAll) {
    EXPECT_EQ(decisionToString(ForgetLogEntry::Decision::Decay), "Decay");
    EXPECT_EQ(decisionToString(ForgetLogEntry::Decision::Archive), "Archive");
    EXPECT_EQ(decisionToString(ForgetLogEntry::Decision::Tombstone), "Tombstone");
    EXPECT_EQ(decisionToString(ForgetLogEntry::Decision::Vacuum), "Vacuum");
    EXPECT_EQ(decisionToString(ForgetLogEntry::Decision::DropFromHNSW), "DropFromHNSW");
    EXPECT_EQ(decisionToString(ForgetLogEntry::Decision::ResolveConflict), "ResolveConflict");
    EXPECT_EQ(decisionToString(ForgetLogEntry::Decision::LineageInvalidate), "LineageInvalidate");
    EXPECT_EQ(decisionToString(ForgetLogEntry::Decision::GateReject), "GateReject");
    EXPECT_EQ(decisionToString(ForgetLogEntry::Decision::GateDefer), "GateDefer");
}

// ── Custom Weight Configuration ─────────────────────────────────────────

TEST_F(ForgetEngineTest, CustomWeightsAffectScore) {
    // Create engine with staleness weight = 0.80 (much higher than default 0.25)
    ForgetConfig custom_cfg;
    custom_cfg.weight_staleness = 0.80f;
    custom_cfg.weight_low_access = 0.05f;
    custom_cfg.weight_low_importance = 0.05f;
    custom_cfg.weight_conflict_penalty = 0.05f;
    custom_cfg.weight_redundancy = 0.05f;
    custom_cfg.weight_verified_bonus = 0.0f;
    custom_cfg.weight_graph_centrality = 0.0f;
    ForgetEngine custom_engine(custom_cfg);

    ForgetSignals signals;
    signals.staleness = 1.0f;

    float default_score = engine_.computeForgetScore(signals);
    float custom_score = custom_engine.computeForgetScore(signals);

    // Custom engine should give much higher score for staleness
    EXPECT_NEAR(default_score, 0.25f, 0.01f);  // default weight = 0.25
    EXPECT_NEAR(custom_score, 0.80f, 0.01f);    // custom weight = 0.80
}

TEST_F(ForgetEngineTest, CustomThresholdsChangeDecisions) {
    // Lower thresholds → more aggressive GC
    ForgetConfig aggressive_cfg;
    aggressive_cfg.decay_threshold = 0.10f;
    aggressive_cfg.archive_threshold = 0.20f;
    aggressive_cfg.tombstone_threshold = 0.30f;
    ForgetEngine aggressive_engine(aggressive_cfg);

    ForgetSignals signals;
    signals.staleness = 1.0f;  // score ~0.25 with default weights

    // Default engine: score ~0.25 < decay_threshold(0.3) → below thresholds
    auto default_eval = engine_.evaluate(42, signals);
    EXPECT_TRUE(default_eval.reason.find("below all thresholds") != std::string::npos);

    // Aggressive engine: score ~0.25 > decay(0.10) and > archive(0.20) → Archive
    auto aggressive_eval = aggressive_engine.evaluate(42, signals);
    EXPECT_EQ(aggressive_eval.recommended_action, ForgetLogEntry::Decision::Archive);
}

TEST_F(ForgetEngineTest, ConfigSnapshotReflectsCustomValues) {
    ForgetConfig custom_cfg;
    custom_cfg.stale_hours = 336.0f;  // 14 days
    custom_cfg.weight_staleness = 0.50f;
    ForgetEngine custom_engine(custom_cfg);

    auto snapshot = custom_engine.config();
    EXPECT_FLOAT_EQ(snapshot.stale_hours, 336.0f);
    EXPECT_FLOAT_EQ(snapshot.weight_staleness, 0.50f);
}
