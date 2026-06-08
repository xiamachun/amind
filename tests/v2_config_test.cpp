
#include <gtest/gtest.h>

#include "config/v2_config.h"

using namespace amind;

class V2ConfigTest : public ::testing::Test {
protected:
    FeatureGate gate_;
};

// ── Defaults ─────────────────────────────────────────────────────────────

TEST_F(V2ConfigTest, AllFeaturesDisabledByDefault) {
    EXPECT_FALSE(gate_.isWriteGateEnabled());
    EXPECT_FALSE(gate_.isLineagePropagationEnabled());
    EXPECT_FALSE(gate_.isForgetScoreEnabled());
    EXPECT_FALSE(gate_.isHnswCompactEnabled());
    EXPECT_FALSE(gate_.isConflictResolverEnabled());
    EXPECT_FALSE(gate_.isConsolidationEnabled());
    EXPECT_EQ(gate_.enabledCount(), 1);  // agent_store_routing defaults to true
}

TEST_F(V2ConfigTest, GlobalShadowModeOnByDefault) {
    EXPECT_TRUE(gate_.isGlobalShadowMode());
}

// ── Individual Toggles ───────────────────────────────────────────────────

TEST_F(V2ConfigTest, ToggleWriteGate) {
    gate_.setWriteGateEnabled(true);
    EXPECT_TRUE(gate_.isWriteGateEnabled());
    EXPECT_EQ(gate_.enabledCount(), 2);  // +agent_store_routing

    gate_.setWriteGateEnabled(false);
    EXPECT_FALSE(gate_.isWriteGateEnabled());
}

TEST_F(V2ConfigTest, ToggleLineagePropagation) {
    gate_.setLineagePropagationEnabled(true);
    EXPECT_TRUE(gate_.isLineagePropagationEnabled());
}

TEST_F(V2ConfigTest, ToggleForgetScore) {
    gate_.setForgetScoreEnabled(true);
    EXPECT_TRUE(gate_.isForgetScoreEnabled());
}

TEST_F(V2ConfigTest, ToggleHnswCompact) {
    gate_.setHnswCompactEnabled(true);
    EXPECT_TRUE(gate_.isHnswCompactEnabled());
}

TEST_F(V2ConfigTest, ToggleConflictResolver) {
    gate_.setConflictResolverEnabled(true);
    EXPECT_TRUE(gate_.isConflictResolverEnabled());
}

TEST_F(V2ConfigTest, ToggleConsolidation) {
    gate_.setConsolidationEnabled(true);
    EXPECT_TRUE(gate_.isConsolidationEnabled());
}

TEST_F(V2ConfigTest, ToggleGlobalShadowMode) {
    gate_.setGlobalShadowMode(false);
    EXPECT_FALSE(gate_.isGlobalShadowMode());
}

// ── Batch Operations ─────────────────────────────────────────────────────

TEST_F(V2ConfigTest, EnableAllTurnsEverythingOn) {
    gate_.enableAll();

    EXPECT_TRUE(gate_.isWriteGateEnabled());
    EXPECT_TRUE(gate_.isLineagePropagationEnabled());
    EXPECT_TRUE(gate_.isForgetScoreEnabled());
    EXPECT_TRUE(gate_.isHnswCompactEnabled());
    EXPECT_TRUE(gate_.isConflictResolverEnabled());
    EXPECT_TRUE(gate_.isConsolidationEnabled());
    EXPECT_FALSE(gate_.isGlobalShadowMode());
    EXPECT_EQ(gate_.enabledCount(), 13);
}

TEST_F(V2ConfigTest, DisableAllKillSwitch) {
    gate_.enableAll();
    gate_.disableAll();

    EXPECT_FALSE(gate_.isWriteGateEnabled());
    EXPECT_FALSE(gate_.isLineagePropagationEnabled());
    EXPECT_FALSE(gate_.isForgetScoreEnabled());
    EXPECT_FALSE(gate_.isHnswCompactEnabled());
    EXPECT_FALSE(gate_.isConflictResolverEnabled());
    EXPECT_FALSE(gate_.isConsolidationEnabled());
    EXPECT_TRUE(gate_.isGlobalShadowMode());
    EXPECT_EQ(gate_.enabledCount(), 1);  // agent_store_routing stays true
}

// ── isEnabled (generic string access) ────────────────────────────────────

TEST_F(V2ConfigTest, IsEnabledByName) {
    gate_.setWriteGateEnabled(true);
    EXPECT_TRUE(gate_.isEnabled("write_gate"));
    EXPECT_FALSE(gate_.isEnabled("forget_score"));
    EXPECT_FALSE(gate_.isEnabled("nonexistent_feature"));
}

// ── Snapshot ─────────────────────────────────────────────────────────────

TEST_F(V2ConfigTest, SnapshotReflectsCurrentState) {
    gate_.setWriteGateEnabled(true);
    gate_.setForgetScoreEnabled(true);

    auto snap = gate_.snapshot();
    EXPECT_TRUE(snap.write_gate_enabled);
    EXPECT_TRUE(snap.forget_score_enabled);
    EXPECT_FALSE(snap.lineage_propagation_enabled);
}

// ── Constructor with Config ──────────────────────────────────────────────

TEST_F(V2ConfigTest, ConstructorWithPresetConfig) {
    V2Config preset;
    preset.write_gate_enabled = true;
    preset.forget_score_enabled = true;
    preset.global_shadow_mode = false;

    FeatureGate custom_gate(preset);
    EXPECT_TRUE(custom_gate.isWriteGateEnabled());
    EXPECT_TRUE(custom_gate.isForgetScoreEnabled());
    EXPECT_FALSE(custom_gate.isGlobalShadowMode());
    EXPECT_EQ(custom_gate.enabledCount(), 3);  // +agent_store_routing default
}

// ── Enabled Count ────────────────────────────────────────────────────────

TEST_F(V2ConfigTest, EnabledCountIncrements) {
    EXPECT_EQ(gate_.enabledCount(), 1);  // agent_store_routing default
    gate_.setWriteGateEnabled(true);
    EXPECT_EQ(gate_.enabledCount(), 2);
    gate_.setForgetScoreEnabled(true);
    EXPECT_EQ(gate_.enabledCount(), 3);
    gate_.setWriteGateEnabled(false);
    EXPECT_EQ(gate_.enabledCount(), 2);
}
