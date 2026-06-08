
#include <gtest/gtest.h>

#include "memory/memory_store.h"
#include "core/types.h"
#include "config/v2_config.h"
#include "vector/recall_scorer.h"

#include <filesystem>
#include <cmath>

using namespace amind;

// ═══════════════════════════════════════════════════════════════════════════
// Test fixture for tiered memory features
// ═══════════════════════════════════════════════════════════════════════════

class TieredMemoryTest : public ::testing::Test {
protected:
    std::filesystem::path test_dir_;
    std::unique_ptr<MemoryStore> store_;

    void SetUp() override {
        test_dir_ = std::filesystem::temp_directory_path() / "amind_tiered_test";
        std::filesystem::remove_all(test_dir_);

        MemoryStore::Config cfg;
        cfg.data_dir = test_dir_.string();
        cfg.embedding_dim = 4;
        cfg.max_cache_size = 1000;
        cfg.decay_rate = 0.1f;
        cfg.promotion_working_access_count = 3;
        cfg.promotion_working_importance = 0.6f;
        cfg.promotion_short_access_count = 5;
        cfg.promotion_short_importance = 0.7f;
        store_ = std::make_unique<MemoryStore>(cfg);
        auto result = store_->init();
        ASSERT_TRUE(result.ok());
    }

    void TearDown() override {
        store_.reset();
        std::filesystem::remove_all(test_dir_);
    }

    MemoryRecord makeRecord(const std::string& content,
                            MemoryTier tier = MemoryTier::Working,
                            float importance = 0.5f) {
        MemoryRecord rec;
        rec.content = content;
        rec.tier = tier;
        rec.importance = importance;
        rec.embedding = {0.1f, 0.2f, 0.3f, 0.4f};
        rec.user_id = "test_user";
        rec.agent_id = "test_agent";
        rec.created_at = MemoryRecord::currentTimeSec();
        rec.last_accessed = rec.created_at;
        return rec;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 改动① 三层 MemoryTier
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(TieredMemoryTest, TierEnumValues) {
    EXPECT_EQ(static_cast<uint8_t>(MemoryTier::Working), 0);
    EXPECT_EQ(static_cast<uint8_t>(MemoryTier::ShortTerm), 1);
    EXPECT_EQ(static_cast<uint8_t>(MemoryTier::LongTerm), 2);
}

TEST_F(TieredMemoryTest, TierToStringConversion) {
    EXPECT_EQ(memoryTierToString(MemoryTier::Working), "working");
    EXPECT_EQ(memoryTierToString(MemoryTier::ShortTerm), "short_term");
    EXPECT_EQ(memoryTierToString(MemoryTier::LongTerm), "long_term");
}

TEST_F(TieredMemoryTest, TierFromStringConversion) {
    EXPECT_EQ(memoryTierFromString("working"), MemoryTier::Working);
    EXPECT_EQ(memoryTierFromString("short_term"), MemoryTier::ShortTerm);
    EXPECT_EQ(memoryTierFromString("long_term"), MemoryTier::LongTerm);
}

TEST_F(TieredMemoryTest, TierFromStringBackwardCompat) {
    // Old enum names should map correctly for backward compatibility
    EXPECT_EQ(memoryTierFromString("ephemeral"), MemoryTier::Working);
    EXPECT_EQ(memoryTierFromString("consolidated"), MemoryTier::ShortTerm);
    EXPECT_EQ(memoryTierFromString("unknown_value"), MemoryTier::Working);
}

TEST_F(TieredMemoryTest, DefaultTierIsWorking) {
    MemoryRecord rec;
    EXPECT_EQ(rec.tier, MemoryTier::Working);
}

TEST_F(TieredMemoryTest, StoreAndRetrieveAllTiers) {
    auto working_id = store_->fastStore(makeRecord("working mem", MemoryTier::Working)).value();
    auto short_id = store_->fastStore(makeRecord("short term mem", MemoryTier::ShortTerm)).value();
    auto long_id = store_->fastStore(makeRecord("long term mem", MemoryTier::LongTerm)).value();

    auto working_rec = store_->get(working_id);
    auto short_rec = store_->get(short_id);
    auto long_rec = store_->get(long_id);

    ASSERT_TRUE(working_rec.ok());
    ASSERT_TRUE(short_rec.ok());
    ASSERT_TRUE(long_rec.ok());

    EXPECT_EQ(working_rec->tier, MemoryTier::Working);
    EXPECT_EQ(short_rec->tier, MemoryTier::ShortTerm);
    EXPECT_EQ(long_rec->tier, MemoryTier::LongTerm);
}

// ═══════════════════════════════════════════════════════════════════════════
// 改动② 差异化衰减
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(TieredMemoryTest, WorkingDecaysFasterThanShortTerm) {
    auto working_rec = makeRecord("working", MemoryTier::Working, 1.0f);
    working_rec.last_accessed = MemoryRecord::currentTimeSec() - 86400;  // 1 day ago
    auto short_rec = makeRecord("short", MemoryTier::ShortTerm, 1.0f);
    short_rec.last_accessed = MemoryRecord::currentTimeSec() - 86400;

    auto working_id = store_->fastStore(working_rec).value();
    auto short_id = store_->fastStore(short_rec).value();

    store_->applyDecay();

    auto after_working = store_->get(working_id);
    auto after_short = store_->get(short_id);
    ASSERT_TRUE(after_working.ok());
    ASSERT_TRUE(after_short.ok());

    // Working (×2.0 multiplier) should decay more than ShortTerm (×1.0)
    EXPECT_LT(after_working->importance, after_short->importance);
}

TEST_F(TieredMemoryTest, LongTermDecaysSlowerThanShortTerm) {
    auto short_rec = makeRecord("short", MemoryTier::ShortTerm, 1.0f);
    short_rec.last_accessed = MemoryRecord::currentTimeSec() - 86400;
    auto long_rec = makeRecord("long", MemoryTier::LongTerm, 1.0f);
    long_rec.last_accessed = MemoryRecord::currentTimeSec() - 86400;

    auto short_id = store_->fastStore(short_rec).value();
    auto long_id = store_->fastStore(long_rec).value();

    store_->applyDecay();

    auto after_short = store_->get(short_id);
    auto after_long = store_->get(long_id);
    ASSERT_TRUE(after_short.ok());
    ASSERT_TRUE(after_long.ok());

    // LongTerm (×0.5 multiplier) should decay less than ShortTerm (×1.0)
    EXPECT_GT(after_long->importance, after_short->importance);
}

TEST_F(TieredMemoryTest, ExponentialDecayMode) {
    // Create a store with exponential decay enabled
    std::filesystem::path exp_dir = test_dir_ / "exp_decay";
    MemoryStore::Config cfg;
    cfg.data_dir = exp_dir.string();
    cfg.embedding_dim = 4;
    cfg.max_cache_size = 1000;
    cfg.decay_rate = 0.1f;
    cfg.exponential_decay_enabled = true;
    auto exp_store = std::make_unique<MemoryStore>(cfg);
    exp_store->init();

    auto rec = makeRecord("exp test", MemoryTier::Working, 1.0f);
    rec.last_accessed = MemoryRecord::currentTimeSec() - 86400;  // 1 day ago
    auto id = exp_store->fastStore(rec).value();

    exp_store->applyDecay();

    auto after = exp_store->get(id);
    ASSERT_TRUE(after.ok());
    // Importance should have decayed but remain positive (exponential never goes negative)
    EXPECT_GT(after->importance, 0.0f);
    EXPECT_LT(after->importance, 1.0f);
}

// ═══════════════════════════════════════════════════════════════════════════
// 改动②b 降级机制
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(TieredMemoryTest, DemotionFromLongTermToShortTerm) {
    auto rec = makeRecord("demote me", MemoryTier::LongTerm, 0.05f);
    rec.last_accessed = MemoryRecord::currentTimeSec() - 86400 * 30;  // 30 days ago
    auto id = store_->fastStore(rec).value();

    store_->applyDecay();

    auto after = store_->get(id);
    ASSERT_TRUE(after.ok());
    // With very low importance, should be demoted from LongTerm
    EXPECT_NE(after->tier, MemoryTier::LongTerm);
}

TEST_F(TieredMemoryTest, WorkingCannotBeDemotedFurther) {
    auto rec = makeRecord("bottom tier", MemoryTier::Working, 0.01f);
    rec.last_accessed = MemoryRecord::currentTimeSec() - 86400;
    auto id = store_->fastStore(rec).value();

    store_->applyDecay();

    auto after = store_->get(id);
    ASSERT_TRUE(after.ok());
    // Working is the lowest tier, should stay Working
    EXPECT_EQ(after->tier, MemoryTier::Working);
}

// ═══════════════════════════════════════════════════════════════════════════
// 改动③ 访问驱动实时晋升
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(TieredMemoryTest, PromotionFromWorkingToShortTermByAccess) {
    auto rec = makeRecord("promote by access", MemoryTier::Working, 0.3f);
    auto id = store_->fastStore(rec).value();

    // Access enough times to trigger promotion (threshold = 3)
    // fastStore already called get() once internally, so access_count starts >= 1
    for (int i = 0; i < 5; ++i) {
        store_->get(id);
    }

    auto after = store_->get(id);
    ASSERT_TRUE(after.ok());
    EXPECT_EQ(after->tier, MemoryTier::ShortTerm);
}

TEST_F(TieredMemoryTest, PromotionFromWorkingToShortTermByImportance) {
    // High importance should also trigger promotion (OR condition)
    auto rec = makeRecord("important working", MemoryTier::Working, 0.8f);
    auto id = store_->fastStore(rec).value();

    // Even a single get() should trigger promotion due to high importance
    auto after = store_->get(id);
    ASSERT_TRUE(after.ok());
    EXPECT_EQ(after->tier, MemoryTier::ShortTerm);
}

TEST_F(TieredMemoryTest, PromotionFromShortTermToLongTerm) {
    auto rec = makeRecord("promote to long", MemoryTier::ShortTerm, 0.8f);
    auto id = store_->fastStore(rec).value();

    // Access many times AND have high importance (AND condition for ShortTerm→LongTerm)
    for (int i = 0; i < 10; ++i) {
        store_->get(id);
    }

    auto after = store_->get(id);
    ASSERT_TRUE(after.ok());
    EXPECT_EQ(after->tier, MemoryTier::LongTerm);
}

TEST_F(TieredMemoryTest, NoPromotionWithLowImportanceForShortToLong) {
    // ShortTerm→LongTerm requires BOTH access_count AND importance
    auto rec = makeRecord("low importance short", MemoryTier::ShortTerm, 0.3f);
    auto id = store_->fastStore(rec).value();

    for (int i = 0; i < 15; ++i) {
        store_->get(id);
    }

    auto after = store_->get(id);
    ASSERT_TRUE(after.ok());
    // Even with many accesses, low importance prevents promotion
    EXPECT_EQ(after->tier, MemoryTier::ShortTerm);
}

// ═══════════════════════════════════════════════════════════════════════════
// 改动④ Recency 乘法门控 (RecallScorer)
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(TieredMemoryTest, RecencyGateMultiplicativeMode) {
    RecallWeights weights;
    weights.semantic = 0.6f;
    weights.recency = 0.2f;
    weights.importance = 0.1f;
    weights.frequency = 0.1f;
    weights.decayRate = 0.01f;
    weights.recencyGateEnabled = true;

    RecallScorer scorer(weights);

    auto now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    // Recent memory: high recency gate
    RecallCandidate recent;
    recent.memoryId = 1;
    recent.cosineDistance = 0.2f;  // high similarity
    recent.importance = 0.8f;
    recent.accessCount = 5;
    recent.createdAtMs = now_ms - 3600 * 1000;  // 1 hour ago

    // Old memory: low recency gate
    RecallCandidate old_mem;
    old_mem.memoryId = 2;
    old_mem.cosineDistance = 0.1f;  // even higher similarity
    old_mem.importance = 0.9f;
    old_mem.accessCount = 10;
    old_mem.createdAtMs = now_ms - 86400ULL * 30 * 1000;  // 30 days ago

    auto recent_score = scorer.score(recent, now_ms);
    auto old_score = scorer.score(old_mem, now_ms);

    // In multiplicative mode, the very old memory should be significantly suppressed
    // even though it has better semantic similarity and importance
    EXPECT_GT(recent_score.totalScore, old_score.totalScore);
}

TEST_F(TieredMemoryTest, RecencyGateDisabledFallsBackToAdditive) {
    RecallWeights weights;
    weights.recencyGateEnabled = false;

    RecallScorer scorer(weights);

    auto now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    RecallCandidate candidate;
    candidate.memoryId = 1;
    candidate.cosineDistance = 0.3f;
    candidate.importance = 0.5f;
    candidate.accessCount = 3;
    candidate.createdAtMs = now_ms - 3600 * 1000;

    auto result = scorer.score(candidate, now_ms);
    // In additive mode, total = α×semantic + β×recency + γ×importance + δ×frequency
    float expected = weights.semantic * result.semanticScore
                   + weights.recency * result.recencyScore
                   + weights.importance * result.importanceScore
                   + weights.frequency * result.frequencyScore;
    EXPECT_NEAR(result.totalScore, expected, 0.001f);
}

// ═══════════════════════════════════════════════════════════════════════════
// FeatureGate 扩展测试
// ═══════════════════════════════════════════════════════════════════════════

TEST(FeatureGateExtTest, NewGatesDefaultFalse) {
    FeatureGate gate;
    EXPECT_FALSE(gate.isTieredDecayEnabled());
    EXPECT_FALSE(gate.isExponentialDecayEnabled());
    EXPECT_FALSE(gate.isAccessPromotionEnabled());
    EXPECT_FALSE(gate.isRecencyGateEnabled());
    EXPECT_FALSE(gate.isTierDemotionEnabled());
}

TEST(FeatureGateExtTest, AgentStoreRoutingDefaultTrue) {
    FeatureGate gate;
    EXPECT_TRUE(gate.isAgentStoreRoutingEnabled());
}

TEST(FeatureGateExtTest, ToggleNewGates) {
    FeatureGate gate;

    gate.setTieredDecayEnabled(true);
    EXPECT_TRUE(gate.isTieredDecayEnabled());

    gate.setExponentialDecayEnabled(true);
    EXPECT_TRUE(gate.isExponentialDecayEnabled());

    gate.setAccessPromotionEnabled(true);
    EXPECT_TRUE(gate.isAccessPromotionEnabled());

    gate.setRecencyGateEnabled(true);
    EXPECT_TRUE(gate.isRecencyGateEnabled());

    gate.setTierDemotionEnabled(true);
    EXPECT_TRUE(gate.isTierDemotionEnabled());
}

TEST(FeatureGateExtTest, EnableAllIncludesNewGates) {
    FeatureGate gate;
    gate.enableAll();

    EXPECT_TRUE(gate.isTieredDecayEnabled());
    EXPECT_TRUE(gate.isExponentialDecayEnabled());
    EXPECT_TRUE(gate.isAccessPromotionEnabled());
    EXPECT_TRUE(gate.isRecencyGateEnabled());
    EXPECT_TRUE(gate.isTierDemotionEnabled());
    EXPECT_TRUE(gate.isAgentStoreRoutingEnabled());
    EXPECT_FALSE(gate.isGlobalShadowMode());
}

TEST(FeatureGateExtTest, DisableAllKeepsAgentStoreRouting) {
    FeatureGate gate;
    gate.enableAll();
    gate.disableAll();

    EXPECT_FALSE(gate.isTieredDecayEnabled());
    EXPECT_FALSE(gate.isExponentialDecayEnabled());
    EXPECT_FALSE(gate.isAccessPromotionEnabled());
    EXPECT_FALSE(gate.isRecencyGateEnabled());
    EXPECT_FALSE(gate.isTierDemotionEnabled());
    // Agent store routing should stay true — it's architectural foundation
    EXPECT_TRUE(gate.isAgentStoreRoutingEnabled());
}

TEST(FeatureGateExtTest, IsEnabledByName) {
    FeatureGate gate;
    gate.setTieredDecayEnabled(true);
    gate.setRecencyGateEnabled(true);

    EXPECT_TRUE(gate.isEnabled("tiered_decay"));
    EXPECT_TRUE(gate.isEnabled("recency_gate"));
    EXPECT_FALSE(gate.isEnabled("exponential_decay"));
    EXPECT_TRUE(gate.isEnabled("agent_store_routing"));  // default true
}

TEST(FeatureGateExtTest, EnabledCountIncludesNewGates) {
    FeatureGate gate;
    int base_count = gate.enabledCount();  // agent_store_routing is true by default

    gate.setTieredDecayEnabled(true);
    gate.setAccessPromotionEnabled(true);
    EXPECT_EQ(gate.enabledCount(), base_count + 2);
}

// ═══════════════════════════════════════════════════════════════════════════
// Serialization round-trip for new tier values
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(TieredMemoryTest, SerializeDeserializeAllTiers) {
    for (auto tier : {MemoryTier::Working, MemoryTier::ShortTerm, MemoryTier::LongTerm}) {
        auto rec = makeRecord("tier test", tier, 0.5f);
        auto id = store_->fastStore(rec).value();
        store_->flush();

        // Force re-read from LSM by clearing cache
        auto retrieved = store_->get(id);
        ASSERT_TRUE(retrieved.ok()) << "Failed for tier " << memoryTierToString(tier);
        EXPECT_EQ(retrieved->tier, tier)
            << "Tier mismatch for " << memoryTierToString(tier);
    }
}
