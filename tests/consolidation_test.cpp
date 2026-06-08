
#include <gtest/gtest.h>

#include "consolidation/consolidation_worker.h"

#include <cmath>

using namespace amind;

class ConsolidationTest : public ::testing::Test {
protected:
    ConsolidationWorker worker_;

    MemoryRecord makeRecord(uint64_t id, float importance = 0.5f,
                            uint32_t access_count = 1,
                            Confidence conf = Confidence::Inferred) {
        MemoryRecord record;
        record.memory_id = id;
        record.importance = importance;
        record.access_count = access_count;
        record.confidence_level = conf;
        record.scope = MemoryScope::Private;
        record.memory_type = MemoryType::Ephemeral;
        record.layer = MemoryLayer::Derived;
        record.embedding = {0.1f, 0.2f, 0.3f};
        record.content = "Memory " + std::to_string(id);
        return record;
    }

    // Simple cosine similarity that always returns a fixed value.
    CosineSimilarityFunc fixedSim(float value) {
        return [value](const std::vector<float>&, const std::vector<float>&) {
            return value;
        };
    }

    // Real dot-product based cosine similarity.
    CosineSimilarityFunc realCosine() {
        return [](const std::vector<float>& a, const std::vector<float>& b) -> float {
            float dot = 0, na = 0, nb = 0;
            for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            float denom = std::sqrt(na) * std::sqrt(nb);
            return (denom > 0) ? dot / denom : 0.0f;
        };
    }
};

// ── Promote Top-K ────────────────────────────────────────────────────────

TEST_F(ConsolidationTest, PromoteTopKByImportance) {
    std::vector<MemoryRecord> memories = {
        makeRecord(1, 0.9f, 5),   // high importance + access
        makeRecord(2, 0.1f, 1),   // low
        makeRecord(3, 0.8f, 10),  // high
        makeRecord(4, 0.5f, 2),   // medium
    };

    auto promoted = worker_.promoteTopK(memories);

    // Default top_k = 5, but only Inferred get promoted
    EXPECT_GE(promoted.size(), 2u);

    // Promoted records should now be Verified + Private scope with UserProfile type
    for (auto& record : memories) {
        bool was_promoted = std::find(promoted.begin(), promoted.end(),
                                       record.memory_id) != promoted.end();
        if (was_promoted) {
            EXPECT_EQ(record.confidence_level, Confidence::Verified);
            EXPECT_EQ(record.scope, MemoryScope::Private);
            EXPECT_EQ(record.memory_type, MemoryType::UserProfile);
        }
    }
}

TEST_F(ConsolidationTest, PromoteSkipsAlreadyVerified) {
    std::vector<MemoryRecord> memories = {
        makeRecord(1, 0.9f, 5),
    };
    memories[0].confidence_level = Confidence::Verified;

    auto promoted = worker_.promoteTopK(memories);
    EXPECT_TRUE(promoted.empty());  // Already Verified, skip
}

// ── Dedup ────────────────────────────────────────────────────────────────

TEST_F(ConsolidationTest, DedupMergesHighSimilarity) {
    std::vector<MemoryRecord> memories = {
        makeRecord(1, 0.5f),
        makeRecord(2, 0.7f),  // higher importance, should be kept
    };

    auto results = worker_.dedup(memories, fixedSim(0.99f));

    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].kept_id, 2u);  // higher importance wins
    ASSERT_EQ(results[0].archived_ids.size(), 1u);
    EXPECT_EQ(results[0].archived_ids[0], 1u);
}

TEST_F(ConsolidationTest, DedupKeepsBothBelowThreshold) {
    std::vector<MemoryRecord> memories = {
        makeRecord(1, 0.5f),
        makeRecord(2, 0.7f),
    };

    auto results = worker_.dedup(memories, fixedSim(0.5f));  // low sim → no dedup
    EXPECT_TRUE(results.empty());
}

TEST_F(ConsolidationTest, DedupSkipsEmptyEmbedding) {
    std::vector<MemoryRecord> memories = {
        makeRecord(1, 0.5f),
        makeRecord(2, 0.7f),
    };
    memories[1].embedding.clear();

    auto results = worker_.dedup(memories, fixedSim(0.99f));
    EXPECT_TRUE(results.empty());  // Can't compare without embedding
}

// ── Drift Check ──────────────────────────────────────────────────────────

TEST_F(ConsolidationTest, DriftCheckDetectsHighDrift) {
    auto derived = makeRecord(10);
    auto raw = makeRecord(1);
    raw.embedding = {0.9f, -0.1f, 0.0f};  // very different from derived

    std::vector<std::pair<MemoryRecord, MemoryRecord>> pairs = {{derived, raw}};

    // Low similarity → high drift
    auto results = worker_.checkDrift(pairs, fixedSim(0.5f));

    ASSERT_EQ(results.size(), 1u);
    EXPECT_GT(results[0].drift_score, 0.3f);
    EXPECT_TRUE(results[0].invalidated);
}

TEST_F(ConsolidationTest, DriftCheckPassesLowDrift) {
    auto derived = makeRecord(10);
    auto raw = makeRecord(1);

    std::vector<std::pair<MemoryRecord, MemoryRecord>> pairs = {{derived, raw}};

    auto results = worker_.checkDrift(pairs, fixedSim(0.95f));

    ASSERT_EQ(results.size(), 1u);
    EXPECT_LE(results[0].drift_score, 0.3f);
    EXPECT_FALSE(results[0].invalidated);
}

TEST_F(ConsolidationTest, DriftCheckSkipsMissingEmbedding) {
    auto derived = makeRecord(10);
    derived.embedding.clear();
    auto raw = makeRecord(1);

    std::vector<std::pair<MemoryRecord, MemoryRecord>> pairs = {{derived, raw}};

    auto results = worker_.checkDrift(pairs, fixedSim(0.5f));

    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].drift_score, 0.0f);
    EXPECT_FALSE(results[0].invalidated);
}

// ── Full Cycle ───────────────────────────────────────────────────────────

TEST_F(ConsolidationTest, RunCycleIntegration) {
    std::vector<MemoryRecord> session1 = {
        makeRecord(1, 0.9f, 5),
        makeRecord(2, 0.1f, 1),
    };
    std::vector<MemoryRecord> session2 = {
        makeRecord(3, 0.8f, 10),
    };

    std::vector<std::vector<MemoryRecord>> sessions = {session1, session2};

    // Drift candidates
    auto derived = makeRecord(100);
    auto raw = makeRecord(50);
    std::vector<std::pair<MemoryRecord, MemoryRecord>> drift_candidates = {{derived, raw}};

    auto result = worker_.runCycle(sessions, fixedSim(0.5f), drift_candidates);

    EXPECT_EQ(result.sessions_processed, 2u);
    EXPECT_GE(result.memories_promoted, 1u);
    EXPECT_EQ(result.drift_checked, 1u);
}

// ── Stats ────────────────────────────────────────────────────────────────

TEST_F(ConsolidationTest, StatsTrackCycles) {
    std::vector<std::vector<MemoryRecord>> sessions;
    std::vector<std::pair<MemoryRecord, MemoryRecord>> no_drift;

    worker_.runCycle(sessions, fixedSim(0.5f), no_drift);

    auto stats = worker_.stats();
    EXPECT_EQ(stats.cycles_run, 1u);
}

TEST_F(ConsolidationTest, StatsResettable) {
    std::vector<std::vector<MemoryRecord>> sessions;
    std::vector<std::pair<MemoryRecord, MemoryRecord>> no_drift;
    worker_.runCycle(sessions, fixedSim(0.5f), no_drift);

    EXPECT_GT(worker_.stats().cycles_run, 0u);
    worker_.resetStats();
    EXPECT_EQ(worker_.stats().cycles_run, 0u);
}

// ── Config ───────────────────────────────────────────────────────────────

TEST_F(ConsolidationTest, DefaultConfig) {
    auto config = worker_.config();
    EXPECT_TRUE(config.enabled);
    EXPECT_EQ(config.interval_seconds, 86400u);
    EXPECT_EQ(config.top_k_per_session, 5u);
    EXPECT_FLOAT_EQ(config.dedup_threshold, 0.95f);
    EXPECT_FLOAT_EQ(config.drift_threshold, 0.3f);
}
