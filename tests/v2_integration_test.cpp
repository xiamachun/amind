
/// V2 End-to-End Integration Test
///
/// Validates that all V2 modules work together in realistic scenarios:
///   FeatureGate → WriteGate → DerivedExtractor → LineageIndex → Propagator
///   → RemoveCoordinator → ForgetEngine → ConflictResolver → ConsolidationWorker
///
/// Each test scenario exercises multiple modules in a single flow.

#include <gtest/gtest.h>

#include "capture/derived_extractor.h"
#include "config/v2_config.h"
#include "conflict/conflict_resolver.h"
#include "consolidation/consolidation_worker.h"
#include "coordinator/remove_coordinator.h"
#include "core/memory_record.h"
#include "core/types.h"
#include "forget/forget_engine.h"
#include "gate/write_gate.h"
#include "lineage/lineage_index.h"
#include "lineage/propagator.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

using namespace amind;

// ═══════════════════════════════════════════════════════════════════════════
// Shared test infrastructure
// ═══════════════════════════════════════════════════════════════════════════

class V2IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // ── FeatureGate: all V2 features enabled ─────────────────────────
        V2Config v2cfg;
        v2cfg.write_gate_enabled = true;
        v2cfg.lineage_propagation_enabled = true;
        v2cfg.forget_score_enabled = true;
        v2cfg.conflict_resolver_enabled = true;
        v2cfg.consolidation_enabled = true;
        v2cfg.global_shadow_mode = false;
        feature_gate_ = std::make_unique<FeatureGate>(v2cfg);

        // ── WriteGate: enforce mode ──────────────────────────────────────
        WriteGateConfig gate_cfg;
        gate_cfg.shadow_mode = false;
        gate_cfg.duplicate_threshold = 0.95f;
        write_gate_ = std::make_unique<WriteGate>(gate_cfg);

        // ── Lineage ──────────────────────────────────────────────────────
        lineage_ = std::make_unique<LineageIndex>();

        // ── Forget ───────────────────────────────────────────────────────
        ForgetConfig forget_cfg;
        forget_cfg.shadow_mode = false;
        forget_engine_ = std::make_unique<ForgetEngine>(forget_cfg);

        // ── Coordinator ──────────────────────────────────────────────────
        remove_coordinator_ = std::make_unique<RemoveCoordinator>(
            *lineage_, *forget_engine_);

        // ── DerivedExtractor ─────────────────────────────────────────────
        derived_extractor_ = std::make_unique<DerivedExtractor>(
            *write_gate_, *lineage_);

        // ── Conflict ─────────────────────────────────────────────────────
        ConflictResolverConfig conflict_cfg;
        conflict_cfg.enable_llm_judge = false;
        conflict_resolver_ = std::make_unique<ConflictResolver>(conflict_cfg);

        // ── Consolidation ────────────────────────────────────────────────
        consolidation_ = std::make_unique<ConsolidationWorker>();
    }

    // ── In-memory record store ───────────────────────────────────────────

    uint64_t next_id_{1};

    uint64_t storeRecord(MemoryRecord record) {
        uint64_t id = next_id_++;
        record.memory_id = id;
        records_[id] = std::move(record);
        return id;
    }

    MemoryRecord makeRawRecord(const std::string& content,
                                float importance = 0.7f) {
        MemoryRecord record;
        record.content = content;
        record.importance = importance;
        record.layer = MemoryLayer::Raw;
        record.confidence_level = Confidence::Inferred;
        record.scope = MemoryScope::Private;
        record.memory_type = MemoryType::Ephemeral;
        record.tier = MemoryTier::Working;
        record.user_id = "test_user";
        record.agent_id = "agent-001";
        record.source_tier = SourceTier::Assertion;
        record.embedding = {0.1f, 0.2f, 0.3f, 0.4f};
        record.created_at = MemoryRecord::currentTimeSec();
        record.last_accessed = record.created_at;
        return record;
    }

    // ── Callback factories ───────────────────────────────────────────────

    SimilaritySearchFunc noDuplicateSearch() {
        return [](const std::vector<float>&, size_t)
            -> std::vector<std::pair<uint64_t, float>> {
            return {};
        };
    }

    SimilaritySearchFunc searchReturning(
        std::vector<std::pair<uint64_t, float>> results) {
        return [results](const std::vector<float>&, size_t)
            -> std::vector<std::pair<uint64_t, float>> {
            return results;
        };
    }

    StoreFunc storeFunc() {
        return [this](MemoryRecord record) -> uint64_t {
            return storeRecord(std::move(record));
        };
    }

    GetRecordFunc getFunc() {
        return [this](uint64_t id) -> MemoryRecord* {
            auto it = records_.find(id);
            return (it != records_.end()) ? &it->second : nullptr;
        };
    }

    PersistFunc persistFunc() {
        return [this](uint64_t id, const MemoryRecord& record) {
            records_[id] = record;
        };
    }

    std::vector<uint64_t> hnsw_deleted_;
    HnswSoftDeleteFunc hnswFunc() {
        return [this](uint64_t id) { hnsw_deleted_.push_back(id); };
    }

    CosineSimilarityFunc cosineSim() {
        return [](const std::vector<float>& a,
                  const std::vector<float>& b) -> float {
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

    // ── Module instances ─────────────────────────────────────────────────

    std::unordered_map<uint64_t, MemoryRecord> records_;

    std::unique_ptr<FeatureGate> feature_gate_;
    std::unique_ptr<WriteGate> write_gate_;
    std::unique_ptr<LineageIndex> lineage_;
    std::unique_ptr<ForgetEngine> forget_engine_;
    std::unique_ptr<RemoveCoordinator> remove_coordinator_;
    std::unique_ptr<DerivedExtractor> derived_extractor_;
    std::unique_ptr<ConflictResolver> conflict_resolver_;
    std::unique_ptr<ConsolidationWorker> consolidation_;
};

// ═══════════════════════════════════════════════════════════════════════════
// Scenario 1: Full Memory Lifecycle
//   Store Raw → Extract Derived → Lineage → Remove Raw → Cascade Invalidate
//   → ForgetLog audit → Consolidation
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(V2IntegrationTest, FullMemoryLifecycle) {
    // ── Step 1: Store raw memory ─────────────────────────────────────────
    auto raw = makeRawRecord("I always start my workday at 9am with coffee");
    uint64_t raw_id = storeRecord(raw);

    ASSERT_NE(raw_id, 0u);
    EXPECT_EQ(records_[raw_id].layer, MemoryLayer::Raw);

    // ── Step 2: Extract derived facts via DerivedExtractor ───────────────
    //    (simulates Stage 2 of CapturePipeline)
    std::vector<DerivedCandidate> candidates = {
        {"User starts work at 9am daily", 0.8f, SourceTier::Behavioral,
         {0.1f, 0.2f, 0.3f, 0.4f}, {}, MemoryScope::Private, MemoryType::UserProfile},
        {"User drinks coffee in the morning", 0.6f, SourceTier::Assertion,
         {0.2f, 0.3f, 0.4f, 0.5f}, {}, MemoryScope::Private, MemoryType::UserProfile},
    };

    auto extract_results = derived_extractor_->processFacts(
        raw_id, "agent-001", candidates,
        noDuplicateSearch(), storeFunc());

    ASSERT_EQ(extract_results.size(), 2u);
    EXPECT_EQ(extract_results[0].decision, GateDecision::Accepted);
    EXPECT_EQ(extract_results[1].decision, GateDecision::Accepted);

    uint64_t derived_1 = extract_results[0].derived_id;
    uint64_t derived_2 = extract_results[1].derived_id;
    ASSERT_NE(derived_1, 0u);
    ASSERT_NE(derived_2, 0u);

    // ── Step 3: Verify lineage was recorded ──────────────────────────────
    auto children = lineage_->getChildren(raw_id);
    EXPECT_EQ(children.size(), 2u);

    auto parents_1 = lineage_->getParents(derived_1);
    ASSERT_EQ(parents_1.size(), 1u);
    EXPECT_EQ(parents_1[0], raw_id);

    // ── Step 4: Delete the raw memory → cascade should invalidate both ──
    ASSERT_TRUE(feature_gate_->isLineagePropagationEnabled());

    auto remove_result = remove_coordinator_->remove(
        raw_id, RemoveReason::UserDelete,
        getFunc(), persistFunc(), hnswFunc());

    EXPECT_TRUE(remove_result.success);
    EXPECT_EQ(remove_result.invalidated_descendants.size(), 2u);

    // Both derived should now be Invalidated
    EXPECT_EQ(records_[derived_1].phase, MemoryPhase::Invalidated);
    EXPECT_EQ(records_[derived_2].phase, MemoryPhase::Invalidated);

    // Raw should be Tombstone
    EXPECT_EQ(records_[raw_id].phase, MemoryPhase::Tombstone);

    // HNSW should have deleted raw + both derived
    EXPECT_EQ(hnsw_deleted_.size(), 3u);

    // ── Step 5: ForgetLog should have audit entries ──────────────────────
    auto log = forget_engine_->getLog();
    ASSERT_GE(log.size(), 3u);  // 2 invalidations + 1 primary tombstone

    // Count LineageInvalidate entries
    size_t invalidate_count = std::count_if(log.begin(), log.end(),
        [](const auto& entry) {
            return entry.decision == ForgetLogEntry::Decision::LineageInvalidate;
        });
    EXPECT_EQ(invalidate_count, 2u);

    // ── Step 6: RemoveCoordinator stats ──────────────────────────────────
    auto rm_stats = remove_coordinator_->stats();
    EXPECT_EQ(rm_stats.total_removes, 1u);
    EXPECT_EQ(rm_stats.total_invalidated, 2u);
}

// ═══════════════════════════════════════════════════════════════════════════
// Scenario 2: WriteGate Blocks Transient + Duplicate Content
//   Transient → rejected; Duplicate → rejected; Good → accepted
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(V2IntegrationTest, WriteGateFiltering) {
    auto raw = makeRawRecord("Tell me about your coding preferences");
    uint64_t raw_id = storeRecord(raw);

    // ── Sub-test A: Good fact accepted ─────────────────────────────────
    std::vector<DerivedCandidate> good_facts = {
        {"User prefers TypeScript for all new projects", 0.8f,
         SourceTier::Assertion, {0.1f, 0.2f, 0.3f, 0.4f}, {}, MemoryScope::Private, MemoryType::UserProfile},
    };

    auto results_good = derived_extractor_->processFacts(
        raw_id, "agent-001", good_facts,
        noDuplicateSearch(), storeFunc());

    ASSERT_EQ(results_good.size(), 1u);
    EXPECT_EQ(results_good[0].decision, GateDecision::Accepted);
    uint64_t first_stored_id = results_good[0].derived_id;

    // ── Sub-test B: Transient content rejected ──────────────────────────
    std::vector<DerivedCandidate> transient_facts = {
        {"ok", 0.1f, SourceTier::Inference, {0.5f, 0.5f, 0.5f, 0.5f}, {}, MemoryScope::Private, MemoryType::UserProfile},
    };

    auto results_transient = derived_extractor_->processFacts(
        raw_id, "agent-001", transient_facts,
        noDuplicateSearch(), storeFunc());

    ASSERT_EQ(results_transient.size(), 1u);
    EXPECT_EQ(results_transient[0].decision, GateDecision::Rejected);

    // ── Sub-test C: Duplicate rejected ──────────────────────────────────
    std::vector<DerivedCandidate> dup_facts = {
        {"User always prefers TypeScript", 0.7f, SourceTier::Behavioral,
         {0.1f, 0.2f, 0.3f, 0.4f}, {}, MemoryScope::Private, MemoryType::UserProfile},
    };

    // Search returns high similarity to previously stored fact
    auto dup_search = searchReturning({{first_stored_id, 0.99f}});

    auto results_dup = derived_extractor_->processFacts(
        raw_id, "agent-001", dup_facts,
        dup_search, storeFunc());

    ASSERT_EQ(results_dup.size(), 1u);
    EXPECT_EQ(results_dup[0].decision, GateDecision::Rejected);

    // Only 1 lineage entry (only good fact was stored as derived)
    auto children = lineage_->getChildren(raw_id);
    EXPECT_EQ(children.size(), 1u);
}

// ═══════════════════════════════════════════════════════════════════════════
// Scenario 3: Multi-Parent Lineage — Partial Invalidation
//   Derived memory has two parents. Deleting one parent should NOT
//   invalidate the derived (still has a live parent).
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(V2IntegrationTest, MultiParentPartialInvalidation) {
    // Two raw memories
    auto raw_a = makeRawRecord("User uses React for frontend");
    auto raw_b = makeRawRecord("User prefers TypeScript");
    uint64_t id_a = storeRecord(raw_a);
    uint64_t id_b = storeRecord(raw_b);

    // One derived memory with BOTH as parents
    MemoryRecord derived;
    derived.content = "User builds React+TypeScript frontends";
    derived.layer = MemoryLayer::Derived;
    derived.confidence_level = Confidence::Inferred;
    derived.importance = 0.8f;
    derived.embedding = {0.3f, 0.3f, 0.3f, 0.3f};
    uint64_t derived_id = storeRecord(derived);

    // Record lineage with two parents
    lineage_->recordLineage(derived_id, {id_a, id_b}, LineageOp::Aggregate);

    // Delete parent A
    auto result = remove_coordinator_->remove(
        id_a, RemoveReason::UserDelete,
        getFunc(), persistFunc(), hnswFunc());

    EXPECT_TRUE(result.success);

    // Derived should NOT be invalidated (parent B is still alive)
    EXPECT_TRUE(result.invalidated_descendants.empty());
    ASSERT_EQ(result.recomputed_descendants.size(), 1u);
    EXPECT_EQ(result.recomputed_descendants[0], derived_id);

    // Derived remains Active
    EXPECT_EQ(records_[derived_id].phase, MemoryPhase::Active);

    // Now delete parent B → derived becomes orphan
    hnsw_deleted_.clear();
    auto result2 = remove_coordinator_->remove(
        id_b, RemoveReason::UserDelete,
        getFunc(), persistFunc(), hnswFunc());

    EXPECT_TRUE(result2.success);
    ASSERT_EQ(result2.invalidated_descendants.size(), 1u);
    EXPECT_EQ(result2.invalidated_descendants[0], derived_id);
    EXPECT_EQ(records_[derived_id].phase, MemoryPhase::Invalidated);
}

// ═══════════════════════════════════════════════════════════════════════════
// Scenario 4: Conflict Detection + Resolution
//   Two conflicting memories → ConflictResolver picks winner
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(V2IntegrationTest, ConflictDetectionAndResolution) {
    // Memory A: old, Inferred
    MemoryRecord mem_a;
    mem_a.memory_id = 100;
    mem_a.content = "User prefers Python for data science";
    mem_a.confidence_level = Confidence::Inferred;
    mem_a.source_tier = SourceTier::Assertion;
    mem_a.created_at = 1000;
    mem_a.importance = 0.6f;
    records_[100] = mem_a;

    // Memory B: newer, Verified
    MemoryRecord mem_b;
    mem_b.memory_id = 200;
    mem_b.content = "User now prefers R for data science";
    mem_b.confidence_level = Confidence::Verified;
    mem_b.source_tier = SourceTier::Assertion;
    mem_b.created_at = 2000;
    mem_b.importance = 0.8f;
    records_[200] = mem_b;

    // Resolve conflict
    auto resolution = conflict_resolver_->resolve(mem_a, mem_b);

    // L2: Confidence → Verified (B) wins over Inferred (A)
    EXPECT_EQ(resolution.level, ResolutionLevel::Confidence);
    EXPECT_EQ(resolution.winner_id, 200u);
    EXPECT_EQ(resolution.loser_id, 100u);
    EXPECT_FALSE(resolution.coexist);

    // In a real system, the loser would be archived via RemoveCoordinator
    auto remove_result = remove_coordinator_->remove(
        resolution.loser_id, RemoveReason::ConflictLoser,
        getFunc(), persistFunc(), hnswFunc());

    EXPECT_TRUE(remove_result.success);
    EXPECT_EQ(records_[100].phase, MemoryPhase::Tombstone);

    // ForgetLog should record the conflict resolution removal
    auto log = forget_engine_->getLog();
    bool found_conflict = false;
    for (const auto& entry : log) {
        if (entry.memory_id == 100 &&
            entry.reason.find("ConflictLoser") != std::string::npos) {
            found_conflict = true;
        }
    }
    EXPECT_TRUE(found_conflict);

    // Conflict resolver stats
    auto cr_stats = conflict_resolver_->stats();
    EXPECT_EQ(cr_stats.total_resolved, 1u);
    EXPECT_EQ(cr_stats.by_confidence, 1u);
}

// ═══════════════════════════════════════════════════════════════════════════
// Scenario 5: ForgetEngine Scoring + GC Evaluation
//   Compute forget scores for a batch, verify sorting + thresholds
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(V2IntegrationTest, ForgetEngineGcCycle) {
    ASSERT_TRUE(feature_gate_->isForgetScoreEnabled());

    // Build a batch of memories with varying signals
    std::vector<std::pair<uint64_t, ForgetSignals>> batch;

    // Memory 1: Healthy — low staleness, verified
    ForgetSignals healthy;
    healthy.staleness = 0.1f;
    healthy.verified_bonus = 1.0f;
    batch.push_back({1, healthy});

    // Memory 2: Stale + low access + conflict → should be archived (score ~0.75)
    ForgetSignals stale;
    stale.staleness = 1.0f;
    stale.low_access = 1.0f;
    stale.low_importance = 0.5f;
    stale.conflict_penalty = 1.0f;
    batch.push_back({2, stale});

    // Memory 3: Very stale + conflicted + redundant → tombstone candidate
    ForgetSignals very_stale;
    very_stale.staleness = 1.0f;
    very_stale.low_access = 1.0f;
    very_stale.low_importance = 1.0f;
    very_stale.conflict_penalty = 1.0f;
    very_stale.redundancy = 1.0f;
    batch.push_back({3, very_stale});

    auto results = forget_engine_->runCycle(batch);

    // Should be sorted by score descending
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0].memory_id, 3u);  // highest score
    EXPECT_EQ(results[2].memory_id, 1u);  // lowest score

    // Memory 3 should be recommended for Tombstone
    EXPECT_EQ(results[0].recommended_action,
              ForgetLogEntry::Decision::Tombstone);

    // Memory 2 should be recommended for Archive
    auto mem2_it = std::find_if(results.begin(), results.end(),
        [](const auto& r) { return r.memory_id == 2; });
    ASSERT_NE(mem2_it, results.end());
    EXPECT_EQ(mem2_it->recommended_action, ForgetLogEntry::Decision::Archive);

    // Memory 1 should be below decay threshold
    auto mem1_it = std::find_if(results.begin(), results.end(),
        [](const auto& r) { return r.memory_id == 1; });
    ASSERT_NE(mem1_it, results.end());
    EXPECT_LT(mem1_it->forget_score, 0.3f);
}

// ═══════════════════════════════════════════════════════════════════════════
// Scenario 6: Consolidation — Promote + Dedup + Drift
//   Session-level promotion → cross-session dedup → drift check
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(V2IntegrationTest, ConsolidationFullCycle) {
    ASSERT_TRUE(feature_gate_->isConsolidationEnabled());

    // Session 1: two memories with different importance
    MemoryRecord s1m1;
    s1m1.memory_id = 10;
    s1m1.importance = 0.9f;
    s1m1.access_count = 5;
    s1m1.confidence_level = Confidence::Inferred;
    s1m1.scope = MemoryScope::Private;
    s1m1.memory_type = MemoryType::Ephemeral;
    s1m1.embedding = {0.1f, 0.2f, 0.3f};

    MemoryRecord s1m2;
    s1m2.memory_id = 11;
    s1m2.importance = 0.2f;
    s1m2.access_count = 1;
    s1m2.confidence_level = Confidence::Inferred;
    s1m2.scope = MemoryScope::Private;
    s1m2.memory_type = MemoryType::Ephemeral;
    s1m2.embedding = {0.1f, 0.2f, 0.3f};

    // Session 2: one high-importance memory
    MemoryRecord s2m1;
    s2m1.memory_id = 20;
    s2m1.importance = 0.8f;
    s2m1.access_count = 10;
    s2m1.confidence_level = Confidence::Inferred;
    s2m1.scope = MemoryScope::Private;
    s2m1.memory_type = MemoryType::Ephemeral;
    s2m1.embedding = {0.4f, 0.5f, 0.6f};

    std::vector<std::vector<MemoryRecord>> sessions = {{s1m1, s1m2}, {s2m1}};

    // Drift candidate: derived vs raw with low similarity
    MemoryRecord drifted_derived;
    drifted_derived.memory_id = 100;
    drifted_derived.embedding = {0.1f, 0.2f, 0.3f};

    MemoryRecord raw_parent;
    raw_parent.memory_id = 50;
    raw_parent.embedding = {0.9f, -0.1f, 0.0f};  // very different

    std::vector<std::pair<MemoryRecord, MemoryRecord>> drift_candidates = {
        {drifted_derived, raw_parent}
    };

    auto result = consolidation_->runCycle(sessions, cosineSim(), drift_candidates);

    EXPECT_EQ(result.sessions_processed, 2u);
    EXPECT_GE(result.memories_promoted, 1u);  // at least s1m1 promoted
    EXPECT_EQ(result.drift_checked, 1u);
    EXPECT_EQ(result.drift_invalidated, 1u);  // drifted → invalidated

    auto stats = consolidation_->stats();
    EXPECT_EQ(stats.cycles_run, 1u);
}

// ═══════════════════════════════════════════════════════════════════════════
// Scenario 7: FeatureGate Controls V2 Behavior
//   When features are disabled, V2 logic is bypassed
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(V2IntegrationTest, FeatureGateControlsFlow) {
    // Start with all features disabled
    feature_gate_->disableAll();

    EXPECT_FALSE(feature_gate_->isWriteGateEnabled());
    EXPECT_FALSE(feature_gate_->isLineagePropagationEnabled());
    EXPECT_FALSE(feature_gate_->isForgetScoreEnabled());
    EXPECT_FALSE(feature_gate_->isConflictResolverEnabled());
    EXPECT_FALSE(feature_gate_->isConsolidationEnabled());
    EXPECT_TRUE(feature_gate_->isGlobalShadowMode());
    EXPECT_EQ(feature_gate_->enabledCount(), 0);

    // Enable one by one (gradual rollout)
    feature_gate_->setWriteGateEnabled(true);
    EXPECT_EQ(feature_gate_->enabledCount(), 1);

    feature_gate_->setLineagePropagationEnabled(true);
    EXPECT_EQ(feature_gate_->enabledCount(), 2);

    feature_gate_->setForgetScoreEnabled(true);
    EXPECT_EQ(feature_gate_->enabledCount(), 3);

    // Snapshot reflects current state
    auto snap = feature_gate_->snapshot();
    EXPECT_TRUE(snap.write_gate_enabled);
    EXPECT_TRUE(snap.lineage_propagation_enabled);
    EXPECT_TRUE(snap.forget_score_enabled);
    EXPECT_FALSE(snap.conflict_resolver_enabled);

    // Emergency kill switch
    feature_gate_->disableAll();
    EXPECT_EQ(feature_gate_->enabledCount(), 0);
    EXPECT_TRUE(feature_gate_->isGlobalShadowMode());
}

// ═══════════════════════════════════════════════════════════════════════════
// Scenario 8: Shadow Mode — Gate computes but doesn't block
//   WriteGate in shadow mode → duplicates pass through
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(V2IntegrationTest, ShadowModePassesThrough) {
    // Switch WriteGate to shadow mode
    WriteGateConfig shadow_cfg;
    shadow_cfg.shadow_mode = true;
    shadow_cfg.duplicate_threshold = 0.95f;
    auto shadow_gate = std::make_unique<WriteGate>(shadow_cfg);

    auto shadow_extractor = std::make_unique<DerivedExtractor>(
        *shadow_gate, *lineage_);

    auto raw = makeRawRecord("Shadow mode test content");
    uint64_t raw_id = storeRecord(raw);

    // This is a "duplicate" — high similarity
    DerivedCandidate dup_candidate;
    dup_candidate.content = "Shadow mode duplicate content";
    dup_candidate.importance = 0.5f;
    dup_candidate.embedding = {0.1f, 0.2f, 0.3f, 0.4f};

    // Search returns high similarity
    auto dup_search = searchReturning({{999, 0.99f}});

    auto results = shadow_extractor->processFacts(
        raw_id, "agent-001",
        {dup_candidate},
        dup_search, storeFunc());

    // In shadow mode, gate decision is overridden to Accepted
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].decision, GateDecision::Accepted);
    EXPECT_NE(results[0].derived_id, 0u);

    // Shadow override should be counted in gate stats
    auto gate_stats = shadow_gate->stats();
    EXPECT_GE(gate_stats.shadow_overrides, 1u);
}

// ═══════════════════════════════════════════════════════════════════════════
// Scenario 9: Deep Cascade Chain
//   A → B → C → D: delete A → all three descendants invalidated
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(V2IntegrationTest, DeepCascadeChain) {
    // Build chain: A → B → C → D
    auto raw_a = makeRawRecord("Root fact");
    uint64_t id_a = storeRecord(raw_a);

    MemoryRecord mem_b;
    mem_b.content = "Derived B from A";
    mem_b.layer = MemoryLayer::Derived;
    mem_b.embedding = {0.2f, 0.3f, 0.4f, 0.5f};
    uint64_t id_b = storeRecord(mem_b);

    MemoryRecord mem_c;
    mem_c.content = "Derived C from B";
    mem_c.layer = MemoryLayer::Derived;
    mem_c.embedding = {0.3f, 0.4f, 0.5f, 0.6f};
    uint64_t id_c = storeRecord(mem_c);

    MemoryRecord mem_d;
    mem_d.content = "Derived D from C";
    mem_d.layer = MemoryLayer::Derived;
    mem_d.embedding = {0.4f, 0.5f, 0.6f, 0.7f};
    uint64_t id_d = storeRecord(mem_d);

    lineage_->recordLineage(id_b, {id_a}, LineageOp::Infer);
    lineage_->recordLineage(id_c, {id_b}, LineageOp::Summarize);
    lineage_->recordLineage(id_d, {id_c}, LineageOp::Infer);

    // Delete root A → cascade through B, C, D
    auto result = remove_coordinator_->remove(
        id_a, RemoveReason::UserDelete,
        getFunc(), persistFunc(), hnswFunc());

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.invalidated_descendants.size(), 3u);

    EXPECT_EQ(records_[id_b].phase, MemoryPhase::Invalidated);
    EXPECT_EQ(records_[id_c].phase, MemoryPhase::Invalidated);
    EXPECT_EQ(records_[id_d].phase, MemoryPhase::Invalidated);

    // HNSW: 1 (root) + 3 (cascade) = 4
    EXPECT_EQ(hnsw_deleted_.size(), 4u);

    // ForgetLog: 3 invalidations + 1 tombstone
    auto log = forget_engine_->getLog();
    EXPECT_EQ(log.size(), 4u);
}

// ═══════════════════════════════════════════════════════════════════════════
// Scenario 10: Cross-Module Stats Consistency
//   After running multiple operations, verify all stats are consistent
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(V2IntegrationTest, CrossModuleStatsConsistency) {
    // ── Phase 1: Store + Extract ─────────────────────────────────────────
    auto raw1 = makeRawRecord("First user observation");
    uint64_t raw1_id = storeRecord(raw1);

    auto raw2 = makeRawRecord("Second user observation");
    uint64_t raw2_id = storeRecord(raw2);

    std::vector<DerivedCandidate> facts1 = {
        {"Derived fact from first observation", 0.8f, SourceTier::Inference,
         {0.1f, 0.2f, 0.3f, 0.4f}, {}, MemoryScope::Private, MemoryType::UserProfile},
    };
    std::vector<DerivedCandidate> facts2 = {
        {"Derived fact from second observation", 0.7f, SourceTier::Assertion,
         {0.5f, 0.6f, 0.7f, 0.8f}, {}, MemoryScope::Private, MemoryType::UserProfile},
    };

    derived_extractor_->processFacts(
        raw1_id, "agent-001", facts1, noDuplicateSearch(), storeFunc());
    derived_extractor_->processFacts(
        raw2_id, "agent-001", facts2, noDuplicateSearch(), storeFunc());

    auto ext_stats = derived_extractor_->stats();
    EXPECT_EQ(ext_stats.total_processed, 2u);
    EXPECT_EQ(ext_stats.accepted, 2u);

    // ── Phase 2: Conflict Resolution ─────────────────────────────────────
    MemoryRecord conf_a;
    conf_a.memory_id = 500;
    conf_a.confidence_level = Confidence::Inferred;
    conf_a.source_tier = SourceTier::Assertion;
    conf_a.created_at = 100;

    MemoryRecord conf_b;
    conf_b.memory_id = 600;
    conf_b.confidence_level = Confidence::Inferred;
    conf_b.source_tier = SourceTier::Behavioral;
    conf_b.created_at = 100;

    conflict_resolver_->resolve(conf_a, conf_b);

    auto cr_stats = conflict_resolver_->stats();
    EXPECT_EQ(cr_stats.total_resolved, 1u);
    EXPECT_EQ(cr_stats.by_source_tier, 1u);  // Behavioral > Assertion

    // ── Phase 3: Remove ──────────────────────────────────────────────────
    remove_coordinator_->remove(
        raw1_id, RemoveReason::UserDelete,
        getFunc(), persistFunc(), hnswFunc());

    auto rm_stats = remove_coordinator_->stats();
    EXPECT_EQ(rm_stats.total_removes, 1u);
    EXPECT_EQ(rm_stats.total_invalidated, 1u);

    // ── Phase 4: Verify ForgetLog aggregation ────────────────────────────
    auto total_log = forget_engine_->logSize();
    EXPECT_GE(total_log, 2u);  // at least invalidation + tombstone

    // ── Phase 5: All stats are resettable ────────────────────────────────
    derived_extractor_->resetStats();
    conflict_resolver_->resetStats();
    remove_coordinator_->resetStats();
    forget_engine_->clearLog();

    EXPECT_EQ(derived_extractor_->stats().total_processed, 0u);
    EXPECT_EQ(conflict_resolver_->stats().total_resolved, 0u);
    EXPECT_EQ(remove_coordinator_->stats().total_removes, 0u);
    EXPECT_EQ(forget_engine_->logSize(), 0u);
}
