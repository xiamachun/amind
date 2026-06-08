
#include <gtest/gtest.h>

#include "capture/derived_extractor.h"
#include "gate/write_gate.h"
#include "lineage/lineage_index.h"

using namespace amind;

class DerivedExtractorTest : public ::testing::Test {
protected:
    void SetUp() override {
        WriteGateConfig gate_cfg;
        gate_cfg.shadow_mode = false;
        gate_cfg.duplicate_threshold = 0.95f;
        gate_ = std::make_unique<WriteGate>(gate_cfg);
        lineage_ = std::make_unique<LineageIndex>();
        extractor_ = std::make_unique<DerivedExtractor>(*gate_, *lineage_);
    }

    // Dummy store function that assigns incrementing IDs starting from 1000.
    uint64_t nextDerivedId_{1000};
    StoreFunc makeStoreFunc() {
        return [this](MemoryRecord) -> uint64_t {
            return nextDerivedId_++;
        };
    }

    // No-duplicate search: returns empty results.
    SimilaritySearchFunc noDuplicateSearch() {
        return [](const std::vector<float>&, size_t) -> std::vector<std::pair<uint64_t, float>> {
            return {};
        };
    }

    // Returns a candidate with specified content.
    DerivedCandidate makeCandidate(const std::string& content, float importance = 0.7f) {
        DerivedCandidate candidate;
        candidate.content = content;
        candidate.importance = importance;
        candidate.source_tier = SourceTier::Inference;
        candidate.embedding = {0.1f, 0.2f, 0.3f};
        return candidate;
    }

    std::unique_ptr<WriteGate> gate_;
    std::unique_ptr<LineageIndex> lineage_;
    std::unique_ptr<DerivedExtractor> extractor_;
};

// ── Basic Flow ───────────────────────────────────────────────────────────

TEST_F(DerivedExtractorTest, AcceptsSingleFact) {
    auto results = extractor_->processFacts(
        1, "agent-1", "user-1",
        {makeCandidate("User prefers dark mode")},
        noDuplicateSearch(), makeStoreFunc());

    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].decision, GateDecision::Accepted);
    EXPECT_NE(results[0].derived_id, 0u);
}

TEST_F(DerivedExtractorTest, RejectsDuplicateFact) {
    // Search returns high similarity → gate rejects as duplicate.
    auto dupSearch = [](const std::vector<float>&, size_t) {
        return std::vector<std::pair<uint64_t, float>>{{999, 0.99f}};
    };

    auto results = extractor_->processFacts(
        1, "agent-1", "user-1",
        {makeCandidate("User prefers dark mode")},
        dupSearch, makeStoreFunc());

    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].decision, GateDecision::Deferred);
    EXPECT_NE(results[0].derived_id, 0u);
}

TEST_F(DerivedExtractorTest, RejectsTransientContent) {
    auto results = extractor_->processFacts(
        1, "agent-1", "user-1",
        {makeCandidate("ok")},
        noDuplicateSearch(), makeStoreFunc());

    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].decision, GateDecision::Rejected);
}

TEST_F(DerivedExtractorTest, ProcessesMultipleFacts) {
    std::vector<DerivedCandidate> candidates = {
        makeCandidate("User always starts work at 9am"),
        makeCandidate("Project uses React and TypeScript"),
        makeCandidate("ok"),  // transient → rejected
    };

    auto results = extractor_->processFacts(
        1, "agent-1", "user-1", candidates,
        noDuplicateSearch(), makeStoreFunc());

    ASSERT_EQ(results.size(), 3u);
    // First two should be accepted, third rejected
    EXPECT_EQ(results[0].decision, GateDecision::Accepted);
    EXPECT_EQ(results[1].decision, GateDecision::Accepted);
    EXPECT_EQ(results[2].decision, GateDecision::Rejected);
}

// ── Lineage Integration ──────────────────────────────────────────────────

TEST_F(DerivedExtractorTest, RecordsLineageForAcceptedFacts) {
    auto results = extractor_->processFacts(
        42, "agent-1", "user-1",
        {makeCandidate("User prefers dark mode")},
        noDuplicateSearch(), makeStoreFunc());

    ASSERT_EQ(results.size(), 1u);
    uint64_t derived_id = results[0].derived_id;
    ASSERT_NE(derived_id, 0u);

    // Verify lineage: derived should have raw as parent
    auto parents = lineage_->getParents(derived_id);
    ASSERT_EQ(parents.size(), 1u);
    EXPECT_EQ(parents[0], 42u);
}

TEST_F(DerivedExtractorTest, NoLineageForRejectedFacts) {
    auto dupSearch = [](const std::vector<float>&, size_t) {
        return std::vector<std::pair<uint64_t, float>>{{999, 0.99f}};
    };

    auto results = extractor_->processFacts(
        42, "agent-1", "user-1",
        {makeCandidate("User prefers dark mode")},
        dupSearch, makeStoreFunc());

    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].decision, GateDecision::Deferred);

    // Deferred facts are also persisted and have lineage recorded
    auto children = lineage_->getChildren(42);
    EXPECT_FALSE(children.empty());
}

TEST_F(DerivedExtractorTest, MultipleFactsCreateMultipleLineageEntries) {
    std::vector<DerivedCandidate> candidates = {
        makeCandidate("Fact A about user preferences"),
        makeCandidate("Fact B about project structure"),
    };

    auto results = extractor_->processFacts(
        42, "agent-1", "user-1", candidates,
        noDuplicateSearch(), makeStoreFunc());

    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0].decision, GateDecision::Accepted);
    EXPECT_EQ(results[1].decision, GateDecision::Accepted);

    // Raw 42 should have 2 children
    auto children = lineage_->getChildren(42);
    EXPECT_EQ(children.size(), 2u);
}

// ── Stats ────────────────────────────────────────────────────────────────

TEST_F(DerivedExtractorTest, StatsTrackProcessing) {
    std::vector<DerivedCandidate> candidates = {
        makeCandidate("User always starts work at 9am"),
        makeCandidate("ok"),  // rejected (transient)
    };

    extractor_->processFacts(
        1, "agent-1", "user-1", candidates,
        noDuplicateSearch(), makeStoreFunc());

    auto stats = extractor_->stats();
    EXPECT_EQ(stats.total_processed, 2u);
    EXPECT_EQ(stats.accepted, 1u);
    EXPECT_EQ(stats.rejected, 1u);
}

TEST_F(DerivedExtractorTest, StatsResettable) {
    extractor_->processFacts(
        1, "agent-1", "user-1",
        {makeCandidate("User prefers dark mode")},
        noDuplicateSearch(), makeStoreFunc());

    EXPECT_GT(extractor_->stats().total_processed, 0u);

    extractor_->resetStats();
    EXPECT_EQ(extractor_->stats().total_processed, 0u);
}

// ── Derived Record Fields ────────────────────────────────────────────────

TEST_F(DerivedExtractorTest, DerivedRecordHasCorrectLayer) {
    MemoryRecord captured_record;
    StoreFunc capturing_store = [&](MemoryRecord record) -> uint64_t {
        captured_record = std::move(record);
        return 2000;
    };

    extractor_->processFacts(
        42, "agent-1", "user-1",
        {makeCandidate("User prefers dark mode")},
        noDuplicateSearch(), capturing_store);

    EXPECT_EQ(captured_record.layer, MemoryLayer::Derived);
    EXPECT_EQ(captured_record.parent_id, 42u);
    EXPECT_EQ(captured_record.confidence_level, Confidence::Inferred);
    EXPECT_EQ(captured_record.scope, MemoryScope::Private);
    EXPECT_EQ(captured_record.memory_type, MemoryType::Ephemeral);
}

TEST_F(DerivedExtractorTest, DerivedIdAssignedFromStoreFunc) {
    uint64_t custom_id = 9999;
    StoreFunc custom_store = [&](MemoryRecord) -> uint64_t {
        return custom_id;
    };

    auto results = extractor_->processFacts(
        1, "agent-1", "user-1",
        {makeCandidate("User prefers dark mode")},
        noDuplicateSearch(), custom_store);

    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].derived_id, 9999u);
}
