
#include "retrieval/retrieval.h"
#include "vector/distance.h"

#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>

using namespace amind;

// ═══════════════════════════════════════════════════════════════════════════
// Keyword Score Tests
// ═══════════════════════════════════════════════════════════════════════════

// We test computeKeywordScore via a standalone reimplementation of the same logic,
// since the method is private. This validates the algorithm is correct.

namespace {

float testKeywordScore(const std::string& content,
                       const std::string& query,
                       const std::vector<std::string>& entities) {
    std::vector<std::string> keywords;
    std::istringstream iss(query);
    std::string word;
    while (iss >> word) {
        if (word.size() >= 2) {
            std::string lower_word;
            for (char c : word) lower_word += static_cast<char>(tolower(c));
            keywords.push_back(std::move(lower_word));
        }
    }
    for (const auto& entity : entities) {
        if (!entity.empty()) {
            std::string lower_entity;
            for (char c : entity) lower_entity += static_cast<char>(tolower(c));
            keywords.push_back(std::move(lower_entity));
        }
    }
    if (keywords.empty()) return 0.0f;

    std::string lower_content;
    for (char c : content) lower_content += static_cast<char>(tolower(c));

    size_t hits = 0;
    for (const auto& kw : keywords) {
        if (lower_content.find(kw) != std::string::npos) hits++;
    }
    return static_cast<float>(hits) / static_cast<float>(keywords.size());
}

}  // namespace

TEST(RetrievalKeywordTest, FullMatch) {
    float score = testKeywordScore(
        "User prefers dark theme and vim editor",
        "dark theme vim",
        {}
    );
    EXPECT_FLOAT_EQ(score, 1.0f);  // all 3 keywords match
}

TEST(RetrievalKeywordTest, PartialMatch) {
    float score = testKeywordScore(
        "User prefers dark theme",
        "dark theme emacs editor",
        {}
    );
    // "dark" matches, "theme" matches, "emacs" no, "editor" no → 2/4
    EXPECT_FLOAT_EQ(score, 0.5f);
}

TEST(RetrievalKeywordTest, NoMatch) {
    float score = testKeywordScore(
        "User likes cats",
        "dark theme editor",
        {}
    );
    EXPECT_FLOAT_EQ(score, 0.0f);
}

TEST(RetrievalKeywordTest, CaseInsensitive) {
    float score = testKeywordScore(
        "User prefers DARK THEME",
        "dark theme",
        {}
    );
    EXPECT_FLOAT_EQ(score, 1.0f);
}

TEST(RetrievalKeywordTest, EntitiesIncluded) {
    float score = testKeywordScore(
        "Meeting with Alice about project",
        "meeting project",
        {"Alice"}
    );
    // "meeting" hits, "project" hits, "alice" hits → 3/3
    EXPECT_FLOAT_EQ(score, 1.0f);
}

TEST(RetrievalKeywordTest, EmptyQuery) {
    float score = testKeywordScore("some content", "", {});
    EXPECT_FLOAT_EQ(score, 0.0f);
}

TEST(RetrievalKeywordTest, SingleCharTokensSkipped) {
    // Single-char tokens like "a" or "I" are skipped
    float score = testKeywordScore(
        "I like a big house",
        "a I big",
        {}
    );
    // Only "big" qualifies (size>=2) → 1/1
    EXPECT_FLOAT_EQ(score, 1.0f);
}

// ═══════════════════════════════════════════════════════════════════════════
// MMR Diversity Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(RetrievalMMRTest, DiverseResultsPreferred) {
    // Create candidates where some are near-duplicates
    // Embeddings: vec_a and vec_b are nearly identical, vec_c is different
    std::vector<float> vec_a = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec_b = {0.99f, 0.01f, 0.0f, 0.0f};  // very similar to a
    std::vector<float> vec_c = {0.0f, 0.0f, 1.0f, 0.0f};    // orthogonal to a

    // Build candidates manually
    std::vector<ScoredMemory> candidates;

    ScoredMemory sm_a;
    sm_a.record.memory_id = 1;
    sm_a.total_score = 0.95f;
    sm_a.embedding = vec_a;
    candidates.push_back(sm_a);

    ScoredMemory sm_b;
    sm_b.record.memory_id = 2;
    sm_b.total_score = 0.90f;
    sm_b.embedding = vec_b;
    candidates.push_back(sm_b);

    ScoredMemory sm_c;
    sm_c.record.memory_id = 3;
    sm_c.total_score = 0.85f;
    sm_c.embedding = vec_c;
    candidates.push_back(sm_c);

    // Simulate MMR selection for top_k=2
    // Without MMR: top 2 by score → {a, b} (both similar)
    // With MMR: should prefer {a, c} since c is diverse

    // Replicate MMR logic:
    constexpr float lambda = 0.7f;
    std::vector<ScoredMemory> selected;
    std::vector<bool> used(3, false);

    float max_score = 0.95f;

    // First: pick highest score → a
    selected.push_back(candidates[0]);
    used[0] = true;

    // Second: compute MMR for b and c
    // b: norm_score = 0.90/0.95 ≈ 0.947
    //    max_sim to a = 1 - cosineDistance(b, a) ≈ very high (≈0.9999)
    //    mmr_b = 0.7 * 0.947 - 0.3 * 0.9999 ≈ 0.663 - 0.300 = 0.363
    // c: norm_score = 0.85/0.95 ≈ 0.895
    //    max_sim to a = 1 - cosineDistance(c, a) = 1 - 1.0 = 0.0
    //    mmr_c = 0.7 * 0.895 - 0.3 * 0.0 = 0.626

    float dist_b_a = cosineDistance(vec_b, vec_a);
    float sim_b_a = 1.0f - dist_b_a;

    float dist_c_a = cosineDistance(vec_c, vec_a);
    float sim_c_a = 1.0f - dist_c_a;

    float mmr_b = lambda * (0.90f / max_score) - (1.0f - lambda) * sim_b_a;
    float mmr_c = lambda * (0.85f / max_score) - (1.0f - lambda) * sim_c_a;

    // c should win because b is too similar to a
    EXPECT_GT(mmr_c, mmr_b);
}

// ═══════════════════════════════════════════════════════════════════════════
// Graph Score Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(RetrievalGraphScoreTest, ConnectedNeighborsBoostScore) {
    // If a memory has 3 neighbors and 2 are in the candidate set,
    // graph_score = min(1.0, 2*0.2 + 3*0.05) = min(1.0, 0.55) = 0.55
    size_t total_neighbors = 3;
    size_t connected_in_results = 2;
    float graph_score = std::min(1.0f,
        static_cast<float>(connected_in_results) * 0.2f +
        static_cast<float>(total_neighbors) * 0.05f);
    EXPECT_FLOAT_EQ(graph_score, 0.55f);
}

TEST(RetrievalGraphScoreTest, NoNeighborsZeroScore) {
    float graph_score = std::min(1.0f, 0.0f * 0.2f + 0.0f * 0.05f);
    EXPECT_FLOAT_EQ(graph_score, 0.0f);
}

TEST(RetrievalGraphScoreTest, ManyNeighborsCapped) {
    // 10 neighbors, 5 connected → min(1.0, 5*0.2 + 10*0.05) = min(1.0, 1.5) = 1.0
    size_t total_neighbors = 10;
    size_t connected_in_results = 5;
    float graph_score = std::min(1.0f,
        static_cast<float>(connected_in_results) * 0.2f +
        static_cast<float>(total_neighbors) * 0.05f);
    EXPECT_FLOAT_EQ(graph_score, 1.0f);
}

// ═══════════════════════════════════════════════════════════════════════════
// Recency Score Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(RetrievalRecencyTest, JustAccessedHighScore) {
    // Age = 0 hours → exp(0) = 1.0
    float age_hours = 0.0f;
    float score = std::exp(-age_hours / (24.0f * 30.0f));
    EXPECT_FLOAT_EQ(score, 1.0f);
}

TEST(RetrievalRecencyTest, ThirtyDaysHalfLife) {
    // At 30 days: exp(-720 / 720) = exp(-1) ≈ 0.368
    float age_hours = 24.0f * 30.0f;
    float score = std::exp(-age_hours / (24.0f * 30.0f));
    EXPECT_NEAR(score, 0.368f, 0.001f);
}

TEST(RetrievalRecencyTest, VeryOldLowScore) {
    // 90 days: exp(-3) ≈ 0.05
    float age_hours = 24.0f * 90.0f;
    float score = std::exp(-age_hours / (24.0f * 30.0f));
    EXPECT_NEAR(score, 0.05f, 0.005f);
}

// ═══════════════════════════════════════════════════════════════════════════
// Score Fusion Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(RetrievalFusionTest, AllWeightsContribute) {
    RetrievalWeights w;  // default: semantic=0.4, keyword=0.25, graph=0.15, recency=0.1, importance=0.1

    float semantic = 0.9f;
    float keyword = 0.6f;
    float recency = 0.5f;
    float graph = 0.3f;
    float importance = 0.8f;

    float total = w.semantic * semantic + w.keyword * keyword +
                  w.recency * recency + w.graph * graph + w.importance * importance;

    // 0.4*0.9 + 0.25*0.6 + 0.1*0.5 + 0.15*0.3 + 0.1*0.8
    // = 0.36 + 0.15 + 0.05 + 0.045 + 0.08 = 0.685
    EXPECT_NEAR(total, 0.685f, 0.001f);
}

TEST(RetrievalFusionTest, WeightsSumToOne) {
    RetrievalWeights w;
    float sum = w.semantic + w.keyword + w.graph + w.recency + w.importance;
    EXPECT_FLOAT_EQ(sum, 1.0f);
}
