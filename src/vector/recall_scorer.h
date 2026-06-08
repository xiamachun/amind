#pragma once

#include "distance.h"

#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace amind {

/// Weights for the recall scoring formula.
///
/// Default (additive):
///   score = α×semantic + β×recency + γ×importance + δ×frequency
///
/// With recency gate enabled (multiplicative):
///   baseScore = α×semantic + γ×importance + δ×frequency
///   recencyGate = e^(-decayRate × hours_since)
///   score = baseScore × recencyGate
///
/// The multiplicative gate ensures that very old memories are suppressed
/// regardless of semantic match quality (avoids stale-but-similar noise).
struct RecallWeights {
    float semantic = 0.6f;     // α: cosine similarity weight (dominant)
    float recency = 0.2f;      // β: time recency weight (unused when gate mode)
    float importance = 0.1f;   // γ: importance weight
    float frequency = 0.1f;    // δ: access frequency weight

    /// Recency decay rate: e^(-decayRate × hours_since)
    float decayRate = 0.01f;

    /// When true, use multiplicative recency gating instead of additive recency.
    bool recencyGateEnabled = false;
};

/// Input data for scoring a single memory candidate.
struct RecallCandidate {
    uint64_t memoryId;
    float cosineDistance;       // from HNSW search (0 = identical, 2 = opposite)
    float importance;          // [0.0, 1.0]
    uint32_t accessCount;
    uint64_t createdAtMs;      // unix timestamp in milliseconds
};

/// Scored result after applying the recall formula.
struct ScoredResult {
    uint64_t memoryId;
    float totalScore;          // combined score (higher = more relevant)
    float semanticScore;       // cosine similarity component
    float recencyScore;        // time recency component
    float importanceScore;     // importance component
    float frequencyScore;      // access frequency component

    bool operator>(const ScoredResult& other) const {
        return totalScore > other.totalScore;
    }

    bool operator<(const ScoredResult& other) const {
        return totalScore < other.totalScore;
    }
};

/// Recall scorer that combines semantic similarity, time recency,
/// importance, and access frequency into a single relevance score.
///
/// Formula (from DESIGN.md Section 6.2):
///   score = α × cosine_similarity
///         + β × time_recency(created_at, now)
///         + γ × importance
///         + δ × log2(access_count + 1)
///
/// where:
///   cosine_similarity = 1 - cosine_distance
///   time_recency(t, now) = e^(-decay_rate × hours_since(t, now))
class RecallScorer {
public:
    explicit RecallScorer(RecallWeights weights = {});

    /// Score a single candidate at the given current time.
    ScoredResult score(const RecallCandidate& candidate,
                       uint64_t currentTimeMs) const;

    /// Score and rank a batch of candidates. Returns results sorted by
    /// totalScore descending (most relevant first).
    std::vector<ScoredResult> scoreAndRank(
        const std::vector<RecallCandidate>& candidates,
        uint64_t currentTimeMs) const;

    /// Score and rank, then return only the top-k results.
    std::vector<ScoredResult> scoreAndRankTopK(
        const std::vector<RecallCandidate>& candidates,
        uint64_t currentTimeMs,
        size_t topK) const;

    /// Get the current weights.
    const RecallWeights& weights() const { return weights_; }

    /// Update the weights.
    void setWeights(const RecallWeights& weights) { weights_ = weights; }

private:
    /// Compute the semantic similarity score from cosine distance.
    /// Maps cosine_distance ∈ [0, 2] → similarity ∈ [-1, 1], then clamps to [0, 1].
    static float computeSemanticScore(float cosineDistance);

    /// Compute the time recency score.
    /// recency = e^(-decay_rate × hours_since)
    float computeRecencyScore(uint64_t createdAtMs,
                              uint64_t currentTimeMs) const;

    /// Compute the frequency score.
    /// frequency = log2(access_count + 1), normalized to [0, 1] range.
    static float computeFrequencyScore(uint32_t accessCount);

    RecallWeights weights_;
};

}  // namespace amind
