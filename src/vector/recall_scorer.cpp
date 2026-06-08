#include "recall_scorer.h"

#include <algorithm>
#include <cmath>

namespace amind {

RecallScorer::RecallScorer(RecallWeights weights)
    : weights_(weights) {}

// ── Semantic score ────────────────────────────────────────────────────────

float RecallScorer::computeSemanticScore(float cosineDistance) {
    // cosine_distance ∈ [0, 2], similarity = 1 - distance ∈ [-1, 1]
    // Clamp to [0, 1] for scoring purposes
    float similarity = 1.0f - cosineDistance;
    return std::clamp(similarity, 0.0f, 1.0f);
}

// ── Recency score ─────────────────────────────────────────────────────────

float RecallScorer::computeRecencyScore(uint64_t createdAtMs,
                                         uint64_t currentTimeMs) const {
    if (currentTimeMs <= createdAtMs) {
        return 1.0f;  // future or same time = fully recent
    }

    double elapsedMs = static_cast<double>(currentTimeMs - createdAtMs);
    double elapsedHours = elapsedMs / (1000.0 * 3600.0);

    // recency = e^(-decay_rate × hours_since)
    float recency = static_cast<float>(std::exp(-weights_.decayRate * elapsedHours));
    return std::clamp(recency, 0.0f, 1.0f);
}

// ── Frequency score ───────────────────────────────────────────────────────

float RecallScorer::computeFrequencyScore(uint32_t accessCount) {
    // log2(access_count + 1), normalized by dividing by log2(1001) ≈ 10
    // so that access_count=1000 maps to ~1.0
    constexpr float normalizer = 9.97f;  // log2(1001)
    float rawScore = std::log2(static_cast<float>(accessCount) + 1.0f);
    return std::clamp(rawScore / normalizer, 0.0f, 1.0f);
}

// ── Score a single candidate ──────────────────────────────────────────────

ScoredResult RecallScorer::score(const RecallCandidate& candidate,
                                  uint64_t currentTimeMs) const {
    ScoredResult result;
    result.memoryId = candidate.memoryId;

    result.semanticScore = computeSemanticScore(candidate.cosineDistance);
    result.recencyScore = computeRecencyScore(candidate.createdAtMs, currentTimeMs);
    result.importanceScore = std::clamp(candidate.importance, 0.0f, 1.0f);
    result.frequencyScore = computeFrequencyScore(candidate.accessCount);

    if (weights_.recencyGateEnabled) {
        // Multiplicative gating: baseScore × recencyGate
        // The recency component acts as a 0-1 multiplier on the entire score,
        // ensuring very old memories are suppressed regardless of semantic match.
        float baseScore = weights_.semantic * result.semanticScore
                        + weights_.importance * result.importanceScore
                        + weights_.frequency * result.frequencyScore;
        result.totalScore = baseScore * result.recencyScore;
    } else {
        // Additive mode (default): traditional weighted sum
        result.totalScore = weights_.semantic * result.semanticScore
                          + weights_.recency * result.recencyScore
                          + weights_.importance * result.importanceScore
                          + weights_.frequency * result.frequencyScore;
    }

    return result;
}

// ── Score and rank ────────────────────────────────────────────────────────

std::vector<ScoredResult> RecallScorer::scoreAndRank(
    const std::vector<RecallCandidate>& candidates,
    uint64_t currentTimeMs) const {

    std::vector<ScoredResult> results;
    results.reserve(candidates.size());

    for (const auto& candidate : candidates) {
        results.push_back(score(candidate, currentTimeMs));
    }

    // Sort by totalScore descending (most relevant first)
    std::sort(results.begin(), results.end(), std::greater<>());

    return results;
}

// ── Score and rank top-k ──────────────────────────────────────────────────

std::vector<ScoredResult> RecallScorer::scoreAndRankTopK(
    const std::vector<RecallCandidate>& candidates,
    uint64_t currentTimeMs,
    size_t topK) const {

    auto results = scoreAndRank(candidates, currentTimeMs);

    if (results.size() > topK) {
        results.resize(topK);
    }

    return results;
}

}  // namespace amind
