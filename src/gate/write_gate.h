
#pragma once

#include "core/types.h"

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace amind {

/// A proposed memory candidate to be evaluated by the WriteGate.
struct ProposedMemory {
    std::string content;
    std::vector<float> embedding;
    MemoryOwner owner{MemoryOwner::Session};
    SourceTier source_tier{SourceTier::Inference};
    float importance{0.5f};
};

/// WriteGate decision with full metadata.
struct GateVerdict {
    GateDecision decision{GateDecision::Accepted};
    SourceTier tier{SourceTier::Inference};
    float marginal_value{0.0f};
    std::string reason;
};

/// Callback to search existing memories for semantic similarity.
/// Returns vector of (memory_id, cosine_similarity) pairs.
using SimilaritySearchFunc = std::function<
    std::vector<std::pair<uint64_t, float>>(const std::vector<float>& embedding, size_t top_k)
>;

/// Configuration for the WriteGate.
struct WriteGateConfig {
    float duplicate_threshold{0.95f};    // Cosine similarity above this → Rejected (duplicate)
    float low_value_threshold{0.15f};    // Marginal value below this → Rejected (low value)
    float deferred_threshold{0.30f};     // Marginal value below this → Deferred (uncertain)
    bool shadow_mode{true};              // Shadow mode: log decisions but always accept
    size_t similarity_top_k{5};          // How many neighbors to check for marginal value
};

/// WriteGate — quality gate for incoming memories.
///
/// Evaluates whether a candidate memory should be stored, based on:
/// 1. **Duplicate detection**: reject if semantically identical to existing memory
/// 2. **Marginal value**: reject if it adds negligible new information
/// 3. **Source tier classification**: Assertion > Behavioral > Inference
///
/// **Shadow mode** (default ON): logs all decisions but always returns Accepted.
/// This allows offline analysis before enabling hard rejection.
class WriteGate {
public:
    explicit WriteGate(WriteGateConfig config = {});

    /// Evaluate a candidate memory against existing store.
    /// `search_fn` is used to find similar existing memories.
    GateVerdict evaluate(const ProposedMemory& candidate,
                         const SimilaritySearchFunc& search_fn);

    // ── Configuration ────────────────────────────────────────────────────

    void setShadowMode(bool enabled);
    bool isShadowMode() const;

    void setConfig(const WriteGateConfig& config);
    WriteGateConfig config() const;

    // ── Stats ────────────────────────────────────────────────────────────

    struct Stats {
        uint64_t total_evaluated{0};
        uint64_t accepted{0};
        uint64_t rejected{0};
        uint64_t deferred{0};
        uint64_t shadow_overrides{0};  // Would have rejected but shadow mode accepted
    };

    Stats stats() const;
    void resetStats();

    /// Check if content is transient/noise (greetings, acknowledgments, etc.).
    /// Public+static so CapturePipeline can reject at Stage 1 before fastStore().
    static bool isTransientContent(const std::string& content);

    /// Check if content is a forget/delete request ("请忘掉", "请删除", "forget").
    /// CapturePipeline uses this to tombstone matching memories instead of storing.
    static bool isForgetRequest(const std::string& content);

    /// Check if content contains secrecy markers ("千万不要告诉别人", "keep secret").
    /// Used at recall time to annotate confidential memories.
    static bool hasSecrecyMarker(const std::string& content);

private:
    /// Compute marginal value of candidate given existing neighbors.
    float computeMarginalValue(const ProposedMemory& candidate,
                                const std::vector<std::pair<uint64_t, float>>& neighbors);

    /// Classify source tier based on content heuristics.
    SourceTier classifySourceTier(const ProposedMemory& candidate);

    mutable std::mutex mutex_;
    WriteGateConfig config_;
    Stats stats_;
};

}  // namespace amind
