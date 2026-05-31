
#pragma once

#include "core/memory_record.h"
#include "core/types.h"

#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace amind {

/// Which strategy level resolved the conflict.
enum class ResolutionLevel : uint8_t {
    VersionAncestor = 1,  // L1: explicit version/parent relationship
    Confidence = 2,       // L2: Verified > Inferred > Stale > Conflicted
    Recency = 3,          // L3: newer timestamp wins
    SourceTier = 4,       // L4: Behavioral > Assertion > Inference
    LlmJudge = 5,         // L5: LLM arbitration
    Coexist = 6,          // L6: both survive + ConflictsWith edge preserved
};

inline std::string resolutionLevelToString(ResolutionLevel level) {
    switch (level) {
        case ResolutionLevel::VersionAncestor: return "VersionAncestor";
        case ResolutionLevel::Confidence:      return "Confidence";
        case ResolutionLevel::Recency:         return "Recency";
        case ResolutionLevel::SourceTier:      return "SourceTier";
        case ResolutionLevel::LlmJudge:        return "LlmJudge";
        case ResolutionLevel::Coexist:         return "Coexist";
    }
    return "Unknown";
}

/// Result of conflict resolution.
struct ConflictResolution {
    uint64_t winner_id;
    uint64_t loser_id;
    ResolutionLevel level;
    std::string reason;
    bool coexist{false};  // true if L6 — both survive
};

/// Callback for LLM-based conflict judgment (L5).
/// Takes two memory contents and returns the winner ID, or nullopt if undecided.
using LlmJudgeFunc = std::function<std::optional<uint64_t>(
    uint64_t id_a, const std::string& content_a,
    uint64_t id_b, const std::string& content_b
)>;

/// Configuration for the ConflictResolver.
struct ConflictResolverConfig {
    bool enable_llm_judge{false};   // L5 is expensive; disabled by default
    float high_importance_threshold{0.8f};  // L6: if both > this, coexist
};

/// ConflictResolver — six-level strategy to resolve memory conflicts.
///
/// Given two conflicting memories (A and B), applies strategies in order:
/// L1: Version ancestor (parent_id chain → ancestor is loser)
/// L2: Confidence (Verified > Inferred > Stale > Conflicted)
/// L3: Recency (newer created_at wins)
/// L4: SourceTier (Behavioral > Assertion > Inference)
/// L5: LLM judge (optional, expensive)
/// L6: Coexist (both survive with ConflictsWith edge)
class ConflictResolver {
public:
    explicit ConflictResolver(ConflictResolverConfig config = {});

    /// Resolve conflict between two memory records.
    ConflictResolution resolve(const MemoryRecord& record_a,
                                const MemoryRecord& record_b);

    /// Set the LLM judge callback for L5 resolution.
    void setLlmJudge(LlmJudgeFunc judge);

    // ── Stats ────────────────────────────────────────────────────────────

    struct Stats {
        uint64_t total_resolved{0};
        uint64_t by_version{0};
        uint64_t by_confidence{0};
        uint64_t by_recency{0};
        uint64_t by_source_tier{0};
        uint64_t by_llm{0};
        uint64_t by_coexist{0};
    };

    Stats stats() const;
    void resetStats();

    ConflictResolverConfig config() const;

private:
    /// L1: Check if one memory is an ancestor of the other.
    std::optional<ConflictResolution> tryVersionAncestor(
        const MemoryRecord& a, const MemoryRecord& b);

    /// L2: Compare confidence levels.
    std::optional<ConflictResolution> tryConfidence(
        const MemoryRecord& a, const MemoryRecord& b);

    /// L3: Compare recency (created_at timestamp).
    std::optional<ConflictResolution> tryRecency(
        const MemoryRecord& a, const MemoryRecord& b);

    /// L4: Compare source tiers.
    std::optional<ConflictResolution> trySourceTier(
        const MemoryRecord& a, const MemoryRecord& b);

    /// L5: Ask LLM to judge.
    std::optional<ConflictResolution> tryLlmJudge(
        const MemoryRecord& a, const MemoryRecord& b);

    /// L6: Coexist — always succeeds.
    ConflictResolution coexist(const MemoryRecord& a, const MemoryRecord& b);

    static int confidenceRank(Confidence conf);
    static int sourceTierRank(SourceTier tier);

    mutable std::mutex mutex_;
    ConflictResolverConfig config_;
    Stats stats_;
    LlmJudgeFunc llm_judge_;
};

}  // namespace amind
