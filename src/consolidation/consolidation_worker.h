
#pragma once

#include "core/memory_record.h"
#include "core/types.h"

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace amind {

/// Configuration for the ConsolidationWorker.
struct ConsolidationConfig {
    bool enabled{true};
    uint32_t interval_seconds{86400};     // 24 hours
    size_t top_k_per_session{5};          // Top-K to promote per session
    float dedup_threshold{0.95f};         // Cosine similarity for dedup
    float drift_threshold{0.3f};          // Drift threshold for invalidation
    float drift_sample_ratio{0.01f};      // Fraction of Derived memories to sample
};

/// Result of processing a single candidate for drift.
struct DriftCheckResult {
    uint64_t memory_id;
    float drift_score;          // 0 = perfect match, 1 = completely drifted
    bool invalidated{false};
    std::string reason;
};

/// Result of a dedup pass.
struct DedupResult {
    uint64_t kept_id;
    std::vector<uint64_t> archived_ids;
};

/// Result of a full consolidation cycle.
struct ConsolidationCycleResult {
    size_t sessions_processed{0};
    size_t memories_promoted{0};
    size_t memories_deduped{0};
    size_t drift_checked{0};
    size_t drift_invalidated{0};
};

/// Callback to compute cosine similarity between two embeddings.
using CosineSimilarityFunc = std::function<float(
    const std::vector<float>&, const std::vector<float>&)>;

/// Callback to re-embed content (for drift check).
using ReEmbedFunc = std::function<std::vector<float>(const std::string&)>;

/// ConsolidationWorker — periodic maintenance of Derived memories:
///   1. Session-level promotion (top-K to User/Project owner)
///   2. Cross-session dedup (merge semantically identical memories)
///   3. Drift check (re-compute Derived from Raw, invalidate if drifted too far)
class ConsolidationWorker {
public:
    explicit ConsolidationWorker(ConsolidationConfig config = {});

    /// Promote top-K memories from a session by importance + access count.
    /// Returns IDs of promoted memories.
    std::vector<uint64_t> promoteTopK(std::vector<MemoryRecord>& session_memories);

    /// Deduplicate a batch of memories by cosine similarity.
    /// Returns dedup results (kept + archived groups).
    std::vector<DedupResult> dedup(std::vector<MemoryRecord>& memories,
                                    const CosineSimilarityFunc& cosine_sim);

    /// Check a batch of Derived memories for drift against their Raw parents.
    /// For each, re-embeds the raw parent content and compares with the derived embedding.
    std::vector<DriftCheckResult> checkDrift(
        const std::vector<std::pair<MemoryRecord, MemoryRecord>>& derived_raw_pairs,
        const CosineSimilarityFunc& cosine_sim);

    /// Run a full consolidation cycle (combines all three steps).
    /// Takes session groups of memories + raw records lookup.
    ConsolidationCycleResult runCycle(
        std::vector<std::vector<MemoryRecord>>& session_groups,
        const CosineSimilarityFunc& cosine_sim,
        const std::vector<std::pair<MemoryRecord, MemoryRecord>>& drift_candidates);

    ConsolidationConfig config() const;

    struct Stats {
        uint64_t cycles_run{0};
        uint64_t total_promoted{0};
        uint64_t total_deduped{0};
        uint64_t total_drift_checked{0};
        uint64_t total_drift_invalidated{0};
    };

    Stats stats() const;
    void resetStats();

private:
    mutable std::mutex mutex_;
    ConsolidationConfig config_;
    Stats stats_;
};

}  // namespace amind
