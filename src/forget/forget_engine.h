
#pragma once

#include "core/types.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace amind {

// ForgetLog class removed in Phase 4 (replaced by observability::MemoryEventLog).
// ForgetLogEntry struct kept because runForgetCycleOnce() still uses its
// Decision enum + decisionToString() helper to label emitted MemoryEvents.

/// Decision record retained as a value type — no longer persisted to its own
/// log file. The amind engine builds a MemoryEvent{kind=GcDecay|Archive|Tombstone}
/// from each instance and writes it to the unified events.log.
struct ForgetLogEntry {
    uint64_t timestamp_ms;
    uint64_t memory_id;
    enum class Decision : uint8_t {
        Decay,             // importance reduced
        Archive,           // moved to Archived phase
        Tombstone,         // moved to Tombstone
        Vacuum,            // physically removed
        DropFromHNSW,      // soft-deleted from HNSW
        ResolveConflict,   // conflict was resolved
        LineageInvalidate, // derived memory invalidated
        GateReject,        // WriteGate rejected
        GateDefer,         // WriteGate deferred
    } decision;
    std::string reason;
    MemoryPhase before_state;
    MemoryPhase after_state;
    std::vector<uint64_t> lineage_affected;
    std::string gc_worker_id;
    // Display-only context (filled by the writer); truncated for storage.
    std::string agent_id;
    std::string content_preview;
};

inline std::string decisionToString(ForgetLogEntry::Decision decision) {
    switch (decision) {
        case ForgetLogEntry::Decision::Decay:             return "Decay";
        case ForgetLogEntry::Decision::Archive:           return "Archive";
        case ForgetLogEntry::Decision::Tombstone:         return "Tombstone";
        case ForgetLogEntry::Decision::Vacuum:            return "Vacuum";
        case ForgetLogEntry::Decision::DropFromHNSW:      return "DropFromHNSW";
        case ForgetLogEntry::Decision::ResolveConflict:   return "ResolveConflict";
        case ForgetLogEntry::Decision::LineageInvalidate: return "LineageInvalidate";
        case ForgetLogEntry::Decision::GateReject:        return "GateReject";
        case ForgetLogEntry::Decision::GateDefer:         return "GateDefer";
    }
    return "Unknown";
}

/// Configuration for the ForgetEngine.
struct ForgetConfig {
    // Weights for forget_score formula
    float weight_staleness{0.25f};
    float weight_low_access{0.20f};
    float weight_low_importance{0.20f};
    float weight_conflict_penalty{0.10f};
    float weight_redundancy{0.10f};
    float weight_verified_bonus{0.10f};
    float weight_graph_centrality{0.05f};

    // Thresholds for GC actions
    float decay_threshold{0.3f};       // forget_score > this → reduce importance
    float archive_threshold{0.6f};     // forget_score > this → Archive
    float tombstone_threshold{0.85f};  // forget_score > this → Tombstone

    // GC Worker config
    float sample_ratio{0.1f};          // Fraction of memories to scan per cycle
    bool shadow_mode{true};            // Shadow mode: compute but don't execute
    uint32_t gc_interval_seconds{3600};

    // Stale threshold
    float stale_hours{168.0f};         // 7 days
};

/// Input signals for computing forget_score.
struct ForgetSignals {
    float staleness{0.0f};        // hours since last access / stale_hours
    float low_access{0.0f};       // 1 - log(1 + access_count) / log(1 + max_access)
    float low_importance{0.0f};   // 1 - importance
    float conflict_penalty{0.0f}; // 1.0 if Conflicted confidence, else 0.0
    float redundancy{0.0f};       // max cosine similarity to any neighbor (0-1)
    float verified_bonus{0.0f};   // 1.0 if Verified confidence, else 0.0
    float graph_centrality{0.0f}; // normalized degree centrality (0-1)
};

/// Result of a GC evaluation for one memory.
struct GcEvaluation {
    uint64_t memory_id;
    float forget_score;
    ForgetLogEntry::Decision recommended_action;
    std::string reason;
};

/// ForgetEngine — computes forget_score and manages GC lifecycle.
///
/// **Shadow mode** (default ON): computes scores and logs recommendations
/// but does not actually modify any memory records.
class ForgetEngine {
public:
    explicit ForgetEngine(ForgetConfig config = {});

    ~ForgetEngine();

    /// Compute forget_score for a single memory given its signals.
    float computeForgetScore(const ForgetSignals& signals) const;

    /// Evaluate a single memory and recommend a GC action.
    GcEvaluation evaluate(uint64_t memory_id, const ForgetSignals& signals) const;

    /// Run a GC cycle: evaluate a batch of memories.
    /// Returns evaluations sorted by forget_score descending.
    std::vector<GcEvaluation> runCycle(const std::vector<std::pair<uint64_t, ForgetSignals>>& batch) const;

    // ── Config ───────────────────────────────────────────────────────────

    void setShadowMode(bool enabled);
    bool isShadowMode() const;
    ForgetConfig config() const;

private:
    mutable std::mutex mutex_;
    ForgetConfig config_;
    // Phase 4: in-memory log_ and persistent_log_ removed. Engine emits
    // MemoryEvents directly to events_log_ instead.
};

}  // namespace amind
