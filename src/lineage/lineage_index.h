
#pragma once

#include "core/types.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace amind {

class LineageWAL;

/// Record of a single lineage relationship.
struct LineageRecord {
    uint64_t child_id;
    std::vector<uint64_t> parent_ids;
    LineageOp op{LineageOp::None};
    uint32_t created_at{0};
};

/// Bidirectional lineage index — tracks parent→children and child→parents relationships.
///
/// This is the core data structure for V2 lineage tracking. When a parent memory is deleted,
/// the reverse index allows efficient lookup of all derived children for invalidation propagation.
class LineageIndex {
public:
    LineageIndex();
    ~LineageIndex();

    /// Construct with WAL persistence. Replays snapshot + WAL on construction.
    explicit LineageIndex(const std::string& data_dir);

    // ── Persistence ─────────────────────────────────────────────────────

    /// Write a snapshot and truncate WAL. No-op if constructed without data_dir.
    bool checkpoint();

    /// Get WAL file size (for deciding when to checkpoint).
    size_t walSize() const;

    // ── Core Operations ──────────────────────────────────────────────────

    /// Record that `child` was derived from `parents` via `op`.
    void recordLineage(uint64_t child, const std::vector<uint64_t>& parents, LineageOp op);

    /// Remove all lineage records involving `memory_id` (as parent or child).
    void removeNode(uint64_t memory_id);

    /// Mark a node as removed (for hasLiveParents checks) WITHOUT clearing reverse index.
    /// Used by propagator to allow continued BFS traversal while marking nodes dead.
    void markRemoved(uint64_t memory_id);

    // ── Queries ──────────────────────────────────────────────────────────

    /// Get direct parent IDs for a given child.
    std::vector<uint64_t> getParents(uint64_t child_id) const;

    /// Get direct child IDs for a given parent.
    std::vector<uint64_t> getChildren(uint64_t parent_id) const;

    /// Get all transitive descendants via BFS (depth-limited).
    std::vector<uint64_t> getTransitiveDescendants(uint64_t parent_id, int max_depth = 5) const;

    /// Get the LineageOp used to derive a child.
    LineageOp getLineageOp(uint64_t child_id) const;

    /// Check if a child has any remaining live parents (non-removed).
    bool hasLiveParents(uint64_t child_id) const;

    // ── Stats ────────────────────────────────────────────────────────────

    /// Number of child→parent mappings.
    size_t childCount() const;

    /// Number of parent→children reverse entries.
    size_t parentCount() const;

    /// Total lineage records tracked.
    size_t totalRecords() const;

private:
    void replayFromWAL();

    mutable std::mutex mutex_;

    /// Forward index: child_id → LineageRecord
    std::unordered_map<uint64_t, LineageRecord> child_to_record_;

    /// Reverse index: parent_id → {child_ids}
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> parent_to_children_;

    /// Set of removed node IDs (for hasLiveParents check)
    std::unordered_set<uint64_t> removed_nodes_;

    /// WAL for persistence (nullptr if constructed without data_dir)
    std::unique_ptr<LineageWAL> wal_;
};

}  // namespace amind
