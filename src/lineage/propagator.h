
#pragma once

#include "lineage_index.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace amind {

/// Result of a single invalidation propagation.
struct PropagationResult {
    uint64_t memory_id;
    bool invalidated;         // true if marked Invalidated; false if still has live parents
    std::string reason;
};

/// Propagates lineage invalidation when a parent memory is deleted.
///
/// When a memory is deleted (Tombstone), the propagator:
/// 1. Looks up all direct children via LineageIndex reverse index
/// 2. For each child: checks if any other live parents remain
/// 3. If no live parents → marks child as Invalidated (lineage orphan)
/// 4. Recursively propagates to grandchildren (BFS, depth-limited)
///
/// Thread-safe: acquires locks on LineageIndex internally.
class LineagePropagator {
public:
    /// Callback invoked for each memory that should be invalidated.
    /// Signature: void(uint64_t memory_id) — caller is responsible for
    /// actually calling markInvalidated() on the MemoryRecord.
    using InvalidateCallback = std::function<void(uint64_t)>;

    explicit LineagePropagator(LineageIndex& index, int max_depth = 5);

    /// Propagate invalidation starting from `deleted_parent_id`.
    /// Calls `on_invalidate` for each memory that loses all live parents.
    /// Returns the list of PropagationResults.
    std::vector<PropagationResult> propagate(uint64_t deleted_parent_id,
                                              const InvalidateCallback& on_invalidate);

    /// Set maximum propagation depth (default: 5).
    void setMaxDepth(int depth);
    int maxDepth() const;

private:
    LineageIndex& index_;
    int max_depth_;
};

}  // namespace amind
