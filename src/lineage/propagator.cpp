
#include "propagator.h"

#include <queue>
#include <spdlog/spdlog.h>
#include <unordered_set>

namespace amind {

LineagePropagator::LineagePropagator(LineageIndex& index, int max_depth)
    : index_(index), max_depth_(max_depth) {}

std::vector<PropagationResult> LineagePropagator::propagate(
    uint64_t deleted_parent_id,
    const InvalidateCallback& on_invalidate) {

    std::vector<PropagationResult> results;
    std::unordered_set<uint64_t> visited;
    std::queue<std::pair<uint64_t, int>> frontier;

    // Mark the parent as removed (but preserve reverse index for BFS traversal).
    // We use markRemoved instead of removeNode so that getChildren still works
    // during propagation — removeNode would clear the reverse index.
    index_.markRemoved(deleted_parent_id);

    // Get direct children (reverse index still intact)
    auto direct_children = index_.getChildren(deleted_parent_id);

    // Seed BFS with direct children
    for (uint64_t child_id : direct_children) {
        if (visited.count(child_id)) continue;
        visited.insert(child_id);
        frontier.push({child_id, 1});
    }

    while (!frontier.empty()) {
        auto [current_id, depth] = frontier.front();
        frontier.pop();

        bool has_live = index_.hasLiveParents(current_id);

        PropagationResult result;
        result.memory_id = current_id;

        if (!has_live) {
            result.invalidated = true;
            result.reason = "All upstream parents deleted/invalidated (lineage orphan)";

            if (on_invalidate) {
                on_invalidate(current_id);
            }

            // Mark this node as removed so downstream hasLiveParents sees it as dead
            index_.markRemoved(current_id);

            spdlog::info("LineagePropagator: invalidated memory {} (depth {})", current_id, depth);

            // Continue propagating to this node's children (it's now effectively dead)
            if (depth < max_depth_) {
                auto grandchildren = index_.getChildren(current_id);
                for (uint64_t gc_id : grandchildren) {
                    if (visited.count(gc_id)) continue;
                    visited.insert(gc_id);
                    frontier.push({gc_id, depth + 1});
                }
            }
        } else {
            result.invalidated = false;
            result.reason = "Still has live parents";
            spdlog::debug("LineagePropagator: memory {} still has live parents", current_id);
        }

        results.push_back(std::move(result));
    }

    spdlog::info("LineagePropagator: propagation from parent {} complete, {} nodes checked, {} invalidated",
                 deleted_parent_id, results.size(),
                 std::count_if(results.begin(), results.end(),
                               [](const auto& r) { return r.invalidated; }));

    return results;
}

void LineagePropagator::setMaxDepth(int depth) {
    max_depth_ = depth;
}

int LineagePropagator::maxDepth() const {
    return max_depth_;
}

}  // namespace amind
