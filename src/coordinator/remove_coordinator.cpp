
#include "remove_coordinator.h"

#include <spdlog/spdlog.h>

namespace amind {

RemoveCoordinator::RemoveCoordinator(LineageIndex& lineage,
                                      ForgetEngine& forget_engine)
    : lineage_(lineage), forget_engine_(forget_engine) {}

RemoveResult RemoveCoordinator::remove(uint64_t memory_id,
                                        RemoveReason reason,
                                        const GetRecordFunc& get_record,
                                        const PersistFunc& persist,
                                        const HnswSoftDeleteFunc& hnsw_delete) {
    RemoveResult result;
    result.primary_id = memory_id;
    result.reason = reason;

    // Step 1: Get and validate the primary record
    auto* primary = get_record(memory_id);
    if (!primary) {
        result.success = false;
        result.error_message = "memory not found: " + std::to_string(memory_id);
        return result;
    }

    // Step 2: Mark primary as Tombstone
    MemoryPhase before_state = primary->phase;
    primary->markTombstone();
    persist(memory_id, *primary);

    spdlog::info("RemoveCoordinator: tombstoned memory {} (reason={})",
                 memory_id, removeReasonToString(reason));

    // Step 3: HNSW soft-delete for primary
    hnsw_delete(memory_id);

    // Step 4: Propagate invalidation through lineage
    LineagePropagator propagator(lineage_);
    auto propagation_results = propagator.propagate(
        memory_id,
        [&](uint64_t invalidated_id) {
            // For each invalidated descendant:
            auto* descendant = get_record(invalidated_id);
            if (descendant) {
                descendant->markInvalidated();
                persist(invalidated_id, *descendant);
                hnsw_delete(invalidated_id);

                // TODO Phase 4 follow-up: emit MemoryEvent{kind=LineagePropagate}
                // via an events_log_ injection here. For now the cascade still
                // works correctly; only its observability event is missing.
                (void)memory_id;
            }
        });

    // Step 5: Categorize results
    for (const auto& prop : propagation_results) {
        if (prop.invalidated) {
            result.invalidated_descendants.push_back(prop.memory_id);
        } else {
            result.recomputed_descendants.push_back(prop.memory_id);
        }
    }

    // Step 6: Log the primary remove — TODO Phase 4 follow-up: emit
    // MemoryEvent{kind=GcTombstone} via events_log_ injection here.
    (void)before_state;
    (void)reason;

    // Step 7: Update stats
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.total_removes++;
        stats_.total_invalidated += result.invalidated_descendants.size();
        stats_.total_recomputed += result.recomputed_descendants.size();
    }

    spdlog::info("RemoveCoordinator: remove complete for {}, {} invalidated, {} recomputed",
                 memory_id, result.invalidated_descendants.size(),
                 result.recomputed_descendants.size());

    return result;
}

RemoveCoordinator::Stats RemoveCoordinator::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void RemoveCoordinator::resetStats() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = {};
}

}  // namespace amind
