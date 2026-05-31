
#pragma once

#include "core/memory_record.h"
#include "core/types.h"
#include "forget/forget_engine.h"
#include "lineage/lineage_index.h"
#include "lineage/propagator.h"

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace amind {

/// Callback to retrieve a MemoryRecord by ID.
using GetRecordFunc = std::function<MemoryRecord*(uint64_t)>;

/// Callback to persist a modified record (after markTombstone/markInvalidated).
using PersistFunc = std::function<void(uint64_t, const MemoryRecord&)>;

/// Callback to soft-delete from HNSW index.
using HnswSoftDeleteFunc = std::function<void(uint64_t)>;

/// Result of a coordinated remove operation.
struct RemoveResult {
    uint64_t primary_id;
    RemoveReason reason;
    std::vector<uint64_t> invalidated_descendants;
    std::vector<uint64_t> recomputed_descendants;  // still have live parents
    bool success{true};
    std::string error_message;
};

/// RemoveCoordinator — orchestrates the V2 remove() flow:
///   1. Mark primary record as Tombstone
///   2. Propagate invalidation through LineagePropagator
///   3. Mark orphaned descendants as Invalidated
///   4. HNSW soft-delete for primary + invalidated
///   5. Log everything to ForgetEngine
class RemoveCoordinator {
public:
    RemoveCoordinator(LineageIndex& lineage,
                      ForgetEngine& forget_engine);

    /// Execute a coordinated remove.
    RemoveResult remove(uint64_t memory_id,
                        RemoveReason reason,
                        const GetRecordFunc& get_record,
                        const PersistFunc& persist,
                        const HnswSoftDeleteFunc& hnsw_delete);

    struct Stats {
        uint64_t total_removes{0};
        uint64_t total_invalidated{0};
        uint64_t total_recomputed{0};
    };

    Stats stats() const;
    void resetStats();

private:
    LineageIndex& lineage_;
    ForgetEngine& forget_engine_;
    mutable std::mutex mutex_;
    Stats stats_;
};

/// Deprecated V1-compatible overload — defaults to UserDelete reason.
[[deprecated("Use remove(id, reason, ...) instead")]]
inline RemoveResult removeCompat(RemoveCoordinator& coord,
                                  uint64_t memory_id,
                                  const GetRecordFunc& get_record,
                                  const PersistFunc& persist,
                                  const HnswSoftDeleteFunc& hnsw_delete) {
    return coord.remove(memory_id, RemoveReason::UserDelete,
                        get_record, persist, hnsw_delete);
}

}  // namespace amind
