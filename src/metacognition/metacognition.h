#pragma once

#include "core/types.h"
#include "core/result.h"
#include "memory/memory_store.h"
#include "graph/graph_store.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace amind {

struct CoverageStats {
    size_t total{0};
    size_t active{0};
    size_t stale{0};
    size_t conflicted{0};
    std::unordered_map<std::string, size_t> topic_distribution;
    uint32_t last_updated{0};
};

struct ConflictInfo {
    uint64_t memory_a;
    uint64_t memory_b;
    std::string conflict_type;
    std::string explanation;
};

/// MetaCognition — "knowing what you know."
/// Provides coverage tracking, confidence mapping, and proactive recall advice.
class MetaCognition {
public:
    MetaCognition(MemoryStore& store, GraphStore& graph);

    /// Get coverage statistics for a namespace (tenant isolation unit).
    CoverageStats getCoverage(const std::string& namespace_) const;

    /// Get list of active conflicts.
    std::vector<ConflictInfo> getConflicts() const;

    /// Get list of stale memories that may need refresh.
    std::vector<uint64_t> getStaleMemories() const;

private:
    MemoryStore& store_;
    GraphStore& graph_;
};

}  // namespace amind
