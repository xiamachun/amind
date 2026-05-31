#pragma once

#include "core/result.h"
#include "memory_store.h"
#include "graph/graph_store.h"

#include <string>

namespace amind {

/// Backup manager: JSONL export/import for memories and graph edges.
class BackupManager {
public:
    BackupManager(MemoryStore& store, GraphStore& graph);

    /// Export all memories to JSONL string (one JSON object per line).
    Result<std::string> exportMemories();

    /// Import memories from JSONL string.
    Result<size_t> importMemories(const std::string& jsonl);

    /// Export all graph edges to JSONL string.
    Result<std::string> exportGraph();

    /// Import graph edges from JSONL string.
    Result<size_t> importGraph(const std::string& jsonl);

private:
    MemoryStore& store_;
    GraphStore& graph_;
};

}  // namespace amind
