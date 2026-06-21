#pragma once

#include "core/types.h"
#include "core/result.h"
#include "graph_wal.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace amind {

/// Graph edge in the memory knowledge graph.
struct GraphEdge {
    uint64_t from_id;
    uint64_t to_id;
    EdgeType type;
    float weight{1.0f};
    uint32_t created_at{0};
};

/// In-memory graph store using adjacency lists.
/// Stores relationships between memories for graph-based retrieval.
/// Now backed by WAL + snapshot for persistence across restarts.
class GraphStore {
public:
    /// Construct with optional data directory for persistence.
    explicit GraphStore(const std::string& data_dir = "");

    /// Initialize: load snapshot + replay WAL.
    void recover();

    /// Add an edge between two memories.
    void addEdge(uint64_t from, uint64_t to, EdgeType type, float weight = 1.0f);

    /// Get all edges from a memory.
    std::vector<GraphEdge> getEdges(uint64_t memory_id) const;

    /// Get neighbors of a memory (optionally filtered by edge type).
    std::vector<uint64_t> getNeighbors(uint64_t memory_id,
                                        std::optional<EdgeType> type_filter = std::nullopt) const;

    /// Remove all edges involving a memory.
    void removeNode(uint64_t memory_id);

    /// Get all conflict edges.
    std::vector<GraphEdge> getConflicts() const;

    /// Get all Supersedes edges (for recall filtering).
    std::vector<GraphEdge> getSupersedes() const;

    /// BFS/DFS traversal up to depth.
    std::vector<uint64_t> traverse(uint64_t start_id, int max_depth = 2) const;

    /// Write a full snapshot and truncate WAL.
    void checkpoint();

    /// Flush WAL to disk.
    void flush();

    size_t edgeCount() const;
    size_t nodeCount() const;


    /// List edges with pagination (for WebUI).
    struct EdgeListResult {
        std::vector<GraphEdge> edges;
        size_t total{0};
        int page{1};
        int per_page{100};
    };
    EdgeListResult listEdges(int page = 1, int per_page = 100) const;

    /// Get neighbors with full edge info (for WebUI detail page).
    std::vector<GraphEdge> getNeighborEdges(uint64_t memory_id) const;

    /// Get edges pointing INTO this memory (e.g. DerivedFrom edges from
    /// derived facts, which are stored only in the derived's adjacency).
    std::vector<GraphEdge> getIncomingEdges(uint64_t memory_id) const;


    /// Get all edges (for backup export).
    std::vector<GraphEdge> getAllEdges() const;

private:
    /// Add edge to in-memory adjacency list only (no WAL write).
    void addEdgeInternal(uint64_t from, uint64_t to, EdgeType type,
                         float weight, uint32_t created_at);

    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, std::vector<GraphEdge>> adjacency_;
    std::unordered_map<uint64_t, std::vector<uint64_t>> reverse_adj_;
    std::unique_ptr<GraphWAL> wal_;
    std::string data_dir_;
};

}  // namespace amind
