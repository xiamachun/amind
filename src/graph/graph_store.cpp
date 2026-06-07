#include "graph_store.h"

#include "core/memory_record.h"

#include <algorithm>
#include <queue>
#include <spdlog/spdlog.h>
#include <unordered_set>

namespace amind {

GraphStore::GraphStore(const std::string& data_dir)
    : data_dir_(data_dir) {
    if (!data_dir_.empty()) {
        wal_ = std::make_unique<GraphWAL>(data_dir_);
    }
}

void GraphStore::recover() {
    if (!wal_) return;
    if (!wal_->open()) {
        spdlog::warn("GraphStore: WAL open failed, starting with empty graph");
        return;
    }

    // Step 1: Load snapshot
    auto snapshot_edges = wal_->loadSnapshot();
    for (const auto& e : snapshot_edges) {
        auto type = static_cast<EdgeType>(e.edge_type);
        addEdgeInternal(e.from_id, e.to_id, type, e.weight, e.timestamp);
    }

    // Step 2: Replay WAL on top of snapshot
    size_t replayed = wal_->replay(
        [this](WalOp op, uint64_t from, uint64_t to, EdgeType type, float weight, uint32_t ts) {
            if (op == WalOp::AddEdge) {
                addEdgeInternal(from, to, type, weight, ts);
            } else if (op == WalOp::RemoveEdge) {
                // Remove all edges for node (from = node_id)
                std::lock_guard lock(mutex_);
                adjacency_.erase(from);
                // Clean reverse_adj_ using the same logic as removeNode
                auto rev_it = reverse_adj_.find(from);
                if (rev_it != reverse_adj_.end()) {
                    for (uint64_t src : rev_it->second) {
                        auto adj_it = adjacency_.find(src);
                        if (adj_it != adjacency_.end()) {
                            auto& edges = adj_it->second;
                            edges.erase(
                                std::remove_if(edges.begin(), edges.end(),
                                               [from](const GraphEdge& e) { return e.to_id == from; }),
                                edges.end());
                        }
                    }
                    reverse_adj_.erase(rev_it);
                }
                // Remove from as a source in other nodes' reverse_adj_ lists
                for (auto& [target, sources] : reverse_adj_) {
                    sources.erase(std::remove(sources.begin(), sources.end(), from), sources.end());
                }
            }
        });

    spdlog::info("GraphStore: recovered {} snapshot edges + {} WAL records, total {} edges",
                 snapshot_edges.size(), replayed, edgeCount());
}

void GraphStore::addEdgeInternal(uint64_t from, uint64_t to, EdgeType type,
                                  float weight, uint32_t created_at) {
    std::lock_guard lock(mutex_);

    // Check for duplicate edge (same from→to+type): update weight/ts instead
    auto& from_edges = adjacency_[from];
    bool found = false;
    for (auto& e : from_edges) {
        if (e.to_id == to && e.type == type) {
            e.weight = weight;
            e.created_at = created_at;
            found = true;
            break;
        }
    }
    if (!found) {
        from_edges.push_back({from, to, type, weight, created_at});
        reverse_adj_[to].push_back(from);
    }

    // Add reverse edge for undirected relationships
    if (type == EdgeType::Related || type == EdgeType::SameSession ||
        type == EdgeType::Temporal || type == EdgeType::Entity) {
        auto& to_edges = adjacency_[to];
        bool found_rev = false;
        for (auto& e : to_edges) {
            if (e.to_id == from && e.type == type) {
                e.weight = weight;
                e.created_at = created_at;
                found_rev = true;
                break;
            }
        }
        if (!found_rev) {
            to_edges.push_back({to, from, type, weight, created_at});
            reverse_adj_[from].push_back(to);
        }
    }
}

void GraphStore::addEdge(uint64_t from, uint64_t to, EdgeType type, float weight) {
    uint32_t ts = MemoryRecord::currentTimeSec();

    // Write to WAL first (before modifying in-memory state)
    if (wal_) {
        wal_->appendAdd(from, to, type, weight, ts);
    }

    addEdgeInternal(from, to, type, weight, ts);
}

std::vector<GraphEdge> GraphStore::getEdges(uint64_t memory_id) const {
    std::lock_guard lock(mutex_);
    auto it = adjacency_.find(memory_id);
    if (it == adjacency_.end()) return {};
    return it->second;
}

std::vector<uint64_t> GraphStore::getNeighbors(
    uint64_t memory_id, std::optional<EdgeType> type_filter) const {
    std::lock_guard lock(mutex_);
    auto it = adjacency_.find(memory_id);
    if (it == adjacency_.end()) return {};

    std::vector<uint64_t> neighbors;
    for (const auto& edge : it->second) {
        if (!type_filter.has_value() || edge.type == type_filter.value()) {
            neighbors.push_back(edge.to_id);
        }
    }
    return neighbors;
}

void GraphStore::removeNode(uint64_t memory_id) {
    // Write to WAL first
    if (wal_) {
        wal_->appendRemove(memory_id);
    }

    std::lock_guard lock(mutex_);
    adjacency_.erase(memory_id);

    // Remove edges pointing to this node using reverse_adj_ (O(in-degree) instead of O(E))
    auto rev_it = reverse_adj_.find(memory_id);
    if (rev_it != reverse_adj_.end()) {
        for (uint64_t src : rev_it->second) {
            auto adj_it = adjacency_.find(src);
            if (adj_it != adjacency_.end()) {
                auto& edges = adj_it->second;
                edges.erase(
                    std::remove_if(edges.begin(), edges.end(),
                                   [memory_id](const GraphEdge& e) { return e.to_id == memory_id; }),
                    edges.end());
            }
        }
        reverse_adj_.erase(rev_it);
    }

    // Remove memory_id from other nodes' reverse_adj_ lists (it was a source)
    for (auto& [target, sources] : reverse_adj_) {
        sources.erase(std::remove(sources.begin(), sources.end(), memory_id), sources.end());
    }
}

std::vector<GraphEdge> GraphStore::getConflicts() const {
    std::lock_guard lock(mutex_);
    std::vector<GraphEdge> conflicts;
    for (const auto& [id, edges] : adjacency_) {
        for (const auto& edge : edges) {
            if (edge.type == EdgeType::Contradicts || edge.type == EdgeType::ConflictsWith) {
                conflicts.push_back(edge);
            }
        }
    }
    return conflicts;
}

std::vector<uint64_t> GraphStore::traverse(uint64_t start_id, int max_depth) const {
    std::lock_guard lock(mutex_);
    std::vector<uint64_t> result;
    std::unordered_set<uint64_t> visited;
    std::queue<std::pair<uint64_t, int>> bfs;

    bfs.push({start_id, 0});
    visited.insert(start_id);

    while (!bfs.empty()) {
        auto [current, depth] = bfs.front();
        bfs.pop();
        result.push_back(current);

        if (depth >= max_depth) continue;

        auto it = adjacency_.find(current);
        if (it == adjacency_.end()) continue;

        for (const auto& edge : it->second) {
            if (visited.find(edge.to_id) == visited.end()) {
                visited.insert(edge.to_id);
                bfs.push({edge.to_id, depth + 1});
            }
        }
    }
    return result;
}

void GraphStore::checkpoint() {
    if (!wal_) return;

    std::lock_guard lock(mutex_);
    std::vector<GraphWAL::EdgeData> edges;
    for (const auto& [node_id, edge_list] : adjacency_) {
        for (const auto& e : edge_list) {
            // Only store forward edges to avoid duplicating reverse edges
            if (e.from_id == node_id) {
                GraphWAL::EdgeData ed;
                ed.from_id = e.from_id;
                ed.to_id = e.to_id;
                ed.edge_type = static_cast<uint8_t>(e.type);
                ed.weight = e.weight;
                ed.timestamp = e.created_at;
                edges.push_back(ed);
            }
        }
    }
    wal_->checkpoint(edges);
}

void GraphStore::flush() {
    if (wal_) wal_->sync();
}

size_t GraphStore::edgeCount() const {
    std::lock_guard lock(mutex_);
    size_t count = 0;
    for (const auto& [id, edges] : adjacency_) {
        count += edges.size();
    }
    return count;
}

size_t GraphStore::nodeCount() const {
    std::lock_guard lock(mutex_);
    return adjacency_.size();
}

std::vector<GraphEdge> GraphStore::getAllEdges() const {
    std::lock_guard lock(mutex_);
    std::vector<GraphEdge> all;
    for (const auto& [id, edges] : adjacency_) {
        for (const auto& e : edges) {
            all.push_back(e);
        }
    }
    return all;
}


GraphStore::EdgeListResult GraphStore::listEdges(int page, int per_page) const {
    std::lock_guard lock(mutex_);
    EdgeListResult result;
    result.page = page;
    result.per_page = per_page;

    // Collect all unique forward edges
    std::vector<GraphEdge> all;
    for (const auto& [id, edges] : adjacency_) {
        for (const auto& e : edges) {
            if (e.from_id == id) {  // only forward edges
                all.push_back(e);
            }
        }
    }
    result.total = all.size();
    int offset = (page - 1) * per_page;
    for (int i = offset; i < static_cast<int>(all.size()) && i < offset + per_page; ++i) {
        result.edges.push_back(all[static_cast<size_t>(i)]);
    }
    return result;
}

std::vector<GraphEdge> GraphStore::getNeighborEdges(uint64_t memory_id) const {
    std::lock_guard lock(mutex_);
    auto it = adjacency_.find(memory_id);
    if (it == adjacency_.end()) return {};
    return it->second;
}

std::vector<GraphEdge> GraphStore::getIncomingEdges(uint64_t memory_id) const {
    std::lock_guard lock(mutex_);
    std::vector<GraphEdge> result;
    auto it = reverse_adj_.find(memory_id);
    if (it == reverse_adj_.end()) return result;
    for (uint64_t source : it->second) {
        auto adj_it = adjacency_.find(source);
        if (adj_it == adjacency_.end()) continue;
        for (const auto& edge : adj_it->second) {
            if (edge.to_id == memory_id) {
                result.push_back(edge);
            }
        }
    }
    return result;
}

}  // namespace amind
