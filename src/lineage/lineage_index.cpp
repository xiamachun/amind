
#include "lineage_index.h"
#include "lineage_wal.h"
#include "core/memory_record.h"

#include <queue>
#include <spdlog/spdlog.h>

namespace amind {

LineageIndex::LineageIndex() = default;
LineageIndex::~LineageIndex() = default;

// ── Constructor with persistence ─────────────────────────────────────────

LineageIndex::LineageIndex(const std::string& data_dir)
    : wal_(std::make_unique<LineageWAL>(data_dir)) {
    if (!wal_->open()) {
        spdlog::error("LineageIndex: failed to open WAL, running without persistence");
        wal_.reset();
        return;
    }
    replayFromWAL();
}

void LineageIndex::replayFromWAL() {
    if (!wal_) return;

    // Load snapshot first
    std::vector<uint64_t> snapshot_removed;
    auto snapshot_records = wal_->loadSnapshot(snapshot_removed);

    for (const auto& rec : snapshot_records) {
        LineageRecord lr;
        lr.child_id = rec.child_id;
        lr.parent_ids = rec.parent_ids;
        lr.op = rec.op;
        lr.created_at = rec.created_at;

        child_to_record_[rec.child_id] = std::move(lr);
        for (uint64_t pid : rec.parent_ids) {
            parent_to_children_[pid].insert(rec.child_id);
        }
    }

    for (uint64_t nid : snapshot_removed) {
        removed_nodes_.insert(nid);
    }

    // Replay WAL on top of snapshot
    wal_->replay([this](const LineageWAL::ReplayEntry& entry) {
        switch (entry.wal_op) {
            case LineageWalOp::RecordLineage: {
                LineageRecord lr;
                lr.child_id = entry.node_id;
                lr.parent_ids = entry.parents;
                lr.op = entry.lineage_op;
                lr.created_at = entry.timestamp;

                child_to_record_[entry.node_id] = std::move(lr);
                for (uint64_t pid : entry.parents) {
                    parent_to_children_[pid].insert(entry.node_id);
                }
                break;
            }
            case LineageWalOp::RemoveNode: {
                removed_nodes_.insert(entry.node_id);

                auto child_it = child_to_record_.find(entry.node_id);
                if (child_it != child_to_record_.end()) {
                    for (uint64_t pid : child_it->second.parent_ids) {
                        auto pit = parent_to_children_.find(pid);
                        if (pit != parent_to_children_.end()) {
                            pit->second.erase(entry.node_id);
                            if (pit->second.empty()) {
                                parent_to_children_.erase(pit);
                            }
                        }
                    }
                    child_to_record_.erase(child_it);
                }
                auto pit = parent_to_children_.find(entry.node_id);
                if (pit != parent_to_children_.end()) {
                    parent_to_children_.erase(pit);
                }
                break;
            }
            case LineageWalOp::MarkRemoved: {
                removed_nodes_.insert(entry.node_id);
                break;
            }
        }
    });

    spdlog::info("LineageIndex: restored {} records, {} removed nodes",
                 child_to_record_.size(), removed_nodes_.size());
}

bool LineageIndex::checkpoint() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!wal_) return false;

    std::vector<LineageWAL::SnapshotRecord> records;
    records.reserve(child_to_record_.size());
    for (const auto& [id, rec] : child_to_record_) {
        LineageWAL::SnapshotRecord sr;
        sr.child_id = rec.child_id;
        sr.parent_ids = rec.parent_ids;
        sr.op = rec.op;
        sr.created_at = rec.created_at;
        records.push_back(std::move(sr));
    }

    std::vector<uint64_t> removed(removed_nodes_.begin(), removed_nodes_.end());
    return wal_->checkpoint(records, removed);
}

size_t LineageIndex::walSize() const {
    if (!wal_) return 0;
    return wal_->walSize();
}

// ── Core Operations ──────────────────────────────────────────────────────

void LineageIndex::recordLineage(uint64_t child,
                                  const std::vector<uint64_t>& parents,
                                  LineageOp op) {
    std::lock_guard<std::mutex> lock(mutex_);

    uint32_t now = MemoryRecord::currentTimeSec();

    LineageRecord record;
    record.child_id = child;
    record.parent_ids = parents;
    record.op = op;
    record.created_at = now;

    child_to_record_[child] = std::move(record);

    for (uint64_t parent_id : parents) {
        parent_to_children_[parent_id].insert(child);
    }

    if (wal_) {
        wal_->appendRecordLineage(child, parents, op, now);
    }

    spdlog::debug("LineageIndex: recorded lineage child={} parents=[{}] op={}",
                  child, parents.size(), lineageOpToString(op));
}

void LineageIndex::removeNode(uint64_t memory_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    removed_nodes_.insert(memory_id);

    // Remove as child
    auto child_it = child_to_record_.find(memory_id);
    if (child_it != child_to_record_.end()) {
        for (uint64_t parent_id : child_it->second.parent_ids) {
            auto parent_it = parent_to_children_.find(parent_id);
            if (parent_it != parent_to_children_.end()) {
                parent_it->second.erase(memory_id);
                if (parent_it->second.empty()) {
                    parent_to_children_.erase(parent_it);
                }
            }
        }
        child_to_record_.erase(child_it);
    }

    // Remove as parent (remove reverse index entry, but keep children's forward records)
    auto parent_it = parent_to_children_.find(memory_id);
    if (parent_it != parent_to_children_.end()) {
        parent_to_children_.erase(parent_it);
    }

    if (wal_) {
        wal_->appendRemoveNode(memory_id);
    }

    spdlog::debug("LineageIndex: removed node {}", memory_id);
}

void LineageIndex::markRemoved(uint64_t memory_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    removed_nodes_.insert(memory_id);

    if (wal_) {
        wal_->appendMarkRemoved(memory_id);
    }

    spdlog::debug("LineageIndex: marked node {} as removed", memory_id);
}

// ── Queries ──────────────────────────────────────────────────────────────

std::vector<uint64_t> LineageIndex::getParents(uint64_t child_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = child_to_record_.find(child_id);
    if (it == child_to_record_.end()) return {};
    return it->second.parent_ids;
}

std::vector<uint64_t> LineageIndex::getChildren(uint64_t parent_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = parent_to_children_.find(parent_id);
    if (it == parent_to_children_.end()) return {};
    return {it->second.begin(), it->second.end()};
}

std::vector<uint64_t> LineageIndex::getTransitiveDescendants(uint64_t parent_id,
                                                              int max_depth) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<uint64_t> result;
    std::unordered_set<uint64_t> visited;
    std::queue<std::pair<uint64_t, int>> frontier;

    frontier.push({parent_id, 0});
    visited.insert(parent_id);

    while (!frontier.empty()) {
        auto [current_id, depth] = frontier.front();
        frontier.pop();

        if (depth >= max_depth) continue;

        auto it = parent_to_children_.find(current_id);
        if (it == parent_to_children_.end()) continue;

        for (uint64_t child_id : it->second) {
            if (visited.count(child_id)) continue;
            visited.insert(child_id);
            result.push_back(child_id);
            frontier.push({child_id, depth + 1});
        }
    }

    return result;
}

LineageOp LineageIndex::getLineageOp(uint64_t child_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = child_to_record_.find(child_id);
    if (it == child_to_record_.end()) return LineageOp::None;
    return it->second.op;
}

bool LineageIndex::hasLiveParents(uint64_t child_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = child_to_record_.find(child_id);
    if (it == child_to_record_.end()) return false;

    for (uint64_t parent_id : it->second.parent_ids) {
        if (removed_nodes_.count(parent_id) == 0) {
            return true;
        }
    }
    return false;
}

// ── Stats ────────────────────────────────────────────────────────────────

size_t LineageIndex::childCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return child_to_record_.size();
}

size_t LineageIndex::parentCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return parent_to_children_.size();
}

size_t LineageIndex::totalRecords() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return child_to_record_.size();
}

}  // namespace amind
