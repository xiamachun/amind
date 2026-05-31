#include "metacognition.h"

namespace amind {

MetaCognition::MetaCognition(MemoryStore& store, GraphStore& graph)
    : store_(store), graph_(graph) {}

CoverageStats MetaCognition::getCoverage(const std::string& namespace_) const {
    CoverageStats stats;
    stats.last_updated = MemoryRecord::currentTimeSec();

    uint64_t filter_hash = 0;
    if (!namespace_.empty()) {
        filter_hash = MemoryRecord::hashNamespace(namespace_);
    }

    store_.scanAll([&](const MemoryRecord& rec) {
        if (filter_hash != 0 && rec.namespace_hash != filter_hash) return;

        stats.total++;

        if (rec.phase == MemoryPhase::Active) {
            stats.active++;
        }

        if (rec.confidence == Confidence::Stale) {
            stats.stale++;
        }

        if (rec.confidence == Confidence::Conflicted) {
            stats.conflicted++;
        }

        // Topic distribution by owner
        std::string owner_key = ownerToString(rec.owner);
        stats.topic_distribution[owner_key]++;
    });

    return stats;
}

std::vector<ConflictInfo> MetaCognition::getConflicts() const {
    auto edges = graph_.getConflicts();
    std::vector<ConflictInfo> conflicts;
    conflicts.reserve(edges.size());
    for (const auto& edge : edges) {
        ConflictInfo info;
        info.memory_a = edge.from_id;
        info.memory_b = edge.to_id;
        info.conflict_type = edgeTypeToString(edge.type);
        info.explanation = "Contradicting memories detected via graph edge";
        conflicts.push_back(std::move(info));
    }
    return conflicts;
}

std::vector<uint64_t> MetaCognition::getStaleMemories() const {
    std::vector<uint64_t> stale_ids;

    store_.scanAll([&](const MemoryRecord& rec) {
        if (rec.confidence == Confidence::Stale && rec.phase == MemoryPhase::Active) {
            stale_ids.push_back(rec.memory_id);
        }
    });

    return stale_ids;
}

}  // namespace amind
