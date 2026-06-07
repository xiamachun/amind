#include "metacognition.h"

namespace amind {

MetaCognition::MetaCognition(MemoryStore& store, GraphStore& graph)
    : store_(store), graph_(graph) {}

CoverageStats MetaCognition::getCoverage(const std::string& agent_id) const {
    CoverageStats stats;
    stats.last_updated = MemoryRecord::currentTimeSec();

    store_.scanAll([&](const MemoryRecord& rec) {
        if (!agent_id.empty() && rec.agent_id != agent_id) return;

        stats.total++;

        if (rec.phase == MemoryPhase::Active) {
            stats.active++;
        }

        if (rec.confidence_level == Confidence::Stale) {
            stats.stale++;
        }

        if (rec.confidence_level == Confidence::Conflicted) {
            stats.conflicted++;
        }

        // Topic distribution by memory type
        std::string type_key = memoryTypeToString(rec.memory_type);
        stats.topic_distribution[type_key]++;
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
        if (rec.confidence_level == Confidence::Stale && rec.phase == MemoryPhase::Active) {
            stale_ids.push_back(rec.memory_id);
        }
    });

    return stale_ids;
}

}  // namespace amind
