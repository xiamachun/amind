#include "backup.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <sstream>

namespace amind {

using json = nlohmann::json;

BackupManager::BackupManager(MemoryStore& store, GraphStore& graph)
    : store_(store), graph_(graph) {}

Result<std::string> BackupManager::exportMemories() {
    std::ostringstream out;
    size_t count = 0;

    store_.scanAll([&](const MemoryRecord& record) {
        json j;
        j["memory_id"] = record.memory_id;
        j["content"] = record.content;
        j["owner"] = ownerToString(record.owner);
        j["phase"] = phaseToString(record.phase);
        j["confidence"] = confidenceToString(record.confidence);
        j["importance"] = record.importance;
        j["version"] = record.mem_version;
        j["parent_id"] = record.parent_id;
        j["created_at"] = record.created_at;
        j["last_accessed"] = record.last_accessed;
        j["access_count"] = record.access_count;
        j["source_turn"] = record.source_turn;
        j["flags"] = record.flags;
        j["namespace_hash"] = record.namespace_hash;

        // Export embedding as array of floats
        if (!record.embedding.empty()) {
            j["embedding"] = record.embedding;
        }

        out << j.dump() << "\n";
        count++;
    });

    spdlog::info("BackupManager: exported {} memories", count);
    return out.str();
}

Result<size_t> BackupManager::importMemories(const std::string& jsonl) {
    std::istringstream in(jsonl);
    std::string line;
    size_t imported = 0;
    size_t errors = 0;

    while (std::getline(in, line)) {
        if (line.empty()) continue;

        try {
            auto j = json::parse(line);

            MemoryRecord record;
            record.memory_id = j.value("memory_id", uint64_t(0));
            record.content = j.value("content", "");
            record.importance = j.value("importance", 0.5f);
            record.mem_version = j.value("version", uint32_t(1));
            record.parent_id = j.value("parent_id", uint64_t(0));
            record.created_at = j.value("created_at", uint32_t(0));
            record.last_accessed = j.value("last_accessed", uint32_t(0));
            record.access_count = j.value("access_count", uint32_t(0));
            record.source_turn = j.value("source_turn", uint16_t(0));
            record.flags = j.value("flags", uint16_t(RecordFlags::ALIVE));
            record.namespace_hash = j.value("namespace_hash", uint64_t(0));

            // Parse owner
            auto owner_str = j.value("owner", "Session");
            if (owner_str == "User") record.owner = MemoryOwner::User;
            else if (owner_str == "Project") record.owner = MemoryOwner::Project;
            else if (owner_str == "Agent") record.owner = MemoryOwner::Agent;
            else if (owner_str == "Shared") record.owner = MemoryOwner::Shared;
            else record.owner = MemoryOwner::Session;

            // Parse phase
            auto phase_str = j.value("phase", "Active");
            if (phase_str == "Versioned") record.phase = MemoryPhase::Versioned;
            else if (phase_str == "Archived") record.phase = MemoryPhase::Archived;
            else if (phase_str == "Tombstone") record.phase = MemoryPhase::Tombstone;
            else record.phase = MemoryPhase::Active;

            // Parse confidence
            auto conf_str = j.value("confidence", "Inferred");
            if (conf_str == "Verified") record.confidence = Confidence::Verified;
            else if (conf_str == "Stale") record.confidence = Confidence::Stale;
            else if (conf_str == "Conflicted") record.confidence = Confidence::Conflicted;
            else record.confidence = Confidence::Inferred;

            // Parse embedding
            if (j.contains("embedding") && j["embedding"].is_array()) {
                record.embedding = j["embedding"].get<std::vector<float>>();
            }

            auto result = store_.fastStore(std::move(record));
            if (result.ok()) {
                imported++;
            } else {
                errors++;
            }
        } catch (const json::exception& e) {
            spdlog::warn("BackupManager: import parse error: {}", e.what());
            errors++;
        }
    }

    spdlog::info("BackupManager: imported {} memories ({} errors)", imported, errors);
    return imported;
}

Result<std::string> BackupManager::exportGraph() {
    std::ostringstream out;
    auto edges = graph_.getAllEdges();

    for (const auto& edge : edges) {
        json j;
        j["from_id"] = edge.from_id;
        j["to_id"] = edge.to_id;
        j["type"] = edgeTypeToString(edge.type);
        j["weight"] = edge.weight;
        j["created_at"] = edge.created_at;
        out << j.dump() << "\n";
    }

    spdlog::info("BackupManager: exported {} graph edges", edges.size());
    return out.str();
}

Result<size_t> BackupManager::importGraph(const std::string& jsonl) {
    std::istringstream in(jsonl);
    std::string line;
    size_t imported = 0;

    while (std::getline(in, line)) {
        if (line.empty()) continue;

        try {
            auto j = json::parse(line);
            uint64_t from = j.value("from_id", uint64_t(0));
            uint64_t to = j.value("to_id", uint64_t(0));
            float weight = j.value("weight", 1.0f);

            auto type_str = j.value("type", "Related");
            EdgeType type = EdgeType::Related;
            if (type_str == "Caused") type = EdgeType::Caused;
            else if (type_str == "Contradicts") type = EdgeType::Contradicts;
            else if (type_str == "Supersedes") type = EdgeType::Supersedes;
            else if (type_str == "DerivedFrom") type = EdgeType::DerivedFrom;
            else if (type_str == "SameSession") type = EdgeType::SameSession;
            else if (type_str == "Corrects") type = EdgeType::Corrects;
            else if (type_str == "Prerequisite") type = EdgeType::Prerequisite;
            else if (type_str == "Temporal") type = EdgeType::Temporal;
            else if (type_str == "Entity") type = EdgeType::Entity;
            else if (type_str == "ConflictsWith") type = EdgeType::ConflictsWith;

            graph_.addEdge(from, to, type, weight);
            imported++;
        } catch (const json::exception& e) {
            spdlog::warn("BackupManager: graph import parse error: {}", e.what());
        }
    }

    spdlog::info("BackupManager: imported {} graph edges", imported);
    return imported;
}

}  // namespace amind
