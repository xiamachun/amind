#include "gate_log.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace amind {

namespace {

int64_t nowMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();
}

const char* decisionToStr(GateDecision d) {
    switch (d) {
        case GateDecision::Accepted: return "Accepted";
        case GateDecision::Rejected: return "Rejected";
        case GateDecision::Deferred: return "Deferred";
    }
    return "Accepted";
}

GateDecision parseDecision(const std::string& s) {
    if (s == "Rejected") return GateDecision::Rejected;
    if (s == "Deferred") return GateDecision::Deferred;
    return GateDecision::Accepted;
}

const char* layerToStr(MemoryLayer l) {
    return (l == MemoryLayer::Derived) ? "Derived" : "Raw";
}

MemoryLayer parseLayer(const std::string& s) {
    return s == "Derived" ? MemoryLayer::Derived : MemoryLayer::Raw;
}

const char* ownerToStr(MemoryOwner o) {
    switch (o) {
        case MemoryOwner::User:    return "User";
        case MemoryOwner::Project: return "Project";
        case MemoryOwner::Agent:   return "Agent";
        case MemoryOwner::Session: return "Session";
        case MemoryOwner::Shared:  return "Shared";
    }
    return "Session";
}

MemoryOwner parseOwner(const std::string& s) {
    if (s == "User")    return MemoryOwner::User;
    if (s == "Project") return MemoryOwner::Project;
    if (s == "Agent")   return MemoryOwner::Agent;
    if (s == "Shared")  return MemoryOwner::Shared;
    return MemoryOwner::Session;
}

nlohmann::json entryToJson(const GateLogEntry& e) {
    nlohmann::json j;
    j["op"] = "append";
    j["entry_id"] = e.entry_id;
    j["ts"] = e.timestamp_ms;
    j["ns"] = e.namespace_;
    j["content"] = e.content;
    if (!e.embedding.empty()) {
        j["embedding"] = e.embedding;
    }
    j["decision"] = decisionToStr(e.decision);
    j["reason"] = e.reason;
    j["marginal_value"] = e.marginal_value;
    j["conflict_with_id"] = e.conflict_with_id;
    j["owner"] = ownerToStr(e.owner);
    j["layer"] = layerToStr(e.layer);
    if (!e.user_metadata.empty()) {
        nlohmann::json meta = nlohmann::json::object();
        for (const auto& [k, v] : e.user_metadata) meta[k] = v;
        j["user_metadata"] = std::move(meta);
    }
    if (e.memory_id != 0) j["memory_id"] = e.memory_id;
    if (!e.reconcile_op.empty()) {
        j["reconcile_op"] = e.reconcile_op;
        j["reconcile_target_id"] = e.reconcile_target_id;
        if (!e.reconcile_rationale.empty()) j["reconcile_rationale"] = e.reconcile_rationale;
    }
    return j;
}

GateLogEntry entryFromJson(const nlohmann::json& j) {
    GateLogEntry e;
    e.entry_id = j.value("entry_id", uint64_t(0));
    e.timestamp_ms = j.value("ts", int64_t(0));
    e.namespace_ = j.value("ns", "");
    e.content = j.value("content", "");
    if (j.contains("embedding") && j["embedding"].is_array()) {
        e.embedding.reserve(j["embedding"].size());
        for (const auto& v : j["embedding"]) e.embedding.push_back(v.get<float>());
    }
    e.decision = parseDecision(j.value("decision", "Accepted"));
    e.reason = j.value("reason", "");
    e.marginal_value = j.value("marginal_value", 0.0f);
    e.conflict_with_id = j.value("conflict_with_id", uint64_t(0));
    e.owner = parseOwner(j.value("owner", "Session"));
    e.layer = parseLayer(j.value("layer", "Raw"));
    if (j.contains("user_metadata") && j["user_metadata"].is_object()) {
        for (auto it = j["user_metadata"].begin(); it != j["user_metadata"].end(); ++it) {
            if (it.value().is_string()) {
                e.user_metadata[it.key()] = it.value().get<std::string>();
            } else {
                e.user_metadata[it.key()] = it.value().dump();
            }
        }
    }
    e.memory_id = j.value("memory_id", uint64_t(0));
    e.reconcile_op = j.value("reconcile_op", "");
    e.reconcile_target_id = j.value("reconcile_target_id", uint64_t(0));
    e.reconcile_rationale = j.value("reconcile_rationale", "");
    return e;
}

}  // namespace

// ── GateLog ─────────────────────────────────────────────────────────────────

GateLog::GateLog(const std::string& data_dir, GateLogConfig config)
    : data_dir_(data_dir), config_(config), id_gen_(2 /*node_id*/) {}

GateLog::~GateLog() {
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

bool GateLog::open() {
    std::lock_guard lock(mutex_);
    namespace fs = std::filesystem;
    if (!fs::exists(data_dir_)) fs::create_directories(data_dir_);
    auto path = currentPath();
    if (fs::exists(path)) {
        current_file_size_ = static_cast<size_t>(fs::file_size(path));
    }
    file_.open(path, std::ios::binary | std::ios::app);
    if (!file_.is_open()) {
        spdlog::error("GateLog: failed to open log file: {}", path);
        return false;
    }
    spdlog::info("GateLog: opened {}, current size: {} bytes",
                 path, current_file_size_);
    return true;
}

void GateLog::replay() {
    std::lock_guard lock(mutex_);
    namespace fs = std::filesystem;

    // Replay only the current WAL into the in-memory ring.
    // Rotated WALs stay on disk for audit but are not paginated by this API.
    auto path = currentPath();
    if (!fs::exists(path)) return;

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return;

    // We replay sequentially; for resurrect ops we need to mutate an earlier
    // entry, so we keep a side-index (entry_id → ring offset).
    std::map<uint64_t, size_t> id_to_pos;

    std::string line;
    size_t count_appends = 0, count_resurrects = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        try {
            auto j = nlohmann::json::parse(line);
            std::string op = j.value("op", "append");
            if (op == "resurrect") {
                uint64_t eid = j.value("entry_id", uint64_t(0));
                auto it = id_to_pos.find(eid);
                if (it != id_to_pos.end() && it->second < recent_.size()) {
                    recent_[it->second].resurrected_to     = j.value("new_memory_id", uint64_t(0));
                    recent_[it->second].resurrected_at_ms  = j.value("ts", int64_t(0));
                    recent_[it->second].resurrect_strategy = j.value("strategy", "");
                }
                count_resurrects++;
            } else {
                GateLogEntry e = entryFromJson(j);
                recent_.push_back(std::move(e));
                id_to_pos[recent_.back().entry_id] = recent_.size() - 1;
                count_appends++;
            }
        } catch (...) {
            continue;
        }
    }

    // Trim ring to its cap (drop oldest); we lose ability to mutate trimmed
    // entries' resurrect status, which is acceptable for an in-memory cache.
    while (recent_.size() > config_.max_memory_entries) {
        recent_.pop_front();
    }

    spdlog::info("GateLog: replayed {} appends + {} resurrects from disk; ring={}",
                 count_appends, count_resurrects, recent_.size());
}

void GateLog::append(const GateLogEntry& entry_in) {
    std::lock_guard lock(mutex_);

    GateLogEntry entry = entry_in;
    if (entry.entry_id == 0) entry.entry_id = id_gen_.nextId();
    if (entry.timestamp_ms == 0) entry.timestamp_ms = nowMs();

    recent_.push_back(entry);
    while (recent_.size() > config_.max_memory_entries) {
        recent_.pop_front();
    }

    if (!file_.is_open()) return;
    rotateIfNeeded();
    writeLine(entryToJson(entry).dump());
}

void GateLog::recordResurrect(uint64_t entry_id, uint64_t new_memory_id,
                              const std::string& strategy) {
    std::lock_guard lock(mutex_);
    int64_t ts = nowMs();

    // Update in-memory ring
    for (auto& e : recent_) {
        if (e.entry_id == entry_id) {
            e.resurrected_to = new_memory_id;
            e.resurrected_at_ms = ts;
            e.resurrect_strategy = strategy;
            break;
        }
    }

    // WAL audit
    if (!file_.is_open()) return;
    rotateIfNeeded();
    nlohmann::json j;
    j["op"] = "resurrect";
    j["entry_id"] = entry_id;
    j["new_memory_id"] = new_memory_id;
    j["strategy"] = strategy;
    j["ts"] = ts;
    writeLine(j.dump());
}

std::optional<GateLogEntry> GateLog::findById(uint64_t entry_id) const {
    std::lock_guard lock(mutex_);
    for (const auto& e : recent_) {
        if (e.entry_id == entry_id) return e;
    }
    return std::nullopt;
}

std::vector<GateLogEntry> GateLog::query(const Filter& f) const {
    std::lock_guard lock(mutex_);
    std::vector<GateLogEntry> out;
    out.reserve(std::min(f.limit, recent_.size()));

    // Iterate newest first
    for (auto it = recent_.rbegin(); it != recent_.rend(); ++it) {
        if (f.decision && it->decision != *f.decision) continue;
        if (!f.namespace_filter.empty() && it->namespace_ != f.namespace_filter) continue;
        if (f.since_ms > 0 && it->timestamp_ms < f.since_ms) continue;
        if (f.only_unresurrected && it->resurrected_to != 0) continue;
        out.push_back(*it);
        if (out.size() >= f.limit) break;
    }
    return out;
}

GateLog::Stats GateLog::stats(int64_t since_ms,
                              const std::string& namespace_filter) const {
    std::lock_guard lock(mutex_);
    Stats s;
    for (const auto& e : recent_) {
        if (since_ms > 0 && e.timestamp_ms < since_ms) continue;
        if (!namespace_filter.empty() && e.namespace_ != namespace_filter) continue;
        switch (e.decision) {
            case GateDecision::Accepted: s.accepted++; break;
            case GateDecision::Rejected: s.rejected++; break;
            case GateDecision::Deferred: s.deferred++; break;
        }
        if (e.resurrected_to != 0) s.resurrected++;
    }
    return s;
}

size_t GateLog::memorySize() const {
    std::lock_guard lock(mutex_);
    return recent_.size();
}

void GateLog::flush() {
    std::lock_guard lock(mutex_);
    if (file_.is_open()) file_.flush();
}

void GateLog::writeLine(const std::string& line_in) {
    // mutex held by caller
    std::string line = line_in + "\n";
    file_.write(line.data(), static_cast<std::streamsize>(line.size()));
    file_.flush();
    current_file_size_ += line.size();
}

void GateLog::rotateIfNeeded() {
    // mutex held by caller
    if (current_file_size_ < config_.max_file_bytes) return;
    namespace fs = std::filesystem;
    file_.close();

    // Find next index — keep all rotated files, never delete (TTL infinite).
    size_t idx = nextRotateIndex();
    fs::rename(currentPath(), rotatedPath(idx));

    file_.open(currentPath(), std::ios::binary | std::ios::app);
    current_file_size_ = 0;
    spdlog::info("GateLog: rotated to gate.log.{} (TTL=infinite, kept on disk)", idx);
}

size_t GateLog::nextRotateIndex() const {
    namespace fs = std::filesystem;
    size_t idx = 1;
    while (fs::exists(rotatedPath(idx))) idx++;
    return idx;
}

std::string GateLog::currentPath() const {
    return data_dir_ + "/gate.log";
}

std::string GateLog::rotatedPath(size_t index) const {
    return data_dir_ + "/gate.log." + std::to_string(index);
}

}  // namespace amind
