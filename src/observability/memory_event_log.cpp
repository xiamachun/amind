#include "memory_event_log.h"

#include "core/snowflake.h"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace amind {

using json = nlohmann::json;

namespace {

// Single process-wide Snowflake for event_id generation. Worker 0 is fine —
// MemoryEventLog is single-process; no cross-host coordination needed.
SnowflakeGenerator& eventIdGen() {
    static SnowflakeGenerator gen(0);
    return gen;
}

json eventToJson(const MemoryEvent& e) {
    json j;
    j["event_id"]        = e.event_id;
    j["parent_event_id"] = e.parent_event_id;
    j["trace_id"]        = e.trace_id;
    j["memory_id"]       = e.memory_id;
    j["ts"]              = e.timestamp_ms;
    j["dur"]             = e.duration_ms;
    j["ns"]              = e.agent_id;
    j["kind"]            = kindToString(e.kind);
    j["status"]          = statusToString(e.status);
    j["summary"]         = e.summary;
    if (!e.attrs.empty()) {
        json attrs = json::object();
        for (const auto& [k, v] : e.attrs) attrs[k] = v;
        j["attrs"] = std::move(attrs);
    }
    return j;
}

MemoryEvent jsonToEvent(const json& j) {
    MemoryEvent e;
    e.event_id        = j.value("event_id", uint64_t(0));
    e.parent_event_id = j.value("parent_event_id", uint64_t(0));
    e.trace_id        = j.value("trace_id", uint64_t(0));
    e.memory_id       = j.value("memory_id", uint64_t(0));
    e.timestamp_ms    = j.value("ts", uint64_t(0));
    e.duration_ms     = j.value("dur", uint32_t(0));
    e.agent_id        = j.value("ns", "");
    e.kind            = kindFromString(j.value("kind", "Store"));
    e.status          = statusFromString(j.value("status", "Ok"));
    e.summary         = j.value("summary", "");
    if (j.contains("attrs") && j["attrs"].is_object()) {
        for (auto it = j["attrs"].begin(); it != j["attrs"].end(); ++it) {
            e.attrs[it.key()] = it.value().get<std::string>();
        }
    }
    return e;
}

}  // namespace

MemoryEventLog::MemoryEventLog(const std::string& data_dir,
                                MemoryEventLogConfig config)
    : data_dir_(data_dir), config_(config) {}

MemoryEventLog::~MemoryEventLog() {
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

bool MemoryEventLog::open() {
    std::lock_guard<std::mutex> lock(mutex_);
    namespace fs = std::filesystem;
    if (!fs::exists(data_dir_)) {
        fs::create_directories(data_dir_);
    }
    auto path = currentPath_();
    if (fs::exists(path)) {
        current_file_size_ = static_cast<size_t>(fs::file_size(path));
    }
    file_.open(path, std::ios::binary | std::ios::app);
    if (!file_.is_open()) {
        spdlog::warn("MemoryEventLog: failed to open {}", path);
        return false;
    }
    return true;
}

void MemoryEventLog::replay() {
    std::lock_guard<std::mutex> lock(mutex_);
    namespace fs = std::filesystem;
    auto path = currentPath_();
    if (!fs::exists(path)) return;

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return;

    std::string line;
    size_t loaded = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        try {
            auto j = json::parse(line);
            recent_.push_back(jsonToEvent(j));
            ++loaded;
        } catch (const std::exception& e) {
            spdlog::debug("MemoryEventLog: skipping bad line during replay: {}",
                          e.what());
        }
    }
    while (recent_.size() > config_.ring_capacity) {
        recent_.pop_front();
    }
    rebuildIndexes_();
    if (loaded > 0) {
        spdlog::info("MemoryEventLog: replayed {} events from {}", loaded, path);
    }
}

void MemoryEventLog::append(MemoryEvent ev) {
    if (ev.event_id == 0) {
        ev.event_id = eventIdGen().nextId();
    }
    if (ev.timestamp_ms == 0) {
        ev.timestamp_ms = nowMs();
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Persist before mutating in-memory state — if disk write fails, we don't
    // leave a phantom event in the ring.
    if (file_.is_open()) {
        rotateIfNeeded_();
        std::string line = eventToJson(ev).dump() + "\n";
        file_.write(line.data(), static_cast<std::streamsize>(line.size()));
        if (config_.flush_per_append) file_.flush();
        current_file_size_ += line.size();
    }

    recent_.push_back(std::move(ev));
    if (recent_.size() > config_.ring_capacity) {
        recent_.pop_front();
        rebuildIndexes_();  // positions shifted; cheapest correct option
    } else {
        addToIndexes_(recent_.size() - 1, recent_.back());
    }
}

void MemoryEventLog::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_.is_open()) file_.flush();
}

// ── Query helpers ──────────────────────────────────────────────────────────

std::vector<MemoryEvent>
MemoryEventLog::query(const Filter& f) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<MemoryEvent> out;

    auto matches = [&](const MemoryEvent& e) -> bool {
        if (f.memory_id && e.memory_id != *f.memory_id) return false;
        if (f.trace_id && e.trace_id != *f.trace_id) return false;
        if (f.kind && e.kind != *f.kind) return false;
        if (f.status && e.status != *f.status) return false;
        if (!f.agent_id_filter.empty() &&
            e.agent_id.substr(0, f.agent_id_filter.size()) != f.agent_id_filter) {
            return false;
        }
        if (f.since_ms && e.timestamp_ms < f.since_ms) return false;
        if (f.until_ms && e.timestamp_ms > f.until_ms) return false;
        return true;
    };

    // Use the most selective index when possible.
    auto collect_from = [&](const std::unordered_multimap<uint64_t, size_t>& idx,
                            uint64_t key) {
        auto range = idx.equal_range(key);
        std::vector<size_t> positions;
        positions.reserve(std::distance(range.first, range.second));
        for (auto it = range.first; it != range.second; ++it) {
            positions.push_back(it->second);
        }
        std::sort(positions.begin(), positions.end(), std::greater<>{});
        for (size_t pos : positions) {
            if (pos >= recent_.size()) continue;  // defensive
            const auto& ev = recent_[pos];
            if (matches(ev)) {
                out.push_back(ev);
                if (out.size() >= f.limit) break;
            }
        }
    };

    if (f.memory_id) {
        collect_from(by_memory_id_, *f.memory_id);
    } else if (f.trace_id) {
        collect_from(by_trace_id_, *f.trace_id);
    } else {
        // Full scan, newest first.
        for (auto it = recent_.rbegin(); it != recent_.rend(); ++it) {
            if (matches(*it)) {
                out.push_back(*it);
                if (out.size() >= f.limit) break;
            }
        }
    }
    return out;
}

std::vector<MemoryEvent>
MemoryEventLog::trace(uint64_t trace_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<MemoryEvent> out;
    auto range = by_trace_id_.equal_range(trace_id);
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second < recent_.size()) out.push_back(recent_[it->second]);
    }
    std::sort(out.begin(), out.end(),
              [](const MemoryEvent& a, const MemoryEvent& b) {
                  return a.timestamp_ms < b.timestamp_ms;
              });
    return out;
}

std::vector<MemoryEvent>
MemoryEventLog::memoryHistory(uint64_t memory_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<MemoryEvent> out;
    auto range = by_memory_id_.equal_range(memory_id);
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second < recent_.size()) out.push_back(recent_[it->second]);
    }
    std::sort(out.begin(), out.end(),
              [](const MemoryEvent& a, const MemoryEvent& b) {
                  return a.timestamp_ms > b.timestamp_ms;  // newest first
              });
    return out;
}

MemoryEventLog::Stats MemoryEventLog::stats(const Filter& f) const {
    Stats s;
    auto rows = query(f);  // reuses the same filter logic; OK for ring scale
    for (const auto& e : rows) {
        ++s.by_kind[kindToString(e.kind)];
        ++s.by_status[statusToString(e.status)];
        ++s.total;
    }
    return s;
}

size_t MemoryEventLog::memorySize() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return recent_.size();
}

void MemoryEventLog::clearMemory() {
    std::lock_guard<std::mutex> lock(mutex_);
    recent_.clear();
    by_memory_id_.clear();
    by_trace_id_.clear();
    by_kind_.clear();
}

// ── Internals ──────────────────────────────────────────────────────────────

void MemoryEventLog::addToIndexes_(size_t pos, const MemoryEvent& ev) {
    if (ev.memory_id != 0) by_memory_id_.emplace(ev.memory_id, pos);
    if (ev.trace_id != 0)  by_trace_id_.emplace(ev.trace_id, pos);
    by_kind_.emplace(ev.kind, pos);
}

void MemoryEventLog::rebuildIndexes_() {
    by_memory_id_.clear();
    by_trace_id_.clear();
    by_kind_.clear();
    for (size_t i = 0; i < recent_.size(); ++i) {
        addToIndexes_(i, recent_[i]);
    }
}

void MemoryEventLog::rotateIfNeeded_() {
    if (current_file_size_ < config_.max_file_bytes) return;
    namespace fs = std::filesystem;
    file_.close();
    // Drop the oldest, then shift .(N-1) → .N, ..., current → .1
    std::string oldest = rotatedPath_(config_.max_rotated_files);
    if (fs::exists(oldest)) fs::remove(oldest);
    for (size_t i = config_.max_rotated_files; i > 1; --i) {
        std::string from = rotatedPath_(i - 1);
        std::string to   = rotatedPath_(i);
        if (fs::exists(from)) fs::rename(from, to);
    }
    if (fs::exists(currentPath_())) {
        fs::rename(currentPath_(), rotatedPath_(1));
    }
    file_.open(currentPath_(), std::ios::binary | std::ios::app);
    current_file_size_ = 0;
    spdlog::info("MemoryEventLog: rotated log files");
}

std::string MemoryEventLog::currentPath_() const {
    return data_dir_ + "/events.log";
}

std::string MemoryEventLog::rotatedPath_(size_t index) const {
    return data_dir_ + "/events.log." + std::to_string(index);
}

}  // namespace amind
