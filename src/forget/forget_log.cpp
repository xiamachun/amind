#include "forget_log.h"

#include <filesystem>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace amind {

namespace {

nlohmann::json entryToJson(const ForgetLogEntry& e) {
    nlohmann::json j;
    j["ts"] = e.timestamp_ms;
    j["id"] = e.memory_id;
    j["decision"] = decisionToString(e.decision);
    j["reason"] = e.reason;
    j["before"] = static_cast<uint8_t>(e.before_state);
    j["after"] = static_cast<uint8_t>(e.after_state);
    if (!e.lineage_affected.empty()) {
        j["lineage"] = e.lineage_affected;
    }
    if (!e.gc_worker_id.empty()) {
        j["worker"] = e.gc_worker_id;
    }
    if (!e.namespace_.empty()) {
        j["ns"] = e.namespace_;
    }
    if (!e.content_preview.empty()) {
        j["content"] = e.content_preview;
    }
    return j;
}

}  // namespace

static ForgetLogEntry::Decision parseDecision(const std::string& s) {
    if (s == "Decay") return ForgetLogEntry::Decision::Decay;
    if (s == "Archive") return ForgetLogEntry::Decision::Archive;
    if (s == "Tombstone") return ForgetLogEntry::Decision::Tombstone;
    if (s == "Vacuum") return ForgetLogEntry::Decision::Vacuum;
    if (s == "DropFromHNSW") return ForgetLogEntry::Decision::DropFromHNSW;
    if (s == "ResolveConflict") return ForgetLogEntry::Decision::ResolveConflict;
    if (s == "LineageInvalidate") return ForgetLogEntry::Decision::LineageInvalidate;
    if (s == "GateReject") return ForgetLogEntry::Decision::GateReject;
    if (s == "GateDefer") return ForgetLogEntry::Decision::GateDefer;
    return ForgetLogEntry::Decision::Decay;
}

ForgetLog::ForgetLog(const std::string& data_dir, ForgetLogConfig config)
    : data_dir_(data_dir), config_(config) {}

ForgetLog::~ForgetLog() {
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

bool ForgetLog::open() {
    std::lock_guard lock(mutex_);
    namespace fs = std::filesystem;

    if (!fs::exists(data_dir_)) {
        fs::create_directories(data_dir_);
    }

    auto path = currentPath();
    if (fs::exists(path)) {
        current_file_size_ = static_cast<size_t>(fs::file_size(path));
    }

    file_.open(path, std::ios::binary | std::ios::app);
    if (!file_.is_open()) {
        spdlog::error("ForgetLog: failed to open log file: {}", path);
        return false;
    }

    spdlog::info("ForgetLog: opened {}, current size: {} bytes", path, current_file_size_);
    return true;
}

void ForgetLog::replay() {
    std::lock_guard lock(mutex_);
    namespace fs = std::filesystem;

    auto path = currentPath();
    if (!fs::exists(path)) return;

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return;

    std::string line;
    size_t count = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        try {
            auto j = nlohmann::json::parse(line);
            ForgetLogEntry entry;
            entry.timestamp_ms = j.value("ts", uint64_t(0));
            entry.memory_id = j.value("id", uint64_t(0));
            entry.decision = parseDecision(j.value("decision", ""));
            entry.reason = j.value("reason", "");
            entry.before_state = static_cast<MemoryPhase>(j.value("before", 0));
            entry.after_state = static_cast<MemoryPhase>(j.value("after", 0));
            if (j.contains("lineage") && j["lineage"].is_array()) {
                for (const auto& id : j["lineage"]) {
                    entry.lineage_affected.push_back(id.get<uint64_t>());
                }
            }
            entry.gc_worker_id = j.value("worker", "");
            entry.namespace_ = j.value("ns", "");
            entry.content_preview = j.value("content", "");
            recent_.push_back(std::move(entry));
            count++;
        } catch (...) {
            continue;
        }
    }

    while (recent_.size() > config_.max_memory_entries) {
        recent_.pop_front();
    }

    spdlog::info("ForgetLog: replayed {} entries from disk", count);
}

void ForgetLog::append(const ForgetLogEntry& entry) {
    std::lock_guard lock(mutex_);

    recent_.push_back(entry);
    while (recent_.size() > config_.max_memory_entries) {
        recent_.pop_front();
    }

    if (!file_.is_open()) return;

    rotateIfNeeded();

    std::string line = entryToJson(entry).dump() + "\n";
    file_.write(line.data(), static_cast<std::streamsize>(line.size()));
    file_.flush();
    current_file_size_ += line.size();
}

std::vector<ForgetLogEntry> ForgetLog::recentEntries() const {
    std::lock_guard lock(mutex_);
    return {recent_.begin(), recent_.end()};
}

size_t ForgetLog::memorySize() const {
    std::lock_guard lock(mutex_);
    return recent_.size();
}

void ForgetLog::flush() {
    std::lock_guard lock(mutex_);
    if (file_.is_open()) {
        file_.flush();
    }
}

void ForgetLog::rotateIfNeeded() {
    if (current_file_size_ < config_.max_file_bytes) return;

    namespace fs = std::filesystem;

    file_.close();

    // Remove oldest rotated file
    std::string oldest = rotatedPath(config_.max_rotated_files);
    if (fs::exists(oldest)) {
        fs::remove(oldest);
    }

    // Shift rotated files: .4 → .5, .3 → .4, etc.
    for (size_t i = config_.max_rotated_files; i > 1; --i) {
        std::string from = rotatedPath(i - 1);
        std::string to = rotatedPath(i);
        if (fs::exists(from)) {
            fs::rename(from, to);
        }
    }

    // Current → .1
    fs::rename(currentPath(), rotatedPath(1));

    // Open fresh file
    file_.open(currentPath(), std::ios::binary | std::ios::app);
    current_file_size_ = 0;

    spdlog::info("ForgetLog: rotated log files");
}

std::string ForgetLog::currentPath() const {
    return data_dir_ + "/forget.log";
}

std::string ForgetLog::rotatedPath(size_t index) const {
    return data_dir_ + "/forget.log." + std::to_string(index);
}

}  // namespace amind
