#pragma once

#include "forget_engine.h"

#include <cstddef>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace amind {

struct ForgetLogConfig {
    size_t max_file_bytes{10 * 1024 * 1024};  // 10 MB
    size_t max_rotated_files{5};
    size_t max_memory_entries{1000};
};

/// Persistent, rotating audit log for ForgetEngine decisions.
/// Format: JSON Lines (one JSON object per line).
class ForgetLog {
public:
    explicit ForgetLog(const std::string& data_dir, ForgetLogConfig config = {});
    ~ForgetLog();

    bool open();

    /// Replay log from disk into recent_ on startup.
    void replay();

    void append(const ForgetLogEntry& entry);

    /// Recent entries kept in memory (most recent last).
    std::vector<ForgetLogEntry> recentEntries() const;

    size_t memorySize() const;

    void flush();

private:
    void rotateIfNeeded();
    std::string currentPath() const;
    std::string rotatedPath(size_t index) const;

    std::string data_dir_;
    ForgetLogConfig config_;
    std::ofstream file_;
    size_t current_file_size_{0};
    mutable std::mutex mutex_;
    std::deque<ForgetLogEntry> recent_;
};

}  // namespace amind
