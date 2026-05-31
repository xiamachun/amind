#pragma once

#include "session_manager.h"

#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace amind {

/// JSONL write-ahead log for session persistence.
/// Each line records a session event (start, turn, close).
class SessionWAL {
public:
    explicit SessionWAL(const std::string& data_dir);
    ~SessionWAL();

    bool open();

    void appendStart(const Session& session);
    void appendTurn(uint64_t session_id, const TurnRecord& turn);
    void appendClose(uint64_t session_id);

    /// Replay WAL and reconstruct sessions.
    std::unordered_map<uint64_t, Session> replay();

private:
    std::string wal_path_;
    std::ofstream wal_file_;
    std::mutex mutex_;

    void writeLine(const std::string& json_line);
};

}  // namespace amind
