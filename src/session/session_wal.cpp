#include "session_wal.h"

#include <filesystem>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace amind {

using json = nlohmann::json;

SessionWAL::SessionWAL(const std::string& data_dir)
    : wal_path_(data_dir + "/session_wal.jsonl") {}

SessionWAL::~SessionWAL() {
    if (wal_file_.is_open()) {
        wal_file_.flush();
        wal_file_.close();
    }
}

bool SessionWAL::open() {
    namespace fs = std::filesystem;
    auto parent = fs::path(wal_path_).parent_path();
    if (!fs::exists(parent)) {
        fs::create_directories(parent);
    }

    wal_file_.open(wal_path_, std::ios::app);
    if (!wal_file_.is_open()) {
        spdlog::error("SessionWAL: failed to open {}", wal_path_);
        return false;
    }
    spdlog::info("SessionWAL: opened {}", wal_path_);
    return true;
}

void SessionWAL::appendStart(const Session& session) {
    json j;
    j["op"] = "start";
    j["session_id"] = session.session_id;
    j["agent_id"] = session.agent_id;
    j["user_id"] = session.user_id;
    j["started_at"] = session.started_at;
    writeLine(j.dump());
}

void SessionWAL::appendTurn(uint64_t session_id, const TurnRecord& turn) {
    json j;
    j["op"] = "turn";
    j["session_id"] = session_id;
    j["turn_number"] = turn.turn_number;
    j["user_input"] = turn.user_input;
    j["agent_response"] = turn.agent_response;
    j["detected_intent"] = turn.detected_intent;
    j["extracted_memory_ids"] = turn.extracted_memory_ids;
    j["timestamp"] = turn.timestamp;
    writeLine(j.dump());
}

void SessionWAL::appendClose(uint64_t session_id) {
    json j;
    j["op"] = "close";
    j["session_id"] = session_id;
    writeLine(j.dump());
}

std::unordered_map<uint64_t, Session> SessionWAL::replay() {
    std::unordered_map<uint64_t, Session> sessions;

    std::ifstream file(wal_path_);
    if (!file.is_open()) return sessions;

    std::string line;
    size_t count = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        try {
            auto j = json::parse(line);
            auto op = j.value("op", "");

            if (op == "start") {
                Session s;
                s.session_id = j["session_id"].get<uint64_t>();
                s.agent_id = j.value("agent_id", j.value("namespace", "default"));
                s.user_id = j.value("user_id", "anonymous");
                s.started_at = j.value("started_at", 0u);
                s.last_turn_at = s.started_at;
                s.active = true;
                sessions[s.session_id] = std::move(s);
            } else if (op == "turn") {
                uint64_t sid = j["session_id"].get<uint64_t>();
                auto it = sessions.find(sid);
                if (it == sessions.end()) continue;
                auto& session = it->second;

                TurnRecord turn;
                turn.turn_number = j.value("turn_number", 0u);
                turn.user_input = j.value("user_input", "");
                turn.agent_response = j.value("agent_response", "");
                turn.detected_intent = j.value("detected_intent", "");
                turn.timestamp = j.value("timestamp", int64_t(0));
                if (j.contains("extracted_memory_ids")) {
                    turn.extracted_memory_ids = j["extracted_memory_ids"].get<std::vector<uint64_t>>();
                    for (auto mid : turn.extracted_memory_ids) {
                        session.memory_ids.push_back(mid);
                    }
                }

                session.turn_count++;
                session.last_turn_at = static_cast<uint32_t>(turn.timestamp);
                if (!turn.detected_intent.empty()) {
                    session.current_intent = turn.detected_intent;
                }
                session.turns.push_back(std::move(turn));
            } else if (op == "close") {
                uint64_t sid = j["session_id"].get<uint64_t>();
                auto it = sessions.find(sid);
                if (it != sessions.end()) {
                    it->second.active = false;
                }
            }
            count++;
        } catch (const json::exception&) {
            spdlog::warn("SessionWAL: skipping malformed line");
        }
    }

    spdlog::info("SessionWAL: replayed {} entries, {} sessions recovered", count, sessions.size());
    return sessions;
}

void SessionWAL::writeLine(const std::string& json_line) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!wal_file_.is_open()) return;
    wal_file_ << json_line << '\n';
    wal_file_.flush();
}

}  // namespace amind
