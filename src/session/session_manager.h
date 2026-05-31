#pragma once

#include "core/memory_record.h"
#include "core/result.h"
#include "core/snowflake.h"
#include "memory/memory_store.h"
#include "provider/provider.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace amind {

class CapturePipeline;  // forward declaration
class SessionWAL;       // forward declaration

/// A single turn in a conversation.
struct TurnRecord {
    uint32_t turn_number{0};
    std::string user_input;
    std::string agent_response;
    std::string detected_intent;
    std::vector<uint64_t> extracted_memory_ids;
    int64_t timestamp{0};
};

/// Session data with full conversation tracking.
struct Session {
    uint64_t session_id{0};
    std::string namespace_;   // tenant isolation key (see core/memory_record.h)
    uint32_t started_at{0};
    uint32_t last_turn_at{0};
    uint16_t turn_count{0};
    bool active{true};
    std::vector<uint64_t> memory_ids;       // memories created in this session
    std::vector<TurnRecord> turns;          // full conversation history
    std::string current_intent;             // latest detected intent
    std::vector<std::string> extracted_facts; // facts extracted during session
};

/// Session summary for API response.
struct SessionSummary {
    uint64_t session_id;
    std::string namespace_;
    uint16_t turn_count;
    std::string current_intent;
    size_t memory_count;
    size_t fact_count;
    uint32_t started_at;
    uint32_t last_turn_at;
    bool active;
};

class SessionManager {
public:
    /// Construct with store, LLM (for intent detection + extraction), and capture pipeline.
    /// data_dir: if non-empty, enables session persistence via JSONL WAL.
    SessionManager(MemoryStore& store, std::shared_ptr<LLMProvider> llm,
                   CapturePipeline& capture, const std::string& data_dir = "");
    ~SessionManager();

    Result<Session> startSession(const std::string& namespace_);

    /// Record a turn with full user input and agent response.
    Result<void, Error> recordTurn(uint64_t session_id,
                                    const std::string& user_input,
                                    const std::string& agent_response);

    /// Backward-compatible: record turn with just turn number.
    Result<void, Error> recordTurn(uint64_t session_id, uint16_t turn_num);

    Result<void, Error> closeSession(uint64_t session_id);
    Result<Session> getSession(uint64_t session_id) const;

    /// Get session summary (lightweight, for API).
    Result<SessionSummary> getSessionSummary(uint64_t session_id) const;


    /// List all sessions (for WebUI).
    std::vector<SessionSummary> listSessions() const;


    void addMemoryToSession(uint64_t session_id, uint64_t memory_id);

private:
    /// Detect intent from recent turns using LLM.
    std::string detectIntent(const Session& session);

    /// Extract facts from user input using LLM and store as memories.
    std::vector<uint64_t> extractFacts(const Session& session,
                                        const std::string& user_input);

    MemoryStore& store_;
    std::shared_ptr<LLMProvider> llm_;
    CapturePipeline& capture_;
    SnowflakeGenerator id_gen_;
    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, Session> sessions_;
    std::unique_ptr<SessionWAL> wal_;
};

}  // namespace amind
