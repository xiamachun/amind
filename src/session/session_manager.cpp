#include "session_manager.h"
#include "session_wal.h"
#include "capture/capture_pipeline.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace amind {

using json = nlohmann::json;

SessionManager::~SessionManager() = default;

SessionManager::SessionManager(MemoryStore& store, std::shared_ptr<LLMProvider> llm,
                                CapturePipeline& capture, const std::string& data_dir)
    : store_(store), llm_(std::move(llm)), capture_(capture), id_gen_(1) {
    if (!data_dir.empty()) {
        wal_ = std::make_unique<SessionWAL>(data_dir);
        if (wal_->open()) {
            sessions_ = wal_->replay();
            spdlog::info("SessionManager: restored {} sessions from WAL", sessions_.size());
        }
    }
}

Result<Session> SessionManager::startSession(const std::string& namespace_) {
    std::lock_guard lock(mutex_);

    // Purge closed sessions older than 1 hour to prevent unbounded growth
    static constexpr uint32_t SESSION_TTL_SEC = 3600;
    auto now = MemoryRecord::currentTimeSec();
    for (auto it = sessions_.begin(); it != sessions_.end(); ) {
        if (!it->second.active && (now - it->second.last_turn_at) > SESSION_TTL_SEC) {
            it = sessions_.erase(it);
        } else {
            ++it;
        }
    }

    Session session;
    session.session_id = id_gen_.nextId();
    session.namespace_ = namespace_;
    session.started_at = now;
    session.last_turn_at = session.started_at;
    sessions_[session.session_id] = session;
    if (wal_) wal_->appendStart(session);
    spdlog::info("Session started: id={}, namespace={}", session.session_id, namespace_);
    return session;
}

Result<void, Error> SessionManager::recordTurn(uint64_t session_id,
                                                const std::string& user_input,
                                                const std::string& agent_response) {
    // Phase 1: validate + update turn counter + snapshot for LLM calls
    Session session_copy;
    bool needs_intent = false;
    bool needs_facts = false;
    {
        std::lock_guard lock(mutex_);
        auto it = sessions_.find(session_id);
        if (it == sessions_.end()) {
            return makeError(Error::SessionNotFound, "session not found");
        }
        if (!it->second.active) {
            return makeError(Error::SessionClosed, "session is closed");
        }
        auto& session = it->second;
        session.turn_count++;
        session.last_turn_at = MemoryRecord::currentTimeSec();
        needs_intent = (llm_ && session.turns.size() >= 2);
        needs_facts = (llm_ && !user_input.empty() && user_input.length() > 10);
        session_copy = session;
    }

    // Phase 2: LLM calls without holding the lock (safe if LLM throws)
    std::string detected_intent;
    std::vector<uint64_t> extracted_ids;

    if (needs_intent) {
        detected_intent = detectIntent(session_copy);
    }
    if (needs_facts) {
        extracted_ids = extractFacts(session_copy, user_input);
    }

    // Phase 3: apply results under lock
    {
        std::lock_guard lock(mutex_);
        auto it = sessions_.find(session_id);
        if (it == sessions_.end()) return Result<void, Error>();

        auto& session = it->second;

        TurnRecord turn;
        turn.turn_number = session_copy.turn_count;
        turn.user_input = user_input;
        turn.agent_response = agent_response;
        turn.timestamp = static_cast<int64_t>(session_copy.last_turn_at);
        turn.detected_intent = std::move(detected_intent);
        turn.extracted_memory_ids = extracted_ids;

        if (!turn.detected_intent.empty()) {
            session.current_intent = turn.detected_intent;
        }
        for (auto mid : extracted_ids) {
            session.memory_ids.push_back(mid);
        }
        if (wal_) wal_->appendTurn(session_id, turn);
        session.turns.push_back(std::move(turn));
    }

    return Result<void, Error>();
}

Result<void, Error> SessionManager::recordTurn(uint64_t session_id, uint16_t turn_num) {
    std::lock_guard lock(mutex_);
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return makeError(Error::SessionNotFound, "session not found");
    }
    if (!it->second.active) {
        return makeError(Error::SessionClosed, "session is closed");
    }
    it->second.turn_count = turn_num;
    it->second.last_turn_at = MemoryRecord::currentTimeSec();
    return Result<void, Error>();
}

Result<void, Error> SessionManager::closeSession(uint64_t session_id) {
    std::lock_guard lock(mutex_);
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return makeError(Error::SessionNotFound, "session not found");
    }
    it->second.active = false;
    if (wal_) wal_->appendClose(session_id);

    // Verify persisted memories count
    size_t persisted = 0;
    for (auto mid : it->second.memory_ids) {
        auto rec = store_.get(mid);
        if (rec.ok()) ++persisted;
    }
    spdlog::info("Session closed: id={}, turns={}, memories={}/{} persisted, facts={}",
                 session_id, it->second.turn_count,
                 persisted, it->second.memory_ids.size(),
                 it->second.extracted_facts.size());
    return Result<void, Error>();
}

Result<Session> SessionManager::getSession(uint64_t session_id) const {
    std::lock_guard lock(mutex_);
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return makeError(Error::SessionNotFound, "session not found");
    }
    return it->second;
}

Result<SessionSummary> SessionManager::getSessionSummary(uint64_t session_id) const {
    std::lock_guard lock(mutex_);
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return makeError(Error::SessionNotFound, "session not found");
    }
    const auto& s = it->second;
    SessionSummary summary;
    summary.session_id = s.session_id;
    summary.namespace_ = s.namespace_;
    summary.turn_count = s.turn_count;
    summary.current_intent = s.current_intent;
    summary.memory_count = s.memory_ids.size();
    summary.fact_count = s.extracted_facts.size();
    summary.started_at = s.started_at;
    summary.last_turn_at = s.last_turn_at;
    summary.active = s.active;
    return summary;
}

void SessionManager::addMemoryToSession(uint64_t session_id, uint64_t memory_id) {
    std::lock_guard lock(mutex_);
    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        it->second.memory_ids.push_back(memory_id);
    }
}

std::string SessionManager::detectIntent(const Session& session) {
    if (!llm_ || session.turns.empty()) return "";

    // Build context from last 3 turns
    std::string context;
    size_t start = (session.turns.size() > 3) ? session.turns.size() - 3 : 0;
    for (size_t i = start; i < session.turns.size(); i++) {
        context += "User: " + session.turns[i].user_input + "\n";
        if (!session.turns[i].agent_response.empty()) {
            context += "Agent: " + session.turns[i].agent_response + "\n";
        }
    }

    std::string prompt =
        "Analyze the intent of the latest user message in this conversation.\n"
        "Conversation:\n" + context + "\n"
        "Respond with exactly one word from: question, task, chat, correction, preference\n"
        "Intent:";

    auto result = llm_->generate(prompt, "You are an intent classifier. Respond with only one word.");
    if (!result.ok()) return "";

    // Clean response
    auto intent = result.value();
    // Take first word only
    auto sp = intent.find_first_of(" \n\r\t");
    if (sp != std::string::npos) intent = intent.substr(0, sp);
    // Lowercase
    for (auto& c : intent) c = static_cast<char>(tolower(c));
    return intent;
}

std::vector<uint64_t> SessionManager::extractFacts(const Session& session,
                                                     const std::string& user_input) {
    std::vector<uint64_t> ids;
    if (!llm_) return ids;

    std::string prompt =
        "Extract factual statements from the following user message that should be remembered.\n"
        "User message: \"" + user_input + "\"\n\n"
        "Respond in JSON format: {\"facts\": [\"fact1\", \"fact2\"]}\n"
        "If no facts worth remembering, respond: {\"facts\": []}\n";

    auto result = llm_->generateJson(prompt,
        "You are a fact extractor. Extract user preferences, personal info, and important facts.");
    if (!result.ok()) return ids;

    try {
        auto j = json::parse(result.value());
        if (j.contains("facts") && j["facts"].is_array()) {
            for (const auto& fact : j["facts"]) {
                std::string fact_str = fact.get<std::string>();
                if (fact_str.empty()) continue;

                // Store fact via capture pipeline (pre_extracted=true: skip redundant LLM extraction)
                auto cap_result = capture_.capture(fact_str, session.namespace_, MemoryOwner::User, {}, true);
                if (cap_result.ok()) {
                    for (auto mid : *cap_result) {
                        ids.push_back(mid);
                    }
                }
            }
        }
    } catch (const json::exception&) {
        // LLM returned non-JSON, skip extraction
    }

    return ids;
}


std::vector<SessionSummary> SessionManager::listSessions() const {
    std::lock_guard lock(mutex_);
    std::vector<SessionSummary> result;
    result.reserve(sessions_.size());
    for (const auto& [id, s] : sessions_) {
        SessionSummary sum;
        sum.session_id = s.session_id;
        sum.namespace_ = s.namespace_;
        sum.turn_count = s.turn_count;
        sum.current_intent = s.current_intent;
        sum.memory_count = s.memory_ids.size();
        sum.fact_count = s.extracted_facts.size();
        sum.started_at = s.started_at;
        sum.last_turn_at = s.last_turn_at;
        sum.active = s.active;
        result.push_back(sum);
    }
    // Sort: active first, then by last_turn_at desc
    std::sort(result.begin(), result.end(), [](const SessionSummary& a, const SessionSummary& b) {
        if (a.active != b.active) return a.active > b.active;
        return a.last_turn_at > b.last_turn_at;
    });
    return result;
}

}  // namespace amind
