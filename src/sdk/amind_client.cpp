#include "amind_client.h"
#include "provider/ollama_provider.h"

#include <nlohmann/json.hpp>

namespace amind {

using json = nlohmann::json;

AmindClient::AmindClient(const std::string& host, int port, int timeout_ms)
    : host_(host), port_(port), timeout_ms_(timeout_ms),
      pool_(std::make_shared<HttpConnectionPool>()) {}

// ── Helpers ────────────────────────────────────────────────────────────────

static Result<json> parseJsonResponse(const HttpClient::Response& resp) {
    if (resp.status_code >= 400) {
        try {
            auto j = json::parse(resp.body);
            return makeError(Error::ProviderError,
                             j.value("error", "HTTP " + std::to_string(resp.status_code)));
        } catch (...) {
            return makeError(Error::ProviderError, "HTTP " + std::to_string(resp.status_code));
        }
    }
    try {
        return json::parse(resp.body);
    } catch (const json::exception& e) {
        return makeError(Error::ProviderError, std::string("JSON parse error: ") + e.what());
    }
}

// ── Memory CRUD ────────────────────────────────────────────────────────────

Result<StoreResponse> AmindClient::store(const std::string& content,
                                          const std::string& agent_id,
                                          const std::string& user_id,
                                          const std::string& scope,
                                          const std::string& memory_type) {
    json body;
    body["content"] = content;
    body["agent_id"] = agent_id;
    body["user_id"] = user_id;
    body["scope"] = scope;
    body["memory_type"] = memory_type;

    auto resp = HttpClient::post(host_, port_, "/v1/memories", body.dump(), timeout_ms_, pool_.get());
    if (!resp.ok()) return resp.error();

    auto j_result = parseJsonResponse(*resp);
    if (!j_result.ok()) return j_result.error();
    auto& j = *j_result;

    StoreResponse result;
    if (j.contains("memory_ids")) {
        for (const auto& id : j["memory_ids"]) {
            result.memory_ids.push_back(id.get<uint64_t>());
        }
    }
    result.async_refinement_scheduled = j.value("async_refinement_scheduled", false);
    return result;
}

Result<std::vector<RecallItem>> AmindClient::recall(const std::string& query,
                                                     const std::string& agent_id,
                                                     const std::string& user_id,
                                                     int top_k) {
    json body;
    body["query"] = query;
    body["agent_id"] = agent_id;
    body["user_id"] = user_id;
    body["top_k"] = top_k;

    auto resp = HttpClient::post(host_, port_, "/v1/memories/recall", body.dump(), timeout_ms_, pool_.get());
    if (!resp.ok()) return resp.error();

    auto j_result = parseJsonResponse(*resp);
    if (!j_result.ok()) return j_result.error();
    auto& j = *j_result;

    std::vector<RecallItem> results;
    if (j.contains("results")) {
        for (const auto& item : j["results"]) {
            RecallItem r;
            r.memory_id = item.value("memory_id", uint64_t(0));
            r.content = item.value("content", "");
            r.agent_id = item.value("agent_id", "");
            r.user_id = item.value("user_id", "");
            r.scope = item.value("scope", "");
            r.memory_type = item.value("memory_type", "");
            r.tier = item.value("tier", "");
            r.phase = item.value("phase", "");
            r.confidence = item.value("confidence", "");
            r.score = item.value("score", 0.0f);
            r.semantic_score = item.value("semantic_score", 0.0f);
            results.push_back(std::move(r));
        }
    }
    return results;
}

Result<MemoryInfo> AmindClient::get(uint64_t memory_id) {
    auto path = "/v1/memories/" + std::to_string(memory_id);
    auto resp = HttpClient::get(host_, port_, path, timeout_ms_, pool_.get());
    if (!resp.ok()) return resp.error();

    auto j_result = parseJsonResponse(*resp);
    if (!j_result.ok()) return j_result.error();
    auto& j = *j_result;

    MemoryInfo info;
    info.memory_id = j.value("memory_id", uint64_t(0));
    info.content = j.value("content", "");
    info.agent_id = j.value("agent_id", "");
    info.user_id = j.value("user_id", "");
    info.scope = j.value("scope", "");
    info.memory_type = j.value("memory_type", "");
    info.tier = j.value("tier", "");
    info.phase = j.value("phase", "");
    info.confidence = j.value("confidence", "");
    info.version = j.value("version", uint32_t(0));
    info.parent_id = j.value("parent_id", uint64_t(0));
    info.importance = j.value("importance", 0.0f);
    return info;
}

Result<std::vector<VersionInfo>> AmindClient::getHistory(uint64_t memory_id) {
    auto path = "/v1/memories/" + std::to_string(memory_id) + "/history";
    auto resp = HttpClient::get(host_, port_, path, timeout_ms_, pool_.get());
    if (!resp.ok()) return resp.error();

    auto j_result = parseJsonResponse(*resp);
    if (!j_result.ok()) return j_result.error();
    auto& j = *j_result;

    std::vector<VersionInfo> versions;
    if (j.contains("versions")) {
        for (const auto& item : j["versions"]) {
            VersionInfo v;
            v.memory_id = item.value("memory_id", uint64_t(0));
            v.content = item.value("content", "");
            v.version = item.value("version", uint32_t(0));
            v.parent_id = item.value("parent_id", uint64_t(0));
            versions.push_back(std::move(v));
        }
    }
    return versions;
}

Result<void, Error> AmindClient::feedback(uint64_t memory_id, const std::string& text) {
    auto path = "/v1/memories/" + std::to_string(memory_id) + "/feedback";
    json body;
    body["feedback"] = text;

    auto resp = HttpClient::post(host_, port_, path, body.dump(), timeout_ms_, pool_.get());
    if (!resp.ok()) return resp.error();

    if (resp->status_code >= 400) {
        return makeError(Error::ProviderError, "feedback failed: HTTP " + std::to_string(resp->status_code));
    }
    return Result<void, Error>();
}

Result<DeleteResponse> AmindClient::remove(uint64_t memory_id) {
    auto path = "/v1/memories/" + std::to_string(memory_id);
    auto resp = HttpClient::del(host_, port_, path, timeout_ms_, pool_.get());
    if (!resp.ok()) return resp.error();

    auto j_result = parseJsonResponse(*resp);
    if (!j_result.ok()) return j_result.error();
    auto& j = *j_result;

    DeleteResponse result;
    result.deleted = j.value("deleted", false);
    result.invalidated_count = j.value("invalidated_count", size_t(0));
    if (j.contains("invalidated_ids")) {
        for (const auto& id : j["invalidated_ids"]) {
            result.invalidated_ids.push_back(id.get<uint64_t>());
        }
    }
    return result;
}

// ── Intercept ──────────────────────────────────────────────────────────────

Result<std::vector<uint64_t>> AmindClient::intercept(
    const std::vector<std::pair<std::string, std::string>>& messages,
    const std::string& agent_id,
    const std::string& user_id) {
    json body;
    body["agent_id"] = agent_id;
    body["user_id"] = user_id;
    json msgs = json::array();
    for (const auto& [role, content] : messages) {
        msgs.push_back({{"role", role}, {"content", content}});
    }
    body["messages"] = msgs;

    auto resp = HttpClient::post(host_, port_, "/v1/intercept", body.dump(), timeout_ms_, pool_.get());
    if (!resp.ok()) return resp.error();

    auto j_result = parseJsonResponse(*resp);
    if (!j_result.ok()) return j_result.error();
    auto& j = *j_result;

    std::vector<uint64_t> ids;
    if (j.contains("captured_ids")) {
        for (const auto& id : j["captured_ids"]) {
            ids.push_back(id.get<uint64_t>());
        }
    }
    return ids;
}

// ── Sessions ───────────────────────────────────────────────────────────────

Result<uint64_t> AmindClient::startSession(const std::string& agent_id,
                                           const std::string& user_id) {
    json body;
    body["agent_id"] = agent_id;
    body["user_id"] = user_id;

    auto resp = HttpClient::post(host_, port_, "/v1/sessions/start", body.dump(), timeout_ms_, pool_.get());
    if (!resp.ok()) return resp.error();

    auto j_result = parseJsonResponse(*resp);
    if (!j_result.ok()) return j_result.error();
    auto& j = *j_result;

    return j.value("session_id", uint64_t(0));
}

Result<void, Error> AmindClient::recordTurn(uint64_t session_id,
                                             const std::string& user_input,
                                             const std::string& agent_response) {
    auto path = "/v1/sessions/" + std::to_string(session_id) + "/turn";
    json body;
    body["user_input"] = user_input;
    body["agent_response"] = agent_response;

    auto resp = HttpClient::post(host_, port_, path, body.dump(), timeout_ms_, pool_.get());
    if (!resp.ok()) return resp.error();

    if (resp->status_code >= 400) {
        return makeError(Error::SessionNotFound, "recordTurn failed");
    }
    return Result<void, Error>();
}

Result<void, Error> AmindClient::closeSession(uint64_t session_id) {
    auto path = "/v1/sessions/" + std::to_string(session_id) + "/close";
    json body = json::object();

    auto resp = HttpClient::post(host_, port_, path, body.dump(), timeout_ms_, pool_.get());
    if (!resp.ok()) return resp.error();

    if (resp->status_code >= 400) {
        return makeError(Error::SessionNotFound, "closeSession failed");
    }
    return Result<void, Error>();
}

Result<SessionInfo> AmindClient::getSessionSummary(uint64_t session_id) {
    auto path = "/v1/sessions/" + std::to_string(session_id) + "/summary";
    auto resp = HttpClient::get(host_, port_, path, timeout_ms_, pool_.get());
    if (!resp.ok()) return resp.error();

    auto j_result = parseJsonResponse(*resp);
    if (!j_result.ok()) return j_result.error();
    auto& j = *j_result;

    SessionInfo info;
    info.session_id = j.value("session_id", uint64_t(0));
    info.agent_id = j.value("agent_id", "");
    info.user_id = j.value("user_id", "");
    info.turn_count = j.value("turn_count", uint16_t(0));
    info.current_intent = j.value("current_intent", "");
    info.memory_count = j.value("memory_count", size_t(0));
    info.fact_count = j.value("fact_count", size_t(0));
    info.started_at = j.value("started_at", uint32_t(0));
    info.last_turn_at = j.value("last_turn_at", uint32_t(0));
    info.active = j.value("active", false);
    return info;
}

// ── MetaCognition ──────────────────────────────────────────────────────────

Result<CoverageInfo> AmindClient::getCoverage(const std::string& agent_id) {
    std::string path = "/v1/metacognition/coverage";
    if (!agent_id.empty()) {
        path += "?agent_id=" + agent_id;
    }
    auto resp = HttpClient::get(host_, port_, path, timeout_ms_, pool_.get());
    if (!resp.ok()) return resp.error();

    auto j_result = parseJsonResponse(*resp);
    if (!j_result.ok()) return j_result.error();
    auto& j = *j_result;

    CoverageInfo info;
    info.total = j.value("total", size_t(0));
    info.active = j.value("active", size_t(0));
    info.stale = j.value("stale", size_t(0));
    info.conflicted = j.value("conflicted", size_t(0));
    return info;
}

Result<std::vector<ConflictItem>> AmindClient::getConflicts() {
    auto resp = HttpClient::get(host_, port_, "/v1/metacognition/conflicts", timeout_ms_, pool_.get());
    if (!resp.ok()) return resp.error();

    auto j_result = parseJsonResponse(*resp);
    if (!j_result.ok()) return j_result.error();
    auto& j = *j_result;

    std::vector<ConflictItem> conflicts;
    for (const auto& item : j) {
        ConflictItem c;
        c.memory_a = item.value("memory_a", "");
        c.memory_b = item.value("memory_b", "");
        c.conflict_type = item.value("conflict_type", "");
        c.explanation = item.value("explanation", "");
        conflicts.push_back(std::move(c));
    }
    return conflicts;
}

// ── Health ─────────────────────────────────────────────────────────────────

Result<std::string> AmindClient::health() {
    auto resp = HttpClient::get(host_, port_, "/v1/health", timeout_ms_, pool_.get());
    if (!resp.ok()) return resp.error();

    if (resp->status_code != 200) {
        return makeError(Error::ProviderUnavailable, "health check failed");
    }
    try {
        auto j = json::parse(resp->body);
        return j.value("status", "unknown");
    } catch (...) {
        return makeError(Error::ProviderError, "invalid health response");
    }
}

}  // namespace amind
