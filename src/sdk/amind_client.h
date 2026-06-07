#pragma once

#include "core/result.h"
#include "provider/connection_pool.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace amind {

// ── SDK Response Types ─────────────────────────────────────────────────────

struct StoreResponse {
    std::vector<uint64_t> memory_ids;
    bool async_refinement_scheduled{false};
};

struct RecallItem {
    uint64_t memory_id{0};
    std::string content;
    std::string agent_id;
    std::string user_id;
    std::string scope;
    std::string memory_type;
    std::string tier;
    std::string phase;
    std::string confidence;
    float score{0.0f};
    float semantic_score{0.0f};
};

struct MemoryInfo {
    uint64_t memory_id{0};
    std::string content;
    std::string agent_id;
    std::string user_id;
    std::string scope;
    std::string memory_type;
    std::string tier;
    std::string phase;
    std::string confidence;
    uint32_t version{0};
    uint64_t parent_id{0};
    float importance{0.0f};
};

struct VersionInfo {
    uint64_t memory_id{0};
    std::string content;
    uint32_t version{0};
    uint64_t parent_id{0};
};

struct SessionInfo {
    uint64_t session_id{0};
    std::string agent_id;
    std::string user_id;
    uint16_t turn_count{0};
    std::string current_intent;
    size_t memory_count{0};
    size_t fact_count{0};
    uint32_t started_at{0};
    uint32_t last_turn_at{0};
    bool active{false};
};

struct CoverageInfo {
    size_t total{0};
    size_t active{0};
    size_t stale{0};
    size_t conflicted{0};
};

struct ConflictItem {
    std::string memory_a;
    std::string memory_b;
    std::string conflict_type;
    std::string explanation;
};

struct DeleteResponse {
    bool deleted{false};
    size_t invalidated_count{0};
    std::vector<uint64_t> invalidated_ids;
};

// ── AmindClient ────────────────────────────────────────────────────────────

class AmindClient {
public:
    AmindClient(const std::string& host, int port, int timeout_ms = 30000);

    // Memory CRUD
    Result<StoreResponse> store(const std::string& content,
                                const std::string& agent_id = "default_agent",
                                const std::string& user_id = "anonymous",
                                const std::string& scope = "private",
                                const std::string& memory_type = "ephemeral");

    Result<std::vector<RecallItem>> recall(const std::string& query,
                                           const std::string& agent_id = "default_agent",
                                           const std::string& user_id = "anonymous",
                                           int top_k = 10);

    Result<MemoryInfo> get(uint64_t memory_id);

    Result<std::vector<VersionInfo>> getHistory(uint64_t memory_id);

    Result<void, Error> feedback(uint64_t memory_id, const std::string& text);

    Result<DeleteResponse> remove(uint64_t memory_id);

    // Intercept
    Result<std::vector<uint64_t>> intercept(
        const std::vector<std::pair<std::string, std::string>>& messages,
        const std::string& agent_id = "default_agent",
        const std::string& user_id = "anonymous");

    // Sessions
    Result<uint64_t> startSession(const std::string& agent_id = "default_agent",
                                  const std::string& user_id = "anonymous");

    Result<void, Error> recordTurn(uint64_t session_id,
                                    const std::string& user_input,
                                    const std::string& agent_response);

    Result<void, Error> closeSession(uint64_t session_id);

    Result<SessionInfo> getSessionSummary(uint64_t session_id);

    // MetaCognition
    Result<CoverageInfo> getCoverage(const std::string& agent_id = "");

    Result<std::vector<ConflictItem>> getConflicts();

    // Health
    Result<std::string> health();

private:
    std::string host_;
    int port_;
    int timeout_ms_;
    std::shared_ptr<HttpConnectionPool> pool_;
};

}  // namespace amind
