#include "rest_server.h"
#include "variable_manager.h"
#include "config/v2_config.h"
#include "coordinator/remove_coordinator.h"
#include "lineage/lineage_index.h"
#include "forget/forget_engine.h"
#include "observability/memory_event_log.h"
#include "retrieval/staleness_filter.h"
#include "reconcile/reconciler.h"
#include "gate/write_gate.h"

#include <arpa/inet.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>

namespace amind {

using json = nlohmann::json;

RestServer::RestServer(Engine& engine, const std::string& host, int port,
                       int max_connections, int request_timeout_ms)
    : engine_(engine), host_(host), port_(port),
      max_connections_(max_connections), request_timeout_ms_(request_timeout_ms),
      api_token_(engine.config().get("api_token")) {}

RestServer::~RestServer() {
    stop();
}

Result<void, Error> RestServer::start() {
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        return makeError(Error::IOError, "socket() failed");
    }

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    inet_pton(AF_INET, host_.c_str(), &addr.sin_addr);

    if (bind(server_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(server_fd_);
        return makeError(Error::IOError, "bind() failed on port " + std::to_string(port_));
    }

    if (listen(server_fd_, max_connections_) < 0) {
        close(server_fd_);
        return makeError(Error::IOError, "listen() failed");
    }

    running_ = true;
    spdlog::info("REST server listening on {}:{}", host_, port_);

    // Start worker thread pool
    size_t hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) hw_threads = 4;
    size_t num_workers = std::min(static_cast<size_t>(max_connections_), hw_threads);
    for (size_t i = 0; i < num_workers; i++) {
        workers_.emplace_back([this]() { workerLoop(); });
    }
    spdlog::info("REST server thread pool: {} workers", num_workers);

    // Accept loop — enqueue connections for workers
    while (running_) {
        struct sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd_,
                               reinterpret_cast<struct sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            if (!running_) break;
            continue;
        }

        struct timeval tv;
        tv.tv_sec = request_timeout_ms_ / 1000;
        tv.tv_usec = (request_timeout_ms_ % 1000) * 1000;
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        {
            std::lock_guard lk(conn_mutex_);
            conn_queue_.push(client_fd);
        }
        conn_cv_.notify_one();
    }

    // Shutdown: wake all workers
    conn_cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
    workers_.clear();

    return Result<void, Error>();
}

void RestServer::workerLoop() {
    while (true) {
        int client_fd = -1;
        {
            std::unique_lock lk(conn_mutex_);
            conn_cv_.wait(lk, [this]() {
                return !conn_queue_.empty() || !running_;
            });
            if (!running_ && conn_queue_.empty()) return;
            if (conn_queue_.empty()) continue;
            client_fd = conn_queue_.front();
            conn_queue_.pop();
        }
        try {
            handleClient(client_fd);
        } catch (const std::exception& e) {
            spdlog::error("workerLoop: unhandled exception: {}", e.what());
            close(client_fd);
        } catch (...) {
            spdlog::error("workerLoop: unknown exception");
            close(client_fd);
        }
    }
}

void RestServer::stop() {
    running_ = false;
    if (server_fd_ >= 0) {
        ::shutdown(server_fd_, SHUT_RDWR);
        close(server_fd_);
        server_fd_ = -1;
    }
    conn_cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
    workers_.clear();
}

bool RestServer::running() const {
    return running_.load();
}

void RestServer::sendAll(int fd, const std::string& data) {
    size_t total_sent = 0;
    while (total_sent < data.size()) {
        ssize_t sent = send(fd, data.c_str() + total_sent,
                            data.size() - total_sent, 0);
        if (sent <= 0) break;
        total_sent += static_cast<size_t>(sent);
    }
}

void RestServer::handleClient(int client_fd) {
    // Read headers first (up to 8KB), then body based on Content-Length
    std::string request;
    char buf[8192];

    // Phase 1: read until we have complete headers
    while (true) {
        ssize_t n = recv(client_fd, buf, sizeof(buf), 0);
        if (n <= 0) { close(client_fd); return; }
        request.append(buf, static_cast<size_t>(n));
        if (request.find("\r\n\r\n") != std::string::npos) break;
        if (request.size() > 64 * 1024) break;  // header too large
    }

    auto header_end = request.find("\r\n\r\n");
    if (header_end == std::string::npos) { close(client_fd); return; }

    // Parse Content-Length from headers (case-insensitive)
    static constexpr size_t MAX_BODY_SIZE = 16 * 1024 * 1024;  // 16 MB
    size_t content_length = 0;
    {
        std::string headers_lower = request.substr(0, header_end);
        for (auto& c : headers_lower) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
        auto cl_pos = headers_lower.find("content-length:");
        if (cl_pos != std::string::npos) {
            auto val_start = cl_pos + 15;
            while (val_start < header_end && request[val_start] == ' ') val_start++;
            auto cl_end = request.find("\r\n", val_start);
            auto val_str = request.substr(val_start, cl_end - val_start);
            try {
                content_length = std::stoull(val_str);
            } catch (...) {
                sendAll(client_fd, errorResponse(400, "invalid Content-Length"));
                close(client_fd);
                return;
            }
            if (content_length > MAX_BODY_SIZE) {
                sendAll(client_fd, errorResponse(413, "request body too large"));
                close(client_fd);
                return;
            }
        }
    }

    // Phase 2: read remaining body if needed
    size_t body_offset = header_end + 4;
    size_t body_received = request.size() - body_offset;
    while (body_received < content_length) {
        size_t to_read = std::min(sizeof(buf), content_length - body_received);
        ssize_t n = recv(client_fd, buf, to_read, 0);
        if (n <= 0) break;
        request.append(buf, static_cast<size_t>(n));
        body_received += static_cast<size_t>(n);
    }

    // Parse request line
    auto first_line_end = request.find("\r\n");
    auto first_line = request.substr(0, first_line_end);
    auto sp1 = first_line.find(' ');
    auto sp2 = first_line.find(' ', sp1 + 1);
    std::string method = first_line.substr(0, sp1);
    std::string path = first_line.substr(sp1 + 1, sp2 - sp1 - 1);

    std::string body;
    if (body_offset < request.size()) {
        body = request.substr(body_offset);
    }
    // Auth check (before routing)
    std::string headers_section = request.substr(0, header_end);
    auto auth_err = checkAuth(headers_section, path);
    if (!auth_err.empty()) {
        sendAll(client_fd, auth_err);
        close(client_fd);
        return;
    }

    std::string response;
    try {
        response = routeRequest(method, path, body);
    } catch (const std::exception& e) {
        spdlog::error("handleClient: unhandled exception in route {}: {}", path, e.what());
        response = errorResponse(500, "internal server error");
    } catch (...) {
        spdlog::error("handleClient: unknown exception in route {}", path);
        response = errorResponse(500, "internal server error");
    }
    sendAll(client_fd, response);
    close(client_fd);
}

std::string RestServer::routeRequest(const std::string& method, const std::string& path,
                                     const std::string& body) {
    // Strip query string for path matching but keep full path for param extraction
    auto qmark = path.find('?');
    std::string clean_path = (qmark != std::string::npos) ? path.substr(0, qmark) : path;

    // Health / metrics / pipeline
    if (method == "GET" && clean_path == "/v1/health") return handleHealth();
    if (method == "GET" && clean_path == "/v1/metrics") return handleMetrics();
    // /v1/pipeline/stats and /v1/pipeline/reconcile-log removed in Phase 4 —
    // use /v1/events?kind=Reconcile and /v1/events/stats instead.

    // Memory CRUD
    if (method == "POST" && clean_path == "/v1/memories") return handleStore(body);
    if (method == "POST" && clean_path == "/v1/memories/recall") return handleRecall(body);

    // Intercept
    if (method == "POST" && clean_path == "/v1/intercept") return handleIntercept(body);

    // MetaCognition
    if (method == "GET" && clean_path == "/v1/metacognition/conflicts") return handleConflicts();

    // Session
    if (method == "POST" && clean_path == "/v1/sessions/start") return handleSessionStart(body);

    // Backup
    if (clean_path == "/v1/backup/export" && method == "GET") {
        return handleBackupExport(extractQueryParam(path, "type"));
    }
    if (clean_path == "/v1/backup/import" && method == "POST") {
        return handleBackupImport(extractQueryParam(path, "type"), body);
    }

    // Variables API
    if (method == "GET" && clean_path == "/v1/variables") return handleShowVariables(path);
    if (method == "POST" && clean_path == "/v1/config/reload") return handleReloadConfig();
    if (method == "PUT" && clean_path.find("/v1/variables/") == 0) {
        auto var_name = clean_path.substr(14);  // "/v1/variables/" is 14 chars
        return handleSetVariable(var_name, body);
    }

    // Auth key management
    if (method == "POST" && clean_path == "/v1/auth/keys") return handleCreateKey(body);
    if (method == "GET" && clean_path == "/v1/auth/keys") return handleListKeys();
    if (method == "DELETE" && clean_path.find("/v1/auth/keys/") == 0) {
        auto key_id = clean_path.substr(14);  // "/v1/auth/keys/" is 14 chars
        return handleRevokeKey(key_id);
    }

    // Agent memory bulk delete (must be before dynamic /v1/memories/{id} handler)
    if (method == "DELETE" && clean_path.find("/v1/memories/agent/") == 0) {
        auto agent_id = urlDecode(clean_path.substr(19));
        return handleDeleteAgentMemories(agent_id);
    }

    // Agent Management Endpoints
    if (method == "POST" && clean_path == "/v1/agents") return handleRegisterAgent(body);
    if (method == "GET" && clean_path == "/v1/agents") return handleListAgents();
    if (method == "DELETE" && clean_path.find("/v1/agents/") == 0) {
        auto agent_id = urlDecode(clean_path.substr(11));
        return handleUnregisterAgent(agent_id);
    }

    // WebUI: list (must be before dynamic path handlers)
    if (method == "GET" && clean_path == "/v1/memories/list") return handleListMemories(path);
    if (method == "GET" && clean_path == "/v1/graph/edges") return handleListEdges(path);
    if (method == "GET" && clean_path == "/v1/sessions/list") return handleListSessions();
    if (method == "GET" && clean_path == "/v1/metacognition/coverage") {
        return handleCoverageStats(extractQueryParam(path, "namespace"));
    }

    // Legacy /v1/gate/log* /v1/forget/log /v1/recall/stale-log routes removed
    // in Phase 4. All observability now flows through the unified endpoints
    // below, and resurrect goes through POST /v1/admin/resurrect/{event_id}.

    // Unified observability — Memory Event Log
    if (method == "GET"  && clean_path == "/v1/events")        return handleEventsQuery(path);
    if (method == "GET"  && clean_path == "/v1/events/stats")  return handleEventsStats(path);
    if (method == "GET"  && clean_path.find("/v1/traces/") == 0) {
        return handleTraceById(clean_path.substr(std::string("/v1/traces/").size()));
    }
    if (method == "POST" && clean_path.find("/v1/admin/resurrect/") == 0) {
        return handleAdminResurrect(
            clean_path.substr(std::string("/v1/admin/resurrect/").size()), body);
    }
    // memory/{id}/trace handled within the existing /v1/memories/{id} block below

    // Manual triggers for V2 background workers
    if (method == "POST" && clean_path == "/v1/admin/forget/run") return handleAdminForgetRun();
    if (method == "POST" && clean_path == "/v1/admin/consolidation/run") return handleAdminConsolidationRun();

    // Dynamic paths: /v1/memories/{id}[/history|/feedback]
    if (clean_path.find("/v1/memories/") == 0 && clean_path != "/v1/memories/recall") {
        auto rest = clean_path.substr(13);
        auto slash_pos = rest.find('/');
        std::string id_str = (slash_pos != std::string::npos) ? rest.substr(0, slash_pos) : rest;
        std::string sub_path = (slash_pos != std::string::npos) ? rest.substr(slash_pos) : "";

        if (method == "GET" && sub_path.empty()) return handleGetMemory(id_str);
        if (method == "GET" && sub_path == "/history") return handleGetHistory(id_str);
        if (method == "GET" && sub_path == "/trace") return handleMemoryTrace(id_str);
        if (method == "POST" && sub_path == "/feedback") return handleFeedback(id_str, body);
        if (method == "DELETE" && sub_path.empty()) return handleDeleteMemory(id_str);
        if (method == "POST" && sub_path == "/archive") return handleArchiveMemory(id_str);
    }

    // Session dynamic: /v1/sessions/{id}/turn|close|summary
    if (clean_path.find("/v1/sessions/") == 0 && clean_path != "/v1/sessions/start") {
        auto rest = clean_path.substr(13);
        auto slash_pos = rest.find('/');
        std::string id_str = (slash_pos != std::string::npos) ? rest.substr(0, slash_pos) : rest;
        std::string sub_path = (slash_pos != std::string::npos) ? rest.substr(slash_pos) : "";

        if (method == "POST" && sub_path == "/turn") return handleSessionTurn(id_str, body);
        if (method == "POST" && sub_path == "/close") return handleSessionClose(id_str, body);
        if (method == "GET" && sub_path == "/summary") return handleSessionSummary(id_str);
    }



    // WebUI: graph neighbors. Pass the full path so the handler can parse
    // optional flags (e.g. ?include_incoming=1) from the query string.
    if (method == "GET" && clean_path.find("/v1/graph/neighbors/") == 0) {
        auto id_str = clean_path.substr(20);
        bool include_incoming = false;
        if (auto v = extractQueryParam(path, "include_incoming"); !v.empty()) {
            include_incoming = (v == "1" || v == "true");
        }
        return handleGraphNeighbors(id_str, include_incoming);
    }

    return errorResponse(404, "not found");
}

// ── Route Handlers ──────────────────────────────────────────────────────────

std::string RestServer::handleStore(const std::string& body) {
    try {
        json j = json::parse(body);
        auto content = j.value("content", "");
        auto agent_id = j.value("agent_id", "default_agent");
        auto user_id = j.value("user_id", "anonymous");
        
        // Parse new scope and type parameters
        auto scope_str = j.value("scope", "private");
        auto type_str = j.value("memory_type", "ephemeral");
        
        MemoryScope scope = scopeFromString(scope_str);
        MemoryType memory_type = memoryTypeFromString(type_str);

        // Client-supplied metadata is flattened to string→string. Nested
        // values are JSON-stringified so we never lose data.
        std::map<std::string, std::string> user_metadata;
        if (j.contains("metadata") && j["metadata"].is_object()) {
            for (auto it = j["metadata"].begin(); it != j["metadata"].end(); ++it) {
                if (it.value().is_string()) {
                    user_metadata[it.key()] = it.value().get<std::string>();
                } else if (it.value().is_null()) {
                    // skip
                } else {
                    user_metadata[it.key()] = it.value().dump();
                }
            }
        }

        // Reject if client explicitly marked this as an error/noise message.
        if (user_metadata.count("is_error") && user_metadata["is_error"] == "true") {
            return jsonResponse(200, R"({"memory_ids":[],"filtered":"is_error metadata flag"})");
        }

        auto result = engine_.capturePipeline().capture(
            content, agent_id, user_id, scope, memory_type, std::move(user_metadata));
        if (!result.ok()) {
            return errorResponse(500, result.error().toString());
        }

        json resp;
        resp["memory_ids"] = *result;
        resp["async_refinement_scheduled"] = true;
        return jsonResponse(200, resp.dump());
    } catch (const json::exception& e) {
        return errorResponse(400, "invalid JSON: " + std::string(e.what()));
    } catch (const std::exception& e) {
        spdlog::error("handleStore: {}", e.what());
        return errorResponse(500, "store failed: " + std::string(e.what()));
    }
}

std::string RestServer::handleRecall(const std::string& body) {
    try {
        auto j = json::parse(body);
        auto query = j.value("query", "");
        auto agent_id = j.value("agent_id", "default_agent");
        auto user_id = j.value("user_id", "anonymous");
        auto top_k = j.value("top_k", 10);
        
        // Optional client-side filters keyed on user_metadata fields.
        // Server applies an exact-match AND filter after the recall pipeline.
        std::map<std::string, std::string> filters;
        if (j.contains("filters") && j["filters"].is_object()) {
            for (auto it = j["filters"].begin(); it != j["filters"].end(); ++it) {
                if (it.value().is_string()) {
                    filters[it.key()] = it.value().get<std::string>();
                }
            }
        }

        // Over-fetch when filters are present so we still return ~top_k after pruning.
        size_t fetch_k = filters.empty() ? static_cast<size_t>(top_k)
                                         : static_cast<size_t>(top_k) * 4;
        auto result = engine_.retrievalPipeline().recall(query, agent_id, user_id, fetch_k);
        if (!result.ok()) {
            return errorResponse(500, result.error().toString());
        }

        json resp = json::array();
        size_t emitted = 0;
        for (const auto& sm : *result) {
            // Apply metadata filters (AND semantics; missing key = mismatch).
            if (!filters.empty()) {
                bool match = true;
                for (const auto& [k, v] : filters) {
                    auto it = sm.record.user_metadata.find(k);
                    if (it == sm.record.user_metadata.end() || it->second != v) {
                        match = false;
                        break;
                    }
                }
                if (!match) continue;
            }
            json item;
            item["memory_id"] = sm.record.memory_id;
            // Annotate confidential memories so client agents know not to reveal them.
            // Check both: (1) LLM-classified confidential flag in metadata (Stage 2),
            // (2) rule-based hasSecrecyMarker fallback for Raw records or pre-V2 data.
            bool is_confidential = false;
            {
                auto it = sm.record.user_metadata.find("confidential");
                if (it != sm.record.user_metadata.end() && it->second == "true") {
                    is_confidential = true;
                }
            }
            if (!is_confidential) {
                is_confidential = WriteGate::hasSecrecyMarker(sm.record.content);
            }
            if (is_confidential) {
                item["content"] = "[CONFIDENTIAL - user marked this as secret, do not reveal the actual content] "
                                  + sm.record.content;
                item["confidential"] = true;
            } else {
                item["content"] = sm.record.content;
            }
            item["scope"] = scopeToString(sm.record.scope);
            item["memory_type"] = memoryTypeToString(sm.record.memory_type);
            item["tier"] = memoryTierToString(sm.record.tier);
            item["phase"] = phaseToString(sm.record.phase);
            item["confidence"] = confidenceToString(sm.record.confidence_level);
            item["score"] = sm.total_score;
            item["semantic_score"] = sm.semantic_score;
            if (!sm.record.user_metadata.empty()) {
                json meta = json::object();
                for (const auto& [k, v] : sm.record.user_metadata) meta[k] = v;
                item["metadata"] = std::move(meta);
            }
            resp.push_back(std::move(item));
            if (++emitted >= static_cast<size_t>(top_k)) break;
        }
        return jsonResponse(200, json{{"results", resp}}.dump());
    } catch (const json::exception& e) {
        return errorResponse(400, "invalid JSON");
    } catch (const std::exception& e) {
        spdlog::error("handleRecall: {}", e.what());
        return errorResponse(500, "recall failed: " + std::string(e.what()));
    }
}

std::string RestServer::handleGetMemory(const std::string& id_str) {
    try {
        uint64_t id = std::stoull(id_str);
        auto result = engine_.memoryStore().get(id);
        if (!result.ok()) return errorResponse(404, "memory not found");

        json resp;
        resp["memory_id"] = result->memory_id;
        resp["content"] = result->content;
        resp["agent_id"] = result->agent_id;
        resp["user_id"] = result->user_id;
        resp["scope"] = scopeToString(result->scope);
        resp["memory_type"] = memoryTypeToString(result->memory_type);
        resp["tier"] = memoryTierToString(result->tier);
        resp["phase"] = phaseToString(result->phase);
        resp["confidence"] = confidenceToString(result->confidence_level);
        resp["version"] = result->mem_version;
        resp["parent_id"] = result->parent_id;
        resp["importance"] = result->importance;
        if (!result->user_metadata.empty()) {
            json meta = json::object();
            for (const auto& [k, v] : result->user_metadata) meta[k] = v;
            resp["metadata"] = std::move(meta);
        }
        return jsonResponse(200, resp.dump());
    } catch (...) {
        return errorResponse(400, "invalid memory ID");
    }
}

std::string RestServer::handleGetHistory(const std::string& id_str) {
    try {
        uint64_t id = std::stoull(id_str);
        auto result = engine_.memoryStore().getHistory(id);
        if (!result.ok()) return errorResponse(404, "not found");

        json resp = json::array();
        for (const auto& r : *result) {
            json item;
            item["memory_id"] = r.memory_id;
            item["content"] = r.content;
            item["version"] = r.mem_version;
            item["parent_id"] = r.parent_id;
            resp.push_back(std::move(item));
        }
        return jsonResponse(200, json{{"versions", resp}}.dump());
    } catch (...) {
        return errorResponse(400, "invalid memory ID");
    }
}

std::string RestServer::handleFeedback(const std::string& id_str, const std::string& body) {
    try {
        auto j = json::parse(body);
        return jsonResponse(200, R"({"action_taken":"acknowledged"})");
    } catch (...) {
        return errorResponse(400, "invalid JSON");
    }
}

std::string RestServer::handleDeleteMemory(const std::string& id_str) {
    try {
        uint64_t id = std::stoull(id_str);

        // V2: Delegate to RemoveCoordinator when lineage propagation is enabled
        if (engine_.featureGate().isLineagePropagationEnabled()) {
            auto get_record = [this](uint64_t mid) -> MemoryRecord* {
                auto result = engine_.memoryStore().get(mid);
                if (!result.ok()) return nullptr;
                // Store in temporary to return pointer (coordinator uses it transiently)
                thread_local MemoryRecord tmp;
                tmp = std::move(result.value());
                return &tmp;
            };
            auto persist = [this](uint64_t mid, const MemoryRecord& rec) {
                // Re-serialize to LSM via update path
                engine_.memoryStore().remove(mid);
            };
            auto hnsw_delete = [](uint64_t /*mid*/) {
                // Already handled inside memoryStore().remove()
            };

            auto remove_result = engine_.removeCoordinator().remove(
                id, RemoveReason::UserDelete, get_record, persist, hnsw_delete);

            if (!remove_result.success) {
                return errorResponse(404, remove_result.error_message.empty() ? "not found" : remove_result.error_message);
            }

            // Clean graph edges for primary + invalidated descendants
            engine_.graphStore().removeNode(id);
            for (uint64_t desc : remove_result.invalidated_descendants) {
                engine_.graphStore().removeNode(desc);
            }

            json resp;
            resp["deleted"] = true;
            resp["invalidated_count"] = remove_result.invalidated_descendants.size();
            resp["invalidated_ids"] = remove_result.invalidated_descendants;
            return jsonResponse(200, resp.dump());
        }

        // V1 fallback: direct remove
        auto result = engine_.memoryStore().remove(id);
        if (!result.ok()) return errorResponse(404, "not found");
        engine_.graphStore().removeNode(id);
        return jsonResponse(200, R"({"deleted":true})");
    } catch (...) {
        return errorResponse(400, "invalid ID");
    }
}

std::string RestServer::handleArchiveMemory(const std::string& id_str) {
    try {
        uint64_t id = std::stoull(id_str);
        auto result = engine_.memoryStore().archive(id);
        if (!result.ok()) return errorResponse(404, "not found");
        return jsonResponse(200, R"({"archived":true})");
    } catch (...) {
        return errorResponse(400, "invalid ID");
    }
}

std::string RestServer::handleDeleteAgentMemories(const std::string& agent_id) {
    // In the new architecture, we delete memories by agent_id.
    // Note: This is a heavy operation. In production, this might be restricted or async.
    // In new architecture, delete all memories from this agent's store via scanAll + remove
    std::vector<uint64_t> deleted_ids;
    engine_.memoryStore().scanAll([&](const MemoryRecord& rec) {
        if (rec.isAlive()) {
            deleted_ids.push_back(rec.memory_id);
        }
    });
    for (uint64_t id : deleted_ids) {
        engine_.memoryStore().remove(id);
    }

    for (uint64_t id : deleted_ids) {
        engine_.graphStore().removeNode(id);
    }

    json resp;
    resp["deleted"] = deleted_ids.size();
    resp["agent_id"] = agent_id;
    return jsonResponse(200, resp.dump());
}

std::string RestServer::handleIntercept(const std::string& body) {
    try {
        auto j = json::parse(body);
        auto agent_id = j.value("agent_id", "default_agent");
        auto user_id = j.value("user_id", "anonymous");
        std::vector<std::pair<std::string, std::string>> messages;
        if (j.contains("messages") && j["messages"].is_array()) {
            for (const auto& m : j["messages"]) {
                messages.emplace_back(m.value("role", ""), m.value("content", ""));
            }
        }
        auto result = engine_.capturePipeline().interceptCapture(messages, agent_id, user_id);
        json resp;
        resp["capture_scheduled"] = true;
        if (result.ok()) resp["captured_ids"] = *result;
        return jsonResponse(200, resp.dump());
    } catch (...) {
        return errorResponse(400, "invalid JSON");
    }
}

std::string RestServer::handleCoverage(const std::string& query) {
    auto stats = engine_.metaCognition().getCoverage("");
    json resp;
    resp["total_memories"] = stats.total;
    resp["active_memories"] = stats.active;
    resp["stale_memories"] = stats.stale;
    resp["conflicted_memories"] = stats.conflicted;
    return jsonResponse(200, resp.dump());
}

std::string RestServer::handleConflicts() {
    auto conflicts = engine_.metaCognition().getConflicts();
    json resp = json::array();
    for (const auto& c : conflicts) {
        json item;
        item["memory_a"] = std::to_string(c.memory_a);
        item["memory_b"] = std::to_string(c.memory_b);
        item["conflict_type"] = c.conflict_type;
        item["explanation"] = "Similarity score exceeds threshold";
        resp.push_back(std::move(item));
    }
    return jsonResponse(200, resp.dump());
}

std::string RestServer::handleSessionStart(const std::string& body) {
    try {
        auto j = json::parse(body);
        auto agent_id = j.value("agent_id", "default_agent");
        auto user_id = j.value("user_id", "anonymous");
        auto result = engine_.sessionManager().startSession(agent_id, user_id);
        if (!result.ok()) return errorResponse(500, result.error().toString());
        json resp;
        resp["session_id"] = result->session_id;
        return jsonResponse(200, resp.dump());
    } catch (...) {
        return errorResponse(400, "invalid JSON");
    }
}

std::string RestServer::handleSessionTurn(const std::string& id_str, const std::string& body) {
    try {
        uint64_t sid = std::stoull(id_str);
        auto j = json::parse(body);
        auto user_input = j.value("user_input", "");
        auto agent_response = j.value("agent_response", "");

        if (!user_input.empty() && !agent_response.empty()) {
            // Enhanced path: full turn with intent detection + fact extraction
            auto result = engine_.sessionManager().recordTurn(sid, user_input, agent_response);
            if (!result.ok()) return errorResponse(404, result.error().toString());

            auto summary = engine_.sessionManager().getSessionSummary(sid);
            json resp;
            resp["ok"] = true;
            if (summary.ok()) {
                resp["current_intent"] = summary->current_intent;
                resp["turn_count"] = summary->turn_count;
                resp["fact_count"] = summary->fact_count;
            }
            return jsonResponse(200, resp.dump());
        } else {
            // Backward-compatible: simple turn record
            auto turn_num = j.value("turn_number", 0);
            auto result = engine_.sessionManager().recordTurn(sid, static_cast<uint16_t>(turn_num));
            if (!result.ok()) return errorResponse(404, result.error().toString());
            return jsonResponse(200, R"({"ok":true})");
        }
    } catch (...) {
        return errorResponse(400, "invalid session ID or JSON");
    }
}

std::string RestServer::handleSessionClose(const std::string& id_str, const std::string& body) {
    try {
        uint64_t sid = std::stoull(id_str);
        auto result = engine_.sessionManager().closeSession(sid);
        if (!result.ok()) return errorResponse(404, result.error().toString());
        return jsonResponse(200, R"({"closed":true})");
    } catch (...) {
        return errorResponse(400, "invalid session ID");
    }
}

std::string RestServer::handleSessionSummary(const std::string& id_str) {
    try {
        uint64_t sid = std::stoull(id_str);
        auto result = engine_.sessionManager().getSessionSummary(sid);
        if (!result.ok()) return errorResponse(404, result.error().toString());

        json resp;
        resp["session_id"] = result->session_id;
        resp["agent_id"] = result->agent_id;
        resp["user_id"] = result->user_id;
        resp["turn_count"] = result->turn_count;
        resp["current_intent"] = result->current_intent;
        resp["memory_count"] = result->memory_count;
        resp["fact_count"] = result->fact_count;
        resp["started_at"] = result->started_at;
        resp["last_turn_at"] = result->last_turn_at;
        resp["active"] = result->active;
        return jsonResponse(200, resp.dump());
    } catch (...) {
        return errorResponse(400, "invalid session ID");
    }
}

// ── Backup Handlers ─────────────────────────────────────────────────────────

std::string RestServer::handleBackupExport(const std::string& type) {
    if (type == "graph") {
        auto result = engine_.backupManager().exportGraph();
        if (!result.ok()) return errorResponse(500, result.error().toString());
        json resp;
        resp["type"] = "graph";
        resp["data"] = *result;
        return jsonResponse(200, resp.dump());
    }
    // Default: export memories
    auto result = engine_.backupManager().exportMemories();
    if (!result.ok()) return errorResponse(500, result.error().toString());
    json resp;
    resp["type"] = "memories";
    resp["data"] = *result;
    return jsonResponse(200, resp.dump());
}

std::string RestServer::handleBackupImport(const std::string& type, const std::string& body) {
    try {
        auto j = json::parse(body);
        auto data = j.value("data", "");

        if (type == "graph") {
            auto result = engine_.backupManager().importGraph(data);
            if (!result.ok()) return errorResponse(500, result.error().toString());
            return jsonResponse(200, json{{"type","graph"},{"imported",*result}}.dump());
        }
        // Default: import memories
        auto result = engine_.backupManager().importMemories(data);
        if (!result.ok()) return errorResponse(500, result.error().toString());
        return jsonResponse(200, json{{"type","memories"},{"imported",*result}}.dump());
    } catch (...) {
        return errorResponse(400, "invalid JSON");
    }
}


// ── WebUI Handlers ─────────────────────────────────────────────────────────

std::string RestServer::handleListMemories(const std::string& path) {
    MemoryStore::ListFilter filter;
    auto page_str = extractQueryParam(path, "page");
    auto pp_str = extractQueryParam(path, "per_page");
    if (!page_str.empty()) filter.page = std::stoi(page_str);
    if (!pp_str.empty()) filter.per_page = std::stoi(pp_str);
    
    // Adapt old filters to new fields if present, but prioritize new ones
    // agent_id filtering is handled at the AgentStore level (physical isolation),
    // not in ListFilter. The correct AgentStore is already selected by the caller.
    // We still parse agent_id from the query param for routing purposes.
    // auto agent_id_param = extractQueryParam(path, "agent_id");
    if (auto v = extractQueryParam(path, "user_id"); !v.empty()) {
        filter.user_id_filter = v;
    }
    if (auto v = extractQueryParam(path, "scope"); !v.empty()) {
        filter.scope_filter = v;
    }
    if (auto v = extractQueryParam(path, "memory_type"); !v.empty()) {
        filter.memory_type_filter = v;
    }
    
    filter.phase_filter = extractQueryParam(path, "phase");
    filter.query = extractQueryParam(path, "q");
    
    if (auto v = extractQueryParam(path, "include_tombstone"); v == "1" || v == "true") {
        filter.include_tombstone = true;
    }
    if (auto v = extractQueryParam(path, "layer"); !v.empty()) {
        filter.layer_filter = v;  // "Raw" | "Derived"
    }

    auto result = engine_.memoryStore().listMemories(filter);

    json resp;
    resp["total"] = result.total;
    resp["page"] = result.page;
    resp["per_page"] = result.per_page;
    resp["memories"] = json::array();
    for (const auto& rec : result.records) {
        json m;
        m["memory_id"] = std::to_string(rec.memory_id);
        m["content"] = rec.content;
        m["agent_id"] = rec.agent_id;
        m["user_id"] = rec.user_id;
        m["scope"] = scopeToString(rec.scope);
        m["memory_type"] = memoryTypeToString(rec.memory_type);
        m["tier"] = memoryTierToString(rec.tier);
        m["phase"] = phaseToString(rec.phase);
        m["confidence"] = confidenceToString(rec.confidence_level);
        m["importance"] = rec.importance;
        m["created_at"] = rec.created_at;
        m["last_accessed"] = rec.last_accessed;
        m["access_count"] = rec.access_count;
        m["version"] = rec.mem_version;
        m["has_embedding"] = (rec.flags & RecordFlags::HAS_EMBEDDING) != 0;
        m["layer"] = layerToString(rec.layer);
        if (!rec.user_metadata.empty()) {
            json meta = json::object();
            for (const auto& [k, v] : rec.user_metadata) meta[k] = v;
            m["metadata"] = std::move(meta);
        }
        resp["memories"].push_back(m);
    }
    return jsonResponse(200, resp.dump());
}

std::string RestServer::handleListEdges(const std::string& path) {
    int page = 1, per_page = 100;
    auto page_str = extractQueryParam(path, "page");
    auto pp_str = extractQueryParam(path, "per_page");
    if (!page_str.empty()) page = std::stoi(page_str);
    if (!pp_str.empty()) per_page = std::stoi(pp_str);

    // Fetch all edges, filter out those referencing dead memories, then paginate
    auto all = engine_.graphStore().listEdges(1, 100000);
    std::vector<GraphEdge> live_edges;
    live_edges.reserve(all.edges.size());
    for (const auto& e : all.edges) {
        auto from_rec = engine_.memoryStore().get(e.from_id);
        auto to_rec = engine_.memoryStore().get(e.to_id);
        if (from_rec.ok() && from_rec.value().isAlive() &&
            to_rec.ok() && to_rec.value().isAlive()) {
            live_edges.push_back(e);
        }
    }

    json resp;
    resp["total"] = live_edges.size();
    resp["page"] = page;
    resp["per_page"] = per_page;
    resp["edges"] = json::array();
    int offset = (page - 1) * per_page;
    for (int i = offset; i < static_cast<int>(live_edges.size()) && i < offset + per_page; ++i) {
        const auto& e = live_edges[static_cast<size_t>(i)];
        json edge;
        edge["from_id"] = std::to_string(e.from_id);
        edge["to_id"] = std::to_string(e.to_id);
        edge["type"] = edgeTypeToString(e.type);
        edge["weight"] = e.weight;
        edge["created_at"] = e.created_at;
        resp["edges"].push_back(edge);
    }
    return jsonResponse(200, resp.dump());
}

std::string RestServer::handleGraphNeighbors(const std::string& id_str,
                                              bool include_incoming) {
    try {
        uint64_t memory_id = std::stoull(id_str);
        auto edges = engine_.graphStore().getNeighborEdges(memory_id);
        if (include_incoming) {
            // Edges that point INTO this memory live only on the source's
            // adjacency list (e.g. DerivedFrom written from derived → raw).
            auto incoming = engine_.graphStore().getIncomingEdges(memory_id);
            edges.insert(edges.end(), incoming.begin(), incoming.end());
        }

        json resp = json::array();
        for (const auto& e : edges) {
            json edge;
            edge["from_id"] = std::to_string(e.from_id);
            edge["to_id"]   = std::to_string(e.to_id);
            edge["type"]    = edgeTypeToString(e.type);
            edge["weight"]  = e.weight;
            // Inline the neighbor's current phase + a short content preview so
            // the WebUI can show "what the other side actually says" without
            // an N+1 fetch per edge. Looking up the OTHER endpoint of the edge.
            uint64_t other = e.from_id == memory_id ? e.to_id : e.from_id;
            if (auto rec = engine_.memoryStore().get(other); rec.ok()) {
                edge["other_phase"]      = phaseToString(rec->phase);
                edge["other_confidence"] = confidenceToString(rec->confidence_level);
                edge["other_preview"]    = truncateUtf8(rec->content, 90);
            } else {
                // Edge points at a memory we can no longer load — usually
                // because it was vacuumed. Surface this so the user knows
                // why content is missing.
                edge["other_phase"]      = "Missing";
                edge["other_confidence"] = "";
                edge["other_preview"]    = "";
            }
            resp.push_back(edge);
        }
        return jsonResponse(200, resp.dump());
    } catch (...) {
        return errorResponse(400, "invalid memory ID");
    }
}

std::string RestServer::handleListSessions() {
    auto sessions = engine_.sessionManager().listSessions();
    json resp = json::array();
    for (const auto& s : sessions) {
        json sess;
        sess["session_id"] = std::to_string(s.session_id);
        sess["agent_id"] = s.agent_id;
        sess["user_id"] = s.user_id;
        sess["turn_count"] = s.turn_count;
        sess["current_intent"] = s.current_intent;
        sess["memory_count"] = s.memory_count;
        sess["fact_count"] = s.fact_count;
        sess["started_at"] = s.started_at;
        sess["last_turn_at"] = s.last_turn_at;
        sess["active"] = s.active;
        resp.push_back(sess);
    }
    return jsonResponse(200, resp.dump());
}

std::string RestServer::handleCoverageStats(const std::string& agent_id) {
    auto stats = engine_.metaCognition().getCoverage(agent_id);
    json resp;
    resp["total"] = stats.total;
    resp["active"] = stats.active;
    resp["stale"] = stats.stale;
    resp["conflicted"] = stats.conflicted;
    resp["last_updated"] = stats.last_updated;

    // Add scope/type/phase/confidence distribution
    json scope_dist = json::object(), type_dist = json::object(), phase_dist = json::object(), confidence_dist = json::object();
    engine_.memoryStore().scanAll([&](const MemoryRecord& rec) {
        if (!agent_id.empty() && rec.agent_id != agent_id) return;
        
        std::string sc = scopeToString(rec.scope);
        std::string tp = memoryTypeToString(rec.memory_type);
        std::string ph = phaseToString(rec.phase);
        std::string cf = confidenceToString(rec.confidence_level);
        
        scope_dist[sc] = (scope_dist.contains(sc) ? scope_dist[sc].get<int>() : 0) + 1;
        type_dist[tp] = (type_dist.contains(tp) ? type_dist[tp].get<int>() : 0) + 1;
        phase_dist[ph] = (phase_dist.contains(ph) ? phase_dist[ph].get<int>() : 0) + 1;
        confidence_dist[cf] = (confidence_dist.contains(cf) ? confidence_dist[cf].get<int>() : 0) + 1;
    });
    resp["scope_distribution"] = scope_dist;
    resp["memory_type_distribution"] = type_dist;
    resp["phase_distribution"] = phase_dist;
    resp["confidence_distribution"] = confidence_dist;
    return jsonResponse(200, resp.dump());
}

// ── Agent Management Handlers ───────────────────────────────────────────────

std::string RestServer::handleRegisterAgent(const std::string& body) {
    try {
        auto j = json::parse(body);
        auto agent_id = j.value("agent_id", "");
        if (agent_id.empty()) {
            return errorResponse(400, "agent_id is required");
        }
        auto result = engine_.registerAgent(agent_id);
        if (!result.ok()) {
            return errorResponse(500, result.error().toString());
        }
        return jsonResponse(201, R"({"status":"registered","agent_id":")" + agent_id + "\"}");
    } catch (...) {
        return errorResponse(400, "invalid JSON");
    }
}

std::string RestServer::handleListAgents() {
    auto agents = engine_.listAgents();
    json resp = json::array();
    for (const auto& a : agents) {
        json item;
        item["agent_id"] = a;
        resp.push_back(std::move(item));
    }
    return jsonResponse(200, resp.dump());
}

std::string RestServer::handleUnregisterAgent(const std::string& agent_id) {
    auto result = engine_.removeAgent(agent_id);
    if (!result.ok()) {
        return errorResponse(404, result.error().toString());
    }
    return jsonResponse(200, R"({"status":"unregistered","agent_id":")" + agent_id + "\"}");
}

// ── Health / Metrics ────────────────────────────────────────────────────────

std::string RestServer::handleShowVariables(const std::string& path) {
    auto pattern = extractQueryParam(path, "like");
    if (pattern.empty()) pattern = "%";

    auto vars = engine_.variableManager().list(pattern);
    json arr = json::array();
    for (const auto& v : vars) {
        json item;
        item["name"] = v.name;
        item["value"] = v.value;
        item["default"] = v.default_value;
        item["type"] = (v.type == VarType::STRING ? "string" :
                        v.type == VarType::INT    ? "int" :
                        v.type == VarType::FLOAT  ? "float" : "bool");
        item["mode"] = (v.mode == VarMode::DYNAMIC ? "DYNAMIC" : "READONLY");
        item["category"] = v.category;
        item["description"] = v.description;
        arr.push_back(std::move(item));
    }
    return jsonResponse(200, json{{"variables", arr}}.dump());
}

std::string RestServer::handleSetVariable(const std::string& name,
                                           const std::string& body) {
    try {
        auto j = json::parse(body);
        auto value = j.value("value", "");
        if (value.empty()) {
            return errorResponse(400, "missing 'value' field");
        }

        // Get old value before setting
        auto& vm = engine_.variableManager();
        auto old_value = vm.get(name);

        auto result = vm.set(name, value);
        if (!result.ok()) {
            return errorResponse(400, result.error().toString());
        }

        // Persist to config file
        vm.persistToFile(engine_.configPath());

        json resp;
        resp["ok"] = true;
        resp["name"] = name;
        resp["old_value"] = old_value;
        resp["new_value"] = value;
        return jsonResponse(200, resp.dump());
    } catch (const json::exception& e) {
        return errorResponse(400, "invalid JSON: " + std::string(e.what()));
    }
}

std::string RestServer::handleReloadConfig() {
    auto result = engine_.variableManager().reloadFromFile(engine_.configPath());
    if (!result.ok()) {
        return errorResponse(500, result.error().toString());
    }
    json resp;
    resp["ok"] = true;
    resp["changed"] = *result;
    return jsonResponse(200, resp.dump());
}

std::string RestServer::handleHealth() {
    return jsonResponse(200, json{{"status","ok"},{"version",AMIND_VERSION}}.dump());
}

std::string RestServer::handleMetrics() {
    json resp;
    resp["total_memories"] = engine_.memoryStore().totalMemories();
    resp["graph_edges"] = engine_.graphStore().edgeCount();
    resp["pool_connections"] = engine_.connectionPool().pooledCount();
    resp["pool_total_acquired"] = engine_.connectionPool().totalAcquired();
    resp["pool_total_reused"] = engine_.connectionPool().totalReused();
    resp["pool_circuit_open"] = engine_.connectionPool().isCircuitOpen();
    return jsonResponse(200, resp.dump());
}



// ── HTTP Helpers ─────────────────────────────────────────────────────────────

std::string RestServer::jsonResponse(int status, const std::string& body) {
    std::string status_text = (status == 200) ? "OK" : "Error";
    return "HTTP/1.1 " + std::to_string(status) + " " + status_text + "\r\n"
         + "Content-Type: application/json\r\n"
         + "Content-Length: " + std::to_string(body.size()) + "\r\n"
         + "Connection: close\r\n\r\n"
         + body;
}

std::string RestServer::errorResponse(int status, const std::string& message) {
    json j;
    j["error"] = message;
    return jsonResponse(status, j.dump());
}

std::string RestServer::urlDecode(const std::string& src) {
    std::string result;
    result.reserve(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        if (src[i] == '%' && i + 2 < src.size()) {
            int hi = 0, lo = 0;
            auto hexVal = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return -1;
            };
            hi = hexVal(src[i+1]);
            lo = hexVal(src[i+2]);
            if (hi >= 0 && lo >= 0) {
                result += static_cast<char>((hi << 4) | lo);
                i += 2;
                continue;
            }
        } else if (src[i] == '+') {
            result += ' ';
            continue;
        }
        result += src[i];
    }
    return result;
}

std::string RestServer::extractQueryParam(const std::string& path, const std::string& key) {
    auto qmark = path.find('?');
    if (qmark == std::string::npos) return "";
    auto query = path.substr(qmark + 1);
    auto target = key + "=";
    auto pos = query.find(target);
    if (pos == std::string::npos) return "";
    auto val_start = pos + target.size();
    auto amp = query.find('&', val_start);
    auto raw = (amp != std::string::npos) ? query.substr(val_start, amp - val_start)
                                          : query.substr(val_start);
    return urlDecode(raw);
}

// Gate/Forget JSON serializers removed in Phase 4 — MemoryEvent handles
// all observability JSON via memoryEventToJson in the unified handler block.

// forgetEntryToJson removed in Phase 4 (see comment above).



std::string RestServer::handleAdminForgetRun() {
    if (!engine_.featureGate().isForgetScoreEnabled()) {
        return errorResponse(409, "ForgetEngine disabled (set v2_forget_score_enabled=true)");
    }
    try {
        auto r = engine_.runForgetCycleOnce();
        json resp;
        resp["evaluated"] = r.evaluated;
        resp["logged"]    = r.logged;
        resp["actioned"]  = r.actioned;
        resp["shadow"]    = engine_.forgetEngine().isShadowMode();
        return jsonResponse(200, resp.dump());
    } catch (const std::exception& e) {
        return errorResponse(500, std::string("forget cycle failed: ") + e.what());
    }
}

std::string RestServer::handleAdminConsolidationRun() {
    if (!engine_.featureGate().isConsolidationEnabled()) {
        return errorResponse(409, "Consolidation disabled (set v2_consolidation_enabled=true)");
    }
    try {
        auto r = engine_.runConsolidationCycleOnce();
        json resp;
        resp["checked"]      = r.checked;
        resp["deduped"]      = r.deduped;
        resp["sampled_full"] = r.sampled_full;
        resp["shadow"]       = engine_.featureGate().isGlobalShadowMode();
        return jsonResponse(200, resp.dump());
    } catch (const std::exception& e) {
        return errorResponse(500, std::string("consolidation cycle failed: ") + e.what());
    }
}

// ── Unified observability (MemoryEventLog) handlers ───────────────────────
// See docs/arch/统一可观测层重构-MemoryEvent.md.

namespace {

json memoryEventToJson(const MemoryEvent& e) {
    json j;
    j["event_id"]        = std::to_string(e.event_id);
    j["parent_event_id"] = std::to_string(e.parent_event_id);
    j["trace_id"]        = std::to_string(e.trace_id);
    j["memory_id"]       = std::to_string(e.memory_id);
    j["timestamp_ms"]    = e.timestamp_ms;
    j["duration_ms"]     = e.duration_ms;
    j["agent_id"]        = e.agent_id; // Use new field
    j["kind"]            = kindToString(e.kind);
    j["status"]          = statusToString(e.status);
    j["summary"]         = e.summary;
    if (!e.attrs.empty()) {
        json a = json::object();
        for (const auto& [k, v] : e.attrs) a[k] = v;
        j["attrs"] = std::move(a);
    }
    return j;
}

}  // namespace

// Helper — populates filter from URL query params. Member function so it
// can call extractQueryParam(). Used by both /v1/events and /v1/events/stats.
static MemoryEventLog::Filter buildEventsFilterFromPath(
    const RestServer* self, const std::string& path, size_t default_limit) {
    MemoryEventLog::Filter f;
    f.limit = default_limit;
    auto p = [&](const char* k) { return RestServer::extractQueryParam(path, k); };
    if (auto s = p("limit"); !s.empty()) {
        try { f.limit = static_cast<size_t>(std::stoul(s)); } catch (...) {}
    }
    if (auto s = p("memory_id"); !s.empty()) {
        try { f.memory_id = std::stoull(s); } catch (...) {}
    }
    if (auto s = p("trace_id"); !s.empty()) {
        try { f.trace_id = std::stoull(s); } catch (...) {}
    }
    if (auto s = p("kind"); !s.empty()) {
        EventKind k = kindFromString(s);
        if (kindToString(k) == s) f.kind = k;
    }
    if (auto s = p("status"); !s.empty()) {
        EventStatus st = statusFromString(s);
        if (statusToString(st) == s) f.status = st;
    }
    // Adapt old namespace filter to agent_id if present
    if (auto s = p("agent_id"); !s.empty()) {
        f.agent_id_filter = s;
    } else if (auto ns = p("namespace"); !ns.empty()) {
        f.agent_id_filter = ns; // Fallback for backward compatibility
    }
    if (auto s = p("since_ms"); !s.empty()) {
        try { f.since_ms = std::stoull(s); } catch (...) {}
    }
    if (auto s = p("until_ms"); !s.empty()) {
        try { f.until_ms = std::stoull(s); } catch (...) {}
    }
    (void)self;
    return f;
}

std::string RestServer::handleEventsQuery(const std::string& path) {
    auto& el = engine_.eventsLog();
    auto f = buildEventsFilterFromPath(this, path, /*default_limit=*/200);
    auto rows = el.query(f);

    json resp;
    json arr = json::array();
    for (const auto& e : rows) arr.push_back(memoryEventToJson(e));
    resp["entries"] = std::move(arr);
    resp["memory_size"] = el.memorySize();
    return jsonResponse(200, resp.dump());
}

std::string RestServer::handleMemoryTrace(const std::string& id_str) {
    uint64_t mid = 0;
    try { mid = std::stoull(id_str); } catch (...) {
        return errorResponse(400, "invalid memory_id");
    }
    auto& el = engine_.eventsLog();
    auto rows = el.memoryHistory(mid);  // newest first; only events touching this memory

    // Group events by trace_id so the user sees "this memory was touched by N
    // separate operations". Within each group we show ONLY events for this
    // memory_id — clicking the trace_id link drills into the full trace
    // (including siblings) on the Events page.
    std::vector<std::string> trace_order;
    std::unordered_map<std::string, json> by_trace;
    std::unordered_map<std::string, std::vector<MemoryEvent>> bucket;
    for (const auto& e : rows) {
        std::string tid = std::to_string(e.trace_id);
        if (!by_trace.count(tid)) {
            json group;
            group["trace_id"] = tid;
            by_trace[tid] = std::move(group);
            trace_order.push_back(tid);
        }
        bucket[tid].push_back(e);
    }
    for (auto& tid : trace_order) {
        auto& evs = bucket[tid];
        std::sort(evs.begin(), evs.end(),
                  [](const MemoryEvent& a, const MemoryEvent& b) {
                      return a.timestamp_ms < b.timestamp_ms;
                  });
        json arr = json::array();
        for (const auto& e : evs) arr.push_back(memoryEventToJson(e));
        by_trace[tid]["events"]      = std::move(arr);
        by_trace[tid]["event_count"] = evs.size();
        by_trace[tid]["start_ms"]    = evs.empty() ? 0 : evs.front().timestamp_ms;
        by_trace[tid]["end_ms"]      = evs.empty() ? 0 : evs.back().timestamp_ms;
    }

    json traces = json::array();
    for (const auto& tid : trace_order) traces.push_back(std::move(by_trace[tid]));

    json resp;
    resp["memory_id"]     = id_str;
    resp["total_events"]  = rows.size();
    resp["traces"]        = std::move(traces);
    return jsonResponse(200, resp.dump());
}

std::string RestServer::handleTraceById(const std::string& trace_id_str) {
    uint64_t tid = 0;
    try { tid = std::stoull(trace_id_str); } catch (...) {
        return errorResponse(400, "invalid trace_id");
    }
    auto& el = engine_.eventsLog();
    auto events = el.trace(tid);  // chronological ASC

    json resp;
    resp["trace_id"] = trace_id_str;
    resp["event_count"] = events.size();
    resp["start_ms"] = events.empty() ? 0 : events.front().timestamp_ms;
    resp["end_ms"]   = events.empty() ? 0 : events.back().timestamp_ms;
    json arr = json::array();
    for (const auto& e : events) arr.push_back(memoryEventToJson(e));
    resp["events"] = std::move(arr);
    return jsonResponse(200, resp.dump());
}

std::string RestServer::handleEventsStats(const std::string& path) {
    auto& el = engine_.eventsLog();
    auto f = buildEventsFilterFromPath(this, path, /*default_limit=*/2000);
    auto s = el.stats(f);

    json resp;
    json bk = json::object();
    for (const auto& [name, count] : s.by_kind)   bk[name] = count;
    json bs = json::object();
    for (const auto& [name, count] : s.by_status) bs[name] = count;
    resp["by_kind"]   = std::move(bk);
    resp["by_status"] = std::move(bs);
    resp["total"]     = s.total;
    resp["memory_size"] = el.memorySize();
    return jsonResponse(200, resp.dump());
}

std::string RestServer::handleAdminResurrect(const std::string& event_id_str,
                                              const std::string& body) {
    uint64_t event_id = 0;
    try { event_id = std::stoull(event_id_str); } catch (...) {
        return errorResponse(400, "invalid event_id");
    }
    // Find the source Gate event by scanning the ring (max 2000 entries; if
    // it's been evicted, we can't resurrect — would need to read events.log
    // JSONL from disk, deferred).
    auto& el = engine_.eventsLog();
    MemoryEventLog::Filter f;
    f.limit = el.memorySize();  // entire ring
    auto all = el.query(f);
    const MemoryEvent* src = nullptr;
    for (const auto& e : all) {
        if (e.event_id == event_id) { src = &e; break; }
    }
    if (!src) {
        return errorResponse(404, "event_id not found in recent ring "
                                  "(may have been evicted; check events.log JSONL)");
    }
    if (src->kind != EventKind::Gate) {
        return errorResponse(400, "resurrect only applies to Gate events");
    }
    if (src->status != EventStatus::Rejected && src->status != EventStatus::Deferred) {
        return errorResponse(400, "source event was not Rejected/Deferred");
    }

    // Pull the original content. Gate event 'summary' is truncated; full
    // content is not stored in the event itself (by design — we don't want
    // to duplicate large payloads). For now use summary as the fact body;
    // if the user needs to preserve the full original wording, they can
    // pass it explicitly in the request body.
    std::string content = src->summary;
    std::string strategy = "coexist";
    try {
        if (!body.empty()) {
            auto j = json::parse(body);
            content = j.value("content", content);
            strategy = j.value("strategy", strategy);
        }
    } catch (...) {
        return errorResponse(400, "invalid JSON body");
    }

    // Store the fact through the same fast-path as a normal POST /v1/memories.
    // For resurrect, we use the agent_id from the original event's attrs if available, 
    // otherwise default to "default_agent". We also need to determine scope/type.
    auto it_agent = src->attrs.find("agent_id");
    std::string agent_id = (it_agent != src->attrs.end()) ? it_agent->second : "default_agent";
    auto it_user = src->attrs.find("user_id");
    std::string user_id = (it_user != src->attrs.end()) ? it_user->second : "anonymous";
    
    // Determine scope and type from attrs or defaults
    auto it_scope = src->attrs.find("scope");
    MemoryScope scope = (it_scope != src->attrs.end()) ? scopeFromString(it_scope->second) : MemoryScope::Private;
    auto it_type = src->attrs.find("memory_type");
    MemoryType memory_type = (it_type != src->attrs.end()) ? memoryTypeFromString(it_type->second) : MemoryType::Ephemeral;

    auto store_result = engine_.capturePipeline().capture(
        content, agent_id, user_id, scope, memory_type, {});
    if (!store_result.ok()) {
        return errorResponse(500, std::string("resurrect store failed: ") +
                                  store_result.error().toString());
    }
    auto ids = store_result.value();
    uint64_t new_id = ids.empty() ? 0 : ids.front();

    // Emit Resurrect event so the trace shows the recovery action.
    MemoryEvent rev;
    rev.memory_id  = new_id;
    rev.trace_id   = new_id;
    rev.agent_id   = agent_id; // Use new field if available in MemoryEvent, otherwise keep namespace_ for backward compat in event log
    rev.kind       = EventKind::Resurrect;
    rev.status     = EventStatus::Ok;
    rev.summary    = content.substr(0, 80);
    rev.attrs["source_event_id"] = std::to_string(event_id);
    rev.attrs["strategy"]        = strategy;
    el.append(std::move(rev));

    json resp;
    resp["memory_id"]       = std::to_string(new_id);
    resp["source_event_id"] = std::to_string(event_id);
    resp["strategy"]        = strategy;
    return jsonResponse(200, resp.dump());
}

std::string RestServer::checkAuth(const std::string& headers, const std::string& path) {
    if (api_token_.empty()) return "";  // auth disabled

    // /v1/health is exempt (health probes don't carry tokens)
    auto qmark = path.find('?');
    std::string clean = (qmark != std::string::npos) ? path.substr(0, qmark) : path;
    if (clean == "/v1/health") return "";

    // Look for "Authorization: Bearer <token>" (case-insensitive header name)
    std::string headers_lower = headers;
    for (auto& c : headers_lower) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));

    auto pos = headers_lower.find("authorization:");
    if (pos == std::string::npos) {
        return errorResponse(401, "unauthorized");
    }

    auto val_start = pos + 14;  // len("authorization:")
    while (val_start < headers_lower.size() && headers_lower[val_start] == ' ') val_start++;
    auto line_end = headers.find("\r\n", val_start);
    std::string auth_value = headers.substr(val_start, line_end - val_start);

    // Expect "Bearer <token>"
    if (auth_value.size() <= 7 || auth_value.substr(0, 7) != "Bearer ") {
        return errorResponse(401, "unauthorized");
    }
    std::string token = auth_value.substr(7);

    // Check master token first (constant-time comparison)
    bool master_ok = false;
    if (!api_token_.empty() && token.size() == api_token_.size()) {
        volatile unsigned char result = 0;
        for (size_t i = 0; i < token.size(); ++i) {
            result |= static_cast<unsigned char>(token[i]) ^ static_cast<unsigned char>(api_token_[i]);
        }
        master_ok = (result == 0);
    }
    if (master_ok) return "";

    // Check AuthManager keys (SHA-256 hash comparison)
    if (engine_.authManager().validateKey(token)) return "";

    return errorResponse(401, "unauthorized");
}

// ── Auth Key Management ──

std::string RestServer::handleCreateKey(const std::string& body) {
    try {
        auto j = nlohmann::json::parse(body);
        std::string label = j.value("label", "");
        auto result = engine_.authManager().createKey(label);
        if (!result.ok()) return errorResponse(500, result.error().message);
        nlohmann::json resp;
        resp["key"] = *result;
        resp["message"] = "Store this key securely — it will not be shown again.";
        return jsonResponse(201, resp.dump());
    } catch (const std::exception& e) {
        return errorResponse(400, std::string("invalid JSON: ") + e.what());
    }
}

std::string RestServer::handleListKeys() {
    auto keys = engine_.authManager().listKeys();
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& k : keys) {
        nlohmann::json item;
        item["id"] = k.id;
        item["label"] = k.label;
        item["key_prefix"] = k.key_prefix;
        item["created_at"] = k.created_at;
        item["last_used_at"] = k.last_used_at;
        arr.push_back(std::move(item));
    }
    nlohmann::json resp;
    resp["keys"] = std::move(arr);
    return jsonResponse(200, resp.dump());
}

std::string RestServer::handleRevokeKey(const std::string& key_id) {
    auto result = engine_.authManager().revokeKey(key_id);
    if (!result.ok()) return errorResponse(400, result.error().message);
    return jsonResponse(200, R"({"status":"revoked"})");
}

}  // namespace amind
