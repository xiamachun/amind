#include "rest_server.h"
#include "variable_manager.h"
#include "config/v2_config.h"
#include "coordinator/remove_coordinator.h"
#include "lineage/lineage_index.h"
#include "forget/forget_engine.h"
#include "retrieval/staleness_filter.h"
#include "gate_log/gate_log.h"
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
    if (method == "GET" && clean_path == "/v1/pipeline/stats") return handlePipelineStats();
    if (method == "GET" && clean_path == "/v1/pipeline/reconcile-log") return handleReconcileLog(path);

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

    // Namespace bulk delete (must be before dynamic /v1/memories/{id} handler)
    if (method == "DELETE" && clean_path.find("/v1/memories/namespace/") == 0) {
        auto ns = urlDecode(clean_path.substr(23));
        return handleDeleteNamespace(ns);
    }

    // WebUI: list (must be before dynamic path handlers)
    if (method == "GET" && clean_path == "/v1/memories/list") return handleListMemories(path);
    if (method == "GET" && clean_path == "/v1/graph/edges") return handleListEdges(path);
    if (method == "GET" && clean_path == "/v1/sessions/list") return handleListSessions();
    if (method == "GET" && clean_path == "/v1/metacognition/coverage") {
        return handleCoverageStats(extractQueryParam(path, "namespace"));
    }

    // Gate log audit
    if (method == "GET" && clean_path == "/v1/gate/log") return handleGateLogList(path);
    if (method == "GET" && clean_path == "/v1/gate/log/stats") return handleGateLogStats(path);
    if (clean_path.find("/v1/gate/log/") == 0) {
        auto rest = clean_path.substr(std::string("/v1/gate/log/").size());
        auto slash = rest.find('/');
        std::string id_str = (slash != std::string::npos) ? rest.substr(0, slash) : rest;
        std::string sub = (slash != std::string::npos) ? rest.substr(slash) : "";
        if (method == "POST" && sub == "/resurrect") {
            return handleGateLogResurrect(id_str, body);
        }
    }

    // Forget log audit (read-only)
    if (method == "GET" && clean_path == "/v1/forget/log") return handleForgetLog(path);
    if (method == "GET" && clean_path == "/v1/recall/stale-log") return handleRecallStaleLog(path);

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



    // WebUI: graph neighbors
    if (method == "GET" && clean_path.find("/v1/graph/neighbors/") == 0) {
        auto id_str = clean_path.substr(20);
        return handleGraphNeighbors(id_str);
    }

    return errorResponse(404, "not found");
}

// ── Route Handlers ──────────────────────────────────────────────────────────

std::string RestServer::handleStore(const std::string& body) {
    try {
        auto j = json::parse(body);
        auto content = j.value("content", "");
        auto ns = j.value("namespace", "default");
        auto owner_str = j.value("owner", "session");

        MemoryOwner owner = MemoryOwner::Session;
        if (owner_str == "user") owner = MemoryOwner::User;
        else if (owner_str == "project") owner = MemoryOwner::Project;
        else if (owner_str == "agent") owner = MemoryOwner::Agent;

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
            content, ns, owner, std::move(user_metadata));
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
        auto ns = j.value("namespace", "default");
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
        auto result = engine_.retrievalPipeline().recall(query, ns, fetch_k);
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
            item["owner"] = ownerToString(sm.record.owner);
            item["phase"] = phaseToString(sm.record.phase);
            item["confidence"] = confidenceToString(sm.record.confidence);
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
        resp["owner"] = ownerToString(result->owner);
        resp["phase"] = phaseToString(result->phase);
        resp["confidence"] = confidenceToString(result->confidence);
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

std::string RestServer::handleDeleteNamespace(const std::string& ns) {
    static const std::vector<std::string> protected_ns = {
        "amazon1", "amazon2",
        "pyclaw:amazon1:pyclaw", "pyclaw:amazon2:pyclaw",
    };
    for (const auto& p : protected_ns) {
        if (ns == p) {
            return errorResponse(403, "namespace '" + ns + "' is protected and cannot be deleted");
        }
    }

    uint64_t ns_hash = MemoryRecord::hashNamespace(ns);
    auto deleted_ids = engine_.memoryStore().removeByNamespace(ns_hash);

    for (uint64_t id : deleted_ids) {
        engine_.graphStore().removeNode(id);
    }

    json resp;
    resp["deleted"] = deleted_ids.size();
    resp["namespace"] = ns;
    return jsonResponse(200, resp.dump());
}

std::string RestServer::handleIntercept(const std::string& body) {
    try {
        auto j = json::parse(body);
        auto ns = j.value("namespace", "default");
        std::vector<std::pair<std::string, std::string>> messages;
        if (j.contains("messages") && j["messages"].is_array()) {
            for (const auto& m : j["messages"]) {
                messages.emplace_back(m.value("role", ""), m.value("content", ""));
            }
        }
        auto result = engine_.capturePipeline().interceptCapture(messages, ns);
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
        auto ns = j.value("namespace", "default");
        auto result = engine_.sessionManager().startSession(ns);
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
        resp["namespace"] = result->namespace_;
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
    filter.owner_filter = extractQueryParam(path, "owner");
    filter.phase_filter = extractQueryParam(path, "phase");
    filter.query = extractQueryParam(path, "q");
    filter.namespace_filter = extractQueryParam(path, "namespace");
    filter.user_id_filter = extractQueryParam(path, "user_id");

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
        m["owner"] = ownerToString(rec.owner);
        m["phase"] = phaseToString(rec.phase);
        m["confidence"] = confidenceToString(rec.confidence);
        m["importance"] = rec.importance;
        m["created_at"] = rec.created_at;
        m["last_accessed"] = rec.last_accessed;
        m["access_count"] = rec.access_count;
        m["version"] = rec.mem_version;
        m["has_embedding"] = (rec.flags & RecordFlags::HAS_EMBEDDING) != 0;
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

std::string RestServer::handleGraphNeighbors(const std::string& id_str) {
    try {
        uint64_t memory_id = std::stoull(id_str);
        auto edges = engine_.graphStore().getNeighborEdges(memory_id);

        json resp = json::array();
        for (const auto& e : edges) {
            json edge;
            edge["from_id"] = std::to_string(e.from_id);
            edge["to_id"] = std::to_string(e.to_id);
            edge["type"] = edgeTypeToString(e.type);
            edge["weight"] = e.weight;
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
        sess["namespace"] = s.namespace_;
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

std::string RestServer::handleCoverageStats(const std::string& query) {
    auto stats = engine_.metaCognition().getCoverage(query);
    json resp;
    resp["total"] = stats.total;
    resp["active"] = stats.active;
    resp["stale"] = stats.stale;
    resp["conflicted"] = stats.conflicted;
    resp["last_updated"] = stats.last_updated;

    // Add owner/phase/confidence distribution
    json owner_dist = json::object(), phase_dist = json::object(), confidence_dist = json::object();
    engine_.memoryStore().scanAll([&](const MemoryRecord& rec) {
        std::string o = ownerToString(rec.owner);
        std::string ph = phaseToString(rec.phase);
        std::string cf = confidenceToString(rec.confidence);
        owner_dist[o] = (owner_dist.contains(o) ? owner_dist[o].get<int>() : 0) + 1;
        phase_dist[ph] = (phase_dist.contains(ph) ? phase_dist[ph].get<int>() : 0) + 1;
        confidence_dist[cf] = (confidence_dist.contains(cf) ? confidence_dist[cf].get<int>() : 0) + 1;
    });
    resp["owner_distribution"] = owner_dist;
    resp["phase_distribution"] = phase_dist;
    resp["confidence_distribution"] = confidence_dist;
    return jsonResponse(200, resp.dump());
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

std::string RestServer::handlePipelineStats() {
    json resp;
    resp["queue_depth"] = engine_.taskQueue().size();
    resp["queue_capacity"] = 10000;  // from TaskQueue constructor default
    resp["executor_threads"] = engine_.taskExecutor().numThreads();
    resp["tasks_completed"] = engine_.taskExecutor().tasksCompleted();
    resp["tasks_failed"] = engine_.taskExecutor().tasksFailed();

    // Reconciler stats (if available)
    json rec_json;
    if (auto* rec = engine_.reconciler()) {
        auto s = rec->stats();
        rec_json["total_calls"] = s.total_calls;
        rec_json["llm_invocations"] = s.llm_invocations;
        rec_json["llm_failures"] = s.llm_failures;
        rec_json["op_add"] = s.op_add;
        rec_json["op_replace"] = s.op_replace;
        rec_json["op_retract"] = s.op_retract;
        rec_json["op_reinforce"] = s.op_reinforce;
        rec_json["op_noop"] = s.op_noop;
    }
    resp["reconciler"] = rec_json;

    return jsonResponse(200, resp.dump());
}

std::string RestServer::handleReconcileLog(const std::string& path) {
    int limit = 100;
    auto limit_str = extractQueryParam(path, "limit");
    if (!limit_str.empty()) {
        try { limit = std::stoi(limit_str); } catch (...) {}
    }

    json resp;
    if (auto* rec = engine_.reconciler()) {
        auto entries = rec->recentLog(static_cast<size_t>(limit));
        json arr = json::array();
        for (const auto& e : entries) {
            json entry;
            entry["timestamp_ms"] = e.timestamp_ms;
            entry["candidate"] = e.candidate;
            entry["op"] = reconcileOpToString(e.op);
            entry["target_id"] = std::to_string(e.target_id);
            entry["latency_ms"] = e.latency_ms;
            entry["from_fallback"] = e.from_fallback;
            arr.push_back(std::move(entry));
        }
        resp["entries"] = std::move(arr);
    } else {
        resp["entries"] = json::array();
    }
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

// ── Gate Log handlers ───────────────────────────────────────────────────────

namespace {
const char* gateDecisionStr(GateDecision d) {
    switch (d) {
        case GateDecision::Accepted: return "Accepted";
        case GateDecision::Rejected: return "Rejected";
        case GateDecision::Deferred: return "Deferred";
    }
    return "Accepted";
}

const char* memLayerStr(MemoryLayer l) {
    return (l == MemoryLayer::Derived) ? "Derived" : "Raw";
}

json gateEntryToJson(const GateLogEntry& e) {
    json j;
    j["entry_id"] = std::to_string(e.entry_id);
    j["timestamp_ms"] = e.timestamp_ms;
    j["namespace"] = e.namespace_;
    j["content"] = e.content;
    j["decision"] = gateDecisionStr(e.decision);
    j["reason"] = e.reason;
    j["marginal_value"] = e.marginal_value;
    j["conflict_with_id"] = std::to_string(e.conflict_with_id);
    j["owner"] = ownerToString(e.owner);
    j["layer"] = memLayerStr(e.layer);
    if (!e.user_metadata.empty()) {
        json meta = json::object();
        for (const auto& [k, v] : e.user_metadata) meta[k] = v;
        j["user_metadata"] = std::move(meta);
    }
    if (e.memory_id != 0) j["memory_id"] = std::to_string(e.memory_id);
    j["resurrected_to"] = std::to_string(e.resurrected_to);
    j["resurrected_at_ms"] = e.resurrected_at_ms;
    if (!e.resurrect_strategy.empty()) j["resurrect_strategy"] = e.resurrect_strategy;
    if (!e.reconcile_op.empty()) {
        j["reconcile_op"] = e.reconcile_op;
        j["reconcile_target_id"] = std::to_string(e.reconcile_target_id);
        if (!e.reconcile_rationale.empty()) j["reconcile_rationale"] = e.reconcile_rationale;
    }
    return j;
}
}

std::string RestServer::handleGateLogList(const std::string& path) {
    GateLog::Filter filter;
    auto limit_str = extractQueryParam(path, "limit");
    if (!limit_str.empty()) {
        try { filter.limit = static_cast<size_t>(std::stoul(limit_str)); } catch (...) {}
    }
    if (filter.limit == 0) filter.limit = 100;

    auto decision_str = extractQueryParam(path, "decision");
    if (!decision_str.empty()) {
        if (decision_str == "Accepted") filter.decision = GateDecision::Accepted;
        else if (decision_str == "Rejected") filter.decision = GateDecision::Rejected;
        else if (decision_str == "Deferred") filter.decision = GateDecision::Deferred;
    }
    filter.namespace_filter = extractQueryParam(path, "namespace");
    auto since_str = extractQueryParam(path, "since_ms");
    if (!since_str.empty()) {
        try { filter.since_ms = std::stoll(since_str); } catch (...) {}
    }
    auto only_un = extractQueryParam(path, "only_unresurrected");
    filter.only_unresurrected = (only_un == "1" || only_un == "true");

    auto entries = engine_.gateLog().query(filter);
    auto stats = engine_.gateLog().stats(filter.since_ms, filter.namespace_filter);

    json resp;
    json arr = json::array();
    for (const auto& e : entries) arr.push_back(gateEntryToJson(e));
    resp["entries"] = std::move(arr);
    json st;
    st["accepted"] = stats.accepted;
    st["rejected"] = stats.rejected;
    st["deferred"] = stats.deferred;
    st["resurrected"] = stats.resurrected;
    st["total"] = stats.total();
    resp["stats"] = std::move(st);
    resp["memory_size"] = engine_.gateLog().memorySize();
    return jsonResponse(200, resp.dump());
}

std::string RestServer::handleGateLogStats(const std::string& path) {
    auto ns = extractQueryParam(path, "namespace");
    int64_t since_ms = 0;
    auto since_str = extractQueryParam(path, "since_ms");
    if (!since_str.empty()) {
        try { since_ms = std::stoll(since_str); } catch (...) {}
    }
    auto stats = engine_.gateLog().stats(since_ms, ns);
    json resp;
    resp["accepted"] = stats.accepted;
    resp["rejected"] = stats.rejected;
    resp["deferred"] = stats.deferred;
    resp["resurrected"] = stats.resurrected;
    resp["total"] = stats.total();
    return jsonResponse(200, resp.dump());
}

std::string RestServer::handleGateLogResurrect(const std::string& id_str,
                                                const std::string& body) {
    uint64_t entry_id = 0;
    try { entry_id = std::stoull(id_str); } catch (...) {
        return errorResponse(400, "invalid entry_id");
    }

    std::string strategy = "coexist";
    try {
        if (!body.empty()) {
            auto j = json::parse(body);
            strategy = j.value("strategy", "coexist");
        }
    } catch (...) {
        return errorResponse(400, "invalid JSON body");
    }
    if (strategy != "coexist" && strategy != "replace_conflict"
        && strategy != "update_existing") {
        return errorResponse(400, "strategy must be one of: coexist|replace_conflict|update_existing");
    }

    auto entry_opt = engine_.gateLog().findById(entry_id);
    if (!entry_opt) {
        return errorResponse(404, "gate log entry not found (or aged out of memory ring)");
    }
    if (entry_opt->resurrected_to != 0) {
        return errorResponse(409, "already resurrected to memory #"
                                  + std::to_string(entry_opt->resurrected_to));
    }

    json resp;
    resp["strategy"] = strategy;

    if (strategy == "update_existing") {
        // No new memory created; we just record the resurrect (the user
        // confirmed the existing memory is still relevant). Importance bump
        // could be added once MemoryStore exposes a public set-importance API.
        if (entry_opt->conflict_with_id == 0) {
            return errorResponse(400, "update_existing requires a conflict_with_id");
        }
        auto rec_r = engine_.memoryStore().get(entry_opt->conflict_with_id);
        if (!rec_r.ok()) return errorResponse(404, "conflict_with target not found");
        engine_.gateLog().recordResurrect(entry_id, entry_opt->conflict_with_id, strategy);
        resp["memory_id"] = std::to_string(entry_opt->conflict_with_id);
        resp["note"] = "kept existing memory; resurrect logged for audit";
        return jsonResponse(200, resp.dump());
    }

    // Build fresh MemoryRecord from the log entry, bypassing WriteGate.
    MemoryRecord rec;
    rec.namespace_hash = MemoryRecord::hashNamespace(entry_opt->namespace_);
    rec.content = entry_opt->content;
    rec.embedding = entry_opt->embedding;
    rec.owner = entry_opt->owner;
    rec.layer = entry_opt->layer;
    rec.confidence = Confidence::Verified;   // user explicitly resurrected → verified
    rec.user_metadata = entry_opt->user_metadata;
    rec.user_metadata["resurrected_from_gate_log"] = std::to_string(entry_id);

    auto store_r = engine_.memoryStore().fastStore(std::move(rec));
    if (!store_r.ok()) return errorResponse(500, "fastStore failed: " + store_r.error().toString());
    uint64_t new_id = *store_r;

    if (strategy == "replace_conflict" && entry_opt->conflict_with_id != 0) {
        // Tombstone the old conflict target (soft delete).
        engine_.memoryStore().remove(entry_opt->conflict_with_id);
    }

    engine_.gateLog().recordResurrect(entry_id, new_id, strategy);

    resp["memory_id"] = std::to_string(new_id);
    if (strategy == "replace_conflict" && entry_opt->conflict_with_id != 0) {
        resp["replaced_id"] = std::to_string(entry_opt->conflict_with_id);
    }
    return jsonResponse(200, resp.dump());
}

namespace {
json forgetEntryToJson(const ForgetLogEntry& e) {
    json j;
    j["timestamp_ms"] = e.timestamp_ms;
    j["memory_id"]    = std::to_string(e.memory_id);
    j["decision"]     = decisionToString(e.decision);
    j["reason"]       = e.reason;
    j["before_state"] = phaseToString(e.before_state);
    j["after_state"]  = phaseToString(e.after_state);
    if (!e.lineage_affected.empty()) {
        json arr = json::array();
        for (auto id : e.lineage_affected) arr.push_back(std::to_string(id));
        j["lineage_affected"] = std::move(arr);
    }
    if (!e.gc_worker_id.empty()) j["gc_worker_id"] = e.gc_worker_id;
    if (!e.namespace_.empty())   j["namespace"] = e.namespace_;
    if (!e.content_preview.empty()) j["content"] = e.content_preview;
    return j;
}
}  // namespace

std::string RestServer::handleForgetLog(const std::string& path) {
    size_t limit = 200;
    if (auto s = extractQueryParam(path, "limit"); !s.empty()) {
        try { limit = static_cast<size_t>(std::stoul(s)); } catch (...) {}
    }
    if (limit == 0) limit = 200;

    auto decision_filter = extractQueryParam(path, "decision");

    auto entries = engine_.forgetEngine().recentEntries(limit * 4);  // headroom for filter

    // Tally stats from the (filtered) window
    size_t decay = 0, archive = 0, tombstone = 0, vacuum = 0, other = 0;
    for (const auto& e : entries) {
        switch (e.decision) {
            case ForgetLogEntry::Decision::Decay:     ++decay; break;
            case ForgetLogEntry::Decision::Archive:   ++archive; break;
            case ForgetLogEntry::Decision::Tombstone: ++tombstone; break;
            case ForgetLogEntry::Decision::Vacuum:    ++vacuum; break;
            default: ++other; break;
        }
    }

    // Most recent first
    std::reverse(entries.begin(), entries.end());

    json arr = json::array();
    for (const auto& e : entries) {
        if (!decision_filter.empty() &&
            decisionToString(e.decision) != decision_filter) continue;
        if (arr.size() >= limit) break;
        arr.push_back(forgetEntryToJson(e));
    }

    json resp;
    resp["entries"] = std::move(arr);

    json st;
    st["decay"]     = decay;
    st["archive"]   = archive;
    st["tombstone"] = tombstone;
    st["vacuum"]    = vacuum;
    st["other"]     = other;
    st["total"]     = decay + archive + tombstone + vacuum + other;
    resp["stats"]   = std::move(st);

    auto cfg = engine_.forgetEngine().config();
    json fcfg;
    fcfg["enabled"]            = engine_.featureGate().isForgetScoreEnabled();
    fcfg["shadow_mode"]        = engine_.forgetEngine().isShadowMode();
    fcfg["decay_threshold"]    = cfg.decay_threshold;
    fcfg["archive_threshold"]  = cfg.archive_threshold;
    fcfg["tombstone_threshold"] = cfg.tombstone_threshold;
    fcfg["gc_interval_seconds"] = cfg.gc_interval_seconds;
    fcfg["sample_ratio"]       = cfg.sample_ratio;
    resp["config"] = std::move(fcfg);

    resp["memory_size"] = engine_.forgetEngine().logSize();
    return jsonResponse(200, resp.dump());
}

std::string RestServer::handleRecallStaleLog(const std::string& path) {
    auto* log = engine_.staleLog();
    json resp;
    resp["enabled"] = engine_.featureGate().isAggregateStalenessFilterEnabled();
    if (!log) {
        resp["entries"] = json::array();
        resp["memory_size"] = 0;
        resp["stats"] = {{"total", 0}};
        return jsonResponse(200, resp.dump());
    }

    size_t limit = 200;
    if (auto s = extractQueryParam(path, "limit"); !s.empty()) {
        try { limit = static_cast<size_t>(std::stoul(s)); } catch (...) {}
    }
    auto ns_filter = extractQueryParam(path, "namespace");

    auto entries = log->recentEntries();
    std::reverse(entries.begin(), entries.end());  // newest first

    json arr = json::array();
    size_t total = 0;
    for (const auto& e : entries) {
        if (!ns_filter.empty() && e.namespace_ != ns_filter) continue;
        ++total;
        if (arr.size() >= limit) continue;
        json j;
        j["timestamp_ms"] = e.timestamp_ms;
        j["query"] = e.query;
        j["namespace"] = e.namespace_;
        j["aggregate_id"] = std::to_string(e.aggregate_id);
        j["aggregate_preview"] = e.aggregate_preview;
        j["aggregate_created_at"] = e.aggregate_created_at;
        j["witness_in_aggregate"] = e.witness_ids_in_aggregate;
        j["witness_in_newer"] = e.witness_ids_in_newer_facts;
        json newer = json::array();
        for (auto id : e.newer_fact_ids) newer.push_back(std::to_string(id));
        j["newer_fact_ids"] = std::move(newer);
        j["action"] = (e.action == StaleFilterEvent::Action::Filter) ? "Filter" : "Downweight";
        j["pre_score"] = e.pre_score;
        j["post_score"] = e.post_score;
        arr.push_back(std::move(j));
    }

    resp["entries"] = std::move(arr);
    resp["memory_size"] = log->memorySize();
    resp["stats"] = {{"total", total}};
    return jsonResponse(200, resp.dump());
}

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
