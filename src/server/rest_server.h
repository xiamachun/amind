#pragma once

#include "engine.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace amind {

/// Minimal REST server using POSIX sockets.
/// Handles HTTP/1.1 requests and routes to engine methods via a thread pool.
class RestServer {
public:
    RestServer(Engine& engine, const std::string& host, int port,
               int max_connections = 128, int request_timeout_ms = 30000);
    ~RestServer();

    /// Start listening (blocking).
    Result<void, Error> start();

    /// Signal to stop.
    void stop();

    [[nodiscard]] bool running() const;

private:
    void handleClient(int client_fd);
    std::string routeRequest(const std::string& method, const std::string& path,
                             const std::string& body);

    // Route handlers
    std::string handleStore(const std::string& body);
    std::string handleRecall(const std::string& body);
    std::string handleGetMemory(const std::string& id_str);
    std::string handleGetHistory(const std::string& id_str);
    std::string handleFeedback(const std::string& id_str, const std::string& body);
    std::string handleDeleteMemory(const std::string& id_str);
    std::string handleArchiveMemory(const std::string& id_str);
    std::string handleDeleteAgentMemories(const std::string& agent_id);
    std::string handleIntercept(const std::string& body);
    std::string handleConflicts();
    std::string handleSessionStart(const std::string& body);
    
    // Agent Management
    std::string handleRegisterAgent(const std::string& body);
    std::string handleListAgents();
    std::string handleUnregisterAgent(const std::string& agent_id);
    std::string handleSessionTurn(const std::string& id_str, const std::string& body);
    std::string handleSessionClose(const std::string& id_str, const std::string& body);
    std::string handleSessionSummary(const std::string& id_str);
    std::string handleBackupExport(const std::string& type);
    std::string handleBackupImport(const std::string& type, const std::string& body);
    std::string handleHealth();
    std::string handleMetrics();

    std::string handleListMemories(const std::string& path);
    std::string handleListEdges(const std::string& path);
    std::string handleGraphNeighbors(const std::string& id_str,
                                     bool include_incoming = false);
    std::string handleListSessions();
    std::string handleCoverageStats(const std::string& query);

    // Unified observability (MemoryEventLog) — single source of truth
    // for Gate/Reconcile/GC/Stale/etc. The legacy per-subsystem handlers
    // (handleGateLog*, handleForgetLog, handleRecallStaleLog, handlePipelineStats,
    // handleReconcileLog) were removed in Phase 4.
    std::string handleEventsQuery(const std::string& path);
    std::string handleMemoryTrace(const std::string& id_str);
    std::string handleTraceById(const std::string& trace_id_str);
    std::string handleEventsStats(const std::string& path);
    std::string handleAdminResurrect(const std::string& event_id_str, const std::string& body);

    // Manual triggers for V2 background workers (synchronous; useful for
    // observation in dev — production scheduler still runs every interval).
    std::string handleAdminForgetRun();
    std::string handleAdminConsolidationRun();

    // Auth key management
    std::string handleCreateKey(const std::string& body);
    std::string handleListKeys();
    std::string handleRevokeKey(const std::string& key_id);

    // Variable management
    std::string handleShowVariables(const std::string& path);
    std::string handleSetVariable(const std::string& name, const std::string& body);
    std::string handleReloadConfig();

public:
    // HTTP helpers — public so free-function helpers in the .cpp can reuse them.
    static std::string jsonResponse(int status, const std::string& body);
    static std::string errorResponse(int status, const std::string& message);
    static std::string extractQueryParam(const std::string& path, const std::string& key);
    static std::string urlDecode(const std::string& src);
    static void sendAll(int fd, const std::string& data);

private:

    void workerLoop();

    /// Check Authorization header. Returns empty string if OK, error response if unauthorized.
    std::string checkAuth(const std::string& headers, const std::string& path);

    Engine& engine_;
    std::string host_;
    int port_;
    int max_connections_;
    int request_timeout_ms_;
    std::string api_token_;
    int server_fd_{-1};
    std::atomic<bool> running_{false};

    // Thread pool
    std::vector<std::thread> workers_;
    std::queue<int> conn_queue_;
    std::mutex conn_mutex_;
    std::condition_variable conn_cv_;
};

}  // namespace amind
