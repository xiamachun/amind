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
    std::string handleDeleteNamespace(const std::string& ns);
    std::string handleIntercept(const std::string& body);
    std::string handleCoverage(const std::string& query);
    std::string handleConflicts();
    std::string handleSessionStart(const std::string& body);
    std::string handleSessionTurn(const std::string& id_str, const std::string& body);
    std::string handleSessionClose(const std::string& id_str, const std::string& body);
    std::string handleSessionSummary(const std::string& id_str);
    std::string handleBackupExport(const std::string& type);
    std::string handleBackupImport(const std::string& type, const std::string& body);
    std::string handleHealth();
    std::string handleMetrics();

    std::string handleListMemories(const std::string& path);
    std::string handleListEdges(const std::string& path);
    std::string handleGraphNeighbors(const std::string& id_str);
    std::string handleListSessions();
    std::string handleCoverageStats(const std::string& query);

    // Gate log audit + resurrect
    std::string handleGateLogList(const std::string& path);
    std::string handleGateLogStats(const std::string& path);
    std::string handleGateLogResurrect(const std::string& id_str, const std::string& body);

    // Forget log audit (read-only — recovery semantics are not in v1)
    std::string handleForgetLog(const std::string& path);

    // Recall staleness filter audit (read-only)
    std::string handleRecallStaleLog(const std::string& path);

    // Manual triggers for V2 background workers (synchronous; useful for
    // observation in dev — production scheduler still runs every interval).
    std::string handleAdminForgetRun();
    std::string handleAdminConsolidationRun();

    // Pipeline observability
    std::string handlePipelineStats();
    std::string handleReconcileLog(const std::string& path);

    // Auth key management
    std::string handleCreateKey(const std::string& body);
    std::string handleListKeys();
    std::string handleRevokeKey(const std::string& key_id);

    // Variable management
    std::string handleShowVariables(const std::string& path);
    std::string handleSetVariable(const std::string& name, const std::string& body);
    std::string handleReloadConfig();

    // HTTP helpers
    static std::string jsonResponse(int status, const std::string& body);
    static std::string errorResponse(int status, const std::string& message);
    static std::string extractQueryParam(const std::string& path, const std::string& key);
    static std::string urlDecode(const std::string& src);
    static void sendAll(int fd, const std::string& data);

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
