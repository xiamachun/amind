#pragma once

#include "core/result.h"

#include <atomic>
#include <string>
#include <semaphore>
#include <thread>
#include <unordered_map>

namespace amind {

/// Lightweight HTTP server for serving the WebUI dashboard.
/// Serves static files from a directory and proxies /api/* to the REST API.
class WebServer {
public:
    WebServer(const std::string& web_root, int port, int api_port,
              const std::string& api_token = "", int max_connections = 256);
    ~WebServer();

    /// Start listening in a background thread (non-blocking).
    void startAsync();

    /// Stop the server.
    void stop();

    [[nodiscard]] bool running() const { return running_.load(); }

private:
    void run();
    void handleClient(int client_fd);

    /// Serve a static file from web_root_.
    std::string serveFile(const std::string& path);

    /// Proxy a request to the REST API server.
    std::string proxyToApi(const std::string& method, const std::string& path,
                           const std::string& headers, const std::string& body);

    /// Get MIME type for a file extension.
    static std::string mimeType(const std::string& path);

    /// Build an HTTP response.
    static std::string httpResponse(int status, const std::string& content_type,
                                     const std::string& body);

    std::string web_root_;
    int port_;
    int api_port_;
    std::string api_token_;
    int server_fd_{-1};
    int max_connections_{256};
    std::unique_ptr<std::counting_semaphore<>> conn_semaphore_;
    std::atomic<bool> running_{false};
    std::thread thread_;
};

}  // namespace amind
