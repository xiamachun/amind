#include "web_server.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace amind {
namespace fs = std::filesystem;

WebServer::WebServer(const std::string& web_root, int port, int api_port,
                     const std::string& api_token, int max_connections)
    : web_root_(web_root), port_(port), api_port_(api_port), api_token_(api_token),
      max_connections_(max_connections),
      conn_semaphore_(std::make_unique<std::counting_semaphore<>>(max_connections)) {}

WebServer::~WebServer() {
    stop();
}

void WebServer::startAsync() {
    thread_ = std::thread([this]() { run(); });
    thread_.detach();
}

void WebServer::stop() {
    running_ = false;
    if (server_fd_ >= 0) {
        ::shutdown(server_fd_, SHUT_RDWR);
        close(server_fd_);
        server_fd_ = -1;
    }
}

void WebServer::run() {
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        spdlog::error("WebServer: socket() failed: {}", strerror(errno));
        return;
    }

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(server_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        spdlog::error("WebServer: bind() failed on port {}: {}", port_, strerror(errno));
        close(server_fd_);
        server_fd_ = -1;
        return;
    }

    if (listen(server_fd_, 64) < 0) {
        spdlog::error("WebServer: listen() failed: {}", strerror(errno));
        close(server_fd_);
        server_fd_ = -1;
        return;
    }

    running_ = true;
    spdlog::info("WebUI server listening on http://0.0.0.0:{}", port_);

    while (running_) {
        struct sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd_,
                               reinterpret_cast<struct sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            if (!running_) break;
            continue;
        }
        // Enforce max concurrent connections via semaphore
        if (!conn_semaphore_->try_acquire_for(std::chrono::seconds(5))) {
            spdlog::warn("WebServer: max connections reached ({}), rejecting", max_connections_);
            const char* busy = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            send(client_fd, busy, strlen(busy), 0);
            close(client_fd);
            continue;
        }
        std::thread([this, client_fd]() {
            handleClient(client_fd);
            conn_semaphore_->release();
        }).detach();
    }
}

void WebServer::handleClient(int client_fd) {
    // Set recv timeout
    struct timeval tv;
    tv.tv_sec = 5;
    tv.tv_usec = 0;
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    char buf[65536];
    ssize_t n = recv(client_fd, buf, sizeof(buf) - 1, 0);
    if (n <= 0) { close(client_fd); return; }
    buf[n] = '\0';
    std::string request(buf, static_cast<size_t>(n));

    // Parse request line
    auto first_line_end = request.find("\r\n");
    if (first_line_end == std::string::npos) { close(client_fd); return; }
    auto first_line = request.substr(0, first_line_end);
    auto sp1 = first_line.find(' ');
    auto sp2 = first_line.find(' ', sp1 + 1);
    std::string method = first_line.substr(0, sp1);
    std::string path = first_line.substr(sp1 + 1, sp2 - sp1 - 1);

    // Extract headers and body
    std::string headers;
    std::string body;
    auto hdr_start = first_line_end + 2;
    auto body_start = request.find("\r\n\r\n");
    if (body_start != std::string::npos) {
        headers = request.substr(hdr_start, body_start - hdr_start);
        body = request.substr(body_start + 4);
    }

    std::string response;

    // Route: API proxy
    if (path.find("/api/") == 0) {
        // Strip /api prefix: /api/v1/health -> /v1/health
        auto api_path = path.substr(4);
        response = proxyToApi(method, api_path, headers, body);
    } else {
        // Static file serving
        response = serveFile(path);
    }

    send(client_fd, response.c_str(), response.size(), 0);
    close(client_fd);
}

static std::string urlDecode(const std::string& encoded) {
    std::string decoded;
    decoded.reserve(encoded.size());
    for (size_t i = 0; i < encoded.size(); ++i) {
        if (encoded[i] == '%' && i + 2 < encoded.size()) {
            auto hexVal = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return -1;
            };
            int high = hexVal(encoded[i + 1]);
            int low = hexVal(encoded[i + 2]);
            if (high >= 0 && low >= 0) {
                decoded += static_cast<char>(high * 16 + low);
                i += 2;
                continue;
            }
        }
        decoded += encoded[i];
    }
    return decoded;
}

std::string WebServer::serveFile(const std::string& url_path) {
    std::string decoded_path = urlDecode(url_path);
    if (decoded_path == "/") decoded_path = "/index.html";

    // Security: canonical path must stay within web_root
    fs::path web_root_abs = fs::weakly_canonical(fs::path(web_root_));
    fs::path full_path = fs::weakly_canonical(web_root_abs / decoded_path.substr(1));
    auto root_str = web_root_abs.string();
    auto full_str = full_path.string();
    if (full_str.compare(0, root_str.size(), root_str) != 0) {
        return httpResponse(403, "text/plain", "Forbidden");
    }

    if (fs::exists(full_path) && fs::is_regular_file(full_path)) {
        std::ifstream ifs(full_path, std::ios::binary);
        if (!ifs) return httpResponse(500, "text/plain", "Internal Server Error");
        std::string content((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());
        return httpResponse(200, mimeType(full_path.string()), content);
    }

    fs::path index_path = web_root_abs / "index.html";
    if (fs::exists(index_path)) {
        std::ifstream ifs(index_path, std::ios::binary);
        if (ifs) {
            std::string content((std::istreambuf_iterator<char>(ifs)),
                                std::istreambuf_iterator<char>());
            return httpResponse(200, "text/html; charset=utf-8", content);
        }
    }
    return httpResponse(404, "text/plain", "Not Found");
}

std::string WebServer::proxyToApi(const std::string& method, const std::string& path,
                                   const std::string& /*headers*/, const std::string& body) {
    // Connect to REST API on localhost:api_port_
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) return httpResponse(502, "text/plain", "Bad Gateway");

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(api_port_));
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    // Timeout for proxy connection
    struct timeval tv;
    tv.tv_sec = 30;
    tv.tv_usec = 0;
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    if (connect(sockfd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(sockfd);
        return httpResponse(502, "text/plain", "Cannot connect to API server");
    }

    // Build proxied request
    std::string auth_header;
    if (!api_token_.empty()) {
        auth_header = "Authorization: Bearer " + api_token_ + "\r\n";
    }
    std::string req = method + " " + path + " HTTP/1.1\r\n"
                    + "Host: 127.0.0.1:" + std::to_string(api_port_) + "\r\n"
                    + "Content-Type: application/json\r\n"
                    + auth_header
                    + "Content-Length: " + std::to_string(body.size()) + "\r\n"
                    + "Connection: close\r\n\r\n"
                    + body;

    send(sockfd, req.c_str(), req.size(), 0);

    // Read response
    std::string response_data;
    char buf[8192];
    ssize_t n;
    while ((n = recv(sockfd, buf, sizeof(buf), 0)) > 0) {
        response_data.append(buf, static_cast<size_t>(n));
    }
    close(sockfd);

    if (response_data.empty()) {
        return httpResponse(502, "text/plain", "Empty response from API");
    }

    // Extract status and body from proxied response, re-wrap with CORS
    auto status_end = response_data.find("\r\n");
    int status = 200;
    if (status_end != std::string::npos) {
        auto line = response_data.substr(0, status_end);
        auto sp = line.find(' ');
        if (sp != std::string::npos) {
            try { status = std::stoi(line.substr(sp + 1, 3)); } catch (...) {}
        }
    }

    auto body_pos = response_data.find("\r\n\r\n");
    std::string resp_body;
    if (body_pos != std::string::npos) {
        resp_body = response_data.substr(body_pos + 4);
    }

    return httpResponse(status, "application/json", resp_body);
}

std::string WebServer::mimeType(const std::string& path) {
    auto ext_pos = path.rfind('.');
    if (ext_pos == std::string::npos) return "application/octet-stream";
    auto ext = path.substr(ext_pos);

    if (ext == ".html" || ext == ".htm") return "text/html; charset=utf-8";
    if (ext == ".css")  return "text/css; charset=utf-8";
    if (ext == ".js")   return "application/javascript; charset=utf-8";
    if (ext == ".json") return "application/json";
    if (ext == ".png")  return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".svg")  return "image/svg+xml";
    if (ext == ".ico")  return "image/x-icon";
    if (ext == ".woff")  return "font/woff";
    if (ext == ".woff2") return "font/woff2";
    if (ext == ".ttf")   return "font/ttf";
    if (ext == ".map")   return "application/json";
    return "application/octet-stream";
}

std::string WebServer::httpResponse(int status, const std::string& content_type,
                                     const std::string& body) {
    std::string status_text;
    switch (status) {
        case 200: status_text = "OK"; break;
        case 403: status_text = "Forbidden"; break;
        case 404: status_text = "Not Found"; break;
        case 500: status_text = "Internal Server Error"; break;
        case 502: status_text = "Bad Gateway"; break;
        default:  status_text = "Error"; break;
    }

    return "HTTP/1.1 " + std::to_string(status) + " " + status_text + "\r\n"
         + "Content-Type: " + content_type + "\r\n"
         + "Content-Length: " + std::to_string(body.size()) + "\r\n"
         + "Access-Control-Allow-Origin: *\r\n"
         + "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n"
         + "Access-Control-Allow-Headers: Content-Type\r\n"
         + "Cache-Control: no-cache\r\n"
         + "Connection: close\r\n\r\n"
         + body;
}

}  // namespace amind
