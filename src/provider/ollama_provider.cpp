#include "ollama_provider.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <netdb.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <sys/socket.h>
#include <unistd.h>

#ifdef AMIND_TLS_ENABLED
#include <openssl/ssl.h>
#include <openssl/err.h>
#endif

namespace amind {

using json = nlohmann::json;

// ── Chunked Transfer-Encoding decoder ───────────────────────────────────

static std::string decodeChunked(const std::string& data) {
    std::string result;
    size_t pos = 0;
    while (pos < data.size()) {
        auto crlf = data.find("\r\n", pos);
        if (crlf == std::string::npos) break;
        size_t chunk_size = 0;
        try {
            chunk_size = std::stoull(data.substr(pos, crlf - pos), nullptr, 16);
        } catch (...) { break; }
        if (chunk_size == 0) break;           // terminal chunk
        pos = crlf + 2;                       // skip size line
        if (pos + chunk_size > data.size()) break;
        result.append(data, pos, chunk_size);
        pos += chunk_size + 2;                // skip chunk data + \r\n
    }
    return result;
}

// ── Helper: loop-based send to handle partial writes ──────────────────────

static bool sendAll(int sockfd, const char* data, size_t len) {
    size_t total_sent = 0;
    while (total_sent < len) {
        ssize_t sent = send(sockfd, data + total_sent, len - total_sent, 0);
        if (sent <= 0) return false;
        total_sent += static_cast<size_t>(sent);
    }
    return true;
}

// ── HttpClient (pool-aware) ─────────────────────────────────────────────

Result<HttpClient::Response> HttpClient::post(
    const std::string& host, int port,
    const std::string& path, const std::string& json_body,
    int timeout_ms, HttpConnectionPool* pool) {
    return post(host, port, path, json_body, {}, timeout_ms, pool);
}

// ── Helper: connect a raw TCP socket ────────────────────────────────────

static Result<int> connectSocket(const std::string& host, int port, int timeout_ms) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        return makeError(Error::ProviderError, "socket() failed: " + std::string(strerror(errno)));
    }

    struct sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(static_cast<uint16_t>(port));

    if (inet_pton(AF_INET, host.c_str(), &server_addr.sin_addr) <= 0) {
        struct addrinfo hints{}, *res;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0) {
            close(sockfd);
            return makeError(Error::ProviderUnavailable, "cannot resolve host: " + host);
        }
        std::memcpy(&server_addr, res->ai_addr, res->ai_addrlen);
        freeaddrinfo(res);
    }

    if (connect(sockfd, reinterpret_cast<struct sockaddr*>(&server_addr), sizeof(server_addr)) < 0) {
        close(sockfd);
        return makeError(Error::ProviderUnavailable,
                         "connect failed: " + host + ":" + std::to_string(port));
    }

    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    return sockfd;
}

// ── Helper: build HTTP request string ──────────────────────────────────

static std::string buildHttpRequest(
    const std::string& method,
    const std::string& host, int /*port*/,
    const std::string& path, const std::string& body,
    const std::vector<std::pair<std::string, std::string>>& extra_headers,
    bool keep_alive) {

    std::string conn_header = keep_alive ? "Connection: keep-alive" : "Connection: close";
    std::string request = method + " " + path + " HTTP/1.1\r\n"
                        + "Host: " + host + "\r\n"
                        + conn_header + "\r\n";
    if (!body.empty()) {
        request += "Content-Type: application/json\r\n";
        request += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    }
    for (const auto& [header_name, header_value] : extra_headers) {
        request += header_name + ": " + header_value + "\r\n";
    }
    request += "\r\n" + body;
    return request;
}

static std::string buildHttpRequest(
    const std::string& host, int port,
    const std::string& path, const std::string& json_body,
    const std::vector<std::pair<std::string, std::string>>& extra_headers,
    bool keep_alive) {
    return buildHttpRequest("POST", host, port, path, json_body, extra_headers, keep_alive);
}

// ── Helper: parse HTTP response ────────────────────────────────────────

static Result<HttpClient::Response> parseHttpResponse(const std::string& response_data) {
    if (response_data.empty()) {
        return makeError(Error::ProviderTimeout, "empty response");
    }

    HttpClient::Response resp;

    auto status_end = response_data.find("\r\n");
    if (status_end != std::string::npos) {
        auto status_line = response_data.substr(0, status_end);
        auto sp1 = status_line.find(' ');
        if (sp1 != std::string::npos) {
            try {
                resp.status_code = std::stoi(status_line.substr(sp1 + 1, 3));
            } catch (...) {
                resp.status_code = 500;
            }
        }
    }

    auto body_start = response_data.find("\r\n\r\n");
    if (body_start != std::string::npos) {
        std::string raw_body = response_data.substr(body_start + 4);
        auto te_pos = response_data.find("Transfer-Encoding: chunked");
        if (te_pos != std::string::npos && te_pos < body_start) {
            resp.body = decodeChunked(raw_body);
        } else {
            resp.body = raw_body;
        }
    }

    return resp;
}

// ── Helper: read full HTTP response from socket ────────────────────────

// Read function type: abstracts over raw socket recv() vs SSL_read()
using ReadFunc = std::function<ssize_t(char* buf, size_t len)>;

static std::string readHttpResponse(const ReadFunc& read_fn, bool detect_end) {
    std::string response_data;
    char buf[8192];
    bool headers_done = false;
    bool is_chunked = false;
    size_t content_length = 0;
    size_t hdr_end_pos = std::string::npos;

    ssize_t n;
    while ((n = read_fn(buf, sizeof(buf))) > 0) {
        response_data.append(buf, static_cast<size_t>(n));

        if (!headers_done) {
            hdr_end_pos = response_data.find("\r\n\r\n");
            if (hdr_end_pos != std::string::npos) {
                headers_done = true;
                auto hdr = response_data.substr(0, hdr_end_pos);
                if (hdr.find("Transfer-Encoding: chunked") != std::string::npos) {
                    is_chunked = true;
                } else {
                    auto cl_pos = hdr.find("Content-Length: ");
                    if (cl_pos != std::string::npos) {
                        auto cl_val = hdr.substr(cl_pos + 16);
                        try {
                            content_length = std::stoull(cl_val.substr(0, cl_val.find("\r\n")));
                        } catch (...) {
                            content_length = 0;
                        }
                    }
                }
            }
        }

        // Detect end-of-body
        if (headers_done && detect_end) {
            if (is_chunked) {
                if (response_data.find("\r\n0\r\n\r\n", hdr_end_pos + 4) != std::string::npos) {
                    break;
                }
            } else if (content_length > 0) {
                size_t body_received = response_data.size() - (hdr_end_pos + 4);
                if (body_received >= content_length) break;
            }
        }
    }
    return response_data;
}

// ── TLS POST implementation ────────────────────────────────────────────

#ifdef AMIND_TLS_ENABLED
static Result<HttpClient::Response> postTls(
    const std::string& host, int port,
    const std::string& request_str, int timeout_ms) {

    auto sock_result = connectSocket(host, port, timeout_ms);
    if (!sock_result.ok()) return sock_result.error();
    int sockfd = *sock_result;

    // Initialize OpenSSL
    SSL_CTX* ctx = SSL_CTX_new(TLS_client_method());
    if (!ctx) {
        close(sockfd);
        return makeError(Error::ProviderError, "SSL_CTX_new failed");
    }

    SSL* ssl = SSL_new(ctx);
    if (!ssl) {
        SSL_CTX_free(ctx);
        close(sockfd);
        return makeError(Error::ProviderError, "SSL_new failed");
    }

    // Set SNI hostname (required by most HTTPS servers)
    SSL_set_tlsext_host_name(ssl, host.c_str());
    SSL_set_fd(ssl, sockfd);

    if (SSL_connect(ssl) <= 0) {
        unsigned long err = ERR_get_error();
        char err_buf[256];
        ERR_error_string_n(err, err_buf, sizeof(err_buf));
        SSL_free(ssl);
        SSL_CTX_free(ctx);
        close(sockfd);
        return makeError(Error::ProviderUnavailable,
                         "TLS handshake failed: " + std::string(err_buf));
    }

    // Send request over TLS
    int sent = SSL_write(ssl, request_str.c_str(), static_cast<int>(request_str.size()));
    if (sent <= 0) {
        SSL_free(ssl);
        SSL_CTX_free(ctx);
        close(sockfd);
        return makeError(Error::ProviderError, "SSL_write failed");
    }

    // Read response over TLS
    ReadFunc ssl_read = [ssl](char* buf, size_t len) -> ssize_t {
        int n = SSL_read(ssl, buf, static_cast<int>(len));
        return n > 0 ? static_cast<ssize_t>(n) : -1;
    };
    auto response_data = readHttpResponse(ssl_read, true);

    // Cleanup
    SSL_shutdown(ssl);
    SSL_free(ssl);
    SSL_CTX_free(ctx);
    close(sockfd);

    return parseHttpResponse(response_data);
}
#endif  // AMIND_TLS_ENABLED

// ── HttpClient::post (with extra_headers) ──────────────────────────────

Result<HttpClient::Response> HttpClient::post(
    const std::string& host, int port,
    const std::string& path, const std::string& json_body,
    const std::vector<std::pair<std::string, std::string>>& extra_headers,
    int timeout_ms, HttpConnectionPool* pool) {

    bool use_tls = (port == 443);

    // TLS path: no connection pooling (each request creates a fresh TLS session)
    if (use_tls) {
#ifdef AMIND_TLS_ENABLED
        auto request_str = buildHttpRequest(host, port, path, json_body, extra_headers, false);
        return postTls(host, port, request_str, timeout_ms);
#else
        return makeError(Error::ConfigError,
                         "TLS required (port 443) but amind was compiled without TLS support. "
                         "Rebuild with -DAMIND_ENABLE_TLS=ON");
#endif
    }

    // Plain HTTP path (original logic with connection pool support)
    int sockfd = -1;

    if (pool) {
        sockfd = pool->acquire(host, port);
    }

    if (sockfd < 0) {
        auto sock_result = connectSocket(host, port, timeout_ms);
        if (!sock_result.ok()) {
            if (pool) pool->recordFailure();
            return sock_result.error();
        }
        sockfd = *sock_result;
    }

    auto request_str = buildHttpRequest(host, port, path, json_body, extra_headers, pool != nullptr);

    if (!sendAll(sockfd, request_str.c_str(), request_str.size())) {
        if (pool) { pool->discard(sockfd); pool->recordFailure(); } else close(sockfd);
        return makeError(Error::ProviderError, "send failed");
    }

    ReadFunc sock_read = [sockfd](char* buf, size_t len) -> ssize_t {
        return recv(sockfd, buf, len, 0);
    };
    auto response_data = readHttpResponse(sock_read, pool != nullptr);

    if (response_data.empty()) {
        if (pool) { pool->discard(sockfd); pool->recordFailure(); } else close(sockfd);
        return makeError(Error::ProviderTimeout, "empty response from " + host);
    }

    // Return connection to pool or close
    if (pool) {
        pool->release(sockfd, host, port);
        pool->recordSuccess();
    } else {
        close(sockfd);
    }

    return parseHttpResponse(response_data);
}

// ── HttpClient::request (generic) ─────────────────────────────────────

Result<HttpClient::Response> HttpClient::request(
    const std::string& method,
    const std::string& host, int port,
    const std::string& path, const std::string& body,
    const std::vector<std::pair<std::string, std::string>>& extra_headers,
    int timeout_ms, HttpConnectionPool* pool) {

    bool use_tls = (port == 443);
    if (use_tls) {
#ifdef AMIND_TLS_ENABLED
        auto request_str = buildHttpRequest(method, host, port, path, body, extra_headers, false);
        return postTls(host, port, request_str, timeout_ms);
#else
        return makeError(Error::ConfigError,
                         "TLS required (port 443) but amind was compiled without TLS support.");
#endif
    }

    int sockfd = -1;
    if (pool) sockfd = pool->acquire(host, port);

    if (sockfd < 0) {
        auto sock_result = connectSocket(host, port, timeout_ms);
        if (!sock_result.ok()) {
            if (pool) pool->recordFailure();
            return sock_result.error();
        }
        sockfd = *sock_result;
    }

    auto request_str = buildHttpRequest(method, host, port, path, body, extra_headers, pool != nullptr);

    if (!sendAll(sockfd, request_str.c_str(), request_str.size())) {
        if (pool) { pool->discard(sockfd); pool->recordFailure(); } else close(sockfd);
        return makeError(Error::ProviderError, "send failed");
    }

    ReadFunc sock_read = [sockfd](char* buf, size_t len) -> ssize_t {
        return recv(sockfd, buf, len, 0);
    };
    auto response_data = readHttpResponse(sock_read, pool != nullptr);

    if (response_data.empty()) {
        if (pool) { pool->discard(sockfd); pool->recordFailure(); } else close(sockfd);
        return makeError(Error::ProviderTimeout, "empty response from " + host);
    }

    if (pool) { pool->release(sockfd, host, port); pool->recordSuccess(); }
    else { close(sockfd); }

    return parseHttpResponse(response_data);
}

// ── HttpClient::get ───────────────────────────────────────────────────

Result<HttpClient::Response> HttpClient::get(
    const std::string& host, int port,
    const std::string& path, int timeout_ms, HttpConnectionPool* pool) {
    return request("GET", host, port, path, "", {}, timeout_ms, pool);
}

// ── HttpClient::del ───────────────────────────────────────────────────

Result<HttpClient::Response> HttpClient::del(
    const std::string& host, int port,
    const std::string& path, int timeout_ms, HttpConnectionPool* pool) {
    return request("DELETE", host, port, path, "", {}, timeout_ms, pool);
}

// ── OllamaLLM ──────────────────────────────────────────────────────────

OllamaLLM::OllamaLLM(std::string model, std::string host, int port,
                       std::shared_ptr<HttpConnectionPool> pool)
    : model_(std::move(model)), host_(std::move(host)), port_(port),
      pool_(std::move(pool)) {}

Result<std::string> OllamaLLM::generate(
    const std::string& prompt, const std::string& system_prompt) {

    json request_body = {
        {"model", model_},
        {"prompt", prompt},
        {"stream", false}
    };
    if (!system_prompt.empty()) {
        request_body["system"] = system_prompt;
    }

    auto resp = HttpClient::post(host_, port_, "/api/generate",
                                  request_body.dump(), 30000, pool_.get());
    if (!resp.ok()) return resp.error();

    if (resp->status_code != 200) {
        return makeError(Error::ProviderError,
                         "Ollama returned HTTP " + std::to_string(resp->status_code));
    }

    try {
        auto j = json::parse(resp->body);
        return j.value("response", "");
    } catch (const json::exception& e) {
        return makeError(Error::ProviderError, "JSON parse error: " + std::string(e.what()));
    }
}

Result<std::string> OllamaLLM::generateJson(
    const std::string& prompt, const std::string& system_prompt) {

    json request_body = {
        {"model", model_},
        {"prompt", prompt},
        {"stream", false},
        {"format", "json"}
    };
    if (!system_prompt.empty()) {
        request_body["system"] = system_prompt;
    }

    auto resp = HttpClient::post(host_, port_, "/api/generate",
                                  request_body.dump(), 30000, pool_.get());
    if (!resp.ok()) return resp.error();

    if (resp->status_code != 200) {
        return makeError(Error::ProviderError,
                         "Ollama returned HTTP " + std::to_string(resp->status_code));
    }

    try {
        auto j = json::parse(resp->body);
        return j.value("response", "");
    } catch (const json::exception& e) {
        return makeError(Error::ProviderError, "JSON parse error: " + std::string(e.what()));
    }
}

// ── OllamaEmbed ─────────────────────────────────────────────────────────

OllamaEmbed::OllamaEmbed(std::string model, std::string host, int port,
                           size_t dim, std::shared_ptr<HttpConnectionPool> pool)
    : model_(std::move(model)), host_(std::move(host)), port_(port),
      dimension_(dim), pool_(std::move(pool)) {}

Result<std::vector<float>> OllamaEmbed::embed(const std::string& text) {
    json request_body = {
        {"model", model_},
        {"input", text}
    };

    auto resp = HttpClient::post(host_, port_, "/api/embed",
                                  request_body.dump(), 30000, pool_.get());
    if (!resp.ok()) return resp.error();

    if (resp->status_code != 200) {
        return makeError(Error::ProviderError,
                         "Ollama embed returned HTTP " + std::to_string(resp->status_code));
    }

    try {
        auto j = json::parse(resp->body);
        if (j.contains("embeddings") && j["embeddings"].is_array()
            && !j["embeddings"].empty()) {
            return j["embeddings"][0].get<std::vector<float>>();
        }
        return makeError(Error::ProviderError, "unexpected embedding response format");
    } catch (const json::exception& e) {
        return makeError(Error::ProviderError, "JSON parse error: " + std::string(e.what()));
    }
}

Result<std::vector<std::vector<float>>> OllamaEmbed::embedBatch(
    const std::vector<std::string>& texts) {

    std::vector<std::vector<float>> results;
    results.reserve(texts.size());
    for (const auto& text : texts) {
        auto result = embed(text);
        if (!result.ok()) return result.error();
        results.push_back(std::move(*result));
    }
    return results;
}

// ── Registration ────────────────────────────────────────────────────────

void registerOllamaProviders() {
    auto& registry = ProviderRegistry::instance();

    registry.registerLLM("ollama",
        [](const std::string& model, const std::string& host, int port,
           const std::string& /*api_key*/, const std::string& /*base_url*/) {
            return std::make_unique<OllamaLLM>(model, host, port);
        });

    registry.registerEmbed("ollama",
        [](const std::string& model, const std::string& host, int port,
           const std::string& /*api_key*/, const std::string& /*base_url*/, size_t dim) {
            return std::make_unique<OllamaEmbed>(model, host, port, dim);
        });
}

}  // namespace amind
