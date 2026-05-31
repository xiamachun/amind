#pragma once

#include "provider.h"
#include "connection_pool.h"

#include <memory>
#include <string>

namespace amind {

/// HTTP/HTTPS helper for API calls.
/// Automatically uses TLS when port == 443 (requires AMIND_TLS_ENABLED at compile time).
/// When a connection pool is provided, reuses TCP connections with keep-alive (HTTP only).
class HttpClient {
public:
    struct Response {
        int status_code{0};
        std::string body;
    };

    /// POST with default headers (Content-Type: application/json).
    [[nodiscard]] static Result<Response> post(
        const std::string& host, int port,
        const std::string& path, const std::string& json_body,
        int timeout_ms = 30000,
        HttpConnectionPool* pool = nullptr);

    /// POST with custom extra headers (e.g. Authorization).
    /// Each pair is {header_name, header_value}.
    /// TLS is auto-enabled when port == 443.
    [[nodiscard]] static Result<Response> post(
        const std::string& host, int port,
        const std::string& path, const std::string& json_body,
        const std::vector<std::pair<std::string, std::string>>& extra_headers,
        int timeout_ms = 30000,
        HttpConnectionPool* pool = nullptr);

    /// GET request.
    [[nodiscard]] static Result<Response> get(
        const std::string& host, int port,
        const std::string& path,
        int timeout_ms = 30000,
        HttpConnectionPool* pool = nullptr);

    /// DELETE request.
    [[nodiscard]] static Result<Response> del(
        const std::string& host, int port,
        const std::string& path,
        int timeout_ms = 30000,
        HttpConnectionPool* pool = nullptr);

    /// Generic request with arbitrary method and optional body.
    [[nodiscard]] static Result<Response> request(
        const std::string& method,
        const std::string& host, int port,
        const std::string& path, const std::string& body = "",
        const std::vector<std::pair<std::string, std::string>>& extra_headers = {},
        int timeout_ms = 30000,
        HttpConnectionPool* pool = nullptr);
};

/// Ollama LLM provider — calls /api/generate endpoint.
class OllamaLLM : public LLMProvider {
public:
    OllamaLLM(std::string model, std::string host, int port,
              std::shared_ptr<HttpConnectionPool> pool = nullptr);

    [[nodiscard]] Result<std::string> generate(
        const std::string& prompt,
        const std::string& system_prompt = "") override;

    [[nodiscard]] Result<std::string> generateJson(
        const std::string& prompt,
        const std::string& system_prompt = "") override;

    [[nodiscard]] std::string name() const override { return "ollama"; }

private:
    std::string model_;
    std::string host_;
    int port_;
    std::shared_ptr<HttpConnectionPool> pool_;
};

/// Ollama Embedding provider — calls /api/embed endpoint.
class OllamaEmbed : public EmbedProvider {
public:
    OllamaEmbed(std::string model, std::string host, int port, size_t dim,
                std::shared_ptr<HttpConnectionPool> pool = nullptr);

    [[nodiscard]] Result<std::vector<float>> embed(
        const std::string& text) override;

    [[nodiscard]] Result<std::vector<std::vector<float>>> embedBatch(
        const std::vector<std::string>& texts) override;

    [[nodiscard]] size_t dimension() const override { return dimension_; }
    [[nodiscard]] std::string name() const override { return "ollama"; }

private:
    std::string model_;
    std::string host_;
    int port_;
    size_t dimension_;
    std::shared_ptr<HttpConnectionPool> pool_;
};

/// Register Ollama providers with the ProviderRegistry.
void registerOllamaProviders();

}  // namespace amind
