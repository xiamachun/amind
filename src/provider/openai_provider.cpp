#include "openai_provider.h"
#include "ollama_provider.h"  // HttpClient

#include <cstdlib>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace amind {

using json = nlohmann::json;

// ── OpenAILLM ───────────────────────────────────────────────────────────

OpenAILLM::OpenAILLM(std::string model, std::string host, int port,
                       std::string api_key, std::string base_path,
                       std::shared_ptr<HttpConnectionPool> pool)
    : model_(std::move(model)), host_(std::move(host)), port_(port),
      api_key_(std::move(api_key)), base_path_(std::move(base_path)),
      pool_(std::move(pool)) {
    // Normalize base_path: ensure it starts with / and has no trailing /
    if (base_path_.empty()) base_path_ = "/v1";
    if (base_path_.front() != '/') base_path_ = "/" + base_path_;
    if (base_path_.back() == '/') base_path_.pop_back();
}

Result<std::string> OpenAILLM::generate(
    const std::string& prompt, const std::string& system_prompt) {
    return chatCompletion(prompt, system_prompt, false);
}

Result<std::string> OpenAILLM::generateJson(
    const std::string& prompt, const std::string& system_prompt) {
    return chatCompletion(prompt, system_prompt, true);
}

Result<std::string> OpenAILLM::chatCompletion(
    const std::string& prompt,
    const std::string& system_prompt,
    bool json_mode) {

    // Build messages array (OpenAI Chat Completions format)
    json messages = json::array();
    if (!system_prompt.empty()) {
        messages.push_back({{"role", "system"}, {"content", system_prompt}});
    }
    messages.push_back({{"role", "user"}, {"content", prompt}});

    json request_body = {
        {"model", model_},
        {"messages", messages},
        {"stream", false}
    };
    if (json_mode) {
        request_body["response_format"] = {{"type", "json_object"}};
    }

    // Build auth header
    std::vector<std::pair<std::string, std::string>> headers;
    if (!api_key_.empty()) {
        headers.emplace_back("Authorization", "Bearer " + api_key_);
    }

    std::string endpoint = base_path_ + "/chat/completions";
    auto resp = HttpClient::post(host_, port_, endpoint,
                                  request_body.dump(), headers, 60000, pool_.get());
    if (!resp.ok()) return resp.error();

    if (resp->status_code != 200) {
        spdlog::warn("OpenAI API returned HTTP {}: {}", resp->status_code, resp->body);
        return makeError(Error::ProviderError,
                         "OpenAI API returned HTTP " + std::to_string(resp->status_code)
                         + ": " + resp->body.substr(0, 200));
    }

    try {
        auto j = json::parse(resp->body);
        if (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty()) {
            return j["choices"][0]["message"]["content"].get<std::string>();
        }
        return makeError(Error::ProviderError, "unexpected response: no choices");
    } catch (const json::exception& e) {
        return makeError(Error::ProviderError, "JSON parse error: " + std::string(e.what()));
    }
}

// ── OpenAIEmbed ─────────────────────────────────────────────────────────

OpenAIEmbed::OpenAIEmbed(std::string model, std::string host, int port,
                           std::string api_key, std::string base_path,
                           size_t dimension,
                           std::shared_ptr<HttpConnectionPool> pool)
    : model_(std::move(model)), host_(std::move(host)), port_(port),
      api_key_(std::move(api_key)), base_path_(std::move(base_path)),
      dimension_(dimension), pool_(std::move(pool)) {
    if (base_path_.empty()) base_path_ = "/v1";
    if (base_path_.front() != '/') base_path_ = "/" + base_path_;
    if (base_path_.back() == '/') base_path_.pop_back();
}

Result<std::vector<float>> OpenAIEmbed::embed(const std::string& text) {
    json request_body = {
        {"model", model_},
        {"input", text}
    };

    // NVIDIA NIM asymmetric models require input_type
    if (model_.find("nvidia/") == 0 || model_.find("nemo") != std::string::npos) {
        request_body["input_type"] = "query";
    }

    std::vector<std::pair<std::string, std::string>> headers;
    if (!api_key_.empty()) {
        headers.emplace_back("Authorization", "Bearer " + api_key_);
    }

    std::string endpoint = base_path_ + "/embeddings";
    auto resp = HttpClient::post(host_, port_, endpoint,
                                  request_body.dump(), headers, 30000, pool_.get());
    if (!resp.ok()) return resp.error();

    if (resp->status_code != 200) {
        spdlog::warn("OpenAI Embed API returned HTTP {}: {}", resp->status_code, resp->body);
        return makeError(Error::ProviderError,
                         "OpenAI Embed API returned HTTP " + std::to_string(resp->status_code));
    }

    try {
        auto j = json::parse(resp->body);
        if (j.contains("data") && j["data"].is_array() && !j["data"].empty()) {
            return j["data"][0]["embedding"].get<std::vector<float>>();
        }
        return makeError(Error::ProviderError, "unexpected embedding response: no data");
    } catch (const json::exception& e) {
        return makeError(Error::ProviderError, "JSON parse error: " + std::string(e.what()));
    }
}

Result<std::vector<std::vector<float>>> OpenAIEmbed::embedBatch(
    const std::vector<std::string>& texts) {

    json request_body = {
        {"model", model_},
        {"input", texts}
    };

    // NVIDIA NIM asymmetric models require input_type
    if (model_.find("nvidia/") == 0 || model_.find("nemo") != std::string::npos) {
        request_body["input_type"] = "query";
    }

    std::vector<std::pair<std::string, std::string>> headers;
    if (!api_key_.empty()) {
        headers.emplace_back("Authorization", "Bearer " + api_key_);
    }

    std::string endpoint = base_path_ + "/embeddings";
    auto resp = HttpClient::post(host_, port_, endpoint,
                                  request_body.dump(), headers, 60000, pool_.get());
    if (!resp.ok()) return resp.error();

    if (resp->status_code != 200) {
        return makeError(Error::ProviderError,
                         "OpenAI Embed batch returned HTTP " + std::to_string(resp->status_code));
    }

    try {
        auto j = json::parse(resp->body);
        std::vector<std::vector<float>> results;
        if (j.contains("data") && j["data"].is_array()) {
            results.reserve(j["data"].size());
            for (const auto& item : j["data"]) {
                results.push_back(item["embedding"].get<std::vector<float>>());
            }
        }
        return results;
    } catch (const json::exception& e) {
        return makeError(Error::ProviderError, "JSON parse error: " + std::string(e.what()));
    }
}

// ── Registration ────────────────────────────────────────────────────────

void registerOpenAIProviders() {
    auto& registry = ProviderRegistry::instance();

    registry.registerLLM("openai",
        [](const std::string& model, const std::string& host, int port,
           const std::string& api_key, const std::string& base_url) {
            return std::make_unique<OpenAILLM>(model, host, port, api_key, base_url);
        });

    // Also register "nvidia" as an alias — same protocol, different defaults
    registry.registerLLM("nvidia",
        [](const std::string& model, const std::string& host, int port,
           const std::string& api_key, const std::string& base_url) {
            return std::make_unique<OpenAILLM>(model, host, port, api_key, base_url);
        });

    registry.registerEmbed("openai",
        [](const std::string& model, const std::string& host, int port,
           const std::string& api_key, const std::string& base_url, size_t dim) {
            return std::make_unique<OpenAIEmbed>(model, host, port, api_key, base_url, dim);
        });

    registry.registerEmbed("nvidia",
        [](const std::string& model, const std::string& host, int port,
           const std::string& api_key, const std::string& base_url, size_t dim) {
            return std::make_unique<OpenAIEmbed>(model, host, port, api_key, base_url, dim);
        });
}

}  // namespace amind
