#include "anthropic_provider.h"
#include "ollama_provider.h"  // HttpClient

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace amind {

using json = nlohmann::json;

AnthropicLLM::AnthropicLLM(std::string model, std::string host, int port,
                             std::string api_key, std::string base_path,
                             std::shared_ptr<HttpConnectionPool> pool,
                             std::vector<std::pair<std::string, std::string>> extra_headers)
    : model_(std::move(model)), host_(std::move(host)), port_(port),
      api_key_(std::move(api_key)), base_path_(std::move(base_path)),
      pool_(std::move(pool)), extra_headers_(std::move(extra_headers)) {
    if (base_path_.empty()) base_path_ = "/v1";
    if (base_path_.front() != '/') base_path_ = "/" + base_path_;
    if (base_path_.back() == '/') base_path_.pop_back();
}

Result<std::string> AnthropicLLM::generate(
    const std::string& prompt, const std::string& system_prompt) {
    return callMessages(prompt, system_prompt, false);
}

Result<std::string> AnthropicLLM::generateJson(
    const std::string& prompt, const std::string& system_prompt) {
    return callMessages(prompt, system_prompt, true);
}

Result<std::string> AnthropicLLM::callMessages(
    const std::string& prompt,
    const std::string& system_prompt,
    bool json_mode) {

    json messages = json::array();
    messages.push_back({{"role", "user"}, {"content", prompt}});

    json request_body = {
        {"model", model_},
        {"max_tokens", 4096},
        {"messages", messages},
        {"stream", false}
    };

    // Anthropic uses top-level "system" field, not a system message
    std::string effective_system = system_prompt;
    if (json_mode) {
        std::string json_instruction =
            "You must respond with valid JSON only. "
            "No markdown fences, no explanation, no trailing text.";
        if (effective_system.empty()) {
            effective_system = json_instruction;
        } else {
            effective_system = json_instruction + "\n\n" + effective_system;
        }
    }
    if (!effective_system.empty()) {
        request_body["system"] = effective_system;
    }

    // Required Anthropic headers
    std::vector<std::pair<std::string, std::string>> headers;
    headers.emplace_back("x-api-key", api_key_);
    headers.emplace_back("anthropic-version", "2023-06-01");
    // Append any extra headers from config (e.g. proxy-specific)
    for (const auto& h : extra_headers_) {
        headers.push_back(h);
    }

    std::string endpoint = base_path_ + "/messages";
    auto resp = HttpClient::post(host_, port_, endpoint,
                                  request_body.dump(), headers, 120000, pool_.get());
    if (!resp.ok()) return resp.error();

    if (resp->status_code != 200) {
        spdlog::warn("Anthropic API returned HTTP {}: {}", resp->status_code, resp->body);
        return makeError(Error::ProviderError,
                         "Anthropic API returned HTTP " + std::to_string(resp->status_code)
                         + ": " + resp->body.substr(0, 200));
    }

    try {
        auto j = json::parse(resp->body);
        if (j.contains("content") && j["content"].is_array() && !j["content"].empty()) {
            for (const auto& block : j["content"]) {
                if (block.value("type", "") == "text") {
                    std::string text = block["text"].get<std::string>();
                    if (json_mode) {
                        // Strip markdown fences: ```json\n...\n```
                        auto pos = text.find("```");
                        if (pos != std::string::npos) {
                            auto start = text.find('\n', pos);
                            if (start != std::string::npos) start++;
                            else start = pos + 3;
                            auto end = text.rfind("```");
                            if (end != std::string::npos && end > pos) {
                                text = text.substr(start, end - start);
                            }
                        }
                    }
                    return text;
                }
            }
            return makeError(Error::ProviderError, "no text block in response content");
        }
        return makeError(Error::ProviderError, "unexpected response: no content array");
    } catch (const json::exception& e) {
        return makeError(Error::ProviderError, "JSON parse error: " + std::string(e.what()));
    }
}

// ── Registration ────────────────────────────────────────────────────────

void registerAnthropicProviders() {
    auto& registry = ProviderRegistry::instance();

    auto factory = [](const std::string& model, const std::string& host, int port,
                      const std::string& api_key, const std::string& base_url) {
        return std::make_unique<AnthropicLLM>(model, host, port, api_key, base_url);
    };

    registry.registerLLM("anthropic", factory);
    registry.registerLLM("claude", factory);
}

}  // namespace amind
