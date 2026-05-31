#pragma once

#include "provider.h"
#include "connection_pool.h"

#include <memory>
#include <string>
#include <vector>

namespace amind {

/// OpenAI-compatible LLM provider.
///
/// Works with any service that implements the OpenAI Chat Completions API:
///   - OpenAI (api.openai.com)
///   - NVIDIA NIM (integrate.api.nvidia.com)
///   - DeepSeek, Moonshot, vLLM, LocalAI, etc.
///
/// Configuration:
///   llm_provider = openai
///   llm_model = nvidia/llama-3.1-nemotron-70b-instruct   (or gpt-4o, etc.)
///   llm_host = integrate.api.nvidia.com   (or api.openai.com)
///   llm_port = 443                         (or 80 for local HTTP)
///   llm_api_key_env = NVIDIA_API_KEY       (reads key from this env var)
///   llm_base_url = /v1                     (API path prefix, default /v1)
class OpenAILLM : public LLMProvider {
public:
    OpenAILLM(std::string model, std::string host, int port,
              std::string api_key, std::string base_path,
              std::shared_ptr<HttpConnectionPool> pool = nullptr);

    [[nodiscard]] Result<std::string> generate(
        const std::string& prompt,
        const std::string& system_prompt = "") override;

    [[nodiscard]] Result<std::string> generateJson(
        const std::string& prompt,
        const std::string& system_prompt = "") override;

    [[nodiscard]] std::string name() const override { return "openai"; }

private:
    Result<std::string> chatCompletion(const std::string& prompt,
                                        const std::string& system_prompt,
                                        bool json_mode);

    std::string model_;
    std::string host_;
    int port_;
    std::string api_key_;
    std::string base_path_;  // e.g. "/v1"
    std::shared_ptr<HttpConnectionPool> pool_;
};

/// OpenAI-compatible Embedding provider.
///
/// Works with any service that implements the OpenAI Embeddings API:
///   - OpenAI text-embedding-3-small / text-embedding-3-large
///   - NVIDIA NIM NV-Embed-QA, nvidia/nv-embedqa-e5-v5
///   - Any OpenAI-compatible embedding endpoint
class OpenAIEmbed : public EmbedProvider {
public:
    OpenAIEmbed(std::string model, std::string host, int port,
                std::string api_key, std::string base_path,
                size_t dimension,
                std::shared_ptr<HttpConnectionPool> pool = nullptr);

    [[nodiscard]] Result<std::vector<float>> embed(
        const std::string& text) override;

    [[nodiscard]] Result<std::vector<std::vector<float>>> embedBatch(
        const std::vector<std::string>& texts) override;

    [[nodiscard]] size_t dimension() const override { return dimension_; }
    [[nodiscard]] std::string name() const override { return "openai"; }

private:
    std::string model_;
    std::string host_;
    int port_;
    std::string api_key_;
    std::string base_path_;
    size_t dimension_;
    std::shared_ptr<HttpConnectionPool> pool_;
};

/// Register OpenAI-compatible providers with the ProviderRegistry.
void registerOpenAIProviders();

}  // namespace amind
