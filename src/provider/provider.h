#pragma once

#include "core/result.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace amind {

/// Ranked result from a reranking provider.
struct RankedResult {
    size_t index;    // original index in the input
    float score;     // reranking score
};

// ── Abstract Provider Interfaces ────────────────────────────────────────────

/// LLM provider interface — generates text from prompt.
class LLMProvider {
public:
    virtual ~LLMProvider() = default;

    /// Generate text from a prompt. Returns generated text or error.
    [[nodiscard]] virtual Result<std::string> generate(
        const std::string& prompt,
        const std::string& system_prompt = "") = 0;

    /// Generate with JSON mode (structured output).
    [[nodiscard]] virtual Result<std::string> generateJson(
        const std::string& prompt,
        const std::string& system_prompt = "") = 0;

    [[nodiscard]] virtual std::string name() const = 0;
};

/// Embedding provider interface — converts text to vectors.
class EmbedProvider {
public:
    virtual ~EmbedProvider() = default;

    /// Embed a single text into a vector.
    [[nodiscard]] virtual Result<std::vector<float>> embed(
        const std::string& text) = 0;

    /// Batch embed multiple texts.
    [[nodiscard]] virtual Result<std::vector<std::vector<float>>> embedBatch(
        const std::vector<std::string>& texts) = 0;

    /// Get the embedding dimension.
    [[nodiscard]] virtual size_t dimension() const = 0;

    [[nodiscard]] virtual std::string name() const = 0;
};

/// Rerank provider interface — reranks documents by relevance to query.
class RerankProvider {
public:
    virtual ~RerankProvider() = default;

    /// Rerank documents given a query.
    [[nodiscard]] virtual Result<std::vector<RankedResult>> rerank(
        const std::string& query,
        const std::vector<std::string>& documents,
        size_t top_k = 0) = 0;

    [[nodiscard]] virtual std::string name() const = 0;
};

// ── Provider Factory ────────────────────────────────────────────────────────

/// Factory function types
using LLMProviderFactory = std::function<std::unique_ptr<LLMProvider>(
    const std::string& model, const std::string& host, int port,
    const std::string& api_key, const std::string& base_url)>;

using EmbedProviderFactory = std::function<std::unique_ptr<EmbedProvider>(
    const std::string& model, const std::string& host, int port,
    const std::string& api_key, const std::string& base_url, size_t dimension)>;

using RerankProviderFactory = std::function<std::unique_ptr<RerankProvider>(
    const std::string& model, const std::string& host, int port,
    const std::string& api_key, const std::string& base_url)>;

/// Provider registry — singleton for runtime provider selection.
class ProviderRegistry {
public:
    static ProviderRegistry& instance();

    void registerLLM(const std::string& name, LLMProviderFactory factory);
    void registerEmbed(const std::string& name, EmbedProviderFactory factory);
    void registerRerank(const std::string& name, RerankProviderFactory factory);

    [[nodiscard]] std::unique_ptr<LLMProvider> createLLM(
        const std::string& name, const std::string& model,
        const std::string& host, int port,
        const std::string& api_key = "", const std::string& base_url = "") const;

    [[nodiscard]] std::unique_ptr<EmbedProvider> createEmbed(
        const std::string& name, const std::string& model,
        const std::string& host, int port,
        const std::string& api_key = "", const std::string& base_url = "",
        size_t dimension = 0) const;

    [[nodiscard]] std::unique_ptr<RerankProvider> createRerank(
        const std::string& name, const std::string& model,
        const std::string& host, int port,
        const std::string& api_key = "", const std::string& base_url = "") const;

private:
    ProviderRegistry() = default;
    std::unordered_map<std::string, LLMProviderFactory> llm_factories_;
    std::unordered_map<std::string, EmbedProviderFactory> embed_factories_;
    std::unordered_map<std::string, RerankProviderFactory> rerank_factories_;
};

}  // namespace amind
