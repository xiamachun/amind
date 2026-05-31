#include "provider.h"

#include <spdlog/spdlog.h>
#include <unordered_map>

namespace amind {

ProviderRegistry& ProviderRegistry::instance() {
    static ProviderRegistry registry;
    return registry;
}

void ProviderRegistry::registerLLM(const std::string& name, LLMProviderFactory factory) {
    llm_factories_[name] = std::move(factory);
    spdlog::info("Registered LLM provider: {}", name);
}

void ProviderRegistry::registerEmbed(const std::string& name, EmbedProviderFactory factory) {
    embed_factories_[name] = std::move(factory);
    spdlog::info("Registered Embed provider: {}", name);
}

void ProviderRegistry::registerRerank(const std::string& name, RerankProviderFactory factory) {
    rerank_factories_[name] = std::move(factory);
    spdlog::info("Registered Rerank provider: {}", name);
}

std::unique_ptr<LLMProvider> ProviderRegistry::createLLM(
    const std::string& name, const std::string& model,
    const std::string& host, int port,
    const std::string& api_key, const std::string& base_url) const {
    auto it = llm_factories_.find(name);
    if (it == llm_factories_.end()) {
        spdlog::error("Unknown LLM provider: {}", name);
        return nullptr;
    }
    return it->second(model, host, port, api_key, base_url);
}

std::unique_ptr<EmbedProvider> ProviderRegistry::createEmbed(
    const std::string& name, const std::string& model,
    const std::string& host, int port,
    const std::string& api_key, const std::string& base_url,
    size_t dimension) const {
    auto it = embed_factories_.find(name);
    if (it == embed_factories_.end()) {
        spdlog::error("Unknown Embed provider: {}", name);
        return nullptr;
    }
    return it->second(model, host, port, api_key, base_url, dimension);
}

std::unique_ptr<RerankProvider> ProviderRegistry::createRerank(
    const std::string& name, const std::string& model,
    const std::string& host, int port,
    const std::string& api_key, const std::string& base_url) const {
    auto it = rerank_factories_.find(name);
    if (it == rerank_factories_.end()) {
        spdlog::error("Unknown Rerank provider: {}", name);
        return nullptr;
    }
    return it->second(model, host, port, api_key, base_url);
}

}  // namespace amind
