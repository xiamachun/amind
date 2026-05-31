#pragma once

#include "provider.h"
#include "connection_pool.h"

#include <memory>
#include <string>
#include <vector>

namespace amind {

class AnthropicLLM : public LLMProvider {
public:
    AnthropicLLM(std::string model, std::string host, int port,
                 std::string api_key, std::string base_path,
                 std::shared_ptr<HttpConnectionPool> pool = nullptr,
                 std::vector<std::pair<std::string, std::string>> extra_headers = {});

    Result<std::string> generate(
        const std::string& prompt,
        const std::string& system_prompt = "") override;

    Result<std::string> generateJson(
        const std::string& prompt,
        const std::string& system_prompt = "") override;

    std::string name() const override { return "anthropic"; }

private:
    Result<std::string> callMessages(const std::string& prompt,
                                      const std::string& system_prompt,
                                      bool json_mode);

    std::string model_;
    std::string host_;
    int port_;
    std::string api_key_;
    std::string base_path_;
    std::shared_ptr<HttpConnectionPool> pool_;
    std::vector<std::pair<std::string, std::string>> extra_headers_;
};

void registerAnthropicProviders();

}  // namespace amind
