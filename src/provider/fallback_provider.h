#pragma once

#include "provider.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace amind {

/// Multi-level failover LLM provider.
/// Wraps N LLMProvider instances (same model, different API keys or hosts).
/// On failure, cascades to the next level. Per-level circuit breaker skips
/// unhealthy levels for a cooldown period before retrying.
class FallbackLLMProvider : public LLMProvider {
public:
    struct Level {
        std::shared_ptr<LLMProvider> provider;
        std::string label;  // e.g. "L1", "L2"
    };

    explicit FallbackLLMProvider(std::vector<Level> levels,
                                  size_t circuit_threshold = 3,
                                  std::chrono::seconds cooldown = std::chrono::seconds(30));

    Result<std::string> generate(
        const std::string& prompt,
        const std::string& system_prompt = "") override;

    Result<std::string> generateJson(
        const std::string& prompt,
        const std::string& system_prompt = "") override;

    std::string name() const override { return "fallback"; }

    /// Check if all levels are tripped (total blackout).
    bool allCircuitsOpen() const;

private:
    struct CircuitState {
        std::atomic<size_t> consecutive_failures{0};
        std::atomic<int64_t> tripped_at_epoch_s{0};  // 0 = not tripped
    };

    bool isOpen(size_t level_idx) const;
    void recordSuccess(size_t level_idx);
    void recordFailure(size_t level_idx);

    std::vector<Level> levels_;
    std::vector<CircuitState> circuits_;
    size_t circuit_threshold_;
    std::chrono::seconds cooldown_;
};

}  // namespace amind
