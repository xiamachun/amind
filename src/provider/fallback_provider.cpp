#include "fallback_provider.h"

#include <spdlog/spdlog.h>

namespace amind {

FallbackLLMProvider::FallbackLLMProvider(std::vector<Level> levels,
                                          size_t circuit_threshold,
                                          std::chrono::seconds cooldown)
    : levels_(std::move(levels)),
      circuits_(levels_.size()),
      circuit_threshold_(circuit_threshold),
      cooldown_(cooldown) {}

bool FallbackLLMProvider::isOpen(size_t idx) const {
    auto tripped = circuits_[idx].tripped_at_epoch_s.load(std::memory_order_relaxed);
    if (tripped == 0) return false;
    auto now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    return (now - tripped) < cooldown_.count();
}

void FallbackLLMProvider::recordSuccess(size_t idx) {
    circuits_[idx].consecutive_failures.store(0, std::memory_order_relaxed);
    circuits_[idx].tripped_at_epoch_s.store(0, std::memory_order_relaxed);
}

void FallbackLLMProvider::recordFailure(size_t idx) {
    auto failures = circuits_[idx].consecutive_failures.fetch_add(1, std::memory_order_relaxed) + 1;
    if (failures >= circuit_threshold_) {
        auto now = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        circuits_[idx].tripped_at_epoch_s.store(now, std::memory_order_relaxed);
        spdlog::warn("FallbackLLM: level {} ({}) circuit OPEN after {} failures, cooldown {}s",
                     idx, levels_[idx].label, failures, cooldown_.count());
    }
}

bool FallbackLLMProvider::allCircuitsOpen() const {
    for (size_t i = 0; i < levels_.size(); ++i) {
        if (!isOpen(i)) return false;
    }
    return true;
}

Result<std::string> FallbackLLMProvider::generate(
    const std::string& prompt, const std::string& system_prompt) {

    Error last_err;
    for (size_t i = 0; i < levels_.size(); ++i) {
        if (isOpen(i)) continue;

        auto result = levels_[i].provider->generate(prompt, system_prompt);
        if (result.ok()) {
            recordSuccess(i);
            return result;
        }
        recordFailure(i);
        last_err = result.error();
        spdlog::warn("FallbackLLM: level {} ({}) failed: {}, trying next...",
                     i, levels_[i].label, last_err.toString());
    }

    // All levels failed or circuit-open — try the first tripped level as last resort
    for (size_t i = 0; i < levels_.size(); ++i) {
        if (!isOpen(i)) continue;
        spdlog::warn("FallbackLLM: all levels exhausted, forcing retry on level {} ({})",
                     i, levels_[i].label);
        auto result = levels_[i].provider->generate(prompt, system_prompt);
        if (result.ok()) {
            recordSuccess(i);
            return result;
        }
        last_err = result.error();
        break;
    }

    return last_err;
}

Result<std::string> FallbackLLMProvider::generateJson(
    const std::string& prompt, const std::string& system_prompt) {

    Error last_err;
    for (size_t i = 0; i < levels_.size(); ++i) {
        if (isOpen(i)) continue;

        auto result = levels_[i].provider->generateJson(prompt, system_prompt);
        if (result.ok()) {
            recordSuccess(i);
            return result;
        }
        recordFailure(i);
        last_err = result.error();
        spdlog::warn("FallbackLLM: level {} ({}) generateJson failed: {}, trying next...",
                     i, levels_[i].label, last_err.toString());
    }

    // Last resort
    for (size_t i = 0; i < levels_.size(); ++i) {
        if (!isOpen(i)) continue;
        auto result = levels_[i].provider->generateJson(prompt, system_prompt);
        if (result.ok()) {
            recordSuccess(i);
            return result;
        }
        last_err = result.error();
        break;
    }

    return last_err;
}

}  // namespace amind
