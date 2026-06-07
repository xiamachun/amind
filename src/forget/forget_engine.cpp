
#include "forget_engine.h"

#include <algorithm>
#include <spdlog/spdlog.h>

namespace amind {

ForgetEngine::ForgetEngine(ForgetConfig config)
    : config_(std::move(config)) {}

ForgetEngine::~ForgetEngine() = default;

float ForgetEngine::computeForgetScore(const ForgetSignals& signals) const {
    ForgetConfig cfg;
    { std::lock_guard<std::mutex> lock(mutex_); cfg = config_; }

    float score = cfg.weight_staleness     * signals.staleness
                + cfg.weight_low_access    * signals.low_access
                + cfg.weight_low_importance * signals.low_importance
                + cfg.weight_conflict_penalty * signals.conflict_penalty
                + cfg.weight_redundancy    * signals.redundancy
                - cfg.weight_verified_bonus * signals.verified_bonus
                - cfg.weight_graph_centrality * signals.graph_centrality;

    return std::clamp(score, 0.0f, 1.0f);
}

GcEvaluation ForgetEngine::evaluate(uint64_t memory_id,
                                     const ForgetSignals& signals) const {
    ForgetConfig cfg;
    { std::lock_guard<std::mutex> lock(mutex_); cfg = config_; }

    GcEvaluation eval;
    eval.memory_id = memory_id;
    eval.forget_score = computeForgetScore(signals);

    if (eval.forget_score >= cfg.tombstone_threshold) {
        eval.recommended_action = ForgetLogEntry::Decision::Tombstone;
        eval.reason = "forget_score >= " + std::to_string(cfg.tombstone_threshold);
    } else if (eval.forget_score >= cfg.archive_threshold) {
        eval.recommended_action = ForgetLogEntry::Decision::Archive;
        eval.reason = "forget_score >= " + std::to_string(cfg.archive_threshold);
    } else if (eval.forget_score >= cfg.decay_threshold) {
        eval.recommended_action = ForgetLogEntry::Decision::Decay;
        eval.reason = "forget_score >= " + std::to_string(cfg.decay_threshold);
    } else {
        eval.recommended_action = ForgetLogEntry::Decision::Decay;
        eval.reason = "below all thresholds, no action needed";
    }

    return eval;
}

std::vector<GcEvaluation> ForgetEngine::runCycle(
    const std::vector<std::pair<uint64_t, ForgetSignals>>& batch) const {

    bool shadow;
    { std::lock_guard<std::mutex> lock(mutex_); shadow = config_.shadow_mode; }

    std::vector<GcEvaluation> results;
    results.reserve(batch.size());

    for (const auto& [memory_id, signals] : batch) {
        results.push_back(evaluate(memory_id, signals));
    }

    // Sort by forget_score descending (highest risk first)
    std::sort(results.begin(), results.end(),
              [](const GcEvaluation& a, const GcEvaluation& b) {
                  return a.forget_score > b.forget_score;
              });

    spdlog::info("ForgetEngine::runCycle: evaluated {} memories, shadow_mode={}",
                 batch.size(), shadow);

    return results;
}

// logEntry / getLog / recentEntries / clearLog / logSize removed in Phase 4.
// GC decisions are emitted as MemoryEvent{kind=Gc*} from runForgetCycleOnce()
// in src/server/engine.cpp instead.

// ── Config ───────────────────────────────────────────────────────────────

void ForgetEngine::setShadowMode(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.shadow_mode = enabled;
    spdlog::info("ForgetEngine: shadow mode {}", enabled ? "enabled" : "disabled");
}

bool ForgetEngine::isShadowMode() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.shadow_mode;
}

ForgetConfig ForgetEngine::config() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

}  // namespace amind
