
#include "v2_config.h"

#include <spdlog/spdlog.h>

namespace amind {

FeatureGate::FeatureGate(V2Config config)
    : config_(std::move(config)) {}

// ── Query gates ──────────────────────────────────────────────────────────

bool FeatureGate::isWriteGateEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.write_gate_enabled;
}

bool FeatureGate::isLineagePropagationEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.lineage_propagation_enabled;
}

bool FeatureGate::isForgetScoreEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.forget_score_enabled;
}

bool FeatureGate::isHnswCompactEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.hnsw_compact_enabled;
}

bool FeatureGate::isConflictResolverEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.conflict_resolver_enabled;
}

bool FeatureGate::isConsolidationEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.consolidation_enabled;
}

bool FeatureGate::isReconcileEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.reconcile_enabled;
}

bool FeatureGate::isAggregateStalenessFilterEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.aggregate_staleness_filter_enabled;
}

bool FeatureGate::isGlobalShadowMode() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.global_shadow_mode;
}

bool FeatureGate::isEnabled(const std::string& feature_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (feature_name == "write_gate")            return config_.write_gate_enabled;
    if (feature_name == "lineage_propagation")   return config_.lineage_propagation_enabled;
    if (feature_name == "forget_score")          return config_.forget_score_enabled;
    if (feature_name == "hnsw_compact")          return config_.hnsw_compact_enabled;
    if (feature_name == "conflict_resolver")     return config_.conflict_resolver_enabled;
    if (feature_name == "consolidation")         return config_.consolidation_enabled;
    if (feature_name == "reconcile")             return config_.reconcile_enabled;
    if (feature_name == "aggregate_staleness_filter") return config_.aggregate_staleness_filter_enabled;
    return false;
}

// ── Update gates ─────────────────────────────────────────────────────────

void FeatureGate::setWriteGateEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.write_gate_enabled = enabled;
    spdlog::info("FeatureGate: write_gate = {}", enabled);
}

void FeatureGate::setLineagePropagationEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.lineage_propagation_enabled = enabled;
    spdlog::info("FeatureGate: lineage_propagation = {}", enabled);
}

void FeatureGate::setForgetScoreEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.forget_score_enabled = enabled;
    spdlog::info("FeatureGate: forget_score = {}", enabled);
}

void FeatureGate::setHnswCompactEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.hnsw_compact_enabled = enabled;
    spdlog::info("FeatureGate: hnsw_compact = {}", enabled);
}

void FeatureGate::setConflictResolverEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.conflict_resolver_enabled = enabled;
    spdlog::info("FeatureGate: conflict_resolver = {}", enabled);
}

void FeatureGate::setConsolidationEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.consolidation_enabled = enabled;
    spdlog::info("FeatureGate: consolidation = {}", enabled);
}

void FeatureGate::setReconcileEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.reconcile_enabled = enabled;
    spdlog::info("FeatureGate: reconcile = {}", enabled);
}

void FeatureGate::setGlobalShadowMode(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.global_shadow_mode = enabled;
    spdlog::info("FeatureGate: global_shadow_mode = {}", enabled);
}

void FeatureGate::enableAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.write_gate_enabled = true;
    config_.lineage_propagation_enabled = true;
    config_.forget_score_enabled = true;
    config_.hnsw_compact_enabled = true;
    config_.conflict_resolver_enabled = true;
    config_.consolidation_enabled = true;
    config_.reconcile_enabled = true;
    config_.global_shadow_mode = false;
    spdlog::info("FeatureGate: all V2 features enabled, shadow mode OFF");
}

void FeatureGate::disableAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.write_gate_enabled = false;
    config_.lineage_propagation_enabled = false;
    config_.forget_score_enabled = false;
    config_.hnsw_compact_enabled = false;
    config_.conflict_resolver_enabled = false;
    config_.consolidation_enabled = false;
    config_.reconcile_enabled = false;
    config_.global_shadow_mode = true;
    spdlog::info("FeatureGate: all V2 features disabled (emergency kill switch)");
}

// ── Config snapshot ──────────────────────────────────────────────────────

V2Config FeatureGate::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

int FeatureGate::enabledCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int count = 0;
    if (config_.write_gate_enabled) count++;
    if (config_.lineage_propagation_enabled) count++;
    if (config_.forget_score_enabled) count++;
    if (config_.hnsw_compact_enabled) count++;
    if (config_.conflict_resolver_enabled) count++;
    if (config_.consolidation_enabled) count++;
    if (config_.reconcile_enabled) count++;
    return count;
}

}  // namespace amind
