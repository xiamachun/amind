
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

bool FeatureGate::isAgentStoreRoutingEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.agent_store_routing_enabled;
}

bool FeatureGate::isTieredDecayEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.tiered_decay_enabled;
}

bool FeatureGate::isExponentialDecayEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.exponential_decay_enabled;
}

bool FeatureGate::isAccessPromotionEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.access_promotion_enabled;
}

bool FeatureGate::isRecencyGateEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.recency_gate_enabled;
}

bool FeatureGate::isTierDemotionEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.tier_demotion_enabled;
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
    if (feature_name == "agent_store_routing")   return config_.agent_store_routing_enabled;
    if (feature_name == "tiered_decay")          return config_.tiered_decay_enabled;
    if (feature_name == "exponential_decay")     return config_.exponential_decay_enabled;
    if (feature_name == "access_promotion")      return config_.access_promotion_enabled;
    if (feature_name == "recency_gate")          return config_.recency_gate_enabled;
    if (feature_name == "tier_demotion")         return config_.tier_demotion_enabled;
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

void FeatureGate::setAgentStoreRoutingEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.agent_store_routing_enabled = enabled;
    spdlog::info("FeatureGate: agent_store_routing = {}", enabled);
}

void FeatureGate::setTieredDecayEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.tiered_decay_enabled = enabled;
    spdlog::info("FeatureGate: tiered_decay = {}", enabled);
}

void FeatureGate::setExponentialDecayEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.exponential_decay_enabled = enabled;
    spdlog::info("FeatureGate: exponential_decay = {}", enabled);
}

void FeatureGate::setAccessPromotionEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.access_promotion_enabled = enabled;
    spdlog::info("FeatureGate: access_promotion = {}", enabled);
}

void FeatureGate::setRecencyGateEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.recency_gate_enabled = enabled;
    spdlog::info("FeatureGate: recency_gate = {}", enabled);
}

void FeatureGate::setTierDemotionEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.tier_demotion_enabled = enabled;
    spdlog::info("FeatureGate: tier_demotion = {}", enabled);
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
    config_.agent_store_routing_enabled = true;
    config_.tiered_decay_enabled = true;
    config_.exponential_decay_enabled = true;
    config_.access_promotion_enabled = true;
    config_.recency_gate_enabled = true;
    config_.tier_demotion_enabled = true;
    config_.global_shadow_mode = false;
    spdlog::info("FeatureGate: all features enabled, shadow mode OFF");
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
    // agent_store_routing stays true — it's architectural foundation
    config_.tiered_decay_enabled = false;
    config_.exponential_decay_enabled = false;
    config_.access_promotion_enabled = false;
    config_.recency_gate_enabled = false;
    config_.tier_demotion_enabled = false;
    config_.global_shadow_mode = true;
    spdlog::info("FeatureGate: all features disabled (emergency kill switch)");
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
    if (config_.agent_store_routing_enabled) count++;
    if (config_.tiered_decay_enabled) count++;
    if (config_.exponential_decay_enabled) count++;
    if (config_.access_promotion_enabled) count++;
    if (config_.recency_gate_enabled) count++;
    if (config_.tier_demotion_enabled) count++;
    return count;
}

}  // namespace amind
