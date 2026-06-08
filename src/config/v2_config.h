
#pragma once

#include <atomic>
#include <mutex>
#include <string>

namespace amind {

/// V2 feature gates — runtime switches for gradual rollout.
///
/// All V2 features start in shadow mode (compute but don't enforce).
/// After 2 weeks of observation, flip to enforce mode.
struct V2Config {
    // ── Feature gates ────────────────────────────────────────────────────

    /// WriteGate: quality filter before writing Derived memories.
    bool write_gate_enabled{false};

    /// Lineage propagation: cascade invalidation on remove().
    bool lineage_propagation_enabled{false};

    /// Forget score: compute forget_score for GC decisions.
    bool forget_score_enabled{false};

    /// HNSW compact: enable periodic compaction of soft-deleted nodes.
    bool hnsw_compact_enabled{false};

    /// Conflict resolver: auto-resolve ConflictsWith edges.
    bool conflict_resolver_enabled{false};

    /// Consolidation worker: periodic session-level consolidation.
    bool consolidation_enabled{false};

    /// Reconciler: LLM-decided ADD/REPLACE/RETRACT/REINFORCE/NOOP for
    /// derived facts that have strongly-similar neighbours in same namespace.
    /// Solves correction/update/retraction generically; see src/reconcile/.
    bool reconcile_enabled{false};

    /// Aggregate staleness filter: at recall time, drop list-style aggregate
    /// memories that are made stale by newer atomic facts in the same result
    /// set. Applied post-rank, storage untouched. See src/retrieval/staleness_filter.h.
    bool aggregate_staleness_filter_enabled{true};

    // ── Three-tier memory lifecycle gates ────────────────────────────────

    /// Agent store routing: route capture/recall to per-agent stores.
    /// Default true — this is the architectural foundation.
    bool agent_store_routing_enabled{true};

    /// Tiered decay: apply tier-differentiated decay multipliers in applyDecay().
    bool tiered_decay_enabled{false};

    /// Exponential decay: use e^(-t/S) instead of linear decay.
    bool exponential_decay_enabled{false};

    /// Access-driven promotion: promote memories on get() when thresholds are met.
    bool access_promotion_enabled{false};

    /// Recency gate: use multiplicative recency gating in recall scoring.
    bool recency_gate_enabled{false};

    /// Tier demotion: demote memories when importance drops below threshold.
    bool tier_demotion_enabled{false};

    // ── Shadow mode overrides ────────────────────────────────────────────

    /// When true, V2 features compute but don't enforce (shadow mode).
    /// Individual features may also have their own shadow mode flags.
    bool global_shadow_mode{true};
};

/// Thread-safe feature gate accessor.
class FeatureGate {
public:
    explicit FeatureGate(V2Config config = {});

    // ── Query gates ──────────────────────────────────────────────────────

    bool isWriteGateEnabled() const;
    bool isLineagePropagationEnabled() const;
    bool isForgetScoreEnabled() const;
    bool isHnswCompactEnabled() const;
    bool isConflictResolverEnabled() const;
    bool isConsolidationEnabled() const;
    bool isReconcileEnabled() const;
    bool isAggregateStalenessFilterEnabled() const;
    bool isAgentStoreRoutingEnabled() const;
    bool isTieredDecayEnabled() const;
    bool isExponentialDecayEnabled() const;
    bool isAccessPromotionEnabled() const;
    bool isRecencyGateEnabled() const;
    bool isTierDemotionEnabled() const;
    bool isGlobalShadowMode() const;

    /// Check if a named feature is enabled (for generic access).
    bool isEnabled(const std::string& feature_name) const;

    // ── Update gates ─────────────────────────────────────────────────────

    void setWriteGateEnabled(bool enabled);
    void setLineagePropagationEnabled(bool enabled);
    void setForgetScoreEnabled(bool enabled);
    void setHnswCompactEnabled(bool enabled);
    void setConflictResolverEnabled(bool enabled);
    void setConsolidationEnabled(bool enabled);
    void setReconcileEnabled(bool enabled);
    void setAgentStoreRoutingEnabled(bool enabled);
    void setTieredDecayEnabled(bool enabled);
    void setExponentialDecayEnabled(bool enabled);
    void setAccessPromotionEnabled(bool enabled);
    void setRecencyGateEnabled(bool enabled);
    void setTierDemotionEnabled(bool enabled);
    void setGlobalShadowMode(bool enabled);

    /// Enable all V2 features (for testing or full rollout).
    void enableAll();

    /// Disable all V2 features (emergency kill switch).
    void disableAll();

    // ── Config snapshot ──────────────────────────────────────────────────

    V2Config snapshot() const;

    /// Count of enabled features.
    int enabledCount() const;

private:
    mutable std::mutex mutex_;
    V2Config config_;
};

}  // namespace amind
