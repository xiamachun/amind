#pragma once

#include "core/memory_record.h"
#include "core/result.h"
#include "provider/provider.h"
#include "memory/memory_store.h"
#include "async/task_queue.h"
#include "graph/graph_store.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace amind {

// Forward declarations for V2 optional dependencies (lifetime managed by Engine)
class FeatureGate;
class DerivedExtractor;
class WriteGate;
// GateLog removed in Phase 4.
class MemoryEventLog;
class Reconciler;

/// Fact extracted from conversation by LLM.
struct ExtractedFact {
    std::string content;
    MemoryScope scope{MemoryScope::Private};
    MemoryType memory_type{MemoryType::Ephemeral};
    Confidence confidence{Confidence::Inferred};
    float importance{0.5f};
    std::vector<std::string> entities;
};

/// Result of LLM-based fact extraction, including intent classification.
struct ExtractionResult {
    std::vector<ExtractedFact> facts;
    bool is_forget_request{false};
    bool is_confidential{false};
};

/// The capture pipeline processes incoming conversations and extracts memories.
/// Two stages:
///   Stage 1 (sync):  Raw content → embed → fast store → return ID
///   Stage 2 (async): LLM extract facts → dedup → conflict detect → graph link
class CapturePipeline {
public:
    CapturePipeline(MemoryStore& store, GraphStore& graph,
                    TaskQueue& queue,
                    std::shared_ptr<LLMProvider> llm,
                    std::shared_ptr<EmbedProvider> embedder,
                    FeatureGate* feature_gate = nullptr,
                    DerivedExtractor* derived_extractor = nullptr,
                    WriteGate* write_gate = nullptr);

    /// Stage 1: Fast capture — embed content, store immediately.
    /// Optional user_metadata is persisted under the record's user_metadata
    /// map and exposed back to clients via recall/get responses.
    /// When pre_extracted=true, Stage 2 skips the LLM extractFacts() call
    /// (content is already a clean fact from SessionManager or external caller).
    Result<std::vector<uint64_t>> capture(
        const std::string& content,
        const std::string& agent_id,
        const std::string& user_id,
        MemoryScope scope = MemoryScope::Private,
        MemoryType memory_type = MemoryType::Ephemeral,
        std::map<std::string, std::string> user_metadata = {},
        bool pre_extracted = false);

    /// Stage 2: Schedule async refinement for a memory.
    /// `trace_id` carries the observability trace across the Stage 1 → Stage 2
    /// boundary so all emitted events (Embed/Gate/Derive/Reconcile) group
    /// together. 0 = no trace context.
    void scheduleRefinement(uint64_t memory_id, 
                            bool pre_extracted = false,
                            uint64_t trace_id = 0);

    /// Intercept: capture from LLM conversation messages.
    Result<std::vector<uint64_t>> interceptCapture(
        const std::vector<std::pair<std::string, std::string>>& messages,
        const std::string& agent_id,
        const std::string& user_id);

    /// Inject the unified MemoryEventLog. When set, the capture pipeline
    /// emits Gate / Reconcile / Store / Derive events into it as the single
    /// source of truth for observability. Phase 4 will remove setGateLog().
    void setEventsLog(MemoryEventLog* el) { events_log_ = el; }

    /// Inject the Reconciler. When set, derived facts get LLM-decided
    /// ADD/REPLACE/RETRACT/REINFORCE/NOOP instead of unconditional ADD.
    void setReconciler(Reconciler* reconciler) { reconciler_ = reconciler; }

private:
    /// LLM-based fact extraction.
    Result<ExtractionResult> extractFacts(const std::string& content);

    /// Deduplicate against existing memories.
    bool isDuplicate(const std::vector<float>& embedding, float threshold = 0.95f);

    MemoryStore& store_;
    GraphStore& graph_;
    TaskQueue& queue_;
    std::shared_ptr<LLMProvider> llm_;
    std::shared_ptr<EmbedProvider> embedder_;

    // V2 optional dependencies (nullptr when V2 not active)
    FeatureGate* feature_gate_{nullptr};
    DerivedExtractor* derived_extractor_{nullptr};
    WriteGate* write_gate_{nullptr};
    MemoryEventLog* events_log_{nullptr};
    Reconciler* reconciler_{nullptr};
    std::atomic<bool> alive_{true};

    // ── Per-key reconcile serialization ──────────────────────────────────
    // When multiple Stage 2 tasks target the same existing memory for
    // REPLACE/RETRACT, they must execute sequentially to avoid stale-read
    // races (e.g. 5 rapid weight updates all seeing the same "old" value).
    // Key = the strongest neighbour's memory_id (i.e. the likely REPLACE target).
    // Only REPLACE/RETRACT paths contend; unrelated ADD tasks run fully parallel.
    mutable std::mutex reconcile_slots_mu_;
    std::unordered_map<uint64_t, std::unique_ptr<std::mutex>> reconcile_slots_;

    /// Get or create the per-target mutex for reconcile serialization.
    std::mutex& getReconcileSlotLock(uint64_t target_memory_id);

    // ── Freshness barrier ─────────────────────────────────────────────────
    // Per-namespace counter of in-flight Stage 2 refinement tasks.
    // RetrievalPipeline can call waitForPendingRefinements() before recall
    // to ensure any recently-written memories have been fully reconciled.
    struct NamespaceFlightInfo {
        int pending{0};
        std::chrono::steady_clock::time_point last_submit{};
    };
    mutable std::mutex flight_mu_;
    mutable std::condition_variable flight_cv_;
    std::unordered_map<uint64_t, NamespaceFlightInfo> ns_in_flight_;

    void incrementPending(uint64_t ns_hash);
    void decrementPending(uint64_t ns_hash);

public:
    /// Block until all in-flight Stage 2 tasks for the given namespace finish,
    /// or until timeout_ms elapses. Returns true if barrier was satisfied,
    /// false on timeout.  Called by RetrievalPipeline before recall.
    bool waitForPendingRefinements(uint64_t ns_hash,
                                   std::chrono::milliseconds timeout = std::chrono::milliseconds(3000)) const;

    /// Check if there are pending refinements for a namespace that were
    /// submitted within the last `recency` window.
    bool hasFreshPending(uint64_t ns_hash,
                         std::chrono::milliseconds recency = std::chrono::milliseconds(5000)) const;

    ~CapturePipeline() { alive_ = false; }
};

}  // namespace amind
