#pragma once

#include "core/memory_record.h"
#include "core/result.h"
#include "provider/provider.h"

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace amind {

/// Operations the Reconciler may decide to perform on a new fact.
enum class ReconcileOp : uint8_t {
    ADD       = 0,   ///< new fact stands on its own; no conflict
    REPLACE   = 1,   ///< new fact supersedes target_id (correction/update)
    RETRACT   = 2,   ///< target_id is withdrawn; no new fact stored
    REINFORCE = 3,   ///< redundant with target_id; just bump its importance
    NOOP      = 4,   ///< new fact already covered; skip
};

const char* reconcileOpToString(ReconcileOp op);
ReconcileOp parseReconcileOp(const std::string& s);

/// Result of asking the Reconciler about one candidate.
struct ReconcileDecision {
    ReconcileOp op{ReconcileOp::ADD};
    uint64_t    target_id{0};       ///< for REPLACE/RETRACT/REINFORCE/NOOP
    std::string rationale;          ///< LLM's stated reason; for audit + UI
    bool        from_fallback{false}; ///< true when LLM call failed and we defaulted to ADD
};

/// Stateless reconciler — given a candidate fact and its strongly-similar
/// neighbours in the same namespace, ask an LLM what to do.
///
/// The Reconciler does NOT execute the op; it returns a decision. The caller
/// (capture pipeline) is responsible for applying the side effects, so all
/// MemoryStore/Graph mutations stay in one place.
class Reconciler {
public:
    struct Config {
        /// Cosine similarity above which a neighbour is "strongly similar"
        /// enough to warrant LLM reconciliation. Below this we ADD blindly.
        float similarity_floor{0.70f};
        /// How many top neighbours to feed the LLM.
        size_t max_neighbours{3};
        /// Truncate neighbour content to this many chars in the prompt to
        /// keep token budget bounded.
        size_t max_content_chars{280};
    };

    Reconciler(std::shared_ptr<LLMProvider> llm, Config config);
    explicit Reconciler(std::shared_ptr<LLMProvider> llm)
        : Reconciler(std::move(llm), Config{}) {}

    /// Decide what to do with `candidate` given `neighbours`.
    /// Neighbours must be sorted by similarity desc and already namespace-filtered.
    /// If neighbours is empty or all below the similarity floor, returns
    /// {ADD, 0, "no strong neighbours"} without calling the LLM.
    /// On LLM failure, returns {ADD, 0, ..., from_fallback=true}.
    ReconcileDecision decide(
        const std::string& candidate_content,
        const std::vector<std::pair<MemoryRecord, float>>& neighbours);

    struct Stats {
        uint64_t total_calls{0};
        uint64_t llm_invocations{0};      ///< when neighbours triggered LLM
        uint64_t llm_failures{0};          ///< LLM call failed → fallback to ADD
        uint64_t op_add{0};
        uint64_t op_replace{0};
        uint64_t op_retract{0};
        uint64_t op_reinforce{0};
        uint64_t op_noop{0};
    };
    Stats stats() const;

    struct LogEntry {
        uint64_t timestamp_ms{0};
        std::string candidate;
        ReconcileOp op{ReconcileOp::ADD};
        uint64_t target_id{0};
        uint64_t latency_ms{0};
        bool from_fallback{false};
    };

    /// Return most recent log entries (newest first), up to `limit`.
    std::vector<LogEntry> recentLog(size_t limit = 100) const;

    /// Build the prompt for one decision. Exposed for golden tests.
    static std::string buildPrompt(
        const std::string& candidate,
        const std::vector<std::pair<MemoryRecord, float>>& neighbours,
        size_t max_content_chars);

    /// Parse a JSON LLM response into a decision. Exposed for tests.
    /// Returns ADD on parse failure (logged).
    static ReconcileDecision parseResponse(const std::string& json_str);

private:
    std::shared_ptr<LLMProvider> llm_;
    Config config_;
    mutable std::mutex stats_mu_;
    Stats stats_;

    static constexpr size_t LOG_CAPACITY = 500;
    mutable std::mutex log_mu_;
    std::deque<LogEntry> log_;
    void appendLog(LogEntry entry);
};

}  // namespace amind
