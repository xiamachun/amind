
#include "derived_extractor.h"

#include <spdlog/spdlog.h>

namespace amind {

DerivedExtractor::DerivedExtractor(WriteGate& gate, LineageIndex& lineage)
    : gate_(gate), lineage_(lineage) {}

std::vector<DerivedResult> DerivedExtractor::processFacts(
    uint64_t raw_memory_id,
    const std::string& agent_id,
    const std::string& user_id,
    const std::vector<DerivedCandidate>& candidates,
    const SimilaritySearchFunc& search_func,
    const StoreFunc& store_func) {

    std::vector<DerivedResult> results;
    results.reserve(candidates.size());

    for (const auto& candidate : candidates) {
        stats_.total_processed++;

        // Build a ProposedMemory for the gate
        ProposedMemory proposed;
        proposed.content = candidate.content;
        proposed.embedding = candidate.embedding;
        proposed.importance = candidate.importance;

        // Evaluate through WriteGate
        GateVerdict verdict = gate_.evaluate(proposed, search_func);

        // Downgrade Near-duplicate AND Low-marginal-value rejections to Deferred
        // so the Reconciler can decide REPLACE — a correction shares high
        // embedding similarity with the fact it corrects (e.g.
        // "用户的手机号是138" vs "用户的手机号是139"), which yields both
        // high near-duplicate similarity AND low marginal_value.
        if (verdict.decision == GateDecision::Rejected
            && (verdict.reason.find("Near-duplicate") != std::string::npos
                || verdict.reason.find("Low marginal value") != std::string::npos)) {
            spdlog::info("DerivedExtractor: downgrading rejection ({}) to deferred for derived fact '{}'",
                          verdict.reason.substr(0, 30),
                          candidate.content.substr(0, 60));
            verdict.decision = GateDecision::Deferred;
        }

        DerivedResult result;
        result.decision = verdict.decision;
        result.reason = verdict.reason;

        if (verdict.decision == GateDecision::Accepted
            || verdict.decision == GateDecision::Deferred) {
            // Build derived MemoryRecord. Deferred facts ALSO get persisted
            // (with the deferred flag in metadata) so the downstream Reconciler
            // can decide REPLACE/RETRACT instead of silently losing the fact.
            // WriteGate-Deferred typically means "looks like an update or
            // restatement" — exactly the case where Reconciler is most useful.
            MemoryRecord derived;
            derived.content = candidate.content;
            derived.agent_id = agent_id;
            derived.user_id = user_id;
            derived.scope = candidate.scope;
            derived.memory_type = candidate.memory_type;
            derived.tier = MemoryTier::Consolidated; // Derived facts are consolidated
            derived.layer = MemoryLayer::Derived;
            derived.source_tier = candidate.source_tier;
            derived.importance = candidate.importance;
            derived.confidence_level = (verdict.decision == GateDecision::Deferred)
                                       ? Confidence::Stale       // lower confidence for Deferred
                                       : Confidence::Inferred;
            derived.gate_decision = verdict.decision;
            derived.marginal_value = verdict.marginal_value;
            derived.parent_id = raw_memory_id;
            derived.embedding = candidate.embedding;
            // Inherit tenant audit trail (user_id, session_id, ...) from raw.
            derived.user_metadata = candidate.user_metadata;

            // Persist
            uint64_t derived_id = store_func(std::move(derived));
            result.derived_id = derived_id;

            // Record lineage: derived is child of raw
            lineage_.recordLineage(derived_id, {raw_memory_id}, LineageOp::Infer);

            spdlog::debug("DerivedExtractor: {} fact '{}' as derived {} from raw {}",
                          (verdict.decision == GateDecision::Accepted ? "accepted" : "deferred"),
                          candidate.content.substr(0, 50), derived_id, raw_memory_id);

            if (verdict.decision == GateDecision::Accepted) {
                stats_.accepted++;
            } else {
                stats_.deferred++;
            }

        } else {
            spdlog::debug("DerivedExtractor: rejected fact from raw {}: {}",
                          raw_memory_id, verdict.reason);
            stats_.rejected++;
        }

        results.push_back(std::move(result));
    }

    spdlog::info("DerivedExtractor: processed {} candidates for raw {}: {} accepted, {} rejected, {} deferred",
                 candidates.size(), raw_memory_id, stats_.accepted, stats_.rejected, stats_.deferred);

    return results;
}

DerivedExtractor::Stats DerivedExtractor::stats() const {
    return stats_;
}

void DerivedExtractor::resetStats() {
    stats_ = {};
}

}  // namespace amind
