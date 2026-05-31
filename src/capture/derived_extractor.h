
#pragma once

#include "core/memory_record.h"
#include "core/types.h"
#include "gate/write_gate.h"
#include "lineage/lineage_index.h"

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace amind {

/// A fact extracted from raw content, ready for gate evaluation.
struct DerivedCandidate {
    std::string content;
    float importance{0.5f};
    SourceTier source_tier{SourceTier::Inference};
    std::vector<float> embedding;  // pre-computed if available
    /// Optional user_metadata inherited from the raw memory so the derived
    /// record carries the same tenant audit trail (user_id, session_id, ...).
    std::map<std::string, std::string> user_metadata;
    /// Owner inherited from the raw memory. A fact derived from a User-owned
    /// raw memory is still about the user, not about the AI itself —
    /// MemoryOwner::Agent is reserved for the AI's own self-knowledge.
    MemoryOwner owner{MemoryOwner::User};
};

/// Result of processing one derived candidate through the gate.
struct DerivedResult {
    uint64_t derived_id{0};   // 0 if rejected/deferred
    GateDecision decision;
    std::string reason;
};

/// Callback to persist a derived MemoryRecord and return its ID.
using StoreFunc = std::function<uint64_t(MemoryRecord)>;

/// DerivedExtractor — bridges Stage 2's extractFacts → WriteGate → Lineage.
///
/// Responsibilities:
/// 1. Take a raw memory ID and its extracted facts
/// 2. Run each fact through WriteGate
/// 3. Store accepted facts as Derived-layer records
/// 4. Record lineage (raw → derived) in LineageIndex
class DerivedExtractor {
public:
    DerivedExtractor(WriteGate& gate, LineageIndex& lineage);

    /// Process a batch of extracted facts for a given raw memory.
    /// Returns per-fact results (accepted/rejected/deferred).
    std::vector<DerivedResult> processFacts(
        uint64_t raw_memory_id,
        const std::string& namespace_,
        const std::vector<DerivedCandidate>& candidates,
        const SimilaritySearchFunc& search_func,
        const StoreFunc& store_func);

    struct Stats {
        uint64_t total_processed{0};
        uint64_t accepted{0};
        uint64_t rejected{0};
        uint64_t deferred{0};
    };

    Stats stats() const;
    void resetStats();

private:
    WriteGate& gate_;
    LineageIndex& lineage_;
    Stats stats_;
};

}  // namespace amind
