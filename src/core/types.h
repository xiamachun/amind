#pragma once

#include <cstdint>
#include <string>

namespace amind {

/// Memory ownership model — classifies memories by who they belong to,
/// not by cognitive category. This is the fundamental axis for isolation
/// and access control.
enum class MemoryOwner : uint8_t {
    User = 0,      // Belongs to the human user (preferences, habits, personal info)
    Project = 1,   // Belongs to the project/codebase (architecture, conventions)
    Agent = 2,     // Belongs to the AI agent itself (lessons learned, skills)
    Session = 3,   // Temporary, belongs to current conversation session
    Shared = 4,    // Cross-agent team knowledge (shared decisions, standards)
};

/// Memory lifecycle phases — every memory transitions through these states.
/// Active → Versioned (when updated) or Active → Archived (cold) or Active → Tombstone (deleted)
enum class MemoryPhase : uint8_t {
    Active = 0,     // Live, searchable, current version
    Versioned = 1,  // Has newer version, kept as history
    Archived = 2,   // Low-frequency, moved to cold storage
    Tombstone = 3,  // Marked for deletion (soft delete)
    Invalidated = 4, // V2: Lineage orphan — all upstream evidence deleted
};

/// Confidence level — indicates how trustworthy a memory is.
/// This is a key differentiator: agents should know when their memories might be wrong.
enum class Confidence : uint8_t {
    Verified = 0,    // User explicitly confirmed this memory
    Inferred = 1,    // Agent extracted/inferred from conversation
    Stale = 2,       // Possibly outdated (exceeded stale_threshold_hours)
    Conflicted = 3,  // Contradicts another memory
};

/// Edge types for the memory relationship graph.
/// Used by GraphStore to build the knowledge graph linking memories.
enum class EdgeType : uint8_t {
    Related = 0,       // Semantic similarity link
    Caused = 1,        // A caused B (temporal causation)
    Contradicts = 2,   // A contradicts B (conflict)
    Supersedes = 3,    // A supersedes B (newer version)
    DerivedFrom = 4,   // A was derived from B (extraction lineage)
    SameSession = 5,   // A and B from same conversation session
    Corrects = 6,      // A is a correction of B (user feedback)
    Prerequisite = 7,  // A is prerequisite for understanding B
    Temporal = 8,      // A and B close in time
    Entity = 9,        // A and B share a named entity
    ConflictsWith = 10, // Explicit conflict edge (created by conflict detector)
};

// ── V2 Enumerations ─────────────────────────────────────────────────────

/// Memory storage layer — physical separation of raw evidence vs derived interpretations.
enum class MemoryLayer : uint8_t {
    Raw = 0,       // Original user utterance, tool output, environment observation (immutable)
    Derived = 1,   // LLM-extracted fact, summary, belief, preference (can be invalidated/recomputed)
};

/// Lineage operation — how a derived memory was produced from its parents.
enum class LineageOp : uint8_t {
    None = 0,       // No lineage (Raw layer memories)
    Summarize = 1,  // Condensed from parent(s)
    Aggregate = 2,  // Merged multiple parents
    Infer = 3,      // LLM inference/extraction
    Distill = 4,    // Skill distillation (V4)
};

/// Source tier — how the information was originally obtained.
/// Behavioral > Assertion > Inference (behavioral evidence is hardest to fake).
enum class SourceTier : uint8_t {
    Assertion = 0,   // User explicitly stated ("I prefer dark mode")
    Behavioral = 1,  // Observed from repeated actions (chose dark mode 5 times)
    Inference = 2,   // Agent inferred from context
};

/// Write gate decision — whether a candidate memory passes the quality gate.
enum class GateDecision : uint8_t {
    Accepted = 0,   // Passes gate, will be stored
    Rejected = 1,   // Fails gate (duplicate, low quality, noise)
    Deferred = 2,   // Uncertain, queued for more evidence
};

/// Reason for memory removal — required by V2 remove(id, reason) signature.
enum class RemoveReason : uint8_t {
    UserDelete = 0,        // User explicitly requested deletion
    LineageOrphan = 1,     // All upstream lineage parents were deleted/invalidated
    GcArchive = 2,         // GC Worker scored → Archive
    GcTombstone = 3,       // GC Worker scored → Tombstone
    ConflictLoser = 4,     // Lost in conflict resolution
};

// ── V2 String conversions ───────────────────────────────────────────────

inline std::string layerToString(MemoryLayer layer) {
    switch (layer) {
        case MemoryLayer::Raw:     return "Raw";
        case MemoryLayer::Derived: return "Derived";
    }
    return "Unknown";
}

inline std::string lineageOpToString(LineageOp op) {
    switch (op) {
        case LineageOp::None:      return "None";
        case LineageOp::Summarize: return "Summarize";
        case LineageOp::Aggregate: return "Aggregate";
        case LineageOp::Infer:     return "Infer";
        case LineageOp::Distill:   return "Distill";
    }
    return "Unknown";
}

inline std::string sourceTierToString(SourceTier tier) {
    switch (tier) {
        case SourceTier::Assertion:  return "Assertion";
        case SourceTier::Behavioral: return "Behavioral";
        case SourceTier::Inference:  return "Inference";
    }
    return "Unknown";
}

inline std::string gateDecisionToString(GateDecision decision) {
    switch (decision) {
        case GateDecision::Accepted: return "Accepted";
        case GateDecision::Rejected: return "Rejected";
        case GateDecision::Deferred: return "Deferred";
    }
    return "Unknown";
}

inline std::string removeReasonToString(RemoveReason reason) {
    switch (reason) {
        case RemoveReason::UserDelete:     return "UserDelete";
        case RemoveReason::LineageOrphan:  return "LineageOrphan";
        case RemoveReason::GcArchive:      return "GcArchive";
        case RemoveReason::GcTombstone:    return "GcTombstone";
        case RemoveReason::ConflictLoser:  return "ConflictLoser";
    }
    return "Unknown";
}

/// Bit flags for the MemoryRecord header.
/// Packed into a uint16_t flags field for compact on-disk representation.
struct RecordFlags {
    static constexpr uint16_t ALIVE       = 1 << 0;  // Record is active
    static constexpr uint16_t VERSIONED   = 1 << 1;  // Record has been superseded by newer version
    static constexpr uint16_t ARCHIVED    = 1 << 2;  // Record moved to cold storage
    static constexpr uint16_t TOMBSTONE   = 1 << 3;  // Record marked for deletion
    static constexpr uint16_t HAS_EMBEDDING = 1 << 4; // Record has embedding vector
    static constexpr uint16_t LLM_REFINED = 1 << 5;  // Stage 2 refinement completed
    static constexpr uint16_t HNSW_EVICTED   = 1 << 6;  // V2: Evicted from HNSW hot layer
    static constexpr uint16_t LINEAGE_ORPHAN = 1 << 7;  // V2: All upstream lineage parents invalidated
};

// ── String conversions ──────────────────────────────────────────────────────

inline std::string ownerToString(MemoryOwner owner) {
    switch (owner) {
        case MemoryOwner::User:    return "User";
        case MemoryOwner::Project: return "Project";
        case MemoryOwner::Agent:   return "Agent";
        case MemoryOwner::Session: return "Session";
        case MemoryOwner::Shared:  return "Shared";
    }
    return "Unknown";
}

inline std::string phaseToString(MemoryPhase phase) {
    switch (phase) {
        case MemoryPhase::Active:      return "Active";
        case MemoryPhase::Versioned:   return "Versioned";
        case MemoryPhase::Archived:    return "Archived";
        case MemoryPhase::Tombstone:   return "Tombstone";
        case MemoryPhase::Invalidated: return "Invalidated";
    }
    return "Unknown";
}

inline std::string confidenceToString(Confidence conf) {
    switch (conf) {
        case Confidence::Verified:   return "Verified";
        case Confidence::Inferred:   return "Inferred";
        case Confidence::Stale:      return "Stale";
        case Confidence::Conflicted: return "Conflicted";
    }
    return "Unknown";
}

inline std::string edgeTypeToString(EdgeType type) {
    switch (type) {
        case EdgeType::Related:       return "Related";
        case EdgeType::Caused:        return "Caused";
        case EdgeType::Contradicts:   return "Contradicts";
        case EdgeType::Supersedes:    return "Supersedes";
        case EdgeType::DerivedFrom:   return "DerivedFrom";
        case EdgeType::SameSession:   return "SameSession";
        case EdgeType::Corrects:      return "Corrects";
        case EdgeType::Prerequisite:  return "Prerequisite";
        case EdgeType::Temporal:      return "Temporal";
        case EdgeType::Entity:        return "Entity";
        case EdgeType::ConflictsWith: return "ConflictsWith";
    }
    return "Unknown";
}

}  // namespace amind
