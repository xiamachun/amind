#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace amind {

/// Unified event type. Replaces the per-subsystem log entry types
/// (GateLogEntry, ForgetLogEntry, StaleFilterEvent, ReconcileLogEntry).
///
/// One MemoryEvent = one Span in the distributed-tracing model:
///   - event_id           → span ID
///   - parent_event_id    → parent span (0 = root of trace)
///   - trace_id           → groups events from the same execution chain
///                          (one Stage 2 pipeline, one recall, one GC cycle, ...)
///   - memory_id          → primary business anchor (0 if not memory-scoped)
///   - kind               → what kind of step this is
///   - status             → outcome
///   - attrs              → free-form key/value bag for subsystem-specific
///                          fields (marginal_value, forget_score, witness_ids, ...)
///
/// See docs/arch/统一可观测层重构-MemoryEvent.md for the full design.
enum class EventKind : uint8_t {
    // Write path
    Store,              // Stage 1 fastStore
    Embed,              // EmbedProvider call
    Gate,               // WriteGate verdict (Accept/Reject/Defer)
    Derive,             // DerivedExtractor split a Raw memory into Derived facts
    Reconcile,          // Reconciler LLM verdict (ADD/REPLACE/RETRACT/REINFORCE/NOOP)
    LineagePropagate,   // LineageIndex cascade invalidation
    GraphEdge,          // Wrote a Supersedes/ConflictsWith/Derives edge
    VersionUpdate,      // New version created in the parent_id chain

    // Read path
    Recall,             // One recall call (trace root)
    RecallSemantic,     // HNSW returned N candidates
    RecallFilter,       // MMR / Supersedes / Old-marker filter decision
    RecallStale,        // Aggregate-staleness filter dropped a stale aggregate

    // Maintenance
    GcDecay,
    GcArchive,
    GcTombstone,
    GcVacuum,           // LSM compaction physical delete
    Consolidate,        // Cross-session dedup or drift check
    Resurrect,          // Manual revive of a Rejected/Deferred memory

    // Infra
    ProviderCall,       // LLM/Embed/Rerank HTTP call (latency/tokens)
    Error,              // Any-stage failure
};

enum class EventStatus : uint8_t {
    Ok,
    Rejected,
    Deferred,
    Failed,
    NoOp,
};

struct MemoryEvent {
    uint64_t event_id{0};         // snowflake, globally unique
    uint64_t parent_event_id{0};  // 0 = root of trace
    uint64_t trace_id{0};         // shared by all events in same chain
    uint64_t memory_id{0};        // 0 = not memory-scoped
    uint64_t timestamp_ms{0};
    uint32_t duration_ms{0};      // span duration; 0 = instantaneous
    std::string agent_id;         // agent isolation key
    EventKind kind{EventKind::Store};
    EventStatus status{EventStatus::Ok};
    std::string summary;          // ≤80 chars, list-view friendly
    std::map<std::string, std::string> attrs;
};

// ── Enum string converters (used in JSON serialization + REST API) ─────────

std::string kindToString(EventKind k);
EventKind   kindFromString(const std::string& s);

std::string statusToString(EventStatus s);
EventStatus statusFromString(const std::string& s);

/// Current wall-clock milliseconds since unix epoch.
uint64_t nowMs();

/// UTF-8 safe byte-budget truncator. Trims `s` to at most `max_bytes`,
/// snapping back to the previous complete code-point boundary so the result
/// is always valid UTF-8 and never throws when serialized via nlohmann::json.
/// Appends "…" if any bytes were dropped.
std::string truncateUtf8(const std::string& s, size_t max_bytes);

}  // namespace amind
