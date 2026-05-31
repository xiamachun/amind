#pragma once

#include "core/memory_record.h"
#include "core/snowflake.h"
#include "gate/write_gate.h"

#include <cstddef>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace amind {

/// One persisted record of a WriteGate decision.
struct GateLogEntry {
    uint64_t entry_id{0};                     // Snowflake; unique
    int64_t  timestamp_ms{0};
    std::string namespace_;                   // tenant
    std::string content;                      // original (full)
    std::vector<float> embedding;             // for resurrect-without-re-embed
    GateDecision decision{GateDecision::Accepted};
    std::string reason;
    float marginal_value{0.0f};
    uint64_t conflict_with_id{0};             // when reason mentions a duplicate
    MemoryOwner owner{MemoryOwner::Session};
    MemoryLayer layer{MemoryLayer::Raw};      // Raw (top-level) or Derived
    std::map<std::string, std::string> user_metadata;

    // Cross-link: id of the actual memory that ended up in store.
    // For Accepted → the new memory_id; for Rejected/Deferred (deferred still
    // stores) → relevant id; for Rejected → 0.
    uint64_t memory_id{0};

    // Resurrect outcome (filled when manually revived)
    uint64_t resurrected_to{0};
    int64_t  resurrected_at_ms{0};
    std::string resurrect_strategy;           // "coexist" / "replace_conflict" / "update_existing"

    // Reconcile decision (filled when Stage 2's Reconciler ran on this fact).
    // Empty op string = no reconcile happened (gate-only audit).
    std::string reconcile_op;                 // "ADD" / "REPLACE" / "RETRACT" / "REINFORCE" / "NOOP"
    uint64_t    reconcile_target_id{0};
    std::string reconcile_rationale;
};

struct GateLogConfig {
    size_t max_file_bytes{100 * 1024 * 1024};  // 100 MB → rotate
    size_t max_memory_entries{1000};            // ring buffer cap
    bool   keep_all_rotated{true};              // rotate but never delete (TTL=infinite)
};

/// Persistent, rotating audit log for WriteGate decisions.
/// Format: JSON Lines; one JSON object per line.
///
/// Each line is one of two ops:
///   {"op":"append", ...full GateLogEntry fields...}
///   {"op":"resurrect","entry_id":N,"new_memory_id":M,"strategy":"coexist","ts":...}
///
/// On replay we apply ops in order so resurrected_to gets updated correctly.
class GateLog {
public:
    explicit GateLog(const std::string& data_dir, GateLogConfig config = {});
    ~GateLog();

    bool open();
    void replay();

    /// Append a new gate verdict.
    void append(const GateLogEntry& entry);

    /// Record a resurrect operation (audit only — does NOT actually create the memory).
    void recordResurrect(uint64_t entry_id,
                         uint64_t new_memory_id,
                         const std::string& strategy);

    /// Look up an entry by id; checks memory ring then scans WAL.
    std::optional<GateLogEntry> findById(uint64_t entry_id) const;

    struct Filter {
        std::optional<GateDecision> decision;
        std::string namespace_filter;
        int64_t since_ms{0};
        size_t limit{100};
        bool only_unresurrected{false};
    };

    /// Query the in-memory ring (does not scan rotated WALs — those are kept
    /// for audit but not paginated by this API; users dump the file directly
    /// if they need the long-tail).
    std::vector<GateLogEntry> query(const Filter& f) const;

    struct Stats {
        size_t accepted{0};
        size_t rejected{0};
        size_t deferred{0};
        size_t resurrected{0};
        size_t total() const { return accepted + rejected + deferred; }
    };
    Stats stats(int64_t since_ms = 0,
                const std::string& namespace_filter = "") const;

    size_t memorySize() const;
    void flush();

private:
    void rotateIfNeeded();
    std::string currentPath() const;
    std::string rotatedPath(size_t index) const;
    size_t nextRotateIndex() const;

    void writeLine(const std::string& line);

    std::string data_dir_;
    GateLogConfig config_;
    std::ofstream file_;
    size_t current_file_size_{0};
    mutable std::mutex mutex_;
    std::deque<GateLogEntry> recent_;
    SnowflakeGenerator id_gen_;
};

}  // namespace amind
