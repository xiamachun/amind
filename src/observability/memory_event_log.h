#pragma once

#include "memory_event.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace amind {

struct MemoryEventLogConfig {
    size_t ring_capacity{2000};
    size_t max_file_bytes{16 * 1024 * 1024};   // 16MB per file
    size_t max_rotated_files{5};
    /// When true, the log writes the JSONL file atomically (flush per append).
    /// Production: true. Tests can set false for speed.
    bool flush_per_append{true};
};

/// Unified persistent + indexed log for all MemoryEvents.
///
/// Replaces GateLog / ForgetLog / StaleLog. All amind subsystems that need to
/// emit observability events call append(); no subsystem persists its own log.
///
/// Storage:
///   - File: ${data_dir}/events.log  (JSONL, one event per line)
///   - Rotation: when file exceeds max_file_bytes, current → .1, .1 → .2, ...
///   - In-memory ring: last `ring_capacity` events kept hot
///
/// Indexes (all in memory, recomputed on append/evict):
///   - by memory_id  → vector<size_t> (indexes into recent_)
///   - by trace_id   → vector<size_t>
///   - by kind       → vector<size_t>
///
/// Indexes are intentionally lightweight (multimaps of ring positions). They
/// are *not* a persistent secondary index — if you need to query events older
/// than the ring (rotated to .1+), use jq / grep on the JSONL files directly
/// or rebuild a fresh log via the migration tool.
class MemoryEventLog {
public:
    explicit MemoryEventLog(const std::string& data_dir,
                            MemoryEventLogConfig config = {});
    ~MemoryEventLog();

    MemoryEventLog(const MemoryEventLog&) = delete;
    MemoryEventLog& operator=(const MemoryEventLog&) = delete;

    /// Open (or create) the events.log file in append mode. Idempotent.
    bool open();

    /// Replay the current events.log into the in-memory ring. Skips malformed
    /// lines. Caller should run this once at startup, after open().
    void replay();

    /// Single entry point for all subsystems.
    /// The log assigns event_id if it's 0 (use Snowflake; caller may pass a
    /// pre-generated event_id if they need a parent reference before append).
    void append(MemoryEvent ev);

    /// Force flush of the file buffer (no-op when flush_per_append is true).
    void flush();

    // ── Queries ───────────────────────────────────────────────────────────

    struct Filter {
        std::optional<uint64_t> memory_id;
        std::optional<uint64_t> trace_id;
        std::optional<EventKind> kind;
        std::optional<EventStatus> status;
        std::string agent_id_filter;  // empty = all
        uint64_t since_ms{0};          // 0 = beginning
        uint64_t until_ms{0};          // 0 = no upper bound
        size_t limit{200};
    };

    /// Generic query against the in-memory ring. Results are returned newest
    /// first.
    std::vector<MemoryEvent> query(const Filter& f) const;

    /// All events sharing the given trace_id, ordered by timestamp ascending
    /// so callers can render a span tree.
    std::vector<MemoryEvent> trace(uint64_t trace_id) const;

    /// All events touching this memory_id (across all traces it participated
    /// in), newest first.
    std::vector<MemoryEvent> memoryHistory(uint64_t memory_id) const;

    struct Stats {
        std::unordered_map<std::string, size_t> by_kind;
        std::unordered_map<std::string, size_t> by_status;
        size_t total{0};
    };
    Stats stats(const Filter& f) const;

    size_t memorySize() const;

    /// Test helper: drop in-memory ring + indexes (does NOT touch disk).
    void clearMemory();

private:
    void addToIndexes_(size_t pos, const MemoryEvent& ev);
    void rebuildIndexes_();
    void rotateIfNeeded_();
    std::string currentPath_() const;
    std::string rotatedPath_(size_t index) const;

    std::string data_dir_;
    MemoryEventLogConfig config_;
    std::ofstream file_;
    size_t current_file_size_{0};

    mutable std::mutex mutex_;
    std::deque<MemoryEvent> recent_;  // ring; oldest at front
    // Indexes hold positions into recent_. Positions stay valid only while
    // we don't pop_front; on eviction we rebuild the maps (cheap because the
    // ring is small, default 2000).
    std::unordered_multimap<uint64_t, size_t> by_memory_id_;
    std::unordered_multimap<uint64_t, size_t> by_trace_id_;
    std::unordered_multimap<EventKind, size_t> by_kind_;
};

}  // namespace amind
