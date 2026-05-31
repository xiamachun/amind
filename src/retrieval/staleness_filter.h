#pragma once

#include "core/memory_record.h"

#include <cstdint>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace amind {

/// One audit entry for a single filter decision.
struct StaleFilterEvent {
    uint64_t timestamp_ms{0};
    std::string query;
    std::string namespace_;
    uint64_t aggregate_id{0};
    std::string aggregate_preview;     // truncated content of the stale aggregate
    uint32_t aggregate_created_at{0};  // unix seconds
    std::vector<std::string> witness_ids_in_aggregate;
    std::vector<std::string> witness_ids_in_newer_facts;
    std::vector<uint64_t> newer_fact_ids;
    enum class Action : uint8_t { Filter, Downweight };
    Action action{Action::Filter};
    float pre_score{0.0f};
    float post_score{0.0f};
};

/// Persistent ring-buffer audit log (mirrors ForgetLog/GateLog patterns).
class StaleLog {
public:
    explicit StaleLog(const std::string& data_dir, size_t ring = 1000);
    ~StaleLog();

    void open();
    void replay();
    void append(const StaleFilterEvent& ev);
    std::vector<StaleFilterEvent> recentEntries() const;
    size_t memorySize() const;

private:
    std::string data_dir_;
    size_t ring_capacity_;
    std::ofstream file_;
    mutable std::mutex mutex_;
    std::deque<StaleFilterEvent> recent_;
};

/// Detector + filter for "list-style aggregate" memories that have been
/// rendered stale by newer atomic facts in the same recall result set.
///
/// Algorithm (cheap, no LLM):
///   1. For each candidate, regex-extract entity IDs of common patterns
///      (e.g. TKT\d+, A-\d+, plain \d{6+}). Two or more matched IDs of the
///      same pattern → flag as a candidate aggregate.
///   2. For each aggregate, scan the OTHER candidates: any whose created_at
///      is later than the aggregate AND that contains an ID matching the
///      same pattern but absent from the aggregate's set → "anti-witness".
///   3. Aggregate with ≥1 anti-witness is filtered (or downweighted).
///
/// Conservative by design: if no clear ID pattern is detected, leaves the
/// candidate alone. False positives (wrongly filtered fresh aggregates) are
/// far more harmful than false negatives (occasional stale aggregate slips
/// through), so we err on the side of leaving things in.
class AggregateStalenessFilter {
public:
    struct Config {
        bool enabled{true};
        size_t min_ids_to_be_aggregate{2};
        bool also_match_table_format{true};
    };

    AggregateStalenessFilter();
    explicit AggregateStalenessFilter(Config config);

    /// Filter a scored result list in place. Returns the count of memories
    /// that were filtered out. If `log` is non-null, every filter event is
    /// appended.
    struct ScoredCandidate {
        uint64_t memory_id;
        float score;
        const MemoryRecord* record;  // non-owning
    };
    size_t apply(std::vector<ScoredCandidate>& candidates,
                 const std::string& query,
                 const std::string& namespace_hint,
                 StaleLog* log) const;

    /// Public for unit tests: extract IDs from text using the built-in
    /// patterns. Returns map of pattern_name → set of matched IDs.
    static std::vector<std::pair<std::string, std::vector<std::string>>>
    extractIds(const std::string& text);

private:
    Config config_;
};

}  // namespace amind
