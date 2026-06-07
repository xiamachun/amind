#pragma once

#include "core/memory_record.h"

#include <cstdint>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace amind {

class MemoryEventLog;

// StaleFilterEvent + StaleLog removed in Phase 4. AggregateStalenessFilter
// now emits MemoryEvent{kind=RecallStale} into the unified MemoryEventLog.

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
                 MemoryEventLog* events) const;

    /// Public for unit tests: extract IDs from text using the built-in
    /// patterns. Returns map of pattern_name → set of matched IDs.
    static std::vector<std::pair<std::string, std::vector<std::string>>>
    extractIds(const std::string& text);

private:
    Config config_;
};

}  // namespace amind
