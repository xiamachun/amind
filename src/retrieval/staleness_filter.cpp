#include "staleness_filter.h"

#include "observability/memory_event_log.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <regex>
#include <set>
#include <spdlog/spdlog.h>
#include <unordered_map>

namespace amind {

using nlohmann::json;

namespace {

// Patterns we treat as "entity IDs". Conservative: we want clear, structured
// identifiers rather than arbitrary numbers. Order matters — first match wins
// per substring, so longer/more-specific patterns should come earlier.
struct IdPattern {
    const char* name;
    std::regex  re;
};

const std::vector<IdPattern>& patterns() {
    static const std::vector<IdPattern> kPatterns = {
        // 1+ capital letters, optional hyphen, then digits. Matches TKT005,
        // A-1001, INV-456, etc. We deliberately exclude the leading-letter
        // case where there's no hyphen (e.g. "A1001") to avoid false hits
        // on plain words.
        {"prefixed_id",  std::regex(R"(\b(?:[A-Z]{2,5}\d{3,6}|[A-Z]{1,5}-\d{3,6})\b)")},
        {"long_numeric", std::regex(R"(\b\d{8,}\b)")},                // long numeric IDs
    };
    return kPatterns;
}

bool looksLikeListLayout(const std::string& s) {
    // Cheap structural heuristics — markdown table, bullet list, "包含/列表/清单"
    if (s.find("|---") != std::string::npos) return true;          // markdown table separator
    if (s.find("\n- ") != std::string::npos) return true;          // bullet list
    if (s.find("\n* ") != std::string::npos) return true;
    if (s.find("\n1.") != std::string::npos) return true;
    if (s.find("以下") != std::string::npos
        || s.find("列表") != std::string::npos
        || s.find("清单") != std::string::npos
        || s.find("汇总") != std::string::npos
        || s.find("包括") != std::string::npos
        || s.find("包含") != std::string::npos) return true;
    return false;
}

// truncateUtf8() lives in observability/memory_event.h — shared helper.
// now_ms() removed — emit path uses MemoryEventLog which timestamps internally.

}  // namespace

// ── ID extraction ────────────────────────────────────────────────────────────

std::vector<std::pair<std::string, std::vector<std::string>>>
AggregateStalenessFilter::extractIds(const std::string& text) {
    std::vector<std::pair<std::string, std::vector<std::string>>> out;
    for (const auto& p : patterns()) {
        std::vector<std::string> ids;
        std::set<std::string> seen;
        auto begin = std::sregex_iterator(text.begin(), text.end(), p.re);
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            std::string id = it->str();
            if (seen.insert(id).second) ids.push_back(id);
        }
        if (!ids.empty()) out.emplace_back(p.name, std::move(ids));
    }
    return out;
}

// ── Filter ───────────────────────────────────────────────────────────────────

AggregateStalenessFilter::AggregateStalenessFilter()
    : config_() {}

AggregateStalenessFilter::AggregateStalenessFilter(Config config)
    : config_(std::move(config)) {}

size_t AggregateStalenessFilter::apply(
    std::vector<ScoredCandidate>& candidates,
    const std::string& query,
    const std::string& namespace_hint,
    MemoryEventLog* events) const {

    if (!config_.enabled) return 0;
    if (candidates.size() < 2) return 0;

    // Pre-compute ID extraction per candidate (avoid quadratic regex).
    struct CandFeat {
        // pattern_name → set of ids in this memory
        std::unordered_map<std::string, std::set<std::string>> ids_by_pattern;
        bool list_layout{false};
        size_t total_ids{0};
    };
    std::vector<CandFeat> feats(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (!candidates[i].record) continue;
        const auto& content = candidates[i].record->content;
        auto extracted = extractIds(content);
        for (auto& [pname, ids] : extracted) {
            auto& set = feats[i].ids_by_pattern[pname];
            for (auto& id : ids) set.insert(std::move(id));
            feats[i].total_ids += set.size();
        }
        feats[i].list_layout = looksLikeListLayout(content);
    }

    // Decide: a candidate is an "aggregate" if it has ≥ N IDs of one pattern,
    // OR it has list layout AND ≥ N-1 IDs.
    auto isAggregate = [&](size_t i) {
        const auto& f = feats[i];
        for (const auto& [_, ids] : f.ids_by_pattern) {
            if (ids.size() >= config_.min_ids_to_be_aggregate) return true;
            if (f.list_layout && config_.also_match_table_format
                && ids.size() + 1 >= config_.min_ids_to_be_aggregate) return true;
        }
        return false;
    };

    std::vector<bool> drop(candidates.size(), false);
    size_t dropped = 0;

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (!isAggregate(i)) continue;
        if (!candidates[i].record) continue;
        const auto& agg_rec = *candidates[i].record;
        const auto& agg_feat = feats[i];

        // For each pattern in this aggregate, look for anti-witnesses.
        std::vector<std::string> witness_ids_in_newer;
        std::vector<uint64_t> newer_ids;
        std::set<std::string> agg_ids_total;
        std::string winning_pattern;

        for (const auto& [pname, agg_ids] : agg_feat.ids_by_pattern) {
            for (const auto& id : agg_ids) agg_ids_total.insert(id);
            for (size_t j = 0; j < candidates.size(); ++j) {
                if (j == i) continue;
                if (!candidates[j].record) continue;
                const auto& other_rec = *candidates[j].record;
                if (other_rec.created_at <= agg_rec.created_at) continue;
                auto it = feats[j].ids_by_pattern.find(pname);
                if (it == feats[j].ids_by_pattern.end()) continue;
                bool any_new = false;
                for (const auto& id : it->second) {
                    if (agg_ids.find(id) == agg_ids.end()) {
                        witness_ids_in_newer.push_back(id);
                        any_new = true;
                    }
                }
                if (any_new) {
                    newer_ids.push_back(other_rec.memory_id);
                    if (winning_pattern.empty()) winning_pattern = pname;
                }
            }
        }

        if (witness_ids_in_newer.empty()) continue;

        drop[i] = true;
        ++dropped;

        if (events) {
            std::sort(witness_ids_in_newer.begin(), witness_ids_in_newer.end());
            witness_ids_in_newer.erase(
                std::unique(witness_ids_in_newer.begin(), witness_ids_in_newer.end()),
                witness_ids_in_newer.end());

            // Pipe-separated lists keep the JSONL line lean and parseable.
            auto join_pipe = [](const auto& items) {
                std::string out;
                bool first = true;
                for (const auto& s : items) {
                    if (!first) out += "|";
                    out += s;
                    first = false;
                }
                return out;
            };

            MemoryEvent ev;
            ev.memory_id = candidates[i].memory_id;
            ev.agent_id = namespace_hint;
            ev.kind = EventKind::RecallStale;
            ev.status = EventStatus::Ok;
            ev.summary = truncateUtf8(agg_rec.content, 80);
            ev.attrs["query"] = query;
            ev.attrs["action"] = "Filter";
            ev.attrs["aggregate_created_at"] = std::to_string(agg_rec.created_at);
            ev.attrs["pre_score"] = std::to_string(candidates[i].score);
            ev.attrs["witness_in_aggregate"] =
                join_pipe(std::vector<std::string>(agg_ids_total.begin(), agg_ids_total.end()));
            ev.attrs["witness_in_newer"] = join_pipe(witness_ids_in_newer);
            ev.attrs["newer_fact_count"] = std::to_string(newer_ids.size());
            events->append(std::move(ev));
        }
    }

    if (dropped > 0) {
        std::vector<ScoredCandidate> kept;
        kept.reserve(candidates.size() - dropped);
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (!drop[i]) kept.push_back(candidates[i]);
        }
        candidates = std::move(kept);
    }
    return dropped;
}

// StaleLog implementation removed in Phase 4 — events flow into the
// unified MemoryEventLog via apply()'s `events` parameter.

}  // namespace amind
