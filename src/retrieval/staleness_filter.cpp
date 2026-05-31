#include "staleness_filter.h"

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

std::string truncateUtf8(const std::string& s, size_t max_bytes) {
    if (s.size() <= max_bytes) return s;
    size_t cut = max_bytes;
    while (cut > 0 && (static_cast<unsigned char>(s[cut]) & 0xC0) == 0x80) --cut;
    return s.substr(0, cut) + "…";
}

uint64_t now_ms() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
}

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
    StaleLog* log) const {

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

        if (log) {
            StaleFilterEvent ev;
            ev.timestamp_ms = now_ms();
            ev.query = query;
            ev.namespace_ = namespace_hint;
            ev.aggregate_id = candidates[i].memory_id;
            ev.aggregate_preview = truncateUtf8(agg_rec.content, 200);
            ev.aggregate_created_at = agg_rec.created_at;
            ev.witness_ids_in_aggregate.assign(agg_ids_total.begin(), agg_ids_total.end());
            std::sort(witness_ids_in_newer.begin(), witness_ids_in_newer.end());
            witness_ids_in_newer.erase(
                std::unique(witness_ids_in_newer.begin(), witness_ids_in_newer.end()),
                witness_ids_in_newer.end());
            ev.witness_ids_in_newer_facts = std::move(witness_ids_in_newer);
            ev.newer_fact_ids = std::move(newer_ids);
            ev.action = StaleFilterEvent::Action::Filter;
            ev.pre_score = candidates[i].score;
            ev.post_score = 0.0f;
            log->append(ev);
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

// ── StaleLog ─────────────────────────────────────────────────────────────────

namespace {

json eventToJson(const StaleFilterEvent& e) {
    json j;
    j["ts"] = e.timestamp_ms;
    j["query"] = e.query;
    j["ns"] = e.namespace_;
    j["aggregate_id"] = std::to_string(e.aggregate_id);
    j["aggregate_preview"] = e.aggregate_preview;
    j["aggregate_created_at"] = e.aggregate_created_at;
    j["witness_in_aggregate"] = e.witness_ids_in_aggregate;
    j["witness_in_newer"] = e.witness_ids_in_newer_facts;
    json ids = json::array();
    for (auto id : e.newer_fact_ids) ids.push_back(std::to_string(id));
    j["newer_fact_ids"] = std::move(ids);
    j["action"] = (e.action == StaleFilterEvent::Action::Filter) ? "Filter" : "Downweight";
    j["pre_score"] = e.pre_score;
    j["post_score"] = e.post_score;
    return j;
}

StaleFilterEvent jsonToEvent(const json& j) {
    StaleFilterEvent e;
    e.timestamp_ms = j.value("ts", uint64_t(0));
    e.query = j.value("query", "");
    e.namespace_ = j.value("ns", "");
    e.aggregate_id = std::stoull(j.value("aggregate_id", "0"));
    e.aggregate_preview = j.value("aggregate_preview", "");
    e.aggregate_created_at = j.value("aggregate_created_at", 0u);
    if (j.contains("witness_in_aggregate") && j["witness_in_aggregate"].is_array()) {
        for (const auto& v : j["witness_in_aggregate"]) e.witness_ids_in_aggregate.push_back(v.get<std::string>());
    }
    if (j.contains("witness_in_newer") && j["witness_in_newer"].is_array()) {
        for (const auto& v : j["witness_in_newer"]) e.witness_ids_in_newer_facts.push_back(v.get<std::string>());
    }
    if (j.contains("newer_fact_ids") && j["newer_fact_ids"].is_array()) {
        for (const auto& v : j["newer_fact_ids"]) e.newer_fact_ids.push_back(std::stoull(v.get<std::string>()));
    }
    auto act = j.value("action", "Filter");
    e.action = (act == "Downweight") ? StaleFilterEvent::Action::Downweight
                                      : StaleFilterEvent::Action::Filter;
    e.pre_score = j.value("pre_score", 0.0f);
    e.post_score = j.value("post_score", 0.0f);
    return e;
}

}  // namespace

StaleLog::StaleLog(const std::string& data_dir, size_t ring)
    : data_dir_(data_dir), ring_capacity_(ring) {}

StaleLog::~StaleLog() {
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

void StaleLog::open() {
    std::lock_guard lock(mutex_);
    namespace fs = std::filesystem;
    if (!fs::exists(data_dir_)) fs::create_directories(data_dir_);
    auto path = data_dir_ + "/recall_stale.log";
    file_.open(path, std::ios::binary | std::ios::app);
    if (!file_.is_open()) {
        spdlog::warn("StaleLog: failed to open {}", path);
    }
}

void StaleLog::replay() {
    std::lock_guard lock(mutex_);
    namespace fs = std::filesystem;
    auto path = data_dir_ + "/recall_stale.log";
    if (!fs::exists(path)) return;
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return;
    std::string line;
    size_t count = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        try {
            auto j = json::parse(line);
            recent_.push_back(jsonToEvent(j));
            ++count;
        } catch (...) {
            continue;
        }
    }
    while (recent_.size() > ring_capacity_) recent_.pop_front();
    spdlog::info("StaleLog: replayed {} entries", count);
}

void StaleLog::append(const StaleFilterEvent& ev) {
    std::lock_guard lock(mutex_);
    recent_.push_back(ev);
    while (recent_.size() > ring_capacity_) recent_.pop_front();
    if (file_.is_open()) {
        std::string line = eventToJson(ev).dump() + "\n";
        file_.write(line.data(), static_cast<std::streamsize>(line.size()));
        file_.flush();
    }
}

std::vector<StaleFilterEvent> StaleLog::recentEntries() const {
    std::lock_guard lock(mutex_);
    return {recent_.begin(), recent_.end()};
}

size_t StaleLog::memorySize() const {
    std::lock_guard lock(mutex_);
    return recent_.size();
}

}  // namespace amind
