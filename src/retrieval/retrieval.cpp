#include "retrieval.h"
#include "capture/capture_pipeline.h"
#include "vector/distance.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <queue>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace amind {

using json = nlohmann::json;

RetrievalPipeline::RetrievalPipeline(
    MemoryStore& store, GraphStore& graph,
    std::shared_ptr<LLMProvider> llm,
    std::shared_ptr<EmbedProvider> embedder,
    RetrievalWeights weights)
    : store_(store), graph_(graph),
      llm_(std::move(llm)), embedder_(std::move(embedder)),
      weights_(weights) {}

Result<std::vector<ScoredMemory>> RetrievalPipeline::recall(
    const std::string& query, 
    const std::string& agent_id,
    const std::string& user_id,
    size_t top_k, 
    bool analyze_intent) {

    // 0. Freshness barrier: if there are recently-submitted Stage 2 tasks for
    //    this agent, wait for them to complete so recall sees the latest state.
    if (capture_pipeline_ && !agent_id.empty()) {
        uint64_t flight_key = std::hash<std::string>{}(agent_id);
        if (capture_pipeline_->hasFreshPending(flight_key)) {
            capture_pipeline_->waitForPendingRefinements(flight_key);
        }
    }

    // 1. Optionally analyze intent
    QueryIntent intent;
    if (analyze_intent && llm_) {
        auto intent_result = analyzeIntent(query);
        if (intent_result.ok()) {
            intent = std::move(*intent_result);
        }
    }

    // 1b. Snapshot retrieval weights (thread-safe)
    auto w = getWeights();

    // 2. Embed the query (or rewritten query)
    std::string search_query = intent.rewritten_query.empty() ? query : intent.rewritten_query;
    spdlog::debug("recall: query='{}' agent_id='{}' user_id='{}' top_k={}", search_query, agent_id, user_id, top_k);
    auto embed_result = embedder_->embed(search_query);
    if (!embed_result.ok()) {
        spdlog::error("recall: embed failed: {}", embed_result.error().toString());
        return embed_result.error();
    }

    // 3. Semantic search (HNSW) with user_id pre-filter.
    // Private-scope memories are filtered at the HNSW level via searchWithFilter
    // so that they don't crowd out results for the requesting user.
    size_t over_fetch = top_k * 10;
    std::vector<std::pair<uint64_t, float>> semantic_hits;
    if (!user_id.empty() && user_id != "anonymous") {
        auto filter = [this, &user_id](uint64_t mid) {
            return store_.peekScopeMatch(mid, user_id);
        };
        semantic_hits = store_.searchSimilar(*embed_result, over_fetch, filter);
    } else {
        semantic_hits = store_.searchSimilar(*embed_result, over_fetch);
    }
    spdlog::debug("recall: searchSimilar returned {} hits (over_fetch={})", semantic_hits.size(), over_fetch);

    // 4. Build scored candidates
    std::vector<ScoredMemory> candidates;
    std::unordered_set<uint64_t> candidate_ids;
    size_t skip_not_ok = 0, skip_not_alive = 0, skip_agent = 0, skip_scope = 0;

    for (const auto& [memory_id, similarity] : semantic_hits) {
        auto record_result = store_.get(memory_id);
        if (!record_result.ok()) { skip_not_ok++; continue; }
        if (!record_result->isAlive()) { skip_not_alive++; continue; }

        // Filter by agent_id
        if (!agent_id.empty() && record_result->agent_id != agent_id) { skip_agent++; continue; }

        // Filter by scope: Private memories must match user_id
        if (record_result->scope == MemoryScope::Private) {
            if (!user_id.empty() && record_result->user_id != user_id) { skip_scope++; continue; }
        }

        ScoredMemory scored;
        scored.record = std::move(*record_result);
        scored.embedding = scored.record.embedding;
        scored.semantic_score = similarity;
        scored.recency_score = computeRecencyScore(scored.record);

        scored.keyword_score = computeKeywordScore(
            scored.record.content, search_query, intent.entities);

        auto neighbors = graph_.getNeighbors(memory_id);
        size_t connected_in_results = 0;
        for (auto neighbor_id : neighbors) {
            if (candidate_ids.count(neighbor_id)) connected_in_results++;
        }
        scored.graph_score = std::min(1.0f,
            static_cast<float>(connected_in_results) * 0.2f +
            static_cast<float>(neighbors.size()) * 0.05f);

        if (w.recency_gate_enabled) {
            float base_score =
                w.semantic * scored.semantic_score +
                w.keyword * scored.keyword_score +
                w.graph * scored.graph_score +
                w.importance * scored.record.importance;
            scored.total_score = base_score * scored.recency_score;
        } else {
            scored.total_score =
                w.semantic * scored.semantic_score +
                w.keyword * scored.keyword_score +
                w.recency * scored.recency_score +
                w.graph * scored.graph_score +
                w.importance * scored.record.importance;
        }

        candidates.push_back(std::move(scored));
        candidate_ids.insert(memory_id);
    }
    spdlog::debug("recall: candidates={} skip_not_ok={} skip_not_alive={} skip_agent={} skip_scope={}",
                  candidates.size(), skip_not_ok, skip_not_alive, skip_agent, skip_scope);

    // Per-fact Derived preference: only exclude Raw memories that have
    // corresponding Derived children (via parent_id lineage). Raw memories
    // without Derived coverage remain searchable.
    std::unordered_set<uint64_t> derived_parents;
    for (const auto& sm : candidates) {
        if (sm.record.layer == MemoryLayer::Derived && sm.record.parent_id != 0) {
            derived_parents.insert(sm.record.parent_id);
        }
    }
    if (!derived_parents.empty()) {
        candidates.erase(
            std::remove_if(candidates.begin(), candidates.end(),
                           [&derived_parents](const ScoredMemory& sm) {
                               return sm.record.layer == MemoryLayer::Raw
                                   && derived_parents.count(sm.record.memory_id);
                           }),
            candidates.end());
    }

    // ── Global Supersedes scan ────────────────────────────────────────────────
    // Fetch all Supersedes edges once. Used by both the Raw parent filter and
    // the content-level filter below. Global scan ensures filtering works even
    // when the superseding memory is not in the recall candidate set.
    auto all_supersedes = graph_.getSupersedes();

    // ── Superseded Raw filter ─────────────────────────────────────────────────
    // If a Derived memory D has been superseded by another Derived memory D',
    // then the Raw memory that D was extracted from should also be suppressed
    // (the fact it describes is no longer current). This prevents recall from
    // surfacing stale facts via their Raw memories.
    std::unordered_set<uint64_t> superseded_derived_parents;
    for (const auto& edge : all_supersedes) {
        auto superseded = store_.peek(edge.to_id);
        if (superseded.ok()
            && superseded->layer == MemoryLayer::Derived
            && superseded->parent_id != 0) {
            superseded_derived_parents.insert(superseded->parent_id);
        }
    }
    if (!superseded_derived_parents.empty()) {
        candidates.erase(
            std::remove_if(candidates.begin(), candidates.end(),
                           [&superseded_derived_parents](const ScoredMemory& sm) {
                               return sm.record.layer == MemoryLayer::Raw
                                   && superseded_derived_parents.count(sm.record.memory_id);
                           }),
            candidates.end());
    }

    // ── Content-level superseded filter ─────────────────────────────────
    // Raw/Derived memories whose content closely overlaps with superseded
    // Derived facts should also be suppressed, even if they aren't direct
    // parents. This catches memories from different source turns that mention
    // the same old entity (e.g. "我的特斯拉每周充两次电" when the Tesla car
    // fact was superseded by "用户的车是比亚迪汉EV").
    // Timestamp guard: only filter candidates OLDER than the superseding fact
    // to avoid filtering the correction itself.
    // Chain walk: when D3 supersedes D2 which supersedes D1, we collect ALL
    // old values (D1, D2, and their Raw parents) so that Raw memories
    // mentioning ANY historical value are caught — not just the most recent
    // superseded content. This handles multi-hop value changes like
    // salary 2万 → 3万 → 4万 where Raw echoes of "2万" would otherwise leak.
    {
        struct SupersededInfo {
            std::string content;
            uint32_t superseding_ts;  // created_at of the terminal superseding fact
        };
        std::vector<SupersededInfo> superseded_infos;

        // Build superseding map: to_id → from_id (superseded → who superseded it)
        std::unordered_map<uint64_t, uint64_t> superseding_map;
        for (const auto& edge : all_supersedes) {
            superseding_map[edge.to_id] = edge.from_id;
        }

        std::unordered_set<uint64_t> visited_global;
        for (const auto& edge : all_supersedes) {
            // Find the terminal (alive) superseding fact by walking UP the chain
            uint64_t terminal_id = edge.from_id;
            {
                std::unordered_set<uint64_t> up_visited;
                uint64_t cur = edge.from_id;
                while (up_visited.insert(cur).second) {
                    auto it = superseding_map.find(cur);
                    if (it == superseding_map.end()) break;
                    cur = it->second;
                    auto frec = store_.peek(cur);
                    if (frec.ok() && frec->isAlive()) terminal_id = cur;
                }
            }
            auto terminal_rec = store_.peek(terminal_id);
            if (!terminal_rec.ok() || !terminal_rec->isAlive()) continue;
            uint32_t terminal_ts = terminal_rec->created_at;

            // Walk DOWN from the superseded node, collecting all transitively
            // superseded Derived content and their Raw parents' content.
            // peek() returns tombstoned records, so we can read old values.
            std::queue<uint64_t> q;
            q.push(edge.to_id);
            while (!q.empty()) {
                uint64_t cur = q.front();
                q.pop();
                if (!visited_global.insert(cur).second) continue;

                auto cur_rec = store_.peek(cur);
                if (!cur_rec.ok()) continue;

                if (cur_rec->layer == MemoryLayer::Derived) {
                    superseded_infos.push_back({cur_rec->content, terminal_ts});
                    // Include Raw parent's content — catches historical values
                    // from earlier turns (e.g. Raw parent of D(3万) mentions
                    // "2万" which is also stale after D(4万) supersedes D(3万))
                    if (cur_rec->parent_id != 0) {
                        auto parent = store_.peek(cur_rec->parent_id);
                        if (parent.ok()
                            && visited_global.insert(cur_rec->parent_id).second) {
                            superseded_infos.push_back({parent->content, terminal_ts});
                        }
                    }
                }

                // Follow outgoing Supersedes edges to older superseded facts
                auto edges = graph_.getEdges(cur);
                for (const auto& e : edges) {
                    if (e.type == EdgeType::Supersedes
                        && !visited_global.count(e.to_id)) {
                        q.push(e.to_id);
                    }
                }
            }
        }
        if (!superseded_infos.empty()) {
            auto makeBigrams = [](const std::string& s) {
                std::unordered_set<std::string> bg;
                std::vector<std::string> chars;
                for (size_t k = 0; k < s.size(); ) {
                    unsigned char c = (unsigned char)s[k];
                    size_t L = 1;
                    if ((c & 0xE0) == 0xC0) L = 2;
                    else if ((c & 0xF0) == 0xE0) L = 3;
                    else if ((c & 0xF8) == 0xF0) L = 4;
                    if (k + L > s.size()) break;
                    chars.emplace_back(s.substr(k, L));
                    k += L;
                }
                for (size_t k = 0; k + 1 < chars.size(); ++k) {
                    bg.insert(chars[k] + chars[k + 1]);
                }
                return bg;
            };
            struct SupersededBg {
                std::unordered_set<std::string> bigrams;
                uint32_t superseding_ts;
            };
            std::vector<SupersededBg> superseded_bgs;
            for (const auto& info : superseded_infos) {
                auto bg = makeBigrams(info.content);
                if (!bg.empty()) {
                    superseded_bgs.push_back({std::move(bg), info.superseding_ts});
                }
            }
            if (!superseded_bgs.empty()) {
                candidates.erase(
                    std::remove_if(candidates.begin(), candidates.end(),
                        [&superseded_bgs, &makeBigrams](const ScoredMemory& sm) {
                            auto bg_set = makeBigrams(sm.record.content);
                            if (bg_set.empty()) return false;
                            for (const auto& sbg : superseded_bgs) {
                                // Only filter if candidate is OLDER than the
                                // superseding fact (avoid filtering corrections)
                                if (sm.record.created_at >= sbg.superseding_ts) continue;
                                size_t hit = 0;
                                for (const auto& bg : bg_set) {
                                    if (sbg.bigrams.count(bg)) hit++;
                                }
                                // Use superseded bigrams as denominator: what %
                                // of the old value's content appears in candidate?
                                float overlap = (float)hit / (float)sbg.bigrams.size();
                                if (overlap >= 0.35f) return true;
                            }
                            return false;
                        }),
                    candidates.end());
            }
        }
    }

    auto ranked = rankAndFuse(candidates, top_k);

    // ── Post-ranking filter: remove superseded and conflict-losing memories ──
    // This ensures callers always receive the "current best" version of each
    // fact, not the full contradiction history.  Storage is untouched — the
    // old versions remain for /history and audit.
    if (ranked.size() > 1) {
        // Collect IDs of memories that have been superseded by a newer version
        // already present in the result set, or that lost a conflict to a newer
        // sibling also in the result set.
        std::unordered_set<uint64_t> result_ids;
        for (const auto& sm : ranked) result_ids.insert(sm.record.memory_id);

        // Debug: log all candidates before filtering
        if (ranked.size() <= 20) {
            for (size_t i = 0; i < ranked.size(); ++i) {
                const auto& rec = ranked[i].record;
                auto edges = graph_.getEdges(rec.memory_id);
                size_t supersedes_count = 0;
                for (const auto& e : edges) {
                    if (e.type == EdgeType::Supersedes) supersedes_count++;
                }
                spdlog::debug("recall candidate[{}] id={} score={:.3f} supersedes_edges={} content={}",
                             i, rec.memory_id, ranked[i].total_score, supersedes_count,
                             rec.content.substr(0, 60));
            }
        }

        std::unordered_set<uint64_t> suppress;

        for (const auto& sm : ranked) {
            auto edges = graph_.getEdges(sm.record.memory_id);
            for (const auto& edge : edges) {
                // Supersedes edge: sm supersedes edge.to_id → suppress the old one
                if (edge.type == EdgeType::Supersedes && result_ids.count(edge.to_id)) {
                    spdlog::debug("suppressing {} (superseded by {})", edge.to_id, sm.record.memory_id);
                    suppress.insert(edge.to_id);
                }
                // ConflictsWith edge: keep the one with the newer created_at
                if (edge.type == EdgeType::ConflictsWith && result_ids.count(edge.to_id)) {
                    // Look up the conflicting peer in the result set
                    for (const auto& peer : ranked) {
                        if (peer.record.memory_id == edge.to_id) {
                            if (sm.record.created_at >= peer.record.created_at) {
                                suppress.insert(peer.record.memory_id);
                            } else {
                                suppress.insert(sm.record.memory_id);
                            }
                            break;
                        }
                    }
                }
            }

            // Version-chain filter via parent_id: if sm has a parent_id that is
            // also in the result set, suppress the parent (older version).
            if (sm.record.parent_id != 0 && result_ids.count(sm.record.parent_id)) {
                suppress.insert(sm.record.parent_id);
            }
        }

        // ── "Old version" semantic filter ────────────────────────────────────
        // If two results describe the same entity and one contains temporal
        // markers indicating it's historical ("旧的"、"之前的"、"以前的"),
        // suppress the historical one. This prevents recall from surfacing
        // e.g. "旧护照号 E73291047" alongside "护照号 E83291047".
        static const std::vector<std::string> old_markers = {
            "旧", "之前", "以前", "原来", "过去", "历史", "曾经", "earlier",
            "previous", "old", "former", "past"
        };
        // Contrastive markers: if a sentence has BOTH an old marker and a
        // contrastive marker, it describes a state CHANGE ("之前不喜欢X，但
        // 现在迷上了"), not an old version of another fact.
        static const std::vector<std::string> contrastive_markers = {
            "但", "不过", "现在", "如今", "已经", "now", "but", "however",
            "currently"
        };
        auto contentHasOldMarker = [&](const std::string& content) -> bool {
            bool has_old = false;
            for (const auto& marker : old_markers) {
                if (content.find(marker) != std::string::npos) { has_old = true; break; }
            }
            if (!has_old) return false;
            for (const auto& marker : contrastive_markers) {
                if (content.find(marker) != std::string::npos) return false;
            }
            return true;
        };

        // For each pair: if both are semantically very similar (same entity),
        // and one has old-markers but the other doesn't, suppress the old one.
        for (size_t i = 0; i < ranked.size(); ++i) {
            if (suppress.count(ranked[i].record.memory_id)) continue;
            bool i_old = contentHasOldMarker(ranked[i].record.content);
            for (size_t j = i + 1; j < ranked.size(); ++j) {
                if (suppress.count(ranked[j].record.memory_id)) continue;
                bool j_old = contentHasOldMarker(ranked[j].record.content);
                if (i_old == j_old) continue;  // same polarity — skip

                // Check semantic similarity between the two (use cached embeddings)
                if (ranked[i].embedding.empty() || ranked[j].embedding.empty()) continue;
                float sim = 0.0f;
                if (ranked[i].embedding.size() == ranked[j].embedding.size()) {
                    float dot = 0.0f, ni = 0.0f, nj = 0.0f;
                    for (size_t d = 0; d < ranked[i].embedding.size(); ++d) {
                        dot += ranked[i].embedding[d] * ranked[j].embedding[d];
                        ni  += ranked[i].embedding[d] * ranked[i].embedding[d];
                        nj  += ranked[j].embedding[d] * ranked[j].embedding[d];
                    }
                    if (ni > 0 && nj > 0) sim = dot / (std::sqrt(ni) * std::sqrt(nj));
                }
                if (sim >= 0.70f) {
                    // The one with old-markers is the "old" candidate, BUT only
                    // if it was created BEFORE the other. In Chinese, "不再是X"
                    // often appears in the NEWER record to describe a state change.
                    // When the old-marker record is newer, it's the current truth
                    // — suppress the other (stale) record instead.
                    const auto& marker_rec = i_old ? ranked[i].record : ranked[j].record;
                    const auto& plain_rec  = i_old ? ranked[j].record : ranked[i].record;
                    float marker_score = i_old ? ranked[i].total_score : ranked[j].total_score;
                    float plain_score  = i_old ? ranked[j].total_score : ranked[i].total_score;

                    uint64_t suppress_id;
                    if (marker_rec.created_at >= plain_rec.created_at) {
                        // Safety: if plain_rec has significantly higher score,
                        // it's likely the correct answer and marker_rec is a
                        // separate historical fact (not a state change).
                        if (plain_score > marker_score + 0.1f) {
                            spdlog::debug("RetrievalPipeline: skipping old-version suppression "
                                         "(plain_score {:.3f} >> marker_score {:.3f})",
                                         plain_score, marker_score);
                            continue;
                        }
                        suppress_id = plain_rec.memory_id;
                    } else {
                        suppress_id = marker_rec.memory_id;
                    }
                    suppress.insert(suppress_id);
                    spdlog::info("RetrievalPipeline: suppressing old-version memory {} (sim={:.3f})",
                                 suppress_id, sim);
                }
            }
        }

        if (!suppress.empty()) {
            spdlog::info("RetrievalPipeline: suppressing {} superseded/conflict-losing memories",
                         suppress.size());
            ranked.erase(
                std::remove_if(ranked.begin(), ranked.end(),
                               [&suppress](const ScoredMemory& sm) {
                                   return suppress.count(sm.record.memory_id) > 0;
                               }),
                ranked.end());
        }
    }

    // ── Aggregate staleness filter ────────────────────────────────────────────
    // Drops list-style aggregate memories that have been rendered stale by
    // newer atomic facts present in the same recall result set. Strictly a
    // post-rank step; storage is untouched, only this query's results change.
    if (staleness_filter_ && ranked.size() > 1) {
        std::vector<AggregateStalenessFilter::ScoredCandidate> sc;
        sc.reserve(ranked.size());
        for (const auto& sm : ranked) {
            sc.push_back({sm.record.memory_id, sm.total_score, &sm.record});
        }
        size_t dropped = staleness_filter_->apply(sc, query, agent_id, events_log_);
        if (dropped > 0) {
            std::unordered_set<uint64_t> kept;
            for (const auto& c : sc) kept.insert(c.memory_id);
            ranked.erase(
                std::remove_if(ranked.begin(), ranked.end(),
                               [&kept](const ScoredMemory& sm) {
                                   return kept.count(sm.record.memory_id) == 0;
                               }),
                ranked.end());
            spdlog::info("RetrievalPipeline: aggregate-staleness dropped {} stale aggregate(s)",
                         dropped);
        }
    }

    return ranked;
}

Result<std::vector<ScoredMemory>> RetrievalPipeline::simpleSearch(
    const std::string& query, size_t top_k) {
    auto embed_result = embedder_->embed(query);
    if (!embed_result.ok()) return embed_result.error();

    auto hits = store_.searchSimilar(*embed_result, top_k);
    std::vector<ScoredMemory> results;
    for (const auto& [memory_id, score] : hits) {
        auto record = store_.get(memory_id);
        if (!record.ok()) continue;
        ScoredMemory sm;
        sm.record = std::move(*record);
        sm.semantic_score = score;
        sm.total_score = score;
        results.push_back(std::move(sm));
    }
    return results;
}

Result<QueryIntent> RetrievalPipeline::analyzeIntent(const std::string& query) {
    std::string prompt =
        "Analyze the search intent of this query. Return JSON with fields: "
        "rewritten_query (optimized search text), entities (list of key entities), "
        "urgency (0-1).\n\nQuery: " + query;

    auto result = llm_->generateJson(prompt);
    if (!result.ok()) return result.error();

    try {
        auto j = json::parse(*result);
        QueryIntent intent;
        intent.rewritten_query = j.value("rewritten_query", query);
        intent.urgency = j.value("urgency", 0.5f);
        if (j.contains("entities") && j["entities"].is_array()) {
            for (const auto& e : j["entities"]) {
                intent.entities.push_back(e.get<std::string>());
            }
        }
        return intent;
    } catch (...) {
        QueryIntent intent;
        intent.rewritten_query = query;
        return intent;
    }
}

float RetrievalPipeline::computeRecencyScore(const MemoryRecord& record) const {
    auto now = MemoryRecord::currentTimeSec();
    float age_hours = static_cast<float>(now - record.last_accessed) / 3600.0f;
    return std::exp(-age_hours / (24.0f * 30.0f));  // 30-day half-life
}

namespace {

/// Decode a UTF-8 string into a vector of codepoint strings.
/// Each output element is one full character (1-4 bytes).
std::vector<std::string> utf8Chars(const std::string& s) {
    std::vector<std::string> out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = 1;
        if      ((c & 0x80) == 0x00) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (i + len > s.size()) break;
        out.emplace_back(s.substr(i, len));
        i += len;
    }
    return out;
}

/// Build the set of character bigrams for a string. Lower-cased ASCII inside.
std::unordered_set<std::string> charBigrams(const std::string& s) {
    auto chars = utf8Chars(s);
    // ASCII-lowercase in place (CJK codepoints are untouched).
    for (auto& ch : chars) {
        if (ch.size() == 1) {
            ch[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(ch[0])));
        }
    }
    std::unordered_set<std::string> out;
    if (chars.size() < 2) return out;
    out.reserve(chars.size());
    for (size_t i = 0; i + 1 < chars.size(); ++i) {
        out.insert(chars[i] + chars[i + 1]);
    }
    return out;
}

}  // namespace

float RetrievalPipeline::computeKeywordScore(
    const std::string& content,
    const std::string& query,
    const std::vector<std::string>& entities) const {

    // ── 1. Whitespace-token overlap (covers Latin script well) ──
    std::vector<std::string> keywords;
    std::istringstream iss(query);
    std::string word;
    while (iss >> word) {
        if (word.size() >= 2) {
            std::string lower_word;
            lower_word.reserve(word.size());
            for (char c : word) lower_word += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            keywords.push_back(std::move(lower_word));
        }
    }
    for (const auto& entity : entities) {
        if (!entity.empty()) {
            std::string lower_entity;
            lower_entity.reserve(entity.size());
            for (char c : entity) lower_entity += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            keywords.push_back(std::move(lower_entity));
        }
    }

    std::string lower_content;
    lower_content.reserve(content.size());
    for (char c : content) lower_content += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    float token_score = 0.0f;
    if (!keywords.empty()) {
        size_t hits = 0;
        for (const auto& kw : keywords) {
            if (lower_content.find(kw) != std::string::npos) hits++;
        }
        token_score = static_cast<float>(hits) / static_cast<float>(keywords.size());
    }

    // ── 2. Char-bigram overlap (CJK-friendly; Latin gets it for free too) ──
    // Whitespace tokens fail for Chinese (no spaces). Bigrams let
    // "古力娜扎" overlap with "用户喜欢古力娜扎" naturally.
    float bigram_score = 0.0f;
    auto q_bigrams = charBigrams(query);
    if (!q_bigrams.empty()) {
        auto c_bigrams = charBigrams(content);
        size_t matched = 0;
        for (const auto& bg : q_bigrams) {
            if (c_bigrams.count(bg)) matched++;
        }
        bigram_score = static_cast<float>(matched) / static_cast<float>(q_bigrams.size());
    }

    // Take the max — either signal alone is sufficient evidence.
    return std::max(token_score, bigram_score);
}

std::vector<ScoredMemory> RetrievalPipeline::rankAndFuse(
    std::vector<ScoredMemory>& candidates, size_t top_k) {
    if (candidates.empty()) return {};

    // Optional reranking pass (before MMR diversity selection)
    if (reranker_ && candidates.size() > 1) {
        std::vector<std::string> docs;
        docs.reserve(candidates.size());
        for (const auto& c : candidates) {
            docs.push_back(c.record.content);
        }
        // Use the first candidate's content as a proxy query (best semantic match)
        std::string query_proxy = candidates.front().record.content;
        auto rerank_result = reranker_->rerank(query_proxy, docs, candidates.size());
        if (rerank_result.ok()) {
            auto& ranked = rerank_result.value();
            for (const auto& r : ranked) {
                if (r.index < candidates.size()) {
                    candidates[r.index].total_score = r.score;
                }
            }
        }
    }

    // Pre-extract entity IDs (used by both branches below).
    std::vector<std::unordered_set<std::string>> entity_ids(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        auto extracted = AggregateStalenessFilter::extractIds(
            candidates[i].record.content);
        for (const auto& [_, ids] : extracted) {
            for (const auto& id : ids) entity_ids[i].insert(id);
        }
    }

    if (candidates.size() <= top_k) {
        // Sort by score then re-balance: a same-entity run longer than 2 in a
        // row gets interrupted by the next different-entity (or no-entity)
        // candidate. Keeps top-rank items adjacent for the agent's primary
        // anchor while preventing one entity from monopolizing the tail.
        std::vector<size_t> order(candidates.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t b) {
                      return candidates[a].total_score > candidates[b].total_score;
                  });

        auto entityKey = [&](size_t i) -> std::string {
            if (entity_ids[i].empty()) return std::string();
            return *entity_ids[i].begin();  // canonical first ID
        };

        std::vector<size_t> rebalanced;
        rebalanced.reserve(order.size());
        std::vector<bool> placed(order.size(), false);
        std::string last_key;
        size_t run = 0;
        for (size_t step = 0; step < order.size(); ++step) {
            // Find the highest-ranked unplaced item whose entity-key differs
            // from last_key when we're already on a 2-run of the same entity.
            size_t pick = order.size();
            for (size_t k = 0; k < order.size(); ++k) {
                size_t cand = order[k];
                if (placed[cand]) continue;
                if (run >= 2 && !last_key.empty() && entityKey(cand) == last_key) continue;
                pick = cand;
                break;
            }
            // Fallback: if no different-entity option exists, take the next
            // unplaced item by score (degenerates to plain score order).
            if (pick == order.size()) {
                for (size_t k = 0; k < order.size(); ++k) {
                    if (!placed[order[k]]) { pick = order[k]; break; }
                }
            }
            placed[pick] = true;
            rebalanced.push_back(pick);
            std::string k = entityKey(pick);
            if (!k.empty() && k == last_key) run++;
            else { last_key = k; run = 1; }
        }

        std::vector<ScoredMemory> out;
        out.reserve(rebalanced.size());
        for (size_t i : rebalanced) out.push_back(std::move(candidates[i]));
        return out;
    }

    // Reserve 1 slot for exploration injection if enough candidates
    size_t select_k = (candidates.size() > top_k + 2) ? top_k - 1 : top_k;

    // ── Entity-aware MMR diversity ────────────────────────────────────────────
    // Reuse the same patterns used by AggregateStalenessFilter so the diversity
    // step recognizes "TKT001 vs TKT005" as different entities and avoids
    // letting a single entity dominate the top_k. (entity_ids was already
    // extracted above for the small-candidate-set branch.)
    auto sharesEntity = [&](size_t a, size_t b) -> bool {
        if (entity_ids[a].empty() || entity_ids[b].empty()) return false;
        for (const auto& id : entity_ids[a]) {
            if (entity_ids[b].count(id)) return true;
        }
        return false;
    };

    // MMR: Maximal Marginal Relevance for diversity
    // MMR(d) = λ * score(d) - (1-λ) * max_sim(d, selected)
    constexpr float lambda = 0.7f;
    // Entity collision is treated as ≥ this similarity. We use 1.0 (saturate)
    // because vector similarity between two memories about the same entity is
    // already very high (>0.9), so anything below would be redundant. With
    // 1.0 the penalty is (1-lambda)·1 = 0.3, the maximum possible, making
    // same-entity candidates strictly dominated by any different-entity
    // candidate with a comparable normalized score. Different entities are
    // only revisited after all distinct entities are exhausted.
    constexpr float kEntityCollisionSim = 1.0f;

    std::vector<ScoredMemory> selected;
    std::vector<size_t> selected_orig_idx;  // original indices into candidates
    selected.reserve(top_k);
    selected_orig_idx.reserve(top_k);
    std::vector<bool> used(candidates.size(), false);

    // Normalize scores to [0,1] for fair MMR computation
    float max_score = 0.0f;
    for (const auto& c : candidates) {
        max_score = std::max(max_score, c.total_score);
    }
    if (max_score <= 0.0f) max_score = 1.0f;

    // Select first: highest total_score
    size_t best_idx = 0;
    for (size_t i = 1; i < candidates.size(); i++) {
        if (candidates[i].total_score > candidates[best_idx].total_score)
            best_idx = i;
    }
    selected.push_back(std::move(candidates[best_idx]));
    selected_orig_idx.push_back(best_idx);
    used[best_idx] = true;

    // Greedy MMR selection
    while (selected.size() < select_k) {
        float best_mmr = -std::numeric_limits<float>::max();
        size_t best_i = 0;

        for (size_t i = 0; i < candidates.size(); i++) {
            if (used[i]) continue;

            float norm_score = candidates[i].total_score / max_score;

            // Max similarity to any already-selected item — combines vector
            // distance with entity-ID overlap so a candidate that shares an
            // entity with a selected one can't claim "novel" status even if
            // its embedding is far from the selected ones.
            float max_sim = 0.0f;
            for (size_t k = 0; k < selected.size(); ++k) {
                const auto& sel = selected[k];
                size_t sel_orig = selected_orig_idx[k];
                if (sharesEntity(i, sel_orig)) {
                    max_sim = std::max(max_sim, kEntityCollisionSim);
                }
                if (candidates[i].embedding.empty() || sel.embedding.empty()) continue;
                if (sel.embedding.size() != candidates[i].embedding.size()) continue;
                float sim = cosineDistance(candidates[i].embedding, sel.embedding);
                sim = 1.0f - sim;  // distance → similarity
                max_sim = std::max(max_sim, sim);
            }

            float mmr = lambda * norm_score - (1.0f - lambda) * max_sim;
            if (mmr > best_mmr) {
                best_mmr = mmr;
                best_i = i;
            }
        }

        selected.push_back(std::move(candidates[best_i]));
        selected_orig_idx.push_back(best_i);
        used[best_i] = true;
    }

    // Exploration injection: pick 1 random unused candidate from the bottom half
    if (select_k < top_k) {
        std::vector<size_t> unused_indices;
        for (size_t i = 0; i < candidates.size(); i++) {
            if (!used[i]) unused_indices.push_back(i);
        }
        if (!unused_indices.empty()) {
            thread_local std::mt19937 rng(std::random_device{}());
            std::uniform_int_distribution<size_t> dist(0, unused_indices.size() - 1);
            size_t explore_i = unused_indices[dist(rng)];
            selected.push_back(std::move(candidates[explore_i]));
        }
    }

    return selected;
}

void RetrievalPipeline::setWeight(const std::string& name, float v) {
    std::lock_guard lock(weights_mutex_);
    if (name == "semantic") weights_.semantic = v;
    else if (name == "keyword") weights_.keyword = v;
    else if (name == "graph") weights_.graph = v;
    else if (name == "recency") weights_.recency = v;
    else if (name == "importance") weights_.importance = v;
}

RetrievalWeights RetrievalPipeline::getWeights() const {
    std::lock_guard lock(weights_mutex_);
    return weights_;
}


}  // namespace amind
