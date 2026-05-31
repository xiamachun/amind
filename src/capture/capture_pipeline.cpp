#include "capture_pipeline.h"
#include "config/v2_config.h"
#include "capture/derived_extractor.h"
#include "gate/write_gate.h"
#include "gate_log/gate_log.h"
#include "reconcile/reconciler.h"

#include <nlohmann/json.hpp>
#include <array>
#include <chrono>
#include <future>
#include <regex>
#include <thread>
#include <unordered_set>
#include <spdlog/spdlog.h>

namespace amind {

using json = nlohmann::json;

CapturePipeline::CapturePipeline(MemoryStore& store, GraphStore& graph,
                                 TaskQueue& queue,
                                 std::shared_ptr<LLMProvider> llm,
                                 std::shared_ptr<EmbedProvider> embedder,
                                 FeatureGate* feature_gate,
                                 DerivedExtractor* derived_extractor,
                                 WriteGate* write_gate)
    : store_(store), graph_(graph), queue_(queue),
      llm_(std::move(llm)), embedder_(std::move(embedder)),
      feature_gate_(feature_gate), derived_extractor_(derived_extractor),
      write_gate_(write_gate) { (void)write_gate_; }

std::mutex& CapturePipeline::getReconcileSlotLock(uint64_t target_memory_id) {
    std::lock_guard guard(reconcile_slots_mu_);
    auto& slot = reconcile_slots_[target_memory_id];
    if (!slot) {
        slot = std::make_unique<std::mutex>();
    }
    return *slot;
}

// ── Freshness barrier implementation ────────────────────────────────────

void CapturePipeline::incrementPending(uint64_t ns_hash) {
    std::lock_guard lock(flight_mu_);
    auto& info = ns_in_flight_[ns_hash];
    info.pending++;
    info.last_submit = std::chrono::steady_clock::now();
}

void CapturePipeline::decrementPending(uint64_t ns_hash) {
    {
        std::lock_guard lock(flight_mu_);
        auto it = ns_in_flight_.find(ns_hash);
        if (it != ns_in_flight_.end() && it->second.pending > 0) {
            it->second.pending--;
        }
    }
    flight_cv_.notify_all();
}

bool CapturePipeline::waitForPendingRefinements(
    uint64_t ns_hash, std::chrono::milliseconds timeout) const {
    std::unique_lock lock(flight_mu_);
    return flight_cv_.wait_for(lock, timeout, [&]() {
        auto it = ns_in_flight_.find(ns_hash);
        return it == ns_in_flight_.end() || it->second.pending <= 0;
    });
}

bool CapturePipeline::hasFreshPending(
    uint64_t ns_hash, std::chrono::milliseconds recency) const {
    std::lock_guard lock(flight_mu_);
    auto it = ns_in_flight_.find(ns_hash);
    if (it == ns_in_flight_.end() || it->second.pending <= 0) return false;
    auto elapsed = std::chrono::steady_clock::now() - it->second.last_submit;
    return elapsed <= recency;
}

Result<std::vector<uint64_t>> CapturePipeline::capture(
    const std::string& content,
    const std::string& namespace_,
    MemoryOwner owner,
    std::map<std::string, std::string> user_metadata,
    bool pre_extracted) {

    // Early rejection of transient/noise content (greetings, acks, errors)
    // before Stage 1 store — WriteGate runs in Stage 2, too late to prevent storage.
    if (WriteGate::isTransientContent(content)) {
        spdlog::debug("CapturePipeline: rejected transient content at Stage 1: '{}'",
                      content.substr(0, 80));
        return std::vector<uint64_t>{};
    }

    // Forget/delete requests: search for matching memories and tombstone them
    // instead of storing the request itself. This is an engine-level feature
    // so every client agent (pyclaw, dingtalk, etc.) gets it for free.
    if (WriteGate::isForgetRequest(content)) {
        spdlog::info("CapturePipeline: forget request detected: '{}'",
                     content.substr(0, 80));
        const uint64_t ns_hash = MemoryRecord::hashNamespace(namespace_);
        if (embedder_) {
            auto emb = embedder_->embed(content);
            if (emb.ok()) {
                auto matches = store_.searchSimilar(*emb, 20);
                int deleted = 0;
                for (const auto& [mid, sim] : matches) {
                    if (sim < 0.30f) continue;
                    auto rec = store_.get(mid);
                    if (!rec.ok() || !rec->isAlive()) continue;
                    if (rec->namespace_hash != ns_hash) continue;
                    store_.remove(mid);
                    deleted++;
                    spdlog::info("CapturePipeline: forget — tombstoned {} (sim={:.3f}): '{}'",
                                 mid, sim, rec->content.substr(0, 60));
                }
                spdlog::info("CapturePipeline: forget complete — {} memories deleted", deleted);
            }
        }
        return std::vector<uint64_t>{};
    }

    // Stage 1 (non-blocking): store content without embedding, return ID
    // immediately. Embedding + WriteGate + HNSW insert happen in Stage 2
    // async task. This keeps the REST thread free.
    store_.registerNamespaceString(namespace_);
    MemoryRecord record;
    record.namespace_hash = MemoryRecord::hashNamespace(namespace_);
    record.content = content;
    record.owner = owner;
    record.confidence = Confidence::Inferred;
    if (!user_metadata.empty()) {
        record.user_metadata = std::move(user_metadata);
    }

    auto store_result = store_.fastStore(std::move(record));
    if (!store_result.ok()) return store_result.error();

    uint64_t memory_id = *store_result;

    // Schedule Stage 2: embed + WriteGate + HNSW insert + fact extraction
    scheduleRefinement(memory_id, namespace_, pre_extracted);

    return std::vector<uint64_t>{memory_id};
}

void CapturePipeline::scheduleRefinement(uint64_t memory_id, const std::string& namespace_,
                                         bool pre_extracted) {
    const uint64_t ns_hash = MemoryRecord::hashNamespace(namespace_);

    AsyncTask task;
    task.memory_id = memory_id;
    task.task_type = "refine";
    task.work = [this, memory_id, namespace_, ns_hash, pre_extracted]() {
        // RAII guard to always decrement pending on exit
        struct PendingGuard {
            CapturePipeline* self; uint64_t ns;
            ~PendingGuard() { self->decrementPending(ns); }
        } guard{this, ns_hash};

        if (!alive_.load(std::memory_order_relaxed)) return;

        auto record_result = store_.get(memory_id);
        if (!record_result.ok()) return;

        auto& record = record_result.value();

        // ── Stage 2a: Embed the raw memory and insert into HNSW ──
        // (moved from the REST thread to avoid blocking accept loop)
        if (embedder_ && record.embedding.empty()) {
            auto embed_result = embedder_->embed(record.content);
            if (embed_result.ok()) {
                store_.setEmbedding(memory_id, std::move(*embed_result));
                // Re-fetch with the embedding populated
                record_result = store_.get(memory_id);
                if (!record_result.ok()) return;
                record = record_result.value();
            } else {
                spdlog::warn("Stage2a: embed failed for {}: {}",
                             memory_id, embed_result.error().toString());
            }
        }

        // Extract facts via LLM (skipped when pre_extracted — content is already a clean fact)
        std::vector<ExtractedFact> fact_list;
        bool extract_failed = false;
        bool is_confidential = false;
        if (pre_extracted) {
            ExtractedFact f;
            f.content = record.content;
            f.importance = 0.5f;
            f.owner = record.owner;
            fact_list.push_back(std::move(f));
            spdlog::debug("Pre-extracted fact for memory {}: skipping LLM extractFacts", memory_id);
        } else {
            auto extraction = extractFacts(record.content);
            if (extraction.ok()) {
                // LLM-based forget: search & tombstone matching memories, discard Raw
                if (extraction->is_forget_request) {
                    spdlog::info("Stage2: LLM detected forget intent for {}: '{}'",
                                 memory_id, record.content.substr(0, 80));
                    if (embedder_ && !record.embedding.empty()) {
                        auto matches = store_.searchSimilar(record.embedding, 20);
                        int deleted = 0;
                        for (const auto& [mid, sim] : matches) {
                            if (mid == memory_id) continue;
                            if (sim < 0.30f) continue;
                            auto rec = store_.get(mid);
                            if (!rec.ok() || !rec->isAlive()) continue;
                            if (rec->namespace_hash != ns_hash) continue;
                            store_.remove(mid);
                            deleted++;
                            spdlog::info("Stage2: forget — tombstoned {} (sim={:.3f}): '{}'",
                                         mid, sim, rec->content.substr(0, 60));
                        }
                        spdlog::info("Stage2: forget complete — {} memories deleted", deleted);
                    }
                    store_.remove(memory_id);
                    return;
                }
                is_confidential = extraction->is_confidential;
                fact_list = std::move(extraction->facts);
            } else {
                extract_failed = true;
                spdlog::warn("Stage2: extractFacts failed for {}: {}",
                             memory_id, extraction.error().toString());
            }
        }
        // No extractable facts AND LLM succeeded → Raw has no memory value.
        if (fact_list.empty() && !pre_extracted && !extract_failed) {
            store_.remove(memory_id);
            spdlog::info("Stage2: removed Raw {} (no extractable facts): '{}'",
                         memory_id, record.content.substr(0, 80));
            return;
        }

        if (!fact_list.empty()) {
            spdlog::debug("Processing {} facts from memory {}", fact_list.size(), memory_id);

            // ── V2: Route through DerivedExtractor if enabled ──
            if (feature_gate_ && derived_extractor_ &&
                feature_gate_->isWriteGateEnabled()) {

                // Snapshot the raw record's audit metadata + owner so derived
                // facts inherit tenant fields and the right owner classification.
                std::map<std::string, std::string> raw_user_meta = record.user_metadata;
                if (is_confidential) {
                    raw_user_meta["confidential"] = "true";
                }
                MemoryOwner raw_owner = record.owner;

                std::vector<DerivedCandidate> candidates;
                candidates.reserve(fact_list.size());
                for (auto& fact : fact_list) {
                    DerivedCandidate candidate;
                    candidate.content = std::move(fact.content);
                    candidate.importance = fact.importance;
                    candidate.source_tier = SourceTier::Inference;
                    candidate.user_metadata = raw_user_meta;
                    candidate.owner = raw_owner;
                    if (!embedder_) continue;
                    auto embed_result = embedder_->embed(candidate.content);
                    if (embed_result.ok()) {
                        candidate.embedding = std::move(*embed_result);
                    }
                    candidates.push_back(std::move(candidate));
                }

                StoreFunc store_fn = [this](MemoryRecord rec) -> uint64_t {
                    auto result = store_.fastStore(std::move(rec));
                    return result.ok() ? *result : 0;
                };

                const uint64_t derived_ns_hash = MemoryRecord::hashNamespace(namespace_);
                SimilaritySearchFunc search_fn =
                    [this, derived_ns_hash, memory_id](const std::vector<float>& emb, size_t top_k)
                        -> std::vector<std::pair<uint64_t, float>> {
                    auto raw = store_.searchSimilar(emb, top_k * 10);
                    std::vector<std::pair<uint64_t, float>> filtered;
                    filtered.reserve(top_k);
                    for (const auto& [id, sim] : raw) {
                        if (id == memory_id) continue;  // exclude parent raw memory
                        auto rec = store_.get(id);
                        if (rec.ok() && rec->namespace_hash == derived_ns_hash) {
                            filtered.emplace_back(id, sim);
                            if (filtered.size() >= top_k) break;
                        }
                    }
                    return filtered;
                };

                auto results = derived_extractor_->processFacts(
                    memory_id, namespace_, candidates,
                    search_fn, store_fn);

                spdlog::info("Stage2 reconciler check: reconciler_={} results={}",
                             (reconciler_ ? "set" : "null"), results.size());
                for (size_t i = 0; i < results.size(); ++i) {
                    const auto& result = results[i];
                    if (result.decision == GateDecision::Accepted && result.derived_id != 0) {
                        graph_.addEdge(result.derived_id, memory_id, EdgeType::DerivedFrom, 1.0f);
                        spdlog::info("V2 derived fact stored: id={} from raw={}",
                                     result.derived_id, memory_id);
                    }
                    spdlog::info("  result[{}] decision={} derived_id={} reconcile_eligible={}",
                                 i, (int)result.decision, result.derived_id,
                                 (reconciler_ && result.decision == GateDecision::Accepted
                                  && result.derived_id != 0 && i < candidates.size()));

                    // ── Stage 2.5: Reconcile (LLM-decided ADD/REPLACE/RETRACT/REINFORCE/NOOP)
                    // Triggers on any fact that DerivedExtractor wrote — both
                    // Accepted AND Deferred. Deferred facts are exactly where
                    // Reconciler shines (looks like update/restatement of an
                    // existing fact), so we MUST consider them.
                    ReconcileDecision reconcile_decision;
                    bool reconciled = false;
                    if (reconciler_
                        && (result.decision == GateDecision::Accepted
                            || result.decision == GateDecision::Deferred)
                        && result.derived_id != 0 && i < candidates.size()) {
                        const auto& cand = candidates[i];
                        // Find strongly-similar neighbours in same namespace,
                        // EXCLUDING the just-stored derived fact itself.
                        const uint64_t derived_ns = MemoryRecord::hashNamespace(namespace_);

                        // Vector path: HNSW neighbours, namespace-filtered, derived-only.
                        std::vector<std::pair<MemoryRecord, float>> neigh;
                        std::unordered_set<uint64_t> seen_ids;
                        seen_ids.insert(result.derived_id);
                        auto raw_neigh = store_.searchSimilar(cand.embedding, 30);
                        for (const auto& [nid, sim] : raw_neigh) {
                            if (seen_ids.count(nid)) continue;
                            auto rec = store_.get(nid);
                            if (!rec.ok() || rec->namespace_hash != derived_ns) continue;
                            if (!rec->isAlive()) continue;
                            if (rec->layer != MemoryLayer::Derived) continue;
                            seen_ids.insert(nid);
                            neigh.emplace_back(std::move(*rec), sim);
                            if (neigh.size() >= 5) break;
                        }

                        // Keyword fallback path (CJK-friendly): scan all derived
                        // records in the same namespace, score by char-bigram
                        // overlap with the candidate content. This catches
                        // entity-swap cases (e.g. "古力娜扎" vs "迪丽热巴")
                        // where embeddings score < 0.5 but the surrounding
                        // template is identical.
                        if (neigh.size() < 5) {
                            // Build candidate bigram set once.
                            std::unordered_set<std::string> cand_bg;
                            {
                                auto utf8 = [](const std::string& s) {
                                    std::vector<std::string> cs;
                                    for (size_t k = 0; k < s.size(); ) {
                                        unsigned char c = (unsigned char)s[k];
                                        size_t L = 1;
                                        if ((c & 0xE0) == 0xC0) L = 2;
                                        else if ((c & 0xF0) == 0xE0) L = 3;
                                        else if ((c & 0xF8) == 0xF0) L = 4;
                                        if (k + L > s.size()) break;
                                        cs.emplace_back(s.substr(k, L));
                                        k += L;
                                    }
                                    return cs;
                                };
                                auto chars = utf8(cand.content);
                                for (size_t k = 0; k + 1 < chars.size(); ++k) {
                                    cand_bg.insert(chars[k] + chars[k + 1]);
                                }
                            }

                            // Scan namespace for derived facts and rank by overlap.
                            std::vector<std::tuple<float, MemoryRecord>> kw_hits;
                            store_.scanAll([&](const MemoryRecord& r) {
                                if (seen_ids.count(r.memory_id)) return;
                                if (r.namespace_hash != derived_ns) return;
                                if (!r.isAlive()) return;
                                if (r.layer != MemoryLayer::Derived) return;
                                if (cand_bg.empty()) return;

                                // Build content bigrams + count overlap.
                                std::unordered_set<std::string> content_bg;
                                {
                                    std::vector<std::string> chars;
                                    for (size_t k = 0; k < r.content.size(); ) {
                                        unsigned char c = (unsigned char)r.content[k];
                                        size_t L = 1;
                                        if ((c & 0xE0) == 0xC0) L = 2;
                                        else if ((c & 0xF0) == 0xE0) L = 3;
                                        else if ((c & 0xF8) == 0xF0) L = 4;
                                        if (k + L > r.content.size()) break;
                                        chars.emplace_back(r.content.substr(k, L));
                                        k += L;
                                    }
                                    for (size_t k = 0; k + 1 < chars.size(); ++k) {
                                        content_bg.insert(chars[k] + chars[k + 1]);
                                    }
                                }
                                size_t hit = 0;
                                for (const auto& bg : cand_bg) {
                                    if (content_bg.count(bg)) hit++;
                                }
                                float overlap = (float)hit / (float)cand_bg.size();
                                if (overlap >= 0.30f) {
                                    kw_hits.emplace_back(overlap, r);
                                }
                            });
                            // Take top by overlap, fill up to 5 total neighbours.
                            std::sort(kw_hits.begin(), kw_hits.end(),
                                      [](const auto& a, const auto& b) {
                                          return std::get<0>(a) > std::get<0>(b);
                                      });
                            for (auto& [score, rec] : kw_hits) {
                                if (seen_ids.count(rec.memory_id)) continue;
                                seen_ids.insert(rec.memory_id);
                                neigh.emplace_back(std::move(rec), score);
                                if (neigh.size() >= 5) break;
                            }
                        }

                        // ── Temporal-aggregate sweep ───────────────────────
                        // Time-bound assertions like "用户最近执行的工单ID是TKT001"
                        // are written when X was the latest, but become stale
                        // when a newer same-domain fact (e.g. "TKT005 已执行成功")
                        // arrives. They have low cosine similarity to atomic
                        // facts so the regular HNSW search misses them. Scan
                        // namespace for memories containing temporal markers
                        // and pull up to 3 into the reconciler's decision pool.
                        static const std::array<const char*, 6> kTemporalMarkers = {
                            "最近", "当前", "目前", "最新", "现在", "刚才"
                        };
                        auto looksTemporal = [&](const std::string& s) {
                            for (auto m : kTemporalMarkers) {
                                if (s.find(m) != std::string::npos) return true;
                            }
                            return false;
                        };
                        // Only sweep if the new fact looks "completion-flavored":
                        // emitting a temporal neighbour for an unrelated atomic
                        // would just waste an LLM call.
                        static const std::array<const char*, 6> kCompletionMarkers = {
                            "已执行", "已完成", "已上线", "执行成功", "已发布", "已部署"
                        };
                        bool cand_is_completion = false;
                        for (auto m : kCompletionMarkers) {
                            if (cand.content.find(m) != std::string::npos) {
                                cand_is_completion = true; break;
                            }
                        }
                        if (cand_is_completion) {
                            std::vector<std::tuple<uint32_t, MemoryRecord>> temporal_hits;
                            store_.scanAll([&](const MemoryRecord& r) {
                                if (seen_ids.count(r.memory_id)) return;
                                if (r.namespace_hash != derived_ns) return;
                                if (!r.isAlive()) return;
                                if (r.layer != MemoryLayer::Derived) return;
                                if (!looksTemporal(r.content)) return;
                                temporal_hits.emplace_back(r.created_at, r);
                            });
                            // Newest temporal aggregates first — they're most
                            // likely to be the load-bearing summary that needs
                            // updating. Cap at 3 to bound LLM cost.
                            std::sort(temporal_hits.begin(), temporal_hits.end(),
                                      [](const auto& a, const auto& b) {
                                          return std::get<0>(a) > std::get<0>(b);
                                      });
                            for (auto& [ts, rec] : temporal_hits) {
                                if (neigh.size() >= 5) break;
                                if (seen_ids.count(rec.memory_id)) continue;
                                seen_ids.insert(rec.memory_id);
                                // Sentinel similarity 0.50 — recognizable in
                                // logs as "temporal-injected" rather than a
                                // real cosine hit.
                                neigh.emplace_back(std::move(rec), 0.50f);
                            }
                        }

                        spdlog::info("Reconciler hook: cand={} → neigh.size={}",
                                     cand.content.substr(0, 60), neigh.size());
                        for (size_t k = 0; k < neigh.size(); ++k) {
                            spdlog::info("  pre-decide neigh[{}] id={} sim={:.3f} content={}",
                                         k, neigh[k].first.memory_id, neigh[k].second,
                                         neigh[k].first.content.substr(0, 80));
                        }

                        // ── Per-key serialization ──────────────────────────
                        // If neighbours exist, lock on the strongest neighbour's
                        // memory_id so concurrent updates to the same fact
                        // (e.g. 5 rapid weight changes) execute sequentially.
                        // Unrelated facts (different target) run fully parallel.
                        std::unique_lock<std::mutex> slot_guard;
                        if (!neigh.empty()) {
                            uint64_t slot_key = neigh.front().first.memory_id;
                            slot_guard = std::unique_lock<std::mutex>(
                                getReconcileSlotLock(slot_key));

                            // Re-check: the target may have been tombstoned by
                            // a prior concurrent reconcile that finished while
                            // we waited for the lock. Refresh the neighbour.
                            auto refreshed = store_.get(slot_key);
                            if (refreshed.ok() && !refreshed->isAlive()) {
                                // Target was already superseded — re-search for
                                // the current live version before deciding.
                                neigh.clear();
                                seen_ids.clear();
                                seen_ids.insert(result.derived_id);
                                auto fresh_neigh = store_.searchSimilar(cand.embedding, 30);
                                for (const auto& [nid, sim] : fresh_neigh) {
                                    if (seen_ids.count(nid)) continue;
                                    auto rec = store_.get(nid);
                                    if (!rec.ok() || rec->namespace_hash != derived_ns) continue;
                                    if (!rec->isAlive()) continue;
                                    if (rec->layer != MemoryLayer::Derived) continue;
                                    seen_ids.insert(nid);
                                    neigh.emplace_back(std::move(*rec), sim);
                                    if (neigh.size() >= 5) break;
                                }
                                spdlog::info("Reconciler slot re-search after lock: neigh.size={}",
                                             neigh.size());
                            }
                        }

                        reconcile_decision = reconciler_->decide(cand.content, neigh);
                        reconciled = true;

                        // Apply non-ADD ops
                        switch (reconcile_decision.op) {
                            case ReconcileOp::ADD:
                                // Keep derived as-is.
                                break;
                            case ReconcileOp::REPLACE: {
                                // Tombstone the older fact + link new.parent_id = old.
                                auto old_rec = store_.get(reconcile_decision.target_id);
                                store_.remove(reconcile_decision.target_id);
                                graph_.addEdge(result.derived_id,
                                               reconcile_decision.target_id,
                                               EdgeType::Supersedes, 1.0f);
                                spdlog::info("Reconciler REPLACE: {} ({}) supersedes {} — {}",
                                             result.derived_id,
                                             cand.content.substr(0, 40),
                                             reconcile_decision.target_id,
                                             reconcile_decision.rationale);

                                // ── Cascade: tombstone sibling memories that
                                // reference the same old value. Parallel LLM
                                // calls — total time = max(single) not sum(all).
                                if (old_rec.ok() && !old_rec->embedding.empty()) {
                                    auto siblings = store_.searchSimilar(old_rec->embedding, 20);

                                    // Collect eligible siblings first
                                    struct SiblingInfo { uint64_t id; float sim; MemoryRecord rec; };
                                    std::vector<SiblingInfo> eligible;
                                    for (const auto& [sid, sim] : siblings) {
                                        if (sid == reconcile_decision.target_id) continue;
                                        if (sid == result.derived_id) continue;
                                        if (sid >= result.derived_id) continue;
                                        if (sim < 0.35f) continue;
                                        auto srec = store_.get(sid);
                                        if (!srec.ok() || !srec->isAlive()) continue;
                                        if (srec->namespace_hash != derived_ns) continue;
                                        if (srec->layer != MemoryLayer::Derived) continue;
                                        eligible.push_back({sid, sim, std::move(*srec)});
                                    }

                                    // Fire LLM calls in parallel
                                    std::vector<std::future<ReconcileDecision>> futures;
                                    futures.reserve(eligible.size());
                                    for (auto& sib : eligible) {
                                        std::vector<std::pair<MemoryRecord, float>> sib_pair;
                                        sib_pair.emplace_back(sib.rec, sib.sim);
                                        futures.push_back(std::async(std::launch::async,
                                            [this, content = cand.content, pair = std::move(sib_pair)]() {
                                                return reconciler_->decide(content, pair);
                                            }));
                                    }

                                    // Collect results and tombstone
                                    for (size_t fi = 0; fi < futures.size(); ++fi) {
                                        auto sib_decision = futures[fi].get();
                                        if (sib_decision.op == ReconcileOp::REPLACE
                                            || sib_decision.op == ReconcileOp::RETRACT) {
                                            store_.remove(eligible[fi].id);
                                            spdlog::info("Reconciler CASCADE: tombstoned sibling {} (sim={:.3f}) — {}",
                                                         eligible[fi].id, eligible[fi].sim, sib_decision.rationale);
                                        }
                                    }
                                }
                                break;
                            }
                            case ReconcileOp::RETRACT: {
                                // Tombstone target AND the just-stored fact
                                // (the candidate is a negation, not new info).
                                auto retract_rec = store_.get(reconcile_decision.target_id);
                                store_.remove(reconcile_decision.target_id);
                                store_.remove(result.derived_id);
                                spdlog::info("Reconciler RETRACT: removed both {} and just-stored {} — {}",
                                             reconcile_decision.target_id, result.derived_id,
                                             reconcile_decision.rationale);

                                // Cascade: tombstone siblings of the retracted fact (parallel)
                                if (retract_rec.ok() && !retract_rec->embedding.empty()) {
                                    auto siblings = store_.searchSimilar(retract_rec->embedding, 20);

                                    struct SiblingInfo { uint64_t id; float sim; MemoryRecord rec; };
                                    std::vector<SiblingInfo> eligible;
                                    for (const auto& [sid, sim] : siblings) {
                                        if (sid == reconcile_decision.target_id) continue;
                                        if (sid == result.derived_id) continue;
                                        if (sid >= result.derived_id) continue;
                                        if (sim < 0.35f) continue;
                                        auto srec = store_.get(sid);
                                        if (!srec.ok() || !srec->isAlive()) continue;
                                        if (srec->namespace_hash != derived_ns) continue;
                                        if (srec->layer != MemoryLayer::Derived) continue;
                                        eligible.push_back({sid, sim, std::move(*srec)});
                                    }

                                    std::vector<std::future<ReconcileDecision>> futures;
                                    futures.reserve(eligible.size());
                                    for (auto& sib : eligible) {
                                        std::vector<std::pair<MemoryRecord, float>> sib_pair;
                                        sib_pair.emplace_back(sib.rec, sib.sim);
                                        futures.push_back(std::async(std::launch::async,
                                            [this, content = cand.content, pair = std::move(sib_pair)]() {
                                                return reconciler_->decide(content, pair);
                                            }));
                                    }

                                    for (size_t fi = 0; fi < futures.size(); ++fi) {
                                        auto sib_decision = futures[fi].get();
                                        if (sib_decision.op == ReconcileOp::REPLACE
                                            || sib_decision.op == ReconcileOp::RETRACT) {
                                            store_.remove(eligible[fi].id);
                                            spdlog::info("Reconciler CASCADE(retract): tombstoned sibling {} (sim={:.3f}) — {}",
                                                         eligible[fi].id, eligible[fi].sim, sib_decision.rationale);
                                        }
                                    }
                                }
                                break;
                            }
                            case ReconcileOp::REINFORCE: {
                                // Drop the just-stored fact; bump importance of
                                // the existing one (repeated mention = stronger signal).
                                store_.remove(result.derived_id);
                                store_.boostImportance(reconcile_decision.target_id, 0.1f);
                                spdlog::info("Reconciler REINFORCE: dropped {}, boosted existing {} (+0.1) — {}",
                                             result.derived_id, reconcile_decision.target_id,
                                             reconcile_decision.rationale);
                                break;
                            }
                            case ReconcileOp::NOOP: {
                                store_.remove(result.derived_id);
                                spdlog::info("Reconciler NOOP: dropped {} (already known) — {}",
                                             result.derived_id, reconcile_decision.rationale);
                                break;
                            }
                        }
                        // slot_guard released here (RAII) — next concurrent
                        // reconcile for the same target can proceed.
                    }

                    // Audit each derived verdict to GateLog (with reconcile op)
                    if (gate_log_ && i < candidates.size()) {
                        const auto& cand = candidates[i];
                        GateLogEntry entry;
                        entry.namespace_ = namespace_;
                        entry.content = cand.content;
                        entry.embedding = cand.embedding;
                        entry.decision = result.decision;
                        entry.reason = result.reason;
                        entry.owner = cand.owner;
                        entry.layer = MemoryLayer::Derived;
                        entry.user_metadata = cand.user_metadata;
                        entry.memory_id = result.derived_id;
                        std::smatch m;
                        std::regex dup_re(R"(Near-duplicate of memory #(\d+))");
                        if (std::regex_search(result.reason, m, dup_re) && m.size() >= 2) {
                            try { entry.conflict_with_id = std::stoull(m[1].str()); } catch (...) {}
                        }
                        if (reconciled) {
                            entry.reconcile_op = reconcileOpToString(reconcile_decision.op);
                            entry.reconcile_target_id = reconcile_decision.target_id;
                            entry.reconcile_rationale = reconcile_decision.rationale;
                        }
                        gate_log_->append(entry);
                    }
                }
            }
        }

        // Graph edge creation: Related + ConflictsWith.
        // Edges only make sense within the same namespace — alice and bob's
        // contradicting beliefs should not be linked as ConflictsWith.
        if (!record.embedding.empty()) {
            const uint64_t edge_ns_hash = record.namespace_hash;
            auto raw_neighbors = store_.searchSimilar(record.embedding, 50);
            std::vector<std::pair<uint64_t, float>> neighbors;
            neighbors.reserve(10);
            for (const auto& [nid, score] : raw_neighbors) {
                auto rec = store_.get(nid);
                if (rec.ok() && rec->namespace_hash == edge_ns_hash) {
                    neighbors.emplace_back(nid, score);
                    if (neighbors.size() >= 10) break;
                }
            }
            for (const auto& [neighbor_id, score] : neighbors) {
                if (neighbor_id == memory_id) continue;
                if (score >= 0.85f) {
                    graph_.addEdge(memory_id, neighbor_id, EdgeType::ConflictsWith, score);
                    spdlog::info("Conflict detected: {} <-> {} (score={})",
                                 memory_id, neighbor_id, score);
                } else if (score >= 0.5f) {
                    graph_.addEdge(memory_id, neighbor_id, EdgeType::Related, score);
                }
            }
        }
    };

    // Freshness barrier: increment before push so recall can detect in-flight work
    incrementPending(ns_hash);

    auto push_result = queue_.push(std::move(task));
    if (!push_result.ok()) {
        // Push failed — decrement immediately since the task won't run
        decrementPending(ns_hash);
        spdlog::warn("Failed to schedule refinement for memory {}: {}",
                     memory_id, push_result.error().toString());
    }
}

Result<std::vector<uint64_t>> CapturePipeline::interceptCapture(
    const std::vector<std::pair<std::string, std::string>>& messages,
    const std::string& namespace_) {

    std::vector<uint64_t> all_ids;

    // Capture assistant messages (they contain the knowledge)
    for (const auto& [role, content] : messages) {
        if (role == "assistant" || role == "user") {
            auto result = capture(content, namespace_, MemoryOwner::Session);
            if (result.ok()) {
                all_ids.insert(all_ids.end(), result->begin(), result->end());
            }
        }
    }
    return all_ids;
}

Result<ExtractionResult> CapturePipeline::extractFacts(const std::string& content) {
    if (!llm_) return makeError(Error::ProviderError, "no LLM provider configured");

    std::string prompt =
        "Analyze the following text and return a JSON object with these fields:\n"
        "1. \"intent\": one of \"store\", \"forget\", \"confidential_store\"\n"
        "   - \"forget\": user wants to delete/forget previously stored information\n"
        "   - \"confidential_store\": user is sharing info but explicitly asks to keep it secret\n"
        "   - \"store\": normal information to remember (default)\n"
        "2. \"facts\": array of extracted facts (empty array if intent is \"forget\")\n"
        "   Each fact: {\"content\": \"...\", \"owner\": \"user/project/agent\", \"importance\": 0-1}\n\n"
        "Fact extraction rules:\n"
        "- State the CURRENT state as a simple assertion "
        "(e.g. '用户的手机号是139-9999-8888' NOT '用户的新手机号是...')\n"
        "- Do NOT use words like 新/旧/之前/曾经/原来 — just state what IS true now\n"
        "- When the text expresses a PREFERENCE switch (e.g. 不再喜欢X, 改成Z), "
        "extract as '用户最喜欢的<类别>是Y'\n\n"
        "Text: " + content;

    constexpr int MAX_RETRIES = 3;
    constexpr int BASE_BACKOFF_MS = 3000;
    for (int attempt = 0; attempt <= MAX_RETRIES; ++attempt) {
        auto llm_result = llm_->generateJson(prompt);
        if (llm_result.ok()) {
            try {
                auto j = json::parse(*llm_result);
                ExtractionResult result;

                // Parse intent
                std::string intent = j.value("intent", "store");
                result.is_forget_request = (intent == "forget");
                result.is_confidential = (intent == "confidential_store");

                // Parse facts (handle both top-level object and legacy array format)
                auto& facts_json = j.contains("facts") ? j["facts"] : j;
                if (facts_json.is_array()) {
                    for (const auto& item : facts_json) {
                        ExtractedFact fact;
                        fact.content = item.value("content", "");
                        fact.importance = item.value("importance", 0.5f);
                        if (!fact.content.empty()) {
                            result.facts.push_back(std::move(fact));
                        }
                    }
                }
                return result;
            } catch (const std::exception& e) {
                return makeError(Error::InternalError, "JSON parse failed: " + std::string(e.what()));
            }
        }
        auto err_msg = llm_result.error().toString();
        bool is_rate_limit = err_msg.find("429") != std::string::npos
                          || err_msg.find("限流") != std::string::npos
                          || err_msg.find("rate") != std::string::npos
                          || err_msg.find("Too many") != std::string::npos;
        if (!is_rate_limit || attempt == MAX_RETRIES) {
            return llm_result.error();
        }
        int wait_ms = BASE_BACKOFF_MS * (1 << attempt);
        spdlog::warn("extractFacts rate-limited, retry {}/{} in {}ms",
                     attempt + 1, MAX_RETRIES, wait_ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
    }
    return ExtractionResult{};
}

bool CapturePipeline::isDuplicate(const std::vector<float>& embedding, float threshold) {
    auto similar = store_.searchSimilar(embedding, 1);
    return !similar.empty() && similar[0].second >= threshold;
}

}  // namespace amind
