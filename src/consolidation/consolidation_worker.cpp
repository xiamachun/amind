
#include "consolidation_worker.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <spdlog/spdlog.h>

namespace amind {

ConsolidationWorker::ConsolidationWorker(ConsolidationConfig config)
    : config_(std::move(config)) {}

std::vector<uint64_t> ConsolidationWorker::promoteTopK(
    std::vector<MemoryRecord>& session_memories) {

    // Sort by importance * (1 + log(1 + access_count)) descending
    std::sort(session_memories.begin(), session_memories.end(),
              [](const MemoryRecord& a, const MemoryRecord& b) {
                  float score_a = a.importance * (1.0f + std::log(1.0f + static_cast<float>(a.access_count)));
                  float score_b = b.importance * (1.0f + std::log(1.0f + static_cast<float>(b.access_count)));
                  return score_a > score_b;
              });

    std::vector<uint64_t> promoted_ids;
    size_t promote_count = std::min(config_.top_k_per_session, session_memories.size());

    for (size_t i = 0; i < promote_count; ++i) {
        auto& record = session_memories[i];
        if (record.confidence_level == Confidence::Inferred) {
            record.scope = MemoryScope::Private;
            record.memory_type = MemoryType::UserProfile;
            record.tier = MemoryTier::Consolidated;
            record.confidence_level = Confidence::Verified;
            promoted_ids.push_back(record.memory_id);
        }
    }

    return promoted_ids;
}

std::vector<DedupResult> ConsolidationWorker::dedup(
    std::vector<MemoryRecord>& memories,
    const CosineSimilarityFunc& cosine_sim) {

    std::vector<DedupResult> results;
    std::vector<bool> archived(memories.size(), false);

    // Precompute L2 norms for early termination
    std::vector<float> norms(memories.size(), 0.0f);
    for (size_t i = 0; i < memories.size(); ++i) {
        if (memories[i].embedding.empty()) continue;
        float sum = 0.0f;
        for (float v : memories[i].embedding) sum += v * v;
        norms[i] = std::sqrt(sum);
    }

    // Sort indices by norm for locality-based pruning
    std::vector<size_t> indices(memories.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&norms](size_t a, size_t b) { return norms[a] < norms[b]; });

    // Max cosine similarity between vectors with norms n1, n2 is bounded by min(n1,n2)/max(n1,n2)
    // If that bound < threshold, skip the pair
    float threshold = config_.dedup_threshold;

    for (size_t ii = 0; ii < indices.size(); ++ii) {
        size_t i = indices[ii];
        if (archived[i]) continue;
        if (memories[i].embedding.empty()) continue;

        DedupResult group;
        group.kept_id = memories[i].memory_id;

        for (size_t jj = ii + 1; jj < indices.size(); ++jj) {
            size_t j = indices[jj];
            if (archived[j]) continue;
            if (memories[j].embedding.empty()) continue;

            // Norm-based early termination: if norm ratio is too extreme, similarity can't reach threshold
            if (norms[i] > 0.0f && norms[j] > 0.0f) {
                float ratio = norms[i] / norms[j];  // i <= j by sort order
                if (ratio < threshold) break;       // all further j have larger norms
            }

            float sim = cosine_sim(memories[i].embedding, memories[j].embedding);
            if (sim >= threshold) {
                if (memories[j].importance > memories[i].importance) {
                    group.archived_ids.push_back(group.kept_id);
                    group.kept_id = memories[j].memory_id;
                    memories[i].phase = MemoryPhase::Archived;
                    archived[i] = true;
                } else {
                    group.archived_ids.push_back(memories[j].memory_id);
                    memories[j].phase = MemoryPhase::Archived;
                    archived[j] = true;
                }
            }
        }

        if (!group.archived_ids.empty()) {
            results.push_back(std::move(group));
        }
    }

    return results;
}

std::vector<DriftCheckResult> ConsolidationWorker::checkDrift(
    const std::vector<std::pair<MemoryRecord, MemoryRecord>>& derived_raw_pairs,
    const CosineSimilarityFunc& cosine_sim) {

    std::vector<DriftCheckResult> results;

    for (const auto& [derived, raw] : derived_raw_pairs) {
        DriftCheckResult check;
        check.memory_id = derived.memory_id;

        if (derived.embedding.empty() || raw.embedding.empty()) {
            check.drift_score = 0.0f;
            check.reason = "missing embedding, skipped";
            results.push_back(std::move(check));
            continue;
        }

        float similarity = cosine_sim(derived.embedding, raw.embedding);
        check.drift_score = 1.0f - similarity;

        if (check.drift_score > config_.drift_threshold) {
            check.invalidated = true;
            check.reason = "drift " + std::to_string(check.drift_score)
                         + " > threshold " + std::to_string(config_.drift_threshold);
            spdlog::info("ConsolidationWorker: drift detected for memory {} (drift={})",
                         derived.memory_id, check.drift_score);
        } else {
            check.reason = "within drift tolerance";
        }

        results.push_back(std::move(check));
    }

    return results;
}

ConsolidationCycleResult ConsolidationWorker::runCycle(
    std::vector<std::vector<MemoryRecord>>& session_groups,
    const CosineSimilarityFunc& cosine_sim,
    const std::vector<std::pair<MemoryRecord, MemoryRecord>>& drift_candidates) {

    ConsolidationCycleResult cycle;

    // Phase 1: Promote top-K per session
    for (auto& session : session_groups) {
        auto promoted = promoteTopK(session);
        cycle.memories_promoted += promoted.size();
        cycle.sessions_processed++;
    }

    // Phase 2: Cross-session dedup (flatten all memories)
    std::vector<MemoryRecord> all_memories;
    for (auto& session : session_groups) {
        all_memories.insert(all_memories.end(),
                            std::make_move_iterator(session.begin()),
                            std::make_move_iterator(session.end()));
    }

    auto dedup_results = dedup(all_memories, cosine_sim);
    for (const auto& group : dedup_results) {
        cycle.memories_deduped += group.archived_ids.size();
    }

    // Phase 3: Drift check
    auto drift_results = checkDrift(drift_candidates, cosine_sim);
    cycle.drift_checked = drift_results.size();
    for (const auto& check : drift_results) {
        if (check.invalidated) cycle.drift_invalidated++;
    }

    // Update stats
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.cycles_run++;
        stats_.total_promoted += cycle.memories_promoted;
        stats_.total_deduped += cycle.memories_deduped;
        stats_.total_drift_checked += cycle.drift_checked;
        stats_.total_drift_invalidated += cycle.drift_invalidated;
    }

    spdlog::info("ConsolidationWorker::runCycle: {} sessions, {} promoted, {} deduped, "
                 "{} drift checked ({} invalidated)",
                 cycle.sessions_processed, cycle.memories_promoted, cycle.memories_deduped,
                 cycle.drift_checked, cycle.drift_invalidated);

    return cycle;
}

ConsolidationConfig ConsolidationWorker::config() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

ConsolidationWorker::Stats ConsolidationWorker::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void ConsolidationWorker::resetStats() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = {};
}

}  // namespace amind
