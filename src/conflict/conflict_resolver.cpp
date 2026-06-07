
#include "conflict_resolver.h"

#include <spdlog/spdlog.h>

namespace amind {

ConflictResolver::ConflictResolver(ConflictResolverConfig config)
    : config_(std::move(config)) {}

ConflictResolution ConflictResolver::resolve(const MemoryRecord& record_a,
                                              const MemoryRecord& record_b) {
    // Try each level in order
    if (auto result = tryVersionAncestor(record_a, record_b)) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.total_resolved++;
        stats_.by_version++;
        return *result;
    }

    if (auto result = tryConfidence(record_a, record_b)) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.total_resolved++;
        stats_.by_confidence++;
        return *result;
    }

    if (auto result = tryRecency(record_a, record_b)) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.total_resolved++;
        stats_.by_recency++;
        return *result;
    }

    if (auto result = trySourceTier(record_a, record_b)) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.total_resolved++;
        stats_.by_source_tier++;
        return *result;
    }

    if (config_.enable_llm_judge) {
        if (auto result = tryLlmJudge(record_a, record_b)) {
            std::lock_guard<std::mutex> lock(mutex_);
            stats_.total_resolved++;
            stats_.by_llm++;
            return *result;
        }
    }

    // L6: Coexist
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.total_resolved++;
        stats_.by_coexist++;
    }
    return coexist(record_a, record_b);
}

void ConflictResolver::setLlmJudge(LlmJudgeFunc judge) {
    std::lock_guard<std::mutex> lock(mutex_);
    llm_judge_ = std::move(judge);
}

ConflictResolver::Stats ConflictResolver::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void ConflictResolver::resetStats() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = {};
}

ConflictResolverConfig ConflictResolver::config() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

// ── L1: Version Ancestor ─────────────────────────────────────────────────

std::optional<ConflictResolution> ConflictResolver::tryVersionAncestor(
    const MemoryRecord& a, const MemoryRecord& b) {
    // If A is parent of B, A is the ancestor → A is loser (B supersedes A)
    if (b.parent_id == a.memory_id) {
        return ConflictResolution{
            .winner_id = b.memory_id,
            .loser_id = a.memory_id,
            .level = ResolutionLevel::VersionAncestor,
            .reason = "B is newer version of A (parent_id chain)"
        };
    }
    // If B is parent of A, B is the ancestor → B is loser
    if (a.parent_id == b.memory_id) {
        return ConflictResolution{
            .winner_id = a.memory_id,
            .loser_id = b.memory_id,
            .level = ResolutionLevel::VersionAncestor,
            .reason = "A is newer version of B (parent_id chain)"
        };
    }
    return std::nullopt;
}

// ── L2: Confidence ───────────────────────────────────────────────────────

int ConflictResolver::confidenceRank(Confidence conf) {
    switch (conf) {
        case Confidence::Verified:   return 4;
        case Confidence::Inferred:   return 3;
        case Confidence::Stale:      return 2;
        case Confidence::Conflicted: return 1;
    }
    return 0;
}

std::optional<ConflictResolution> ConflictResolver::tryConfidence(
    const MemoryRecord& a, const MemoryRecord& b) {
    int rank_a = confidenceRank(a.confidence_level);
    int rank_b = confidenceRank(b.confidence_level);

    if (rank_a == rank_b) return std::nullopt;

    bool a_wins = rank_a > rank_b;
    return ConflictResolution{
        .winner_id = a_wins ? a.memory_id : b.memory_id,
        .loser_id = a_wins ? b.memory_id : a.memory_id,
        .level = ResolutionLevel::Confidence,
        .reason = std::string(a_wins ? "A" : "B") + " has higher confidence ("
                  + confidenceToString(a_wins ? a.confidence_level : b.confidence_level) + " > "
                  + confidenceToString(a_wins ? b.confidence_level : a.confidence_level) + ")"
    };
}

// ── L3: Recency ──────────────────────────────────────────────────────────

std::optional<ConflictResolution> ConflictResolver::tryRecency(
    const MemoryRecord& a, const MemoryRecord& b) {
    if (a.created_at == b.created_at) return std::nullopt;

    bool a_newer = a.created_at > b.created_at;
    return ConflictResolution{
        .winner_id = a_newer ? a.memory_id : b.memory_id,
        .loser_id = a_newer ? b.memory_id : a.memory_id,
        .level = ResolutionLevel::Recency,
        .reason = std::string(a_newer ? "A" : "B") + " is more recent"
    };
}

// ── L4: SourceTier ───────────────────────────────────────────────────────

int ConflictResolver::sourceTierRank(SourceTier tier) {
    switch (tier) {
        case SourceTier::Behavioral: return 3;
        case SourceTier::Assertion:  return 2;
        case SourceTier::Inference:  return 1;
    }
    return 0;
}

std::optional<ConflictResolution> ConflictResolver::trySourceTier(
    const MemoryRecord& a, const MemoryRecord& b) {
    int rank_a = sourceTierRank(a.source_tier);
    int rank_b = sourceTierRank(b.source_tier);

    if (rank_a == rank_b) return std::nullopt;

    bool a_wins = rank_a > rank_b;
    return ConflictResolution{
        .winner_id = a_wins ? a.memory_id : b.memory_id,
        .loser_id = a_wins ? b.memory_id : a.memory_id,
        .level = ResolutionLevel::SourceTier,
        .reason = std::string(a_wins ? "A" : "B") + " has higher source tier ("
                  + sourceTierToString(a_wins ? a.source_tier : b.source_tier) + " > "
                  + sourceTierToString(a_wins ? b.source_tier : a.source_tier) + ")"
    };
}

// ── L5: LLM Judge ────────────────────────────────────────────────────────

std::optional<ConflictResolution> ConflictResolver::tryLlmJudge(
    const MemoryRecord& a, const MemoryRecord& b) {
    if (!llm_judge_) return std::nullopt;

    auto winner = llm_judge_(a.memory_id, a.content, b.memory_id, b.content);
    if (!winner.has_value()) return std::nullopt;

    uint64_t winner_id = *winner;
    uint64_t loser_id = (winner_id == a.memory_id) ? b.memory_id : a.memory_id;

    return ConflictResolution{
        .winner_id = winner_id,
        .loser_id = loser_id,
        .level = ResolutionLevel::LlmJudge,
        .reason = "LLM judge determined winner"
    };
}

// ── L6: Coexist ──────────────────────────────────────────────────────────

ConflictResolution ConflictResolver::coexist(const MemoryRecord& a,
                                              const MemoryRecord& b) {
    spdlog::info("ConflictResolver: L6 coexist for memories {} and {}", a.memory_id, b.memory_id);
    return ConflictResolution{
        .winner_id = a.memory_id,
        .loser_id = b.memory_id,
        .level = ResolutionLevel::Coexist,
        .reason = "All strategies tied — both memories coexist",
        .coexist = true
    };
}

}  // namespace amind
