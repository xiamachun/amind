
#include "write_gate.h"

#include <algorithm>
#include <cmath>
#include <regex>
#include <spdlog/spdlog.h>

namespace amind {

WriteGate::WriteGate(WriteGateConfig config)
    : config_(std::move(config)) {}

GateVerdict WriteGate::evaluate(const ProposedMemory& candidate,
                                 const SimilaritySearchFunc& search_fn) {
    GateVerdict verdict;

    // Step 1: Classify source tier
    verdict.tier = classifySourceTier(candidate);

    // Step 2: Check for transient/noise content
    if (isTransientContent(candidate.content)) {
        verdict.decision = GateDecision::Rejected;
        verdict.marginal_value = 0.0f;
        verdict.reason = "Transient/noise content (greeting, acknowledgment, etc.)";

        std::lock_guard<std::mutex> lock(mutex_);
        stats_.total_evaluated++;
        if (config_.shadow_mode) {
            stats_.shadow_overrides++;
            verdict.decision = GateDecision::Accepted;
            verdict.reason += " [shadow: would reject]";
        } else {
            stats_.rejected++;
        }
        spdlog::debug("WriteGate: {} content='{}' reason='{}'",
                      gateDecisionToString(verdict.decision), candidate.content, verdict.reason);
        return verdict;
    }

    // Step 3: Search for similar existing memories
    std::vector<std::pair<uint64_t, float>> neighbors;
    if (search_fn && !candidate.embedding.empty()) {
        neighbors = search_fn(candidate.embedding, config_.similarity_top_k);
    }

    // Step 4: Check for exact/near duplicates
    for (const auto& [id, similarity] : neighbors) {
        if (similarity >= config_.duplicate_threshold) {
            verdict.decision = GateDecision::Rejected;
            verdict.marginal_value = 0.0f;
            verdict.reason = "Near-duplicate of memory #" + std::to_string(id)
                           + " (similarity=" + std::to_string(similarity) + ")";

            std::lock_guard<std::mutex> lock(mutex_);
            stats_.total_evaluated++;
            if (config_.shadow_mode) {
                stats_.shadow_overrides++;
                verdict.decision = GateDecision::Accepted;
                verdict.reason += " [shadow: would reject]";
            } else {
                stats_.rejected++;
            }
            spdlog::debug("WriteGate: {} duplicate detected, reason='{}'",
                          gateDecisionToString(verdict.decision), verdict.reason);
            return verdict;
        }
    }

    // Step 5: Compute marginal value
    verdict.marginal_value = computeMarginalValue(candidate, neighbors);

    // Step 6: Make decision based on marginal value
    if (verdict.marginal_value < config_.low_value_threshold) {
        verdict.decision = GateDecision::Rejected;
        verdict.reason = "Low marginal value (" + std::to_string(verdict.marginal_value) + ")";
    } else if (verdict.marginal_value < config_.deferred_threshold) {
        verdict.decision = GateDecision::Deferred;
        verdict.reason = "Uncertain marginal value (" + std::to_string(verdict.marginal_value) + ")";
    } else {
        verdict.decision = GateDecision::Accepted;
        verdict.reason = "Sufficient marginal value (" + std::to_string(verdict.marginal_value) + ")";
    }

    // Apply shadow mode override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.total_evaluated++;
        if (config_.shadow_mode && verdict.decision != GateDecision::Accepted) {
            stats_.shadow_overrides++;
            std::string original_reason = verdict.reason;
            verdict.decision = GateDecision::Accepted;
            verdict.reason = original_reason + " [shadow: would " +
                             (verdict.marginal_value < config_.low_value_threshold ? "reject" : "defer") + "]";
        } else {
            switch (verdict.decision) {
                case GateDecision::Accepted: stats_.accepted++; break;
                case GateDecision::Rejected: stats_.rejected++; break;
                case GateDecision::Deferred: stats_.deferred++; break;
            }
        }
    }

    spdlog::debug("WriteGate: {} marginal_value={:.3f} tier={} reason='{}'",
                  gateDecisionToString(verdict.decision), verdict.marginal_value,
                  sourceTierToString(verdict.tier), verdict.reason);

    return verdict;
}

// ── Configuration ────────────────────────────────────────────────────────

void WriteGate::setShadowMode(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.shadow_mode = enabled;
    spdlog::info("WriteGate: shadow mode {}", enabled ? "enabled" : "disabled");
}

bool WriteGate::isShadowMode() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.shadow_mode;
}

void WriteGate::setConfig(const WriteGateConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
}

WriteGateConfig WriteGate::config() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

// ── Stats ────────────────────────────────────────────────────────────────

WriteGate::Stats WriteGate::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void WriteGate::resetStats() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = {};
}

// ── Private helpers ──────────────────────────────────────────────────────

float WriteGate::computeMarginalValue(
    const ProposedMemory& candidate,
    const std::vector<std::pair<uint64_t, float>>& neighbors) {

    if (neighbors.empty()) {
        // No existing memories → max marginal value (everything is new)
        return 1.0f;
    }

    // Marginal value = 1 - max_similarity_to_any_neighbor
    // Higher uniqueness → higher marginal value
    float max_similarity = 0.0f;
    for (const auto& [id, similarity] : neighbors) {
        max_similarity = std::max(max_similarity, similarity);
    }

    float base_value = 1.0f - max_similarity;

    // Boost marginal value based on source tier (behavioral evidence is more valuable)
    float tier_boost = 0.0f;
    switch (candidate.source_tier) {
        case SourceTier::Behavioral: tier_boost = 0.1f; break;
        case SourceTier::Assertion:  tier_boost = 0.05f; break;
        case SourceTier::Inference:  tier_boost = 0.0f; break;
    }

    // Boost by importance
    float importance_boost = candidate.importance * 0.1f;

    return std::clamp(base_value + tier_boost + importance_boost, 0.0f, 1.0f);
}

SourceTier WriteGate::classifySourceTier(const ProposedMemory& candidate) {
    // Use the source_tier provided by the caller if it's not the default
    if (candidate.source_tier != SourceTier::Inference) {
        return candidate.source_tier;
    }

    // Simple heuristic classification based on content patterns
    const auto& content = candidate.content;

    // Assertion patterns: "I like", "I prefer", "I want", "I am", "my name is"
    // Checked FIRST because assertions are explicit user statements.
    static const std::regex assertion_pattern(
        R"(\b(I\s+(like|prefer|want|need|am|have|use)|my\s+(name|preference|favorite))\b)",
        std::regex::icase);

    // Behavioral patterns: "always", "usually", "every time", "5 times", etc.
    // Requires frequency/repetition signals, not just preference verbs.
    static const std::regex behavioral_pattern(
        R"(\b(always|usually|every\s+time|repeatedly|often|habit|pattern|\d+\s*times)\b)",
        std::regex::icase);

    if (std::regex_search(content, assertion_pattern)) {
        return SourceTier::Assertion;
    }
    if (std::regex_search(content, behavioral_pattern)) {
        return SourceTier::Behavioral;
    }
    return SourceTier::Inference;
}

bool WriteGate::isTransientContent(const std::string& content) {
    if (content.size() < 3) return true;

    // ── Error / system noise (structural detection, not exhaustive) ──
    // Check prefixes that signal error messages from any upstream client.
    // UTF-8 prefixes for Chinese error patterns:
    //   出错了 = \xe5\x87\xba\xe9\x94\x99\xe4\xba\x86
    //   抱歉   = \xe6\x8a\xb1\xe6\xad\x89
    //   ⏳     = \xe2\x8f\xb3
    static const std::vector<std::string> error_prefixes = {
        "\xe5\x87\xba\xe9\x94\x99\xe4\xba\x86",  // 出错了
        "\xe6\x8a\xb1\xe6\xad\x89",                // 抱歉
        "\xe2\x8f\xb3",                             // ⏳ (timeout marker)
        "ERROR", "Error:", "[Error",
    };
    for (const auto& prefix : error_prefixes) {
        if (content.size() >= prefix.size() &&
            content.compare(0, prefix.size(), prefix) == 0) {
            return true;
        }
    }

    // Structural error signals: HTTP status codes, rate limiting, LLM failures.
    // These appear anywhere in the content regardless of language.
    // HTTP status codes must follow "HTTP", "status", or "code" to avoid false
    // positives on normal numeric content like "预算500万".
    static const std::regex error_signal_pattern(
        R"((?:HTTP|status|code|error)\s*[:= ]?\s*\b(429|500|502|503|504)\b|rate.?limit|empty.?response|timed?\s*out|connection.?refused)",
        std::regex::icase);
    if (std::regex_search(content, error_signal_pattern)) {
        return true;
    }

    // ── Greetings / acknowledgments / no-info queries ──
    if (content.size() > 200) return false;

    // Chinese transient patterns (UTF-8):
    //   你好, 你是谁, 谢谢, 好的, 嗯, 哦, 再见, 拜拜, 是的, 不是, 对, 没有,
    //   你叫什么, 你能做什么, 你会什么, 今日上海天气, etc.
    static const std::regex cn_transient_pattern(
        "^("
        "\xe4\xbd\xa0\xe5\xa5\xbd"                     // 你好
        "|\xe4\xbd\xa0\xe6\x98\xaf\xe8\xb0\x81"       // 你是谁
        "|\xe8\xb0\xa2\xe8\xb0\xa2"                     // 谢谢
        "|\xe5\xa5\xbd\xe7\x9a\x84"                     // 好的
        "|\xe5\x97\xaf"                                   // 嗯
        "|\xe5\x93\xa6"                                   // 哦
        "|\xe5\x86\x8d\xe8\xa7\x81"                     // 再见
        "|\xe6\x8b\x9c\xe6\x8b\x9c"                     // 拜拜
        "|\xe6\x98\xaf\xe7\x9a\x84"                     // 是的
        "|\xe4\xb8\x8d\xe6\x98\xaf"                     // 不是
        "|\xe5\xaf\xb9"                                   // 对
        "|\xe6\xb2\xa1\xe6\x9c\x89"                     // 没有
        "|\xe4\xbd\xa0\xe5\x8f\xab\xe4\xbb\x80\xe4\xb9\x88"           // 你叫什么
        "|\xe4\xbd\xa0\xe8\x83\xbd\xe5\x81\x9a\xe4\xbb\x80\xe4\xb9\x88"   // 你能做什么
        "|\xe4\xbd\xa0\xe4\xbc\x9a\xe4\xbb\x80\xe4\xb9\x88"           // 你会什么
        "|\xe4\xbd\xa0\xe5\x8f\xab\xe4\xbb\x80\xe4\xb9\x88\xe5\x90\x8d\xe5\xad\x97"  // 你叫什么名字
        "|\xe4\xbb\x8a\xe6\x97\xa5.*\xe5\xa4\xa9\xe6\xb0\x94"         // 今日...天气
        ")[\xe5\x95\x8a\xe5\x91\x80\xe5\x91\xa2\xe5\x90\x97\xe4\xb9\x9f\xe5\x91\xb3\\s\xef\xbc\x8c\xef\xbc\x9f?!.]*$");
    if (std::regex_match(content, cn_transient_pattern)) {
        return true;
    }

    static const std::regex transient_pattern(
        R"(^(hi|hello|hey|ok|okay|thanks|thank\s+you|sure|got\s+it|yes|no|bye|goodbye|see\s+you|alright|fine|cool|great|nice|wow|lol|haha|hmm|um|uh)[\s!.?]*$)",
        std::regex::icase);

    return std::regex_match(content, transient_pattern);
}

bool WriteGate::isForgetRequest(const std::string& content) {
    if (content.size() < 6 || content.size() > 500) return false;

    // Chinese forget/delete patterns:
    //   请忘掉 = \xe8\xaf\xb7\xe5\xbf\x98\xe6\x8e\x89
    //   请忘记 = \xe8\xaf\xb7\xe5\xbf\x98\xe8\xae\xb0
    //   请删除 = \xe8\xaf\xb7\xe5\x88\xa0\xe9\x99\xa4
    //   帮我忘 = \xe5\xb8\xae\xe6\x88\x91\xe5\xbf\x98
    //   帮我删 = \xe5\xb8\xae\xe6\x88\x91\xe5\x88\xa0
    //   忘掉   = \xe5\xbf\x98\xe6\x8e\x89
    //   忘了吧 = \xe5\xbf\x98\xe4\xba\x86\xe5\x90\xa7
    //   删掉   = \xe5\x88\xa0\xe6\x8e\x89
    //   别记了 = \xe5\x88\xab\xe8\xae\xb0\xe4\xba\x86
    //   不要记 = \xe4\xb8\x8d\xe8\xa6\x81\xe8\xae\xb0
    //   清除   = \xe6\xb8\x85\xe9\x99\xa4
    static const std::vector<std::string> cn_forget = {
        "\xe8\xaf\xb7\xe5\xbf\x98\xe6\x8e\x89",   // 请忘掉
        "\xe8\xaf\xb7\xe5\xbf\x98\xe8\xae\xb0",   // 请忘记
        "\xe8\xaf\xb7\xe5\x88\xa0\xe9\x99\xa4",   // 请删除
        "\xe5\xb8\xae\xe6\x88\x91\xe5\xbf\x98",   // 帮我忘
        "\xe5\xb8\xae\xe6\x88\x91\xe5\x88\xa0",   // 帮我删
        "\xe5\xbf\x98\xe6\x8e\x89",               // 忘掉
        "\xe5\xbf\x98\xe4\xba\x86\xe5\x90\xa7",   // 忘了吧
        "\xe5\x88\xa0\xe6\x8e\x89",               // 删掉
        "\xe5\x88\xab\xe8\xae\xb0\xe4\xba\x86",   // 别记了
        "\xe4\xb8\x8d\xe8\xa6\x81\xe8\xae\xb0",   // 不要记
        "\xe6\xb8\x85\xe9\x99\xa4",               // 清除
    };
    for (const auto& pat : cn_forget) {
        if (content.find(pat) != std::string::npos) return true;
    }

    static const std::regex en_forget(
        R"(\b(forget|delete|remove|erase)\b.*(memory|info|data|about|what\s+i))",
        std::regex::icase);
    if (std::regex_search(content, en_forget)) return true;

    return false;
}

bool WriteGate::hasSecrecyMarker(const std::string& content) {
    // Chinese secrecy markers:
    //   千万不要告诉别人 = \xe5\x8d\x83\xe4\xb8\x87\xe4\xb8\x8d\xe8\xa6\x81\xe5\x91\x8a\xe8\xaf\x89\xe5\x88\xab\xe4\xba\xba
    //   别告诉别人     = \xe5\x88\xab\xe5\x91\x8a\xe8\xaf\x89\xe5\x88\xab\xe4\xba\xba
    //   不要告诉任何人 = \xe4\xb8\x8d\xe8\xa6\x81\xe5\x91\x8a\xe8\xaf\x89\xe4\xbb\xbb\xe4\xbd\x95\xe4\xba\xba
    //   千万别说       = \xe5\x8d\x83\xe4\xb8\x87\xe5\x88\xab\xe8\xaf\xb4
    //   千万别告诉     = \xe5\x8d\x83\xe4\xb8\x87\xe5\x88\xab\xe5\x91\x8a\xe8\xaf\x89
    //   保密           = \xe4\xbf\x9d\xe5\xaf\x86
    //   不要透露       = \xe4\xb8\x8d\xe8\xa6\x81\xe9\x80\x8f\xe9\x9c\xb2
    static const std::vector<std::string> cn_secrecy = {
        "\xe5\x8d\x83\xe4\xb8\x87\xe4\xb8\x8d\xe8\xa6\x81\xe5\x91\x8a\xe8\xaf\x89",  // 千万不要告诉
        "\xe5\x88\xab\xe5\x91\x8a\xe8\xaf\x89\xe5\x88\xab\xe4\xba\xba",              // 别告诉别人
        "\xe4\xb8\x8d\xe8\xa6\x81\xe5\x91\x8a\xe8\xaf\x89\xe4\xbb\xbb\xe4\xbd\x95",  // 不要告诉任何
        "\xe5\x8d\x83\xe4\xb8\x87\xe5\x88\xab\xe8\xaf\xb4",                          // 千万别说
        "\xe5\x8d\x83\xe4\xb8\x87\xe5\x88\xab\xe5\x91\x8a\xe8\xaf\x89",              // 千万别告诉
        "\xe4\xbf\x9d\xe5\xaf\x86",                                                    // 保密
        "\xe4\xb8\x8d\xe8\xa6\x81\xe9\x80\x8f\xe9\x9c\xb2",                          // 不要透露
    };
    for (const auto& pat : cn_secrecy) {
        if (content.find(pat) != std::string::npos) return true;
    }

    static const std::regex en_secrecy(
        R"(\b(don'?t\s+tell|keep\s+(it\s+)?secret|confidential|private|do\s+not\s+share)\b)",
        std::regex::icase);
    if (std::regex_search(content, en_secrecy)) return true;

    return false;
}

}  // namespace amind
