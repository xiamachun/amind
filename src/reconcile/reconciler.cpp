#include "reconciler.h"

#include <algorithm>
#include <chrono>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <sstream>
#include <unordered_set>

namespace amind {

const char* reconcileOpToString(ReconcileOp op) {
    switch (op) {
        case ReconcileOp::ADD:       return "ADD";
        case ReconcileOp::REPLACE:   return "REPLACE";
        case ReconcileOp::RETRACT:   return "RETRACT";
        case ReconcileOp::REINFORCE: return "REINFORCE";
        case ReconcileOp::NOOP:      return "NOOP";
    }
    return "ADD";
}

ReconcileOp parseReconcileOp(const std::string& s) {
    if (s == "REPLACE")   return ReconcileOp::REPLACE;
    if (s == "RETRACT")   return ReconcileOp::RETRACT;
    if (s == "REINFORCE") return ReconcileOp::REINFORCE;
    if (s == "NOOP")      return ReconcileOp::NOOP;
    return ReconcileOp::ADD;
}

namespace {

// The system instruction is bilingual on purpose: amind clients write in many
// languages (we've already seen Chinese/English mixed). The instruction
// itself stays in English so it doesn't compete with content tokens.
const char* SYSTEM_PROMPT = R"(You are a fact-reconciliation engine for a memory store.

The store contains user-specific facts. When a NEW fact arrives, you decide
whether it conflicts with EXISTING facts and pick ONE operation:

  ADD        — new fact is independent; no existing fact addresses the same property
  REPLACE    — same property of the same entity, but a NEW value
               → retire the old fact, keep the new one
  RETRACT    — new fact negates ("不" / "no longer" / "stop") an existing fact
               → retire the old fact, do NOT store the new statement
  REINFORCE  — new fact says the SAME thing in different words
               → do NOT store new (duplicate), bump importance of the existing
  NOOP       — new fact is fully covered already; skip silently

CRITICAL RULES — apply these first. Examples are given bilingually so the
rules generalize across languages — apply the SAME logic regardless of
input language (English, Chinese, Spanish, Japanese, mixed-script, …).

  1. SAME-SLOT VALUE CHANGE → REPLACE.
     If existing says "X's Y is A" and new says "X's Y is B" with the SAME
     entity X and same property Y, pick REPLACE.
       existing "user's dog is named Wangcai"  new "user's dog is named Xiaobai"
       existing "用户家的狗叫旺财"               new "用户家的狗叫小白"
       existing "the cat is 3 years old"        new "the cat is 5 years old"
       existing "猫3岁"                         new "猫5岁"
       existing "user's phone is 13800138000"   new "user's phone is 15900159000"
       → all REPLACE

  2. PREFERENCE properties — "favorite / most-liked / best X" is ONE slot per
     person. A new favorite value REPLACES the old one even if the values
     look unrelated.
       existing "user's favorite restaurant is Haidilao"
                                          new "user loves Dadong roast duck, goes weekly"
       existing "用户最喜欢的餐厅是海底捞"     new "用户喜欢大董烤鸭,几乎每周都去"
       existing "user's favorite movie is Star Wars"
                                          new "user's favorite movie is Ready Player One"
       existing "user's favorite drink is Coke"  new "user prefers fruit juice now"
       → all REPLACE
     The values looking "different" is the WHOLE POINT of a preference update.

  3. BINDING/ASSOCIATION about a SUPERSEDED anchor → RETRACT.
       existing "user's bank card uses 138-phone"
                                            new "user's 138-phone has been cancelled"
       existing "用户的银行卡绑定138号码"        new "用户的138号码已注销"
       → RETRACT (drop the bank-card binding, since 138 is gone)

  4. NEGATION over an existing claim → RETRACT.
       existing "user uses IntelliJ"     new "user no longer uses IntelliJ"
       existing "用户用 IntelliJ"         new "用户不再用 IntelliJ"
       existing "user is allergic to peanuts"  new "user is no longer allergic to peanuts"
       existing "用户对花生过敏"            new "用户对花生不过敏"
       → RETRACT
     Negation cues across languages: "no", "not", "never", "no longer",
     "stopped", "不", "没", "不再", "已注销", "ya no", "ne ... pas".

  5. PARAPHRASE / SAME FACT → REINFORCE (do NOT add a duplicate copy).
     If new and existing assert the SAME claim about the SAME entity/property
     even with very different wording, it is a duplicate.
       existing "user likes coffee"             new "user is a coffee lover"
       existing "用户喜欢喝咖啡"                 new "用户是咖啡爱好者"
       existing "user lives in Beijing"         new "user's current residence is Beijing"
       existing "用户住在北京"                   new "用户现在的居住地是北京"
       existing "user goes to the gym every Wednesday"
                                          new "every Wednesday user works out at the gym"
       existing "用户每周三去健身房"             new "用户每周三都会去健身房锻炼"
       existing "user's daughter is named Xiaohong"
                                          new "user has a daughter named Xiaohong"
       existing "user is good at Python"        new "Python is user's strongest language"
       existing "user's favorite is Dilraba"    new "Dilraba is user's idol"
       existing "user is allergic to peanuts"   new "peanuts cause user to have allergies"
       existing "user's annual salary is 800K"  new "user's yearly income is 800K"
       → all REINFORCE
     Heuristic: ask "do these two statements entail each other?" If yes →
     REINFORCE. Different surface forms, equivalent meaning, same slot.

  6. ADD is only correct when the new fact introduces information about a
     DIFFERENT property or entity that no existing fact covers. Default to
     REINFORCE / REPLACE / RETRACT before ADD whenever the new fact and an
     existing fact share the same subject and the same property slot.

  7. TIMESTAMP ORDERING — each fact has a created_at unix timestamp.
     When the EXISTING fact has a LATER timestamp than the NEW fact,
     the "existing" fact is actually more recent information. In this case:
     - If they conflict, return NOOP (discard the stale newcomer).
     - If the new fact is a paraphrase of the existing, return NOOP.
     The newer timestamp wins — never REPLACE a newer fact with an older one.

  8. TEMPORAL QUALIFIERS — if the NEW fact contains words like "之前的",
     "旧的", "previous", "old", "former", "past", it describes a SEPARATE
     historical fact, NOT a replacement of the current value.
       existing "user's passport is E83291047"
                              new "user's previous passport was E73291047"
       existing "用户的护照是 E83291047"       new "用户之前的旧护照是 E73291047"
       → ADD (the old passport is a different historical fact, not a new value)
     Do NOT pick REPLACE just because both mention "passport" — the temporal
     qualifier signals they are DIFFERENT facts about DIFFERENT time periods.

Output STRICT JSON, nothing else, no markdown fences:
  {"op":"<OP>","target_ids":[<id1>,<id2>,...],"rationale":"<one short Chinese or English sentence>"}

target_ids is an array of integer ids of existing facts you act on.
For ADD, target_ids MUST be an empty array [].
For REPLACE, target_ids MUST contain ALL existing facts that reference the
old/outdated value being superseded — not just the most similar one.
Example: if existing facts include "user drives a white Tesla Model 3",
"user charges Tesla twice a week", "user loves Tesla autopilot", and
"user drove Tesla to Dongguan", and the new fact is "user sold Tesla
and bought a BYD Han EV", then target_ids MUST include ALL four Tesla
facts because they all reference the old value (Tesla) being replaced.
For RETRACT, target_ids can also contain multiple ids if the new fact
invalidates several existing facts.
For REINFORCE and NOOP, target_ids typically contains one id.
You may also use "target_id" (singular integer) instead of a single-element
array for backward compatibility, but prefer "target_ids" (array).)";

std::string truncate(const std::string& s, size_t max_chars) {
    if (s.size() <= max_chars) return s;
    // Make sure we don't cut a UTF-8 byte sequence in half: walk back to a
    // byte that doesn't have the 10xxxxxx continuation bit pattern.
    size_t cut = max_chars;
    while (cut > 0 && (static_cast<unsigned char>(s[cut]) & 0xC0) == 0x80) {
        cut--;
    }
    return s.substr(0, cut) + "…";
}

}  // namespace

Reconciler::Reconciler(std::shared_ptr<LLMProvider> llm, Config config)
    : llm_(std::move(llm)), config_(config) {}

std::string Reconciler::buildPrompt(
    const std::string& candidate,
    const std::vector<std::pair<MemoryRecord, float>>& neighbours,
    size_t max_content_chars) {

    std::ostringstream oss;
    oss << "Existing facts about this user:\n";
    if (neighbours.empty()) {
        oss << "  (none)\n";
    } else {
        for (const auto& [rec, sim] : neighbours) {
            oss << "  - id=" << rec.memory_id
                << " created_at=" << rec.created_at
                << " sim=" << std::to_string(sim).substr(0, 4)
                << " content=\"" << truncate(rec.content, max_content_chars) << "\"\n";
        }
    }
    oss << "\nNew fact (just extracted from latest conversation):\n";
    oss << "  \"" << truncate(candidate, max_content_chars) << "\"\n";
    oss << "\nDecide the op as instructed and respond with the JSON object only.";
    return oss.str();
}

ReconcileDecision Reconciler::parseResponse(const std::string& json_str) {
    ReconcileDecision out;
    try {
        // Tolerant parsing: strip ```json fences and stray prose around JSON.
        auto first_brace = json_str.find('{');
        auto last_brace = json_str.rfind('}');
        if (first_brace == std::string::npos || last_brace == std::string::npos
            || last_brace < first_brace) {
            spdlog::warn("Reconciler: no JSON object found in LLM response, defaulting to ADD");
            return out;
        }
        std::string trimmed = json_str.substr(first_brace, last_brace - first_brace + 1);
        auto j = nlohmann::json::parse(trimmed);
        out.op = parseReconcileOp(j.value("op", "ADD"));

        // Parse target_ids (array) or target_id (single int) — prefer array.
        if (j.contains("target_ids") && j["target_ids"].is_array()) {
            for (const auto& tid : j["target_ids"]) {
                uint64_t id = 0;
                if (tid.is_number_integer())       id = tid.get<uint64_t>();
                else if (tid.is_string()) {
                    try { id = std::stoull(tid.get<std::string>()); } catch (...) {}
                }
                if (id != 0) out.target_ids.push_back(id);
            }
            out.target_id = out.target_ids.empty() ? 0 : out.target_ids[0];
        } else if (j.contains("target_id")) {
            const auto& t = j["target_id"];
            if (t.is_number_integer())       out.target_id = t.get<uint64_t>();
            else if (t.is_string()) {
                try { out.target_id = std::stoull(t.get<std::string>()); } catch (...) {}
            }
            if (out.target_id != 0) out.target_ids.push_back(out.target_id);
        }

        out.rationale = j.value("rationale", "");
        // Sanity: ADD must have empty target_ids; non-ADD must have non-empty.
        if (out.op == ReconcileOp::ADD && !out.target_ids.empty()) {
            spdlog::debug("Reconciler: ADD with non-empty target_ids, normalising to empty");
            out.target_ids.clear();
            out.target_id = 0;
        } else if (out.op != ReconcileOp::ADD && out.target_ids.empty()) {
            spdlog::warn("Reconciler: {} with empty target_ids — degrading to ADD",
                         reconcileOpToString(out.op));
            out.op = ReconcileOp::ADD;
            out.target_id = 0;
        }
    } catch (const std::exception& e) {
        spdlog::warn("Reconciler: response parse failed ({}); defaulting to ADD: {}",
                     e.what(), json_str.substr(0, 200));
    }
    return out;
}

ReconcileDecision Reconciler::decide(
    const std::string& candidate_content,
    const std::vector<std::pair<MemoryRecord, float>>& neighbours) {

    auto start_time = std::chrono::steady_clock::now();

    {
        std::lock_guard lock(stats_mu_);
        stats_.total_calls++;
    }

    // Filter to neighbours above the similarity floor; preserve order.
    std::vector<std::pair<MemoryRecord, float>> strong;
    strong.reserve(std::min(neighbours.size(), config_.max_neighbours));
    for (const auto& nb : neighbours) {
        if (nb.second < config_.similarity_floor) continue;
        strong.push_back(nb);
        if (strong.size() >= config_.max_neighbours) break;
    }

    if (strong.empty() || !llm_) {
        // No strong neighbours → fact stands on its own.
        std::lock_guard lock(stats_mu_);
        stats_.op_add++;
        return ReconcileDecision{
            .op = ReconcileOp::ADD, .target_id = 0,
            .rationale = "no strong neighbours", .from_fallback = false,
        };
    }

    std::string user_prompt = buildPrompt(candidate_content, strong, config_.max_content_chars);
    std::string sys_prompt = SYSTEM_PROMPT;

    {
        std::lock_guard lock(stats_mu_);
        stats_.llm_invocations++;
    }

    auto resp = llm_->generateJson(user_prompt, sys_prompt);
    if (!resp.ok()) {
        spdlog::warn("Reconciler: LLM call failed ({}); defaulting to ADD",
                     resp.error().toString());
        std::lock_guard lock(stats_mu_);
        stats_.llm_failures++;
        stats_.op_add++;
        return ReconcileDecision{
            .op = ReconcileOp::ADD, .target_id = 0,
            .rationale = "LLM failure: " + resp.error().toString(),
            .from_fallback = true,
        };
    }

    ReconcileDecision decision = parseResponse(*resp);

    // Diagnostic: log what we asked + what we got. Helps tune the prompt and
    // similarity floor when REPLACE/RETRACT aren't firing despite obvious
    // overlap.  Trimmed to keep log lines bounded.
    spdlog::info("Reconciler decision: op={} target_id={} rationale={}",
                 reconcileOpToString(decision.op), decision.target_id,
                 decision.rationale.substr(0, 120));
    spdlog::info("Reconciler candidate: {}", candidate_content.substr(0, 120));
    for (const auto& [rec, sim] : strong) {
        spdlog::info("  neighbour id={} sim={:.3f} content={}",
                     rec.memory_id, sim, rec.content.substr(0, 100));
    }
    spdlog::info("Reconciler raw LLM response: {}", resp->substr(0, 300));

    // Sanity: each target_id must be one of the listed neighbours,
    // otherwise the LLM hallucinated. Filter invalid ones out; if none
    // remain, degrade to ADD.
    if (!decision.target_ids.empty()) {
        std::unordered_set<uint64_t> valid_ids;
        for (const auto& nb : strong) valid_ids.insert(nb.first.memory_id);

        std::vector<uint64_t> filtered;
        for (auto tid : decision.target_ids) {
            if (valid_ids.count(tid)) filtered.push_back(tid);
            else spdlog::warn("Reconciler: LLM returned target_id={} not in neighbours; dropping",
                              tid);
        }
        decision.target_ids = std::move(filtered);
        decision.target_id = decision.target_ids.empty() ? 0 : decision.target_ids[0];

        if (decision.target_ids.empty() && decision.op != ReconcileOp::ADD) {
            spdlog::warn("Reconciler: all target_ids invalid — degrading to ADD");
            decision.op = ReconcileOp::ADD;
            decision.target_id = 0;
            decision.from_fallback = true;
            decision.rationale = "LLM target_ids not in neighbours; " + decision.rationale;
        }
    }

    // Tally
    {
        std::lock_guard lock(stats_mu_);
        switch (decision.op) {
            case ReconcileOp::ADD:       stats_.op_add++;       break;
            case ReconcileOp::REPLACE:   stats_.op_replace++;   break;
            case ReconcileOp::RETRACT:   stats_.op_retract++;   break;
            case ReconcileOp::REINFORCE: stats_.op_reinforce++; break;
            case ReconcileOp::NOOP:      stats_.op_noop++;      break;
        }
    }

    // Log entry with latency
    auto elapsed = std::chrono::steady_clock::now() - start_time;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    appendLog(LogEntry{
        .timestamp_ms = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()),
        .candidate = candidate_content.substr(0, 200),
        .op = decision.op,
        .target_id = decision.target_id,
        .latency_ms = static_cast<uint64_t>(ms),
        .from_fallback = decision.from_fallback,
    });

    return decision;
}

Reconciler::Stats Reconciler::stats() const {
    std::lock_guard lock(stats_mu_);
    return stats_;
}

void Reconciler::appendLog(LogEntry entry) {
    std::lock_guard lock(log_mu_);
    log_.push_back(std::move(entry));
    if (log_.size() > LOG_CAPACITY) {
        log_.pop_front();
    }
}

std::vector<Reconciler::LogEntry> Reconciler::recentLog(size_t limit) const {
    std::lock_guard lock(log_mu_);
    std::vector<LogEntry> result;
    size_t count = std::min(limit, log_.size());
    result.reserve(count);
    // Newest first
    for (auto it = log_.rbegin(); it != log_.rend() && result.size() < count; ++it) {
        result.push_back(*it);
    }
    return result;
}

}  // namespace amind
