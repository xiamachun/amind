#include "auth_manager.h"
#include "memory/memory_store.h"

#include <chrono>
#include <cstring>
#include <nlohmann/json.hpp>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <random>
#include <spdlog/spdlog.h>

namespace amind {

using json = nlohmann::json;

static const std::string KEY_PREFIX = "amk_";
static const std::string CONTENT_PREFIX = "__apikey__:";

AuthManager::AuthManager(MemoryStore& store) : store_(store) {
    refreshCache();
}

std::string AuthManager::generateRawKey() {
    unsigned char buf[32];
    RAND_bytes(buf, sizeof(buf));

    // URL-safe base64 encoding (no padding)
    static const char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    std::string result;
    result.reserve(43);
    for (int i = 0; i < 32; i += 3) {
        uint32_t val = static_cast<uint32_t>(buf[i]) << 16;
        if (i + 1 < 32) val |= static_cast<uint32_t>(buf[i + 1]) << 8;
        if (i + 2 < 32) val |= static_cast<uint32_t>(buf[i + 2]);

        result += table[(val >> 18) & 0x3F];
        result += table[(val >> 12) & 0x3F];
        if (i + 1 < 32) result += table[(val >> 6) & 0x3F];
        if (i + 2 < 32) result += table[val & 0x3F];
    }
    return result;
}

std::string AuthManager::sha256Hex(const std::string& input) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(input.data()),
           input.size(), hash);

    char hex[SHA256_DIGEST_LENGTH * 2 + 1];
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        snprintf(hex + i * 2, 3, "%02x", hash[i]);
    }
    return std::string(hex, SHA256_DIGEST_LENGTH * 2);
}

Result<std::string> AuthManager::createKey(const std::string& label) {
    std::string raw = generateRawKey();
    std::string full_key = KEY_PREFIX + raw;
    std::string hash = sha256Hex(full_key);
    std::string prefix = full_key.substr(0, 12);

    auto now = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    // Store as a MemoryRecord with special content prefix
    json meta;
    meta["label"] = label;
    meta["key_prefix"] = prefix;
    meta["key_hash"] = hash;
    meta["created_at"] = now;
    meta["last_used_at"] = 0;
    meta["revoked"] = false;

    MemoryRecord record;
    record.content = CONTENT_PREFIX + meta.dump();
    record.scope = MemoryScope::AgentShared;
    record.memory_type = MemoryType::DomainKnowledge;
    record.confidence_level = Confidence::Verified;
    record.importance = 0.0f;

    auto result = store_.fastStore(std::move(record));
    if (!result.ok()) return result.error();

    spdlog::info("API key created: prefix={} id={}", prefix, *result);

    std::lock_guard lock(mu_);
    cache_.push_back(CachedKey{std::to_string(*result), hash, 0});

    return full_key;
}

std::vector<ApiKeyInfo> AuthManager::listKeys() const {
    std::vector<ApiKeyInfo> keys;
    auto all = store_.scanByContentPrefix(CONTENT_PREFIX);
    for (const auto& rec : all) {
        auto json_str = rec.content.substr(CONTENT_PREFIX.size());
        try {
            auto j = json::parse(json_str);
            if (j.value("revoked", false)) continue;
            ApiKeyInfo info;
            info.id = std::to_string(rec.memory_id);
            info.label = j.value("label", "");
            info.key_prefix = j.value("key_prefix", "");
            info.created_at = j.value("created_at", uint64_t(0));
            info.last_used_at = j.value("last_used_at", uint64_t(0));
            info.revoked = j.value("revoked", false);
            keys.push_back(std::move(info));
        } catch (...) {
            continue;
        }
    }
    return keys;
}

Result<void, Error> AuthManager::revokeKey(const std::string& key_id) {
    uint64_t id = 0;
    try { id = std::stoull(key_id); } catch (...) {
        return Error{Error::InvalidArgument, "invalid key ID"};
    }

    auto rec_result = store_.peek(id);
    if (!rec_result.ok()) return rec_result.error();

    auto& rec = *rec_result;
    if (rec.content.find(CONTENT_PREFIX) != 0) {
        return Error{Error::InvalidArgument, "not an API key record"};
    }

    auto json_str = rec.content.substr(CONTENT_PREFIX.size());
    try {
        auto j = json::parse(json_str);
        j["revoked"] = true;
        rec.content = CONTENT_PREFIX + j.dump();
        store_.updateInPlace(id, rec.content);
    } catch (...) {
        return Error{Error::InternalError, "failed to update key"};
    }

    spdlog::info("API key revoked: id={}", key_id);

    std::lock_guard lock(mu_);
    cache_.erase(
        std::remove_if(cache_.begin(), cache_.end(),
                       [&](const CachedKey& k) { return k.id == key_id; }),
        cache_.end());

    return {};
}

bool AuthManager::validateKey(const std::string& token) {
    std::string hash = sha256Hex(token);

    std::lock_guard lock(mu_);
    for (auto& cached : cache_) {
        if (cached.hash == hash) {
            cached.last_used = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count());
            return true;
        }
    }
    return false;
}

void AuthManager::refreshCache() {
    auto all = store_.scanByContentPrefix(CONTENT_PREFIX);
    std::lock_guard lock(mu_);
    cache_.clear();
    for (const auto& rec : all) {
        auto json_str = rec.content.substr(CONTENT_PREFIX.size());
        try {
            auto j = json::parse(json_str);
            if (j.value("revoked", false)) continue;
            cache_.push_back(CachedKey{
                std::to_string(rec.memory_id),
                j.value("key_hash", std::string{}),
                j.value("last_used_at", uint64_t(0)),
            });
        } catch (...) {}
    }
    spdlog::info("AuthManager: loaded {} active API keys", cache_.size());
}

}  // namespace amind
