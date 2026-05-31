#pragma once

#include "core/result.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace amind {

struct ApiKeyInfo {
    std::string id;
    std::string label;
    std::string key_prefix;
    std::string key_hash;
    uint64_t created_at{0};
    uint64_t last_used_at{0};
    bool revoked{false};
};

class MemoryStore;

/// Manages API keys stored in LSM via MemoryStore.
/// Keys are stored as special records with prefix "apikey:" in content.
class AuthManager {
public:
    explicit AuthManager(MemoryStore& store);

    /// Create a new API key. Returns the full plaintext key (only shown once).
    Result<std::string> createKey(const std::string& label);

    /// List all keys (without hashes, only prefix + metadata).
    std::vector<ApiKeyInfo> listKeys() const;

    /// Revoke a key by ID.
    Result<void, Error> revokeKey(const std::string& key_id);

    /// Validate a bearer token against stored keys. Returns true if valid.
    bool validateKey(const std::string& token);

private:
    static std::string generateRawKey();
    static std::string sha256Hex(const std::string& input);

    MemoryStore& store_;
    mutable std::mutex mu_;

    // In-memory cache for fast validation (refreshed on create/revoke)
    struct CachedKey {
        std::string id;
        std::string hash;
        uint64_t last_used{0};
    };
    std::vector<CachedKey> cache_;
    void refreshCache();
};

}  // namespace amind
