#pragma once

#include "core/memory_record.h"
#include "core/result.h"
#include "core/snowflake.h"
#include "lsm/lsm_engine.h"
#include "vector/hnsw_index.h"

#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace amind {

class EmbedProvider;

/// Central memory store — manages the lifecycle of all memories.
/// Handles two-stage store, version history, and phase transitions.
class MemoryStore {
public:
    struct Config {
        std::string data_dir = "./amind_data";
        size_t embedding_dim = 4096;
        float conflict_similarity_threshold = 0.85f;
        uint32_t stale_threshold_hours = 720;
        float decay_rate = 0.01f;
        float importance_boost = 0.1f;

        // HNSW index parameters
        size_t hnsw_max_connections = 16;
        size_t hnsw_ef_construction = 200;
        size_t hnsw_hot_capacity = 50000;
        std::string hnsw_cold_tier_path;
        float hnsw_eviction_ratio = 0.1f;

        // HNSW persistence
        uint32_t hnsw_save_interval = 60;

        // LSM engine parameters
        uint32_t lsm_flush_interval = 30;

        // Cache limits
        size_t max_cache_size = 100000;
    };

    explicit MemoryStore(Config config);
    ~MemoryStore();

    /// Initialize storage engines (LSM, HNSW).
    Result<void, Error> init();

    /// Stage 1: Fast store — write to LSM + HNSW, return memory_id immediately.
    Result<uint64_t> fastStore(MemoryRecord record);

    /// Get a memory by ID.
    Result<MemoryRecord> get(uint64_t memory_id);

    /// Update a memory (creates new version, marks old as Versioned).
    Result<uint64_t> update(uint64_t memory_id, const std::string& new_content,
                            const std::vector<float>& new_embedding);

    /// Soft delete (mark as Tombstone).
    Result<void, Error> remove(uint64_t memory_id);

    /// Archive a memory (Active → Archived, no cascade).
    Result<void, Error> archive(uint64_t memory_id);

    /// Get version history chain for a memory.
    Result<std::vector<MemoryRecord>> getHistory(uint64_t memory_id, int max_depth = 10);

    /// Search by vector similarity (top-k nearest neighbors).
    std::vector<std::pair<uint64_t, float>> searchSimilar(
        const std::vector<float>& query_embedding, size_t top_k);

    /// Find potentially conflicting memories (similarity > threshold).
    std::vector<std::pair<uint64_t, float>> findConflicts(
        const std::vector<float>& embedding);

    /// Boost importance of a specific memory (clamped to [0, 1]).
    void boostImportance(uint64_t memory_id, float delta);

    /// Set embedding for a memory (deferred embed) and insert into HNSW.
    void setEmbedding(uint64_t memory_id, std::vector<float> embedding);

    /// Apply time-based decay to importance scores.
    void applyDecay();

    /// Get total memory count.
    size_t totalMemories() const;


    /// List memories with pagination and optional filters.
    struct ListFilter {
        int page{1};
        int per_page{50};
        std::string scope_filter;          // empty = all
        std::string memory_type_filter;    // empty = all
        std::string phase_filter;          // empty = all
        std::string query;                 // keyword substring search
        std::string user_id_filter;        // empty = all; matches user_metadata["user_id"]
        std::string layer_filter;          // "Raw" | "Derived" | empty=all
        bool include_tombstone{false};     // when true, return Tombstoned records too
    };
    struct ListResult {
        std::vector<MemoryRecord> records;
        size_t total{0};
        int page{1};
        int per_page{50};
    };
    ListResult listMemories(const ListFilter& filter) const;


    /// Iterate over all records in cache.
    void scanAll(std::function<void(const MemoryRecord&)> callback) const;

    /// Scan records whose content starts with the given prefix.
    std::vector<MemoryRecord> scanByContentPrefix(const std::string& prefix) const;

    /// In-place content update (no versioning). Used for internal metadata records.
    void updateInPlace(uint64_t memory_id, const std::string& new_content);

    /// Flush to disk.
    void flush();

    // ── Dynamic variable setters (thread-safe) ──
    void setConflictThreshold(float v) { std::unique_lock lk(mutex_); config_.conflict_similarity_threshold = v; }
    void setDecayRate(float v) { std::unique_lock lk(mutex_); config_.decay_rate = v; }
    void setImportanceBoost(float v) { std::unique_lock lk(mutex_); config_.importance_boost = v; }
    void setStaleThresholdHours(uint32_t v) { std::unique_lock lk(mutex_); config_.stale_threshold_hours = v; }


private:
    Config config_;
    std::unique_ptr<LSMEngine> lsm_;
    std::unique_ptr<HNSWIndex> hnsw_;
    SnowflakeGenerator id_gen_;
    mutable std::shared_mutex mutex_;

    // In-memory index: memory_id → MemoryRecord (hot cache)
    std::unordered_map<uint64_t, MemoryRecord> cache_;

    void evictIfNeeded();
};

}  // namespace amind
