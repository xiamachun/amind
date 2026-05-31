#pragma once

#include "cold_tier.h"
#include "distance.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <shared_mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace amind {

/// Result of a nearest-neighbor search.
struct SearchResult {
    uint64_t id;        // memory_id
    float distance;     // cosine distance to query

    bool operator<(const SearchResult& other) const {
        return distance < other.distance;
    }

    bool operator>(const SearchResult& other) const {
        return distance > other.distance;
    }
};

/// Configuration for the HNSW index.
struct HNSWConfig {
    size_t maxConnections = 16;
    size_t maxConnectionsLayer0 = 32;
    size_t efConstruction = 200;
    size_t efSearch = 50;
    size_t dimension = 0;
    DistanceMetric metric = DistanceMetric::Cosine;

    // Hot/cold tiering
    size_t hotCapacity = 50000;         // max vectors in hot HNSW layer
    std::string coldTierPath;           // path for cold tier storage
    float evictionRatio = 0.1f;         // evict 10% when capacity exceeded
};

/// HNSW index with hot/cold tiering support.
/// Hot layer: full HNSW graph (fast ANN search).
/// Cold layer: disk-backed brute-force (slower, but unbounded capacity).
class HNSWIndex {
public:
    explicit HNSWIndex(HNSWConfig config);

    void insert(uint64_t id, std::span<const float> vector);

    using FilterFunc = std::function<bool(uint64_t)>;

    std::vector<SearchResult> search(std::span<const float> query,
                                     size_t topK) const;

    std::vector<SearchResult> searchWithFilter(std::span<const float> query,
                                               size_t topK,
                                               const FilterFunc& filter) const;

    bool remove(uint64_t id);
    bool contains(uint64_t id) const;
    size_t size() const;
    bool empty() const;

    // ── V2: Enhanced deletion & compaction ────────────────────────────────
    /// Soft-delete a vector (same as remove, but tracks deleted count for compact).
    bool softDelete(uint64_t id);

    /// Physically remove all soft-deleted nodes and rebuild graph connections.
    /// Returns the number of nodes purged.
    size_t compact();

    /// Move specified IDs from hot layer to cold tier.
    /// Returns the number of vectors actually relegated.
    size_t relegateToCold(const std::vector<uint64_t>& ids);

    /// Count of soft-deleted nodes still occupying memory (not yet compacted).
    size_t deletedCount() const;

    size_t dimension() const { return config_.dimension; }
    void setEfSearch(size_t efSearch) { config_.efSearch = efSearch; }
    size_t efSearch() const { return config_.efSearch; }
    int maxLevel() const;

    bool save(const std::string& path) const;
    bool load(const std::string& path);

    /// Initialize cold tier (call after config is set).
    void initColdTier();

    /// Force eviction of cold vectors from hot layer.
    size_t evictCold();

    /// Get hot/cold stats.
    size_t hotSize() const;
    size_t coldSize() const;

private:
    struct Node {
        uint64_t id;
        std::vector<float> vector;
        int level;
        bool deleted = false;
        uint32_t last_accessed{0};  // for hot/cold eviction
        uint32_t access_count{0};   // access frequency
        std::vector<std::vector<uint64_t>> connections;
    };

    float computeDistance(std::span<const float> vectorA,
                          std::span<const float> vectorB) const;
    int randomLevel();

    uint64_t greedyClosest(std::span<const float> query,
                           uint64_t entryId,
                           int layer) const;

    std::vector<SearchResult> searchLayer(std::span<const float> query,
                                          const std::vector<uint64_t>& entryPoints,
                                          size_t ef,
                                          int layer,
                                          const FilterFunc& filter = nullptr) const;

    std::vector<uint64_t> selectNeighbors(std::span<const float> nodeVector,
                                           const std::vector<SearchResult>& candidates,
                                           size_t maxConnections) const;

    size_t maxConnectionsForLayer(int layer) const;

    const Node* getNode(uint64_t id) const;
    Node* getNode(uint64_t id);

    std::vector<SearchResult> bruteForceSearch(std::span<const float> query,
                                               size_t topK,
                                               const FilterFunc& filter) const;

    /// Merge hot and cold search results.
    std::vector<SearchResult> mergeResults(std::vector<SearchResult>& hot,
                                            std::vector<SearchResult>& cold,
                                            size_t topK) const;

    static constexpr float kBruteForceMatchRateThreshold = 0.07f;

    HNSWConfig config_;
    std::unordered_map<uint64_t, Node> nodes_;
    uint64_t entryPointId_ = 0;
    int maxLevel_ = -1;
    size_t activeCount_ = 0;

    mutable std::shared_mutex mutex_;
    std::mt19937 rng_{42};
    double levelMultiplier_;

    // Cold tier
    std::unique_ptr<ColdTierIndex> cold_tier_;
};

}  // namespace amind
