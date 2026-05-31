#pragma once

#include "distance.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace amind {

struct SearchResult;  // forward declaration

/// Cold tier index: disk-backed vector storage with brute-force search.
/// When the hot HNSW index exceeds capacity, cold (least-recently-accessed)
/// vectors are evicted here. Stored as a flat file on disk, loaded via mmap
/// or buffered read for search.
class ColdTierIndex {
public:
    static constexpr uint32_t COLD_MAGIC = 0x434F4C44;  // "COLD"
    static constexpr uint32_t COLD_VERSION = 1;

    ColdTierIndex(const std::string& path, size_t dimension, DistanceMetric metric);
    ~ColdTierIndex();

    /// Load existing cold tier data from disk.
    bool load();

    /// Save current state to disk.
    bool save() const;

    /// Append a vector to cold storage.
    void append(uint64_t id, const std::vector<float>& vector);

    /// Remove a vector by ID.
    bool remove(uint64_t id);

    /// Brute-force search the cold tier.
    std::vector<SearchResult> search(const std::vector<float>& query, size_t topK) const;

    /// Check if an ID exists in cold tier.
    bool contains(uint64_t id) const;

    /// Number of vectors in cold tier.
    size_t size() const;

    /// Get a vector by ID (returns copy for thread safety).
    std::optional<std::vector<float>> getVector(uint64_t id) const;

private:
    float computeDistance(const std::vector<float>& a, const std::vector<float>& b) const;

    std::string path_;
    size_t dimension_;
    DistanceMetric metric_;

    // In-memory storage (loaded from disk)
    struct ColdEntry {
        uint64_t id;
        std::vector<float> vector;
    };
    std::vector<ColdEntry> entries_;
    std::unordered_map<uint64_t, size_t> id_to_index_;  // id → index in entries_

    mutable std::mutex mutex_;
};

}  // namespace amind
