#include "cold_tier.h"
#include "hnsw_index.h"
#include "core/crc32.h"
#include "core/file_utils.h"

#include <algorithm>
#include <fstream>
#include <queue>
#include <spdlog/spdlog.h>

namespace amind {

ColdTierIndex::ColdTierIndex(const std::string& path, size_t dimension, DistanceMetric metric)
    : path_(path), dimension_(dimension), metric_(metric) {}

ColdTierIndex::~ColdTierIndex() {
    save();
}

float ColdTierIndex::computeDistance(const std::vector<float>& a,
                                      const std::vector<float>& b) const {
    switch (metric_) {
        case DistanceMetric::Cosine:
            return cosineDistance(a, b);
        case DistanceMetric::L2:
            return l2DistanceSquared(a, b);
        case DistanceMetric::InnerProduct:
            return -innerProduct(a, b);
    }
    return cosineDistance(a, b);
}

bool ColdTierIndex::load() {
    std::lock_guard lock(mutex_);

    std::ifstream file(path_, std::ios::binary);
    if (!file.is_open()) return false;

    uint32_t magic, version;
    uint32_t dim, count;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    file.read(reinterpret_cast<char*>(&count), sizeof(count));

    if (!file.good() || magic != COLD_MAGIC || version != COLD_VERSION) {
        spdlog::warn("ColdTier: invalid file format");
        return false;
    }
    if (dim != dimension_) {
        spdlog::warn("ColdTier: dimension mismatch (file: {}, config: {})", dim, dimension_);
        return false;
    }

    entries_.clear();
    id_to_index_.clear();
    entries_.reserve(count);

    for (uint32_t i = 0; i < count; i++) {
        uint64_t id;
        file.read(reinterpret_cast<char*>(&id), sizeof(id));

        std::vector<float> vec(dimension_);
        file.read(reinterpret_cast<char*>(vec.data()), dimension_ * sizeof(float));

        if (!file.good()) {
            spdlog::warn("ColdTier: truncated file at entry {}", i);
            break;
        }

        id_to_index_[id] = entries_.size();
        entries_.push_back({id, std::move(vec)});
    }

    file.close();
    spdlog::info("ColdTier: loaded {} vectors from {}", entries_.size(), path_);
    return true;
}

bool ColdTierIndex::save() const {
    std::lock_guard lock(mutex_);
    if (entries_.empty()) return true;

    std::string tmp_path = path_ + ".tmp";
    std::ofstream file(tmp_path, std::ios::binary);
    if (!file.is_open()) {
        spdlog::error("ColdTier: failed to open {} for writing", tmp_path);
        return false;
    }

    uint32_t magic = COLD_MAGIC;
    uint32_t version = COLD_VERSION;
    uint32_t dim = static_cast<uint32_t>(dimension_);
    uint32_t count = static_cast<uint32_t>(entries_.size());

    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));

    // CRC over all entry data for integrity
    amind::CRC32Builder crcBuilder;

    for (const auto& entry : entries_) {
        file.write(reinterpret_cast<const char*>(&entry.id), sizeof(entry.id));
        file.write(reinterpret_cast<const char*>(entry.vector.data()),
                   dimension_ * sizeof(float));
        crcBuilder.update(&entry.id, sizeof(entry.id));
        crcBuilder.update(entry.vector.data(), dimension_ * sizeof(float));
    }

    uint32_t final_crc = crcBuilder.finalize();
    file.write(reinterpret_cast<const char*>(&final_crc), sizeof(final_crc));
    file.close();

    // fsync before rename ensures data is durable
    amind::fsyncFile(tmp_path);

    // Atomic rename
    std::rename(tmp_path.c_str(), path_.c_str());
    spdlog::info("ColdTier: saved {} vectors to {} (atomic+CRC)", entries_.size(), path_);
    return true;
}

void ColdTierIndex::append(uint64_t id, const std::vector<float>& vector) {
    std::lock_guard lock(mutex_);

    if (id_to_index_.count(id)) {
        // Update existing
        entries_[id_to_index_[id]].vector = vector;
        return;
    }

    id_to_index_[id] = entries_.size();
    entries_.push_back({id, vector});
}

bool ColdTierIndex::remove(uint64_t id) {
    std::lock_guard lock(mutex_);

    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end()) return false;

    size_t idx = it->second;
    // Swap with last entry for O(1) removal
    if (idx != entries_.size() - 1) {
        auto& last = entries_.back();
        id_to_index_[last.id] = idx;
        entries_[idx] = std::move(last);
    }
    entries_.pop_back();
    id_to_index_.erase(it);
    return true;
}

std::vector<SearchResult> ColdTierIndex::search(const std::vector<float>& query,
                                                  size_t topK) const {
    std::lock_guard lock(mutex_);

    if (entries_.empty()) return {};

    // Max-heap for top-K (farthest on top for pruning)
    auto cmp = [](const SearchResult& a, const SearchResult& b) {
        return a.distance < b.distance;
    };
    std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(cmp)> heap(cmp);

    for (const auto& entry : entries_) {
        float dist = computeDistance(query, entry.vector);

        if (heap.size() < topK) {
            heap.push({entry.id, dist});
        } else if (dist < heap.top().distance) {
            heap.pop();
            heap.push({entry.id, dist});
        }
    }

    std::vector<SearchResult> results;
    results.reserve(heap.size());
    while (!heap.empty()) {
        results.push_back(heap.top());
        heap.pop();
    }
    std::sort(results.begin(), results.end());
    return results;
}

bool ColdTierIndex::contains(uint64_t id) const {
    std::lock_guard lock(mutex_);
    return id_to_index_.count(id) > 0;
}

size_t ColdTierIndex::size() const {
    std::lock_guard lock(mutex_);
    return entries_.size();
}

std::optional<std::vector<float>> ColdTierIndex::getVector(uint64_t id) const {
    std::lock_guard lock(mutex_);
    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end()) return std::nullopt;
    return entries_[it->second].vector;
}

}  // namespace amind
