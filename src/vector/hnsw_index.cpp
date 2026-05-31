#include "hnsw_index.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <queue>
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace amind {

// CRC32 implementation for data integrity verification
namespace {
    uint32_t crc32Table[256];
    std::once_flag crc32InitFlag;

    void initCRC32Table() {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t crc = i;
            for (int j = 0; j < 8; j++) {
                crc = (crc >> 1) ^ ((crc & 1) ? 0xEDB88320 : 0);
            }
            crc32Table[i] = crc;
        }
    }

    uint32_t computeCRC32(const uint8_t* data, size_t length) {
        std::call_once(crc32InitFlag, initCRC32Table);
        uint32_t crc = 0xFFFFFFFF;
        for (size_t i = 0; i < length; i++) {
            crc = (crc >> 8) ^ crc32Table[(crc ^ data[i]) & 0xFF];
        }
        return ~crc;
    }
}

// File format constants
constexpr uint32_t HNSW_MAGIC = 0x484E5357;  // "HNSW"
constexpr uint32_t HNSW_VERSION = 1;

HNSWIndex::HNSWIndex(HNSWConfig config)
    : config_(std::move(config)) {
    if (config_.dimension == 0) {
        throw std::invalid_argument("HNSWIndex: dimension must be > 0");
    }
    if (config_.maxConnections == 0) {
        throw std::invalid_argument("HNSWIndex: maxConnections must be > 0");
    }

    // levelMultiplier = 1 / ln(M), used to generate random levels
    levelMultiplier_ = 1.0 / std::log(static_cast<double>(config_.maxConnections));
}

// ── Distance computation ──────────────────────────────────────────────────

float HNSWIndex::computeDistance(std::span<const float> vectorA,
                                  std::span<const float> vectorB) const {
    switch (config_.metric) {
        case DistanceMetric::Cosine:
            return cosineDistance(vectorA, vectorB);
        case DistanceMetric::L2:
            return l2DistanceSquared(vectorA, vectorB);
        case DistanceMetric::InnerProduct:
            // For inner product, higher is better, so negate for min-distance semantics
            return -innerProduct(vectorA, vectorB);
    }
    return cosineDistance(vectorA, vectorB);
}

// ── Random level generation ───────────────────────────────────────────────

int HNSWIndex::randomLevel() {
    // Generate level from geometric distribution: floor(-ln(uniform) * levelMultiplier)
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double randomValue = distribution(rng_);
    int level = static_cast<int>(-std::log(randomValue) * levelMultiplier_);
    return level;
}

// ── Node access ───────────────────────────────────────────────────────────

const HNSWIndex::Node* HNSWIndex::getNode(uint64_t id) const {
    auto iterator = nodes_.find(id);
    if (iterator == nodes_.end()) return nullptr;
    return &iterator->second;
}

HNSWIndex::Node* HNSWIndex::getNode(uint64_t id) {
    auto iterator = nodes_.find(id);
    if (iterator == nodes_.end()) return nullptr;
    return &iterator->second;
}

size_t HNSWIndex::maxConnectionsForLayer(int layer) const {
    return (layer == 0) ? config_.maxConnectionsLayer0 : config_.maxConnections;
}

// ── Insert ────────────────────────────────────────────────────────────────

void HNSWIndex::insert(uint64_t id, std::span<const float> vector) {
    if (vector.size() != config_.dimension) {
        spdlog::error("HNSWIndex::insert: dimension mismatch (got {}, expected {})",
                      vector.size(), config_.dimension);
        return;
    }

    std::unique_lock<std::shared_mutex> lock(mutex_);

    // Check if ID already exists — update the vector
    auto existingNode = getNode(id);
    if (existingNode != nullptr) {
        existingNode->vector.assign(vector.begin(), vector.end());
        // Check deleted status BEFORE clearing it so we can fix the active count.
        if (existingNode->deleted) {
            existingNode->deleted = false;
            activeCount_++;
        }
        return;
    }

    int newLevel = randomLevel();

    // Create the new node
    Node newNode;
    newNode.id = id;
    newNode.vector.assign(vector.begin(), vector.end());
    newNode.level = newLevel;
    newNode.connections.resize(newLevel + 1);

    // First node: just set as entry point
    if (nodes_.empty()) {
        nodes_.emplace(id, std::move(newNode));
        entryPointId_ = id;
        maxLevel_ = newLevel;
        activeCount_ = 1;
        return;
    }

    // Phase 1: Greedy descent from top layer to newLevel+1
    uint64_t currentEntryId = entryPointId_;
    for (int layer = maxLevel_; layer > newLevel; --layer) {
        currentEntryId = greedyClosest(vector, currentEntryId, layer);
    }

    // Phase 2: Insert at each layer from min(newLevel, maxLevel_) down to 0
    std::vector<uint64_t> entryPoints = {currentEntryId};

    for (int layer = std::min(newLevel, maxLevel_); layer >= 0; --layer) {
        // Find ef_construction nearest neighbors at this layer
        auto candidates = searchLayer(vector, entryPoints, config_.efConstruction, layer);

        // Select the best neighbors
        size_t maxConn = maxConnectionsForLayer(layer);
        auto selectedNeighbors = selectNeighbors(vector, candidates, maxConn);

        // Store the node first so we can reference it
        if (nodes_.find(id) == nodes_.end()) {
            nodes_.emplace(id, std::move(newNode));
        }

        // Add bidirectional connections
        auto* insertedNode = getNode(id);
        insertedNode->connections[layer] = selectedNeighbors;

        for (uint64_t neighborId : selectedNeighbors) {
            auto* neighborNode = getNode(neighborId);
            if (neighborNode == nullptr || neighborNode->deleted) continue;

            if (layer < static_cast<int>(neighborNode->connections.size())) {
                neighborNode->connections[layer].push_back(id);

                // Prune if over capacity
                if (neighborNode->connections[layer].size() > maxConn) {
                    // Build candidates for pruning
                    std::vector<SearchResult> pruningCandidates;
                    pruningCandidates.reserve(neighborNode->connections[layer].size());
                    for (uint64_t connId : neighborNode->connections[layer]) {
                        auto* connNode = getNode(connId);
                        if (connNode == nullptr || connNode->deleted) continue;
                        float dist = computeDistance(neighborNode->vector, connNode->vector);
                        pruningCandidates.push_back({connId, dist});
                    }
                    neighborNode->connections[layer] =
                        selectNeighbors(neighborNode->vector, pruningCandidates, maxConn);
                }
            }
        }

        // Use the found candidates as entry points for the next layer down
        entryPoints.clear();
        for (const auto& candidate : candidates) {
            entryPoints.push_back(candidate.id);
            if (entryPoints.size() >= config_.efConstruction) break;
        }
    }

    // Update entry point if new node has a higher level
    if (newLevel > maxLevel_) {
        entryPointId_ = id;
        maxLevel_ = newLevel;
    }

    activeCount_++;

    // Remove from cold tier if it was there (promoted back to hot)
    if (cold_tier_) {
        cold_tier_->remove(id);
    }

    // Check if hot layer needs eviction (done outside lock)
    // Note: evictCold() acquires its own lock, so we unlock first
}

// ── Search ────────────────────────────────────────────────────────────────

std::vector<SearchResult> HNSWIndex::search(std::span<const float> query,
                                             size_t topK) const {
    if (query.size() != config_.dimension) {
        spdlog::error("HNSWIndex::search: dimension mismatch (got {}, expected {})",
                      query.size(), config_.dimension);
        return {};
    }

    std::shared_lock<std::shared_mutex> lock(mutex_);

    if (nodes_.empty() || activeCount_ == 0) {
        return {};
    }

    // Phase 1: Greedy descent from top layer to layer 1
    uint64_t currentEntryId = entryPointId_;
    for (int layer = maxLevel_; layer > 0; --layer) {
        currentEntryId = greedyClosest(query, currentEntryId, layer);
    }

    // Phase 2: Beam search at layer 0 with ef candidates
    size_t efSearch = std::max(topK, config_.efSearch);
    auto candidates = searchLayer(query, {currentEntryId}, efSearch, 0, nullptr);

    // Sort by distance ascending and take top-k
    std::sort(candidates.begin(), candidates.end());

    if (candidates.size() > topK) {
        candidates.resize(topK);
    }

    // Also search cold tier and merge results
    if (cold_tier_ && cold_tier_->size() > 0) {
        std::vector<float> query_vec(query.begin(), query.end());
        auto cold_results = cold_tier_->search(query_vec, topK);
        if (!cold_results.empty()) {
            return mergeResults(candidates, cold_results, topK);
        }
    }

    return candidates;
}

// ── Pre-filtered search (Milvus Knowhere pattern) ─────────────────────────

std::vector<SearchResult> HNSWIndex::searchWithFilter(
    std::span<const float> query, size_t topK, const FilterFunc& filter) const {

    if (!filter) {
        return search(query, topK);
    }

    std::shared_lock<std::shared_mutex> lock(mutex_);

    if (nodes_.empty() || activeCount_ == 0) {
        return {};
    }

    // Milvus strategy: estimate filter match rate and decide search method.
    // Reference: Milvus WhetherPerformBruteForceSearch() with kHnswSearchKnnBFFilterThreshold = 0.93
    size_t matchCount = 0;
    for (const auto& [nodeId, node] : nodes_) {
        if (!node.deleted && filter(nodeId)) {
            matchCount++;
        }
    }

    float matchRate = (activeCount_ > 0)
        ? static_cast<float>(matchCount) / static_cast<float>(activeCount_)
        : 0.0f;

    // Fallback to brute-force when match rate is too low (Milvus threshold: < 7%)
    if (matchRate < kBruteForceMatchRateThreshold) {
        return bruteForceSearch(query, topK, filter);
    }

    // HNSW search with pre-filter
    uint64_t currentEntryId = entryPointId_;
    for (int layer = maxLevel_; layer > 0; --layer) {
        currentEntryId = greedyClosest(query, currentEntryId, layer);
    }

    // Use larger ef to compensate for filtered-out nodes (Milvus kAlpha pattern)
    size_t adjustedEf = std::max(topK, config_.efSearch);
    if (matchRate < 0.5f) {
        // When many nodes are filtered, increase ef to explore more candidates
        adjustedEf = static_cast<size_t>(
            static_cast<float>(adjustedEf) / std::max(matchRate, 0.1f));
        adjustedEf = std::min(adjustedEf, activeCount_);
    }

    auto candidates = searchLayer(query, {currentEntryId}, adjustedEf, 0, filter);

    std::sort(candidates.begin(), candidates.end());

    if (candidates.size() > topK) {
        candidates.resize(topK);
    }

    // Milvus runtime fallback: if HNSW didn't find enough results, try brute-force
    if (candidates.size() < topK && candidates.size() < matchCount) {
        spdlog::debug("HNSW filtered search found {}/{} results, falling back to brute-force",
                      candidates.size(), topK);
        return bruteForceSearch(query, topK, filter);
    }

    return candidates;
}

// ── Remove ────────────────────────────────────────────────────────────────

bool HNSWIndex::remove(uint64_t id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto* node = getNode(id);
    if (node == nullptr || node->deleted) {
        return false;
    }

    node->deleted = true;
    activeCount_--;
    return true;
}

// ── Queries ───────────────────────────────────────────────────────────────

bool HNSWIndex::contains(uint64_t id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto* node = getNode(id);
    return node != nullptr && !node->deleted;
}

size_t HNSWIndex::size() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return activeCount_;
}

bool HNSWIndex::empty() const {
    return size() == 0;
}

int HNSWIndex::maxLevel() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return maxLevel_;
}

// ── Greedy closest ────────────────────────────────────────────────────────

uint64_t HNSWIndex::greedyClosest(std::span<const float> query,
                                   uint64_t entryId,
                                   int layer) const {
    const auto* currentNode = getNode(entryId);
    if (currentNode == nullptr) return entryId;

    float currentDistance = computeDistance(query, currentNode->vector);
    bool improved = true;

    while (improved) {
        improved = false;

        if (layer >= static_cast<int>(currentNode->connections.size())) break;

        for (uint64_t neighborId : currentNode->connections[layer]) {
            const auto* neighborNode = getNode(neighborId);
            if (neighborNode == nullptr || neighborNode->deleted) continue;

            float neighborDistance = computeDistance(query, neighborNode->vector);
            if (neighborDistance < currentDistance) {
                currentDistance = neighborDistance;
                currentNode = neighborNode;
                entryId = neighborId;
                improved = true;
            }
        }
    }

    return entryId;
}

// ── Search layer (beam search) ────────────────────────────────────────────

std::vector<SearchResult> HNSWIndex::searchLayer(
    std::span<const float> query,
    const std::vector<uint64_t>& entryPoints,
    size_t ef,
    int layer,
    const FilterFunc& filter) const {

    // Min-heap for candidates (closest first)
    auto candidateMinCmp = [](const SearchResult& a, const SearchResult& b) {
        return a.distance > b.distance;  // min-heap
    };
    std::priority_queue<SearchResult, std::vector<SearchResult>,
                        decltype(candidateMinCmp)> candidateQueue(candidateMinCmp);

    // Max-heap for results (farthest first, for pruning)
    auto resultMaxCmp = [](const SearchResult& a, const SearchResult& b) {
        return a.distance < b.distance;  // max-heap
    };
    std::priority_queue<SearchResult, std::vector<SearchResult>,
                        decltype(resultMaxCmp)> resultQueue(resultMaxCmp);

    std::unordered_set<uint64_t> visited;

    // Initialize with entry points
    for (uint64_t entryId : entryPoints) {
        const auto* entryNode = getNode(entryId);
        if (entryNode == nullptr) continue;

        float dist = computeDistance(query, entryNode->vector);
        // Always add to candidate queue so we explore neighbors
        candidateQueue.push({entryId, dist});

        // Milvus pattern: add to result queue only if node passes filter AND is not deleted.
        // Filtered-out nodes are still explored (their neighbors are traversed)
        // to maintain graph connectivity.
        bool passesFilter = !entryNode->deleted
            && (!filter || filter(entryId));
        if (passesFilter) {
            resultQueue.push({entryId, dist});
        }
        visited.insert(entryId);
    }

    while (!candidateQueue.empty()) {
        auto closest = candidateQueue.top();
        candidateQueue.pop();

        // If the closest candidate is farther than the farthest result, stop
        if (!resultQueue.empty() && closest.distance > resultQueue.top().distance
            && resultQueue.size() >= ef) {
            break;
        }

        const auto* closestNode = getNode(closest.id);
        if (closestNode == nullptr) continue;
        if (layer >= static_cast<int>(closestNode->connections.size())) continue;

        for (uint64_t neighborId : closestNode->connections[layer]) {
            if (visited.count(neighborId) > 0) continue;
            visited.insert(neighborId);

            const auto* neighborNode = getNode(neighborId);
            if (neighborNode == nullptr) continue;

            float neighborDist = computeDistance(query, neighborNode->vector);

            bool shouldAdd = resultQueue.size() < ef
                             || neighborDist < resultQueue.top().distance;

            // Milvus Knowhere pattern: check both deleted status AND filter.
            // Nodes that fail the filter are treated like deleted nodes —
            // we still explore their neighbors but don't add them to results.
            bool passesFilter = !neighborNode->deleted
                && (!filter || filter(neighborId));

            if (shouldAdd && passesFilter) {
                candidateQueue.push({neighborId, neighborDist});
                resultQueue.push({neighborId, neighborDist});

                if (resultQueue.size() > ef) {
                    resultQueue.pop();
                }
            } else if (shouldAdd) {
                // Filtered/deleted node: still explore its neighbors for graph connectivity
                candidateQueue.push({neighborId, neighborDist});
            }
        }
    }

    // Extract results from the max-heap
    std::vector<SearchResult> results;
    results.reserve(resultQueue.size());
    while (!resultQueue.empty()) {
        results.push_back(resultQueue.top());
        resultQueue.pop();
    }

    // Sort by distance ascending
    std::sort(results.begin(), results.end());
    return results;
}

// ── Select neighbors ──────────────────────────────────────────────────────

std::vector<uint64_t> HNSWIndex::selectNeighbors(
    std::span<const float> nodeVector,
    const std::vector<SearchResult>& candidates,
    size_t maxConnections) const {

    // Simple heuristic: keep the M closest non-deleted neighbors
    auto sorted = candidates;
    std::sort(sorted.begin(), sorted.end());

    std::vector<uint64_t> selected;
    selected.reserve(maxConnections);

    for (const auto& candidate : sorted) {
        if (selected.size() >= maxConnections) break;

        const auto* candidateNode = getNode(candidate.id);
        if (candidateNode == nullptr || candidateNode->deleted) continue;

        selected.push_back(candidate.id);
    }

    return selected;
}

// ── Brute-force search (Milvus fallback) ──────────────────────────────────

std::vector<SearchResult> HNSWIndex::bruteForceSearch(
    std::span<const float> query, size_t topK, const FilterFunc& filter) const {

    // Reference: Milvus IndexBruteForceWrapper — linear scan all nodes,
    // apply filter, keep top-K by distance using a max-heap.
    auto resultMaxCmp = [](const SearchResult& a, const SearchResult& b) {
        return a.distance < b.distance;  // max-heap: farthest on top
    };
    std::priority_queue<SearchResult, std::vector<SearchResult>,
                        decltype(resultMaxCmp)> resultHeap(resultMaxCmp);

    for (const auto& [nodeId, node] : nodes_) {
        if (node.deleted) continue;
        if (filter && !filter(nodeId)) continue;

        float dist = computeDistance(query, node.vector);

        if (resultHeap.size() < topK) {
            resultHeap.push({nodeId, dist});
        } else if (dist < resultHeap.top().distance) {
            resultHeap.pop();
            resultHeap.push({nodeId, dist});
        }
    }

    std::vector<SearchResult> results;
    results.reserve(resultHeap.size());
    while (!resultHeap.empty()) {
        results.push_back(resultHeap.top());
        resultHeap.pop();
    }

    std::sort(results.begin(), results.end());
    return results;
}

// ── Serialization: Save/Load ───────────────────────────────────────────────

bool HNSWIndex::save(const std::string& path) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    // Write to temp file, then atomic rename to avoid partial writes on crash
    std::string tmp_path = path + ".tmp";
    std::ofstream file(tmp_path, std::ios::binary);
    if (!file.is_open()) {
        spdlog::error("HNSWIndex::save: failed to open file {}", tmp_path);
        return false;
    }

    // Streaming CRC: accumulate bytes as we write
    uint32_t crc = 0xFFFFFFFF;
    auto crcWrite = [&](const void* data, size_t len) {
        file.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(len));
        auto* bytes = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < len; i++) {
            crc = (crc >> 8) ^ crc32Table[(crc ^ bytes[i]) & 0xFF];
        }
    };

    // Ensure CRC table is initialized
    std::call_once(crc32InitFlag, initCRC32Table);

    // Write header
    uint32_t magic = HNSW_MAGIC;
    uint32_t version = HNSW_VERSION;
    uint32_t dimension = static_cast<uint32_t>(config_.dimension);
    uint32_t nodeCount = static_cast<uint32_t>(nodes_.size());
    uint64_t entryPointId = entryPointId_;
    int32_t maxLevel = static_cast<int32_t>(maxLevel_);
    uint32_t maxConnections = static_cast<uint32_t>(config_.maxConnections);
    uint32_t maxConnectionsLayer0 = static_cast<uint32_t>(config_.maxConnectionsLayer0);
    uint32_t efConstruction = static_cast<uint32_t>(config_.efConstruction);
    uint32_t efSearch = static_cast<uint32_t>(config_.efSearch);
    uint8_t metric = static_cast<uint8_t>(config_.metric);

    crcWrite(&magic, sizeof(magic));
    crcWrite(&version, sizeof(version));
    crcWrite(&dimension, sizeof(dimension));
    crcWrite(&nodeCount, sizeof(nodeCount));
    crcWrite(&entryPointId, sizeof(entryPointId));
    crcWrite(&maxLevel, sizeof(maxLevel));
    crcWrite(&maxConnections, sizeof(maxConnections));
    crcWrite(&maxConnectionsLayer0, sizeof(maxConnectionsLayer0));
    crcWrite(&efConstruction, sizeof(efConstruction));
    crcWrite(&efSearch, sizeof(efSearch));
    crcWrite(&metric, sizeof(metric));

    if (!file.good()) {
        spdlog::error("HNSWIndex::save: failed to write header");
        return false;
    }

    // Write nodes
    for (const auto& [id, node] : nodes_) {
        uint64_t nodeId = node.id;
        int32_t level = static_cast<int32_t>(node.level);
        uint8_t deleted = node.deleted ? 1 : 0;

        crcWrite(&nodeId, sizeof(nodeId));
        crcWrite(&level, sizeof(level));
        crcWrite(&deleted, sizeof(deleted));

        if (node.vector.size() != config_.dimension) {
            spdlog::error("HNSWIndex::save: vector dimension mismatch for node {}", nodeId);
            return false;
        }
        crcWrite(node.vector.data(), config_.dimension * sizeof(float));

        for (int l = 0; l <= node.level; ++l) {
            uint32_t connectionCount = static_cast<uint32_t>(node.connections[l].size());
            crcWrite(&connectionCount, sizeof(connectionCount));
            for (uint64_t neighborId : node.connections[l]) {
                crcWrite(&neighborId, sizeof(neighborId));
            }
        }
    }

    if (!file.good()) {
        spdlog::error("HNSWIndex::save: failed to write nodes");
        return false;
    }

    // Finalize CRC and append (not included in the CRC itself)
    uint32_t final_crc = ~crc;
    file.write(reinterpret_cast<const char*>(&final_crc), sizeof(final_crc));
    file.close();

    // Atomic rename: tmp → final path
    std::rename(tmp_path.c_str(), path.c_str());

    spdlog::info("HNSWIndex::save: successfully saved {} nodes to {}", nodes_.size(), path);
    return true;
}

bool HNSWIndex::load(const std::string& path) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        spdlog::warn("HNSWIndex::load: failed to open file {}", path);
        return false;
    }

    // Read header
    uint32_t magic, version, dimension, nodeCount;
    uint64_t entryPointId;
    int32_t maxLevel;
    uint32_t maxConnections, maxConnectionsLayer0, efConstruction, efSearch;
    uint8_t metric;

    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
    file.read(reinterpret_cast<char*>(&nodeCount), sizeof(nodeCount));
    file.read(reinterpret_cast<char*>(&entryPointId), sizeof(entryPointId));
    file.read(reinterpret_cast<char*>(&maxLevel), sizeof(maxLevel));
    file.read(reinterpret_cast<char*>(&maxConnections), sizeof(maxConnections));
    file.read(reinterpret_cast<char*>(&maxConnectionsLayer0), sizeof(maxConnectionsLayer0));
    file.read(reinterpret_cast<char*>(&efConstruction), sizeof(efConstruction));
    file.read(reinterpret_cast<char*>(&efSearch), sizeof(efSearch));
    file.read(reinterpret_cast<char*>(&metric), sizeof(metric));

    if (!file.good()) {
        spdlog::error("HNSWIndex::load: failed to read header");
        return false;
    }

    // Validate magic number
    if (magic != HNSW_MAGIC) {
        spdlog::error("HNSWIndex::load: invalid magic number {:#x}", magic);
        return false;
    }

    // Validate version — if the file was written by a different version of the
    // index format, we cannot safely load it. Return false so the caller can
    // rebuild the index from the LSM store (graceful degradation).
    if (version != HNSW_VERSION) {
        spdlog::warn(
            "HNSWIndex::load: file version {} is incompatible with current version {}. "
            "The index will be rebuilt from the LSM store.",
            version, HNSW_VERSION);
        return false;
    }

    // Validate dimension
    if (dimension != config_.dimension) {
        spdlog::error("HNSWIndex::load: dimension mismatch (file: {}, config: {})", 
                      dimension, config_.dimension);
        return false;
    }

    // Clear existing data
    nodes_.clear();
    activeCount_ = 0;
    entryPointId_ = entryPointId;
    maxLevel_ = maxLevel;

    // Read nodes
    for (uint32_t i = 0; i < nodeCount; ++i) {
        uint64_t nodeId;
        int32_t level;
        uint8_t deleted;

        file.read(reinterpret_cast<char*>(&nodeId), sizeof(nodeId));
        file.read(reinterpret_cast<char*>(&level), sizeof(level));
        file.read(reinterpret_cast<char*>(&deleted), sizeof(deleted));

        if (!file.good()) {
            spdlog::error("HNSWIndex::load: failed to read node header");
            return false;
        }

        Node node;
        node.id = nodeId;
        node.level = level;
        node.deleted = (deleted == 1);

        // Read vector data
        node.vector.resize(config_.dimension);
        file.read(reinterpret_cast<char*>(node.vector.data()), 
                  config_.dimension * sizeof(float));

        if (!file.good()) {
            spdlog::error("HNSWIndex::load: failed to read vector data for node {}", nodeId);
            return false;
        }

        // Read connections for each level
        node.connections.resize(level + 1);
        for (int l = 0; l <= level; ++l) {
            uint32_t connectionCount;
            file.read(reinterpret_cast<char*>(&connectionCount), sizeof(connectionCount));

            if (!file.good()) {
                spdlog::error("HNSWIndex::load: failed to read connection count for node {}", nodeId);
                return false;
            }

            node.connections[l].resize(connectionCount);
            for (uint32_t j = 0; j < connectionCount; ++j) {
                uint64_t neighborId;
                file.read(reinterpret_cast<char*>(&neighborId), sizeof(neighborId));
                node.connections[l][j] = neighborId;
            }

            if (!file.good()) {
                spdlog::error("HNSWIndex::load: failed to read connections for node {}", nodeId);
                return false;
            }
        }

        bool isDeleted = node.deleted;
        nodes_.emplace(nodeId, std::move(node));
        if (!isDeleted) {
            activeCount_++;
        }
    }

    // Read and verify CRC32 checksum
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    size_t dataSize = fileSize - sizeof(uint32_t);
    std::vector<uint8_t> fileData(dataSize);
    
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(fileData.data()), dataSize);
    
    uint32_t storedCRC;
    file.read(reinterpret_cast<char*>(&storedCRC), sizeof(storedCRC));

    if (!file.good()) {
        spdlog::error("HNSWIndex::load: failed to read CRC32");
        return false;
    }

    uint32_t computedCRC = computeCRC32(fileData.data(), dataSize);
    if (computedCRC != storedCRC) {
        spdlog::error("HNSWIndex::load: CRC32 mismatch (stored: {:#x}, computed: {:#x})", 
                      storedCRC, computedCRC);
        return false;
    }

    file.close();
    spdlog::info("HNSWIndex::load: successfully loaded {} nodes from {}", nodes_.size(), path);
    return true;
}

// ── V2: Enhanced deletion & compaction ─────────────────────────────────────

bool HNSWIndex::softDelete(uint64_t id) {
    // Semantically identical to remove() for now.
    // In future PRs, callers will pass RemoveReason to higher-level APIs.
    return remove(id);
}

size_t HNSWIndex::compact() {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    // Collect surviving nodes
    std::vector<std::pair<uint64_t, std::vector<float>>> survivors;
    survivors.reserve(activeCount_);
    for (const auto& [id, node] : nodes_) {
        if (!node.deleted) {
            survivors.emplace_back(id, node.vector);
        }
    }

    size_t purged = nodes_.size() - survivors.size();
    if (purged == 0) return 0;

    // Clear everything and rebuild
    nodes_.clear();
    entryPointId_ = 0;
    maxLevel_ = -1;
    activeCount_ = 0;

    // Temporarily release lock for re-insertion (insert acquires lock)
    // We do unlocked insertions using internal helpers instead.
    // But since insert() takes the lock, we need to do raw node creation here.
    // For simplicity and correctness, we rebuild by re-inserting with unlocked mutex.
    // We already hold the lock, so we call a lockless insert path.

    // Re-insert all survivors (lockless, since we hold mutex_)
    for (const auto& [id, vec] : survivors) {
        // Create node
        int level = randomLevel();
        Node node;
        node.id = id;
        node.vector = vec;
        node.level = level;
        node.deleted = false;
        node.connections.resize(level + 1);

        nodes_[id] = std::move(node);
        activeCount_++;

        if (maxLevel_ == -1 || level > maxLevel_) {
            maxLevel_ = level;
            entryPointId_ = id;
        }
    }

    // Rebuild connections for all nodes
    for (auto& [id, node] : nodes_) {
        for (int layer = 0; layer <= node.level; ++layer) {
            size_t maxConn = maxConnectionsForLayer(layer);

            // Find neighbors via search on this layer
            std::vector<uint64_t> entryPoints = {entryPointId_};
            auto candidates = searchLayer(node.vector, entryPoints,
                                          config_.efConstruction, layer);
            auto neighbors = selectNeighbors(node.vector, candidates, maxConn);

            node.connections[layer] = neighbors;

            // Add reverse connections
            for (uint64_t neighborId : neighbors) {
                auto* neighbor = getNode(neighborId);
                if (!neighbor) continue;
                if (layer < static_cast<int>(neighbor->connections.size())) {
                    auto& nconns = neighbor->connections[layer];
                    if (nconns.size() < maxConn) {
                        nconns.push_back(id);
                    }
                }
            }
        }
    }

    spdlog::info("HNSWIndex::compact: purged {} deleted nodes, rebuilt graph with {} nodes",
                 purged, activeCount_);
    return purged;
}

size_t HNSWIndex::relegateToCold(const std::vector<uint64_t>& ids) {
    if (!cold_tier_) return 0;

    std::unique_lock<std::shared_mutex> lock(mutex_);

    size_t relegated = 0;
    for (uint64_t id : ids) {
        auto* node = getNode(id);
        if (!node || node->deleted) continue;

        cold_tier_->append(id, node->vector);
        node->deleted = true;
        activeCount_--;
        relegated++;
    }

    if (relegated > 0) {
        spdlog::info("HNSWIndex::relegateToCold: moved {} vectors to cold tier (hot: {}, cold: {})",
                     relegated, activeCount_, cold_tier_->size());
    }
    return relegated;
}

size_t HNSWIndex::deletedCount() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    size_t total = 0;
    for (const auto& [id, node] : nodes_) {
        if (node.deleted) total++;
    }
    return total;
}

// ── Hot/Cold Tiering ──────────────────────────────────────────────────────

void HNSWIndex::initColdTier() {
    if (config_.coldTierPath.empty()) return;
    cold_tier_ = std::make_unique<ColdTierIndex>(
        config_.coldTierPath, config_.dimension, config_.metric);
    cold_tier_->load();
    spdlog::info("HNSWIndex: cold tier initialized at {}, {} vectors",
                 config_.coldTierPath, cold_tier_->size());
}

size_t HNSWIndex::evictCold() {
    if (!cold_tier_) return 0;

    std::unique_lock<std::shared_mutex> lock(mutex_);

    if (activeCount_ <= config_.hotCapacity) return 0;

    // Collect nodes sorted by last_accessed (oldest first)
    std::vector<std::pair<uint32_t, uint64_t>> candidates;
    for (const auto& [id, node] : nodes_) {
        if (!node.deleted) {
            candidates.emplace_back(node.last_accessed, id);
        }
    }
    std::sort(candidates.begin(), candidates.end());

    size_t evict_count = static_cast<size_t>(
        static_cast<float>(activeCount_) * config_.evictionRatio);
    evict_count = std::min(evict_count, candidates.size());

    size_t evicted = 0;
    for (size_t i = 0; i < evict_count; i++) {
        uint64_t victim_id = candidates[i].second;
        auto* node = getNode(victim_id);
        if (!node || node->deleted) continue;

        // Move vector to cold tier
        cold_tier_->append(victim_id, node->vector);

        // Lazy delete from HNSW (connections not repaired)
        node->deleted = true;
        activeCount_--;
        evicted++;
    }

    if (evicted > 0) {
        spdlog::info("HNSWIndex: evicted {} vectors to cold tier (hot: {}, cold: {})",
                     evicted, activeCount_, cold_tier_->size());
    }
    return evicted;
}

std::vector<SearchResult> HNSWIndex::mergeResults(
    std::vector<SearchResult>& hot, std::vector<SearchResult>& cold, size_t topK) const {

    std::vector<SearchResult> merged;
    merged.reserve(hot.size() + cold.size());
    merged.insert(merged.end(), hot.begin(), hot.end());
    merged.insert(merged.end(), cold.begin(), cold.end());

    // Deduplicate by ID (keep closer distance)
    std::unordered_map<uint64_t, float> best;
    for (const auto& r : merged) {
        auto it = best.find(r.id);
        if (it == best.end() || r.distance < it->second) {
            best[r.id] = r.distance;
        }
    }

    std::vector<SearchResult> result;
    result.reserve(best.size());
    for (const auto& [id, dist] : best) {
        result.push_back({id, dist});
    }
    std::sort(result.begin(), result.end());

    if (result.size() > topK) {
        result.resize(topK);
    }
    return result;
}

size_t HNSWIndex::hotSize() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return activeCount_;
}

size_t HNSWIndex::coldSize() const {
    if (!cold_tier_) return 0;
    return cold_tier_->size();
}

}  // namespace amind
