#include "memory_store.h"

#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>
#include <filesystem>

namespace amind {

MemoryStore::MemoryStore(Config config)
    : config_(std::move(config)), id_gen_(0) {}

MemoryStore::~MemoryStore() {
    flush();
}

Result<void, Error> MemoryStore::init() {
    namespace fs = std::filesystem;

    try {
        fs::create_directories(config_.data_dir);
        fs::create_directories(config_.data_dir + "/lsm");
        fs::create_directories(config_.data_dir + "/hnsw");
    } catch (const fs::filesystem_error& e) {
        return makeError(Error::IOError, "cannot create data dirs: " + std::string(e.what()));
    }

    // Initialize LSM engine
    lsm_ = std::make_unique<LSMEngine>(fs::path(config_.data_dir + "/lsm"),
                                        config_.lsm_flush_interval);

    // Initialize HNSW index
    HNSWConfig hnsw_cfg;
    hnsw_cfg.dimension = config_.embedding_dim;
    hnsw_cfg.maxConnections = config_.hnsw_max_connections;
    hnsw_cfg.efConstruction = config_.hnsw_ef_construction;
    hnsw_cfg.hotCapacity = config_.hnsw_hot_capacity;
    hnsw_cfg.coldTierPath = config_.hnsw_cold_tier_path;
    hnsw_cfg.evictionRatio = config_.hnsw_eviction_ratio;
    hnsw_ = std::make_unique<HNSWIndex>(hnsw_cfg);

    // Try to load persisted HNSW index
    auto hnsw_path = config_.data_dir + "/hnsw/index.bin";
    if (fs::exists(hnsw_path)) {
        if (hnsw_->load(hnsw_path)) {
            spdlog::info("Loaded HNSW index from {}", hnsw_path);
        }
    }

    // Reconcile: re-insert any LSM records whose embeddings are missing from HNSW.
    // This recovers from unclean shutdowns where LSM flushed but HNSW didn't save.
    size_t reconciled = 0;
    lsm_->forEachLive([&](uint64_t memory_id, const std::vector<uint8_t>& data) {
        if (hnsw_->contains(memory_id)) return;
        auto result = MemoryRecord::deserialize(data);
        if (!result.ok()) return;
        auto& rec = result.value();
        if (!rec.isAlive()) return;
        if (rec.embedding.empty()) return;
        hnsw_->insert(memory_id, rec.embedding);
        cache_[memory_id] = std::move(rec);
        ++reconciled;
    });
    if (reconciled > 0) {
        spdlog::info("HNSW reconciliation: re-inserted {} orphaned records", reconciled);
        hnsw_->save(hnsw_path);
    }

    // Warm cache with all live records so totalMemories() and listMemories()
    // reflect the full dataset immediately after startup.
    size_t warmed = 0;
    lsm_->forEachLive([&](uint64_t memory_id, const std::vector<uint8_t>& data) {
        if (cache_.count(memory_id)) return;
        auto result = MemoryRecord::deserialize(data);
        if (!result.ok()) return;
        auto& rec = result.value();
        if (!rec.isAlive()) return;
        cache_[memory_id] = std::move(rec);
        ++warmed;
    });
    spdlog::info("Cache warmup: loaded {} records from LSM (total cache={})",
                 warmed, cache_.size());

    spdlog::info("MemoryStore initialized: data_dir={}", config_.data_dir);
    return Result<void, Error>();
}

Result<uint64_t> MemoryStore::fastStore(MemoryRecord record) {
    std::unique_lock lock(mutex_);

    // Generate ID if not set
    if (record.memory_id == 0) {
        record.memory_id = id_gen_.nextId();
    }
    if (record.created_at == 0) {
        record.created_at = MemoryRecord::currentTimeSec();
        record.last_accessed = record.created_at;
    }

    uint64_t mid = record.memory_id;

    // Insert embedding into HNSW
    if (!record.embedding.empty()) {
        hnsw_->insert(mid, record.embedding);
        record.flags |= RecordFlags::HAS_EMBEDDING;
    }

    // Write to LSM via raw serialized bytes
    auto serialized = record.serialize();
    lsm_->putRaw(mid, std::move(serialized));

    // Cache
    cache_[mid] = std::move(record);
    evictIfNeeded();

    return mid;
}

bool MemoryStore::contains(uint64_t memory_id) const {
    // Check cache first (read-only, no side effects)
    {
        std::shared_lock lock(mutex_);
        if (cache_.count(memory_id)) return true;
    }
    // Fall back to LSM
    return lsm_->get(memory_id).has_value();
}

Result<MemoryRecord> MemoryStore::peek(uint64_t memory_id) const {
    // Read-only access: no access_count increment, no tier promotion.
    // Used by internal pipelines (Stage 2 refinement, forget matching, etc.).
    {
        std::shared_lock lock(mutex_);
        auto it = cache_.find(memory_id);
        if (it != cache_.end()) {
            return it->second;
        }
    }
    // Fall back to LSM
    auto raw = lsm_->get(memory_id);
    if (!raw.has_value()) {
        return makeError(Error::NotFound, "memory not found: " + std::to_string(memory_id));
    }
    return MemoryRecord::deserialize(raw.value());
}

Result<MemoryRecord> MemoryStore::get(uint64_t memory_id) {
    // Fast path: check cache under exclusive lock (needed for access_count mutation)
    {
        std::unique_lock lock(mutex_);
        auto it = cache_.find(memory_id);
        if (it != cache_.end()) {
            it->second.access_count++;
            it->second.last_accessed = MemoryRecord::currentTimeSec();
            tryPromote(it->second);
            dirty_ids_.insert(memory_id);
            return it->second;
        }
    }

    // Slow path: read from LSM (no lock held — LSM is internally thread-safe)
    auto raw = lsm_->get(memory_id);
    if (!raw.has_value()) {
        return makeError(Error::NotFound, "memory not found: " + std::to_string(memory_id));
    }

    auto result = MemoryRecord::deserialize(raw.value());
    if (!result.ok()) return result.error();

    auto& record = result.value();
    record.access_count++;
    record.last_accessed = MemoryRecord::currentTimeSec();

    // Populate cache under exclusive lock (double-checked)
    std::unique_lock wlock(mutex_);
    auto [it, inserted] = cache_.try_emplace(memory_id, record);
    if (!inserted) {
        it->second.access_count++;
        it->second.last_accessed = MemoryRecord::currentTimeSec();
    }
    tryPromote(it->second);
    dirty_ids_.insert(memory_id);
    return it->second;
}

Result<uint64_t> MemoryStore::update(uint64_t memory_id, const std::string& new_content,
                                     const std::vector<float>& new_embedding) {
    std::unique_lock lock(mutex_);

    // Read old record under the same lock to prevent TOCTOU race
    MemoryRecord old_record;
    auto cache_it = cache_.find(memory_id);
    if (cache_it != cache_.end()) {
        old_record = cache_it->second;
    } else {
        auto raw = lsm_->get(memory_id);
        if (!raw.has_value()) {
            return makeError(Error::NotFound, "memory not found: " + std::to_string(memory_id));
        }
        auto deser = MemoryRecord::deserialize(raw.value());
        if (!deser.ok()) return deser.error();
        old_record = std::move(deser.value());
    }

    // Create new version
    uint64_t new_id = id_gen_.nextId();
    auto new_record = old_record.createNewVersion(new_id);
    new_record.content = new_content;
    if (!new_embedding.empty()) {
        new_record.embedding = new_embedding;
    }

    // Mark old as Versioned and write back
    old_record.markVersioned();
    lsm_->putRaw(memory_id, old_record.serialize());
    cache_[memory_id] = old_record;

    // Store new version
    if (!new_record.embedding.empty()) {
        hnsw_->insert(new_id, new_record.embedding);
    }
    lsm_->putRaw(new_id, new_record.serialize());
    cache_[new_id] = std::move(new_record);

    return new_id;
}

Result<void, Error> MemoryStore::remove(uint64_t memory_id) {
    std::unique_lock lock(mutex_);

    // Remove from HNSW index so tombstoned records are not searchable
    hnsw_->softDelete(memory_id);

    auto it = cache_.find(memory_id);
    if (it != cache_.end()) {
        it->second.markTombstone();
        lsm_->putRaw(memory_id, it->second.serialize());
        cache_.erase(it);
        return Result<void, Error>();
    }

    // Not in cache, check LSM
    auto raw = lsm_->get(memory_id);
    if (!raw.has_value()) {
        return makeError(Error::NotFound, "memory not found");
    }

    auto result = MemoryRecord::deserialize(raw.value());
    if (!result.ok()) return result.error();

    result.value().markTombstone();
    lsm_->putRaw(memory_id, result.value().serialize());

    return Result<void, Error>();
}

Result<void, Error> MemoryStore::archive(uint64_t memory_id) {
    std::unique_lock lock(mutex_);

    // Remove from HNSW so archived records are not searchable
    hnsw_->softDelete(memory_id);

    auto it = cache_.find(memory_id);
    if (it != cache_.end()) {
        it->second.phase = MemoryPhase::Archived;
        lsm_->putRaw(memory_id, it->second.serialize());
        return Result<void, Error>();
    }

    auto raw = lsm_->get(memory_id);
    if (!raw.has_value()) {
        return makeError(Error::NotFound, "memory not found");
    }

    auto result = MemoryRecord::deserialize(raw.value());
    if (!result.ok()) return result.error();

    result.value().phase = MemoryPhase::Archived;
    lsm_->putRaw(memory_id, result.value().serialize());

    return Result<void, Error>();
}

void MemoryStore::setEmbedding(uint64_t memory_id, std::vector<float> embedding) {
    std::unique_lock lock(mutex_);
    auto it = cache_.find(memory_id);
    if (it != cache_.end()) {
        it->second.embedding = std::move(embedding);
        it->second.flags |= RecordFlags::HAS_EMBEDDING;
        hnsw_->insert(memory_id, it->second.embedding);
        lsm_->putRaw(memory_id, it->second.serialize());
        return;
    }
    // Not in cache — load, set, persist
    auto raw = lsm_->get(memory_id);
    if (!raw.has_value()) return;
    auto result = MemoryRecord::deserialize(raw.value());
    if (!result.ok()) return;
    result.value().embedding = std::move(embedding);
    result.value().flags |= RecordFlags::HAS_EMBEDDING;
    hnsw_->insert(memory_id, result.value().embedding);
    lsm_->putRaw(memory_id, result.value().serialize());
    cache_[memory_id] = std::move(result.value());
}

void MemoryStore::boostImportance(uint64_t memory_id, float delta) {
    std::unique_lock lock(mutex_);
    auto it = cache_.find(memory_id);
    if (it != cache_.end()) {
        it->second.importance = std::clamp(it->second.importance + delta, 0.0f, 1.0f);
        lsm_->putRaw(memory_id, it->second.serialize());
        return;
    }
    // Not in cache — load from LSM, boost, write back
    auto raw = lsm_->get(memory_id);
    if (!raw.has_value()) return;
    auto result = MemoryRecord::deserialize(raw.value());
    if (!result.ok()) return;
    result.value().importance = std::clamp(result.value().importance + delta, 0.0f, 1.0f);
    lsm_->putRaw(memory_id, result.value().serialize());
    cache_[memory_id] = std::move(result.value());
}

Result<std::vector<MemoryRecord>> MemoryStore::getHistory(uint64_t memory_id, int max_depth) {
    std::vector<MemoryRecord> chain;
    uint64_t current_id = memory_id;

    for (int i = 0; i < max_depth && current_id != 0; ++i) {
        auto result = get(current_id);
        if (!result.ok()) break;
        current_id = result.value().parent_id;
        chain.push_back(std::move(result.value()));
    }
    return chain;
}

std::vector<std::pair<uint64_t, float>> MemoryStore::searchSimilar(
    const std::vector<float>& query_embedding, size_t top_k) {
    std::shared_lock lock(mutex_);
    auto results = hnsw_->search(query_embedding, top_k);
    std::vector<std::pair<uint64_t, float>> pairs;
    pairs.reserve(results.size());
    for (const auto& r : results) {
        // Convert distance to similarity (cosine distance to similarity)
        float similarity = 1.0f - r.distance;
        pairs.emplace_back(r.id, similarity);
    }
    return pairs;
}

std::vector<std::pair<uint64_t, float>> MemoryStore::searchSimilar(
    const std::vector<float>& query_embedding, size_t top_k,
    const HNSWIndex::FilterFunc& filter) {
    // Note: no mutex_ lock here — HNSW has its own internal lock, and the
    // filter callback may need to inspect cache_ which requires separate locking.
    auto results = hnsw_->searchWithFilter(query_embedding, top_k, filter);
    std::vector<std::pair<uint64_t, float>> pairs;
    pairs.reserve(results.size());
    for (const auto& r : results) {
        float similarity = 1.0f - r.distance;
        pairs.emplace_back(r.id, similarity);
    }
    return pairs;
}

bool MemoryStore::peekScopeMatch(uint64_t memory_id, const std::string& user_id) const {
    // Read-only check: does this memory's scope allow access by user_id?
    // Returns true if the memory is Shared or if user_id matches.
    // Lock-free cache peek (caller ensures thread safety or accepts races).
    std::shared_lock lock(mutex_);
    auto it = cache_.find(memory_id);
    if (it == cache_.end()) {
        // Not in cache — optimistically include (will be filtered post-search)
        return true;
    }
    const auto& rec = it->second;
    if (!rec.isAlive()) return false;
    if (rec.scope != MemoryScope::Private) return true;
    return user_id.empty() || rec.user_id == user_id;
}

std::vector<std::pair<uint64_t, float>> MemoryStore::findConflicts(
    const std::vector<float>& embedding) {
    auto candidates = searchSimilar(embedding, 10);
    std::vector<std::pair<uint64_t, float>> conflicts;
    for (const auto& [id, score] : candidates) {
        if (score >= config_.conflict_similarity_threshold) {
            conflicts.push_back({id, score});
        }
    }
    return conflicts;
}

void MemoryStore::tryPromote(MemoryRecord& record) {
    // Access-driven tier promotion (called under exclusive lock).
    // Working→ShortTerm: access_count >= threshold OR importance >= threshold
    // ShortTerm→LongTerm: access_count >= threshold AND importance >= threshold
    if (record.tier == MemoryTier::Working) {
        if (record.access_count >= config_.promotion_working_access_count
            || record.importance >= config_.promotion_working_importance) {
            spdlog::debug("Tier promotion: memory {} Working → ShortTerm "
                         "(access={}, importance={:.3f})",
                         record.memory_id, record.access_count, record.importance);
            record.tier = MemoryTier::ShortTerm;
        }
    } else if (record.tier == MemoryTier::ShortTerm) {
        if (record.access_count >= config_.promotion_short_access_count
            && record.importance >= config_.promotion_short_importance) {
            spdlog::debug("Tier promotion: memory {} ShortTerm → LongTerm "
                         "(access={}, importance={:.3f})",
                         record.memory_id, record.access_count, record.importance);
            record.tier = MemoryTier::LongTerm;
        }
    }
    // LongTerm: no further promotion.
}

void MemoryStore::applyDecay() {
    std::unique_lock lock(mutex_);
    auto now = MemoryRecord::currentTimeSec();

    // Tier-aware decay multipliers: Working decays fastest, LongTerm slowest.
    // Multiplier is applied to the base decay_rate.
    static constexpr float kTierDecayMultiplier[] = {
        2.0f,  // Working   — fast decay (×2.0)
        1.0f,  // ShortTerm — baseline   (×1.0)
        0.5f,  // LongTerm  — slow decay (×0.5)
    };

    // Tier-aware staleness thresholds (hours before marking Stale).
    static constexpr float kTierStaleHours[] = {
        72.0f,   // Working   — 3 days
        360.0f,  // ShortTerm — 15 days
        720.0f,  // LongTerm  — 30 days
    };

    // Demotion threshold: if importance drops below this, demote one tier level.
    static constexpr float kDemotionThreshold = 0.1f;

    for (auto& [id, record] : cache_) {
        if (!record.isActive()) continue;
        float age_hours = static_cast<float>(now - record.last_accessed) / 3600.0f;

        // Apply tier-differentiated decay
        uint8_t tier_idx = static_cast<uint8_t>(record.tier);
        if (tier_idx > 2) tier_idx = 0;  // safety
        float multiplier = kTierDecayMultiplier[tier_idx];

        if (config_.exponential_decay_enabled) {
            // Exponential decay: R = importance * e^(-rate * mult * t/24)
            // More accurate but slightly more expensive.
            float exponent = -config_.decay_rate * multiplier * age_hours / 24.0f;
            record.importance *= std::exp(exponent);
        } else {
            // Linear decay (default): importance *= (1 - rate * mult * t/24)
            record.importance *= (1.0f - config_.decay_rate * multiplier * age_hours / 24.0f);
        }
        if (record.importance < 0.0f) record.importance = 0.0f;

        // Mark stale if tier-specific threshold exceeded
        float stale_hours = kTierStaleHours[tier_idx];
        if (age_hours > stale_hours
            && record.confidence_level != Confidence::Verified) {
            record.confidence_level = Confidence::Stale;
        }

        // Demotion: if importance drops too low, demote one tier level.
        // LongTerm→ShortTerm or ShortTerm→Working. Working cannot demote further.
        if (record.importance < kDemotionThreshold && record.tier != MemoryTier::Working) {
            auto old_tier = record.tier;
            if (record.tier == MemoryTier::LongTerm) {
                record.tier = MemoryTier::ShortTerm;
            } else if (record.tier == MemoryTier::ShortTerm) {
                record.tier = MemoryTier::Working;
            }
            spdlog::debug("Decay demotion: memory {} tier {} → {} (importance={:.3f})",
                         id, memoryTierToString(old_tier), memoryTierToString(record.tier),
                         record.importance);
        }

        // Mark as dirty so changes persist on next flush
        dirty_ids_.insert(id);
    }
}

void MemoryStore::scanAll(std::function<void(const MemoryRecord&)> callback) const {
    std::shared_lock lock(mutex_);
    for (const auto& [id, record] : cache_) {
        callback(record);
    }
}

std::vector<MemoryRecord> MemoryStore::scanByContentPrefix(const std::string& prefix) const {
    std::shared_lock lock(mutex_);
    std::vector<MemoryRecord> results;
    for (const auto& [id, record] : cache_) {
        if (record.content.size() >= prefix.size() &&
            record.content.compare(0, prefix.size(), prefix) == 0) {
            results.push_back(record);
        }
    }
    return results;
}

void MemoryStore::updateInPlace(uint64_t memory_id, const std::string& new_content) {
    std::unique_lock lock(mutex_);
    auto it = cache_.find(memory_id);
    if (it != cache_.end()) {
        it->second.content = new_content;
        lsm_->putRaw(memory_id, it->second.serialize());
    }
}

size_t MemoryStore::totalMemories() const {
    std::shared_lock lock(mutex_);
    return cache_.size();
}

void MemoryStore::flushDirtyRecords() {
    // Write back all dirty cache entries to LSM. Must be called under exclusive lock.
    // Uses WAL batch mode to group all writes into a single fsync, reducing
    // disk I/O from O(N) fsyncs to O(1) for N dirty records.
    if (dirty_ids_.empty()) return;

    lsm_->beginBatch();
    size_t flushed = 0;
    for (auto it = dirty_ids_.begin(); it != dirty_ids_.end(); ) {
        auto cache_it = cache_.find(*it);
        if (cache_it != cache_.end()) {
            lsm_->putRaw(*it, cache_it->second.serialize());
            ++flushed;
        }
        it = dirty_ids_.erase(it);
    }
    lsm_->endBatch();
    if (flushed > 0) {
        spdlog::info("Flushed {} dirty cache entries to LSM (batch fsync)", flushed);
    }
}

void MemoryStore::flush() {
    // Write back dirty cache entries before flushing LSM engines
    {
        std::unique_lock lock(mutex_);
        flushDirtyRecords();
    }
    if (lsm_) lsm_->flush();
    if (hnsw_) {
        auto hnsw_path = config_.data_dir + "/hnsw/index.bin";
        hnsw_->save(hnsw_path);
    }
}

MemoryStore::ListResult MemoryStore::listMemories(const ListFilter& filter) const {
    std::shared_lock lock(mutex_);
    ListResult result;
    result.page = filter.page;
    result.per_page = filter.per_page;

    // Collect matching records
    std::vector<const MemoryRecord*> matched;
    matched.reserve(cache_.size());
    for (const auto& [id, rec] : cache_) {
        if (!rec.isAlive() && !filter.include_tombstone) continue;
        if (!filter.scope_filter.empty() && scopeToString(rec.scope) != filter.scope_filter) continue;
        if (!filter.memory_type_filter.empty() && memoryTypeToString(rec.memory_type) != filter.memory_type_filter) continue;
        if (!filter.phase_filter.empty() && phaseToString(rec.phase) != filter.phase_filter) continue;
        if (!filter.layer_filter.empty() && layerToString(rec.layer) != filter.layer_filter) continue;
        if (!filter.tier_filter.empty() && memoryTierToString(rec.tier) != filter.tier_filter) continue;
        if (!filter.query.empty() && rec.content.find(filter.query) == std::string::npos) continue;
        if (!filter.user_id_filter.empty() && rec.user_id != filter.user_id_filter) continue;
        matched.push_back(&rec);
    }

    std::sort(matched.begin(), matched.end(),
              [](const MemoryRecord* a, const MemoryRecord* b) {
                  return a->created_at > b->created_at;
              });

    result.total = matched.size();
    int offset = (filter.page - 1) * filter.per_page;
    for (int i = offset; i < static_cast<int>(matched.size()) && i < offset + filter.per_page; ++i) {
        result.records.push_back(*matched[static_cast<size_t>(i)]);
    }
    return result;
}

void MemoryStore::evictIfNeeded() {
    if (cache_.size() <= config_.max_cache_size) return;

    // Write back dirty entries before eviction to avoid data loss
    flushDirtyRecords();

    // Evict 20% of entries with lowest access_count
    size_t evict_count = cache_.size() / 5;
    std::vector<std::pair<uint32_t, uint64_t>> candidates;
    candidates.reserve(cache_.size());
    for (const auto& [id, rec] : cache_) {
        if (rec.phase == MemoryPhase::Tombstone) {
            candidates.push_back({0, id});
        } else {
            candidates.push_back({rec.access_count, id});
        }
    }
    std::partial_sort(candidates.begin(),
                      candidates.begin() + static_cast<ptrdiff_t>(evict_count),
                      candidates.end());
    for (size_t i = 0; i < evict_count; ++i) {
        cache_.erase(candidates[i].second);
    }
}

}  // namespace amind
