#include "lsm_engine.h"

#include <algorithm>
#include <map>
#include <queue>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace amind {

LSMEngine::LSMEngine(const std::filesystem::path& db_dir,
                     uint32_t flushIntervalSeconds)
    : db_dir_(db_dir)
    , active_memtable_(std::make_unique<MemTable>())
    , flushIntervalSeconds_(flushIntervalSeconds) {
    std::filesystem::create_directories(db_dir_);

    // ── 1. Create and load Manifest ─────────────────────────────────────
    manifest_ = std::make_unique<Manifest>(db_dir_);
    bool hasManifest = manifest_->load();

    if (hasManifest) {
        // Load SSTables from Manifest (no directory scan)
        next_sst_id_.store(manifest_->nextSstId());
        
        // Load L0 files
        for (const auto& f : manifest_->l0Files()) {
            auto path = db_dir_ / f;
            try {
                l0_tables_.push_back(std::make_unique<SSTableReader>(path));
            } catch (const std::exception& ex) {
                spdlog::warn("Failed to load L0 SSTable {}: {}", path.string(), ex.what());
            }
        }
        
        // Load L1 files
        for (const auto& f : manifest_->l1Files()) {
            auto path = db_dir_ / f;
            try {
                l1_tables_.push_back(std::make_unique<SSTableReader>(path));
            } catch (const std::exception& ex) {
                spdlog::warn("Failed to load L1 SSTable {}: {}", path.string(), ex.what());
            }
        }
        
        // Load L2 files
        for (const auto& f : manifest_->l2Files()) {
            auto path = db_dir_ / f;
            try {
                l2_tables_.push_back(std::make_unique<SSTableReader>(path));
            } catch (const std::exception& ex) {
                spdlog::warn("Failed to load L2 SSTable {}: {}", path.string(), ex.what());
            }
        }
    } else {
        // First startup or migration from old version: scan directory
        std::vector<std::filesystem::path> l0_files, l1_files, l2_files;
        if (std::filesystem::exists(db_dir_)) {
            for (const auto& entry : std::filesystem::directory_iterator(db_dir_)) {
                if (entry.path().extension() == ".sst") {
                    auto stem = entry.path().stem().string();
                    if (stem.starts_with("L1_")) {
                        l1_files.push_back(entry.path());
                    } else if (stem.starts_with("L2_")) {
                        l2_files.push_back(entry.path());
                    } else {
                        l0_files.push_back(entry.path());
                    }
                }
            }
        }

        // Sort by filename descending (newest first)
        std::sort(l0_files.begin(), l0_files.end(), std::greater<>());
        std::sort(l1_files.begin(), l1_files.end(), std::greater<>());
        std::sort(l2_files.begin(), l2_files.end(), std::greater<>());

        // Load L0 files
        for (const auto& path : l0_files) {
            try {
                l0_tables_.push_back(std::make_unique<SSTableReader>(path));
                auto stem = path.stem().string();
                if (stem.size() > 4) {  // "sst_XXXX"
                    uint64_t file_id = std::stoull(stem.substr(4));
                    if (file_id >= next_sst_id_.load()) {
                        next_sst_id_.store(file_id + 1);
                    }
                }
            } catch (const std::exception& ex) {
                spdlog::warn("Failed to load SSTable {}: {}", path.string(), ex.what());
            }
        }

        // Load L1 files
        for (const auto& path : l1_files) {
            try {
                l1_tables_.push_back(std::make_unique<SSTableReader>(path));
            } catch (const std::exception& ex) {
                spdlog::warn("Failed to load SSTable {}: {}", path.string(), ex.what());
            }
        }

        // Load L2 files
        for (const auto& path : l2_files) {
            try {
                l2_tables_.push_back(std::make_unique<SSTableReader>(path));
            } catch (const std::exception& ex) {
                spdlog::warn("Failed to load SSTable {}: {}", path.string(), ex.what());
            }
        }

        // Create initial Manifest from scanned files
        std::vector<std::string> l0Names, l1Names, l2Names;
        for (auto& t : l0_tables_) {
            l0Names.push_back(t->filePath().filename().string());
        }
        for (auto& t : l1_tables_) {
            l1Names.push_back(t->filePath().filename().string());
        }
        for (auto& t : l2_tables_) {
            l2Names.push_back(t->filePath().filename().string());
        }
        manifest_->setL0Files(l0Names);
        manifest_->setL1Files(l1Names);
        manifest_->setL2Files(l2Names);
        manifest_->setNextSstId(next_sst_id_.load());
        manifest_->save();
    }

    spdlog::info("LSMEngine opened at {}: {} L0, {} L1, {} L2 SSTables loaded",
                 db_dir_.string(), l0_tables_.size(), l1_tables_.size(), l2_tables_.size());

    // ── 2. WAL incremental replay (only replay seq > checkpoint_seq) ─────
    auto walPath = db_dir_ / "wal.log";
    uint64_t checkpointSeq = manifest_->checkpointSeq();
    uint64_t maxSeq = LsmWriteAheadLog::replay(walPath,
        [this](uint64_t seq, uint64_t key, std::vector<uint8_t> data, bool isTombstone) {
            if (isTombstone) {
                active_memtable_->putRaw(key, {});
            } else {
                active_memtable_->putRaw(key, std::move(data));
            }
        },
        checkpointSeq);  // Incremental replay!

    // ── 3. Set sequence number to max(checkpoint_seq, maxReplayedSeq) ─────
    sequence_number_.store(std::max(checkpointSeq, maxSeq));

    // ── 4. Open WAL ────────────────────────────────────────────────────────
    wal_ = std::make_unique<LsmWriteAheadLog>(walPath);

    // ── 5. If WAL replayed data, flush to SSTable ─────────────────────────
    if (!active_memtable_->empty()) {
        spdlog::info("WAL recovery: flushing {} recovered entries to SSTable",
                     active_memtable_->size());
        flushMemTable(*active_memtable_);
        active_memtable_ = std::make_unique<MemTable>();
        wal_->truncate();
        // Update checkpoint_seq and save Manifest
        manifest_->setCheckpointSeq(sequence_number_.load());
        manifest_->save();
        if (l0_tables_.size() >= SSTableFormat::L0_COMPACTION_THRESHOLD) {
            compactL0();
        }
    }

    // ── 6. Start lazy-flush background thread ─────────────────────────────
    if (flushIntervalSeconds_ > 0) {
        flushThread_ = std::thread(&LSMEngine::flushThreadLoop, this);
        spdlog::info("Lazy flush thread started (interval={}s)", flushIntervalSeconds_);
    }
}

LSMEngine::~LSMEngine() {
    // Stop the background flush thread
    stopFlushThread_.store(true);
    flushCv_.notify_all();
    if (flushThread_.joinable()) {
        flushThread_.join();
    }

    // Final flush: write any remaining MemTable data to SSTable.
    // Catch exceptions to prevent terminate() — destructors must not throw.
    try {
        std::lock_guard lock(mutex_);
        flushLocked();
    } catch (const std::exception& ex) {
        spdlog::error("Failed to flush LSMEngine on shutdown: {}", ex.what());
    }
}

std::filesystem::path LSMEngine::generateSSTPath() {
    uint64_t id;
    if (manifest_) {
        id = manifest_->allocateSstId();
    } else {
        id = next_sst_id_.fetch_add(1);
    }
    char filename[32];
    std::snprintf(filename, sizeof(filename), "sst_%06llu.sst",
                  static_cast<unsigned long long>(id));
    return db_dir_ / filename;
}

// ─── Write Path ────────────────────────────────────────────────────────────────

void LSMEngine::put(const MemoryRecord& record) {
    std::lock_guard lock(mutex_);
    uint64_t seq = nextSeq();
    auto serialized = record.serialize();
    wal_->appendPut(seq, record.memory_id, serialized);
    active_memtable_->put(record);
    if (active_memtable_->shouldFlush()) {
        flushAndCompact();
    }
}

void LSMEngine::putRaw(uint64_t memory_id, std::vector<uint8_t> data) {
    std::lock_guard lock(mutex_);
    uint64_t seq = nextSeq();
    wal_->appendPut(seq, memory_id, data);
    active_memtable_->putRaw(memory_id, std::move(data));
    if (active_memtable_->shouldFlush()) {
        flushAndCompact();
    }
}

void LSMEngine::beginBatch() {
    if (wal_) wal_->beginBatch();
}

void LSMEngine::endBatch() {
    if (wal_) wal_->endBatch();
}

void LSMEngine::flushAndCompact() {
    // Move active memtable to immutable.
    immutable_memtable_ = std::move(active_memtable_);
    active_memtable_ = std::make_unique<MemTable>();

    try {
        flushMemTable(*immutable_memtable_);
    } catch (...) {
        // Restore the immutable memtable so data is not lost.
        active_memtable_ = std::move(immutable_memtable_);
        throw;
    }

    immutable_memtable_.reset();

    // Save manifest BEFORE truncating WAL so that a crash between the two
    // operations leaves the WAL intact (recoverable), not the reverse.
    manifest_->setCheckpointSeq(sequence_number_.load());
    manifest_->save();
    wal_->truncate();

    if (l0_tables_.size() >= SSTableFormat::L0_COMPACTION_THRESHOLD) {
        compactL0();
    }
}

void LSMEngine::remove(uint64_t memory_id) {
    std::lock_guard lock(mutex_);
    uint64_t seq = nextSeq();
    wal_->appendDelete(seq, memory_id);
    active_memtable_->remove(memory_id);
}

// ─── Read Path ─────────────────────────────────────────────────────────────────

std::optional<std::vector<uint8_t>> LSMEngine::get(uint64_t memory_id) const {
    std::shared_lock lock(mutex_);

    // 1. Search active MemTable
    auto result = active_memtable_->getRaw(memory_id);
    if (result.has_value()) {
        // Empty value means tombstone (deleted)
        if (result->empty()) {
            return std::nullopt;
        }
        return result;
    }

    // 2. Search immutable MemTable (if being flushed)
    if (immutable_memtable_) {
        result = immutable_memtable_->getRaw(memory_id);
        if (result.has_value()) {
            // Empty value means tombstone (deleted)
            if (result->empty()) {
                return std::nullopt;
            }
            return result;
        }
    }

    // 3. Search L0 SSTables (newest first)
    for (const auto& table : l0_tables_) {
        // Bloom filter check first
        if (!table->mayContain(memory_id)) {
            continue;
        }
        result = table->get(memory_id);
        if (result.has_value()) {
            // Empty value means tombstone (deleted)
            if (result->empty()) {
                return std::nullopt;
            }
            return result;
        }
    }

    // 4. Search L1 SSTables (newest first)
    for (const auto& table : l1_tables_) {
        // Bloom filter check first
        if (!table->mayContain(memory_id)) {
            continue;
        }
        result = table->get(memory_id);
        if (result.has_value()) {
            // Empty value means tombstone (deleted)
            if (result->empty()) {
                return std::nullopt;
            }
            return result;
        }
    }

    // 5. Search L2 SSTables (newest first)
    for (const auto& table : l2_tables_) {
        // Bloom filter check first
        if (!table->mayContain(memory_id)) {
            continue;
        }
        result = table->get(memory_id);
        if (result.has_value()) {
            // Empty value means tombstone (deleted)
            if (result->empty()) {
                return std::nullopt;
            }
            return result;
        }
    }

    return std::nullopt;
}

// ─── Flush ─────────────────────────────────────────────────────────────────────

void LSMEngine::flush() {
    std::lock_guard lock(mutex_);
    flushLocked();
}

void LSMEngine::flushLocked() {
    if (active_memtable_->empty()) {
        return;
    }

    flushMemTable(*active_memtable_);
    active_memtable_ = std::make_unique<MemTable>();

    // All MemTable data is now in SSTables — truncate the WAL
    if (wal_) {
        wal_->truncate();
    }

    // Update checkpoint_seq and save Manifest
    if (manifest_) {
        manifest_->setCheckpointSeq(sequence_number_.load());
        manifest_->save();
    }
}

// ─── Lazy Flush Background Thread ──────────────────────────────────────────────

void LSMEngine::flushThreadLoop() {
    while (!stopFlushThread_.load()) {
        std::unique_lock lock(flushMutex_);
        flushCv_.wait_for(lock, std::chrono::seconds(flushIntervalSeconds_),
                          [this] { return stopFlushThread_.load(); });

        if (stopFlushThread_.load()) {
            break;
        }

        // Periodic flush
        try {
            std::lock_guard dataLock(mutex_);
            if (!active_memtable_->empty()) {
                spdlog::info("Lazy flush: flushing {} entries to SSTable",
                             active_memtable_->size());
                flushLocked();

                if (l0_tables_.size() >= SSTableFormat::L0_COMPACTION_THRESHOLD) {
                    compactL0();
                }
            }
        } catch (const std::exception& e) {
            spdlog::error("Lazy flush failed: {}", e.what());
        }
    }
}

void LSMEngine::flushMemTable(const MemTable& memtable) {
    auto sst_path = generateSSTPath();
    size_t entry_count = memtable.size();

    SSTableWriter writer(sst_path, entry_count);

    memtable.forEach([&writer](uint64_t memory_id,
                               const std::vector<uint8_t>& data,
                               bool is_tombstone) {
        // Write all entries including tombstones to SSTable.
        // Tombstones are represented as empty values.
        if (is_tombstone) {
            writer.add(memory_id, {});
        } else {
            writer.add(memory_id, data);
        }
    });

    size_t file_size = writer.finish();
    spdlog::info("Flushed MemTable to {}: {} entries, {} bytes",
                 sst_path.string(), entry_count, file_size);

    // Load the new SSTable and insert at the front (newest first)
    auto reader = std::make_unique<SSTableReader>(sst_path);
    l0_tables_.insert(l0_tables_.begin(), std::move(reader));

    // Update Manifest with new L0 file
    if (manifest_) {
        manifest_->addL0File(sst_path.filename().string());
    }
}

// ─── Compaction ────────────────────────────────────────────────────────────────

void LSMEngine::maybeCompact() {
    std::lock_guard lock(mutex_);
    if (l0_tables_.size() >= SSTableFormat::L0_COMPACTION_THRESHOLD) {
        compactL0();
    }
}

void LSMEngine::forceCompact() {
    std::lock_guard lock(mutex_);
    if (l0_tables_.size() >= 2) {
        compactL0();
    }
}

void LSMEngine::compactL0() {
    compactLevel(0);
}

void LSMEngine::compactLevel(int level) {
    auto& source_tables = (level == 0) ? l0_tables_ : l1_tables_;
    const char* src_label = (level == 0) ? "L0" : "L1";
    const char* dst_label = (level == 0) ? "L1" : "L2";

    if (source_tables.size() < 2) {
        return;
    }

    spdlog::info("Starting {}→{} compaction: merging {} SSTables",
                 src_label, dst_label, source_tables.size());

    // Streaming k-way merge using a min-heap of SSTable cursors.
    // For L0, newer tables (lower index) have priority on duplicate keys.
    struct HeapEntry {
        uint64_t key;
        size_t table_index;
        SSTableCursor* cursor;

        bool operator>(const HeapEntry& other) const {
            if (key != other.key) return key > other.key;
            return table_index > other.table_index;
        }
    };

    std::vector<SSTableCursor> cursors;
    cursors.reserve(source_tables.size());

    for (size_t i = 0; i < source_tables.size(); ++i) {
        try {
            cursors.push_back(source_tables[i]->cursor());
        } catch (const std::exception& ex) {
            spdlog::error("Corrupted {} SSTable {}: {} — skipping",
                          src_label, source_tables[i]->filePath().string(), ex.what());
            cursors.emplace_back();
        }
    }

    std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<>> heap;
    for (size_t i = 0; i < cursors.size(); ++i) {
        if (cursors[i].valid()) {
            heap.push({cursors[i].current().key, i, &cursors[i]});
        }
    }

    // Generate output filename
    uint64_t out_id = next_sst_id_.fetch_add(1);
    std::string prefix = (level == 0) ? "L1_" : "L2_";
    auto output_path = db_dir_ / (prefix + std::to_string(out_id) + ".sst");

    size_t estimated_entries = 0;
    for (const auto& t : source_tables) {
        estimated_entries += t->entryCount();
    }
    SSTableWriter writer(output_path, estimated_entries);

    uint64_t prev_key = 0;
    bool first = true;

    while (!heap.empty()) {
        auto top = heap.top();
        heap.pop();

        bool duplicate = (!first && top.key == prev_key);

        if (!duplicate) {
            const auto& val = top.cursor->current().value;
            if (!val.empty()) {
                writer.add(top.key, val);
            }
        }

        prev_key = top.key;
        first = false;

        top.cursor->next();
        if (top.cursor->valid()) {
            heap.push({top.cursor->current().key, top.table_index, top.cursor});
        }
    }

    size_t file_size = writer.finish();

    // Collect old file names and paths
    std::vector<std::string> oldNames;
    std::vector<std::filesystem::path> oldPaths;
    for (const auto& table : source_tables) {
        oldNames.push_back(table->filePath().filename().string());
        oldPaths.push_back(table->filePath());
    }

    // Update Manifest before deleting files
    if (manifest_) {
        manifest_->setNextSstId(next_sst_id_.load());
        if (level == 0) {
            manifest_->compactL0ToL1(oldNames, output_path.filename().string());
        } else {
            manifest_->compactL1ToL2(oldNames, output_path.filename().string());
        }
        manifest_->save();
    }

    source_tables.clear();

    for (const auto& path : oldPaths) {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }

    auto& dest_tables = (level == 0) ? l1_tables_ : l2_tables_;
    dest_tables.push_back(std::make_unique<SSTableReader>(output_path));

    spdlog::info("{}→{} compaction complete: merged into {} ({} bytes)",
                 src_label, dst_label, output_path.string(), file_size);

    // Cascade: check if destination level needs compaction
    if (level == 0 && l1_tables_.size() >= SSTableFormat::L1_COMPACTION_THRESHOLD) {
        compactLevel(1);
    }
}

size_t LSMEngine::l0TableCount() const {
    std::shared_lock lock(mutex_);
    return l0_tables_.size();
}

size_t LSMEngine::l1TableCount() const {
    std::shared_lock lock(mutex_);
    return l1_tables_.size();
}

size_t LSMEngine::l2TableCount() const {
    std::shared_lock lock(mutex_);
    return l2_tables_.size();
}

size_t LSMEngine::approximateTotalEntries() const {
    std::shared_lock lock(mutex_);
    size_t total = active_memtable_->size();
    if (immutable_memtable_) {
        total += immutable_memtable_->size();
    }
    for (const auto& table : l0_tables_) {
        total += table->entryCount();
    }
    return total;
}

void LSMEngine::forEachLive(const LiveEntryVisitor& visitor) const {
    std::shared_lock lock(mutex_);

    // Pass 1: Determine which keys are live using only key + tombstone status.
    // This avoids buffering all value data in memory.
    // For duplicate keys, the newest source wins (memtable > L0 > L1 > L2).
    std::unordered_map<uint64_t, bool> key_state;  // key → isTombstone

    // L2 SSTables (oldest layer)
    for (int i = static_cast<int>(l2_tables_.size()) - 1; i >= 0; --i) {
        try {
            l2_tables_[i]->forEach(
                [&key_state](uint64_t key, std::span<const uint8_t> value) {
                    key_state[key] = value.empty();
                });
        } catch (const std::exception& ex) {
            spdlog::error("Corrupted L2 SSTable {}: {} — skipping",
                          l2_tables_[i]->filePath().string(), ex.what());
        }
    }

    // L1 SSTables
    for (int i = static_cast<int>(l1_tables_.size()) - 1; i >= 0; --i) {
        try {
            l1_tables_[i]->forEach(
                [&key_state](uint64_t key, std::span<const uint8_t> value) {
                    key_state[key] = value.empty();
                });
        } catch (const std::exception& ex) {
            spdlog::error("Corrupted L1 SSTable {}: {} — skipping",
                          l1_tables_[i]->filePath().string(), ex.what());
        }
    }

    // L0 SSTables
    for (int i = static_cast<int>(l0_tables_.size()) - 1; i >= 0; --i) {
        try {
            l0_tables_[i]->forEach(
                [&key_state](uint64_t key, std::span<const uint8_t> value) {
                    key_state[key] = value.empty();
                });
        } catch (const std::exception& ex) {
            spdlog::error("Corrupted L0 SSTable {}: {} — skipping",
                          l0_tables_[i]->filePath().string(), ex.what());
        }
    }

    // Immutable memtable
    if (immutable_memtable_) {
        immutable_memtable_->forEach(
            [&key_state](uint64_t key, const std::vector<uint8_t>&, bool isTombstone) {
                key_state[key] = isTombstone;
            });
    }

    // Active memtable (newest, final authority)
    active_memtable_->forEach(
        [&key_state](uint64_t key, const std::vector<uint8_t>&, bool isTombstone) {
            key_state[key] = isTombstone;
        });

    // Pass 2: For each live key, fetch its data via get() and emit.
    // This reads one record at a time instead of buffering all data.
    for (const auto& [key, is_tombstone] : key_state) {
        if (is_tombstone) continue;

        // Check memtables first (fast path)
        auto mem_data = active_memtable_->getRaw(key);
        if (mem_data.has_value() && !mem_data->empty()) {
            visitor(key, *mem_data);
            continue;
        }
        if (immutable_memtable_) {
            mem_data = immutable_memtable_->getRaw(key);
            if (mem_data.has_value() && !mem_data->empty()) {
                visitor(key, *mem_data);
                continue;
            }
        }

        // Check SSTables (newest first)
        bool found = false;
        for (const auto& table : l0_tables_) {
            auto val = table->get(key);
            if (val.has_value()) {
                if (!val->empty()) visitor(key, *val);
                found = true;
                break;
            }
        }
        if (found) continue;
        for (const auto& table : l1_tables_) {
            auto val = table->get(key);
            if (val.has_value()) {
                if (!val->empty()) visitor(key, *val);
                found = true;
                break;
            }
        }
        if (found) continue;
        for (const auto& table : l2_tables_) {
            auto val = table->get(key);
            if (val.has_value()) {
                if (!val->empty()) visitor(key, *val);
                break;
            }
        }
    }
}

}  // namespace amind
