#pragma once

#include "memtable.h"
#include "sstable_reader.h"
#include "sstable_writer.h"
#include "lsm_wal.h"
#include "manifest.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <vector>

namespace amind {

/// LSMEngine — the top-level LSM-Tree engine that ties together MemTable,
/// SSTable flush, WAL (Write-Ahead Log), and L0→L1 compaction.
///
/// Write path:  put(record) → WAL append → active MemTable → (when full) flush → L0 SSTable → truncate WAL
/// Read path:   get(key) → active MemTable → immutable MemTable → L0 SSTables (newest first)
/// Recovery:    On startup, replay WAL into MemTable to recover unflushed writes.
///
/// Compaction:  When L0 has ≥ 4 SSTables, merge-sort them into a single L1 SSTable.
///
/// Lazy flush:  A background thread periodically flushes the MemTable to disk
///              (default every 30 seconds) to bound memory usage and reduce
///              data loss window.
class LSMEngine {
public:
    /// @param db_dir  Directory where SSTable and WAL files are stored.
    /// @param flushIntervalSeconds  How often the background thread flushes
    ///                              the MemTable (0 = disabled).
    explicit LSMEngine(const std::filesystem::path& db_dir,
                       uint32_t flushIntervalSeconds = 30);
    ~LSMEngine();

    LSMEngine(const LSMEngine&) = delete;
    LSMEngine& operator=(const LSMEngine&) = delete;

    /// Insert or update a record.
    void put(const MemoryRecord& record);

    /// Insert a pre-serialized record.
    void putRaw(uint64_t memory_id, std::vector<uint8_t> data);

    /// Look up a record by memory_id.
    /// Searches: active MemTable → immutable MemTable → L0 SSTables (newest first).
    [[nodiscard]] std::optional<std::vector<uint8_t>> get(uint64_t memory_id) const;

    /// Delete a record (insert tombstone).
    void remove(uint64_t memory_id);

    /// Force flush the active MemTable to an L0 SSTable and truncate the WAL.
    void flush();

    /// Begin a batch write. Subsequent put/putRaw calls will not fsync the WAL
    /// individually; a single fsync is performed when endBatch() is called.
    /// This reduces fsync overhead when writing many records at once (e.g.
    /// flushing dirty cache entries). Must be paired with endBatch().
    /// NOT reentrant — caller must not nest beginBatch() calls.
    void beginBatch();

    /// End a batch write and fsync the WAL once for all accumulated writes.
    void endBatch();

    /// Run L0 → L1 compaction if the threshold is reached.
    void maybeCompact();

    /// Force compaction regardless of threshold.
    void forceCompact();

    /// Dynamically update the flush interval (seconds). Wakes the flush thread.
    void setFlushInterval(uint32_t seconds) {
        flushIntervalSeconds_ = seconds;
        flushCv_.notify_one();
    }



    /// Number of L0 SSTables.
    [[nodiscard]] size_t l0TableCount() const;

    /// Number of L1 SSTables.
    [[nodiscard]] size_t l1TableCount() const;

    /// Number of L2 SSTables.
    [[nodiscard]] size_t l2TableCount() const;

    /// Total number of entries across all MemTables and SSTables.
    [[nodiscard]] size_t approximateTotalEntries() const;

    /// Iterate over all live entries (excluding tombstones), deduplicating by key.
    using LiveEntryVisitor = std::function<void(uint64_t memory_id,
                                                const std::vector<uint8_t>& data)>;
    void forEachLive(const LiveEntryVisitor& visitor) const;

    /// Get the current sequence number (monotonically increasing).
    [[nodiscard]] uint64_t currentSeq() const { return sequence_number_.load(); }

private:
    std::filesystem::path db_dir_;

    mutable std::shared_mutex mutex_;
    std::unique_ptr<MemTable> active_memtable_;
    std::unique_ptr<MemTable> immutable_memtable_;  // Being flushed

    /// L0 SSTables, ordered from newest to oldest.
    std::vector<std::unique_ptr<SSTableReader>> l0_tables_;

    /// L1 SSTables, ordered from newest to oldest.
    std::vector<std::unique_ptr<SSTableReader>> l1_tables_;

    /// L2 SSTables, ordered from newest to oldest.
    std::vector<std::unique_ptr<SSTableReader>> l2_tables_;

    std::atomic<uint64_t> next_sst_id_{0};

    /// Write-Ahead Log for crash recovery.
    std::unique_ptr<LsmWriteAheadLog> wal_;

    /// Manifest for SSTable metadata and checkpoint sequence.
    std::unique_ptr<Manifest> manifest_;

    /// Global monotonically increasing sequence number for WAL entries.
    std::atomic<uint64_t> sequence_number_{0};

    // ── Lazy flush background thread ──────────────────────────────────────
    uint32_t flushIntervalSeconds_;
    std::thread flushThread_;
    std::condition_variable flushCv_;
    std::mutex flushMutex_;
    std::atomic<bool> stopFlushThread_{false};

    void flushThreadLoop();

    /// Get the next sequence number (atomically incremented).
    uint64_t nextSeq() { return sequence_number_.fetch_add(1) + 1; }

    /// Generate a unique SSTable file path.
    [[nodiscard]] std::filesystem::path generateSSTPath();

    /// Flush the given MemTable to an L0 SSTable file.
    void flushMemTable(const MemTable& memtable);

    /// Flush active memtable and run compaction if needed (caller must hold mutex_).
    /// Handles exception safety (restores immutable_memtable_ on flush failure)
    /// and ensures manifest is saved before WAL is truncated.
    void flushAndCompact();

    /// Compact all L0 SSTables into L1 SSTables.
    void compactL0();

    /// Compact all SSTables at a given level to the next level.
    /// @param level 0 = L0→L1, 1 = L1→L2
    void compactLevel(int level);

    /// Internal flush (caller must hold mutex_).
    void flushLocked();
};

}  // namespace amind