#pragma once

#include "skip_list.h"
#include "../core/memory_record.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace amind {

/// MemTable — an in-memory sorted buffer backed by a skip list.
///
/// Records are stored as serialized byte blobs, keyed by memory_id (uint64_t).
/// When the approximate memory usage exceeds the configured threshold,
/// the MemTable should be frozen and flushed to an SSTable on disk.
///
/// The MemTable also tracks tombstones (deletions) so that they can be
/// propagated to SSTables during flush and eventually removed during compaction.
class MemTable {
public:
    /// Default size threshold: 4 MB.
    static constexpr size_t DEFAULT_SIZE_THRESHOLD = 4 * 1024 * 1024;

    explicit MemTable(size_t size_threshold = DEFAULT_SIZE_THRESHOLD);
    ~MemTable() = default;

    MemTable(const MemTable&) = delete;
    MemTable& operator=(const MemTable&) = delete;
    MemTable(MemTable&&) = delete;
    MemTable& operator=(MemTable&&) = delete;

    /// Insert or update a MemoryRecord.
    /// The record is serialized and stored in the skip list keyed by memory_id.
    void put(const MemoryRecord& record);

    /// Insert a pre-serialized record by memory_id.
    void putRaw(uint64_t memory_id, std::vector<uint8_t> serialized_data);

    /// Look up a record by memory_id.
    /// Returns std::nullopt if not found or if the entry is a tombstone.
    [[nodiscard]] std::optional<MemoryRecord> get(uint64_t memory_id) const;

    /// Look up raw serialized bytes by memory_id.
    /// Returns std::nullopt if not found or if the entry is a tombstone.
    [[nodiscard]] std::optional<std::vector<uint8_t>> getRaw(uint64_t memory_id) const;

    /// Mark a memory_id as deleted (insert a tombstone).
    /// Returns true if the key existed.
    bool remove(uint64_t memory_id);

    /// Check if a memory_id exists (excluding tombstones).
    [[nodiscard]] bool contains(uint64_t memory_id) const;

    /// Number of entries (including tombstones).
    [[nodiscard]] size_t size() const;

    /// Whether the MemTable is empty.
    [[nodiscard]] bool empty() const;

    /// Approximate memory usage in bytes.
    [[nodiscard]] size_t approximateMemoryUsage() const;

    /// Whether the MemTable has reached its size threshold and should be flushed.
    [[nodiscard]] bool shouldFlush() const;

    /// Iterate over all entries in key order (including tombstones).
    /// The callback receives (memory_id, serialized_data, is_tombstone).
    /// Used during flush to SSTable.
    using EntryVisitor = std::function<void(uint64_t memory_id,
                                           const std::vector<uint8_t>& data,
                                           bool is_tombstone)>;
    void forEach(const EntryVisitor& visitor) const;

    /// Collect all memory_ids (for bloom filter construction).
    [[nodiscard]] std::vector<uint64_t> allKeys() const;

private:
    SkipList<uint64_t, std::vector<uint8_t>> skip_list_;
    size_t size_threshold_;
};

}  // namespace amind
