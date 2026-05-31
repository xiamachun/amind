#pragma once

#include "sstable_format.h"

#include <cstdint>
#include <span>
#include <vector>

namespace amind {

/// BlockBuilder — accumulates key-value entries and produces a compressed Data Block.
///
/// Entries are appended in sorted key order. When the uncompressed size exceeds
/// the target block size (4 KB), the caller should call finish() to produce the
/// compressed block and start a new BlockBuilder.
///
/// Uncompressed block layout:
///   [entry_count: u32]
///   [entry 0: key(u64) + value_len(u32) + value(bytes)]
///   [entry 1: ...]
///   ...
///   [entry N: ...]
class BlockBuilder {
public:
    explicit BlockBuilder(size_t target_size = SSTableFormat::DATA_BLOCK_SIZE);

    /// Append a key-value entry. Keys must be added in ascending order.
    void add(uint64_t key, std::span<const uint8_t> value);

    /// Finalize the block: serialize entries and compress with LZ4.
    /// Returns the compressed bytes. Resets the builder for reuse.
    [[nodiscard]] std::vector<uint8_t> finish();

    /// Current uncompressed size estimate.
    [[nodiscard]] size_t currentSize() const { return uncompressed_size_; }

    /// Whether the block has reached its target size.
    [[nodiscard]] bool isFull() const { return uncompressed_size_ >= target_size_; }

    /// Whether the block has any entries.
    [[nodiscard]] bool empty() const { return entries_.empty(); }

    /// Number of entries in the current block.
    [[nodiscard]] size_t entryCount() const { return entries_.size(); }

    /// The first key in the block (undefined if empty).
    [[nodiscard]] uint64_t firstKey() const { return first_key_; }

    /// Reset the builder for a new block.
    void reset();

private:
    struct Entry {
        uint64_t key;
        std::vector<uint8_t> value;
    };

    std::vector<Entry> entries_;
    size_t target_size_;
    size_t uncompressed_size_;
    uint64_t first_key_;
};

/// BlockReader — reads entries from a compressed Data Block.
///
/// Decompresses the LZ4 block and provides iteration over key-value entries.
class BlockReader {
public:
    /// Construct a reader from compressed block data.
    /// @param compressed_data  The LZ4-compressed block bytes.
    explicit BlockReader(std::span<const uint8_t> compressed_data);

    /// A single decoded entry from the block.
    struct Entry {
        uint64_t key;
        std::span<const uint8_t> value;
    };

    /// Number of entries in the block.
    [[nodiscard]] size_t entryCount() const { return entry_count_; }

    /// Get the entry at the given index.
    [[nodiscard]] Entry entryAt(size_t index) const;

    /// Binary search for a key. Returns the entry index, or -1 if not found.
    [[nodiscard]] int findKey(uint64_t key) const;

    /// Get the first key in the block.
    [[nodiscard]] uint64_t firstKey() const;

    /// Get the last key in the block.
    [[nodiscard]] uint64_t lastKey() const;

private:
    std::vector<uint8_t> decompressed_;
    size_t entry_count_{0};

    /// Offsets into decompressed_ where each entry starts (after the entry_count header).
    std::vector<size_t> entry_offsets_;

    void parseEntries();
};

}  // namespace amind
