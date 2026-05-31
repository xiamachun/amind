#pragma once

#include "block.h"
#include "bloom_filter.h"
#include "sstable_format.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace amind {

class SSTableReader;

/// SSTableCursor — sequential iterator over an SSTable's entries in key order.
/// Holds one decompressed block at a time, advancing lazily.
class SSTableCursor {
public:
    struct Entry {
        uint64_t key;
        std::vector<uint8_t> value;
    };

    SSTableCursor() = default;
    ~SSTableCursor();
    SSTableCursor(SSTableCursor&&) noexcept;
    SSTableCursor& operator=(SSTableCursor&&) noexcept;

    bool valid() const { return valid_; }
    const Entry& current() const { return current_; }
    void next();

private:
    friend class SSTableReader;
    SSTableCursor(const SSTableReader* reader, size_t block_idx, size_t entry_idx);

    void loadCurrentEntry();

    const SSTableReader* reader_{nullptr};
    size_t block_idx_{0};
    size_t entry_idx_{0};
    bool valid_{false};
    Entry current_;
    std::unique_ptr<BlockReader> current_block_;
};

/// SSTableReader — reads an SSTable file and supports point lookups and range scans.
///
/// On open, the reader uses mmap to map the file into memory.
/// The Footer, Index Block, and Bloom Filter are parsed from the mapped memory.
/// Data Blocks are accessed directly from the mapped memory during lookups.
///
/// Usage:
///   SSTableReader reader("/path/to/table.sst");
///   auto value = reader.get(memory_id);
///   reader.scan(min_key, max_key, [](uint64_t key, span<const uint8_t> val) { ... });
class SSTableReader {
public:
    /// Open an SSTable file for reading.
    explicit SSTableReader(const std::filesystem::path& file_path);

    ~SSTableReader();

    SSTableReader(const SSTableReader&) = delete;
    SSTableReader& operator=(const SSTableReader&) = delete;
    SSTableReader(SSTableReader&&) = delete;
    SSTableReader& operator=(SSTableReader&&) = delete;

    /// Point lookup: find the value for a given key.
    /// Returns std::nullopt if the key is not found.
    /// Uses the Bloom Filter to skip unnecessary block reads.
    [[nodiscard]] std::optional<std::vector<uint8_t>> get(uint64_t key) const;

    /// Range scan: iterate over all entries with keys in [min_key, max_key].
    /// The callback receives (key, value_bytes) for each matching entry.
    using ScanCallback = std::function<void(uint64_t key, std::span<const uint8_t> value)>;
    void scan(uint64_t min_key, uint64_t max_key, const ScanCallback& callback) const;

    /// Iterate over all entries in key order.
    void forEach(const ScanCallback& callback) const;

    /// Total number of entries in the SSTable.
    [[nodiscard]] uint32_t entryCount() const { return footer_.entry_count; }

    /// The file path of this SSTable.
    [[nodiscard]] const std::filesystem::path& filePath() const { return file_path_; }

    /// Check if a key might exist (Bloom Filter check only, no disk I/O).
    [[nodiscard]] bool mayContain(uint64_t key) const;

    /// Get the smallest key in the SSTable (first key of first index entry).
    [[nodiscard]] uint64_t minKey() const;

    /// Get the largest key in the SSTable (requires reading the last block).
    [[nodiscard]] uint64_t maxKey() const;

    /// Create a cursor for sequential iteration over all entries.
    [[nodiscard]] SSTableCursor cursor() const;

private:
    friend class SSTableCursor;
    std::filesystem::path file_path_;
    
    // mmap related members
    const uint8_t* mapped_data_;  // mmap mapped data pointer
    size_t mapped_size_;          // mapped size
    int fd_;                      // file descriptor
    
    // Fallback: traditional file read if mmap fails
    std::vector<uint8_t> file_data_;
    bool using_mmap_;             // true if mmap is being used

    SSTableFooter footer_;
    std::vector<IndexEntry> index_entries_;
    BloomFilter bloom_filter_;

    /// Load and parse the file.
    void load();

    /// Find the index of the Data Block that might contain the given key.
    /// Returns -1 if the key is outside the range of all blocks.
    [[nodiscard]] int findBlockIndex(uint64_t key) const;

    /// Read and decompress a Data Block by its index entry.
    [[nodiscard]] BlockReader readBlock(const IndexEntry& entry) const;
};

}  // namespace amind
