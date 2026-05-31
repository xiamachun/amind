#pragma once

#include "block.h"
#include "bloom_filter.h"
#include "sstable_format.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace amind {

/// SSTableWriter — writes a sorted sequence of key-value entries to an SSTable file.
///
/// Usage:
///   SSTableWriter writer("/path/to/output.sst");
///   writer.add(key1, value1);
///   writer.add(key2, value2);  // keys must be in ascending order
///   ...
///   writer.finish();           // flushes remaining data and writes index + bloom + footer
///
/// The writer accumulates entries into Data Blocks (LZ4 compressed, ~4 KB each),
/// builds an Index Block and Bloom Filter, and writes the Footer at the end.
class SSTableWriter {
public:
    /// Construct a writer that will create an SSTable at the given path.
    /// @param file_path  Output file path.
    /// @param expected_entries  Estimated number of entries (for bloom filter sizing).
    explicit SSTableWriter(const std::filesystem::path& file_path,
                           size_t expected_entries = 1024);

    ~SSTableWriter();

    SSTableWriter(const SSTableWriter&) = delete;
    SSTableWriter& operator=(const SSTableWriter&) = delete;
    SSTableWriter(SSTableWriter&&) = delete;
    SSTableWriter& operator=(SSTableWriter&&) = delete;

    /// Add a key-value entry. Keys must be added in strictly ascending order.
    /// @param key    The memory_id.
    /// @param value  Serialized MemoryRecord bytes.
    void add(uint64_t key, std::span<const uint8_t> value);

    /// Finalize the SSTable: flush the last data block, write index block,
    /// bloom filter block, and footer. Closes the file.
    /// Returns the total file size in bytes.
    size_t finish();

    /// Number of entries written so far.
    [[nodiscard]] size_t entryCount() const { return entry_count_; }

    /// Whether finish() has been called.
    [[nodiscard]] bool isFinished() const { return finished_; }

private:
    std::filesystem::path file_path_;
    std::vector<uint8_t> file_buffer_;  // Accumulate all output bytes

    BlockBuilder block_builder_;
    BloomFilter bloom_filter_;
    std::vector<IndexEntry> index_entries_;

    size_t entry_count_{0};
    uint64_t last_key_{0};
    bool finished_{false};

    /// Flush the current data block to the file buffer.
    void flushDataBlock();

    /// Write the file buffer to disk.
    void writeFile();
};

}  // namespace amind
