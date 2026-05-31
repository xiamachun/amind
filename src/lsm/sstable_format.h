#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace amind {

/// SSTable file format constants.
///
/// On-disk layout:
///   ┌──────────────────────────────────────────┐
///   │  Data Block 0  (LZ4 compressed)          │
///   │  Data Block 1  (LZ4 compressed)          │
///   │  ...                                     │
///   │  Data Block N  (LZ4 compressed)          │
///   ├──────────────────────────────────────────┤
///   │  Index Block (block offsets + first keys) │
///   ├──────────────────────────────────────────┤
///   │  Bloom Filter Block                       │
///   ├──────────────────────────────────────────┤
///   │  Footer (40 bytes)                        │
///   └──────────────────────────────────────────┘
struct SSTableFormat {
    static constexpr uint64_t MAGIC = 0x436C61774D696E64ULL;  // "ClawMind" in ASCII
    static constexpr uint16_t VERSION = 1;

    /// Target size for uncompressed data blocks before LZ4 compression.
    static constexpr size_t DATA_BLOCK_SIZE = 4 * 1024;  // 4 KB

    /// Bloom filter: bits per key. 10 bits/key ≈ 1% false positive rate.
    static constexpr uint32_t BLOOM_BITS_PER_KEY = 10;

    /// Number of L0 SSTables that triggers compaction into L1.
    static constexpr size_t L0_COMPACTION_THRESHOLD = 4;

    /// Number of L1 SSTables that triggers compaction into L2.
    static constexpr size_t L1_COMPACTION_THRESHOLD = 4;

    /// Target file size for L1 SSTables (16 MB).
    static constexpr size_t L1_TARGET_FILE_SIZE = 16 * 1024 * 1024;

    /// Target file size for L2 SSTables (64 MB).
    static constexpr size_t L2_TARGET_FILE_SIZE = 64 * 1024 * 1024;

    /// Maximum number of levels in the LSM tree (L0, L1, L2).
    static constexpr size_t MAX_LEVELS = 3;
};

/// Entry within a Data Block.
/// Each entry is a key-value pair where key = memory_id (uint64_t)
/// and value = serialized MemoryRecord bytes.
///
/// On-disk layout of a single entry:
///   [0..8)    key:        uint64_t (memory_id, little-endian)
///   [8..12)   value_len:  uint32_t (length of value bytes, little-endian)
///   [12..12+value_len)    value:    raw bytes
struct BlockEntry {
    uint64_t key;
    std::vector<uint8_t> value;
};

/// Index entry: maps a Data Block's first key to its position in the file.
///
/// On-disk layout:
///   [0..8)    first_key:  uint64_t (smallest key in the block)
///   [8..16)   offset:     uint64_t (byte offset of compressed block in file)
///   [16..20)  size:       uint32_t (compressed block size in bytes)
struct IndexEntry {
    uint64_t first_key;
    uint64_t offset;
    uint32_t compressed_size;

    /// Serialized size of one index entry.
    static constexpr size_t SERIALIZED_SIZE = 20;

    /// Serialize this entry to bytes (little-endian).
    void writeTo(uint8_t* dest) const {
        std::memcpy(dest, &first_key, 8);
        std::memcpy(dest + 8, &offset, 8);
        std::memcpy(dest + 16, &compressed_size, 4);
    }

    /// Deserialize an entry from bytes (little-endian).
    static IndexEntry readFrom(const uint8_t* src) {
        IndexEntry entry{};
        std::memcpy(&entry.first_key, src, 8);
        std::memcpy(&entry.offset, src + 8, 8);
        std::memcpy(&entry.compressed_size, src + 16, 4);
        return entry;
    }
};

/// SSTable Footer — the last 40 bytes of the file.
///
/// On-disk layout:
///   [0..8)    index_offset:        uint64_t
///   [8..12)   index_size:          uint32_t
///   [12..20)  bloom_filter_offset: uint64_t
///   [20..24)  bloom_filter_size:   uint32_t
///   [24..28)  entry_count:         uint32_t (total number of KV entries)
///   [28..30)  version:             uint16_t
///   [30..32)  _padding:            uint16_t
///   [32..40)  magic:               uint64_t
#pragma pack(push, 1)
struct SSTableFooter {
    uint64_t index_offset{0};
    uint32_t index_size{0};
    uint64_t bloom_filter_offset{0};
    uint32_t bloom_filter_size{0};
    uint32_t entry_count{0};
    uint16_t version{SSTableFormat::VERSION};
    uint16_t _padding{0};
    uint64_t magic{SSTableFormat::MAGIC};

    static constexpr size_t SIZE = 40;

    /// Validate the footer's magic number and version.
    [[nodiscard]] bool isValid() const {
        return magic == SSTableFormat::MAGIC && version == SSTableFormat::VERSION;
    }
};
#pragma pack(pop)

static_assert(sizeof(SSTableFooter) == SSTableFooter::SIZE,
              "SSTableFooter must be exactly 40 bytes");

}  // namespace amind
