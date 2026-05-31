#include "block.h"

#include <lz4.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace amind {

// ─── BlockBuilder ──────────────────────────────────────────────────────────────

BlockBuilder::BlockBuilder(size_t target_size)
    : target_size_(target_size)
    , uncompressed_size_(4)  // 4 bytes for entry_count header
    , first_key_(0) {
}

void BlockBuilder::add(uint64_t key, std::span<const uint8_t> value) {
    if (entries_.empty()) {
        first_key_ = key;
    }
    entries_.push_back({key, {value.begin(), value.end()}});
    // 8 bytes key + 4 bytes value_len + value bytes
    uncompressed_size_ += 8 + 4 + value.size();
}

std::vector<uint8_t> BlockBuilder::finish() {
    // Serialize entries into an uncompressed buffer
    std::vector<uint8_t> uncompressed(uncompressed_size_);
    size_t offset = 0;

    // Write entry count
    auto entry_count = static_cast<uint32_t>(entries_.size());
    std::memcpy(uncompressed.data() + offset, &entry_count, 4);
    offset += 4;

    // Write each entry: key(u64) + value_len(u32) + value(bytes)
    for (const auto& entry : entries_) {
        std::memcpy(uncompressed.data() + offset, &entry.key, 8);
        offset += 8;

        auto value_len = static_cast<uint32_t>(entry.value.size());
        std::memcpy(uncompressed.data() + offset, &value_len, 4);
        offset += 4;

        std::memcpy(uncompressed.data() + offset, entry.value.data(), entry.value.size());
        offset += entry.value.size();
    }

    // LZ4 compress
    int max_compressed_size = LZ4_compressBound(static_cast<int>(uncompressed.size()));
    // Output format: [uncompressed_size: u32] [compressed_data: bytes...]
    std::vector<uint8_t> compressed(4 + max_compressed_size);

    auto uncompressed_len = static_cast<uint32_t>(uncompressed.size());
    std::memcpy(compressed.data(), &uncompressed_len, 4);

    int actual_compressed = LZ4_compress_default(
        reinterpret_cast<const char*>(uncompressed.data()),
        reinterpret_cast<char*>(compressed.data() + 4),
        static_cast<int>(uncompressed.size()),
        max_compressed_size);

    if (actual_compressed <= 0) {
        throw std::runtime_error("LZ4 compression failed");
    }

    compressed.resize(4 + actual_compressed);

    reset();
    return compressed;
}

void BlockBuilder::reset() {
    entries_.clear();
    uncompressed_size_ = 4;
    first_key_ = 0;
}

// ─── BlockReader ───────────────────────────────────────────────────────────────

BlockReader::BlockReader(std::span<const uint8_t> compressed_data) {
    if (compressed_data.size() < 4) {
        throw std::invalid_argument("Compressed block too small");
    }

    // Read uncompressed size
    uint32_t uncompressed_size = 0;
    std::memcpy(&uncompressed_size, compressed_data.data(), 4);

    if (uncompressed_size == 0 || uncompressed_size > 64 * 1024 * 1024) {
        throw std::invalid_argument("Invalid uncompressed size in block header");
    }

    // LZ4 decompress
    decompressed_.resize(uncompressed_size);
    int decompressed_bytes = LZ4_decompress_safe(
        reinterpret_cast<const char*>(compressed_data.data() + 4),
        reinterpret_cast<char*>(decompressed_.data()),
        static_cast<int>(compressed_data.size() - 4),
        static_cast<int>(uncompressed_size));

    if (decompressed_bytes < 0 ||
        static_cast<uint32_t>(decompressed_bytes) != uncompressed_size) {
        throw std::runtime_error("LZ4 decompression failed");
    }

    parseEntries();
}

void BlockReader::parseEntries() {
    if (decompressed_.size() < 4) {
        throw std::invalid_argument("Block data too small for entry count");
    }

    std::memcpy(&entry_count_, decompressed_.data(), 4);
    entry_offsets_.reserve(entry_count_);

    size_t offset = 4;
    for (size_t i = 0; i < entry_count_; ++i) {
        if (offset + 12 > decompressed_.size()) {
            throw std::invalid_argument("Block data truncated at entry header");
        }
        entry_offsets_.push_back(offset);

        // Skip key (8 bytes)
        offset += 8;

        // Read value_len
        uint32_t value_len = 0;
        std::memcpy(&value_len, decompressed_.data() + offset, 4);
        offset += 4;

        // Skip value
        offset += value_len;
    }
}

BlockReader::Entry BlockReader::entryAt(size_t index) const {
    if (index >= entry_count_) {
        throw std::out_of_range("Block entry index out of range");
    }

    size_t offset = entry_offsets_[index];

    uint64_t key = 0;
    std::memcpy(&key, decompressed_.data() + offset, 8);
    offset += 8;

    uint32_t value_len = 0;
    std::memcpy(&value_len, decompressed_.data() + offset, 4);
    offset += 4;

    return {key, {decompressed_.data() + offset, value_len}};
}

int BlockReader::findKey(uint64_t key) const {
    // Binary search over sorted entries
    int low = 0;
    int high = static_cast<int>(entry_count_) - 1;

    while (low <= high) {
        int mid = low + (high - low) / 2;
        uint64_t mid_key = 0;
        std::memcpy(&mid_key, decompressed_.data() + entry_offsets_[mid], 8);

        if (mid_key == key) {
            return mid;
        } else if (mid_key < key) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return -1;
}

uint64_t BlockReader::firstKey() const {
    if (entry_count_ == 0) {
        throw std::runtime_error("Cannot get first key of empty block");
    }
    uint64_t key = 0;
    std::memcpy(&key, decompressed_.data() + entry_offsets_[0], 8);
    return key;
}

uint64_t BlockReader::lastKey() const {
    if (entry_count_ == 0) {
        throw std::runtime_error("Cannot get last key of empty block");
    }
    uint64_t key = 0;
    std::memcpy(&key, decompressed_.data() + entry_offsets_[entry_count_ - 1], 8);
    return key;
}

}  // namespace amind
