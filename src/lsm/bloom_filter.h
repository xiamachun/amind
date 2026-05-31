#pragma once

#include <cstdint>
#include <span>
#include <vector>

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "BloomFilter serialization assumes little-endian byte order");

namespace amind {

/// Bloom Filter — a space-efficient probabilistic data structure for set membership.
///
/// Uses double hashing with xxHash64 to generate k hash functions from two base hashes.
/// Formula: h_i(key) = (h1 + i * h2) % num_bits, for i in [0, k).
///
/// With 10 bits per key, the optimal number of hash functions is k ≈ 7,
/// yielding a false positive rate of approximately 0.82%.
class BloomFilter {
public:
    /// Construct a bloom filter sized for the expected number of keys.
    /// @param expected_key_count  Number of keys to be inserted.
    /// @param bits_per_key        Bits allocated per key (default: 10 ≈ 1% FPR).
    explicit BloomFilter(size_t expected_key_count, uint32_t bits_per_key = 10);

    /// Reconstruct a bloom filter from serialized data (for reading from SSTable).
    /// @param serialized_bits  The raw bit array.
    /// @param num_hash_funcs   Number of hash functions used.
    BloomFilter(std::vector<uint8_t> serialized_bits, uint32_t num_hash_funcs);

    /// Insert a key into the filter.
    void add(uint64_t key);

    /// Check if a key might be in the set.
    /// Returns false → definitely not in set.
    /// Returns true  → probably in set (with ~1% false positive rate).
    [[nodiscard]] bool mayContain(uint64_t key) const;

    /// Serialize the filter's bit array for writing to disk.
    [[nodiscard]] const std::vector<uint8_t>& data() const { return bits_; }

    /// Number of hash functions used.
    [[nodiscard]] uint32_t numHashFunctions() const { return num_hash_funcs_; }

    /// Total number of bits in the filter.
    [[nodiscard]] size_t numBits() const { return num_bits_; }

    /// Serialized size: 4 bytes (num_hash_funcs) + bit array size.
    [[nodiscard]] size_t serializedSize() const { return 4 + bits_.size(); }

    /// Serialize to bytes: [num_hash_funcs: u32] [bit_array: bytes...]
    [[nodiscard]] std::vector<uint8_t> serialize() const;

    /// Deserialize from bytes.
    [[nodiscard]] static BloomFilter deserialize(std::span<const uint8_t> data);

private:
    std::vector<uint8_t> bits_;
    size_t num_bits_;
    uint32_t num_hash_funcs_;

    /// Set the bit at position `bit_index`.
    void setBit(size_t bit_index);

    /// Test the bit at position `bit_index`.
    [[nodiscard]] bool testBit(size_t bit_index) const;
};

}  // namespace amind
