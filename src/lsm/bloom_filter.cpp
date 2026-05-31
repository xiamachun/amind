#include "bloom_filter.h"

#include <xxhash.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace amind {

BloomFilter::BloomFilter(size_t expected_key_count, uint32_t bits_per_key) {
    // Minimum 64 bits to avoid degenerate cases
    num_bits_ = std::max<size_t>(64, expected_key_count * bits_per_key);

    // Round up to the next multiple of 8 for byte alignment
    num_bits_ = (num_bits_ + 7) & ~static_cast<size_t>(7);

    bits_.resize(num_bits_ / 8, 0);

    // Optimal number of hash functions: k = (m/n) * ln(2) ≈ bits_per_key * 0.693
    num_hash_funcs_ = static_cast<uint32_t>(
        std::round(static_cast<double>(bits_per_key) * 0.6931471805599453));
    num_hash_funcs_ = std::clamp(num_hash_funcs_, 1u, 30u);
}

BloomFilter::BloomFilter(std::vector<uint8_t> serialized_bits, uint32_t num_hash_funcs)
    : bits_(std::move(serialized_bits))
    , num_bits_(bits_.size() * 8)
    , num_hash_funcs_(num_hash_funcs) {
}

void BloomFilter::add(uint64_t key) {
    // Double hashing: h_i(key) = (h1 + i * h2) % num_bits
    uint64_t hash1 = XXH64(&key, sizeof(key), 0);
    uint64_t hash2 = XXH64(&key, sizeof(key), 1);

    for (uint32_t i = 0; i < num_hash_funcs_; ++i) {
        size_t bit_index = (hash1 + static_cast<uint64_t>(i) * hash2) % num_bits_;
        setBit(bit_index);
    }
}

bool BloomFilter::mayContain(uint64_t key) const {
    uint64_t hash1 = XXH64(&key, sizeof(key), 0);
    uint64_t hash2 = XXH64(&key, sizeof(key), 1);

    for (uint32_t i = 0; i < num_hash_funcs_; ++i) {
        size_t bit_index = (hash1 + static_cast<uint64_t>(i) * hash2) % num_bits_;
        if (!testBit(bit_index)) {
            return false;
        }
    }
    return true;
}

std::vector<uint8_t> BloomFilter::serialize() const {
    std::vector<uint8_t> result;
    result.reserve(4 + bits_.size());

    // Write num_hash_funcs as little-endian uint32_t
    uint32_t hash_funcs = num_hash_funcs_;
    result.push_back(static_cast<uint8_t>(hash_funcs & 0xFF));
    result.push_back(static_cast<uint8_t>((hash_funcs >> 8) & 0xFF));
    result.push_back(static_cast<uint8_t>((hash_funcs >> 16) & 0xFF));
    result.push_back(static_cast<uint8_t>((hash_funcs >> 24) & 0xFF));

    // Append the bit array
    result.insert(result.end(), bits_.begin(), bits_.end());
    return result;
}

BloomFilter BloomFilter::deserialize(std::span<const uint8_t> data) {
    if (data.size() < 4) {
        throw std::invalid_argument("BloomFilter data too small: need at least 4 bytes");
    }

    uint32_t num_hash_funcs = 0;
    std::memcpy(&num_hash_funcs, data.data(), 4);

    std::vector<uint8_t> bits(data.begin() + 4, data.end());
    return BloomFilter(std::move(bits), num_hash_funcs);
}

void BloomFilter::setBit(size_t bit_index) {
    bits_[bit_index / 8] |= (1u << (bit_index % 8));
}

bool BloomFilter::testBit(size_t bit_index) const {
    return (bits_[bit_index / 8] & (1u << (bit_index % 8))) != 0;
}

}  // namespace amind
