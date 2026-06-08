
#include "vector/hnsw_index.h"

#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace amind;

namespace {

constexpr size_t kDim = 32;

std::vector<float> randomVector(std::mt19937& rng, size_t dim) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> vec(dim);
    float norm = 0.0f;
    for (auto& v : vec) {
        v = dist(rng);
        norm += v * v;
    }
    norm = std::sqrt(norm);
    for (auto& v : vec) v /= norm;
    return vec;
}

HNSWConfig makeConfig() {
    HNSWConfig config;
    config.dimension = kDim;
    config.maxConnections = 8;
    config.maxConnectionsLayer0 = 16;
    config.efConstruction = 50;
    config.efSearch = 30;
    config.hotCapacity = 10000;
    return config;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// softDelete
// ═══════════════════════════════════════════════════════════════════════════

TEST(HNSWv2Test, SoftDeleteRemovesFromSearch) {
    auto index = HNSWIndex(makeConfig());
    std::mt19937 rng(42);

    for (uint64_t i = 1; i <= 100; ++i) {
        index.insert(i, randomVector(rng, kDim));
    }
    ASSERT_EQ(index.size(), 100u);

    EXPECT_TRUE(index.softDelete(50));
    EXPECT_EQ(index.size(), 99u);
    EXPECT_FALSE(index.contains(50));

    // Double delete returns false
    EXPECT_FALSE(index.softDelete(50));

    // Non-existent ID returns false
    EXPECT_FALSE(index.softDelete(999));
}

TEST(HNSWv2Test, SoftDeletedNodeNotInSearchResults) {
    auto index = HNSWIndex(makeConfig());
    std::mt19937 rng(42);

    auto targetVec = randomVector(rng, kDim);
    index.insert(1, targetVec);

    for (uint64_t i = 2; i <= 50; ++i) {
        index.insert(i, randomVector(rng, kDim));
    }

    // Search should find ID 1
    auto results = index.search(targetVec, 10);
    bool foundBefore = false;
    for (const auto& r : results) {
        if (r.id == 1) foundBefore = true;
    }
    EXPECT_TRUE(foundBefore);

    // After soft delete, ID 1 should not appear
    index.softDelete(1);
    auto resultsAfter = index.search(targetVec, 10);
    for (const auto& r : resultsAfter) {
        EXPECT_NE(r.id, 1u);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// deletedCount
// ═══════════════════════════════════════════════════════════════════════════

TEST(HNSWv2Test, DeletedCountTracksDeleted) {
    auto index = HNSWIndex(makeConfig());
    std::mt19937 rng(42);

    for (uint64_t i = 1; i <= 50; ++i) {
        index.insert(i, randomVector(rng, kDim));
    }
    EXPECT_EQ(index.deletedCount(), 0u);

    index.softDelete(10);
    index.softDelete(20);
    index.softDelete(30);
    EXPECT_EQ(index.deletedCount(), 3u);
    EXPECT_EQ(index.size(), 47u);
}

// ═══════════════════════════════════════════════════════════════════════════
// compact
// ═══════════════════════════════════════════════════════════════════════════

TEST(HNSWv2Test, CompactPurgesDeletedNodes) {
    auto index = HNSWIndex(makeConfig());
    std::mt19937 rng(42);

    for (uint64_t i = 1; i <= 100; ++i) {
        index.insert(i, randomVector(rng, kDim));
    }

    // Delete 30 nodes
    for (uint64_t i = 1; i <= 30; ++i) {
        index.softDelete(i);
    }
    EXPECT_EQ(index.size(), 70u);
    EXPECT_EQ(index.deletedCount(), 30u);

    // Compact should purge all deleted nodes
    size_t purged = index.compact();
    EXPECT_EQ(purged, 30u);
    EXPECT_EQ(index.size(), 70u);
    EXPECT_EQ(index.deletedCount(), 0u);

    // Deleted nodes should not be containable
    for (uint64_t i = 1; i <= 30; ++i) {
        EXPECT_FALSE(index.contains(i));
    }
    // Surviving nodes should still be present
    for (uint64_t i = 31; i <= 100; ++i) {
        EXPECT_TRUE(index.contains(i));
    }
}

TEST(HNSWv2Test, CompactWithNoDeletionsIsNoop) {
    auto index = HNSWIndex(makeConfig());
    std::mt19937 rng(42);

    for (uint64_t i = 1; i <= 50; ++i) {
        index.insert(i, randomVector(rng, kDim));
    }

    size_t purged = index.compact();
    EXPECT_EQ(purged, 0u);
    EXPECT_EQ(index.size(), 50u);
}

TEST(HNSWv2Test, CompactPreservesSearchQuality) {
    auto index = HNSWIndex(makeConfig());
    std::mt19937 rng(42);

    // Insert vectors and remember one
    auto targetVec = randomVector(rng, kDim);
    index.insert(999, targetVec);

    for (uint64_t i = 1; i <= 100; ++i) {
        index.insert(i, randomVector(rng, kDim));
    }

    // Delete half the random vectors
    for (uint64_t i = 1; i <= 50; ++i) {
        index.softDelete(i);
    }

    index.compact();

    // Search for the target — it should be the closest match to itself
    auto results = index.search(targetVec, 5);
    ASSERT_FALSE(results.empty());
    EXPECT_EQ(results[0].id, 999u);
    EXPECT_LT(results[0].distance, 0.01f);
}

// ═══════════════════════════════════════════════════════════════════════════
// relegateToCold
// ═══════════════════════════════════════════════════════════════════════════

TEST(HNSWv2Test, RelegateToColdWithoutColdTierReturnsZero) {
    auto index = HNSWIndex(makeConfig());
    std::mt19937 rng(42);

    index.insert(1, randomVector(rng, kDim));
    EXPECT_EQ(index.relegateToCold({1}), 0u);
}

TEST(HNSWv2Test, RelegateToColdMovesVectors) {
    auto config = makeConfig();
    config.coldTierPath = "/tmp/amind_test_cold";
    auto index = HNSWIndex(config);
    index.initColdTier();

    std::mt19937 rng(42);
    for (uint64_t i = 1; i <= 20; ++i) {
        index.insert(i, randomVector(rng, kDim));
    }
    EXPECT_EQ(index.hotSize(), 20u);

    // Relegate 5 vectors
    size_t relegated = index.relegateToCold({1, 2, 3, 4, 5});
    EXPECT_EQ(relegated, 5u);
    EXPECT_EQ(index.hotSize(), 15u);
    EXPECT_EQ(index.coldSize(), 5u);

    // Relegated IDs no longer in hot layer
    for (uint64_t i = 1; i <= 5; ++i) {
        EXPECT_FALSE(index.contains(i));
    }
}

TEST(HNSWv2Test, RelegateToColdSkipsDeletedAndMissing) {
    auto config = makeConfig();
    config.coldTierPath = "/tmp/amind_test_cold2";
    auto index = HNSWIndex(config);
    index.initColdTier();

    std::mt19937 rng(42);
    for (uint64_t i = 1; i <= 10; ++i) {
        index.insert(i, randomVector(rng, kDim));
    }

    // Delete one, then try to relegate it + a non-existent ID
    index.softDelete(5);

    size_t relegated = index.relegateToCold({5, 999, 3, 7});
    EXPECT_EQ(relegated, 2u);  // Only 3 and 7
    EXPECT_EQ(index.hotSize(), 7u);  // 10 - 1(deleted) - 2(relegated)
}
