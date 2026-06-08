#include <gtest/gtest.h>

#include "lsm/lsm_engine.h"
#include "lsm/bloom_filter.h"
#include "lsm/memtable.h"
#include "core/memory_record.h"

#include <filesystem>
#include <random>

using namespace amind;

class LSMTest : public ::testing::Test {
protected:
    std::filesystem::path test_dir_;

    void SetUp() override {
        test_dir_ = std::filesystem::temp_directory_path() / "amind_lsm_test";
        std::filesystem::remove_all(test_dir_);
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override {
        std::filesystem::remove_all(test_dir_);
    }

    MemoryRecord makeRecord(uint64_t id, const std::string& content = "test") {
        MemoryRecord rec;
        rec.memory_id = id;
        rec.content = content;
        rec.scope = MemoryScope::Private;
        rec.memory_type = MemoryType::UserProfile;
        rec.phase = MemoryPhase::Active;
        rec.confidence_level = Confidence::Inferred;
        rec.importance = 0.5f;
        rec.created_at = MemoryRecord::currentTimeSec();
        rec.last_accessed = rec.created_at;
        rec.flags = RecordFlags::ALIVE;
        return rec;
    }
};

// ── MemTable Tests ──────────────────────────────────────────────────────

TEST_F(LSMTest, MemTablePutAndGet) {
    MemTable mt;
    auto rec = makeRecord(42, "hello world");
    mt.put(rec);

    auto result = mt.getRaw(42);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result->empty());
}

TEST_F(LSMTest, MemTableRemoveCreatesTombstone) {
    MemTable mt;
    mt.put(makeRecord(1, "data"));
    mt.remove(1);

    // getRaw returns nullopt for tombstones
    auto result = mt.getRaw(1);
    EXPECT_FALSE(result.has_value());

    // But forEach reveals the tombstone entry
    bool found_tombstone = false;
    mt.forEach([&](uint64_t id, const std::vector<uint8_t>&, bool is_tombstone) {
        if (id == 1 && is_tombstone) found_tombstone = true;
    });
    EXPECT_TRUE(found_tombstone);
}

TEST_F(LSMTest, MemTableForEach) {
    MemTable mt;
    mt.put(makeRecord(10, "a"));
    mt.put(makeRecord(20, "b"));
    mt.put(makeRecord(30, "c"));

    size_t count = 0;
    mt.forEach([&](uint64_t /*key*/, const std::vector<uint8_t>& /*data*/, bool /*tombstone*/) {
        count++;
    });
    EXPECT_EQ(count, 3u);
}

TEST_F(LSMTest, MemTableSize) {
    MemTable mt;
    EXPECT_EQ(mt.size(), 0u);
    mt.put(makeRecord(1));
    EXPECT_EQ(mt.size(), 1u);
    mt.put(makeRecord(2));
    EXPECT_EQ(mt.size(), 2u);
}

// ── BloomFilter Tests ──────────────────────────────────────────────────

TEST_F(LSMTest, BloomFilterBasicInsertAndQuery) {
    BloomFilter bf(100, 10);
    bf.add(42);
    bf.add(100);
    bf.add(999);

    EXPECT_TRUE(bf.mayContain(42));
    EXPECT_TRUE(bf.mayContain(100));
    EXPECT_TRUE(bf.mayContain(999));
}

TEST_F(LSMTest, BloomFilterNoFalseNegatives) {
    BloomFilter bf(1000, 10);
    for (uint64_t i = 0; i < 1000; ++i) {
        bf.add(i);
    }
    for (uint64_t i = 0; i < 1000; ++i) {
        EXPECT_TRUE(bf.mayContain(i));
    }
}

TEST_F(LSMTest, BloomFilterFalsePositiveRate) {
    BloomFilter bf(1000, 10);
    for (uint64_t i = 0; i < 1000; ++i) {
        bf.add(i);
    }
    size_t false_positives = 0;
    for (uint64_t i = 10000; i < 20000; ++i) {
        if (bf.mayContain(i)) false_positives++;
    }
    double fpr = static_cast<double>(false_positives) / 10000.0;
    EXPECT_LT(fpr, 0.02);  // Should be < 1% with 10 bits/key
}

TEST_F(LSMTest, BloomFilterSerializeDeserialize) {
    BloomFilter bf(100, 10);
    bf.add(1);
    bf.add(2);
    bf.add(3);

    auto serialized = bf.serialize();
    auto bf2 = BloomFilter::deserialize(serialized);

    EXPECT_TRUE(bf2.mayContain(1));
    EXPECT_TRUE(bf2.mayContain(2));
    EXPECT_TRUE(bf2.mayContain(3));
}

// ── LSMEngine Tests ─────────────────────────────────────────────────────

TEST_F(LSMTest, EnginePutAndGet) {
    LSMEngine engine(test_dir_, 0);
    auto rec = makeRecord(1, "test content");
    engine.putRaw(1, rec.serialize());

    auto result = engine.get(1);
    ASSERT_TRUE(result.has_value());

    auto deserialized = MemoryRecord::deserialize(*result);
    ASSERT_TRUE(deserialized.ok());
    EXPECT_EQ(deserialized.value().memory_id, 1u);
    EXPECT_EQ(deserialized.value().content, "test content");
}

TEST_F(LSMTest, EngineGetMissing) {
    LSMEngine engine(test_dir_, 0);
    auto result = engine.get(999);
    EXPECT_FALSE(result.has_value());
}

TEST_F(LSMTest, EngineFlushCreatesSSTable) {
    LSMEngine engine(test_dir_, 0);
    for (uint64_t i = 1; i <= 10; ++i) {
        engine.putRaw(i, makeRecord(i, "entry " + std::to_string(i)).serialize());
    }
    engine.flush();
    EXPECT_GE(engine.l0TableCount(), 1u);

    // Data still retrievable after flush
    auto result = engine.get(5);
    ASSERT_TRUE(result.has_value());
}

TEST_F(LSMTest, EngineRemoveThenGet) {
    LSMEngine engine(test_dir_, 0);
    engine.putRaw(1, makeRecord(1).serialize());
    engine.remove(1);

    auto result = engine.get(1);
    EXPECT_FALSE(result.has_value());
}

TEST_F(LSMTest, EngineRecoveryFromWAL) {
    {
        LSMEngine engine(test_dir_, 0);
        engine.putRaw(100, makeRecord(100, "persistent").serialize());
        engine.putRaw(200, makeRecord(200, "also persistent").serialize());
        // Don't flush — data is only in WAL + memtable
    }

    // Reopen — should recover from WAL
    LSMEngine engine2(test_dir_, 0);
    auto r1 = engine2.get(100);
    auto r2 = engine2.get(200);
    ASSERT_TRUE(r1.has_value());
    ASSERT_TRUE(r2.has_value());
}

TEST_F(LSMTest, EngineCompaction) {
    LSMEngine engine(test_dir_, 0);

    // Write enough to trigger multiple flushes
    for (uint64_t i = 0; i < 500; ++i) {
        engine.putRaw(i, makeRecord(i, "data " + std::to_string(i)).serialize());
    }
    engine.flush();
    engine.forceCompact();

    // All data still accessible
    for (uint64_t i = 0; i < 500; ++i) {
        auto result = engine.get(i);
        ASSERT_TRUE(result.has_value()) << "Missing key " << i;
    }
}

TEST_F(LSMTest, EngineForEachLive) {
    LSMEngine engine(test_dir_, 0);
    engine.putRaw(1, makeRecord(1).serialize());
    engine.putRaw(2, makeRecord(2).serialize());
    engine.putRaw(3, makeRecord(3).serialize());
    engine.remove(2);  // tombstone

    size_t live_count = 0;
    engine.forEachLive([&](uint64_t /*id*/, const std::vector<uint8_t>& /*data*/) {
        live_count++;
    });
    EXPECT_EQ(live_count, 2u);  // 1 and 3 are live
}
