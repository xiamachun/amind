#include <gtest/gtest.h>

#include "memory/memory_store.h"

#include <filesystem>

using namespace amind;

class MemoryStoreTest : public ::testing::Test {
protected:
    std::filesystem::path test_dir_;
    std::unique_ptr<MemoryStore> store_;

    void SetUp() override {
        test_dir_ = std::filesystem::temp_directory_path() / "amind_memstore_test";
        std::filesystem::remove_all(test_dir_);

        MemoryStore::Config cfg;
        cfg.data_dir = test_dir_.string();
        cfg.embedding_dim = 4;
        cfg.max_cache_size = 1000;
        cfg.conflict_similarity_threshold = 0.85f;
        store_ = std::make_unique<MemoryStore>(cfg);
        auto result = store_->init();
        ASSERT_TRUE(result.ok());
    }

    void TearDown() override {
        store_.reset();
        std::filesystem::remove_all(test_dir_);
    }

    MemoryRecord makeRecord(const std::string& content,
                            std::vector<float> embedding = {0.1f, 0.2f, 0.3f, 0.4f}) {
        MemoryRecord rec;
        rec.content = content;
        rec.embedding = std::move(embedding);
        rec.scope = MemoryScope::Private;
        rec.memory_type = MemoryType::UserProfile;
        rec.phase = MemoryPhase::Active;
        rec.confidence_level = Confidence::Inferred;
        rec.importance = 0.7f;
        rec.flags = RecordFlags::ALIVE;
        return rec;
    }
};

TEST_F(MemoryStoreTest, FastStoreAndGet) {
    auto rec = makeRecord("test memory");
    auto store_result = store_->fastStore(rec);
    ASSERT_TRUE(store_result.ok());

    uint64_t id = store_result.value();
    EXPECT_GT(id, 0u);

    auto get_result = store_->get(id);
    ASSERT_TRUE(get_result.ok());
    EXPECT_EQ(get_result.value().content, "test memory");
    EXPECT_EQ(get_result.value().memory_id, id);
}

TEST_F(MemoryStoreTest, GetNonExistent) {
    auto result = store_->get(99999);
    EXPECT_FALSE(result.ok());
}

TEST_F(MemoryStoreTest, UpdateCreatesVersionChain) {
    auto rec = makeRecord("original content");
    auto id1 = store_->fastStore(rec).value();

    auto id2_result = store_->update(id1, "updated content", {0.5f, 0.6f, 0.7f, 0.8f});
    ASSERT_TRUE(id2_result.ok());
    uint64_t id2 = id2_result.value();
    EXPECT_NE(id1, id2);

    // New version has updated content
    auto new_rec = store_->get(id2);
    ASSERT_TRUE(new_rec.ok());
    EXPECT_EQ(new_rec.value().content, "updated content");
    EXPECT_EQ(new_rec.value().parent_id, id1);

    // Old version marked as Versioned
    auto old_rec = store_->get(id1);
    ASSERT_TRUE(old_rec.ok());
    EXPECT_EQ(old_rec.value().phase, MemoryPhase::Versioned);
}

TEST_F(MemoryStoreTest, RemoveMarksTombstone) {
    auto id = store_->fastStore(makeRecord("to delete")).value();

    auto remove_result = store_->remove(id);
    ASSERT_TRUE(remove_result.ok());

    auto get_result = store_->get(id);
    ASSERT_TRUE(get_result.ok());
    EXPECT_EQ(get_result.value().phase, MemoryPhase::Tombstone);
}

TEST_F(MemoryStoreTest, SearchSimilar) {
    // Insert records with known embeddings
    store_->fastStore(makeRecord("cat", {1.0f, 0.0f, 0.0f, 0.0f}));
    store_->fastStore(makeRecord("dog", {0.9f, 0.1f, 0.0f, 0.0f}));
    store_->fastStore(makeRecord("car", {0.0f, 0.0f, 1.0f, 0.0f}));

    auto results = store_->searchSimilar({1.0f, 0.0f, 0.0f, 0.0f}, 2);
    ASSERT_GE(results.size(), 2u);
    // First result should be the most similar (cat)
    EXPECT_GT(results[0].second, results[1].second);
}

TEST_F(MemoryStoreTest, ApplyDecay) {
    auto rec = makeRecord("decaying memory");
    rec.importance = 1.0f;
    rec.created_at = MemoryRecord::currentTimeSec() - 86400;  // 1 day ago
    rec.last_accessed = rec.created_at;  // also 1 day ago
    auto id = store_->fastStore(rec).value();

    store_->applyDecay();

    auto after = store_->get(id);
    ASSERT_TRUE(after.ok());
    EXPECT_LT(after.value().importance, 1.0f);
}

TEST_F(MemoryStoreTest, CacheEviction) {
    MemoryStore::Config cfg;
    cfg.data_dir = (test_dir_ / "small_cache").string();
    cfg.embedding_dim = 4;
    cfg.max_cache_size = 5;
    auto small_store = std::make_unique<MemoryStore>(cfg);
    small_store->init();

    // Insert more than cache limit
    for (int i = 0; i < 10; ++i) {
        small_store->fastStore(makeRecord("entry " + std::to_string(i)));
    }

    // Should not crash; total memories reported might be less than 10 after eviction
    EXPECT_LE(small_store->totalMemories(), 10u);
}

TEST_F(MemoryStoreTest, GetHistory) {
    auto id1 = store_->fastStore(makeRecord("v1")).value();
    auto id2 = store_->update(id1, "v2", {0.2f, 0.3f, 0.4f, 0.5f}).value();
    auto id3 = store_->update(id2, "v3", {0.3f, 0.4f, 0.5f, 0.6f}).value();

    auto history = store_->getHistory(id3);
    ASSERT_TRUE(history.ok());
    EXPECT_GE(history.value().size(), 2u);
}

TEST_F(MemoryStoreTest, TotalMemories) {
    EXPECT_EQ(store_->totalMemories(), 0u);
    store_->fastStore(makeRecord("a"));
    store_->fastStore(makeRecord("b"));
    EXPECT_EQ(store_->totalMemories(), 2u);
}

TEST_F(MemoryStoreTest, ListMemoriesWithFilters) {
    auto rec1 = makeRecord("user pref");
    rec1.scope = MemoryScope::Private;
    rec1.memory_type = MemoryType::UserProfile;
    store_->fastStore(rec1);

    auto rec2 = makeRecord("agent note");
    rec2.scope = MemoryScope::AgentShared;
    rec2.memory_type = MemoryType::DomainKnowledge;
    store_->fastStore(rec2);

    MemoryStore::ListFilter filter;
    filter.scope_filter = "private";
    auto result = store_->listMemories(filter);
    EXPECT_EQ(result.total, 1u);
    EXPECT_EQ(result.records[0].content, "user pref");
}

// ── Dirty Writeback Tests ───────────────────────────────────────────────

TEST_F(MemoryStoreTest, GetIncrementsAccessCountAndPersists) {
    auto rec = makeRecord("dirty writeback test");
    auto id = store_->fastStore(rec).value();

    // get() should increment access_count and mark dirty
    store_->get(id);
    store_->get(id);
    store_->get(id);

    auto before_flush = store_->get(id);
    ASSERT_TRUE(before_flush.ok());
    EXPECT_GE(before_flush->access_count, 4u);  // fastStore peek + 3 gets + this get

    // Flush dirty records to LSM
    store_->flush();

    // Destroy and recreate store — simulates process restart
    store_.reset();
    MemoryStore::Config cfg;
    cfg.data_dir = test_dir_.string();
    cfg.embedding_dim = 4;
    cfg.max_cache_size = 1000;
    store_ = std::make_unique<MemoryStore>(cfg);
    ASSERT_TRUE(store_->init().ok());

    // access_count should be persisted after restart
    auto after_restart = store_->get(id);
    ASSERT_TRUE(after_restart.ok());
    EXPECT_GE(after_restart->access_count, 4u);
}

TEST_F(MemoryStoreTest, ApplyDecayPersistsImportance) {
    auto rec = makeRecord("decay persist test");
    rec.importance = 1.0f;
    rec.created_at = MemoryRecord::currentTimeSec() - 86400;  // 1 day ago
    rec.last_accessed = rec.created_at;
    auto id = store_->fastStore(rec).value();

    store_->applyDecay();

    auto decayed = store_->peek(id);
    ASSERT_TRUE(decayed.ok());
    float importance_after_decay = decayed->importance;
    EXPECT_LT(importance_after_decay, 1.0f);

    // Flush and restart
    store_->flush();
    store_.reset();
    MemoryStore::Config cfg;
    cfg.data_dir = test_dir_.string();
    cfg.embedding_dim = 4;
    cfg.max_cache_size = 1000;
    store_ = std::make_unique<MemoryStore>(cfg);
    ASSERT_TRUE(store_->init().ok());

    // Importance should survive restart
    auto after_restart = store_->get(id);
    ASSERT_TRUE(after_restart.ok());
    EXPECT_NEAR(after_restart->importance, importance_after_decay, 0.01f);
}

TEST_F(MemoryStoreTest, EvictFlushsDirtyBeforeEviction) {
    // Create a store with very small cache
    auto small_dir = test_dir_ / "evict_dirty_test";
    MemoryStore::Config cfg;
    cfg.data_dir = small_dir.string();
    cfg.embedding_dim = 4;
    cfg.max_cache_size = 5;
    auto small_store = std::make_unique<MemoryStore>(cfg);
    ASSERT_TRUE(small_store->init().ok());

    // Store records and get() them to make them dirty
    std::vector<uint64_t> ids;
    for (int i = 0; i < 5; ++i) {
        auto id = small_store->fastStore(makeRecord("entry " + std::to_string(i))).value();
        ids.push_back(id);
        small_store->get(id);  // increment access_count, mark dirty
        small_store->get(id);
    }

    // Insert more to trigger eviction — dirty records should be flushed first
    for (int i = 5; i < 12; ++i) {
        small_store->fastStore(makeRecord("overflow " + std::to_string(i)));
    }

    // Flush and restart
    small_store->flush();
    small_store.reset();

    small_store = std::make_unique<MemoryStore>(cfg);
    ASSERT_TRUE(small_store->init().ok());

    // At least some of the original records should have persisted access_count
    int persisted_count = 0;
    for (auto id : ids) {
        auto result = small_store->get(id);
        if (result.ok() && result->access_count >= 2) {
            persisted_count++;
        }
    }
    EXPECT_GT(persisted_count, 0);
}

TEST_F(MemoryStoreTest, TierPromotionPersistsAfterRestart) {
    auto rec = makeRecord("tier promotion persist");
    rec.importance = 0.5f;
    auto id = store_->fastStore(rec).value();

    // get() increments access_count; at threshold (3) triggers Working→ShortTerm
    for (int i = 0; i < 4; ++i) {
        store_->get(id);
    }

    auto promoted = store_->peek(id);
    ASSERT_TRUE(promoted.ok());
    EXPECT_EQ(promoted->tier, MemoryTier::ShortTerm);

    store_->flush();
    store_.reset();
    MemoryStore::Config cfg;
    cfg.data_dir = test_dir_.string();
    cfg.embedding_dim = 4;
    cfg.max_cache_size = 1000;
    store_ = std::make_unique<MemoryStore>(cfg);
    ASSERT_TRUE(store_->init().ok());

    auto after = store_->get(id);
    ASSERT_TRUE(after.ok());
    EXPECT_EQ(after->tier, MemoryTier::ShortTerm);
}

TEST_F(MemoryStoreTest, ApplyDecayConfidenceStalePersistedOnRestart) {
    auto rec = makeRecord("stale confidence test");
    rec.importance = 0.5f;
    rec.created_at = MemoryRecord::currentTimeSec() - 86400 * 4;  // 4 days ago
    rec.last_accessed = rec.created_at;
    auto id = store_->fastStore(rec).value();

    store_->applyDecay();

    auto decayed = store_->peek(id);
    ASSERT_TRUE(decayed.ok());
    EXPECT_EQ(decayed->confidence_level, Confidence::Stale);

    store_->flush();
    store_.reset();
    MemoryStore::Config cfg;
    cfg.data_dir = test_dir_.string();
    cfg.embedding_dim = 4;
    cfg.max_cache_size = 1000;
    store_ = std::make_unique<MemoryStore>(cfg);
    ASSERT_TRUE(store_->init().ok());

    auto after = store_->get(id);
    ASSERT_TRUE(after.ok());
    EXPECT_EQ(after->confidence_level, Confidence::Stale);
}

TEST_F(MemoryStoreTest, DirtyFlushThenCompactionPreservesData) {
    for (int i = 0; i < 10; ++i) {
        auto rec = makeRecord("compaction_test_" + std::to_string(i));
        store_->fastStore(rec);
    }

    auto all_ids = std::vector<uint64_t>();
    store_->scanAll([&](const MemoryRecord& r) {
        all_ids.push_back(r.memory_id);
    });

    for (auto id : all_ids) {
        store_->get(id);  // mark dirty
    }

    store_->flush();

    store_.reset();
    MemoryStore::Config cfg;
    cfg.data_dir = test_dir_.string();
    cfg.embedding_dim = 4;
    cfg.max_cache_size = 1000;
    store_ = std::make_unique<MemoryStore>(cfg);
    ASSERT_TRUE(store_->init().ok());

    int found = 0;
    for (auto id : all_ids) {
        if (store_->get(id).ok()) found++;
    }
    EXPECT_EQ(found, static_cast<int>(all_ids.size()));
}
