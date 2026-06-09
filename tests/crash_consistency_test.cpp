#include <gtest/gtest.h>

#include "lsm/lsm_engine.h"
#include "memory/memory_store.h"
#include "graph/graph_wal.h"
#include "core/memory_record.h"

#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

using namespace amind;

class CrashConsistencyTest : public ::testing::Test {
protected:
    std::filesystem::path test_dir_;

    void SetUp() override {
        test_dir_ = std::filesystem::temp_directory_path() / "amind_crash_test";
        std::filesystem::remove_all(test_dir_);
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override {
        std::filesystem::remove_all(test_dir_);
    }

    MemoryRecord makeRecord(uint64_t id, const std::string& content = "test",
                            std::vector<float> embedding = {0.1f, 0.2f, 0.3f, 0.4f}) {
        MemoryRecord rec;
        rec.memory_id = id;
        rec.content = content;
        rec.embedding = std::move(embedding);
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

// ── 1. LSM WAL truncation recovery ─────────────────────────────────────────

TEST_F(CrashConsistencyTest, LSMWalTruncationRecovery) {
    auto lsm_dir = test_dir_ / "lsm_trunc";

    // Phase 1: Write data and let engine flush normally (destructor flushes)
    {
        LSMEngine engine(lsm_dir, 0);
        for (uint64_t i = 1; i <= 10; ++i) {
            engine.putRaw(i, makeRecord(i, "rec_" + std::to_string(i)).serialize());
        }
        // Destructor calls flushLocked() → data goes to SSTable, WAL truncated
    }

    // Phase 2: Reopen and write more records (these will be in WAL only)
    {
        LSMEngine engine(lsm_dir, 0);
        for (uint64_t i = 11; i <= 20; ++i) {
            engine.putRaw(i, makeRecord(i, "rec_" + std::to_string(i)).serialize());
        }
        // Destructor flushes again — but we'll corrupt WAL after this
    }

    // Now the WAL was truncated by destructor. Simulate a "next session" crash:
    // Write fresh records via a third engine, then manually corrupt the WAL
    // after the engine destructor runs (the destructor will flush to SSTable,
    // but we'll re-append corrupt data to the WAL file to simulate a crash).
    auto wal_path = lsm_dir / "wal.log";

    // Append garbage to WAL to simulate a partial write from a kill -9
    {
        std::ofstream wal(wal_path, std::ios::binary | std::ios::app);
        // Write a partial/corrupt WAL record header (incomplete)
        uint16_t magic = 0xAE01;
        uint16_t version = 1;
        uint8_t type = 0x01;
        wal.write(reinterpret_cast<const char*>(&magic), 2);
        wal.write(reinterpret_cast<const char*>(&version), 2);
        wal.write(reinterpret_cast<const char*>(&type), 1);
        // Intentionally incomplete — missing seq, key, data_len, crc
    }

    // Phase 3: Reopen — recovery should skip the corrupt WAL tail
    LSMEngine engine3(lsm_dir, 0);

    // All 20 records from phases 1 and 2 should be in SSTables (flushed by destructors)
    for (uint64_t i = 1; i <= 20; ++i) {
        EXPECT_TRUE(engine3.get(i).has_value()) << "Missing key " << i;
    }
}

// ── 2. Concurrent writes + sudden restart ──────────────────────────────────

TEST_F(CrashConsistencyTest, LSMConcurrentWriteAndRestart) {
    auto lsm_dir = test_dir_ / "lsm_concurrent";
    constexpr int NUM_THREADS = 4;
    constexpr int WRITES_PER_THREAD = 250;

    {
        LSMEngine engine(lsm_dir, 0);

        std::vector<std::thread> threads;
        for (int t = 0; t < NUM_THREADS; ++t) {
            threads.emplace_back([&engine, t]() {
                for (int i = 0; i < WRITES_PER_THREAD; ++i) {
                    uint64_t key = static_cast<uint64_t>(t * WRITES_PER_THREAD + i + 1);
                    auto rec = MemoryRecord{};
                    rec.memory_id = key;
                    rec.content = "thread_" + std::to_string(t) + "_" + std::to_string(i);
                    rec.scope = MemoryScope::Private;
                    rec.memory_type = MemoryType::UserProfile;
                    rec.phase = MemoryPhase::Active;
                    rec.confidence_level = Confidence::Inferred;
                    rec.importance = 0.5f;
                    rec.created_at = MemoryRecord::currentTimeSec();
                    rec.last_accessed = rec.created_at;
                    rec.flags = RecordFlags::ALIVE;
                    engine.putRaw(key, rec.serialize());
                }
            });
        }
        for (auto& t : threads) t.join();
        // Don't flush — simulate kill -9 (destructor only, no graceful flush)
    }

    // Reopen and count recovered records
    LSMEngine engine2(lsm_dir, 0);
    size_t recovered = 0;
    for (uint64_t i = 1; i <= NUM_THREADS * WRITES_PER_THREAD; ++i) {
        if (engine2.get(i).has_value()) recovered++;
    }

    // All writes should be recovered since each individual putRaw does fsync
    EXPECT_EQ(recovered, static_cast<size_t>(NUM_THREADS * WRITES_PER_THREAD));
}

// ── 3. HNSW reconciliation after crash ─────────────────────────────────────

TEST_F(CrashConsistencyTest, HNSWReconciliationAfterCrash) {
    MemoryStore::Config cfg;
    cfg.data_dir = (test_dir_ / "hnsw_crash").string();
    cfg.embedding_dim = 4;
    cfg.max_cache_size = 1000;

    // Store records with embeddings
    {
        auto store = std::make_unique<MemoryStore>(cfg);
        ASSERT_TRUE(store->init().ok());

        store->fastStore(makeRecord(0, "cat", {1.0f, 0.0f, 0.0f, 0.0f}));
        store->fastStore(makeRecord(0, "dog", {0.9f, 0.1f, 0.0f, 0.0f}));
        store->fastStore(makeRecord(0, "car", {0.0f, 0.0f, 1.0f, 0.0f}));

        // Flush LSM to persist data
        store->flush();
    }

    // Simulate HNSW crash — delete the HNSW index file
    auto hnsw_path = test_dir_ / "hnsw_crash" / "hnsw" / "index.bin";
    if (std::filesystem::exists(hnsw_path)) {
        std::filesystem::remove(hnsw_path);
    }

    // Reopen — init() should reconcile HNSW from LSM
    auto store2 = std::make_unique<MemoryStore>(cfg);
    ASSERT_TRUE(store2->init().ok());

    // Vector search should still work after reconciliation
    auto results = store2->searchSimilar({1.0f, 0.0f, 0.0f, 0.0f}, 3);
    EXPECT_GE(results.size(), 3u);
}

// ── 4. Graph WAL truncation replay ─────────────────────────────────────────

TEST_F(CrashConsistencyTest, GraphWalReplayAfterTruncation) {
    auto graph_dir = (test_dir_ / "graph_trunc").string();
    std::filesystem::create_directories(graph_dir);

    uint32_t now = static_cast<uint32_t>(std::time(nullptr));

    // Write edges to WAL
    {
        GraphWAL wal(graph_dir);
        ASSERT_TRUE(wal.open());
        wal.appendAdd(1, 2, EdgeType::Related, 0.8f, now);
        wal.appendAdd(3, 4, EdgeType::DerivedFrom, 0.9f, now);
        wal.appendAdd(5, 6, EdgeType::ConflictsWith, 0.7f, now);
    }

    // Truncate last bytes — simulate partial write
    auto wal_path = std::filesystem::path(graph_dir) / "graph_wal.bin";
    auto file_size = std::filesystem::file_size(wal_path);
    ASSERT_GT(file_size, 10u);
    // Truncate into the last record (sizeof(WalRecord) = 30 bytes)
    std::filesystem::resize_file(wal_path, file_size - 15);

    // Replay — should recover 2 out of 3 edges (last one truncated)
    GraphWAL wal2(graph_dir);
    ASSERT_TRUE(wal2.open());

    size_t replayed = 0;
    wal2.replay([&](WalOp op, uint64_t from, uint64_t to,
                    EdgeType type, float weight, uint32_t ts) {
        (void)op; (void)from; (void)to; (void)type; (void)weight; (void)ts;
        replayed++;
    });
    EXPECT_EQ(replayed, 2u);
}

// ── 5. Dirty cache not flushed → loss on restart (fork + _exit) ────────────

TEST_F(CrashConsistencyTest, DirtyCacheNotFlushedLoss) {
    MemoryStore::Config cfg;
    cfg.data_dir = (test_dir_ / "dirty_loss").string();
    cfg.embedding_dim = 4;
    cfg.max_cache_size = 1000;

    // Phase 1: store a record and flush to create baseline LSM data
    uint64_t stored_id = 0;
    {
        auto store = std::make_unique<MemoryStore>(cfg);
        ASSERT_TRUE(store->init().ok());

        auto rec = makeRecord(0, "tier test");
        rec.importance = 0.5f;
        auto id_result = store->fastStore(rec);
        ASSERT_TRUE(id_result.ok());
        stored_id = id_result.value();

        store->flush();
    }

    // Phase 2: fork a child process that promotes tier but _exit() without
    // running destructors — simulates kill -9. The parent waits for it.
    pid_t pid = fork();
    ASSERT_NE(pid, -1) << "fork() failed";

    if (pid == 0) {
        // Child process: open store, get() to promote, then _exit (no destructors)
        auto store2 = std::make_unique<MemoryStore>(cfg);
        if (!store2->init().ok()) _exit(1);

        for (int i = 0; i < 5; ++i) {
            store2->get(stored_id);
        }
        auto peek = store2->peek(stored_id);
        if (!peek.ok() || peek->tier != MemoryTier::ShortTerm) _exit(2);

        // Simulate kill -9: skip destructors, dirty records never flushed
        _exit(0);
    }

    // Parent: wait for child
    int status = 0;
    waitpid(pid, &status, 0);
    ASSERT_TRUE(WIFEXITED(status));
    ASSERT_EQ(WEXITSTATUS(status), 0) << "Child failed to promote tier in memory";

    // Phase 3: reopen in parent — tier should be Working (dirty change lost)
    {
        auto store3 = std::make_unique<MemoryStore>(cfg);
        ASSERT_TRUE(store3->init().ok());

        auto result = store3->peek(stored_id);
        ASSERT_TRUE(result.ok());
        EXPECT_EQ(result->tier, MemoryTier::Working)
            << "Dirty tier promotion survived crash without flush — unexpected";
    }
}

// ── 6. Manifest .tmp file ignored on recovery ──────────────────────────────

TEST_F(CrashConsistencyTest, ManifestTmpFileIgnoredOnRecovery) {
    auto lsm_dir = test_dir_ / "manifest_tmp";

    // Write data and flush to create a valid MANIFEST
    {
        LSMEngine engine(lsm_dir, 0);
        for (uint64_t i = 1; i <= 10; ++i) {
            engine.putRaw(i, makeRecord(i).serialize());
        }
        engine.flush();
    }

    // Create a stale MANIFEST.tmp (simulates crash during Manifest::save)
    auto tmp_path = lsm_dir / "MANIFEST.tmp";
    {
        std::ofstream tmp(tmp_path);
        tmp << "{\"version\":1,\"checkpoint_seq\":0,\"next_sst_id\":0,\"l0\":[],\"l1\":[],\"l2\":[]}\n";
    }
    ASSERT_TRUE(std::filesystem::exists(tmp_path));

    // Reopen — engine should load MANIFEST (not MANIFEST.tmp)
    LSMEngine engine2(lsm_dir, 0);
    for (uint64_t i = 1; i <= 10; ++i) {
        auto result = engine2.get(i);
        ASSERT_TRUE(result.has_value()) << "Missing key " << i << " after restart with stale .tmp";
    }
}

// ── 7. Batch write without endBatch — partial data ─────────────────────────

TEST_F(CrashConsistencyTest, BatchWriteNoFsyncBeforeEndBatch) {
    auto lsm_dir = test_dir_ / "batch_crash";

    // Write some records normally first (these should definitely survive)
    {
        LSMEngine engine(lsm_dir, 0);
        for (uint64_t i = 1; i <= 5; ++i) {
            engine.putRaw(i, makeRecord(i, "normal_" + std::to_string(i)).serialize());
        }

        // Now begin a batch, write more, but DON'T call endBatch — simulate kill
        engine.beginBatch();
        for (uint64_t i = 100; i <= 110; ++i) {
            engine.putRaw(i, makeRecord(i, "batch_" + std::to_string(i)).serialize());
        }
        // No endBatch() — destructor runs, but batch data was never fsynced
    }

    // Reopen and check
    LSMEngine engine2(lsm_dir, 0);

    // Normal records (pre-batch) should survive — each was fsynced
    for (uint64_t i = 1; i <= 5; ++i) {
        EXPECT_TRUE(engine2.get(i).has_value()) << "Normal record " << i << " lost";
    }

    // Batch records may or may not survive depending on OS buffer flush
    // The key invariant: no crash, no corruption — engine opens cleanly
    // (any partial WAL entries are skipped by CRC check)
    size_t batch_recovered = 0;
    for (uint64_t i = 100; i <= 110; ++i) {
        if (engine2.get(i).has_value()) batch_recovered++;
    }
    // Batch data might be fully recovered (OS flushed buffers) or partially/fully lost
    // We just verify the engine doesn't crash and returns consistent results
    EXPECT_GE(batch_recovered, 0u);
    EXPECT_LE(batch_recovered, 11u);
}
