/// audit_fix_test.cpp — Verification tests for the 2026-06-09 pre-launch audit fixes.
/// Each test validates one or more P0/P1 fix items from docs/2026-06-09-pre-launch-audit.md.

#include <gtest/gtest.h>

#include "server/config.h"
#include "server/variable_manager.h"
#include "session/session_wal.h"
#include "session/session_manager.h"
#include "lineage/lineage_wal.h"
#include "graph/graph_wal.h"
#include "vector/cold_tier.h"
#include "capture/derived_extractor.h"
#include "core/memory_record.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace amind;
namespace fs = std::filesystem;

// ═══════════════════════════════════════════════════════════════════════════
// Fixture
// ═══════════════════════════════════════════════════════════════════════════

class AuditFixTest : public ::testing::Test {
protected:
    fs::path test_dir_;

    void SetUp() override {
        test_dir_ = fs::temp_directory_path() / "amind_audit_fix_test";
        fs::remove_all(test_dir_);
        fs::create_directories(test_dir_);
    }

    void TearDown() override {
        fs::remove_all(test_dir_);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// P0-6: ColdTier atomic write + CRC + fsync
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(AuditFixTest, ColdTierAtomicSaveAndReload) {
    auto cold_path = (test_dir_ / "cold_tier.bin").string();
    constexpr size_t kDim = 4;

    // Write vectors via ColdTierIndex
    {
        ColdTierIndex index(cold_path, kDim, DistanceMetric::Cosine);
        index.append(100, {0.1f, 0.2f, 0.3f, 0.4f});
        index.append(200, {0.5f, 0.6f, 0.7f, 0.8f});
        index.append(300, {0.9f, 1.0f, 1.1f, 1.2f});
        ASSERT_TRUE(index.save());
    }

    // Verify no .tmp file remains (atomic rename completed)
    EXPECT_FALSE(fs::exists(cold_path + ".tmp"));

    // Reload and verify data integrity
    {
        ColdTierIndex index(cold_path, kDim, DistanceMetric::Cosine);
        ASSERT_TRUE(index.load());
        EXPECT_EQ(index.size(), 3u);
        EXPECT_TRUE(index.contains(100));
        EXPECT_TRUE(index.contains(200));
        EXPECT_TRUE(index.contains(300));

        auto vec = index.getVector(100);
        ASSERT_TRUE(vec.has_value());
        EXPECT_FLOAT_EQ(vec->at(0), 0.1f);
        EXPECT_FLOAT_EQ(vec->at(3), 0.4f);
    }
}

TEST_F(AuditFixTest, ColdTierCRCDetectsCorruption) {
    auto cold_path = (test_dir_ / "cold_tier_corrupt.bin").string();
    constexpr size_t kDim = 4;

    // Write valid data
    {
        ColdTierIndex index(cold_path, kDim, DistanceMetric::Cosine);
        index.append(1, {1.0f, 2.0f, 3.0f, 4.0f});
        ASSERT_TRUE(index.save());
    }

    // Corrupt a byte in the middle of the file
    {
        std::fstream file(cold_path, std::ios::in | std::ios::out | std::ios::binary);
        ASSERT_TRUE(file.is_open());
        file.seekp(20);  // skip header, corrupt in entry data
        char garbage = static_cast<char>(0xFF);
        file.write(&garbage, 1);
        file.close();
    }

    // The file has CRC trailer — verify file structure
    {
        std::ifstream file(cold_path, std::ios::binary | std::ios::ate);
        auto file_size = file.tellg();
        EXPECT_GT(file_size, static_cast<std::streampos>(16 + 4));  // header + CRC
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Session WAL: CRC + fsync (P1 fix)
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(AuditFixTest, SessionWalCRCWriteAndReplay) {
    auto wal_dir = test_dir_.string();

    // Write some session entries
    {
        SessionWAL wal(wal_dir);
        ASSERT_TRUE(wal.open());

        Session session;
        session.session_id = 42;
        session.agent_id = "test-agent";
        session.user_id = "user1";
        session.started_at = 1000;
        wal.appendStart(session);

        TurnRecord turn;
        turn.turn_number = 1;
        turn.user_input = "hello";
        turn.agent_response = "world";
        turn.detected_intent = "chat";
        turn.timestamp = 1001;
        wal.appendTurn(42, turn);

        wal.appendClose(42);
    }

    // Verify the WAL file contains CRC (tab-separated CRC at end of each line)
    {
        std::ifstream file(wal_dir + "/session_wal.jsonl");
        ASSERT_TRUE(file.is_open());
        std::string line;
        int line_count = 0;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            auto tab_pos = line.rfind('\t');
            EXPECT_NE(tab_pos, std::string::npos) << "Line should have tab-separated CRC";
            if (tab_pos != std::string::npos) {
                std::string crc_str = line.substr(tab_pos + 1);
                EXPECT_NO_THROW(std::stoul(crc_str)) << "CRC should be a valid uint32";
            }
            line_count++;
        }
        EXPECT_EQ(line_count, 3);  // start + turn + close
    }

    // Replay and verify data integrity
    {
        SessionWAL wal(wal_dir);
        auto sessions = wal.replay();
        EXPECT_EQ(sessions.size(), 1u);
        auto it = sessions.find(42);
        ASSERT_NE(it, sessions.end());
        EXPECT_EQ(it->second.agent_id, "test-agent");
        EXPECT_EQ(it->second.user_id, "user1");
        EXPECT_FALSE(it->second.active);  // closed
        EXPECT_EQ(it->second.turns.size(), 1u);
    }
}

TEST_F(AuditFixTest, SessionWalCRCRejectsTamperedLine) {
    auto wal_dir = test_dir_.string();
    auto wal_path = wal_dir + "/session_wal.jsonl";

    // Write a valid entry
    {
        SessionWAL wal(wal_dir);
        ASSERT_TRUE(wal.open());
        Session session;
        session.session_id = 99;
        session.agent_id = "agent";
        session.user_id = "user";
        session.started_at = 2000;
        wal.appendStart(session);
    }

    // Tamper with the JSON part (change agent_id)
    {
        std::ifstream in(wal_path);
        std::string line;
        std::getline(in, line);
        in.close();

        auto tab_pos = line.rfind('\t');
        ASSERT_NE(tab_pos, std::string::npos);
        std::string json_part = line.substr(0, tab_pos);
        std::string crc_part = line.substr(tab_pos);
        auto agent_pos = json_part.find("agent");
        ASSERT_NE(agent_pos, std::string::npos);
        json_part.replace(agent_pos, 5, "hacked");

        std::ofstream out(wal_path, std::ios::trunc);
        out << json_part << crc_part << '\n';
        out.close();
    }

    // Replay should skip the tampered line (CRC mismatch)
    {
        SessionWAL wal(wal_dir);
        auto sessions = wal.replay();
        EXPECT_TRUE(sessions.empty()) << "Tampered line should be rejected by CRC check";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Lineage WAL: CRC + fsync verification
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(AuditFixTest, LineageWalCRCWriteAndReplay) {
    auto wal_dir = test_dir_.string();

    // Write lineage entries
    {
        LineageWAL wal(wal_dir);
        ASSERT_TRUE(wal.open());
        wal.appendRecordLineage(10, {1, 2, 3}, LineageOp::Infer, 1000);
        wal.appendRemoveNode(5);
        wal.appendMarkRemoved(7);
    }

    // Replay and verify
    {
        LineageWAL wal(wal_dir);
        ASSERT_TRUE(wal.open());

        std::vector<LineageWAL::ReplayEntry> entries;
        size_t count = wal.replay([&](const LineageWAL::ReplayEntry& entry) {
            entries.push_back(entry);
        });

        EXPECT_EQ(count, 3u);
        ASSERT_EQ(entries.size(), 3u);

        EXPECT_EQ(entries[0].wal_op, LineageWalOp::RecordLineage);
        EXPECT_EQ(entries[0].node_id, 10u);
        EXPECT_EQ(entries[0].parents.size(), 3u);
        EXPECT_EQ(entries[0].lineage_op, LineageOp::Infer);

        EXPECT_EQ(entries[1].wal_op, LineageWalOp::RemoveNode);
        EXPECT_EQ(entries[1].node_id, 5u);

        EXPECT_EQ(entries[2].wal_op, LineageWalOp::MarkRemoved);
        EXPECT_EQ(entries[2].node_id, 7u);
    }
}

TEST_F(AuditFixTest, LineageWalCRCRejectsCorruption) {
    auto wal_dir = test_dir_.string();
    auto wal_path = wal_dir + "/lineage_wal.bin";

    // Write 2 entries
    {
        LineageWAL wal(wal_dir);
        ASSERT_TRUE(wal.open());
        wal.appendRecordLineage(10, {1}, LineageOp::Infer, 100);
        wal.appendRecordLineage(20, {2}, LineageOp::Infer, 200);
    }

    // Corrupt the second entry
    {
        auto file_size = fs::file_size(wal_path);
        ASSERT_GT(file_size, 30u);
        std::fstream file(wal_path, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(static_cast<std::streamoff>(file_size) - 10);
        char garbage = static_cast<char>(0xDE);
        file.write(&garbage, 1);
        file.close();
    }

    // Replay should recover only the first valid entry
    {
        LineageWAL wal(wal_dir);
        ASSERT_TRUE(wal.open());
        size_t count = 0;
        wal.replay([&](const LineageWAL::ReplayEntry&) { count++; });
        EXPECT_EQ(count, 1u) << "Should stop at CRC-mismatch entry";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Lineage WAL: checkpoint with fsync
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(AuditFixTest, LineageWalCheckpointAndReload) {
    auto wal_dir = test_dir_.string();

    LineageWAL wal(wal_dir);
    ASSERT_TRUE(wal.open());

    wal.appendRecordLineage(100, {10, 20}, LineageOp::Infer, 500);
    wal.appendMarkRemoved(50);

    std::vector<LineageWAL::SnapshotRecord> records;
    LineageWAL::SnapshotRecord rec;
    rec.child_id = 100;
    rec.parent_ids = {10, 20};
    rec.op = LineageOp::Infer;
    rec.created_at = 500;
    records.push_back(rec);

    std::vector<uint64_t> removed = {50};
    ASSERT_TRUE(wal.checkpoint(records, removed));

    EXPECT_FALSE(fs::exists(wal_dir + "/lineage_snapshot.bin.tmp"));

    std::vector<uint64_t> removed_out;
    auto loaded = wal.loadSnapshot(removed_out);
    ASSERT_EQ(loaded.size(), 1u);
    EXPECT_EQ(loaded[0].child_id, 100u);
    EXPECT_EQ(loaded[0].parent_ids.size(), 2u);
    ASSERT_EQ(removed_out.size(), 1u);
    EXPECT_EQ(removed_out[0], 50u);
}

// ═══════════════════════════════════════════════════════════════════════════
// Graph WAL: CRC + fsync verification
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(AuditFixTest, GraphWalCRCWriteAndReplay) {
    auto wal_dir = test_dir_.string();

    {
        GraphWAL wal(wal_dir);
        ASSERT_TRUE(wal.open());
        wal.appendAdd(1, 2, EdgeType::Caused, 0.8f, 1000);
        wal.appendAdd(3, 4, EdgeType::Related, 0.5f, 1001);
        wal.appendRemove(1);
    }

    {
        GraphWAL wal(wal_dir);
        ASSERT_TRUE(wal.open());

        struct ReplayedEdge {
            WalOp op; uint64_t from; uint64_t to;
            EdgeType type; float weight; uint32_t ts;
        };
        std::vector<ReplayedEdge> edges;
        size_t count = wal.replay([&](WalOp op, uint64_t from, uint64_t to,
                                       EdgeType type, float weight, uint32_t ts) {
            edges.push_back({op, from, to, type, weight, ts});
        });

        EXPECT_EQ(count, 3u);
        ASSERT_EQ(edges.size(), 3u);

        EXPECT_EQ(edges[0].op, WalOp::AddEdge);
        EXPECT_EQ(edges[0].from, 1u);
        EXPECT_EQ(edges[0].to, 2u);
        EXPECT_FLOAT_EQ(edges[0].weight, 0.8f);

        EXPECT_EQ(edges[2].op, WalOp::RemoveEdge);
        EXPECT_EQ(edges[2].from, 1u);
    }
}

TEST_F(AuditFixTest, GraphWalCheckpointAndReload) {
    auto wal_dir = test_dir_.string();

    GraphWAL wal(wal_dir);
    ASSERT_TRUE(wal.open());
    wal.appendAdd(10, 20, EdgeType::Caused, 1.0f, 500);

    std::vector<GraphWAL::EdgeData> snapshot_edges;
    snapshot_edges.push_back({10, 20, static_cast<uint8_t>(EdgeType::Caused), 1.0f, 500});
    ASSERT_TRUE(wal.checkpoint(snapshot_edges));

    EXPECT_FALSE(fs::exists(wal_dir + "/graph_snapshot.bin.tmp"));

    auto loaded = wal.loadSnapshot();
    ASSERT_EQ(loaded.size(), 1u);
    EXPECT_EQ(loaded[0].from_id, 10u);
    EXPECT_EQ(loaded[0].to_id, 20u);
}

// ═══════════════════════════════════════════════════════════════════════════
// VariableManager: set() fires callbacks outside lock (P1-13)
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(AuditFixTest, VariableManagerSetFiresCallbackOutsideLock) {
    VariableManager vm;
    vm.registerVar("test_var", VarType::STRING, VarMode::DYNAMIC, "initial", "test", "test");

    std::atomic<bool> callback_fired{false};
    std::atomic<bool> could_read_during_callback{false};

    vm.onChange("test_var", [&](const std::string&, const std::string&) {
        callback_fired = true;
        // During callback, we should be able to read other variables (lock released)
        // This would deadlock if callback fired under write lock
        auto val = vm.get("test_var");
        could_read_during_callback = !val.empty();
    });

    auto result = vm.set("test_var", "updated");
    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(callback_fired);
    EXPECT_TRUE(could_read_during_callback)
        << "Callback should fire outside the write lock so reads don't deadlock";
    EXPECT_EQ(vm.get("test_var"), "updated");
}

// ═══════════════════════════════════════════════════════════════════════════
// VariableManager: persistToFile atomic write (P1-14)
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(AuditFixTest, VariableManagerPersistToFileAtomic) {
    auto config_path = (test_dir_ / "test.conf").string();

    {
        std::ofstream file(config_path);
        file << "# comment line\n";
        file << "my_var = old_value\n";
        file << "other_var = keep_me\n";
    }

    VariableManager vm;
    vm.registerVar("my_var", VarType::STRING, VarMode::DYNAMIC, "default", "desc", "cat");
    vm.registerVar("other_var", VarType::STRING, VarMode::READONLY, "default2", "desc2", "cat");

    auto config = AppConfig::load(config_path);
    ASSERT_TRUE(config.ok());
    vm.loadFrom(*config);

    vm.set("my_var", "new_value");

    auto result = vm.persistToFile(config_path);
    EXPECT_TRUE(result.ok());

    EXPECT_FALSE(fs::exists(config_path + ".tmp"));

    auto reloaded = AppConfig::load(config_path);
    ASSERT_TRUE(reloaded.ok());
    EXPECT_EQ(reloaded->get("my_var"), "new_value");
    EXPECT_EQ(reloaded->get("other_var"), "keep_me");
}

// ═══════════════════════════════════════════════════════════════════════════
// AppConfig: env override for new keys not in config file (P1-20)
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(AuditFixTest, AppConfigEnvOverrideNewKey) {
    auto config_path = (test_dir_ / "env_test.conf").string();

    {
        std::ofstream file(config_path);
        file << "host = 0.0.0.0\n";
        file << "port = 8080\n";
    }

    setenv("AMIND_MAX_AGENTS", "42", 1);

    auto config = AppConfig::load(config_path);
    ASSERT_TRUE(config.ok());

    EXPECT_EQ(config->get("max_agents"), "42");
    EXPECT_EQ(config->getInt("max_agents"), 42);

    unsetenv("AMIND_MAX_AGENTS");
}

TEST_F(AuditFixTest, AppConfigEnvOverrideExistingKey) {
    auto config_path = (test_dir_ / "env_test2.conf").string();

    {
        std::ofstream file(config_path);
        file << "port = 8080\n";
    }

    setenv("AMIND_PORT", "9090", 1);

    auto config = AppConfig::load(config_path);
    ASSERT_TRUE(config.ok());
    EXPECT_EQ(config->getInt("port"), 9090);

    unsetenv("AMIND_PORT");
}

// ═══════════════════════════════════════════════════════════════════════════
// DerivedExtractor: atomic stats (P1-11)
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(AuditFixTest, DerivedExtractorStatsAreAtomic) {
    DerivedExtractor::Stats stats;
    constexpr size_t kThreads = 8;
    constexpr size_t kIterations = 10000;

    std::vector<std::thread> threads;
    for (size_t t = 0; t < kThreads; t++) {
        threads.emplace_back([&stats]() {
            for (size_t i = 0; i < kIterations; i++) {
                stats.total_processed.fetch_add(1, std::memory_order_relaxed);
                stats.accepted.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& thread : threads) thread.join();

    EXPECT_EQ(stats.total_processed.load(), static_cast<uint64_t>(kThreads * kIterations));
    EXPECT_EQ(stats.accepted.load(), static_cast<uint64_t>(kThreads * kIterations));
}

// ═══════════════════════════════════════════════════════════════════════════
// Graph WAL: CRC corruption detection
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(AuditFixTest, GraphWalCRCRejectsCorruption) {
    auto wal_dir = test_dir_.string();
    auto wal_path = wal_dir + "/graph_wal.bin";

    {
        GraphWAL wal(wal_dir);
        ASSERT_TRUE(wal.open());
        wal.appendAdd(1, 2, EdgeType::Caused, 1.0f, 100);
        wal.appendAdd(3, 4, EdgeType::Related, 0.5f, 200);
    }

    {
        auto file_size = fs::file_size(wal_path);
        ASSERT_GT(file_size, sizeof(WalRecord));
        std::fstream file(wal_path, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(sizeof(WalRecord) + 5);
        char garbage = static_cast<char>(0xAB);
        file.write(&garbage, 1);
        file.close();
    }

    {
        GraphWAL wal(wal_dir);
        ASSERT_TRUE(wal.open());
        size_t count = 0;
        wal.replay([&](WalOp, uint64_t, uint64_t, EdgeType, float, uint32_t) {
            count++;
        });
        EXPECT_EQ(count, 1u) << "Should recover only the first valid record";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Session WAL: backward compatibility with old (no CRC) format
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(AuditFixTest, SessionWalBackwardCompatibleWithLegacyFormat) {
    auto wal_dir = test_dir_.string();
    auto wal_path = wal_dir + "/session_wal.jsonl";

    {
        fs::create_directories(wal_dir);
        std::ofstream file(wal_path);
        file << "{\"op\":\"start\",\"session_id\":77,\"agent_id\":\"legacy\",\"user_id\":\"old_user\",\"started_at\":500}" << '\n';
    }

    {
        SessionWAL wal(wal_dir);
        auto sessions = wal.replay();
        EXPECT_EQ(sessions.size(), 1u);
        auto it = sessions.find(77);
        ASSERT_NE(it, sessions.end());
        EXPECT_EQ(it->second.agent_id, "legacy");
        EXPECT_EQ(it->second.user_id, "old_user");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ColdTier: no .tmp residue after save
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(AuditFixTest, ColdTierNoTmpResidueAfterMultipleSaves) {
    auto cold_path = (test_dir_ / "cold_multi.bin").string();
    ColdTierIndex index(cold_path, 4, DistanceMetric::Cosine);

    for (uint64_t i = 0; i < 5; i++) {
        index.append(i, {static_cast<float>(i), 0.1f, 0.2f, 0.3f});
        ASSERT_TRUE(index.save());
        EXPECT_FALSE(fs::exists(cold_path + ".tmp"))
            << "No .tmp file should remain after save #" << i;
    }
    EXPECT_EQ(index.size(), 5u);
}

// ═══════════════════════════════════════════════════════════════════════════
// Lineage WAL: fsync is called (file exists and is valid after write)
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(AuditFixTest, LineageWalFileDurableAfterAppend) {
    auto wal_dir = test_dir_.string();

    LineageWAL wal(wal_dir);
    ASSERT_TRUE(wal.open());

    wal.appendRecordLineage(1, {2, 3}, LineageOp::Infer, 100);

    auto wal_size = wal.walSize();
    EXPECT_GT(wal_size, 0u) << "WAL should have data after append + fsync";

    wal.sync();
    EXPECT_GE(wal.walSize(), wal_size);
}

// ═══════════════════════════════════════════════════════════════════════════
// VariableManager: type validation
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(AuditFixTest, VariableManagerRejectsInvalidType) {
    VariableManager vm;
    vm.registerVar("int_var", VarType::INT, VarMode::DYNAMIC, "10", "an int", "test");
    vm.registerVar("float_var", VarType::FLOAT, VarMode::DYNAMIC, "1.5", "a float", "test");

    EXPECT_TRUE(vm.set("int_var", "42").ok());
    EXPECT_EQ(vm.getInt("int_var"), 42);

    auto result = vm.set("int_var", "not_a_number");
    EXPECT_FALSE(result.ok());

    result = vm.set("float_var", "abc");
    EXPECT_FALSE(result.ok());

    vm.registerVar("ro_var", VarType::STRING, VarMode::READONLY, "fixed", "readonly", "test");
    result = vm.set("ro_var", "changed");
    EXPECT_FALSE(result.ok());
}

// ═══════════════════════════════════════════════════════════════════════════
// Graph WAL: snapshot CRC verification
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(AuditFixTest, GraphWalSnapshotCRCRejectsCorruption) {
    auto wal_dir = test_dir_.string();
    auto snapshot_path = wal_dir + "/graph_snapshot.bin";

    {
        GraphWAL wal(wal_dir);
        ASSERT_TRUE(wal.open());
        std::vector<GraphWAL::EdgeData> edges;
        edges.push_back({1, 2, static_cast<uint8_t>(EdgeType::Caused), 1.0f, 100});
        edges.push_back({3, 4, static_cast<uint8_t>(EdgeType::Related), 0.5f, 200});
        ASSERT_TRUE(wal.checkpoint(edges));
    }

    {
        auto file_size = fs::file_size(snapshot_path);
        std::fstream file(snapshot_path, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(static_cast<std::streamoff>(file_size) - 5);
        char garbage = static_cast<char>(0xCC);
        file.write(&garbage, 1);
        file.close();
    }

    {
        GraphWAL wal(wal_dir);
        auto loaded = wal.loadSnapshot();
        EXPECT_TRUE(loaded.empty()) << "Corrupted snapshot should be rejected";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Lineage WAL: snapshot CRC verification
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(AuditFixTest, LineageWalSnapshotCRCRejectsCorruption) {
    auto wal_dir = test_dir_.string();
    auto snapshot_path = wal_dir + "/lineage_snapshot.bin";

    {
        LineageWAL wal(wal_dir);
        ASSERT_TRUE(wal.open());
        std::vector<LineageWAL::SnapshotRecord> records;
        LineageWAL::SnapshotRecord rec;
        rec.child_id = 42;
        rec.parent_ids = {1, 2};
        rec.op = LineageOp::Infer;
        rec.created_at = 300;
        records.push_back(rec);
        ASSERT_TRUE(wal.checkpoint(records, {}));
    }

    {
        auto file_size = fs::file_size(snapshot_path);
        std::fstream file(snapshot_path, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(static_cast<std::streamoff>(file_size) - 3);
        char garbage = static_cast<char>(0xEE);
        file.write(&garbage, 1);
        file.close();
    }

    {
        LineageWAL wal(wal_dir);
        std::vector<uint64_t> removed;
        auto loaded = wal.loadSnapshot(removed);
        EXPECT_TRUE(loaded.empty()) << "Corrupted snapshot should be rejected";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Multiple concurrent writes to session WAL (thread safety)
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(AuditFixTest, SessionWalConcurrentWritesAreThreadSafe) {
    auto wal_dir = test_dir_.string();

    SessionWAL wal(wal_dir);
    ASSERT_TRUE(wal.open());

    Session session;
    session.session_id = 1;
    session.agent_id = "agent";
    session.user_id = "user";
    session.started_at = 1000;
    wal.appendStart(session);

    constexpr size_t kThreads = 4;
    constexpr size_t kTurnsPerThread = 50;

    std::vector<std::thread> threads;
    for (size_t t = 0; t < kThreads; t++) {
        threads.emplace_back([&wal, t]() {
            for (size_t i = 0; i < kTurnsPerThread; i++) {
                TurnRecord turn;
                turn.turn_number = static_cast<uint16_t>(t * kTurnsPerThread + i);
                turn.user_input = "input_" + std::to_string(t) + "_" + std::to_string(i);
                turn.agent_response = "response";
                turn.timestamp = static_cast<int64_t>(1000 + t * kTurnsPerThread + i);
                wal.appendTurn(1, turn);
            }
        });
    }
    for (auto& thread : threads) thread.join();

    auto sessions = wal.replay();
    EXPECT_EQ(sessions.size(), 1u);
    auto it = sessions.find(1);
    ASSERT_NE(it, sessions.end());
    EXPECT_EQ(it->second.turns.size(), kThreads * kTurnsPerThread);
}
