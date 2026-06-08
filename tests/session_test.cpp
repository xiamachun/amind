#include <gtest/gtest.h>

#include "session/session_manager.h"
#include "session/session_wal.h"
#include "memory/memory_store.h"
#include "graph/graph_store.h"
#include "capture/capture_pipeline.h"
#include "async/task_queue.h"

#include <filesystem>

using namespace amind;

class SessionTest : public ::testing::Test {
protected:
    std::filesystem::path test_dir_;
    std::unique_ptr<MemoryStore> store_;
    std::unique_ptr<GraphStore> graph_;
    std::unique_ptr<TaskQueue> queue_;
    std::unique_ptr<CapturePipeline> capture_;
    std::unique_ptr<SessionManager> session_mgr_;

    void SetUp() override {
        test_dir_ = std::filesystem::temp_directory_path() / "amind_session_test";
        std::filesystem::remove_all(test_dir_);

        MemoryStore::Config cfg;
        cfg.data_dir = test_dir_.string();
        cfg.embedding_dim = 4;
        store_ = std::make_unique<MemoryStore>(cfg);
        store_->init();

        graph_ = std::make_unique<GraphStore>((test_dir_ / "graph").string());
        graph_->recover();

        queue_ = std::make_unique<TaskQueue>(100);

        capture_ = std::make_unique<CapturePipeline>(
            *store_, *graph_, *queue_,
            nullptr, nullptr, nullptr, nullptr, nullptr);

        session_mgr_ = std::make_unique<SessionManager>(
            *store_, nullptr, *capture_, test_dir_.string());
    }

    void TearDown() override {
        session_mgr_.reset();
        capture_.reset();
        queue_.reset();
        graph_.reset();
        store_.reset();
        std::filesystem::remove_all(test_dir_);
    }
};

TEST_F(SessionTest, StartSession) {
    auto result = session_mgr_->startSession("agent-1", "test_user");
    ASSERT_TRUE(result.ok());

    auto& session = result.value();
    EXPECT_GT(session.session_id, 0u);
    EXPECT_EQ(session.agent_id, "agent-1");
    EXPECT_EQ(session.user_id, "test_user");
    EXPECT_TRUE(session.active);
    EXPECT_EQ(session.turn_count, 0);
}

TEST_F(SessionTest, RecordTurnSimple) {
    auto session = session_mgr_->startSession("agent-1", "test_user").value();

    // Simple turn recording (no LLM)
    auto result = session_mgr_->recordTurn(session.session_id, uint16_t(1));
    ASSERT_TRUE(result.ok());

    auto updated = session_mgr_->getSession(session.session_id);
    ASSERT_TRUE(updated.ok());
    EXPECT_EQ(updated.value().turn_count, 1);
}

TEST_F(SessionTest, CloseSession) {
    auto session = session_mgr_->startSession("agent-1", "test_user").value();
    auto close_result = session_mgr_->closeSession(session.session_id);
    ASSERT_TRUE(close_result.ok());

    auto closed = session_mgr_->getSession(session.session_id);
    ASSERT_TRUE(closed.ok());
    EXPECT_FALSE(closed.value().active);
}

TEST_F(SessionTest, CloseNonExistent) {
    auto result = session_mgr_->closeSession(99999);
    EXPECT_FALSE(result.ok());
}

TEST_F(SessionTest, RecordTurnOnClosedSession) {
    auto session = session_mgr_->startSession("test", "test_user").value();
    session_mgr_->closeSession(session.session_id);

    auto result = session_mgr_->recordTurn(session.session_id, uint16_t(1));
    EXPECT_FALSE(result.ok());
}

TEST_F(SessionTest, GetSessionSummary) {
    auto session = session_mgr_->startSession("agent-1", "test_user").value();
    session_mgr_->recordTurn(session.session_id, uint16_t(1));
    session_mgr_->recordTurn(session.session_id, uint16_t(2));

    auto summary = session_mgr_->getSessionSummary(session.session_id);
    ASSERT_TRUE(summary.ok());
    EXPECT_EQ(summary.value().agent_id, "agent-1");
    EXPECT_EQ(summary.value().turn_count, 2);
    EXPECT_TRUE(summary.value().active);
}

TEST_F(SessionTest, ListSessions) {
    session_mgr_->startSession("a1", "test_user");
    session_mgr_->startSession("a2", "test_user");

    auto list = session_mgr_->listSessions();
    EXPECT_EQ(list.size(), 2u);
}

// ── SessionWAL Tests ────────────────────────────────────────────────────

TEST(SessionWALTest, WriteAndReplay) {
    auto test_dir = std::filesystem::temp_directory_path() / "amind_swal_test";
    std::filesystem::remove_all(test_dir);
    std::filesystem::create_directories(test_dir);

    {
        SessionWAL wal(test_dir.string());
        ASSERT_TRUE(wal.open());

        Session s;
        s.session_id = 42;
        s.agent_id = "test-agent";
        s.user_id = "test-user";
        s.started_at = 1000;
        wal.appendStart(s);

        TurnRecord turn;
        turn.turn_number = 1;
        turn.user_input = "hello";
        turn.agent_response = "hi there";
        turn.timestamp = 1001;
        wal.appendTurn(42, turn);

        wal.appendClose(42);
    }

    // Replay
    SessionWAL wal2(test_dir.string());
    wal2.open();
    auto sessions = wal2.replay();

    ASSERT_EQ(sessions.size(), 1u);
    ASSERT_TRUE(sessions.count(42));
    EXPECT_EQ(sessions[42].agent_id, "test-agent");
    EXPECT_EQ(sessions[42].turn_count, 1);
    EXPECT_FALSE(sessions[42].active);
    EXPECT_EQ(sessions[42].turns.size(), 1u);
    EXPECT_EQ(sessions[42].turns[0].user_input, "hello");

    std::filesystem::remove_all(test_dir);
}
