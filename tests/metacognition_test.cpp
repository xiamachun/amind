#include <gtest/gtest.h>

#include "metacognition/metacognition.h"
#include "memory/memory_store.h"
#include "graph/graph_store.h"

#include <filesystem>

using namespace amind;

class MetaCognitionTest : public ::testing::Test {
protected:
    std::filesystem::path test_dir_;
    std::unique_ptr<MemoryStore> store_;
    std::unique_ptr<GraphStore> graph_;
    std::unique_ptr<MetaCognition> metacog_;

    void SetUp() override {
        test_dir_ = std::filesystem::temp_directory_path() / "amind_metacog_test";
        std::filesystem::remove_all(test_dir_);

        MemoryStore::Config cfg;
        cfg.data_dir = test_dir_.string();
        cfg.embedding_dim = 4;
        store_ = std::make_unique<MemoryStore>(cfg);
        store_->init();

        auto graph_dir = test_dir_ / "graph";
        graph_ = std::make_unique<GraphStore>(graph_dir.string());
        graph_->recover();

        metacog_ = std::make_unique<MetaCognition>(*store_, *graph_);
    }

    void TearDown() override {
        metacog_.reset();
        graph_.reset();
        store_.reset();
        std::filesystem::remove_all(test_dir_);
    }

    MemoryRecord makeRecord(const std::string& content, MemoryScope scope, MemoryType memory_type,
                            const std::string& agent_id = "default", const std::string& user_id = "test_user") {
        MemoryRecord rec;
        rec.content = content;
        rec.scope = scope;
        rec.memory_type = memory_type;
        rec.agent_id = agent_id;
        rec.user_id = user_id;
        rec.phase = MemoryPhase::Active;
        rec.confidence_level = Confidence::Inferred;
        rec.importance = 0.5f;
        rec.flags = RecordFlags::ALIVE;
        rec.embedding = {0.1f, 0.2f, 0.3f, 0.4f};
        return rec;
    }
};

TEST_F(MetaCognitionTest, GetCoverageGlobal) {
    store_->fastStore(makeRecord("fact 1", MemoryScope::Private, MemoryType::UserProfile));
    store_->fastStore(makeRecord("fact 2", MemoryScope::AgentShared, MemoryType::DomainKnowledge));
    store_->fastStore(makeRecord("fact 3", MemoryScope::AgentShared, MemoryType::Reference));

    auto stats = metacog_->getCoverage("");
    EXPECT_EQ(stats.total, 3u);
    EXPECT_EQ(stats.active, 3u);
    EXPECT_EQ(stats.stale, 0u);
    EXPECT_EQ(stats.conflicted, 0u);
}

TEST_F(MetaCognitionTest, GetCoverageFilteredByAgent) {
    store_->fastStore(makeRecord("fact A", MemoryScope::Private, MemoryType::UserProfile, "agent-1"));
    store_->fastStore(makeRecord("fact B", MemoryScope::Private, MemoryType::UserProfile, "agent-1"));
    store_->fastStore(makeRecord("fact C", MemoryScope::Private, MemoryType::UserProfile, "agent-2"));

    auto stats_1 = metacog_->getCoverage("agent-1");
    EXPECT_EQ(stats_1.total, 2u);

    auto stats_2 = metacog_->getCoverage("agent-2");
    EXPECT_EQ(stats_2.total, 1u);
}

TEST_F(MetaCognitionTest, GetStaleMemories) {
    auto rec = makeRecord("stale fact", MemoryScope::Private, MemoryType::UserProfile);
    rec.confidence_level = Confidence::Stale;
    store_->fastStore(rec);

    store_->fastStore(makeRecord("fresh fact", MemoryScope::Private, MemoryType::UserProfile));

    auto stale = metacog_->getStaleMemories();
    EXPECT_EQ(stale.size(), 1u);
}

TEST_F(MetaCognitionTest, GetConflicts) {
    auto id1 = store_->fastStore(makeRecord("A is true", MemoryScope::Private, MemoryType::UserProfile)).value();
    auto id2 = store_->fastStore(makeRecord("A is false", MemoryScope::Private, MemoryType::UserProfile)).value();
    graph_->addEdge(id1, id2, EdgeType::ConflictsWith, 1.0f);

    auto conflicts = metacog_->getConflicts();
    EXPECT_GE(conflicts.size(), 1u);
    EXPECT_EQ(conflicts[0].memory_a, id1);
    EXPECT_EQ(conflicts[0].memory_b, id2);
}

TEST_F(MetaCognitionTest, EmptyCoverage) {
    auto stats = metacog_->getCoverage("");
    EXPECT_EQ(stats.total, 0u);
    EXPECT_EQ(stats.active, 0u);
}
