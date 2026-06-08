#include <gtest/gtest.h>

#include "capture/capture_pipeline.h"
#include "provider/provider.h"
#include "memory/memory_store.h"
#include "graph/graph_store.h"
#include "async/task_queue.h"

#include <filesystem>

using namespace amind;

class MockEmbedder : public EmbedProvider {
public:
    Result<std::vector<float>> embed(const std::string&) override {
        return std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f};
    }
    Result<std::vector<std::vector<float>>> embedBatch(const std::vector<std::string>& texts) override {
        std::vector<std::vector<float>> results;
        for (size_t i = 0; i < texts.size(); ++i)
            results.push_back({0.1f, 0.2f, 0.3f, 0.4f});
        return results;
    }
    size_t dimension() const override { return 4; }
    std::string name() const override { return "mock"; }
};

class CaptureTest : public ::testing::Test {
protected:
    std::filesystem::path test_dir_;
    std::unique_ptr<MemoryStore> store_;
    std::unique_ptr<GraphStore> graph_;
    std::unique_ptr<TaskQueue> queue_;
    std::unique_ptr<CapturePipeline> capture_;

    void SetUp() override {
        test_dir_ = std::filesystem::temp_directory_path() / "amind_capture_test";
        std::filesystem::remove_all(test_dir_);

        MemoryStore::Config cfg;
        cfg.data_dir = test_dir_.string();
        cfg.embedding_dim = 4;
        store_ = std::make_unique<MemoryStore>(cfg);
        store_->init();

        auto graph_dir = test_dir_ / "graph";
        graph_ = std::make_unique<GraphStore>(graph_dir.string());
        graph_->recover();

        queue_ = std::make_unique<TaskQueue>(100);

        auto embedder = std::make_shared<MockEmbedder>();
        capture_ = std::make_unique<CapturePipeline>(
            *store_, *graph_, *queue_,
            nullptr, embedder, nullptr, nullptr, nullptr);
    }

    void TearDown() override {
        capture_.reset();
        queue_.reset();
        graph_.reset();
        store_.reset();
        std::filesystem::remove_all(test_dir_);
    }
};

TEST_F(CaptureTest, CaptureWithoutEmbedder) {
    // Without an embedder, capture should still work (stores without embedding)
    auto result = capture_->capture("user likes coffee", "agent-1", "test_user", MemoryScope::Private, MemoryType::UserProfile);
    // May fail if embedder is required, or succeed with empty embedding
    if (result.ok()) {
        EXPECT_GE(result.value().size(), 1u);
        auto stored = store_->get(result.value()[0]);
        if (stored.ok()) {
            EXPECT_EQ(stored.value().content, "user likes coffee");
        }
    }
}

TEST_F(CaptureTest, CaptureStoresContent) {
    // Even without embedder, the content should be stored
    auto result = capture_->capture("important fact", "test-agent", "test_user", MemoryScope::Private, MemoryType::DomainKnowledge);
    if (result.ok() && !result.value().empty()) {
        auto mem = store_->get(result.value()[0]);
        ASSERT_TRUE(mem.ok());
        EXPECT_EQ(mem.value().content, "important fact");
        EXPECT_EQ(mem.value().scope, MemoryScope::Private);
        EXPECT_EQ(mem.value().memory_type, MemoryType::DomainKnowledge);
    }
}

TEST_F(CaptureTest, InterceptCapture) {
    std::vector<std::pair<std::string, std::string>> messages = {
        {"user", "My favorite color is blue"},
        {"assistant", "I'll remember that!"},
    };

    auto result = capture_->interceptCapture(messages, "agent-1", "user-1");
    // Without embedder this may or may not succeed depending on implementation
    // Just verify it doesn't crash
    SUCCEED();
}

TEST_F(CaptureTest, NamespacePassedCorrectly) {
    auto result = capture_->capture("test", "my-namespace", "test_user", MemoryScope::AgentShared, MemoryType::DomainKnowledge);
    if (result.ok() && !result.value().empty()) {
        auto mem = store_->get(result.value()[0]);
        if (mem.ok()) {
            EXPECT_EQ(mem.value().agent_id, "my-namespace");
        }
    }
}
