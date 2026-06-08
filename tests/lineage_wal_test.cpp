#include "lineage/lineage_index.h"
#include "lineage/lineage_wal.h"

#include <gtest/gtest.h>
#include <filesystem>
#include <string>

using namespace amind;

namespace {
std::string createTempDir() {
    auto path = std::filesystem::temp_directory_path() / "lineage_wal_test_XXXXXX";
    std::string dir = path.string();
    std::filesystem::create_directories(dir);
    return dir;
}

void cleanDir(const std::string& dir) {
    std::filesystem::remove_all(dir);
}
}

class LineageWALTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = createTempDir();
    }
    void TearDown() override {
        cleanDir(test_dir_);
    }
    std::string test_dir_;
};

TEST_F(LineageWALTest, PersistAndRecover) {
    {
        LineageIndex idx(test_dir_);
        idx.recordLineage(100, {1, 2, 3}, LineageOp::Summarize);
        idx.recordLineage(200, {1}, LineageOp::Infer);
        idx.markRemoved(1);
    }

    // Reconstruct from WAL
    LineageIndex idx2(test_dir_);
    auto parents_100 = idx2.getParents(100);
    ASSERT_EQ(parents_100.size(), 3u);

    auto parents_200 = idx2.getParents(200);
    ASSERT_EQ(parents_200.size(), 1u);
    EXPECT_EQ(parents_200[0], 1u);

    EXPECT_EQ(idx2.getLineageOp(100), LineageOp::Summarize);
    EXPECT_EQ(idx2.getLineageOp(200), LineageOp::Infer);

    // node 1 is removed, but 100 still has parents 2 and 3 alive
    EXPECT_TRUE(idx2.hasLiveParents(100));
    // node 200's only parent is 1, which is removed
    EXPECT_FALSE(idx2.hasLiveParents(200));
}

TEST_F(LineageWALTest, RemoveNodePersists) {
    {
        LineageIndex idx(test_dir_);
        idx.recordLineage(100, {1, 2}, LineageOp::Aggregate);
        idx.removeNode(100);
    }

    LineageIndex idx2(test_dir_);
    EXPECT_TRUE(idx2.getParents(100).empty());
    EXPECT_EQ(idx2.childCount(), 0u);
}

TEST_F(LineageWALTest, CheckpointAndRecover) {
    {
        LineageIndex idx(test_dir_);
        idx.recordLineage(10, {1}, LineageOp::Summarize);
        idx.recordLineage(20, {1, 2}, LineageOp::Aggregate);
        idx.recordLineage(30, {2}, LineageOp::Infer);
        idx.markRemoved(2);

        ASSERT_TRUE(idx.checkpoint());
    }

    // After checkpoint, WAL is truncated — recovery is from snapshot only
    LineageIndex idx2(test_dir_);
    EXPECT_EQ(idx2.childCount(), 3u);

    auto p10 = idx2.getParents(10);
    ASSERT_EQ(p10.size(), 1u);
    EXPECT_EQ(p10[0], 1u);

    auto p20 = idx2.getParents(20);
    ASSERT_EQ(p20.size(), 2u);

    EXPECT_FALSE(idx2.hasLiveParents(30));
    EXPECT_TRUE(idx2.hasLiveParents(10));
}

TEST_F(LineageWALTest, PostCheckpointWALAppends) {
    {
        LineageIndex idx(test_dir_);
        idx.recordLineage(10, {1}, LineageOp::Summarize);
        ASSERT_TRUE(idx.checkpoint());

        // Operations after checkpoint go to new WAL
        idx.recordLineage(20, {1, 2}, LineageOp::Aggregate);
        idx.markRemoved(1);
    }

    LineageIndex idx2(test_dir_);
    EXPECT_EQ(idx2.childCount(), 2u);
    // node 10's only parent is 1, which is removed
    EXPECT_FALSE(idx2.hasLiveParents(10));
    // node 20 has parents [1, 2]; 1 is removed but 2 is alive
    EXPECT_TRUE(idx2.hasLiveParents(20));
}

TEST_F(LineageWALTest, EmptyIndexNoError) {
    {
        LineageIndex idx(test_dir_);
    }

    LineageIndex idx2(test_dir_);
    EXPECT_EQ(idx2.childCount(), 0u);
    EXPECT_EQ(idx2.parentCount(), 0u);
}

TEST_F(LineageWALTest, WalSizeGrowsWithOperations) {
    LineageIndex idx(test_dir_);
    EXPECT_EQ(idx.walSize(), 0u);

    idx.recordLineage(100, {1, 2, 3}, LineageOp::Summarize);
    EXPECT_GT(idx.walSize(), 0u);

    size_t size_after_one = idx.walSize();
    idx.recordLineage(200, {4, 5}, LineageOp::Infer);
    EXPECT_GT(idx.walSize(), size_after_one);
}

TEST_F(LineageWALTest, ReverseIndexRestoredCorrectly) {
    {
        LineageIndex idx(test_dir_);
        idx.recordLineage(10, {1}, LineageOp::Summarize);
        idx.recordLineage(20, {1}, LineageOp::Aggregate);
        idx.recordLineage(30, {1, 2}, LineageOp::Infer);
    }

    LineageIndex idx2(test_dir_);
    auto children_1 = idx2.getChildren(1);
    EXPECT_EQ(children_1.size(), 3u);

    auto children_2 = idx2.getChildren(2);
    EXPECT_EQ(children_2.size(), 1u);
}
