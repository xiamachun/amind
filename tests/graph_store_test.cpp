#include <gtest/gtest.h>

#include "graph/graph_store.h"

#include <filesystem>

using namespace amind;

class GraphStoreTest : public ::testing::Test {
protected:
    std::filesystem::path test_dir_;
    std::unique_ptr<GraphStore> graph_;

    void SetUp() override {
        test_dir_ = std::filesystem::temp_directory_path() / "amind_graph_test";
        std::filesystem::remove_all(test_dir_);
        std::filesystem::create_directories(test_dir_);
        graph_ = std::make_unique<GraphStore>(test_dir_.string());
        graph_->recover();
    }

    void TearDown() override {
        graph_.reset();
        std::filesystem::remove_all(test_dir_);
    }
};

TEST_F(GraphStoreTest, AddAndGetEdges) {
    graph_->addEdge(1, 2, EdgeType::Related, 0.8f);
    graph_->addEdge(1, 3, EdgeType::Contradicts, 0.5f);

    auto edges = graph_->getEdges(1);
    EXPECT_GE(edges.size(), 2u);
}

TEST_F(GraphStoreTest, UndirectedEdgesCreateBothDirections) {
    graph_->addEdge(10, 20, EdgeType::Related, 1.0f);

    auto from10 = graph_->getEdges(10);
    auto from20 = graph_->getEdges(20);

    bool found_10_to_20 = false;
    for (const auto& e : from10) {
        if (e.to_id == 20) found_10_to_20 = true;
    }
    bool found_20_to_10 = false;
    for (const auto& e : from20) {
        if (e.to_id == 10) found_20_to_10 = true;
    }

    EXPECT_TRUE(found_10_to_20);
    EXPECT_TRUE(found_20_to_10);
}

TEST_F(GraphStoreTest, DirectedEdgeOnlyOneDirection) {
    graph_->addEdge(1, 2, EdgeType::Contradicts, 0.5f);

    auto from1 = graph_->getEdges(1);
    auto from2 = graph_->getEdges(2);

    bool found_in_1 = false;
    for (const auto& e : from1) {
        if (e.to_id == 2 && e.type == EdgeType::Contradicts) found_in_1 = true;
    }
    EXPECT_TRUE(found_in_1);

    bool found_in_2 = false;
    for (const auto& e : from2) {
        if (e.to_id == 1 && e.type == EdgeType::Contradicts) found_in_2 = true;
    }
    EXPECT_FALSE(found_in_2);
}

TEST_F(GraphStoreTest, GetNeighbors) {
    graph_->addEdge(1, 2, EdgeType::Related);
    graph_->addEdge(1, 3, EdgeType::Temporal);
    graph_->addEdge(1, 4, EdgeType::Related);

    auto all_neighbors = graph_->getNeighbors(1);
    EXPECT_EQ(all_neighbors.size(), 3u);

    auto related_only = graph_->getNeighbors(1, EdgeType::Related);
    EXPECT_EQ(related_only.size(), 2u);
}

TEST_F(GraphStoreTest, RemoveNode) {
    graph_->addEdge(1, 2, EdgeType::Related);
    graph_->addEdge(3, 1, EdgeType::Contradicts);

    graph_->removeNode(1);

    auto edges_1 = graph_->getEdges(1);
    EXPECT_TRUE(edges_1.empty());

    // Edge from 3→1 should be removed
    auto edges_3 = graph_->getEdges(3);
    for (const auto& e : edges_3) {
        EXPECT_NE(e.to_id, 1u);
    }
}

TEST_F(GraphStoreTest, EdgeDeduplication) {
    graph_->addEdge(1, 2, EdgeType::Related, 0.5f);
    graph_->addEdge(1, 2, EdgeType::Related, 0.9f);  // duplicate, should update

    auto edges = graph_->getEdges(1);
    int related_to_2 = 0;
    float weight = 0.0f;
    for (const auto& e : edges) {
        if (e.to_id == 2 && e.type == EdgeType::Related) {
            related_to_2++;
            weight = e.weight;
        }
    }
    EXPECT_EQ(related_to_2, 1);
    EXPECT_FLOAT_EQ(weight, 0.9f);
}

TEST_F(GraphStoreTest, TraverseBFS) {
    graph_->addEdge(1, 2, EdgeType::Related);
    graph_->addEdge(2, 3, EdgeType::Related);
    graph_->addEdge(3, 4, EdgeType::Related);

    auto reachable = graph_->traverse(1, 2);
    EXPECT_GE(reachable.size(), 2u);  // Should reach at least 2 and 3
}

TEST_F(GraphStoreTest, CheckpointAndRecover) {
    graph_->addEdge(100, 200, EdgeType::Entity, 0.7f);
    graph_->addEdge(200, 300, EdgeType::SameSession, 1.0f);
    graph_->checkpoint();

    // Create new graph from same directory
    auto graph2 = std::make_unique<GraphStore>(test_dir_.string());
    graph2->recover();

    auto edges = graph2->getEdges(100);
    bool found = false;
    for (const auto& e : edges) {
        if (e.to_id == 200 && e.type == EdgeType::Entity) found = true;
    }
    EXPECT_TRUE(found);
}

TEST_F(GraphStoreTest, EdgeAndNodeCount) {
    EXPECT_EQ(graph_->nodeCount(), 0u);
    graph_->addEdge(1, 2, EdgeType::Related);
    EXPECT_GE(graph_->nodeCount(), 2u);
    EXPECT_GE(graph_->edgeCount(), 1u);
}
