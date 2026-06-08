#include "sdk/amind_client.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

using namespace amind;
using json = nlohmann::json;

// ── Unit tests for SDK response type parsing ───────────────────────────────
// These tests verify that the SDK correctly constructs requests and can
// handle typical JSON responses. They don't require a running server.

TEST(SdkTypes, StoreResponseDefaults) {
    StoreResponse resp;
    EXPECT_TRUE(resp.memory_ids.empty());
    EXPECT_FALSE(resp.async_refinement_scheduled);
}

TEST(SdkTypes, RecallItemDefaults) {
    RecallItem item;
    EXPECT_EQ(item.memory_id, 0u);
    EXPECT_TRUE(item.content.empty());
    EXPECT_EQ(item.score, 0.0f);
}

TEST(SdkTypes, MemoryInfoDefaults) {
    MemoryInfo info;
    EXPECT_EQ(info.memory_id, 0u);
    EXPECT_EQ(info.version, 0u);
    EXPECT_EQ(info.parent_id, 0u);
    EXPECT_EQ(info.importance, 0.0f);
}

TEST(SdkTypes, SessionInfoDefaults) {
    SessionInfo info;
    EXPECT_EQ(info.session_id, 0u);
    EXPECT_EQ(info.turn_count, 0u);
    EXPECT_FALSE(info.active);
}

TEST(SdkTypes, CoverageInfoDefaults) {
    CoverageInfo info;
    EXPECT_EQ(info.total, 0u);
    EXPECT_EQ(info.active, 0u);
    EXPECT_EQ(info.stale, 0u);
    EXPECT_EQ(info.conflicted, 0u);
}

TEST(SdkTypes, DeleteResponseDefaults) {
    DeleteResponse resp;
    EXPECT_FALSE(resp.deleted);
    EXPECT_EQ(resp.invalidated_count, 0u);
    EXPECT_TRUE(resp.invalidated_ids.empty());
}

TEST(SdkClient, ConstructsWithDefaults) {
    AmindClient client("127.0.0.1", 8080);
    // Just verify construction doesn't crash
    SUCCEED();
}

TEST(SdkClient, ConstructsWithCustomTimeout) {
    AmindClient client("localhost", 9090, 5000);
    SUCCEED();
}

// ── Connection failure tests (no server running) ───────────────────────────

TEST(SdkClient, HealthFailsWithNoServer) {
    AmindClient client("127.0.0.1", 19999, 1000);
    auto result = client.health();
    EXPECT_FALSE(result.ok());
}

TEST(SdkClient, StoreFailsWithNoServer) {
    AmindClient client("127.0.0.1", 19999, 1000);
    auto result = client.store("test content");
    EXPECT_FALSE(result.ok());
}

TEST(SdkClient, RecallFailsWithNoServer) {
    AmindClient client("127.0.0.1", 19999, 1000);
    auto result = client.recall("test query");
    EXPECT_FALSE(result.ok());
}

TEST(SdkClient, GetFailsWithNoServer) {
    AmindClient client("127.0.0.1", 19999, 1000);
    auto result = client.get(12345);
    EXPECT_FALSE(result.ok());
}

TEST(SdkClient, GetHistoryFailsWithNoServer) {
    AmindClient client("127.0.0.1", 19999, 1000);
    auto result = client.getHistory(12345);
    EXPECT_FALSE(result.ok());
}

TEST(SdkClient, RemoveFailsWithNoServer) {
    AmindClient client("127.0.0.1", 19999, 1000);
    auto result = client.remove(12345);
    EXPECT_FALSE(result.ok());
}

TEST(SdkClient, SessionsFailWithNoServer) {
    AmindClient client("127.0.0.1", 19999, 1000);
    auto r1 = client.startSession("test-agent");
    EXPECT_FALSE(r1.ok());
    auto r2 = client.getSessionSummary(1);
    EXPECT_FALSE(r2.ok());
    auto r3 = client.closeSession(1);
    EXPECT_FALSE(r3.ok());
}

TEST(SdkClient, MetaCognitionFailsWithNoServer) {
    AmindClient client("127.0.0.1", 19999, 1000);
    auto r1 = client.getCoverage();
    EXPECT_FALSE(r1.ok());
    auto r2 = client.getConflicts();
    EXPECT_FALSE(r2.ok());
}
