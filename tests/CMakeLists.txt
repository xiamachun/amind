# ── amind Tests ───────────────────────────────────────────────────────────

# ── Core types tests ─────────────────────────────────────────────────────
add_executable(types_test types_test.cpp)
target_include_directories(types_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(types_test PRIVATE GTest::gtest GTest::gtest_main)
add_test(NAME types_test COMMAND types_test)

# ── Memory record tests ─────────────────────────────────────────────────
add_executable(memory_record_test memory_record_test.cpp)
target_include_directories(memory_record_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(memory_record_test PRIVATE
    amind_core
    GTest::gtest GTest::gtest_main
)
add_test(NAME memory_record_test COMMAND memory_record_test)

# ── HNSW V2 tests ───────────────────────────────────────────────────────
add_executable(hnsw_v2_test hnsw_v2_test.cpp)
target_include_directories(hnsw_v2_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(hnsw_v2_test PRIVATE
    amind_vector
    GTest::gtest GTest::gtest_main
)
add_test(NAME hnsw_v2_test COMMAND hnsw_v2_test)

# ── Lineage tests ────────────────────────────────────────────────────────
add_executable(lineage_test lineage_test.cpp)
target_include_directories(lineage_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(lineage_test PRIVATE
    amind_lineage
    GTest::gtest GTest::gtest_main
)
add_test(NAME lineage_test COMMAND lineage_test)

# ── Lineage WAL persistence tests ───────────────────────────────────────
add_executable(lineage_wal_test lineage_wal_test.cpp)
target_include_directories(lineage_wal_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(lineage_wal_test PRIVATE
    amind_lineage
    GTest::gtest GTest::gtest_main
)
add_test(NAME lineage_wal_test COMMAND lineage_wal_test)

# ── WriteGate tests ──────────────────────────────────────────────────────
add_executable(write_gate_test write_gate_test.cpp)
target_include_directories(write_gate_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(write_gate_test PRIVATE
    amind_gate
    GTest::gtest GTest::gtest_main
)
add_test(NAME write_gate_test COMMAND write_gate_test)

# ── DerivedExtractor tests ───────────────────────────────────────────────
add_executable(derived_extractor_test
    derived_extractor_test.cpp
    ${CMAKE_SOURCE_DIR}/src/capture/derived_extractor.cpp
)
target_include_directories(derived_extractor_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(derived_extractor_test PRIVATE
    amind_gate
    amind_lineage
    amind_core
    spdlog::spdlog
    GTest::gtest GTest::gtest_main
)
add_test(NAME derived_extractor_test COMMAND derived_extractor_test)

# ── ForgetEngine tests ───────────────────────────────────────────────────
add_executable(forget_engine_test forget_engine_test.cpp)
target_include_directories(forget_engine_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(forget_engine_test PRIVATE
    amind_forget
    GTest::gtest GTest::gtest_main
)
add_test(NAME forget_engine_test COMMAND forget_engine_test)

# ── ConflictResolver tests ───────────────────────────────────────────────
add_executable(conflict_resolver_test conflict_resolver_test.cpp)
target_include_directories(conflict_resolver_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(conflict_resolver_test PRIVATE
    amind_conflict
    GTest::gtest GTest::gtest_main
)
add_test(NAME conflict_resolver_test COMMAND conflict_resolver_test)

# ── RemoveCoordinator tests ──────────────────────────────────────────────
add_executable(remove_coordinator_test remove_coordinator_test.cpp)
target_include_directories(remove_coordinator_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(remove_coordinator_test PRIVATE
    amind_coordinator
    amind_lineage
    amind_forget
    amind_core
    GTest::gtest GTest::gtest_main
)
add_test(NAME remove_coordinator_test COMMAND remove_coordinator_test)

# ── Consolidation tests ──────────────────────────────────────────────────
add_executable(consolidation_test consolidation_test.cpp)
target_include_directories(consolidation_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(consolidation_test PRIVATE
    amind_consolidation
    amind_core
    GTest::gtest GTest::gtest_main
)
add_test(NAME consolidation_test COMMAND consolidation_test)

# ── V2 Config / FeatureGate tests ────────────────────────────────────────
add_executable(v2_config_test v2_config_test.cpp)
target_include_directories(v2_config_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(v2_config_test PRIVATE
    amind_config
    GTest::gtest GTest::gtest_main
)
add_test(NAME v2_config_test COMMAND v2_config_test)

# ── Retrieval pipeline tests ────────────────────────────────────────────
add_executable(retrieval_test retrieval_test.cpp)
target_include_directories(retrieval_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(retrieval_test PRIVATE
    amind_retrieval
    amind_vector
    amind_core
    GTest::gtest GTest::gtest_main
)
add_test(NAME retrieval_test COMMAND retrieval_test)

# ── V2 End-to-End Integration tests ─────────────────────────────────────
# Disabled in Phase 4: this test queries the legacy ForgetEngine.getLog()
# API which was deleted. Rewriting to query MemoryEventLog is a follow-up.
# Until then, end-to-end coverage of the GC pipeline lives in the eval/
# scripts (stress_test.py, reconcile_test.py).
# add_executable(v2_integration_test
#     v2_integration_test.cpp
#     ${CMAKE_SOURCE_DIR}/src/capture/derived_extractor.cpp
# )
# target_link_libraries(v2_integration_test PRIVATE
#     amind_coordinator amind_consolidation amind_conflict amind_config
#     amind_gate amind_lineage amind_forget amind_core
#     spdlog::spdlog GTest::gtest GTest::gtest_main)
# add_test(NAME v2_integration_test COMMAND v2_integration_test)

# ── SDK Client tests ─────────────────────────────────────────────────────
add_executable(sdk_test sdk_test.cpp)
target_include_directories(sdk_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(sdk_test PRIVATE
    amind_sdk
    GTest::gtest GTest::gtest_main
)
add_test(NAME sdk_test COMMAND sdk_test)

# ── LSM Engine tests ─────────────────────────────────────────────────────
add_executable(lsm_test lsm_test.cpp)
target_include_directories(lsm_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(lsm_test PRIVATE
    amind_lsm
    amind_core
    GTest::gtest GTest::gtest_main
)
add_test(NAME lsm_test COMMAND lsm_test)

# ── Memory Store tests ───────────────────────────────────────────────────
add_executable(memory_store_test memory_store_test.cpp)
target_include_directories(memory_store_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(memory_store_test PRIVATE
    amind_memory
    amind_vector
    amind_lsm
    amind_core
    GTest::gtest GTest::gtest_main
)
add_test(NAME memory_store_test COMMAND memory_store_test)

# ── Tiered Memory tests (三层记忆 + 衰减 + 晋升 + 门控 + FeatureGate) ────
add_executable(tiered_memory_test tiered_memory_test.cpp)
target_include_directories(tiered_memory_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(tiered_memory_test PRIVATE
    amind_memory
    amind_vector
    amind_lsm
    amind_core
    amind_config
    GTest::gtest GTest::gtest_main
)
add_test(NAME tiered_memory_test COMMAND tiered_memory_test)

# ── Graph Store tests ────────────────────────────────────────────────────
add_executable(graph_store_test graph_store_test.cpp)
target_include_directories(graph_store_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(graph_store_test PRIVATE
    amind_graph
    amind_core
    GTest::gtest GTest::gtest_main
)
add_test(NAME graph_store_test COMMAND graph_store_test)

# ── Async (TaskQueue / TaskExecutor) tests ───────────────────────────────
add_executable(async_test async_test.cpp)
target_include_directories(async_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(async_test PRIVATE
    amind_async
    amind_core
    GTest::gtest GTest::gtest_main
)
add_test(NAME async_test COMMAND async_test)

# ── Provider (ConnectionPool) tests ──────────────────────────────────────
add_executable(provider_test provider_test.cpp)
target_include_directories(provider_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(provider_test PRIVATE
    amind_provider
    amind_core
    GTest::gtest GTest::gtest_main
)
add_test(NAME provider_test COMMAND provider_test)

# ── Session tests ────────────────────────────────────────────────────────
add_executable(session_test session_test.cpp)
target_include_directories(session_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(session_test PRIVATE
    amind_session
    amind_capture
    amind_memory
    amind_vector
    amind_lsm
    amind_graph
    amind_async
    amind_provider
    amind_core
    GTest::gtest GTest::gtest_main
)
add_test(NAME session_test COMMAND session_test)

# ── Capture pipeline tests ───────────────────────────────────────────────
add_executable(capture_test capture_test.cpp)
target_include_directories(capture_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(capture_test PRIVATE
    amind_capture
    amind_memory
    amind_vector
    amind_lsm
    amind_graph
    amind_async
    amind_provider
    amind_core
    GTest::gtest GTest::gtest_main
)
add_test(NAME capture_test COMMAND capture_test)

# ── MetaCognition tests ──────────────────────────────────────────────────
add_executable(metacognition_test metacognition_test.cpp)
target_include_directories(metacognition_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(metacognition_test PRIVATE
    amind_metacognition
    amind_memory
    amind_vector
    amind_lsm
    amind_graph
    amind_core
    GTest::gtest GTest::gtest_main
)
add_test(NAME metacognition_test COMMAND metacognition_test)

# ── Aggregate Staleness Filter tests ────────────────────────────────────
add_executable(staleness_filter_test staleness_filter_test.cpp)
target_include_directories(staleness_filter_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(staleness_filter_test PRIVATE
    amind_retrieval
    amind_observability
    amind_core
    spdlog::spdlog
    fmt::fmt
    GTest::gtest GTest::gtest_main
)
add_test(NAME staleness_filter_test COMMAND staleness_filter_test)

# ── Memory Event Log (unified observability) tests ──────────────────────
add_executable(memory_event_log_test memory_event_log_test.cpp)
target_include_directories(memory_event_log_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(memory_event_log_test PRIVATE
    amind_observability
    amind_core
    spdlog::spdlog
    fmt::fmt
    GTest::gtest GTest::gtest_main
)
add_test(NAME memory_event_log_test COMMAND memory_event_log_test)

# ── Crash Consistency tests ─────────────────────────────────────────────
add_executable(crash_consistency_test crash_consistency_test.cpp)
target_include_directories(crash_consistency_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(crash_consistency_test PRIVATE
    amind_memory
    amind_vector
    amind_lsm
    amind_graph
    amind_core
    GTest::gtest GTest::gtest_main
)
add_test(NAME crash_consistency_test COMMAND crash_consistency_test)

# ── Audit Fix Verification tests ────────────────────────────────────────
add_executable(audit_fix_test audit_fix_test.cpp)
target_include_directories(audit_fix_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(audit_fix_test PRIVATE
    amind_server
    amind_session
    amind_lineage
    amind_graph
    amind_vector
    amind_capture
    amind_gate
    amind_core
    GTest::gtest GTest::gtest_main
)
add_test(NAME audit_fix_test COMMAND audit_fix_test)
