# ── amind Server Library ──────────────────────────────────────────────────
add_library(amind_server
    config.cpp
    engine.cpp
    rest_server.cpp
    variable_manager.cpp
)

target_include_directories(amind_server
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(amind_server
    PUBLIC
        amind_core
        amind_observability
        amind_memory
        amind_graph
        amind_async
        amind_provider
        amind_capture
        amind_retrieval
        amind_metacognition
        amind_session
        amind_gate
        amind_lineage
        amind_forget
        amind_conflict
        amind_coordinator
        amind_consolidation
        amind_reconcile
        amind_auth
        amind_config
        nlohmann_json::nlohmann_json
        spdlog::spdlog
)
