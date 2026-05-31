# ── amind Capture Library ─────────────────────────────────────────────────
add_library(amind_capture
    capture_pipeline.cpp
    derived_extractor.cpp
)

target_include_directories(amind_capture
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(amind_capture
    PUBLIC
        amind_core
        amind_memory
        amind_graph
        amind_provider
        amind_async
        amind_gate
        amind_gate_log
        amind_reconcile
        amind_lineage
        amind_config
        nlohmann_json::nlohmann_json
        spdlog::spdlog
)
