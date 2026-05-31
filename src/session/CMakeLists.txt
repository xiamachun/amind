# ── amind Session Library ─────────────────────────────────────────────────
add_library(amind_session
    session_manager.cpp
    session_wal.cpp
)

target_include_directories(amind_session
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(amind_session
    PUBLIC
        amind_core
        amind_memory
        amind_capture
        amind_provider
        nlohmann_json::nlohmann_json
        spdlog::spdlog
)
