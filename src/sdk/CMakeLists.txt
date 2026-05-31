# ── amind SDK Library ─────────────────────────────────────────────────────
add_library(amind_sdk
    amind_client.cpp
)

target_include_directories(amind_sdk
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(amind_sdk
    PUBLIC
        amind_core
        amind_provider
        nlohmann_json::nlohmann_json
        spdlog::spdlog
)
