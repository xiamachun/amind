# ── amind WebUI Server ────────────────────────────────────────────────────
add_library(amind_web
    web_server.cpp
)

target_include_directories(amind_web
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(amind_web
    PUBLIC
        amind_core
        spdlog::spdlog
)
