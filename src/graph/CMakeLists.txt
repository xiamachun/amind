# ── amind Graph Library ───────────────────────────────────────────────────
add_library(amind_graph
    graph_store.cpp
    graph_wal.cpp
)

target_include_directories(amind_graph
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(amind_graph
    PUBLIC
        amind_core
        spdlog::spdlog
)
