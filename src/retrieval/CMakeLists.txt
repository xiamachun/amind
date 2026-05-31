# ── amind Retrieval Library ───────────────────────────────────────────────
add_library(amind_retrieval
    retrieval.cpp
    staleness_filter.cpp
)

target_include_directories(amind_retrieval
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(amind_retrieval
    PUBLIC
        amind_core
        amind_memory
        amind_graph
        amind_provider
        nlohmann_json::nlohmann_json
        spdlog::spdlog
)
