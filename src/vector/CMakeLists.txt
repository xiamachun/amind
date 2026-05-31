# ── amind Vector Library ──────────────────────────────────────────────────
add_library(amind_vector
    hnsw_index.cpp
    cold_tier.cpp
    recall_scorer.cpp
)

target_include_directories(amind_vector
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(amind_vector
    PUBLIC
        amind_core
        spdlog::spdlog
)

