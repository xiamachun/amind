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

# Enable SIMD instructions for distance computation
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|i[3-6]86)")
    target_compile_options(amind_vector PRIVATE -msse3 -msse4.1)
endif()
