# ── amind Memory Library ──────────────────────────────────────────────────
add_library(amind_memory
    memory_store.cpp
    backup.cpp
)

target_include_directories(amind_memory
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(amind_memory
    PUBLIC
        amind_core
        amind_lsm
        amind_vector
        amind_graph
        nlohmann_json::nlohmann_json
        spdlog::spdlog
)
