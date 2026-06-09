# ── amind Core Library ────────────────────────────────────────────────────
add_library(amind_core
    memory_record.cpp
    snowflake.cpp
)

target_include_directories(amind_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(amind_core
    PUBLIC
        xxHash
        crc32c
        msgpack-cxx
        nlohmann_json::nlohmann_json
        spdlog::spdlog
)
