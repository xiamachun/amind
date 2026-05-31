# ── amind LSM Library ─────────────────────────────────────────────────────
add_library(amind_lsm
    block.cpp
    bloom_filter.cpp
    memtable.cpp
    sstable_reader.cpp
    sstable_writer.cpp
    manifest.cpp
    lsm_wal.cpp
    lsm_engine.cpp
)

target_include_directories(amind_lsm
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(amind_lsm
    PUBLIC
        amind_core
        amind_storage
        xxHash
        lz4_static
        crc32c
        nlohmann_json::nlohmann_json
)
