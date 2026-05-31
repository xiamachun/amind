# ── amind Storage Library ─────────────────────────────────────────────────
add_library(amind_storage
    wal.cpp
    page_cache.cpp
    mmap_file.cpp
    recovery.cpp
)

target_include_directories(amind_storage
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(amind_storage
    PUBLIC
        amind_core
)
