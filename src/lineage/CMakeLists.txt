
add_library(amind_lineage STATIC
    lineage_index.cpp
    lineage_wal.cpp
    propagator.cpp
)

target_include_directories(amind_lineage
    PUBLIC ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(amind_lineage
    PUBLIC
        amind_core
        spdlog::spdlog
)
