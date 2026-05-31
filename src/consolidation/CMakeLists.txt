
add_library(amind_consolidation STATIC
    consolidation_worker.cpp
)

target_include_directories(amind_consolidation
    PUBLIC ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(amind_consolidation
    PUBLIC
        amind_core
        spdlog::spdlog
)
