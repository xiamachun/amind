
add_library(amind_conflict STATIC
    conflict_resolver.cpp
)

target_include_directories(amind_conflict
    PUBLIC ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(amind_conflict
    PUBLIC
        amind_core
        spdlog::spdlog
)
