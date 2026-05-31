
add_library(amind_coordinator STATIC
    remove_coordinator.cpp
)

target_include_directories(amind_coordinator
    PUBLIC ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(amind_coordinator
    PUBLIC
        amind_core
        amind_lineage
        amind_forget
        spdlog::spdlog
)
