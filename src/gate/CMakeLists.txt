
add_library(amind_gate STATIC
    write_gate.cpp
)

target_include_directories(amind_gate
    PUBLIC ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(amind_gate
    PUBLIC
        amind_core
        spdlog::spdlog
)
