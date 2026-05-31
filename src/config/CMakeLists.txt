
add_library(amind_config STATIC
    v2_config.cpp
)

target_include_directories(amind_config
    PUBLIC ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(amind_config
    PUBLIC
        spdlog::spdlog
)
