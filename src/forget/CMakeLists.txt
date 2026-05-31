
add_library(amind_forget STATIC
    forget_engine.cpp
    forget_log.cpp
)

target_include_directories(amind_forget
    PUBLIC ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(amind_forget
    PUBLIC
        amind_core
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)
