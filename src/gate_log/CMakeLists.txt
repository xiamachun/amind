add_library(amind_gate_log STATIC
    gate_log.cpp
)

target_include_directories(amind_gate_log
    PUBLIC ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(amind_gate_log
    PUBLIC
        amind_core
        amind_gate
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)
