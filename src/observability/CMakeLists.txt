add_library(amind_observability STATIC
    memory_event.cpp
    memory_event_log.cpp
)

target_include_directories(amind_observability
    PUBLIC ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(amind_observability
    PUBLIC
        amind_core
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)
