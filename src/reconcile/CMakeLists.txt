add_library(amind_reconcile STATIC
    reconciler.cpp
)

target_include_directories(amind_reconcile
    PUBLIC ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(amind_reconcile
    PUBLIC
        amind_core
        amind_provider
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)
