# ── amind Provider Library ────────────────────────────────────────────────
add_library(amind_provider
    provider.cpp
    ollama_provider.cpp
    openai_provider.cpp
    anthropic_provider.cpp
    fallback_provider.cpp
    connection_pool.cpp
)

target_include_directories(amind_provider
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(amind_provider
    PUBLIC
        amind_core
        nlohmann_json::nlohmann_json
        spdlog::spdlog
)

# TLS/HTTPS support via OpenSSL
if(AMIND_ENABLE_TLS)
    target_compile_definitions(amind_provider PUBLIC AMIND_TLS_ENABLED)
    target_link_libraries(amind_provider PUBLIC OpenSSL::SSL OpenSSL::Crypto)
endif()
