add_library(amind_auth
    auth_manager.cpp
)

target_include_directories(amind_auth PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(amind_auth PUBLIC
    amind_memory
    nlohmann_json::nlohmann_json
    OpenSSL::Crypto
    spdlog::spdlog
)
