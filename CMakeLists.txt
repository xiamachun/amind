cmake_minimum_required(VERSION 3.20)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Minimum policy version for compatibility")

project(amind VERSION 0.1.0 LANGUAGES CXX)

# ── C++20 standard ──────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── TLS/HTTPS support ─────────────────────────────────────────────────────
option(AMIND_ENABLE_TLS "Enable TLS/HTTPS support" OFF)
if(AMIND_ENABLE_TLS)
    find_package(OpenSSL REQUIRED)
    message(STATUS "OpenSSL found: ${OPENSSL_VERSION}")
endif()

# ── Compiler warnings ───────────────────────────────────────────────────────
add_compile_options(
    -Wall -Wextra -Wpedantic -Werror
    -Wno-unused-parameter
    -Wshadow
    -Wnon-virtual-dtor
    -Woverloaded-virtual
)

# ── Sanitizers (Debug builds only) ──────────────────────────────────────────
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    option(AMIND_ENABLE_ASAN "Enable AddressSanitizer" ON)
    option(AMIND_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
    if(AMIND_ENABLE_ASAN AND NOT AMIND_ENABLE_TSAN)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    elseif(AMIND_ENABLE_TSAN)
        add_compile_options(-fsanitize=thread)
        add_link_options(-fsanitize=thread)
    endif()
endif()

# ── System dependencies ────────────────────────────────────────────────────
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# Auto-detect library prefix:
#   macOS (Homebrew): /usr/local (Intel) or /opt/homebrew (Apple Silicon)
#   Linux: /usr (system packages via apt/yum)
# Override with: cmake .. -DBREW_PREFIX=/your/path
if(NOT DEFINED BREW_PREFIX)
    if(APPLE)
        execute_process(
            COMMAND brew --prefix
            OUTPUT_VARIABLE BREW_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(NOT BREW_PREFIX)
            set(BREW_PREFIX "/usr/local")
        endif()
    else()
        set(BREW_PREFIX "/usr")
    endif()
endif()
message(STATUS "Library prefix: ${BREW_PREFIX}")

# Google Test
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    list(APPEND CMAKE_PREFIX_PATH "${BREW_PREFIX}")
    find_package(GTest REQUIRED)
endif()
# Ensure gtest headers are findable (Homebrew may not symlink to /usr/local/include).
if(APPLE)
    execute_process(COMMAND brew --prefix googletest
        OUTPUT_VARIABLE _gtest_prefix OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_gtest_prefix AND EXISTS "${_gtest_prefix}/include")
        include_directories("${_gtest_prefix}/include")
    endif()
endif()

# xxHash
add_library(xxHash INTERFACE)
target_include_directories(xxHash INTERFACE ${BREW_PREFIX}/include)
target_link_directories(xxHash INTERFACE ${BREW_PREFIX}/lib)
target_link_libraries(xxHash INTERFACE -lxxhash)

# crc32c
find_package(Crc32c QUIET)
if(NOT Crc32c_FOUND)
    add_library(crc32c INTERFACE)
    target_include_directories(crc32c INTERFACE ${BREW_PREFIX}/include)
    target_link_directories(crc32c INTERFACE ${BREW_PREFIX}/lib)
    target_link_libraries(crc32c INTERFACE -lcrc32c)
endif()

# msgpack-c (header-only)
add_library(msgpack-cxx INTERFACE)
target_include_directories(msgpack-cxx INTERFACE ${BREW_PREFIX}/include)

# nlohmann/json (header-only)
find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
    add_library(nlohmann_json INTERFACE)
    target_include_directories(nlohmann_json INTERFACE ${BREW_PREFIX}/include)
    add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
endif()

# spdlog
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    add_library(spdlog INTERFACE)
    target_include_directories(spdlog INTERFACE ${BREW_PREFIX}/include)
    target_link_directories(spdlog INTERFACE ${BREW_PREFIX}/lib)
    target_link_libraries(spdlog INTERFACE -lspdlog -lfmt)
    add_library(spdlog::spdlog ALIAS spdlog)
endif()

# lz4
if(EXISTS "${BREW_PREFIX}/lib/liblz4.a")
    add_library(lz4_static STATIC IMPORTED GLOBAL)
    set_target_properties(lz4_static PROPERTIES
        IMPORTED_LOCATION "${BREW_PREFIX}/lib/liblz4.a"
        INTERFACE_INCLUDE_DIRECTORIES "${BREW_PREFIX}/include"
    )
else()
    add_library(lz4_static INTERFACE)
    target_link_libraries(lz4_static INTERFACE -llz4)
    target_include_directories(lz4_static INTERFACE ${BREW_PREFIX}/include)
endif()

# Google Benchmark
option(AMIND_BUILD_BENCHMARKS "Build benchmarks" ON)
if(AMIND_BUILD_BENCHMARKS)
    find_package(benchmark QUIET)
    if(NOT benchmark_FOUND)
        add_library(benchmark INTERFACE)
        target_include_directories(benchmark INTERFACE ${BREW_PREFIX}/include)
        target_link_directories(benchmark INTERFACE ${BREW_PREFIX}/lib)
        target_link_libraries(benchmark INTERFACE -lbenchmark -lbenchmark_main)
        add_library(benchmark::benchmark ALIAS benchmark)
        add_library(benchmark::benchmark_main ALIAS benchmark)
    endif()
endif()

# ── Source tree ─────────────────────────────────────────────────────────────
add_subdirectory(src)

# ── Main executable ─────────────────────────────────────────────────────────
add_executable(amind src/main.cpp)
target_link_libraries(amind PRIVATE amind_server amind_web amind_sdk)
if(AMIND_ENABLE_TLS)
    target_compile_definitions(amind PRIVATE AMIND_TLS_ENABLED)
endif()

# ── Tests ───────────────────────────────────────────────────────────────────
option(AMIND_BUILD_TESTS "Build unit tests" ON)
if(AMIND_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(AMIND_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
