# ── CMake toolchain file for cross-compiling to Linux ARM64 (aarch64) ───────
#
# Usage:
#   cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64.cmake
#
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Cross-compiler
set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Search paths: cross-compile sysroot + multiarch lib paths
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)

# OpenSSL: installed via libssl-dev:arm64 to multiarch path
set(OPENSSL_ROOT_DIR "/usr")
set(OPENSSL_CRYPTO_LIBRARY "/usr/lib/aarch64-linux-gnu/libcrypto.so")
set(OPENSSL_SSL_LIBRARY "/usr/lib/aarch64-linux-gnu/libssl.so")
set(OPENSSL_INCLUDE_DIR "/usr/include")

# Library search paths for multiarch
list(APPEND CMAKE_LIBRARY_PATH "/usr/lib/aarch64-linux-gnu")
list(APPEND CMAKE_INCLUDE_PATH "/usr/include/aarch64-linux-gnu")

# Ensure pkg-config also searches arm64 paths
set(ENV{PKG_CONFIG_PATH} "/usr/lib/aarch64-linux-gnu/pkgconfig")
