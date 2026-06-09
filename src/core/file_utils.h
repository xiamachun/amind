#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <string>

namespace amind {

/// Fsync a file by path: open → fsync → close.
/// Silently ignores failures (matching original behavior across WAL modules).
inline void fsyncFile(const char* path) {
    int fd = ::open(path, O_WRONLY);
    if (fd >= 0) {
        ::fsync(fd);
        ::close(fd);
    }
}

inline void fsyncFile(const std::string& path) {
    fsyncFile(path.c_str());
}

} // namespace amind
