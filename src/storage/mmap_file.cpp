#include "mmap_file.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace amind {

MmapFile::MmapFile(const std::filesystem::path& filepath, bool sequential)
    : filepath_(filepath) {
    // Open file for reading
    int fd = ::open(filepath_.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(
            "MmapFile: failed to open file: " + filepath_.string() +
            " — " + std::strerror(errno));
    }

    // Get file size
    struct stat file_stat{};
    if (::fstat(fd, &file_stat) != 0) {
        ::close(fd);
        throw std::runtime_error(
            "MmapFile: failed to stat file: " + filepath_.string() +
            " — " + std::strerror(errno));
    }
    file_size_ = static_cast<size_t>(file_stat.st_size);

    if (file_size_ == 0) {
        // Empty file — nothing to map
        ::close(fd);
        return;
    }

    // Memory-map the file (read-only, private mapping)
    void* mapped = ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        ::close(fd);
        throw std::runtime_error(
            "MmapFile: mmap failed for file: " + filepath_.string() +
            " — " + std::strerror(errno));
    }

    mapped_data_ = static_cast<uint8_t*>(mapped);

    ::madvise(mapped_data_, file_size_, sequential ? MADV_SEQUENTIAL : MADV_RANDOM);

    // Close the fd — the mapping remains valid after close
    ::close(fd);
}

MmapFile::~MmapFile() {
    unmap();
}

MmapFile::MmapFile(MmapFile&& other) noexcept
    : filepath_(std::move(other.filepath_)),
      mapped_data_(other.mapped_data_),
      file_size_(other.file_size_) {
    other.mapped_data_ = nullptr;
    other.file_size_ = 0;
}

MmapFile& MmapFile::operator=(MmapFile&& other) noexcept {
    if (this != &other) {
        unmap();
        filepath_ = std::move(other.filepath_);
        mapped_data_ = other.mapped_data_;
        file_size_ = other.file_size_;
        other.mapped_data_ = nullptr;
        other.file_size_ = 0;
    }
    return *this;
}

std::span<const uint8_t> MmapFile::data() const {
    if (mapped_data_ == nullptr) {
        return {};
    }
    return {mapped_data_, file_size_};
}

std::span<const uint8_t> MmapFile::slice(size_t offset, size_t length) const {
    if (offset + length > file_size_) {
        throw std::out_of_range(
            "MmapFile::slice: offset=" + std::to_string(offset) +
            " length=" + std::to_string(length) +
            " exceeds file size=" + std::to_string(file_size_));
    }
    return {mapped_data_ + offset, length};
}

size_t MmapFile::fileSize() const {
    return file_size_;
}

const std::filesystem::path& MmapFile::filepath() const {
    return filepath_;
}

bool MmapFile::isMapped() const {
    return mapped_data_ != nullptr;
}

void MmapFile::unmap() {
    if (mapped_data_ != nullptr) {
        ::munmap(mapped_data_, file_size_);
        mapped_data_ = nullptr;
        file_size_ = 0;
    }
}

}  // namespace amind
