#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>

namespace amind {

/// Read-only memory-mapped file.
///
/// Uses POSIX mmap() to map a file into the process's virtual address space.
/// This enables zero-copy reads — the OS handles paging data in/out of RAM
/// transparently, and the data can be accessed as a regular byte array.
///
/// Benefits:
///   - Zero-copy: no need to read() into a buffer
///   - OS manages caching via the page cache
///   - Multiple processes can share the same physical pages
///   - Random access is as fast as sequential access (no seek overhead)
///
/// Thread-safe for concurrent reads (the mapped memory is read-only).
class MmapFile {
public:
    /// Open and memory-map a file for reading.
    /// Throws std::runtime_error if the file cannot be opened or mapped.
    /// Set sequential=false for random-access patterns (e.g. SSTable point lookups).
    explicit MmapFile(const std::filesystem::path& filepath, bool sequential = true);

    /// Destructor — unmaps the file.
    ~MmapFile();

    // Non-copyable (owns a mapping), but movable
    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;
    MmapFile(MmapFile&& other) noexcept;
    MmapFile& operator=(MmapFile&& other) noexcept;

    /// Get a read-only view of the entire file contents.
    [[nodiscard]] std::span<const uint8_t> data() const;

    /// Get a read-only view of a sub-range of the file.
    /// Throws std::out_of_range if offset + length > fileSize().
    [[nodiscard]] std::span<const uint8_t> slice(size_t offset, size_t length) const;

    /// Get the file size in bytes.
    [[nodiscard]] size_t fileSize() const;

    /// Get the file path.
    [[nodiscard]] const std::filesystem::path& filepath() const;

    /// Check if the file is currently mapped.
    [[nodiscard]] bool isMapped() const;

private:
    void unmap();

    std::filesystem::path filepath_;
    uint8_t* mapped_data_{nullptr};
    size_t file_size_{0};
};

}  // namespace amind
