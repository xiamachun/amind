#include "wal.h"

#include <cerrno>
#include <crc32c/crc32c.h>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace amind {

// ── WalEntry ───────────────────────────────────────────────────────────────────

size_t WalEntry::diskSize() const {
    // 4 (length field) + 1 (type) + payload.size() + 4 (crc)
    return 4 + 1 + payload.size() + 4;
}

// ── WriteAheadLog ──────────────────────────────────────────────────────────────

WriteAheadLog::WriteAheadLog(const std::filesystem::path& filepath)
    : filepath_(filepath) {
    // Create parent directories if they don't exist
    if (filepath_.has_parent_path()) {
        std::filesystem::create_directories(filepath_.parent_path());
    }

    // Open file for append (create if not exists)
    fd_ = ::open(filepath_.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd_ < 0) {
        throw std::runtime_error(
            "Failed to open WAL file: " + filepath_.string() +
            " — " + std::strerror(errno));
    }
}

WriteAheadLog::~WriteAheadLog() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

void WriteAheadLog::appendPut(std::span<const uint8_t> payload) {
    appendEntry(WalEntryType::Put, payload);
}

void WriteAheadLog::appendDelete(std::span<const uint8_t> payload) {
    appendEntry(WalEntryType::Delete, payload);
}

void WriteAheadLog::appendEntry(WalEntryType type, std::span<const uint8_t> payload) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Build the on-disk entry:
    //   [length: u32] [type: u8] [payload: N bytes] [crc32: u32]
    // where length = 1 (type) + payload.size()

    uint32_t length = static_cast<uint32_t>(1 + payload.size());

    // Assemble the bytes that CRC covers: length + type + payload
    size_t crc_region_size = sizeof(uint32_t) + sizeof(uint8_t) + payload.size();
    std::vector<uint8_t> buffer(crc_region_size + sizeof(uint32_t));  // + crc trailer

    size_t offset = 0;

    // Write length (little-endian, native on x86/ARM)
    std::memcpy(buffer.data() + offset, &length, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    // Write type
    auto type_byte = static_cast<uint8_t>(type);
    buffer[offset] = type_byte;
    offset += sizeof(uint8_t);

    // Write payload
    if (!payload.empty()) {
        std::memcpy(buffer.data() + offset, payload.data(), payload.size());
        offset += payload.size();
    }

    // Compute CRC32 over (length + type + payload)
    uint32_t checksum = crc32c::Crc32c(buffer.data(), offset);
    std::memcpy(buffer.data() + offset, &checksum, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    // Write entire entry atomically (single write call)
    ssize_t written = ::write(fd_, buffer.data(), offset);
    if (written < 0 || static_cast<size_t>(written) != offset) {
        throw std::runtime_error(
            "WAL write failed: " + std::string(std::strerror(errno)));
    }

    // fsync to guarantee durability
    if (::fsync(fd_) != 0) {
        throw std::runtime_error(
            "WAL fsync failed: " + std::string(std::strerror(errno)));
    }

    ++entry_count_;
}

std::vector<WalEntry> WriteAheadLog::readAll() const {
    std::lock_guard<std::mutex> lock(mutex_);

    // Open file for reading (separate fd to not disturb the write fd)
    int read_fd = ::open(filepath_.c_str(), O_RDONLY);
    if (read_fd < 0) {
        if (errno == ENOENT) {
            return {};  // File doesn't exist yet — no entries
        }
        throw std::runtime_error(
            "Failed to open WAL for reading: " + filepath_.string() +
            " — " + std::strerror(errno));
    }

    // Get file size
    struct stat file_stat{};
    if (::fstat(read_fd, &file_stat) != 0) {
        ::close(read_fd);
        throw std::runtime_error(
            "Failed to stat WAL file: " + std::string(std::strerror(errno)));
    }
    size_t file_size = static_cast<size_t>(file_stat.st_size);

    // Read entire file into memory
    std::vector<uint8_t> file_data(file_size);
    if (file_size > 0) {
        ssize_t bytes_read = ::pread(read_fd, file_data.data(), file_size, 0);
        if (bytes_read < 0 || static_cast<size_t>(bytes_read) != file_size) {
            ::close(read_fd);
            throw std::runtime_error(
                "Failed to read WAL file: " + std::string(std::strerror(errno)));
        }
    }
    ::close(read_fd);

    // Parse entries
    std::vector<WalEntry> entries;
    size_t offset = 0;

    while (offset + sizeof(uint32_t) <= file_size) {
        // Read length
        uint32_t length = 0;
        std::memcpy(&length, file_data.data() + offset, sizeof(uint32_t));

        // Sanity check: length must be at least 1 (type byte)
        if (length < 1) {
            break;  // Corrupted — stop reading
        }

        // Check if we have enough bytes for the full entry
        size_t entry_total = sizeof(uint32_t) + length + sizeof(uint32_t);  // length + body + crc
        if (offset + entry_total > file_size) {
            break;  // Partial write from crash — stop reading
        }

        // Verify CRC32
        size_t crc_region_size = sizeof(uint32_t) + length;  // length field + body
        uint32_t computed_crc = crc32c::Crc32c(file_data.data() + offset, crc_region_size);

        uint32_t stored_crc = 0;
        std::memcpy(&stored_crc, file_data.data() + offset + crc_region_size, sizeof(uint32_t));

        if (computed_crc != stored_crc) {
            break;  // CRC mismatch — corrupted entry, stop reading
        }

        // Parse the entry
        WalEntry entry;
        size_t body_offset = offset + sizeof(uint32_t);

        // Read type
        entry.type = static_cast<WalEntryType>(file_data[body_offset]);
        body_offset += sizeof(uint8_t);

        // Read payload (length - 1 bytes, since type takes 1 byte)
        size_t payload_size = length - 1;
        if (payload_size > 0) {
            entry.payload.assign(
                file_data.data() + body_offset,
                file_data.data() + body_offset + payload_size);
        }

        entries.push_back(std::move(entry));
        offset += entry_total;
    }

    return entries;
}

std::filesystem::path WriteAheadLog::rotate() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Close current file
    if (fd_ >= 0) {
        ::fsync(fd_);
        ::close(fd_);
        fd_ = -1;
    }

    // Rename current WAL to archived name
    ++rotation_seq_;
    std::filesystem::path archived_path =
        filepath_.string() + ".old." + std::to_string(rotation_seq_);

    std::error_code error_code;
    std::filesystem::rename(filepath_, archived_path, error_code);
    if (error_code) {
        throw std::runtime_error(
            "Failed to rotate WAL: " + error_code.message());
    }

    // Open new WAL file
    fd_ = ::open(filepath_.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd_ < 0) {
        throw std::runtime_error(
            "Failed to create new WAL after rotation: " +
            filepath_.string() + " — " + std::strerror(errno));
    }

    entry_count_ = 0;
    return archived_path;
}

void WriteAheadLog::sync() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (fd_ >= 0) {
        if (::fsync(fd_) != 0) {
            throw std::runtime_error(
                "WAL sync failed: " + std::string(std::strerror(errno)));
        }
    }
}

size_t WriteAheadLog::fileSize() const {
    std::lock_guard<std::mutex> lock(mutex_);
    struct stat file_stat{};
    if (::fstat(fd_, &file_stat) != 0) {
        return 0;
    }
    return static_cast<size_t>(file_stat.st_size);
}

const std::filesystem::path& WriteAheadLog::filepath() const {
    return filepath_;
}

uint64_t WriteAheadLog::entryCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entry_count_;
}

}  // namespace amind
