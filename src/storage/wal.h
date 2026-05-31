#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace amind {

/// Type of WAL entry — determines the operation to replay.
enum class WalEntryType : uint8_t {
    Put = 1,     // Insert or update a record
    Delete = 2,  // Delete a record by memory_id
};

/// A single WAL entry on disk.
///
/// Binary layout:
///   [0..4)   length:  u32 — size of (type + payload), NOT including length/crc fields
///   [4..5)   type:    u8  — WalEntryType
///   [5..5+N) payload: [u8; N] where N = length - 1
///   [5+N..9+N) crc32: u32 — CRC32C over bytes [0..5+N) (length + type + payload)
///
/// Total on-disk size = 4 (length) + length + 4 (crc) = length + 8
struct WalEntry {
    WalEntryType type{WalEntryType::Put};
    std::vector<uint8_t> payload;

    /// Compute the total on-disk size of this entry.
    [[nodiscard]] size_t diskSize() const;
};

/// Write-Ahead Log for crash recovery.
///
/// The WAL is an append-only file. Every mutation (put/delete) is first written
/// to the WAL before being applied to the in-memory data structure (MemTable).
/// On crash, the WAL is replayed to reconstruct the MemTable.
///
/// Features:
///   - Append entries with CRC32 integrity check
///   - Read all entries (for recovery)
///   - Rotate: close current WAL, start a new one
///   - Sync: fsync to guarantee durability
///   - Thread-safe (mutex-protected)
class WriteAheadLog {
public:
    /// Create or open a WAL file at the given path.
    /// If the file exists, it is opened for appending.
    /// If it does not exist, it is created.
    /// Throws std::runtime_error on I/O failure.
    explicit WriteAheadLog(const std::filesystem::path& filepath);

    /// Destructor — closes the file descriptor.
    ~WriteAheadLog();

    // Non-copyable, non-movable (owns a file descriptor)
    WriteAheadLog(const WriteAheadLog&) = delete;
    WriteAheadLog& operator=(const WriteAheadLog&) = delete;
    WriteAheadLog(WriteAheadLog&&) = delete;
    WriteAheadLog& operator=(WriteAheadLog&&) = delete;

    /// Append a Put entry to the WAL.
    /// The payload is typically a serialized MemoryRecord.
    /// Calls fsync after writing to guarantee durability.
    void appendPut(std::span<const uint8_t> payload);

    /// Append a Delete entry to the WAL.
    /// The payload is typically the 8-byte memory_id.
    void appendDelete(std::span<const uint8_t> payload);

    /// Read all valid entries from the WAL file.
    /// Stops at the first corrupted entry (partial write from crash).
    /// Returns entries in the order they were written.
    [[nodiscard]] std::vector<WalEntry> readAll() const;

    /// Rotate the WAL: close the current file and start a new empty one.
    /// The old file is renamed to <filepath>.old.<sequence_number>.
    /// Returns the path of the archived old WAL file.
    std::filesystem::path rotate();

    /// Force fsync to flush OS buffers to disk.
    void sync();

    /// Get the current file size in bytes.
    [[nodiscard]] size_t fileSize() const;

    /// Get the current WAL file path.
    [[nodiscard]] const std::filesystem::path& filepath() const;

    /// Get the number of entries written since the WAL was opened/rotated.
    [[nodiscard]] uint64_t entryCount() const;

private:
    /// Append a raw entry (type + payload) to the WAL file.
    void appendEntry(WalEntryType type, std::span<const uint8_t> payload);

    std::filesystem::path filepath_;
    int fd_{-1};                  // POSIX file descriptor
    uint64_t entry_count_{0};     // Entries written since open/rotate
    uint64_t rotation_seq_{0};    // Rotation sequence counter
    mutable std::mutex mutex_;    // Thread safety
};

}  // namespace amind
