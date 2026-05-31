#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <vector>

namespace amind {

// WAL record format constants
constexpr uint16_t WAL_MAGIC = 0x574C;  // 'WL'
constexpr uint16_t WAL_VERSION = 1;
constexpr uint32_t WAL_HEADER_SIZE = 32;

// Record type enumeration
enum class WalRecordType : uint8_t {
    PUT = 0,
    DELETE = 1
};

/// LSM-specific Write-Ahead Log for crash recovery.
///
/// Every put/remove operation is appended to the WAL *before* being applied to
/// the MemTable.  On a clean flush the WAL is truncated.  If the process
/// crashes before flush, the WAL is replayed on the next startup to recover
/// the MemTable contents that were lost.
///
/// On-disk record format (little-endian):
///   Record Header (32 bytes):
///     [0..2)              magic      uint16_t   (0x574C 'WL')
///     [2..4)              version    uint16_t   (1)
///     [4..5)              type       uint8_t    (0=PUT, 1=DELETE)
///     [5..8)              reserved   uint8_t[3]
///     [8..16)             seq        uint64_t   (global sequence number)
///     [16..24)            key        uint64_t   (memory_id)
///     [24..28)            data_len   uint32_t   (0 for tombstone)
///     [28..32)            header_crc uint32_t   (CRC32 over first 28 bytes)
///   Record Data:
///     [0..data_len)       data       bytes
///     [data_len..data_len+4)  data_crc  uint32_t  (CRC32 over data, only if data_len > 0)
///
/// Note: This class is distinct from amind::WriteAheadLog in src/storage/wal.h,
/// which provides a general-purpose WAL with POSIX fd, mutex, and CRC32C.
/// This LSM-specific version uses fstream and a simpler CRC32 implementation.
class LsmWriteAheadLog {
public:
    /// Open (or create) the WAL file at the given path.
    /// maxSegmentSize: maximum size of a WAL segment before rotation (default: 32MB)
    /// maxSegmentAge: maximum age of WAL segments before expiration (default: 7 days in seconds)
    explicit LsmWriteAheadLog(const std::filesystem::path& walPath,
                              uint64_t maxSegmentSize = 32 * 1024 * 1024,
                              uint64_t maxSegmentAge = 7 * 24 * 3600);
    ~LsmWriteAheadLog();

    LsmWriteAheadLog(const LsmWriteAheadLog&) = delete;
    LsmWriteAheadLog& operator=(const LsmWriteAheadLog&) = delete;

    /// Append a put entry.  Calls fsync to guarantee durability unless in batch mode.
    void appendPut(uint64_t seq, uint64_t key, const std::vector<uint8_t>& data);

    /// Append a tombstone (delete) entry.  Calls fsync to guarantee durability unless in batch mode.
    void appendDelete(uint64_t seq, uint64_t key);

    /// Begin a batch write operation.  Subsequent writes will not fsync until endBatch() is called.
    void beginBatch();

    /// End a batch write operation and fsync to guarantee durability.
    void endBatch();

    /// Truncate the WAL (called after a successful MemTable flush).
    void truncate();

    /// Rotate the WAL when current segment size exceeds maxSegmentSize_.
    /// Renames current file to wal.log.timestamp and creates a new empty WAL file.
    void rotate();

    /// Expire old WAL segments that exceed maxSegmentAge_.
    /// Scans the WAL directory and deletes segment files older than the threshold.
    void expireOldSegments();

    /// Get the current segment size in bytes.
    uint64_t currentSegmentSize() const { return currentSize_; }

    /// Get the maximum sequence number replayed from the WAL.
    uint64_t maxReplayedSeq() const { return maxReplayedSeq_; }

    /// Replay all entries in the WAL.
    /// The callback receives (seq, key, data, is_tombstone).
    /// data is empty for tombstones.
    /// @param minSeq  Only replay records with seq > minSeq (for incremental recovery).
    ///                Pass 0 to replay all records.
    /// @return The maximum sequence number encountered during replay.
    using ReplayVisitor = std::function<void(uint64_t seq,
                                            uint64_t key,
                                            std::vector<uint8_t> data,
                                            bool is_tombstone)>;
    static uint64_t replay(const std::filesystem::path& walPath,
                           const ReplayVisitor& visitor,
                           uint64_t minSeq = 0);

private:
    std::filesystem::path path_;
    std::ofstream file_;
    int fd_;  // Cached POSIX file descriptor for efficient fsync
    
    uint64_t maxSegmentSize_;
    uint64_t maxSegmentAge_;
    uint64_t currentSize_;
    bool inBatch_;  // Flag indicating if we're in batch write mode
    uint64_t maxReplayedSeq_;  // Maximum sequence number seen during replay

    void writeRecord(uint8_t type, uint64_t seq, uint64_t key, const std::vector<uint8_t>& data);
    void flushAndSync();
};

}  // namespace amind