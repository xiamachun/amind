#include "lsm_wal.h"

#include <chrono>
#include <crc32c/crc32c.h>
#include <cstring>
#include <fcntl.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <unistd.h>
#include <algorithm>

static uint32_t crc32Simple(const uint8_t* data, size_t length) {
    return crc32c::Crc32c(data, length);
}

namespace amind {

LsmWriteAheadLog::LsmWriteAheadLog(const std::filesystem::path& walPath,
                                   uint64_t maxSegmentSize,
                                   uint64_t maxSegmentAge)
    : path_(walPath),
      fd_(-1),
      maxSegmentSize_(maxSegmentSize),
      maxSegmentAge_(maxSegmentAge),
      currentSize_(0),
      inBatch_(false),
      maxReplayedSeq_(0) {
    file_.open(path_, std::ios::binary | std::ios::app);
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open WAL file: " + path_.string());
    }
    
    // Get POSIX fd for efficient fsync
    fd_ = ::open(path_.c_str(), O_WRONLY);
    if (fd_ < 0) {
        spdlog::warn("Failed to get POSIX fd for WAL: {}, fsync may be slower", path_.string());
    }
    
    // Initialize currentSize_ with existing file size
    if (std::filesystem::exists(path_)) {
        currentSize_ = std::filesystem::file_size(path_);
    }
}

LsmWriteAheadLog::~LsmWriteAheadLog() {
    if (file_.is_open()) {
        file_.close();
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

void LsmWriteAheadLog::flushAndSync() {
    file_.flush();

    // Use cached POSIX fd for efficient fsync
    if (fd_ >= 0) {
        ::fsync(fd_);
    } else {
        // Fallback: open a temporary fd just for fsync
        int fd = ::open(path_.c_str(), O_WRONLY);
        if (fd >= 0) {
            ::fsync(fd);
            ::close(fd);
        }
    }
}

void LsmWriteAheadLog::appendPut(uint64_t seq, uint64_t key, const std::vector<uint8_t>& data) {
    writeRecord(static_cast<uint8_t>(WalRecordType::Put), seq, key, data);
}

void LsmWriteAheadLog::appendDelete(uint64_t seq, uint64_t key) {
    writeRecord(static_cast<uint8_t>(WalRecordType::Delete), seq, key, {});
}

void LsmWriteAheadLog::beginBatch() {
    inBatch_ = true;
}

void LsmWriteAheadLog::endBatch() {
    inBatch_ = false;
    flushAndSync();
}

void LsmWriteAheadLog::writeRecord(uint8_t type, uint64_t seq, uint64_t key, const std::vector<uint8_t>& data) {
    uint32_t dataLen = static_cast<uint32_t>(data.size());

    // Build the 32-byte header
    std::vector<uint8_t> header(WAL_HEADER_SIZE);
    size_t offset = 0;
    
    // magic (2 bytes)
    uint16_t magic = WAL_MAGIC;
    std::memcpy(header.data() + offset, &magic, 2);
    offset += 2;
    
    // version (2 bytes)
    uint16_t version = WAL_VERSION;
    std::memcpy(header.data() + offset, &version, 2);
    offset += 2;
    
    // type (1 byte)
    header[offset] = type;
    offset += 1;
    
    // reserved (3 bytes)
    std::memset(header.data() + offset, 0, 3);
    offset += 3;
    
    // seq (8 bytes)
    std::memcpy(header.data() + offset, &seq, 8);
    offset += 8;
    
    // key (8 bytes)
    std::memcpy(header.data() + offset, &key, 8);
    offset += 8;
    
    // data_len (4 bytes)
    std::memcpy(header.data() + offset, &dataLen, 4);
    offset += 4;
    
    // header_crc32 (4 bytes) - CRC over first 28 bytes
    uint32_t headerCrc = crc32Simple(header.data(), 28);
    std::memcpy(header.data() + offset, &headerCrc, 4);
    
    // Write header
    file_.write(reinterpret_cast<const char*>(header.data()), WAL_HEADER_SIZE);
    
    // Write data and data_crc32 if data_len > 0
    if (dataLen > 0) {
        // Compute data CRC32
        uint32_t dataCrc = crc32Simple(data.data(), dataLen);
        
        // Write data
        file_.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(dataLen));
        // Write data CRC32
        file_.write(reinterpret_cast<const char*>(&dataCrc), 4);
    }
    
    // Update current size
    currentSize_ += WAL_HEADER_SIZE + dataLen + (dataLen > 0 ? 4 : 0);

    // Flush and fsync if not in batch mode
    if (!inBatch_) {
        flushAndSync();
    }
    
    // Check if rotation is needed
    if (currentSize_ >= maxSegmentSize_) {
        rotate();
    }
}

void LsmWriteAheadLog::truncate() {
    file_.close();
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    
    // Re-open in truncate mode to clear the file
    file_.open(path_, std::ios::binary | std::ios::trunc);
    if (!file_.is_open()) {
        spdlog::error("Failed to truncate WAL file: {}", path_.string());
        return;
    }
    
    // Get new POSIX fd
    fd_ = ::open(path_.c_str(), O_WRONLY);
    if (fd_ < 0) {
        spdlog::warn("Failed to get POSIX fd after truncate: {}", path_.string());
    }
    
    currentSize_ = 0;
    file_.flush();
    spdlog::info("WAL truncated: {}", path_.string());
}

void LsmWriteAheadLog::rotate() {
    file_.close();
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    
    // Generate timestamp for the segment file
    uint64_t timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Rename current file to wal.log.timestamp
    std::filesystem::path segmentPath = path_.string() + "." + std::to_string(timestamp);
    try {
        if (std::filesystem::exists(path_)) {
            std::filesystem::rename(path_, segmentPath);
            spdlog::info("WAL rotated: {} -> {}", path_.string(), segmentPath.string());
        }
    } catch (const std::exception& e) {
        spdlog::error("Failed to rotate WAL: {}", e.what());
    }
    
    // Create new empty WAL file
    file_.open(path_, std::ios::binary | std::ios::trunc);
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to create new WAL file after rotation: " + path_.string());
    }
    
    // Get new POSIX fd
    fd_ = ::open(path_.c_str(), O_WRONLY);
    if (fd_ < 0) {
        spdlog::warn("Failed to get POSIX fd after rotation: {}", path_.string());
    }
    
    currentSize_ = 0;
    
    // Expire old segments
    expireOldSegments();
}

void LsmWriteAheadLog::expireOldSegments() {
    std::filesystem::path walDir = path_.parent_path();
    if (walDir.empty()) {
        walDir = ".";
    }
    
    std::string walFileName = path_.filename().string();
    
    try {
        uint64_t now = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        for (const auto& entry : std::filesystem::directory_iterator(walDir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            
            std::string entryName = entry.path().filename().string();
            
            // Check if file matches pattern: wal.log.timestamp
            if (entryName.size() > walFileName.size() + 1 &&
                entryName.substr(0, walFileName.size()) == walFileName &&
                entryName[walFileName.size()] == '.') {
                
                // Try to parse timestamp
                std::string timestampStr = entryName.substr(walFileName.size() + 1);
                try {
                    uint64_t timestamp = std::stoull(timestampStr);
                    uint64_t age = now - timestamp;
                    
                    if (age > maxSegmentAge_) {
                        std::filesystem::remove(entry.path());
                        spdlog::info("Expired old WAL segment: {} (age: {}s)", 
                                     entry.path().string(), age);
                    }
                } catch (const std::exception&) {
                    // Not a valid timestamp, skip
                    continue;
                }
            }
        }
    } catch (const std::exception& e) {
        spdlog::warn("Failed to expire old WAL segments: {}", e.what());
    }
}

uint64_t LsmWriteAheadLog::replay(const std::filesystem::path& walPath,
                                  const ReplayVisitor& visitor,
                                  uint64_t minSeq) {
    std::filesystem::path walDir = walPath.parent_path();
    if (walDir.empty()) {
        walDir = ".";
    }
    
    std::string walFileName = walPath.filename().string();
    
    // Collect all WAL segment files and sort by timestamp
    std::vector<std::pair<uint64_t, std::filesystem::path>> segmentFiles;
    
    // First, try the main WAL file
    if (std::filesystem::exists(walPath)) {
        segmentFiles.push_back({0, walPath});
    }
    
    // Then collect segment files (wal.log.timestamp)
    try {
        for (const auto& entry : std::filesystem::directory_iterator(walDir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            
            std::string entryName = entry.path().filename().string();
            
            // Check if file matches pattern: wal.log.timestamp
            if (entryName.size() > walFileName.size() + 1 &&
                entryName.substr(0, walFileName.size()) == walFileName &&
                entryName[walFileName.size()] == '.') {
                
                // Try to parse timestamp
                std::string timestampStr = entryName.substr(walFileName.size() + 1);
                try {
                    uint64_t timestamp = std::stoull(timestampStr);
                    segmentFiles.push_back({timestamp, entry.path()});
                } catch (const std::exception&) {
                    // Not a valid timestamp, skip
                    continue;
                }
            }
        }
    } catch (const std::exception& e) {
        spdlog::warn("Failed to scan WAL directory: {}", e.what());
    }
    
    // Sort by timestamp (oldest first)
    std::sort(segmentFiles.begin(), segmentFiles.end());
    
    size_t totalRecoveredCount = 0;
    size_t totalCorruptedCount = 0;
    uint64_t maxSeq = 0;
    
    // Replay each segment file
    for (const auto& [timestamp, segmentPath] : segmentFiles) {
        if (!std::filesystem::exists(segmentPath)) {
            continue;
        }

        auto fileSize = std::filesystem::file_size(segmentPath);
        if (fileSize == 0) {
            continue;
        }

        std::ifstream file(segmentPath, std::ios::binary);
        if (!file.is_open()) {
            spdlog::warn("Failed to open WAL segment for replay: {}", segmentPath.string());
            continue;
        }

        size_t recoveredCount = 0;
        size_t corruptedCount = 0;

        while (file.good() && !file.eof()) {
            // Read header (32 bytes)
            std::vector<uint8_t> header(WAL_HEADER_SIZE);
            file.read(reinterpret_cast<char*>(header.data()), WAL_HEADER_SIZE);
            if (file.gcount() < static_cast<std::streamsize>(WAL_HEADER_SIZE)) {
                break;  // Incomplete record at end of file
            }

            // Parse header
            size_t offset = 0;
            
            // magic (2 bytes)
            uint16_t magic;
            std::memcpy(&magic, header.data() + offset, 2);
            offset += 2;
            
            // Verify magic number
            if (magic != WAL_MAGIC) {
                spdlog::warn("WAL replay: invalid magic number 0x{:04X}, stopping", magic);
                ++corruptedCount;
                break;
            }
            
            // version (2 bytes)
            uint16_t version;
            std::memcpy(&version, header.data() + offset, 2);
            offset += 2;
            
            if (version != WAL_VERSION) {
                spdlog::warn("WAL replay: unsupported version {}, stopping", version);
                ++corruptedCount;
                break;
            }
            
            // type (1 byte)
            uint8_t type = header[offset];
            offset += 1;
            
            // reserved (3 bytes)
            offset += 3;
            
            // seq (8 bytes)
            uint64_t seq;
            std::memcpy(&seq, header.data() + offset, 8);
            offset += 8;
            
            // key (8 bytes)
            uint64_t key;
            std::memcpy(&key, header.data() + offset, 8);
            offset += 8;
            
            // data_len (4 bytes)
            uint32_t dataLen;
            std::memcpy(&dataLen, header.data() + offset, 4);
            offset += 4;
            
            // header_crc32 (4 bytes)
            uint32_t headerCrc;
            std::memcpy(&headerCrc, header.data() + offset, 4);
            
            // Verify header CRC32 (over first 28 bytes)
            uint32_t computedHeaderCrc = crc32Simple(header.data(), 28);
            if (computedHeaderCrc != headerCrc) {
                spdlog::warn("WAL replay: header CRC mismatch at seq {}, skipping record", seq);
                ++corruptedCount;
                // Skip the data and data CRC to move to next record
                if (dataLen > 0) {
                    file.seekg(static_cast<std::streamoff>(dataLen + 4), std::ios::cur);
                }
                continue;  // Skip this record and try to read next
            }

            // Sanity check: data_len should not exceed 64 MB
            if (dataLen > 64 * 1024 * 1024) {
                spdlog::warn("WAL replay: suspicious data_len={}, stopping", dataLen);
                ++corruptedCount;
                break;
            }

            // Read data and data_crc32 if data_len > 0
            std::vector<uint8_t> data(dataLen);
            if (dataLen > 0) {
                // Read data
                file.read(reinterpret_cast<char*>(data.data()),
                          static_cast<std::streamsize>(dataLen));
                if (static_cast<uint32_t>(file.gcount()) < dataLen) {
                    ++corruptedCount;
                    break;  // Truncated data, stop replay
                }

                // Read data CRC32 (4 bytes)
                uint32_t dataCrc;
                file.read(reinterpret_cast<char*>(&dataCrc), 4);
                if (file.gcount() < 4) {
                    ++corruptedCount;
                    break;  // Truncated CRC, stop replay
                }

                // Verify data CRC32
                uint32_t computedDataCrc = crc32Simple(data.data(), dataLen);
                if (computedDataCrc != dataCrc) {
                    spdlog::warn("WAL replay: data CRC mismatch at seq {}, skipping record", seq);
                    ++corruptedCount;
                    // Already read past the data CRC, just continue
                    continue;  // Skip this record and try to read next
                }
            }

            bool isTombstone = (type == static_cast<uint8_t>(WalRecordType::Delete));
            bool isDelete = (type == static_cast<uint8_t>(WalRecordType::Delete));
            
            // Sanity check: DELETE should have data_len == 0
            if (isDelete && dataLen > 0) {
                spdlog::warn("WAL replay: DELETE record with data_len > 0 at seq {}, stopping", seq);
                ++corruptedCount;
                break;
            }
            
            // Track maximum sequence number regardless of filter
            if (seq > maxSeq) {
                maxSeq = seq;
            }

            // Skip records with seq <= minSeq (incremental recovery)
            if (seq <= minSeq) {
                continue;
            }

            visitor(seq, key, std::move(data), isTombstone);
            ++recoveredCount;
        }

        totalRecoveredCount += recoveredCount;
        totalCorruptedCount += corruptedCount;
        
        if (recoveredCount > 0 || corruptedCount > 0) {
            spdlog::info("WAL segment replay complete: {} records recovered, {} corrupted/truncated",
                         recoveredCount, corruptedCount);
        }
    }
    
    if (totalRecoveredCount > 0 || totalCorruptedCount > 0) {
        spdlog::info("WAL replay complete: {} total records recovered, {} total corrupted/truncated",
                     totalRecoveredCount, totalCorruptedCount);
        spdlog::info("WAL replay: maximum sequence number = {}", maxSeq);
    }

    return maxSeq;
}

}  // namespace amind