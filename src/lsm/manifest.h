#pragma once
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace amind {

/// Manifest — tracks SSTable metadata and checkpoint sequence number.
/// Uses write-rename for atomic updates (inspired by OceanBase's Manifest design).
///
/// The manifest records:
/// - checkpoint_seq: the maximum WAL sequence number that has been
///   fully persisted to SSTables. On recovery, only WAL records with
///   seq > checkpoint_seq need to be replayed.
/// - next_sst_id: the next SSTable file ID to use.
/// - sstables: lists of SSTable filenames at each level (L0, L1, L2).
class Manifest {
public:
    static constexpr const char* FILENAME = "MANIFEST";
    static constexpr const char* TEMP_FILENAME = "MANIFEST.tmp";
    static constexpr int CURRENT_VERSION = 1;

    explicit Manifest(const std::filesystem::path& dataDir);

    /// Load manifest from disk. Returns false if file doesn't exist.
    bool load();

    /// Save manifest to disk atomically (write tmp + rename).
    void save();

    // Getters
    uint64_t checkpointSeq() const { return checkpointSeq_; }
    uint64_t nextSstId() const { return nextSstId_; }
    const std::vector<std::string>& l0Files() const { return l0Files_; }
    const std::vector<std::string>& l1Files() const { return l1Files_; }
    const std::vector<std::string>& l2Files() const { return l2Files_; }

    // Setters
    void setCheckpointSeq(uint64_t seq) { checkpointSeq_ = seq; }
    void setNextSstId(uint64_t id) { nextSstId_ = id; }
    void setL0Files(std::vector<std::string> files) { l0Files_ = std::move(files); }
    void setL1Files(std::vector<std::string> files) { l1Files_ = std::move(files); }
    void setL2Files(std::vector<std::string> files) { l2Files_ = std::move(files); }

    /// Add an L0 SSTable file (inserted at front = newest first).
    void addL0File(const std::string& filename);

    /// Replace L0 files with a new L1 file (after compaction).
    void compactL0ToL1(const std::vector<std::string>& oldL0Files,
                       const std::string& newL1File);

    /// Replace L1 files with a new L2 file (after compaction).
    void compactL1ToL2(const std::vector<std::string>& oldL1Files,
                       const std::string& newL2File);

    /// Allocate the next SSTable ID and return it.
    uint64_t allocateSstId();

private:
    std::filesystem::path dataDir_;
    uint64_t checkpointSeq_ = 0;
    uint64_t nextSstId_ = 0;
    std::vector<std::string> l0Files_;
    std::vector<std::string> l1Files_;
    std::vector<std::string> l2Files_;
};

}  // namespace amind
