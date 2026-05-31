#pragma once

#include "wal.h"
#include "memory_record.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace amind {

/// Statistics from a WAL recovery operation.
struct RecoveryStats {
    uint64_t entries_replayed{0};   // Total entries successfully replayed
    uint64_t puts_applied{0};       // Number of Put operations applied
    uint64_t deletes_applied{0};    // Number of Delete operations applied
    uint64_t entries_skipped{0};    // Entries skipped (e.g., corrupted payload)
    uint64_t wal_files_processed{0}; // Number of WAL files processed
};

/// WAL Recovery Engine.
///
/// Reads WAL entries and replays them into an in-memory data structure
/// via user-provided callbacks. This decouples recovery from the specific
/// MemTable implementation (which will be built in Phase 2).
///
/// Recovery order:
///   1. Archived WAL files (oldest first, sorted by rotation sequence)
///   2. Current (active) WAL file
///
/// Usage:
///   WalRecovery recovery(data_dir);
///   auto stats = recovery.recover(
///       [&](const MemoryRecord& record) { memtable.put(record); },
///       [&](uint64_t memory_id) { memtable.remove(memory_id); }
///   );
class WalRecovery {
public:
    /// Callback invoked for each Put entry during recovery.
    /// The MemoryRecord has been deserialized from the WAL payload.
    using PutHandler = std::function<void(const MemoryRecord& record)>;

    /// Callback invoked for each Delete entry during recovery.
    /// The parameter is the memory_id to delete.
    using DeleteHandler = std::function<void(uint64_t memory_id)>;

    /// Construct a recovery engine for the given data directory.
    /// @param data_dir Directory containing WAL files
    /// @param wal_filename Base name of the WAL file (default: "clawmind.wal")
    explicit WalRecovery(const std::filesystem::path& data_dir,
                         const std::string& wal_filename = "clawmind.wal");

    /// Replay all WAL entries through the provided handlers.
    /// Processes archived WAL files first (oldest to newest), then the active WAL.
    /// @param on_put Called for each Put entry with the deserialized MemoryRecord
    /// @param on_delete Called for each Delete entry with the memory_id
    /// @return Statistics about the recovery operation
    [[nodiscard]] RecoveryStats recover(PutHandler on_put, DeleteHandler on_delete) const;

    /// Check if there are any WAL files to recover from.
    [[nodiscard]] bool hasWalFiles() const;

    /// Get the list of WAL files that would be processed (in recovery order).
    [[nodiscard]] std::vector<std::filesystem::path> walFiles() const;

private:
    /// Replay entries from a single WAL file.
    void replayWalFile(const std::filesystem::path& wal_path,
                       const PutHandler& on_put,
                       const DeleteHandler& on_delete,
                       RecoveryStats& stats) const;

    std::filesystem::path data_dir_;
    std::string wal_filename_;
};

}  // namespace amind
