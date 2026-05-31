#include "recovery.h"

#include <algorithm>
#include <cstring>
#include <regex>
#include <stdexcept>

namespace amind {

WalRecovery::WalRecovery(const std::filesystem::path& data_dir,
                         const std::string& wal_filename)
    : data_dir_(data_dir),
      wal_filename_(wal_filename) {
}

RecoveryStats WalRecovery::recover(PutHandler on_put, DeleteHandler on_delete) const {
    RecoveryStats stats;

    auto files = walFiles();
    for (const auto& wal_path : files) {
        replayWalFile(wal_path, on_put, on_delete, stats);
        ++stats.wal_files_processed;
    }

    return stats;
}

bool WalRecovery::hasWalFiles() const {
    return !walFiles().empty();
}

std::vector<std::filesystem::path> WalRecovery::walFiles() const {
    std::vector<std::filesystem::path> files;

    if (!std::filesystem::exists(data_dir_)) {
        return files;
    }

    // Collect archived WAL files: <wal_filename>.old.<N>
    std::string archived_prefix = wal_filename_ + ".old.";

    for (const auto& entry : std::filesystem::directory_iterator(data_dir_)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto& filename = entry.path().filename().string();

        if (filename.rfind(archived_prefix, 0) == 0) {
            // This is an archived WAL file
            files.push_back(entry.path());
        }
    }

    // Sort archived files by sequence number (ascending = oldest first)
    std::sort(files.begin(), files.end(),
        [&archived_prefix](const std::filesystem::path& left,
                           const std::filesystem::path& right) {
            auto extract_sequence = [&archived_prefix](const std::filesystem::path& path) -> uint64_t {
                std::string filename = path.filename().string();
                std::string suffix = filename.substr(archived_prefix.size());
                try {
                    return std::stoull(suffix);
                } catch (...) {
                    return 0;
                }
            };
            return extract_sequence(left) < extract_sequence(right);
        });

    // Append the active WAL file (processed last = most recent)
    auto active_wal = data_dir_ / wal_filename_;
    if (std::filesystem::exists(active_wal) && std::filesystem::file_size(active_wal) > 0) {
        files.push_back(active_wal);
    }

    return files;
}

void WalRecovery::replayWalFile(const std::filesystem::path& wal_path,
                                const PutHandler& on_put,
                                const DeleteHandler& on_delete,
                                RecoveryStats& stats) const {
    WriteAheadLog wal(wal_path);
    auto entries = wal.readAll();

    for (const auto& entry : entries) {
        switch (entry.type) {
            case WalEntryType::Put: {
                auto record = MemoryRecord::deserialize(entry.payload);
                if (record.ok()) {
                    on_put(record.value());
                    ++stats.puts_applied;
                } else {
                    ++stats.entries_skipped;
                }
                break;
            }
            case WalEntryType::Delete: {
                if (entry.payload.size() >= sizeof(uint64_t)) {
                    uint64_t memory_id = 0;
                    std::memcpy(&memory_id, entry.payload.data(), sizeof(uint64_t));
                    on_delete(memory_id);
                    ++stats.deletes_applied;
                } else {
                    ++stats.entries_skipped;
                }
                break;
            }
            default:
                ++stats.entries_skipped;
                break;
        }
        ++stats.entries_replayed;
    }
}

}  // namespace amind
