#pragma once

#include "core/types.h"

#include <cstdint>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace amind {

/// WAL record operation type.
enum class WalOp : uint8_t {
    AddEdge = 0x01,
    RemoveEdge = 0x02,
};

/// Binary WAL record: 34 bytes + 4 bytes CRC32 = 38 bytes per entry.
///   [op:1B][from_id:8B][to_id:8B][edge_type:1B][weight:4B][timestamp:4B][crc32:4B]
#pragma pack(push, 1)
struct WalRecord {
    uint8_t op;           // WalOp
    uint64_t from_id;
    uint64_t to_id;
    uint8_t edge_type;    // EdgeType
    float weight;
    uint32_t timestamp;
    uint32_t crc32;
};
#pragma pack(pop)

static_assert(sizeof(WalRecord) == 30, "WalRecord must be 30 bytes");

/// Graph snapshot file header.
#pragma pack(push, 1)
struct SnapshotHeader {
    uint32_t magic;       // 0x47535348 ("GSSH")
    uint32_t version;     // 1
    uint64_t edge_count;
    uint32_t crc32;       // CRC32 of all edge data after header
};
#pragma pack(pop)

/// Write-Ahead Log for graph edges.
/// Append-only binary log that records addEdge/removeEdge operations.
/// Supports replay on startup to restore graph state.
class GraphWAL {
public:
    static constexpr uint32_t SNAPSHOT_MAGIC = 0x47535348;  // "GSSH"
    static constexpr uint32_t SNAPSHOT_VERSION = 1;

    explicit GraphWAL(const std::string& data_dir);
    ~GraphWAL();

    /// Open or create the WAL file.
    bool open();

    /// Append an AddEdge record.
    void appendAdd(uint64_t from, uint64_t to, EdgeType type, float weight, uint32_t timestamp);

    /// Append a RemoveEdge record (remove all edges for a node).
    void appendRemove(uint64_t node_id);

    /// Replay WAL records into a callback.
    /// Callback signature: void(WalOp op, uint64_t from, uint64_t to, EdgeType type, float weight, uint32_t ts)
    size_t replay(std::function<void(WalOp, uint64_t, uint64_t, EdgeType, float, uint32_t)> callback);

    /// Write a full snapshot of current edges, then truncate WAL.
    struct EdgeData {
        uint64_t from_id;
        uint64_t to_id;
        uint8_t edge_type;
        float weight;
        uint32_t timestamp;
    };
    bool checkpoint(const std::vector<EdgeData>& edges);

    /// Load snapshot (called before replay).
    /// Returns edges from the snapshot file.
    std::vector<EdgeData> loadSnapshot();

    /// Sync WAL to disk.
    void sync();

    /// Get WAL file size for deciding when to checkpoint.
    size_t walSize() const;

private:
    static uint32_t computeCRC(const void* data, size_t len);

    std::string wal_path_;
    std::string snapshot_path_;
    std::ofstream wal_file_;
    mutable std::mutex mutex_;
};

}  // namespace amind
