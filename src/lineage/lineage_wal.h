#pragma once

#include "core/types.h"

#include <cstdint>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace amind {

enum class LineageWalOp : uint8_t {
    RecordLineage = 0x01,
    RemoveNode = 0x02,
    MarkRemoved = 0x03,
};

/// Fixed-size WAL record for lineage operations.
/// For RecordLineage: from_id = child, parent_count = number of parents (parents follow as 8B each)
/// For RemoveNode/MarkRemoved: from_id = memory_id, rest zeroed.
///
/// Variable-length format per entry:
///   [op:1B][child_id:8B][lineage_op:1B][parent_count:2B][timestamp:4B]
///   [parent_id:8B] * parent_count
///   [crc32:4B]
///
/// For RemoveNode/MarkRemoved:
///   [op:1B][node_id:8B][lineage_op:1B(0)][parent_count:2B(0)][timestamp:4B]
///   [crc32:4B]

#pragma pack(push, 1)
struct LineageWalHeader {
    uint8_t op;
    uint64_t node_id;
    uint8_t lineage_op;
    uint16_t parent_count;
    uint32_t timestamp;
};
#pragma pack(pop)

static_assert(sizeof(LineageWalHeader) == 16, "LineageWalHeader must be 16 bytes");

#pragma pack(push, 1)
struct LineageSnapshotHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t record_count;
    uint32_t crc32;
};
#pragma pack(pop)

/// Snapshot entry: one lineage record (variable length).
#pragma pack(push, 1)
struct LineageSnapshotEntry {
    uint64_t child_id;
    uint8_t lineage_op;
    uint16_t parent_count;
    uint32_t created_at;
};
#pragma pack(pop)

class LineageWAL {
public:
    static constexpr uint32_t SNAPSHOT_MAGIC = 0x4C574C53;  // "LWLS"
    static constexpr uint32_t SNAPSHOT_VERSION = 1;

    explicit LineageWAL(const std::string& data_dir);
    ~LineageWAL();

    bool open();

    void appendRecordLineage(uint64_t child, const std::vector<uint64_t>& parents,
                             LineageOp op, uint32_t timestamp);
    void appendRemoveNode(uint64_t node_id);
    void appendMarkRemoved(uint64_t node_id);

    struct ReplayEntry {
        LineageWalOp wal_op;
        uint64_t node_id;
        LineageOp lineage_op;
        std::vector<uint64_t> parents;
        uint32_t timestamp;
    };

    size_t replay(std::function<void(const ReplayEntry&)> callback);

    struct SnapshotRecord {
        uint64_t child_id;
        std::vector<uint64_t> parent_ids;
        LineageOp op;
        uint32_t created_at;
    };

    bool checkpoint(const std::vector<SnapshotRecord>& records,
                    const std::vector<uint64_t>& removed_nodes);
    std::vector<SnapshotRecord> loadSnapshot(std::vector<uint64_t>& removed_nodes_out);

    void sync();
    size_t walSize() const;

private:
    static uint32_t computeCRC(const void* data, size_t len);

    std::string wal_path_;
    std::string snapshot_path_;
    std::ofstream wal_file_;
    mutable std::mutex mutex_;
};

}  // namespace amind
