#include "graph_wal.h"

#include "core/crc32.h"
#include "core/file_utils.h"

#include <cstring>
#include <filesystem>
#include <mutex>
#include <spdlog/spdlog.h>

namespace amind {

GraphWAL::GraphWAL(const std::string& data_dir)
    : wal_path_(data_dir + "/graph_wal.bin"),
      snapshot_path_(data_dir + "/graph_snapshot.bin") {}

GraphWAL::~GraphWAL() {
    if (wal_file_.is_open()) {
        wal_file_.flush();
        wal_file_.close();
    }
}

bool GraphWAL::open() {
    std::lock_guard lock(mutex_);
    namespace fs = std::filesystem;

    // Ensure parent directory exists
    auto parent = fs::path(wal_path_).parent_path();
    if (!fs::exists(parent)) {
        fs::create_directories(parent);
    }

    wal_file_.open(wal_path_, std::ios::binary | std::ios::app);
    if (!wal_file_.is_open()) {
        spdlog::error("GraphWAL: failed to open WAL file: {}", wal_path_);
        return false;
    }
    spdlog::info("GraphWAL: opened WAL file: {}", wal_path_);
    return true;
}

void GraphWAL::appendAdd(uint64_t from, uint64_t to, EdgeType type,
                          float weight, uint32_t timestamp) {
    std::lock_guard lock(mutex_);
    if (!wal_file_.is_open()) return;

    WalRecord rec{};
    rec.op = static_cast<uint8_t>(WalOp::AddEdge);
    rec.from_id = from;
    rec.to_id = to;
    rec.edge_type = static_cast<uint8_t>(type);
    rec.weight = weight;
    rec.timestamp = timestamp;

    // CRC over all fields except the crc field itself
    size_t payload_size = sizeof(WalRecord) - sizeof(uint32_t);
    rec.crc32 = amind::crc32(&rec, payload_size);

    wal_file_.write(reinterpret_cast<const char*>(&rec), sizeof(rec));
    wal_file_.flush();
    // fsync to ensure durability
    amind::fsyncFile(wal_path_);
}

void GraphWAL::appendRemove(uint64_t node_id) {
    std::lock_guard lock(mutex_);
    if (!wal_file_.is_open()) return;

    WalRecord rec{};
    rec.op = static_cast<uint8_t>(WalOp::RemoveEdge);
    rec.from_id = node_id;
    rec.to_id = 0;
    rec.edge_type = 0;
    rec.weight = 0.0f;
    rec.timestamp = 0;

    size_t payload_size = sizeof(WalRecord) - sizeof(uint32_t);
    rec.crc32 = amind::crc32(&rec, payload_size);

    wal_file_.write(reinterpret_cast<const char*>(&rec), sizeof(rec));
    wal_file_.flush();
    amind::fsyncFile(wal_path_);
}

size_t GraphWAL::replay(
    std::function<void(WalOp, uint64_t, uint64_t, EdgeType, float, uint32_t)> callback) {
    std::lock_guard lock(mutex_);

    std::ifstream file(wal_path_, std::ios::binary);
    if (!file.is_open()) return 0;

    size_t count = 0;
    WalRecord rec{};
    while (file.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
        // Verify CRC
        size_t payload_size = sizeof(WalRecord) - sizeof(uint32_t);
        uint32_t expected_crc = amind::crc32(&rec, payload_size);
        if (expected_crc != rec.crc32) {
            spdlog::warn("GraphWAL: CRC mismatch at record {}, stopping replay", count);
            break;
        }

        auto op = static_cast<WalOp>(rec.op);
        auto edge_type = static_cast<EdgeType>(rec.edge_type);
        callback(op, rec.from_id, rec.to_id, edge_type, rec.weight, rec.timestamp);
        count++;
    }
    file.close();
    spdlog::info("GraphWAL: replayed {} records from WAL", count);
    return count;
}

bool GraphWAL::checkpoint(const std::vector<EdgeData>& edges) {
    std::lock_guard lock(mutex_);
    namespace fs = std::filesystem;

    // Write snapshot
    std::string tmp_path = snapshot_path_ + ".tmp";
    std::ofstream snap(tmp_path, std::ios::binary);
    if (!snap.is_open()) {
        spdlog::error("GraphWAL: failed to create snapshot file");
        return false;
    }

    // Write header placeholder
    SnapshotHeader header{};
    header.magic = SNAPSHOT_MAGIC;
    header.version = SNAPSHOT_VERSION;
    header.edge_count = edges.size();
    snap.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write edges and compute CRC
    std::vector<uint8_t> edge_bytes;
    for (const auto& e : edges) {
        size_t offset = edge_bytes.size();
        edge_bytes.resize(offset + sizeof(EdgeData));
        std::memcpy(edge_bytes.data() + offset, &e, sizeof(EdgeData));
    }

    if (!edge_bytes.empty()) {
        snap.write(reinterpret_cast<const char*>(edge_bytes.data()), edge_bytes.size());
    }

    // Compute CRC over edge data
    header.crc32 = edge_bytes.empty() ? 0 : amind::crc32(edge_bytes.data(), edge_bytes.size());

    // Rewrite header with CRC
    snap.seekp(0);
    snap.write(reinterpret_cast<const char*>(&header), sizeof(header));
    snap.close();

    // fsync before atomic rename
    amind::fsyncFile(tmp_path);

    // Atomic rename
    if (fs::exists(snapshot_path_)) {
        fs::remove(snapshot_path_);
    }
    fs::rename(tmp_path, snapshot_path_);

    // Truncate WAL
    if (wal_file_.is_open()) {
        wal_file_.close();
    }
    wal_file_.open(wal_path_, std::ios::binary | std::ios::trunc);

    spdlog::info("GraphWAL: checkpoint complete, {} edges saved", edges.size());
    return true;
}

std::vector<GraphWAL::EdgeData> GraphWAL::loadSnapshot() {
    std::lock_guard lock(mutex_);
    std::vector<EdgeData> edges;

    std::ifstream file(snapshot_path_, std::ios::binary);
    if (!file.is_open()) return edges;

    SnapshotHeader header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (!file.good() || header.magic != SNAPSHOT_MAGIC || header.version != SNAPSHOT_VERSION) {
        spdlog::warn("GraphWAL: invalid snapshot file, skipping");
        return edges;
    }

    edges.resize(header.edge_count);
    if (header.edge_count > 0) {
        file.read(reinterpret_cast<char*>(edges.data()),
                  header.edge_count * sizeof(EdgeData));

        if (!file.good()) {
            spdlog::warn("GraphWAL: truncated snapshot, read partial data");
            edges.clear();
            return edges;
        }

        // Verify CRC
        uint32_t crc = amind::crc32(edges.data(), edges.size() * sizeof(EdgeData));
        if (crc != header.crc32) {
            spdlog::warn("GraphWAL: snapshot CRC mismatch, discarding");
            edges.clear();
            return edges;
        }
    }

    file.close();
    spdlog::info("GraphWAL: loaded snapshot with {} edges", edges.size());
    return edges;
}

void GraphWAL::sync() {
    std::lock_guard lock(mutex_);
    if (wal_file_.is_open()) {
        wal_file_.flush();
        amind::fsyncFile(wal_path_);
    }
}

size_t GraphWAL::walSize() const {
    std::lock_guard lock(mutex_);
    namespace fs = std::filesystem;
    if (fs::exists(wal_path_)) {
        return static_cast<size_t>(fs::file_size(wal_path_));
    }
    return 0;
}

}  // namespace amind
