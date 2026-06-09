#include "lineage_wal.h"

#include "core/crc32.h"
#include "core/file_utils.h"

#include <cstring>
#include <filesystem>
#include <spdlog/spdlog.h>

namespace amind {

LineageWAL::LineageWAL(const std::string& data_dir)
    : wal_path_(data_dir + "/lineage_wal.bin"),
      snapshot_path_(data_dir + "/lineage_snapshot.bin") {}

LineageWAL::~LineageWAL() {
    if (wal_file_.is_open()) {
        wal_file_.flush();
        wal_file_.close();
    }
}

bool LineageWAL::open() {
    std::lock_guard lock(mutex_);
    namespace fs = std::filesystem;

    auto parent = fs::path(wal_path_).parent_path();
    if (!fs::exists(parent)) {
        fs::create_directories(parent);
    }

    wal_file_.open(wal_path_, std::ios::binary | std::ios::app);
    if (!wal_file_.is_open()) {
        spdlog::error("LineageWAL: failed to open WAL file: {}", wal_path_);
        return false;
    }
    spdlog::info("LineageWAL: opened WAL file: {}", wal_path_);
    return true;
}

void LineageWAL::appendRecordLineage(uint64_t child, const std::vector<uint64_t>& parents,
                                     LineageOp op, uint32_t timestamp) {
    std::lock_guard lock(mutex_);
    if (!wal_file_.is_open()) return;

    LineageWalHeader hdr{};
    hdr.op = static_cast<uint8_t>(LineageWalOp::RecordLineage);
    hdr.node_id = child;
    hdr.lineage_op = static_cast<uint8_t>(op);
    hdr.parent_count = static_cast<uint16_t>(parents.size());
    hdr.timestamp = timestamp;

    // Build buffer: header + parent_ids + crc
    std::vector<uint8_t> buf;
    buf.resize(sizeof(LineageWalHeader) + parents.size() * sizeof(uint64_t));
    std::memcpy(buf.data(), &hdr, sizeof(hdr));
    if (!parents.empty()) {
        std::memcpy(buf.data() + sizeof(hdr), parents.data(),
                    parents.size() * sizeof(uint64_t));
    }

    uint32_t crc = amind::crc32(buf.data(), buf.size());

    wal_file_.write(reinterpret_cast<const char*>(buf.data()),
                    static_cast<std::streamsize>(buf.size()));
    wal_file_.write(reinterpret_cast<const char*>(&crc), sizeof(crc));
    wal_file_.flush();
    amind::fsyncFile(wal_path_);
}

void LineageWAL::appendRemoveNode(uint64_t node_id) {
    std::lock_guard lock(mutex_);
    if (!wal_file_.is_open()) return;

    LineageWalHeader hdr{};
    hdr.op = static_cast<uint8_t>(LineageWalOp::RemoveNode);
    hdr.node_id = node_id;
    hdr.lineage_op = 0;
    hdr.parent_count = 0;
    hdr.timestamp = 0;

    uint32_t crc = amind::crc32(&hdr, sizeof(hdr));

    wal_file_.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    wal_file_.write(reinterpret_cast<const char*>(&crc), sizeof(crc));
    wal_file_.flush();
    amind::fsyncFile(wal_path_);
}

void LineageWAL::appendMarkRemoved(uint64_t node_id) {
    std::lock_guard lock(mutex_);
    if (!wal_file_.is_open()) return;

    LineageWalHeader hdr{};
    hdr.op = static_cast<uint8_t>(LineageWalOp::MarkRemoved);
    hdr.node_id = node_id;
    hdr.lineage_op = 0;
    hdr.parent_count = 0;
    hdr.timestamp = 0;

    uint32_t crc = amind::crc32(&hdr, sizeof(hdr));

    wal_file_.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    wal_file_.write(reinterpret_cast<const char*>(&crc), sizeof(crc));
    wal_file_.flush();
    amind::fsyncFile(wal_path_);
}

size_t LineageWAL::replay(std::function<void(const ReplayEntry&)> callback) {
    std::lock_guard lock(mutex_);

    std::ifstream file(wal_path_, std::ios::binary);
    if (!file.is_open()) return 0;

    size_t count = 0;
    while (true) {
        LineageWalHeader hdr{};
        file.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        if (!file.good()) break;

        auto wal_op = static_cast<LineageWalOp>(hdr.op);

        std::vector<uint64_t> parents;
        if (hdr.parent_count > 0) {
            parents.resize(hdr.parent_count);
            file.read(reinterpret_cast<char*>(parents.data()),
                      hdr.parent_count * sizeof(uint64_t));
            if (!file.good()) break;
        }

        uint32_t stored_crc = 0;
        file.read(reinterpret_cast<char*>(&stored_crc), sizeof(stored_crc));
        if (!file.good()) break;

        // Verify CRC
        std::vector<uint8_t> buf(sizeof(hdr) + parents.size() * sizeof(uint64_t));
        std::memcpy(buf.data(), &hdr, sizeof(hdr));
        if (!parents.empty()) {
            std::memcpy(buf.data() + sizeof(hdr), parents.data(),
                        parents.size() * sizeof(uint64_t));
        }
        uint32_t expected_crc = amind::crc32(buf.data(), buf.size());
        if (expected_crc != stored_crc) {
            spdlog::warn("LineageWAL: CRC mismatch at record {}, stopping replay", count);
            break;
        }

        ReplayEntry entry;
        entry.wal_op = wal_op;
        entry.node_id = hdr.node_id;
        entry.lineage_op = static_cast<LineageOp>(hdr.lineage_op);
        entry.parents = std::move(parents);
        entry.timestamp = hdr.timestamp;

        callback(entry);
        count++;
    }

    file.close();
    spdlog::info("LineageWAL: replayed {} records", count);
    return count;
}

bool LineageWAL::checkpoint(const std::vector<SnapshotRecord>& records,
                            const std::vector<uint64_t>& removed_nodes) {
    std::lock_guard lock(mutex_);
    namespace fs = std::filesystem;

    std::string tmp_path = snapshot_path_ + ".tmp";
    std::ofstream snap(tmp_path, std::ios::binary);
    if (!snap.is_open()) {
        spdlog::error("LineageWAL: failed to create snapshot file");
        return false;
    }

    // Header placeholder
    LineageSnapshotHeader header{};
    header.magic = SNAPSHOT_MAGIC;
    header.version = SNAPSHOT_VERSION;
    header.record_count = records.size();
    snap.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Serialize records: for each record, write entry + parent_ids
    std::vector<uint8_t> data_buf;
    for (const auto& rec : records) {
        LineageSnapshotEntry entry{};
        entry.child_id = rec.child_id;
        entry.lineage_op = static_cast<uint8_t>(rec.op);
        entry.parent_count = static_cast<uint16_t>(rec.parent_ids.size());
        entry.created_at = rec.created_at;

        size_t offset = data_buf.size();
        data_buf.resize(offset + sizeof(entry) + rec.parent_ids.size() * sizeof(uint64_t));
        std::memcpy(data_buf.data() + offset, &entry, sizeof(entry));
        if (!rec.parent_ids.empty()) {
            std::memcpy(data_buf.data() + offset + sizeof(entry),
                        rec.parent_ids.data(), rec.parent_ids.size() * sizeof(uint64_t));
        }
    }

    // Append removed_nodes: [count:8B][node_id:8B]*count
    {
        uint64_t removed_count = removed_nodes.size();
        size_t offset = data_buf.size();
        data_buf.resize(offset + sizeof(uint64_t) + removed_nodes.size() * sizeof(uint64_t));
        std::memcpy(data_buf.data() + offset, &removed_count, sizeof(uint64_t));
        if (!removed_nodes.empty()) {
            std::memcpy(data_buf.data() + offset + sizeof(uint64_t),
                        removed_nodes.data(), removed_nodes.size() * sizeof(uint64_t));
        }
    }

    if (!data_buf.empty()) {
        snap.write(reinterpret_cast<const char*>(data_buf.data()),
                   static_cast<std::streamsize>(data_buf.size()));
    }

    header.crc32 = data_buf.empty() ? 0 : amind::crc32(data_buf.data(), data_buf.size());
    snap.seekp(0);
    snap.write(reinterpret_cast<const char*>(&header), sizeof(header));
    snap.close();

    // fsync snapshot before rename
    amind::fsyncFile(tmp_path);

    if (fs::exists(snapshot_path_)) {
        fs::remove(snapshot_path_);
    }
    fs::rename(tmp_path, snapshot_path_);

    // Truncate WAL
    if (wal_file_.is_open()) {
        wal_file_.close();
    }
    wal_file_.open(wal_path_, std::ios::binary | std::ios::trunc);

    spdlog::info("LineageWAL: checkpoint complete, {} records saved", records.size());
    return true;
}

std::vector<LineageWAL::SnapshotRecord> LineageWAL::loadSnapshot(
    std::vector<uint64_t>& removed_nodes_out) {
    std::lock_guard lock(mutex_);
    std::vector<SnapshotRecord> records;
    removed_nodes_out.clear();

    std::ifstream file(snapshot_path_, std::ios::binary);
    if (!file.is_open()) return records;

    LineageSnapshotHeader header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file.good() || header.magic != SNAPSHOT_MAGIC || header.version != SNAPSHOT_VERSION) {
        spdlog::warn("LineageWAL: invalid snapshot file, skipping");
        return records;
    }

    // Read all remaining bytes for CRC verification
    auto data_start = file.tellg();
    file.seekg(0, std::ios::end);
    auto data_size = static_cast<size_t>(file.tellg() - data_start);
    file.seekg(data_start);

    std::vector<uint8_t> data_buf(data_size);
    file.read(reinterpret_cast<char*>(data_buf.data()), static_cast<std::streamsize>(data_size));
    file.close();

    if (data_size > 0) {
        uint32_t crc = amind::crc32(data_buf.data(), data_buf.size());
        if (crc != header.crc32) {
            spdlog::warn("LineageWAL: snapshot CRC mismatch, discarding");
            return records;
        }
    }

    // Parse records
    size_t offset = 0;
    for (uint64_t i = 0; i < header.record_count; i++) {
        if (offset + sizeof(LineageSnapshotEntry) > data_size) break;

        LineageSnapshotEntry entry{};
        std::memcpy(&entry, data_buf.data() + offset, sizeof(entry));
        offset += sizeof(entry);

        SnapshotRecord rec;
        rec.child_id = entry.child_id;
        rec.op = static_cast<LineageOp>(entry.lineage_op);
        rec.created_at = entry.created_at;

        if (entry.parent_count > 0) {
            size_t parents_bytes = entry.parent_count * sizeof(uint64_t);
            if (offset + parents_bytes > data_size) break;
            rec.parent_ids.resize(entry.parent_count);
            std::memcpy(rec.parent_ids.data(), data_buf.data() + offset, parents_bytes);
            offset += parents_bytes;
        }

        records.push_back(std::move(rec));
    }

    // Parse removed_nodes
    if (offset + sizeof(uint64_t) <= data_size) {
        uint64_t removed_count = 0;
        std::memcpy(&removed_count, data_buf.data() + offset, sizeof(uint64_t));
        offset += sizeof(uint64_t);

        size_t removed_bytes = static_cast<size_t>(removed_count) * sizeof(uint64_t);
        if (offset + removed_bytes <= data_size) {
            removed_nodes_out.resize(static_cast<size_t>(removed_count));
            std::memcpy(removed_nodes_out.data(), data_buf.data() + offset, removed_bytes);
        }
    }

    spdlog::info("LineageWAL: loaded snapshot with {} records, {} removed nodes",
                 records.size(), removed_nodes_out.size());
    return records;
}

void LineageWAL::sync() {
    std::lock_guard lock(mutex_);
    if (wal_file_.is_open()) {
        wal_file_.flush();
        amind::fsyncFile(wal_path_);
    }
}

size_t LineageWAL::walSize() const {
    std::lock_guard lock(mutex_);
    namespace fs = std::filesystem;
    if (fs::exists(wal_path_)) {
        return static_cast<size_t>(fs::file_size(wal_path_));
    }
    return 0;
}

}  // namespace amind
