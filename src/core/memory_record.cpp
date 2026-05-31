#include "memory_record.h"

#include <chrono>
#include <crc32c/crc32c.h>
#include <cstring>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <xxhash.h>

namespace amind {

// ── Serialization ───────────────────────────────────────────────────────────

std::vector<uint8_t> MemoryRecord::serialize() const {
    // Bounds check: warn and clamp if fields exceed uint16 capacity
    if (content.size() > WireFormat::MAX_CONTENT_SIZE) {
        spdlog::error("MemoryRecord::serialize: content size {} exceeds max {}, truncating",
                      content.size(), WireFormat::MAX_CONTENT_SIZE);
    }
    if (embedding.size() > WireFormat::MAX_EMBEDDING_DIM) {
        spdlog::error("MemoryRecord::serialize: embedding dim {} exceeds max {}",
                      embedding.size(), WireFormat::MAX_EMBEDDING_DIM);
    }

    // Encode V2 fields and user_metadata into metadata blob as JSON.
    // user_metadata lives under the "user" sub-object so V2 fields and
    // client-supplied keys can coexist without collisions.
    std::vector<uint8_t> effective_metadata = metadata;
    bool has_v2 = (layer != MemoryLayer::Raw) || !lineage_parents.empty() ||
                  (lineage_op != LineageOp::None) || (source_tier != SourceTier::Inference) ||
                  (gate_decision != GateDecision::Accepted) || (marginal_value != 0.0f) ||
                  (forget_score != 0.0f) || (last_gc_visit != 0) || (resurrection_count != 0);
    bool has_user = !user_metadata.empty();
    if (has_v2 || has_user) {
        nlohmann::json j;
        if (has_v2) {
            j["layer"] = static_cast<uint8_t>(layer);
            if (!lineage_parents.empty()) j["lineage_parents"] = lineage_parents;
            if (lineage_op != LineageOp::None) j["lineage_op"] = static_cast<uint8_t>(lineage_op);
            if (source_tier != SourceTier::Inference) j["source_tier"] = static_cast<uint8_t>(source_tier);
            if (gate_decision != GateDecision::Accepted) j["gate_decision"] = static_cast<uint8_t>(gate_decision);
            if (marginal_value != 0.0f) j["marginal_value"] = marginal_value;
            if (forget_score != 0.0f) j["forget_score"] = forget_score;
            if (last_gc_visit != 0) j["last_gc_visit"] = last_gc_visit;
            if (resurrection_count != 0) j["resurrection_count"] = resurrection_count;
        }
        if (has_user) {
            nlohmann::json u = nlohmann::json::object();
            for (const auto& [k, v] : user_metadata) u[k] = v;
            j["user"] = std::move(u);
        }
        auto j_str = j.dump();
        effective_metadata.assign(j_str.begin(), j_str.end());
    }

    if (effective_metadata.size() > WireFormat::MAX_METADATA_SIZE) {
        spdlog::error("MemoryRecord::serialize: metadata size {} exceeds max {}, truncating",
                      effective_metadata.size(), WireFormat::MAX_METADATA_SIZE);
    }

    uint16_t emb_dim = static_cast<uint16_t>(std::min(embedding.size(), WireFormat::MAX_EMBEDDING_DIM));
    uint16_t cont_len = static_cast<uint16_t>(std::min(content.size(), WireFormat::MAX_CONTENT_SIZE));
    uint16_t meta_len = static_cast<uint16_t>(std::min(effective_metadata.size(), WireFormat::MAX_METADATA_SIZE));

    size_t total_size = WireFormat::HEADER_SIZE
                      + static_cast<size_t>(emb_dim) * sizeof(float)
                      + cont_len + meta_len + sizeof(uint32_t);
    std::vector<uint8_t> buffer(total_size);

    // Build header
    RecordHeader header;
    header.magic = WireFormat::MAGIC;
    header.version = WireFormat::VERSION;
    header.flags = flags;
    header.memory_id = memory_id;
    header.namespace_hash = namespace_hash;
    header.owner = static_cast<uint8_t>(owner);
    header.phase = static_cast<uint8_t>(phase);
    header.confidence = static_cast<uint8_t>(confidence);
    header._reserved = 0;
    header.importance = importance;
    header.access_count = access_count;
    header.mem_version = mem_version;
    header.parent_id = parent_id;
    header.created_at = created_at;
    header.last_accessed = last_accessed;
    header.embedding_dim = emb_dim;
    header.content_len = cont_len;
    header.metadata_len = meta_len;
    header.source_turn = source_turn;

    // Write header (64 bytes)
    std::memcpy(buffer.data(), &header, WireFormat::HEADER_SIZE);

    size_t offset = WireFormat::HEADER_SIZE;

    // Write embedding (f32 array)
    if (emb_dim > 0) {
        size_t embedding_bytes = static_cast<size_t>(emb_dim) * sizeof(float);
        std::memcpy(buffer.data() + offset, embedding.data(), embedding_bytes);
        offset += embedding_bytes;
    }

    // Write content (UTF-8 string)
    if (cont_len > 0) {
        std::memcpy(buffer.data() + offset, content.data(), cont_len);
        offset += cont_len;
    }

    // Write metadata
    if (meta_len > 0) {
        std::memcpy(buffer.data() + offset, effective_metadata.data(), meta_len);
        offset += meta_len;
    }

    // Compute and write CRC32C trailer
    uint32_t checksum = crc32c::Crc32c(buffer.data(), offset);
    std::memcpy(buffer.data() + offset, &checksum, sizeof(uint32_t));

    return buffer;
}

Result<MemoryRecord> MemoryRecord::deserialize(std::span<const uint8_t> data) {
    // Minimum: header (64) + CRC32 trailer (4)
    if (data.size() < WireFormat::HEADER_SIZE + sizeof(uint32_t)) {
        return makeError(Error::CorruptedData, "data too small for header + CRC");
    }

    // Read header
    RecordHeader header;
    std::memcpy(&header, data.data(), WireFormat::HEADER_SIZE);

    // Validate magic
    if (header.magic != WireFormat::MAGIC) {
        return makeError(Error::CorruptedData, "bad magic number");
    }

    // Validate wire format version
    if (header.version != WireFormat::VERSION) {
        return makeError(Error::CorruptedData,
                         "unsupported version: " + std::to_string(header.version));
    }

    // Validate enum ranges
    if (header.owner > static_cast<uint8_t>(MemoryOwner::Shared)) {
        return makeError(Error::CorruptedData, "invalid owner value");
    }
    if (header.phase > static_cast<uint8_t>(MemoryPhase::Invalidated)) {
        return makeError(Error::CorruptedData, "invalid phase value");
    }
    if (header.confidence > static_cast<uint8_t>(Confidence::Conflicted)) {
        return makeError(Error::CorruptedData, "invalid confidence value");
    }

    // Calculate expected total size
    size_t body_size = static_cast<size_t>(header.embedding_dim) * sizeof(float)
                     + header.content_len
                     + header.metadata_len;
    size_t expected_size = WireFormat::HEADER_SIZE + body_size + sizeof(uint32_t);

    if (data.size() < expected_size) {
        return makeError(Error::CorruptedData, "data too small for declared body");
    }

    // Verify CRC32C
    size_t payload_size = WireFormat::HEADER_SIZE + body_size;
    uint32_t stored_checksum;
    std::memcpy(&stored_checksum, data.data() + payload_size, sizeof(uint32_t));
    uint32_t computed_checksum = crc32c::Crc32c(data.data(), payload_size);
    if (stored_checksum != computed_checksum) {
        return makeError(Error::CrcMismatch, "CRC32C checksum mismatch");
    }

    // Build MemoryRecord
    MemoryRecord record;
    record.memory_id = header.memory_id;
    record.namespace_hash = header.namespace_hash;
    record.owner = static_cast<MemoryOwner>(header.owner);
    record.phase = static_cast<MemoryPhase>(header.phase);
    record.confidence = static_cast<Confidence>(header.confidence);
    record.flags = header.flags;
    record.importance = header.importance;
    record.access_count = header.access_count;
    record.mem_version = header.mem_version;
    record.parent_id = header.parent_id;
    record.created_at = header.created_at;
    record.last_accessed = header.last_accessed;
    record.source_turn = header.source_turn;

    size_t offset = WireFormat::HEADER_SIZE;

    // Read embedding
    if (header.embedding_dim > 0) {
        record.embedding.resize(header.embedding_dim);
        std::memcpy(record.embedding.data(), data.data() + offset,
                    header.embedding_dim * sizeof(float));
        offset += header.embedding_dim * sizeof(float);
    }

    // Read content
    if (header.content_len > 0) {
        record.content.assign(
            reinterpret_cast<const char*>(data.data() + offset),
            header.content_len);
        offset += header.content_len;
    }

    // Read metadata
    if (header.metadata_len > 0) {
        record.metadata.assign(
            data.data() + offset,
            data.data() + offset + header.metadata_len);

        // Attempt to decode V2 fields from metadata (JSON format)
        try {
            auto j = nlohmann::json::parse(record.metadata.begin(), record.metadata.end());
            if (j.contains("layer"))
                record.layer = static_cast<MemoryLayer>(j["layer"].get<uint8_t>());
            if (j.contains("lineage_parents"))
                record.lineage_parents = j["lineage_parents"].get<std::vector<uint64_t>>();
            if (j.contains("lineage_op"))
                record.lineage_op = static_cast<LineageOp>(j["lineage_op"].get<uint8_t>());
            if (j.contains("source_tier"))
                record.source_tier = static_cast<SourceTier>(j["source_tier"].get<uint8_t>());
            if (j.contains("gate_decision"))
                record.gate_decision = static_cast<GateDecision>(j["gate_decision"].get<uint8_t>());
            if (j.contains("marginal_value"))
                record.marginal_value = j["marginal_value"].get<float>();
            if (j.contains("forget_score"))
                record.forget_score = j["forget_score"].get<float>();
            if (j.contains("last_gc_visit"))
                record.last_gc_visit = j["last_gc_visit"].get<uint32_t>();
            if (j.contains("resurrection_count"))
                record.resurrection_count = j["resurrection_count"].get<uint8_t>();
            if (j.contains("user") && j["user"].is_object()) {
                for (auto it = j["user"].begin(); it != j["user"].end(); ++it) {
                    if (it.value().is_string()) {
                        record.user_metadata[it.key()] = it.value().get<std::string>();
                    } else {
                        // Non-string values: stringify so we never lose data.
                        record.user_metadata[it.key()] = it.value().dump();
                    }
                }
            }
        } catch (...) {
            // metadata is not V2 JSON — leave V2 fields at defaults
        }
    }

    return record;
}

size_t MemoryRecord::serializedSize() const {
    return WireFormat::HEADER_SIZE
         + embedding.size() * sizeof(float)
         + content.size()
         + metadata.size()
         + sizeof(uint32_t);  // CRC32 trailer
}

// ── Convenience ─────────────────────────────────────────────────────────────

uint64_t MemoryRecord::hashNamespace(std::string_view ns) {
    return XXH64(ns.data(), ns.size(), 0);
}

uint32_t MemoryRecord::currentTimeSec() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::seconds>(duration).count());
}

uint64_t MemoryRecord::currentTimeMs() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());
}

// ── Phase transitions ───────────────────────────────────────────────────────

bool MemoryRecord::isActive() const {
    return phase == MemoryPhase::Active && (flags & RecordFlags::ALIVE) != 0;
}

bool MemoryRecord::isAlive() const {
    return (flags & RecordFlags::TOMBSTONE) == 0
        && phase != MemoryPhase::Tombstone
        && phase != MemoryPhase::Invalidated;
}

void MemoryRecord::markVersioned() {
    phase = MemoryPhase::Versioned;
    flags &= ~RecordFlags::ALIVE;
    flags |= RecordFlags::VERSIONED;
}

void MemoryRecord::markArchived() {
    phase = MemoryPhase::Archived;
    flags |= RecordFlags::ARCHIVED;
}

void MemoryRecord::markTombstone() {
    phase = MemoryPhase::Tombstone;
    flags &= ~RecordFlags::ALIVE;
    flags |= RecordFlags::TOMBSTONE;
}

void MemoryRecord::markInvalidated() {
    phase = MemoryPhase::Invalidated;
    flags &= ~RecordFlags::ALIVE;
    flags |= RecordFlags::LINEAGE_ORPHAN;
}

// ── Version management ──────────────────────────────────────────────────────

MemoryRecord MemoryRecord::createNewVersion(uint64_t new_memory_id) const {
    MemoryRecord new_record = *this;
    new_record.memory_id = new_memory_id;
    new_record.parent_id = this->memory_id;
    new_record.mem_version = this->mem_version + 1;
    new_record.phase = MemoryPhase::Active;
    new_record.flags = RecordFlags::ALIVE;
    new_record.created_at = currentTimeSec();
    new_record.last_accessed = new_record.created_at;
    new_record.access_count = 0;
    return new_record;
}

}  // namespace amind
