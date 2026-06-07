#pragma once

#include "result.h"
#include "types.h"

#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace amind {

/// Wire format constants for the binary record header.
struct WireFormat {
    static constexpr uint32_t MAGIC = 0xAB1D0001;
    static constexpr uint16_t VERSION = 1;
    static constexpr size_t HEADER_SIZE = 64;
    static constexpr size_t MAX_CONTENT_SIZE = 65535;
    static constexpr size_t MAX_METADATA_SIZE = 65535;
    static constexpr size_t MAX_EMBEDDING_DIM = 65535;
};

/// On-disk / on-wire header for a MemoryRecord.
/// Exactly 64 bytes, mmap-friendly, zero-copy readable.
///
/// V2 Layout (multi-user/multi-agent — replaces namespace_hash + owner with
///            scope/memory_type/tier and moves user_id/agent_id to variable-length section):
///   [0..4)    magic:            u32 = 0xAB1D0001
///   [4..6)    version:          u16 = 2 (wire format version)
///   [6..8)    flags:            u16 (RecordFlags bitmask)
///   [8..16)   memory_id:        u64 (Snowflake ID)
///   [16]      scope:            u8  (MemoryScope enum)
///   [17]      memory_type:      u8  (MemoryType enum)
///   [18]      tier:             u8  (MemoryTier enum)
///   [19]      phase:            u8  (MemoryPhase enum)
///   [20]      confidence_level: u8  (Confidence enum)
///   [21..24)  _reserved:        u8[3]
///   [24..28)  importance:       f32 [0.0, 1.0]
///   [28..32)  confidence_score: f32 [0.0, 1.0]
///   [32..36)  access_count:     u32
///   [36..40)  mem_version:      u32 (version counter, 1-based)
///   [40..48)  parent_id:        u64 (previous version's memory_id, 0 if first)
///   [48..52)  created_at:       u32 (unix timestamp seconds)
///   [52..56)  last_accessed:    u32 (unix timestamp seconds)
///   [56..58)  embedding_dim:    u16
///   [58..60)  content_len:      u16
///   [60..62)  metadata_len:     u16
///   [62..64)  source_turn:      u16 (conversation turn number)
///
/// After header (variable-length fields):
///   user_id:    u16 length + UTF-8 bytes
///   agent_id:   u16 length + UTF-8 bytes
///   embedding:  f32[] (if HAS_EMBEDDING flag set, embedding_dim elements)
///   content:    UTF-8 bytes (content_len bytes)
///   metadata:   MessagePack bytes (metadata_len bytes)
///   checksum:   CRC32C trailer (4 bytes)
#pragma pack(push, 1)
struct RecordHeader {
    uint32_t magic{WireFormat::MAGIC};
    uint16_t version{2};              // V2 wire format
    uint16_t flags{RecordFlags::ALIVE};
    uint64_t memory_id{0};
    uint8_t scope{0};                 // MemoryScope
    uint8_t memory_type{0};           // MemoryType
    uint8_t tier{0};                  // MemoryTier
    uint8_t phase{0};                 // MemoryPhase
    uint8_t confidence_level{0};      // Confidence enum
    uint8_t _reserved[3]{};
    float importance{0.0f};
    float confidence_score{0.5f};     // floating-point confidence [0.0, 1.0]
    uint32_t access_count{0};
    uint32_t mem_version{1};          // version counter (1 = first version)
    uint64_t parent_id{0};            // 0 means no parent (first version)
    uint32_t created_at{0};           // unix timestamp seconds
    uint32_t last_accessed{0};
    uint16_t embedding_dim{0};
    uint16_t content_len{0};
    uint16_t metadata_len{0};
    uint16_t source_turn{0};
};
#pragma pack(pop)

static_assert(sizeof(RecordHeader) == WireFormat::HEADER_SIZE,
              "RecordHeader must be exactly 64 bytes");

/// In-memory representation of a memory record.
/// V2: Multi-user/multi-agent model with scope, memory_type, tier.
struct MemoryRecord {
    // ── Identity ──────────────────────────────────────────────────────────
    uint64_t memory_id{0};
    std::string user_id;                           // creator user ID
    std::string agent_id;                          // owning agent ID
    MemoryScope scope{MemoryScope::Private};       // visibility scope
    MemoryType memory_type{MemoryType::Ephemeral}; // semantic type
    MemoryTier tier{MemoryTier::Ephemeral};        // quality tier (fast/slow)
    MemoryPhase phase{MemoryPhase::Active};
    Confidence confidence_level{Confidence::Inferred};
    float confidence_score{0.5f};                  // numeric confidence [0.0, 1.0]
    uint16_t flags{RecordFlags::ALIVE};

    // ── Scoring ───────────────────────────────────────────────────────────
    float importance{0.5f};
    uint32_t access_count{0};

    // ── Versioning ────────────────────────────────────────────────────────
    uint32_t mem_version{1};   // starts at 1
    uint64_t parent_id{0};     // 0 = first version (no parent)

    // ── Timestamps ────────────────────────────────────────────────────────
    uint32_t created_at{0};     // unix timestamp seconds
    uint32_t last_accessed{0};  // unix timestamp seconds

    // ── Context ───────────────────────────────────────────────────────────
    uint16_t source_turn{0};    // which conversation turn produced this memory

    // ── Payload ───────────────────────────────────────────────────────────
    std::vector<float> embedding;
    std::string content;
    std::vector<uint8_t> metadata;  // MessagePack-encoded (V2 fields + user_metadata as JSON)

    // ── User-supplied metadata (string→string map) ────────────────────────
    // Persisted by merging into the metadata JSON under the "user" key.
    // Used by clients (pyclaw/hermes) for session_id, role, filename, etc.
    std::map<std::string, std::string> user_metadata;

    // ── V2: Storage Layer & Lineage ───────────────────────────────────────
    MemoryLayer layer{MemoryLayer::Raw};
    std::vector<uint64_t> lineage_parents;   // IDs of parent memories this was derived from
    LineageOp lineage_op{LineageOp::None};   // How this memory was derived

    // ── V2: Write Gate Metadata ───────────────────────────────────────────
    SourceTier source_tier{SourceTier::Inference};
    GateDecision gate_decision{GateDecision::Accepted};
    float marginal_value{0.0f};              // WriteGate marginal value score

    // ── V2: Forget Engine Metadata ────────────────────────────────────────
    float forget_score{0.0f};                // Last GC-computed forget score
    uint32_t last_gc_visit{0};               // Timestamp of last GC evaluation
    uint8_t resurrection_count{0};           // Times revived from Archived/Stale

    // ── Serialization ─────────────────────────────────────────────────────

    /// Serialize to binary (header + body + CRC32 trailer).
    [[nodiscard]] std::vector<uint8_t> serialize() const;

    /// Deserialize from binary. Returns error on corrupt data.
    [[nodiscard]] static Result<MemoryRecord> deserialize(std::span<const uint8_t> data);

    /// Total serialized size.
    [[nodiscard]] size_t serializedSize() const;

    // ── Convenience ───────────────────────────────────────────────────────

    /// Current time in seconds since Unix epoch.
    [[nodiscard]] static uint32_t currentTimeSec();

    /// Current time in milliseconds since Unix epoch.
    [[nodiscard]] static uint64_t currentTimeMs();

    // ── Phase transitions ─────────────────────────────────────────────────

    [[nodiscard]] bool isActive() const;
    [[nodiscard]] bool isAlive() const;  // Active and not tombstoned
    void markVersioned();
    void markArchived();
    void markTombstone();
    void markInvalidated();  // V2: Mark as lineage orphan

    // ── Version management ────────────────────────────────────────────────

    /// Create a new version of this record. Copies content, bumps version,
    /// links parent_id, generates new memory_id.
    [[nodiscard]] MemoryRecord createNewVersion(uint64_t new_memory_id) const;
};

}  // namespace amind
