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
/// Layout (differs from ClawMind — adds owner/phase/confidence/version/parent_id):
///   [0..4)    magic:          u32 = 0xAB1D0001
///   [4..6)    version:        u16 = 1 (wire format version)
///   [6..8)    flags:          u16 (RecordFlags bitmask)
///   [8..16)   memory_id:      u64 (Snowflake ID)
///   [16..24)  namespace_hash: u64 (xxHash64 of namespace string — tenant isolation key)
///   [24]      owner:          u8  (MemoryOwner enum)
///   [25]      phase:          u8  (MemoryPhase enum)
///   [26]      confidence:     u8  (Confidence enum)
///   [27]      _reserved:      u8
///   [28..32)  importance:     f32 [0.0, 1.0]
///   [32..36)  access_count:   u32
///   [36..40)  mem_version:    u32 (version counter, 1-based)
///   [40..48)  parent_id:      u64 (previous version's memory_id, 0 if first)
///   [48..52)  created_at:     u32 (unix timestamp seconds)
///   [52..56)  last_accessed:  u32 (unix timestamp seconds)
///   [56..58)  embedding_dim:  u16
///   [58..60)  content_len:    u16
///   [60..62)  metadata_len:   u16
///   [62..64)  source_turn:    u16 (conversation turn number)
///
/// After header:
///   [64 .. 64 + embedding_dim*4)  embedding: f32[] (if HAS_EMBEDDING flag set)
///   [... + content_len)           content:   UTF-8 bytes
///   [... + metadata_len)          metadata:  MessagePack bytes
///   [last 4 bytes)                checksum:  CRC32C trailer
#pragma pack(push, 1)
struct RecordHeader {
    uint32_t magic{WireFormat::MAGIC};
    uint16_t version{WireFormat::VERSION};
    uint16_t flags{RecordFlags::ALIVE};
    uint64_t memory_id{0};
    uint64_t namespace_hash{0};
    uint8_t owner{0};         // MemoryOwner
    uint8_t phase{0};         // MemoryPhase
    uint8_t confidence{0};    // Confidence
    uint8_t _reserved{0};
    float importance{0.0f};
    uint32_t access_count{0};
    uint32_t mem_version{1};  // version counter (1 = first version)
    uint64_t parent_id{0};    // 0 means no parent (first version)
    uint32_t created_at{0};   // unix timestamp seconds
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
/// Extends ClawMind's MemoryRecord with ownership, phases, confidence, and version history.
struct MemoryRecord {
    // ── Identity ──────────────────────────────────────────────────────────
    uint64_t memory_id{0};
    uint64_t namespace_hash{0};
    MemoryOwner owner{MemoryOwner::Session};
    MemoryPhase phase{MemoryPhase::Active};
    Confidence confidence{Confidence::Inferred};
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

    /// Hash a namespace string → xxHash64 (the tenant isolation key).
    [[nodiscard]] static uint64_t hashNamespace(std::string_view ns);

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
