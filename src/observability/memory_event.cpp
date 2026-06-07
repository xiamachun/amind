#include "memory_event.h"

#include <chrono>

namespace amind {

namespace {
// Keep this table in sync with EventKind. Order matters for the reverse lookup.
constexpr struct { EventKind k; const char* s; } kKindTable[] = {
    {EventKind::Store,            "Store"},
    {EventKind::Embed,            "Embed"},
    {EventKind::Gate,             "Gate"},
    {EventKind::Derive,           "Derive"},
    {EventKind::Reconcile,        "Reconcile"},
    {EventKind::LineagePropagate, "LineagePropagate"},
    {EventKind::GraphEdge,        "GraphEdge"},
    {EventKind::VersionUpdate,    "VersionUpdate"},
    {EventKind::Recall,           "Recall"},
    {EventKind::RecallSemantic,   "RecallSemantic"},
    {EventKind::RecallFilter,     "RecallFilter"},
    {EventKind::RecallStale,      "RecallStale"},
    {EventKind::GcDecay,          "GcDecay"},
    {EventKind::GcArchive,        "GcArchive"},
    {EventKind::GcTombstone,      "GcTombstone"},
    {EventKind::GcVacuum,         "GcVacuum"},
    {EventKind::Consolidate,      "Consolidate"},
    {EventKind::Resurrect,        "Resurrect"},
    {EventKind::ProviderCall,     "ProviderCall"},
    {EventKind::Error,            "Error"},
};

constexpr struct { EventStatus s; const char* str; } kStatusTable[] = {
    {EventStatus::Ok,       "Ok"},
    {EventStatus::Rejected, "Rejected"},
    {EventStatus::Deferred, "Deferred"},
    {EventStatus::Failed,   "Failed"},
    {EventStatus::NoOp,     "NoOp"},
};
}  // namespace

std::string kindToString(EventKind k) {
    for (const auto& e : kKindTable) {
        if (e.k == k) return e.s;
    }
    return "Unknown";
}

EventKind kindFromString(const std::string& s) {
    for (const auto& e : kKindTable) {
        if (s == e.s) return e.k;
    }
    return EventKind::Error;  // fallback signals an unknown kind to the caller
}

std::string statusToString(EventStatus s) {
    for (const auto& e : kStatusTable) {
        if (e.s == s) return e.str;
    }
    return "Unknown";
}

EventStatus statusFromString(const std::string& s) {
    for (const auto& e : kStatusTable) {
        if (s == e.str) return e.s;
    }
    return EventStatus::Ok;
}

uint64_t nowMs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
}

std::string truncateUtf8(const std::string& s, size_t max_bytes) {
    if (s.size() <= max_bytes) return s;
    size_t cut = max_bytes;
    // UTF-8 continuation bytes have top two bits 10. Walk back while we're
    // on one, so we land on a code-point boundary. Bounded by max_bytes
    // iterations in the worst case (impossible if the source was valid).
    while (cut > 0 && (static_cast<unsigned char>(s[cut]) & 0xC0) == 0x80) {
        --cut;
    }
    return s.substr(0, cut) + "…";
}

}  // namespace amind
