#include "memtable.h"

namespace amind {

MemTable::MemTable(size_t size_threshold)
    : size_threshold_(size_threshold) {}

void MemTable::put(const MemoryRecord& record) {
    auto serialized = record.serialize();
    uint64_t key = record.memory_id;
    skip_list_.put(key, std::move(serialized));
}

void MemTable::putRaw(uint64_t memory_id, std::vector<uint8_t> serialized_data) {
    skip_list_.put(memory_id, std::move(serialized_data));
}

std::optional<MemoryRecord> MemTable::get(uint64_t memory_id) const {
    auto raw = skip_list_.get(memory_id);
    if (!raw.has_value()) {
        return std::nullopt;
    }
    auto result = MemoryRecord::deserialize(raw.value());
    if (!result.ok()) {
        return std::nullopt;
    }
    return std::move(result.value());
}

std::optional<std::vector<uint8_t>> MemTable::getRaw(uint64_t memory_id) const {
    return skip_list_.get(memory_id);
}

bool MemTable::remove(uint64_t memory_id) {
    return skip_list_.remove(memory_id);
}

bool MemTable::contains(uint64_t memory_id) const {
    return skip_list_.contains(memory_id);
}

size_t MemTable::size() const {
    return skip_list_.size();
}

bool MemTable::empty() const {
    return skip_list_.empty();
}

size_t MemTable::approximateMemoryUsage() const {
    return skip_list_.approximateMemoryUsage();
}

bool MemTable::shouldFlush() const {
    return skip_list_.approximateMemoryUsage() >= size_threshold_;
}

void MemTable::forEach(const EntryVisitor& visitor) const {
    auto it = skip_list_.begin();
    while (it.valid()) {
        visitor(it.key(), it.value(), it.isTombstone());
        it.next();
    }
}

std::vector<uint64_t> MemTable::allKeys() const {
    return skip_list_.allKeys();
}

}  // namespace amind
