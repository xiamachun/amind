#include "sstable_writer.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

#include <fcntl.h>
#include <unistd.h>

namespace amind {

SSTableWriter::SSTableWriter(const std::filesystem::path& file_path,
                             size_t expected_entries)
    : file_path_(file_path)
    , bloom_filter_(expected_entries, SSTableFormat::BLOOM_BITS_PER_KEY) {
    file_buffer_.reserve(64 * 1024);  // Pre-allocate 64 KB
}

SSTableWriter::~SSTableWriter() {
    if (!finished_ && entry_count_ > 0) {
        try {
            finish();
        } catch (...) {
            // Suppress exceptions in destructor
        }
    }
}

void SSTableWriter::add(uint64_t key, std::span<const uint8_t> value) {
    if (finished_) {
        throw std::logic_error("Cannot add entries after finish()");
    }
    if (entry_count_ > 0 && key <= last_key_) {
        throw std::invalid_argument("Keys must be added in strictly ascending order");
    }

    // Add to bloom filter
    bloom_filter_.add(key);

    // Add to current data block
    block_builder_.add(key, value);
    ++entry_count_;
    last_key_ = key;

    // Flush block if it has reached the target size
    if (block_builder_.isFull()) {
        flushDataBlock();
    }
}

size_t SSTableWriter::finish() {
    if (finished_) {
        throw std::logic_error("finish() already called");
    }
    finished_ = true;

    // Flush any remaining entries in the current block
    if (!block_builder_.empty()) {
        flushDataBlock();
    }

    // ── Write Index Block ──────────────────────────────────────────────────
    uint64_t index_offset = file_buffer_.size();

    // Serialize all index entries
    std::vector<uint8_t> index_data(index_entries_.size() * IndexEntry::SERIALIZED_SIZE);
    for (size_t i = 0; i < index_entries_.size(); ++i) {
        index_entries_[i].writeTo(index_data.data() + i * IndexEntry::SERIALIZED_SIZE);
    }
    file_buffer_.insert(file_buffer_.end(), index_data.begin(), index_data.end());
    auto index_size = static_cast<uint32_t>(index_data.size());

    // ── Write Bloom Filter Block ───────────────────────────────────────────
    uint64_t bloom_offset = file_buffer_.size();
    std::vector<uint8_t> bloom_data = bloom_filter_.serialize();
    file_buffer_.insert(file_buffer_.end(), bloom_data.begin(), bloom_data.end());
    auto bloom_size = static_cast<uint32_t>(bloom_data.size());

    // ── Write Footer ───────────────────────────────────────────────────────
    SSTableFooter footer{};
    footer.index_offset = index_offset;
    footer.index_size = index_size;
    footer.bloom_filter_offset = bloom_offset;
    footer.bloom_filter_size = bloom_size;
    footer.entry_count = static_cast<uint32_t>(entry_count_);
    footer.version = SSTableFormat::VERSION;
    footer.magic = SSTableFormat::MAGIC;

    const auto* footer_bytes = reinterpret_cast<const uint8_t*>(&footer);
    file_buffer_.insert(file_buffer_.end(), footer_bytes, footer_bytes + SSTableFooter::SIZE);

    // ── Write to disk ──────────────────────────────────────────────────────
    writeFile();

    return file_buffer_.size();
}

void SSTableWriter::flushDataBlock() {
    if (block_builder_.empty()) {
        return;
    }

    uint64_t block_offset = file_buffer_.size();
    uint64_t first_key = block_builder_.firstKey();

    // Compress and get the block bytes
    std::vector<uint8_t> compressed = block_builder_.finish();

    // Record index entry
    IndexEntry index_entry{};
    index_entry.first_key = first_key;
    index_entry.offset = block_offset;
    index_entry.compressed_size = static_cast<uint32_t>(compressed.size());
    index_entries_.push_back(index_entry);

    // Append compressed block to file buffer
    file_buffer_.insert(file_buffer_.end(), compressed.begin(), compressed.end());
}

void SSTableWriter::writeFile() {
    std::ofstream out(file_path_, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open SSTable file for writing: " +
                                 file_path_.string());
    }
    out.write(reinterpret_cast<const char*>(file_buffer_.data()),
              static_cast<std::streamsize>(file_buffer_.size()));
    if (!out.good()) {
        throw std::runtime_error("Failed to write SSTable file: " + file_path_.string());
    }
    out.flush();

    int fd = ::open(file_path_.c_str(), O_RDONLY);
    if (fd >= 0) {
        ::fsync(fd);
        ::close(fd);
    }
}

}  // namespace amind
