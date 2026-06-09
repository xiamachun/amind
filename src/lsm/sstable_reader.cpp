#include "sstable_reader.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace amind {

SSTableReader::SSTableReader(const std::filesystem::path& file_path)
    : file_path_(file_path)
    , mapped_data_(nullptr)
    , mapped_size_(0)
    , fd_(-1)
    , using_mmap_(false)
    , bloom_filter_(1, 10) {  // Placeholder; will be replaced in load()
    load();
}

SSTableReader::~SSTableReader() {
    if (using_mmap_ && mapped_data_ != nullptr) {
        munmap(const_cast<uint8_t*>(mapped_data_), mapped_size_);
        mapped_data_ = nullptr;
        mapped_size_ = 0;
    }
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
}

void SSTableReader::load() {
    // Try mmap first
    fd_ = open(file_path_.c_str(), O_RDONLY);
    if (fd_ != -1) {
        struct stat st;
        if (fstat(fd_, &st) == 0) {
            mapped_size_ = static_cast<size_t>(st.st_size);
            if (mapped_size_ >= SSTableFooter::SIZE) {
                mapped_data_ = static_cast<const uint8_t*>(
                    mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
                
                if (mapped_data_ != MAP_FAILED) {
                    // Advise sequential access pattern
                    madvise(const_cast<uint8_t*>(mapped_data_), mapped_size_, MADV_SEQUENTIAL);
                    using_mmap_ = true;
                    spdlog::debug("SSTableReader: using mmap for {}", file_path_.string());
                } else {
                    mapped_data_ = nullptr;
                    close(fd_);
                    fd_ = -1;
                    spdlog::debug("SSTableReader: mmap failed, falling back to traditional read for {}", file_path_.string());
                }
            } else {
                close(fd_);
                fd_ = -1;
            }
        } else {
            close(fd_);
            fd_ = -1;
        }
    }

    // Fallback to traditional read if mmap failed
    if (!using_mmap_) {
        std::ifstream in(file_path_, std::ios::binary | std::ios::ate);
        if (!in.is_open()) {
            throw std::runtime_error("Failed to open SSTable file: " + file_path_.string());
        }

        auto file_size = static_cast<size_t>(in.tellg());
        if (file_size < SSTableFooter::SIZE) {
            throw std::invalid_argument("SSTable file too small: " + file_path_.string());
        }

        file_data_.resize(file_size);
        in.seekg(0);
        in.read(reinterpret_cast<char*>(file_data_.data()),
                static_cast<std::streamsize>(file_size));
    }

    // Get data pointer and size based on method used
    const uint8_t* data_ptr = using_mmap_ ? mapped_data_ : file_data_.data();
    size_t data_size = using_mmap_ ? mapped_size_ : file_data_.size();

    // ── Parse Footer (last 40 bytes) ───────────────────────────────────────
    std::memcpy(&footer_, data_ptr + data_size - SSTableFooter::SIZE,
                SSTableFooter::SIZE);

    if (!footer_.isValid()) {
        throw std::invalid_argument("Invalid SSTable footer (bad magic or version)");
    }

    // ── Parse Index Block ──────────────────────────────────────────────────
    if (footer_.index_offset + footer_.index_size > data_size - SSTableFooter::SIZE) {
        throw std::invalid_argument("Index block extends beyond file bounds");
    }

    size_t num_index_entries = footer_.index_size / IndexEntry::SERIALIZED_SIZE;
    index_entries_.reserve(num_index_entries);

    const uint8_t* index_ptr = data_ptr + footer_.index_offset;
    for (size_t i = 0; i < num_index_entries; ++i) {
        index_entries_.push_back(
            IndexEntry::readFrom(index_ptr + i * IndexEntry::SERIALIZED_SIZE));
    }

    // ── Parse Bloom Filter Block ───────────────────────────────────────────
    if (footer_.bloom_filter_offset + footer_.bloom_filter_size >
        data_size - SSTableFooter::SIZE) {
        throw std::invalid_argument("Bloom filter block extends beyond file bounds");
    }

    std::span<const uint8_t> bloom_data(
        data_ptr + footer_.bloom_filter_offset,
        footer_.bloom_filter_size);
    bloom_filter_ = BloomFilter::deserialize(bloom_data);
}

bool SSTableReader::mayContain(uint64_t key) const {
    return bloom_filter_.mayContain(key);
}

int SSTableReader::findBlockIndex(uint64_t key) const {
    if (index_entries_.empty()) {
        return -1;
    }

    // Binary search: find the last block whose first_key <= key
    int low = 0;
    int high = static_cast<int>(index_entries_.size()) - 1;
    int result = -1;

    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (index_entries_[mid].first_key <= key) {
            result = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    return result;
}

BlockReader SSTableReader::readBlock(const IndexEntry& entry) const {
    const uint8_t* data_ptr = using_mmap_ ? mapped_data_ : file_data_.data();
    size_t data_size = using_mmap_ ? mapped_size_ : file_data_.size();

    if (entry.offset + entry.compressed_size > data_size) {
        throw std::runtime_error("Data block extends beyond file bounds");
    }

    std::span<const uint8_t> block_data(
        data_ptr + entry.offset,
        entry.compressed_size);

    return BlockReader(block_data);
}

std::optional<std::vector<uint8_t>> SSTableReader::get(uint64_t key) const {
    // Bloom filter check: skip if definitely not present
    if (!bloom_filter_.mayContain(key)) {
        return std::nullopt;
    }

    // Find the candidate block
    int block_idx = findBlockIndex(key);
    if (block_idx < 0) {
        return std::nullopt;
    }

    // Read and search the block
    BlockReader reader = readBlock(index_entries_[block_idx]);
    int entry_idx = reader.findKey(key);
    if (entry_idx < 0) {
        return std::nullopt;
    }

    auto entry = reader.entryAt(entry_idx);
    return std::vector<uint8_t>(entry.value.begin(), entry.value.end());
}

void SSTableReader::scan(uint64_t min_key, uint64_t max_key,
                         const ScanCallback& callback) const {
    if (index_entries_.empty() || min_key > max_key) {
        return;
    }

    // Find the first block that might contain min_key
    int start_block = findBlockIndex(min_key);
    if (start_block < 0) {
        start_block = 0;
    }

    for (size_t block_idx = start_block; block_idx < index_entries_.size(); ++block_idx) {
        // If this block's first key is beyond max_key, we're done
        if (index_entries_[block_idx].first_key > max_key) {
            break;
        }

        BlockReader reader = readBlock(index_entries_[block_idx]);

        for (size_t entry_idx = 0; entry_idx < reader.entryCount(); ++entry_idx) {
            auto entry = reader.entryAt(entry_idx);
            if (entry.key > max_key) {
                return;
            }
            if (entry.key >= min_key) {
                callback(entry.key, entry.value);
            }
        }
    }
}

void SSTableReader::forEach(const ScanCallback& callback) const {
    for (const auto& index_entry : index_entries_) {
        BlockReader reader = readBlock(index_entry);
        for (size_t entry_idx = 0; entry_idx < reader.entryCount(); ++entry_idx) {
            auto entry = reader.entryAt(entry_idx);
            callback(entry.key, entry.value);
        }
    }
}

uint64_t SSTableReader::minKey() const {
    if (index_entries_.empty()) {
        throw std::runtime_error("Cannot get minKey of empty SSTable");
    }
    return index_entries_.front().first_key;
}

uint64_t SSTableReader::maxKey() const {
    if (index_entries_.empty()) {
        throw std::runtime_error("Cannot get maxKey of empty SSTable");
    }
    BlockReader reader = readBlock(index_entries_.back());
    return reader.lastKey();
}

SSTableCursor SSTableReader::cursor() const {
    if (index_entries_.empty()) {
        return SSTableCursor{};
    }
    return SSTableCursor{this, 0, 0};
}

// ─── SSTableCursor ────────────────────────────────────────────────────────────

SSTableCursor::~SSTableCursor() = default;
SSTableCursor::SSTableCursor(SSTableCursor&&) noexcept = default;
SSTableCursor& SSTableCursor::operator=(SSTableCursor&&) noexcept = default;

SSTableCursor::SSTableCursor(const SSTableReader* reader, size_t block_idx, size_t entry_idx)
    : reader_(reader), block_idx_(block_idx), entry_idx_(entry_idx) {
    loadCurrentEntry();
}

void SSTableCursor::loadCurrentEntry() {
    if (!reader_ || block_idx_ >= reader_->index_entries_.size()) {
        valid_ = false;
        return;
    }

    current_block_ = std::make_unique<BlockReader>(
        reader_->readBlock(reader_->index_entries_[block_idx_]));

    if (entry_idx_ >= current_block_->entryCount()) {
        valid_ = false;
        return;
    }

    auto entry = current_block_->entryAt(entry_idx_);
    current_.key = entry.key;
    current_.value.assign(entry.value.begin(), entry.value.end());
    valid_ = true;
}

void SSTableCursor::next() {
    if (!valid_) return;

    entry_idx_++;
    if (current_block_ && entry_idx_ < current_block_->entryCount()) {
        auto entry = current_block_->entryAt(entry_idx_);
        current_.key = entry.key;
        current_.value.assign(entry.value.begin(), entry.value.end());
        return;
    }

    // Move to next block
    block_idx_++;
    entry_idx_ = 0;
    current_block_.reset();

    if (block_idx_ >= reader_->index_entries_.size()) {
        valid_ = false;
        return;
    }

    loadCurrentEntry();
}

}  // namespace amind
