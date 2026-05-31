#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace amind {

/// Unique identifier for a cached page.
/// Combines a file_id and a page_number to form a unique key.
struct PageId {
    uint64_t file_id{0};
    uint64_t page_number{0};

    bool operator==(const PageId& other) const {
        return file_id == other.file_id && page_number == other.page_number;
    }
};

/// Hash function for PageId, used by unordered_map.
struct PageIdHash {
    size_t operator()(const PageId& page_id) const {
        // Combine file_id and page_number using a simple hash mix
        size_t hash = std::hash<uint64_t>{}(page_id.file_id);
        hash ^= std::hash<uint64_t>{}(page_id.page_number) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};

/// A single page in the cache.
/// Contains the raw data and a "referenced" bit for the Clock algorithm.
struct CachedPage {
    PageId page_id;
    std::vector<uint8_t> data;
    bool referenced{true};   // Clock algorithm: set to true on access
    bool dirty{false};       // Whether the page has been modified
};

/// Page Cache with Clock (second-chance) eviction algorithm.
///
/// The Clock algorithm is an approximation of LRU that is more efficient:
/// - Each page has a "referenced" bit, set to true on every access
/// - When eviction is needed, the clock hand sweeps through pages:
///   - If referenced=true → set to false, move on (give a second chance)
///   - If referenced=false → evict this page
///
/// This avoids the overhead of maintaining a doubly-linked list (LRU)
/// while providing similar hit rates.
///
/// Thread-safe (mutex-protected).
class PageCache {
public:
    /// Callback type for loading a page from disk.
    /// Parameters: file_id, page_number, page_size
    /// Returns: the page data, or empty vector on failure.
    using PageLoader = std::function<std::vector<uint8_t>(uint64_t file_id, uint64_t page_number, size_t page_size)>;

    /// Callback type for flushing a dirty page to disk.
    /// Parameters: file_id, page_number, data
    using PageFlusher = std::function<void(uint64_t file_id, uint64_t page_number, std::span<const uint8_t> data)>;

    /// Construct a page cache.
    /// @param capacity_pages Maximum number of pages to cache
    /// @param page_size Size of each page in bytes (default 4096)
    /// @param loader Callback to load pages from disk
    /// @param flusher Callback to flush dirty pages to disk (optional)
    PageCache(size_t capacity_pages, size_t page_size,
              PageLoader loader, PageFlusher flusher = nullptr);

    /// Fetch a page. Returns a const reference to the cached page data.
    /// If the page is not in cache, it is loaded via the PageLoader callback.
    /// Throws std::runtime_error if the page cannot be loaded.
    [[nodiscard]] const std::vector<uint8_t>& fetchPage(const PageId& page_id);

    /// Write data to a cached page, marking it as dirty.
    /// If the page is not in cache, it is loaded first.
    void writePage(const PageId& page_id, std::span<const uint8_t> data);

    /// Flush all dirty pages to disk via the PageFlusher callback.
    void flushAll();

    /// Flush a specific dirty page to disk.
    /// Returns true if the page was dirty and flushed, false otherwise.
    bool flushPage(const PageId& page_id);

    /// Evict all pages for a specific file (e.g., when the file is deleted).
    void evictFile(uint64_t file_id);

    /// Get the number of pages currently in cache.
    [[nodiscard]] size_t size() const;

    /// Get the cache capacity (max pages).
    [[nodiscard]] size_t capacity() const;

    /// Get the configured page size.
    [[nodiscard]] size_t pageSize() const;

    /// Get cache hit statistics.
    [[nodiscard]] uint64_t hitCount() const;
    [[nodiscard]] uint64_t missCount() const;
    [[nodiscard]] double hitRate() const;

private:
    /// Find a free slot or evict a page using the Clock algorithm.
    /// Returns the index of the available slot.
    size_t findVictim();

    size_t capacity_pages_;
    size_t page_size_;
    PageLoader loader_;
    PageFlusher flusher_;

    // Clock data structures
    std::vector<std::optional<CachedPage>> slots_;  // Fixed-size array of page slots
    std::unordered_map<PageId, size_t, PageIdHash> page_to_slot_;  // PageId → slot index
    size_t clock_hand_{0};  // Current position of the clock hand

    // Statistics
    uint64_t hit_count_{0};
    uint64_t miss_count_{0};

    mutable std::mutex mutex_;
};

}  // namespace amind
