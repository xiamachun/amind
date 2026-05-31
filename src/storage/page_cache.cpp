#include "page_cache.h"

#include <cstring>
#include <span>
#include <stdexcept>

namespace amind {

PageCache::PageCache(size_t capacity_pages, size_t page_size,
                     PageLoader loader, PageFlusher flusher)
    : capacity_pages_(capacity_pages),
      page_size_(page_size),
      loader_(std::move(loader)),
      flusher_(std::move(flusher)),
      slots_(capacity_pages) {
}

const std::vector<uint8_t>& PageCache::fetchPage(const PageId& page_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if page is already in cache
    auto found = page_to_slot_.find(page_id);
    if (found != page_to_slot_.end()) {
        // Cache hit — mark as referenced
        auto& page = slots_[found->second];
        page->referenced = true;
        ++hit_count_;
        return page->data;
    }

    // Cache miss — load from disk
    ++miss_count_;

    std::vector<uint8_t> data = loader_(page_id.file_id, page_id.page_number, page_size_);
    if (data.empty()) {
        throw std::runtime_error(
            "PageCache: failed to load page (file=" +
            std::to_string(page_id.file_id) +
            ", page=" + std::to_string(page_id.page_number) + ")");
    }

    // Find a slot (evict if necessary)
    size_t slot_index = findVictim();

    // Place the new page in the slot
    CachedPage cached_page;
    cached_page.page_id = page_id;
    cached_page.data = std::move(data);
    cached_page.referenced = true;
    cached_page.dirty = false;

    slots_[slot_index] = std::move(cached_page);
    page_to_slot_[page_id] = slot_index;

    return slots_[slot_index]->data;
}

void PageCache::writePage(const PageId& page_id, std::span<const uint8_t> data) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto found = page_to_slot_.find(page_id);
    size_t slot_index;

    if (found != page_to_slot_.end()) {
        // Page is in cache — update it
        slot_index = found->second;
        ++hit_count_;
    } else {
        // Page not in cache — allocate a slot
        ++miss_count_;
        slot_index = findVictim();

        CachedPage cached_page;
        cached_page.page_id = page_id;
        cached_page.data.resize(page_size_, 0);
        slots_[slot_index] = std::move(cached_page);
        page_to_slot_[page_id] = slot_index;
    }

    auto& page = slots_[slot_index].value();
    // Copy data into the page (truncate or pad to page_size)
    size_t copy_size = std::min(data.size(), page_size_);
    std::memcpy(page.data.data(), data.data(), copy_size);
    page.referenced = true;
    page.dirty = true;
}

void PageCache::flushAll() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!flusher_) {
        return;
    }

    for (auto& slot : slots_) {
        if (slot.has_value() && slot->dirty) {
            flusher_(slot->page_id.file_id, slot->page_id.page_number,
                     std::span<const uint8_t>(slot->data));
            slot->dirty = false;
        }
    }
}

bool PageCache::flushPage(const PageId& page_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!flusher_) {
        return false;
    }

    auto found = page_to_slot_.find(page_id);
    if (found == page_to_slot_.end()) {
        return false;
    }

    auto& page = slots_[found->second];
    if (!page.has_value() || !page->dirty) {
        return false;
    }

    flusher_(page->page_id.file_id, page->page_id.page_number,
             std::span<const uint8_t>(page->data));
    page->dirty = false;
    return true;
}

void PageCache::evictFile(uint64_t file_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Collect all page IDs for this file
    std::vector<PageId> to_evict;
    for (const auto& [pid, slot_idx] : page_to_slot_) {
        if (pid.file_id == file_id) {
            to_evict.push_back(pid);
        }
    }

    for (const auto& pid : to_evict) {
        auto found = page_to_slot_.find(pid);
        if (found != page_to_slot_.end()) {
            auto& slot = slots_[found->second];
            // Flush dirty page before evicting
            if (slot.has_value() && slot->dirty && flusher_) {
                flusher_(slot->page_id.file_id, slot->page_id.page_number,
                         std::span<const uint8_t>(slot->data));
            }
            slot.reset();
            page_to_slot_.erase(found);
        }
    }
}

size_t PageCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return page_to_slot_.size();
}

size_t PageCache::capacity() const {
    return capacity_pages_;
}

size_t PageCache::pageSize() const {
    return page_size_;
}

uint64_t PageCache::hitCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return hit_count_;
}

uint64_t PageCache::missCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return miss_count_;
}

double PageCache::hitRate() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t total = hit_count_ + miss_count_;
    if (total == 0) {
        return 0.0;
    }
    return static_cast<double>(hit_count_) / static_cast<double>(total);
}

size_t PageCache::findVictim() {
    // If there are empty slots, use the first one
    for (size_t i = 0; i < capacity_pages_; ++i) {
        if (!slots_[i].has_value()) {
            return i;
        }
    }

    // All slots are full — run the Clock algorithm
    // Sweep until we find a page with referenced=false
    size_t sweep_count = 0;
    size_t max_sweeps = capacity_pages_ * 2;  // Prevent infinite loop

    while (sweep_count < max_sweeps) {
        auto& slot = slots_[clock_hand_];
        if (slot.has_value()) {
            if (!slot->referenced) {
                // Found victim — evict this page
                // Flush if dirty
                if (slot->dirty && flusher_) {
                    flusher_(slot->page_id.file_id, slot->page_id.page_number,
                             std::span<const uint8_t>(slot->data));
                }
                page_to_slot_.erase(slot->page_id);
                size_t victim_index = clock_hand_;
                slot.reset();
                clock_hand_ = (clock_hand_ + 1) % capacity_pages_;
                return victim_index;
            }
            // Give second chance — clear referenced bit
            slot->referenced = false;
        }
        clock_hand_ = (clock_hand_ + 1) % capacity_pages_;
        ++sweep_count;
    }

    // Fallback: evict the current clock hand position (should not happen normally)
    auto& fallback_slot = slots_[clock_hand_];
    if (fallback_slot.has_value()) {
        if (fallback_slot->dirty && flusher_) {
            flusher_(fallback_slot->page_id.file_id, fallback_slot->page_id.page_number,
                     std::span<const uint8_t>(fallback_slot->data));
        }
        page_to_slot_.erase(fallback_slot->page_id);
        fallback_slot.reset();
    }
    size_t victim_index = clock_hand_;
    clock_hand_ = (clock_hand_ + 1) % capacity_pages_;
    return victim_index;
}

}  // namespace amind
