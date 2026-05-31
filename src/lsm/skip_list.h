#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <vector>

namespace amind {

/// A concurrent skip list implementation for the MemTable.
///
/// Template parameters:
///   K — key type (must be comparable with <)
///   V — value type
///
/// The skip list provides O(log n) average-case insert, lookup, and delete.
/// It uses a probabilistic balancing scheme (coin flips) instead of
/// deterministic rebalancing like AVL/Red-Black trees.
///
/// Thread safety: all public methods are protected by a mutex.
template <typename K, typename V>
class SkipList {
public:
    static constexpr int MAX_LEVEL = 16;
    static constexpr double PROBABILITY = 0.5;

    struct Node {
        K key;
        V value;
        bool is_tombstone{false};
        int level;
        std::vector<Node*> forward;

        Node(K k, V v, int lvl)
            : key(std::move(k)),
              value(std::move(v)),
              level(lvl),
              forward(lvl + 1, nullptr) {}

        /// Sentinel node constructor (head node with no meaningful key/value).
        explicit Node(int lvl)
            : key{},
              value{},
              level(lvl),
              forward(lvl + 1, nullptr) {}
    };

    /// Iterator for in-order traversal of the skip list.
    class Iterator {
    public:
        Iterator() : current_(nullptr) {}
        explicit Iterator(Node* node) : current_(node) {}

        bool valid() const { return current_ != nullptr; }

        const K& key() const {
            assert(current_ != nullptr);
            return current_->key;
        }

        const V& value() const {
            assert(current_ != nullptr);
            return current_->value;
        }

        bool isTombstone() const {
            assert(current_ != nullptr);
            return current_->is_tombstone;
        }

        Node* node() const { return current_; }

        void next() {
            assert(current_ != nullptr);
            current_ = current_->forward[0];
        }

        bool operator==(const Iterator& other) const {
            return current_ == other.current_;
        }

        bool operator!=(const Iterator& other) const {
            return current_ != other.current_;
        }

    private:
        Node* current_;
    };

    SkipList()
        : head_(std::make_unique<Node>(MAX_LEVEL)),
          current_level_(0),
          size_(0),
          approximate_memory_usage_(0),
          rng_(std::random_device{}()) {}

    ~SkipList() {
        Node* current = head_->forward[0];
        while (current != nullptr) {
            Node* next = current->forward[0];
            delete current;
            current = next;
        }
    }

    SkipList(const SkipList&) = delete;
    SkipList& operator=(const SkipList&) = delete;
    SkipList(SkipList&&) = delete;
    SkipList& operator=(SkipList&&) = delete;

    /// Insert or update a key-value pair.
    /// If the key already exists, its value is replaced.
    void put(const K& key, V value) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::array<Node*, MAX_LEVEL + 1> update{};
        Node* current = head_.get();

        for (int i = current_level_; i >= 0; --i) {
            while (current->forward[i] != nullptr &&
                   current->forward[i]->key < key) {
                current = current->forward[i];
            }
            update[i] = current;
        }

        current = current->forward[0];

        if (current != nullptr && current->key == key) {
            size_t old_size = estimateNodeSize(current);
            current->value = std::move(value);
            current->is_tombstone = false;
            size_t new_size = estimateNodeSize(current);
            approximate_memory_usage_ += (new_size - old_size);
            return;
        }

        int new_level = randomLevel();
        if (new_level > current_level_) {
            for (int i = current_level_ + 1; i <= new_level; ++i) {
                update[i] = head_.get();
            }
            current_level_ = new_level;
        }

        auto* new_node = new Node(key, std::move(value), new_level);

        for (int i = 0; i <= new_level; ++i) {
            new_node->forward[i] = update[i]->forward[i];
            update[i]->forward[i] = new_node;
        }

        ++size_;
        approximate_memory_usage_ += estimateNodeSize(new_node);
    }

    /// Look up a key. Returns the value if found and not tombstoned.
    [[nodiscard]] std::optional<V> get(const K& key) const {
        std::lock_guard<std::mutex> lock(mutex_);

        const Node* current = head_.get();
        for (int i = current_level_; i >= 0; --i) {
            while (current->forward[i] != nullptr &&
                   current->forward[i]->key < key) {
                current = current->forward[i];
            }
        }

        current = current->forward[0];

        if (current != nullptr && current->key == key && !current->is_tombstone) {
            return current->value;
        }
        return std::nullopt;
    }

    /// Mark a key as deleted (tombstone). Returns true if the key was found.
    bool remove(const K& key) {
        std::lock_guard<std::mutex> lock(mutex_);

        Node* current = head_.get();
        for (int i = current_level_; i >= 0; --i) {
            while (current->forward[i] != nullptr &&
                   current->forward[i]->key < key) {
                current = current->forward[i];
            }
        }

        current = current->forward[0];

        if (current != nullptr && current->key == key) {
            current->is_tombstone = true;
            return true;
        }
        return false;
    }

    /// Check if a key exists (including tombstones).
    [[nodiscard]] bool contains(const K& key) const {
        std::lock_guard<std::mutex> lock(mutex_);

        const Node* current = head_.get();
        for (int i = current_level_; i >= 0; --i) {
            while (current->forward[i] != nullptr &&
                   current->forward[i]->key < key) {
                current = current->forward[i];
            }
        }

        current = current->forward[0];
        return current != nullptr && current->key == key && !current->is_tombstone;
    }

    /// Return an iterator pointing to the first element.
    [[nodiscard]] Iterator begin() const {
        return Iterator(head_->forward[0]);
    }

    /// Return an iterator pointing to the first element >= key.
    [[nodiscard]] Iterator lowerBound(const K& key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const Node* current = head_.get();
        for (int i = current_level_; i >= 0; --i) {
            while (current->forward[i] != nullptr &&
                   current->forward[i]->key < key) {
                current = current->forward[i];
            }
        }
        return Iterator(current->forward[0]);
    }

    /// Return a sentinel end iterator.
    [[nodiscard]] Iterator end() const {
        return Iterator(nullptr);
    }

    /// Number of entries (including tombstones).
    [[nodiscard]] size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_;
    }

    [[nodiscard]] bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_ == 0;
    }

    /// Approximate memory usage in bytes.
    [[nodiscard]] size_t approximateMemoryUsage() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return approximate_memory_usage_;
    }

    /// Collect all keys (for bloom filter construction during flush).
    [[nodiscard]] std::vector<K> allKeys() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<K> keys;
        keys.reserve(size_);
        Node* current = head_->forward[0];
        while (current != nullptr) {
            keys.push_back(current->key);
            current = current->forward[0];
        }
        return keys;
    }

private:
    int randomLevel() {
        int level = 0;
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        while (dist(rng_) < PROBABILITY && level < MAX_LEVEL) {
            ++level;
        }
        return level;
    }

    static size_t estimateNodeSize(const Node* node) {
        size_t size = sizeof(Node);
        size += node->forward.capacity() * sizeof(Node*);
        if constexpr (std::is_same_v<V, std::vector<uint8_t>>) {
            size += node->value.capacity();
        }
        return size;
    }

    std::unique_ptr<Node> head_;
    int current_level_;
    size_t size_;
    size_t approximate_memory_usage_;
    mutable std::mutex mutex_;
    std::mt19937 rng_;
};

}  // namespace amind
