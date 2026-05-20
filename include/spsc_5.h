#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <type_traits>

#include <immintrin.h>

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

template <typename T, size_t SIZE>
class LockFreeQueueStage5 {
    static_assert((SIZE & (SIZE - 1)) == 0, "size must be power of 2");
    static_assert(std::is_trivially_copyable_v<T>,
                  "LockFreeQueueStage5 requires trivially copyable T");
    static_assert(std::is_trivially_default_constructible_v<T>,
                  "LockFreeQueueStage5 direct-slot path requires trivially default-constructible T");

    static constexpr size_t CAPACITY = SIZE;
    static constexpr size_t MASK = CAPACITY - 1;
    static constexpr size_t PADDING = (CACHE_LINE_SIZE - 1) / sizeof(T) + 1;
    static constexpr size_t NSLOTS = CAPACITY + 2 * PADDING;

    struct alignas(CACHE_LINE_SIZE) ProducerState {
        size_t write_index_{0};
        size_t read_index_cache_{0};
        const size_t padding_cache_{PADDING};
    } producer_;

    struct alignas(CACHE_LINE_SIZE) PublishedWriteIndex {
        std::atomic<size_t> write_index_{0};
    } published_write_;

    struct alignas(CACHE_LINE_SIZE) ConsumerState {
        size_t read_index_{0};
        size_t write_index_cache_{0};
        const size_t capacity_cache_{CAPACITY};
    } consumer_;

    struct alignas(CACHE_LINE_SIZE) PublishedReadIndex {
        std::atomic<size_t> read_index_{0};
    } published_read_;

    alignas(CACHE_LINE_SIZE) std::array<T, NSLOTS> buffer_;

    [[nodiscard]] static constexpr size_t slot_index(size_t index) noexcept {
        return (index & MASK) + PADDING;
    }

    [[nodiscard]] T* slot_ptr(size_t index) noexcept {
        return std::addressof(buffer_[slot_index(index)]);
    }

    static inline void spin_pause() noexcept {
        _mm_pause();
        _mm_pause();
    }

public:
    explicit LockFreeQueueStage5() = default;

    LockFreeQueueStage5(const LockFreeQueueStage5&) = delete;
    LockFreeQueueStage5& operator=(const LockFreeQueueStage5&) = delete;

    void emplace(const T& value) noexcept {
        const size_t write_index = producer_.write_index_;

        while (write_index == producer_.read_index_cache_ + MASK) [[unlikely]] {
            producer_.read_index_cache_ = published_read_.read_index_.load(std::memory_order_acquire);
            if (write_index == producer_.read_index_cache_ + MASK) [[unlikely]] {
                spin_pause();
            }
        }

        buffer_[slot_index(write_index)] = value;
        const size_t next_write_index = write_index + 1;
        producer_.write_index_ = next_write_index;
        published_write_.write_index_.store(next_write_index, std::memory_order_release);
    }

    void pop(T& out) noexcept {
        const size_t read_index = consumer_.read_index_;

        while (read_index == consumer_.write_index_cache_) [[unlikely]] {
            consumer_.write_index_cache_ = published_write_.write_index_.load(std::memory_order_acquire);
            if (read_index == consumer_.write_index_cache_) [[unlikely]] {
                spin_pause();
            }
        }

        out = buffer_[slot_index(read_index)];
        const size_t next_read_index = read_index + 1;
        consumer_.read_index_ = next_read_index;
        published_read_.read_index_.store(next_read_index, std::memory_order_release);
    }

    [[nodiscard]] bool try_emplace(const T& value) noexcept {
        const size_t write_index = producer_.write_index_;
        size_t read_index_cache = producer_.read_index_cache_;

        if (write_index == read_index_cache + MASK) [[unlikely]] {
            read_index_cache = published_read_.read_index_.load(std::memory_order_acquire);
            producer_.read_index_cache_ = read_index_cache;
            if (write_index == read_index_cache + MASK) [[unlikely]] {
                return false;
            }
        }

        buffer_[slot_index(write_index)] = value;
        const size_t next_write_index = write_index + 1;
        producer_.write_index_ = next_write_index;
        published_write_.write_index_.store(next_write_index, std::memory_order_release);
        return true;
    }

    [[nodiscard]] bool try_pop(T& out) noexcept {
        const size_t read_index = consumer_.read_index_;
        size_t write_index_cache = consumer_.write_index_cache_;

        if (read_index == write_index_cache) [[unlikely]] {
            write_index_cache = published_write_.write_index_.load(std::memory_order_acquire);
            consumer_.write_index_cache_ = write_index_cache;
            if (read_index == write_index_cache) [[unlikely]] {
                return false;
            }
        }

        out = buffer_[slot_index(read_index)];
        const size_t next_read_index = read_index + 1;
        consumer_.read_index_ = next_read_index;
        published_read_.read_index_.store(next_read_index, std::memory_order_release);
        return true;
    }

    bool enqueue(const T& item) {
        return try_emplace(item);
    }

    T* front() {
        const size_t read_index = consumer_.read_index_;
        size_t write_index_cache = consumer_.write_index_cache_;
        if (read_index == write_index_cache) [[unlikely]] {
            write_index_cache = published_write_.write_index_.load(std::memory_order_acquire);
            consumer_.write_index_cache_ = write_index_cache;
            if (read_index == write_index_cache) {
                return nullptr;
            }
        }

        return slot_ptr(read_index);
    }

    void pop() {
        const size_t next_read_index = consumer_.read_index_ + 1;
        consumer_.read_index_ = next_read_index;
        published_read_.read_index_.store(next_read_index, std::memory_order_release);
    }

    std::optional<T> dequeue() {
        const size_t read_index = consumer_.read_index_;
        size_t write_index_cache = consumer_.write_index_cache_;
        if (read_index == write_index_cache) [[unlikely]] {
            write_index_cache = published_write_.write_index_.load(std::memory_order_acquire);
            consumer_.write_index_cache_ = write_index_cache;
            if (read_index == write_index_cache) {
                return std::nullopt;
            }
        }

        std::optional<T> result{};
        result.emplace(buffer_[slot_index(read_index)]);
        const size_t next_read_index = read_index + 1;
        consumer_.read_index_ = next_read_index;
        published_read_.read_index_.store(next_read_index, std::memory_order_release);
        return result;
    }

    bool try_dequeue(T* out) {
        const size_t read_index = consumer_.read_index_;
        size_t write_index_cache = consumer_.write_index_cache_;
        if (read_index == write_index_cache) [[unlikely]] {
            write_index_cache = published_write_.write_index_.load(std::memory_order_acquire);
            consumer_.write_index_cache_ = write_index_cache;
            if (read_index == write_index_cache) {
                return false;
            }
        }

        if (out) {
            *out = buffer_[slot_index(read_index)];
        }
        const size_t next_read_index = read_index + 1;
        consumer_.read_index_ = next_read_index;
        published_read_.read_index_.store(next_read_index, std::memory_order_relaxed);
        return true;
    }

    bool empty() const {
        return published_write_.write_index_.load(std::memory_order_acquire) ==
               published_read_.read_index_.load(std::memory_order_acquire);
    }

    size_t size() const {
        const size_t read_index = published_read_.read_index_.load(std::memory_order_acquire);
        const size_t write_index = published_write_.write_index_.load(std::memory_order_acquire);
        return write_index - read_index;
    }

    size_t capacity() const {
        return CAPACITY - 1;
    }
};
