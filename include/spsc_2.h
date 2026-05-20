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
class LockFreeQueueStage2 {
    static_assert((SIZE & (SIZE - 1)) == 0, "size must be power of 2");
    static_assert(std::is_trivially_copyable_v<T>,
                  "LockFreeQueueStage2 requires trivially copyable T");
    static_assert(std::is_trivially_default_constructible_v<T>,
                  "LockFreeQueueStage2 direct-slot path requires trivially default-constructible T");

    static constexpr size_t CAPACITY = SIZE;
    static constexpr size_t MASK = CAPACITY - 1;
    static constexpr size_t PADDING = (CACHE_LINE_SIZE - 1) / sizeof(T) + 1;

    struct alignas(CACHE_LINE_SIZE) WriterCacheLine {
        std::atomic<size_t> write_index_{0};
        size_t read_index_cache_{0};
        const size_t padding_cache_{PADDING};
    } writer_;

    struct alignas(CACHE_LINE_SIZE) ReaderCacheLine {
        std::atomic<size_t> read_index_{0};
        size_t write_index_cache_{0};
        size_t capacity_cache_{CAPACITY};
    } reader_;

    static constexpr size_t NSLOTS = CAPACITY + 2 * PADDING;

    alignas(CACHE_LINE_SIZE) std::array<T, NSLOTS> buffer_;

    [[nodiscard]] T* slot_ptr(size_t index) noexcept {
        return std::addressof(buffer_[index + PADDING]);
    }

public:
    explicit LockFreeQueueStage2() = default;

    LockFreeQueueStage2(const LockFreeQueueStage2&) = delete;
    LockFreeQueueStage2& operator=(const LockFreeQueueStage2&) = delete;

    void emplace(const T& value) noexcept {
        const size_t write_index = writer_.write_index_.load(std::memory_order_relaxed);
        const size_t next_write_index = (write_index + 1) & MASK;

        while (next_write_index == writer_.read_index_cache_) [[unlikely]] {
            writer_.read_index_cache_ = reader_.read_index_.load(std::memory_order_acquire);
            _mm_pause();
        }

        buffer_[write_index + PADDING] = value;
        writer_.write_index_.store(next_write_index, std::memory_order_release);
    }

    void pop(T& out) noexcept {
        const size_t read_index = reader_.read_index_.load(std::memory_order_relaxed);

        while (read_index == reader_.write_index_cache_) [[unlikely]] {
            reader_.write_index_cache_ = writer_.write_index_.load(std::memory_order_acquire);
            _mm_pause();
        }

        out = buffer_[read_index + PADDING];
        reader_.read_index_.store((read_index + 1) & MASK, std::memory_order_release);
    }

    [[nodiscard]] bool try_emplace(const T& value) noexcept {
        const size_t write_index = writer_.write_index_.load(std::memory_order_relaxed);
        const size_t next_write_index = (write_index + 1) & MASK;

        if (next_write_index == writer_.read_index_cache_) [[unlikely]] {
            writer_.read_index_cache_ = reader_.read_index_.load(std::memory_order_acquire);
            if (next_write_index == writer_.read_index_cache_) {
                return false;
            }
        }

        buffer_[write_index + PADDING] = value;
        writer_.write_index_.store(next_write_index, std::memory_order_release);
        return true;
    }

    [[nodiscard]] bool try_pop(T& out) noexcept {
        const size_t read_index = reader_.read_index_.load(std::memory_order_relaxed);

        if (read_index == reader_.write_index_cache_) [[unlikely]] {
            reader_.write_index_cache_ = writer_.write_index_.load(std::memory_order_acquire);
            if (read_index == reader_.write_index_cache_) {
                return false;
            }
        }

        out = buffer_[read_index + PADDING];
        reader_.read_index_.store((read_index + 1) & MASK, std::memory_order_release);
        return true;
    }

    bool enqueue(const T& item) {
        return try_emplace(item);
    }

    T* front() {
        const size_t read_index = reader_.read_index_.load(std::memory_order_relaxed);
        if (read_index == reader_.write_index_cache_) [[unlikely]] {
            reader_.write_index_cache_ = writer_.write_index_.load(std::memory_order_acquire);
            if (read_index == reader_.write_index_cache_) {
                return nullptr;
            }
        }

        return slot_ptr(read_index);
    }

    void pop() {
        const size_t read_index = reader_.read_index_.load(std::memory_order_relaxed);
        reader_.read_index_.store((read_index + 1) & MASK, std::memory_order_release);
    }

    std::optional<T> dequeue() {
        const size_t read_index = reader_.read_index_.load(std::memory_order_relaxed);
        if (read_index == reader_.write_index_cache_) [[unlikely]] {
            reader_.write_index_cache_ = writer_.write_index_.load(std::memory_order_acquire);
            if (read_index == reader_.write_index_cache_) {
                return std::nullopt;
            }
        }

        std::optional<T> result{};
        result.emplace(buffer_[read_index + PADDING]);
        reader_.read_index_.store((read_index + 1) & MASK, std::memory_order_release);
        return result;
    }

    bool try_dequeue(T* out) {
        const size_t read_index = reader_.read_index_.load(std::memory_order_relaxed);
        if (read_index == reader_.write_index_cache_) [[unlikely]] {
            reader_.write_index_cache_ = writer_.write_index_.load(std::memory_order_acquire);
            if (read_index == reader_.write_index_cache_) {
                return false;
            }
        }

        if (out) {
            *out = buffer_[read_index + PADDING];
        }
        reader_.read_index_.store((read_index + 1) & MASK, std::memory_order_relaxed);
        return true;
    }

    bool empty() const {
        return writer_.write_index_.load(std::memory_order_acquire) ==
               reader_.read_index_.load(std::memory_order_acquire);
    }

    size_t size() const {
        const size_t read_index = reader_.read_index_.load(std::memory_order_acquire);
        const size_t write_index = writer_.write_index_.load(std::memory_order_acquire);
        return (write_index - read_index) & MASK;
    }

    size_t capacity() const {
        return CAPACITY - 1;
    }
};
