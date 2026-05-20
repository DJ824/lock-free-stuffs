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
class LockFreeQueue {
    static_assert((SIZE & (SIZE - 1)) == 0, "size must be power of 2");
    static_assert(std::is_trivially_copyable_v<T>,
                  "req trivially copyable T");
    static_assert(std::is_trivially_default_constructible_v<T>,
                  "req trivially default-constructible T");

    static constexpr size_t CAPACITY = SIZE;
    static constexpr size_t MASK = CAPACITY - 1;
    static constexpr size_t PADDING = (CACHE_LINE_SIZE - 1) / sizeof(T) + 1;
    static constexpr size_t NSLOTS = CAPACITY + 2 * PADDING;

    alignas(CACHE_LINE_SIZE) std::atomic<size_t> write_index_{0};
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> read_index_{0};
    alignas(CACHE_LINE_SIZE) std::array<T, NSLOTS> buffer_;

    [[nodiscard]] T* slot_ptr(size_t index) noexcept {
        return std::addressof(buffer_[index + PADDING]);
    }

public:
    explicit LockFreeQueue() = default;

    LockFreeQueue(const LockFreeQueue&) = delete;
    LockFreeQueue& operator=(const LockFreeQueue&) = delete;

    void emplace(const T& value) noexcept {
        const size_t write_index = write_index_.load(std::memory_order_relaxed);
        const size_t next_write_index = (write_index + 1) & MASK;

        while (next_write_index == read_index_.load(std::memory_order_acquire)) [[unlikely]] {
        }

        buffer_[write_index + PADDING] = value;
        write_index_.store(next_write_index, std::memory_order_release);
    }

    void pop(T& out) noexcept {
        const size_t read_index = read_index_.load(std::memory_order_relaxed);

        while (read_index == write_index_.load(std::memory_order_acquire)) [[unlikely]] {
        }

        out = buffer_[read_index + PADDING];
        read_index_.store((read_index + 1) & MASK, std::memory_order_release);
    }

    [[nodiscard]] bool try_emplace(const T& value) {
        const size_t write_index = write_index_.load(std::memory_order_relaxed);
        const size_t next_write_index = (write_index + 1) & MASK;

        if (next_write_index == read_index_.load(std::memory_order_acquire)) [[unlikely]] {
            return false;
        }

        buffer_[write_index + PADDING] = value;
        write_index_.store(next_write_index, std::memory_order_release);
        return true;
    }

    [[nodiscard]] bool try_pop(T& out) {
        const size_t read_index = read_index_.load(std::memory_order_relaxed);

        if (read_index == write_index_.load(std::memory_order_acquire)) [[unlikely]] {
            return false;
        }

        out = buffer_[read_index + PADDING];
        read_index_.store((read_index + 1) & MASK, std::memory_order_release);
        return true;
    }

    bool enqueue(const T& item) {
        return try_emplace(item);
    }

    T* front() {
        const size_t read_index = read_index_.load(std::memory_order_relaxed);
        if (read_index == write_index_.load(std::memory_order_acquire)) [[unlikely]] {
            return nullptr;
        }

        return slot_ptr(read_index);
    }

    void pop() {
        const size_t read_index = read_index_.load(std::memory_order_relaxed);
        read_index_.store((read_index + 1) & MASK, std::memory_order_release);
    }

    std::optional<T> dequeue() {
        const size_t read_index = read_index_.load(std::memory_order_relaxed);
        if (read_index == write_index_.load(std::memory_order_acquire)) [[unlikely]] {
            return std::nullopt;
        }

        std::optional<T> result{};
        result.emplace(buffer_[read_index + PADDING]);
        read_index_.store((read_index + 1) & MASK, std::memory_order_release);
        return result;
    }

    bool try_dequeue(T* out) {
        const size_t read_index = read_index_.load(std::memory_order_relaxed);
        if (read_index == write_index_.load(std::memory_order_acquire)) [[unlikely]] {
            return false;
        }

        if (out) {
            *out = buffer_[read_index + PADDING];
        }
        read_index_.store((read_index + 1) & MASK, std::memory_order_relaxed);
        return true;
    }

    bool empty() const {
        return write_index_.load(std::memory_order_acquire) ==
               read_index_.load(std::memory_order_acquire);
    }

    size_t size() const {
        const size_t read_index = read_index_.load(std::memory_order_acquire);
        const size_t write_index = write_index_.load(std::memory_order_acquire);
        return (write_index - read_index) & MASK;
    }

    size_t capacity() const {
        return CAPACITY - 1;
    }
};
