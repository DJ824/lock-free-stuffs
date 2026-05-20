#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>

#include <immintrin.h>

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

template <typename T, size_t SIZE>
class LockFreeQueueTest {
    static_assert((SIZE & (SIZE - 1)) == 0, "size must be power of 2");
    static_assert(std::is_move_constructible_v<T>,
                  "requires move-constructible T");
    static_assert(std::is_destructible_v<T>,
                  "requires destructible T");

    static constexpr size_t CAPACITY = SIZE;
    static constexpr size_t MASK = CAPACITY - 1;
    static constexpr size_t PADDING = (CACHE_LINE_SIZE - 1) / sizeof(T) + 1;
    static constexpr size_t NSLOTS = CAPACITY + 2 * PADDING;
    static constexpr size_t STATE_PADDING =
            (CACHE_LINE_SIZE - 1) / sizeof(std::atomic<std::uint8_t>) + 1;
    static constexpr size_t NSTATES = CAPACITY + 2 * STATE_PADDING;
    static constexpr bool USE_DIRECT_SLOTS =
            std::is_trivially_copyable_v<T> &&
            std::is_trivially_default_constructible_v<T> &&
            std::is_trivially_destructible_v<T>;

    enum SlotState : std::uint8_t {
        Empty = 0,
        Stored = 1,
    };

    using Storage = std::conditional_t<
            USE_DIRECT_SLOTS,
            T,
            std::aligned_storage_t<sizeof(T), alignof(T)>>;

    struct alignas(CACHE_LINE_SIZE) ProducerState {
        size_t write_index_{0};
        const size_t padding_cache_{PADDING};
    } producer_;

    struct alignas(CACHE_LINE_SIZE) ConsumerState {
        size_t read_index_{0};
        const size_t capacity_cache_{CAPACITY};
    } consumer_;

    alignas(CACHE_LINE_SIZE) std::array<std::atomic<std::uint8_t>, NSTATES> states_{};
    alignas(CACHE_LINE_SIZE) std::array<Storage, NSLOTS> buffer_;

    [[nodiscard]] static constexpr size_t slot_index(size_t index) noexcept {
        return (index & MASK) + PADDING;
    }

    [[nodiscard]] static constexpr size_t state_index(size_t index) noexcept {
        return (index & MASK) + STATE_PADDING;
    }

    [[nodiscard]] T* slot_ptr(size_t index) noexcept {
        if constexpr (USE_DIRECT_SLOTS) {
            return std::addressof(buffer_[index]);
        } else {
            return std::launder(reinterpret_cast<T*>(std::addressof(buffer_[index])));
        }
    }

    [[nodiscard]] const T* slot_ptr(size_t index) const noexcept {
        if constexpr (USE_DIRECT_SLOTS) {
            return std::addressof(buffer_[index]);
        } else {
            return std::launder(reinterpret_cast<const T*>(std::addressof(buffer_[index])));
        }
    }

    template <typename... Args>
    void construct_slot(size_t index, Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args&&...>) {
        if constexpr (USE_DIRECT_SLOTS) {
            buffer_[index] = T(std::forward<Args>(args)...);
        } else {
            std::construct_at(slot_ptr(index), std::forward<Args>(args)...);
        }
    }

    void destroy_slot(size_t index) noexcept {
        if constexpr (!USE_DIRECT_SLOTS && !std::is_trivially_destructible_v<T>) {
            std::destroy_at(slot_ptr(index));
        }
    }

    static inline void spin_pause() noexcept {
        _mm_pause();
        _mm_pause();
        _mm_pause();
        _mm_pause();
    }

public:
    explicit LockFreeQueueTest() {
        for (auto& state : states_) {
            state.store(Empty, std::memory_order_relaxed);
        }
    }

    ~LockFreeQueueTest() {
        if constexpr (!USE_DIRECT_SLOTS && !std::is_trivially_destructible_v<T>) {
            size_t read_index = consumer_.read_index_;
            const size_t write_index = producer_.write_index_;
            while (read_index != write_index) {
                destroy_slot(slot_index(read_index));
                ++read_index;
            }
        }
    }

    LockFreeQueueTest(const LockFreeQueueTest&) = delete;
    LockFreeQueueTest& operator=(const LockFreeQueueTest&) = delete;

    template <typename... Args>
    void emplace(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args&&...>) {
        static_assert(std::is_constructible_v<T, Args&&...>,
                      "[emplace] requires constructible T");
        const size_t write_index = producer_.write_index_;
        const size_t slot = slot_index(write_index);
        const size_t state = state_index(write_index);

        while (states_[state].load(std::memory_order_acquire) != Empty) [[unlikely]] {
            spin_pause();
        }

        construct_slot(slot, std::forward<Args>(args)...);
        states_[state].store(Stored, std::memory_order_release);
        producer_.write_index_ = write_index + 1;
    }

    void pop(T& out) noexcept {
        static_assert(std::is_move_assignable_v<T>,
                      "[pop] requires move-assignable T");
        const size_t read_index = consumer_.read_index_;
        const size_t slot = slot_index(read_index);
        const size_t state = state_index(read_index);

        while (states_[state].load(std::memory_order_acquire) != Stored) [[unlikely]] {
            spin_pause();
        }

        out = std::move(*slot_ptr(slot));
        destroy_slot(slot);
        states_[state].store(Empty, std::memory_order_release);
        consumer_.read_index_ = read_index + 1;
    }

    template <typename... Args>
    [[nodiscard]] bool try_emplace(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args&&...>) {
        static_assert(std::is_constructible_v<T, Args&&...>,
                      "requires constructible T");
        const size_t write_index = producer_.write_index_;
        const size_t slot = slot_index(write_index);
        const size_t state = state_index(write_index);
        if (states_[state].load(std::memory_order_acquire) != Empty) [[unlikely]] {
            return false;
        }

        construct_slot(slot, std::forward<Args>(args)...);
        states_[state].store(Stored, std::memory_order_release);
        producer_.write_index_ = write_index + 1;
        return true;
    }

    [[nodiscard]] bool try_pop(T& out) noexcept {
        static_assert(std::is_move_assignable_v<T>,
                      "requires move-assignable T");
        const size_t read_index = consumer_.read_index_;
        const size_t slot = slot_index(read_index);
        const size_t state = state_index(read_index);
        if (states_[state].load(std::memory_order_acquire) != Stored) [[unlikely]] {
            return false;
        }

        out = std::move(*slot_ptr(slot));
        destroy_slot(slot);
        states_[state].store(Empty, std::memory_order_release);
        consumer_.read_index_ = read_index + 1;
        return true;
    }

    bool empty() const noexcept {
        return states_[state_index(consumer_.read_index_)].load(std::memory_order_acquire) != Stored;
    }

    size_t size() const noexcept {
        return producer_.write_index_ - consumer_.read_index_;
    }

    size_t capacity() const noexcept {
        return CAPACITY;
    }
};
