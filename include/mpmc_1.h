#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

#include <immintrin.h>

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

template <typename T, size_t SIZE>
class LockFreeQueueMpmc1 {
    using AtomicState = std::atomic<std::uint8_t>;

    static_assert((SIZE & (SIZE - 1)) == 0, "size must be power of 2");
    static_assert(std::is_trivially_copyable_v<T>,
                  "LockFreeQueueMpmc1 requires trivially copyable T");
    static_assert(std::is_trivially_default_constructible_v<T>,
                  "LockFreeQueueMpmc1 direct-slot path requires trivially default-constructible T");

    static constexpr size_t CAPACITY = SIZE;
    static constexpr size_t MASK = CAPACITY - 1;
    static constexpr size_t BUFFER_PADDING = (CACHE_LINE_SIZE - 1) / sizeof(T) + 1;
    static constexpr size_t BUFFER_NSLOTS = CAPACITY + 2 * BUFFER_PADDING;
    static constexpr size_t STATE_PADDING = (CACHE_LINE_SIZE - 1) / sizeof(AtomicState) + 1;
    static constexpr size_t STATE_NSLOTS = CAPACITY + 2 * STATE_PADDING;

    enum SlotState : uint8_t {
        Empty = 0,
        Storing = 1,
        Stored = 2,
        Loading = 3,
    };

    struct alignas(CACHE_LINE_SIZE) ProducerState {
        std::atomic<size_t> head_{0};
    } producer_;

    struct alignas(CACHE_LINE_SIZE) ConsumerState {
        std::atomic<size_t> tail_{0};
    } consumer_;

    alignas(CACHE_LINE_SIZE) std::array<T, BUFFER_NSLOTS> buffer_;
    alignas(CACHE_LINE_SIZE) std::array<AtomicState, STATE_NSLOTS> states_{};

    [[nodiscard]] static constexpr int cache_line_index_bits(size_t elements_per_cache_line) noexcept {
        switch (elements_per_cache_line) {
            case 256: return 8;
            case 128: return 7;
            case 64: return 6;
            case 32: return 5;
            case 16: return 4;
            case 8: return 3;
            case 4: return 2;
            case 2: return 1;
            default: return 0;
        }
    }

    [[nodiscard]] static constexpr int shuffle_bits(size_t elements_per_cache_line) noexcept {
        const int bits = cache_line_index_bits(elements_per_cache_line);
        const size_t minimum_size = bits == 0 ? 1 : (size_t{1} << (bits * 2));
        return CAPACITY < minimum_size ? 0 : bits;
    }

    template <int BITS>
    [[nodiscard]] static constexpr size_t remap_index(size_t index) noexcept {
        if constexpr (BITS == 0) {
            return index;
        } else {
            const size_t mix_mask = (size_t{1} << BITS) - 1;
            const size_t mix = (index ^ (index >> BITS)) & mix_mask;
            return index ^ mix ^ (mix << BITS);
        }
    }

    static constexpr size_t BUFFER_ELEMENTS_PER_CACHE_LINE = CACHE_LINE_SIZE / sizeof(T);
    static constexpr size_t STATE_ELEMENTS_PER_CACHE_LINE = CACHE_LINE_SIZE / sizeof(AtomicState);
    static constexpr int BUFFER_SHUFFLE_BITS = shuffle_bits(BUFFER_ELEMENTS_PER_CACHE_LINE);
    static constexpr int STATE_SHUFFLE_BITS = shuffle_bits(STATE_ELEMENTS_PER_CACHE_LINE);

    [[nodiscard]] static constexpr size_t buffer_index(size_t index) noexcept {
        return remap_index<BUFFER_SHUFFLE_BITS>(index & MASK) + BUFFER_PADDING;
    }

    [[nodiscard]] static constexpr size_t state_index(size_t index) noexcept {
        return remap_index<STATE_SHUFFLE_BITS>(index & MASK) + STATE_PADDING;
    }

    [[nodiscard]] AtomicState& state_ref(size_t index) noexcept {
        return states_[state_index(index)];
    }

    [[nodiscard]] T& slot_ref(size_t index) noexcept {
        return buffer_[buffer_index(index)];
    }

    static void spin_pause() noexcept {
        _mm_pause();
        _mm_pause();
    }

    template <typename U>
    void push_direct(U&& value, const size_t head) noexcept {
        std::atomic<std::uint8_t>& state = state_ref(head);
        T& slot = slot_ref(head);

        for (;;) {
            std::uint8_t expected = Empty;
            if (state.compare_exchange_weak(expected, Storing,
                                            std::memory_order_acquire,
                                            std::memory_order_relaxed)) {
                slot = std::forward<U>(value);
                state.store(Stored, std::memory_order_release);
                return;
            }

            do {
                spin_pause();
            } while (state.load(std::memory_order_relaxed) != Empty);
        }
    }

    [[nodiscard]] T pop_direct(const size_t tail) noexcept {
        std::atomic<std::uint8_t>& state = state_ref(tail);
        T& slot = slot_ref(tail);

        for (;;) {
            std::uint8_t expected = Stored;
            if (state.compare_exchange_weak(expected, Loading,
                                            std::memory_order_acquire,
                                            std::memory_order_relaxed)) {
                T value = slot;
                state.store(Empty, std::memory_order_release);
                return value;
            }

            do {
                spin_pause();
            } while (state.load(std::memory_order_relaxed) != Stored);
        }
    }

public:
    explicit LockFreeQueueMpmc1() noexcept {
        for (size_t i = 0; i < STATE_NSLOTS; ++i) {
            states_[i].store(Empty, std::memory_order_relaxed);
        }
    }

    LockFreeQueueMpmc1(const LockFreeQueueMpmc1&) = delete;
    LockFreeQueueMpmc1& operator=(const LockFreeQueueMpmc1&) = delete;

    void emplace(const T& value) noexcept {
        push(value);
    }

    [[nodiscard]] bool try_emplace(const T& value) noexcept {
        size_t head = producer_.head_.load(std::memory_order_relaxed);
        do {
            const size_t tail = consumer_.tail_.load(std::memory_order_relaxed);
            const ptrdiff_t used = head >= tail
                                       ? static_cast<ptrdiff_t>(head - tail)
                                       : -static_cast<ptrdiff_t>(tail - head);
            if (used >= static_cast<ptrdiff_t>(CAPACITY)) [[unlikely]] {
                return false;
            }
        } while (!producer_.head_.compare_exchange_weak(head, head + 1,
                                                        std::memory_order_relaxed,
                                                        std::memory_order_relaxed));

        push_direct(value, head);
        return true;
    }

    void push(const T& value) noexcept {
        const size_t head = producer_.head_.fetch_add(1, std::memory_order_relaxed);
        push_direct(value, head);
    }

    [[nodiscard]] bool try_push(const T& value) noexcept {
        return try_emplace(value);
    }

    void pop(T& out) noexcept {
        const size_t tail = consumer_.tail_.fetch_add(1, std::memory_order_relaxed);
        out = pop_direct(tail);
    }

    [[nodiscard]] bool try_pop(T& out) noexcept {
        size_t tail = consumer_.tail_.load(std::memory_order_relaxed);
        do {
            const size_t head = producer_.head_.load(std::memory_order_relaxed);
            const ptrdiff_t available = head >= tail
                                            ? static_cast<ptrdiff_t>(head - tail)
                                            : -static_cast<ptrdiff_t>(tail - head);
            if (available <= 0) [[unlikely]] {
                return false;
            }
        } while (!consumer_.tail_.compare_exchange_weak(tail, tail + 1,
                                                        std::memory_order_relaxed,
                                                        std::memory_order_relaxed));

        out = pop_direct(tail);
        return true;
    }

    [[nodiscard]] std::optional<T> dequeue() noexcept {
        T value{};
        if (!try_pop(value)) {
            return std::nullopt;
        }
        return value;
    }

    [[nodiscard]] bool try_dequeue(T* out) noexcept {
        T value{};
        if (!try_pop(value)) {
            return false;
        }

        if (out) {
            *out = value;
        }
        return true;
    }

    bool enqueue(const T& item) noexcept {
        return try_emplace(item);
    }

    [[nodiscard]] bool empty() const noexcept {
        return size() == 0;
    }

    [[nodiscard]] size_t size() const noexcept {
        const size_t tail = consumer_.tail_.load(std::memory_order_acquire);
        const size_t head = producer_.head_.load(std::memory_order_acquire);
        return head >= tail ? head - tail : 0;
    }

    [[nodiscard]] constexpr size_t capacity() const noexcept {
        return CAPACITY;
    }
};
