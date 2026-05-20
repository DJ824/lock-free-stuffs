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
class LockFreeQueueMpmcSeqSplit {
    using AtomicSeq = std::atomic<size_t>;

    static_assert((SIZE & (SIZE - 1)) == 0, "size must be power of 2");
    static_assert(std::is_trivially_copyable_v<T>,
                  "LockFreeQueueMpmcSeqSplit requires trivially copyable T");
    static_assert(std::is_trivially_default_constructible_v<T>,
                  "LockFreeQueueMpmcSeqSplit requires trivially default-constructible T");

    static constexpr size_t CAPACITY = SIZE;
    static constexpr size_t MASK = CAPACITY - 1;

    struct alignas(CACHE_LINE_SIZE) SeqSlot {
        AtomicSeq seq{0};
    };

    static constexpr size_t VALUE_PADDING = (CACHE_LINE_SIZE - 1) / sizeof(T) + 1;
    static constexpr size_t VALUE_NSLOTS = CAPACITY + 2 * VALUE_PADDING;
    static constexpr size_t SEQ_PADDING = (CACHE_LINE_SIZE - 1) / sizeof(SeqSlot) + 1;
    static constexpr size_t SEQ_NSLOTS = CAPACITY + 2 * SEQ_PADDING;

    struct alignas(CACHE_LINE_SIZE) ProducerState {
        std::atomic<size_t> head_{0};
    } producer_;

    struct alignas(CACHE_LINE_SIZE) ConsumerState {
        std::atomic<size_t> tail_{0};
    } consumer_;

    alignas(CACHE_LINE_SIZE) std::array<T, VALUE_NSLOTS> values_{};
    alignas(CACHE_LINE_SIZE) std::array<SeqSlot, SEQ_NSLOTS> seqs_{};

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

    static constexpr size_t VALUE_ELEMENTS_PER_CACHE_LINE = CACHE_LINE_SIZE / sizeof(T);
    static constexpr int VALUE_SHUFFLE_BITS = shuffle_bits(VALUE_ELEMENTS_PER_CACHE_LINE);

    [[nodiscard]] static constexpr size_t value_index(size_t index) noexcept {
        return remap_index<VALUE_SHUFFLE_BITS>(index & MASK) + VALUE_PADDING;
    }

    [[nodiscard]] static constexpr size_t seq_index(size_t index) noexcept {
        return (index & MASK) + SEQ_PADDING;
    }

    [[nodiscard]] T& value_ref(size_t index) noexcept {
        return values_[value_index(index)];
    }

    [[nodiscard]] SeqSlot& seq_ref(size_t index) noexcept {
        return seqs_[seq_index(index)];
    }

    static void spin_pause() noexcept {
        _mm_pause();
        _mm_pause();
    }

    static void wait_for_sequence(AtomicSeq& seq, size_t expected) noexcept {
        while (seq.load(std::memory_order_acquire) != expected) {
            spin_pause();
        }
    }

    template <typename U>
    void push_direct(U&& value, size_t pos) noexcept {
        SeqSlot& seq_slot = seq_ref(pos);
        wait_for_sequence(seq_slot.seq, pos);
        value_ref(pos) = std::forward<U>(value);
        seq_slot.seq.store(pos + 1, std::memory_order_release);
    }

    void pop_direct(size_t pos, T& out) noexcept {
        SeqSlot& seq_slot = seq_ref(pos);
        wait_for_sequence(seq_slot.seq, pos + 1);
        out = value_ref(pos);
        seq_slot.seq.store(pos + CAPACITY, std::memory_order_release);
    }

public:
    explicit LockFreeQueueMpmcSeqSplit() noexcept {
        for (size_t i = 0; i < CAPACITY; ++i) {
            seq_ref(i).seq.store(i, std::memory_order_relaxed);
        }
    }

    LockFreeQueueMpmcSeqSplit(const LockFreeQueueMpmcSeqSplit&) = delete;
    LockFreeQueueMpmcSeqSplit& operator=(const LockFreeQueueMpmcSeqSplit&) = delete;

    void emplace(const T& value) noexcept {
        push(value);
    }

    void pop(T& out) noexcept {
        const size_t pos = consumer_.tail_.fetch_add(1, std::memory_order_relaxed);
        pop_direct(pos, out);
    }

    void push(const T& value) noexcept {
        const size_t pos = producer_.head_.fetch_add(1, std::memory_order_relaxed);
        push_direct(value, pos);
    }

    [[nodiscard]] bool try_emplace(const T& value) noexcept {
        for (;;) {
            size_t pos = producer_.head_.load(std::memory_order_relaxed);
            SeqSlot& seq_slot = seq_ref(pos);
            const size_t observed = seq_slot.seq.load(std::memory_order_acquire);
            const std::intptr_t diff =
                static_cast<std::intptr_t>(observed) - static_cast<std::intptr_t>(pos);

            if (diff == 0) {
                if (producer_.head_.compare_exchange_weak(pos, pos + 1,
                                                          std::memory_order_relaxed,
                                                          std::memory_order_relaxed)) {
                    value_ref(pos) = value;
                    seq_slot.seq.store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false;
            } else {
                spin_pause();
            }
        }
    }

    [[nodiscard]] bool try_push(const T& value) noexcept {
        return try_emplace(value);
    }

    [[nodiscard]] bool try_pop(T& out) noexcept {
        for (;;) {
            size_t pos = consumer_.tail_.load(std::memory_order_relaxed);
            SeqSlot& seq_slot = seq_ref(pos);
            const size_t observed = seq_slot.seq.load(std::memory_order_acquire);
            const std::intptr_t diff =
                static_cast<std::intptr_t>(observed) - static_cast<std::intptr_t>(pos + 1);

            if (diff == 0) {
                if (consumer_.tail_.compare_exchange_weak(pos, pos + 1,
                                                          std::memory_order_relaxed,
                                                          std::memory_order_relaxed)) {
                    out = value_ref(pos);
                    seq_slot.seq.store(pos + CAPACITY, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false;
            } else {
                spin_pause();
            }
        }
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
