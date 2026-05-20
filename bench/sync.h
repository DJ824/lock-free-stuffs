#pragma once

#include <atomic>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <emmintrin.h>
#endif

namespace support {

struct Context {
    unsigned producers;
    unsigned consumers;
};

struct NoContext {
    template <class... Args>
    constexpr NoContext(Args&&...) noexcept {
    }
};

template <class T>
typename T::ContextType context_of_(int);
template <class T>
NoContext context_of_(long);
template <class T>
using ContextOf = decltype(context_of_<T>(0));

static inline void spin_loop_pause() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    _mm_pause();
#elif defined(__aarch64__) || defined(_M_ARM64)
    asm volatile("yield" ::: "memory");
#else
#endif
}

class Barrier {
    std::atomic<unsigned> counter_{0};

public:
    void wait() noexcept {
        counter_.fetch_add(1, std::memory_order_acquire);
        while (counter_.load(std::memory_order_relaxed) != 0) {
            spin_loop_pause();
        }
    }

    void release(unsigned expected_counter) noexcept {
        while (counter_.load(std::memory_order_relaxed) != expected_counter) {
            spin_loop_pause();
        }
        counter_.store(0, std::memory_order_release);
    }
};

} // namespace support
