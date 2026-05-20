#pragma once

#include "memory.h"
#include "runtime.h"
#include "sync.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>

#include <x86intrin.h>

namespace measure {

using sum_t = long long;
using cycles_t = std::uint64_t;

enum class BenchMode {
    Blocking,
    NonBlocking,
};

struct RetryStats {
    std::uint64_t false_returns{0};
    std::uint64_t stalled_successes{0};
    std::uint64_t max_streak{0};
    std::uint64_t total_stall_cycles{0};

    void record(std::uint64_t streak, std::uint64_t cycles) noexcept {
        false_returns += streak;
        ++stalled_successes;
        max_streak = std::max(max_streak, streak);
        total_stall_cycles += cycles;
    }

    double average_streak() const noexcept {
        return stalled_successes ? static_cast<double>(false_returns) / stalled_successes : 0.0;
    }

    double average_stall_cycles() const noexcept {
        return stalled_successes ? static_cast<double>(total_stall_cycles) / stalled_successes : 0.0;
    }

    double average_cycles_per_false() const noexcept {
        return false_returns ? static_cast<double>(total_stall_cycles) / false_returns : 0.0;
    }
};

struct RetryProfile {
    RetryStats push;
    RetryStats pop;
};

inline void print_retry_stats(char const* label, RetryStats const& stats) {
    std::printf(
        "%36s false=%'11llu stalls=%'11llu avg_streak=%8.2f max=%'8llu avg_stall=%9.1f cyc avg_false=%7.1f cyc\n",
        label,
        static_cast<unsigned long long>(stats.false_returns),
        static_cast<unsigned long long>(stats.stalled_successes),
        stats.average_streak(),
        static_cast<unsigned long long>(stats.max_streak),
        stats.average_stall_cycles(),
        stats.average_cycles_per_false());
}

template <BenchMode Mode, class Queue>
inline void queue_push(Queue& queue,
                       typename Queue::value_type element,
                       RetryStats* stats = nullptr) noexcept {
    if constexpr (Mode == BenchMode::Blocking) {
        queue.push(element);
    } else {
        std::uint64_t streak = 0;
        cycles_t start = 0;
        while (!queue.try_push(element)) {
            if (stats && streak == 0) {
                start = __rdtsc();
            }
            ++streak;
            support::spin_loop_pause();
        }
        if (stats && streak != 0) {
            stats->record(streak, __rdtsc() - start);
        }
    }
}

template <BenchMode Mode, class Queue>
inline typename Queue::value_type queue_pop(Queue& queue,
                                            RetryStats* stats = nullptr) noexcept {
    if constexpr (Mode == BenchMode::Blocking) {
        return queue.pop();
    } else {
        typename Queue::value_type element{};
        std::uint64_t streak = 0;
        cycles_t start = 0;
        while (!queue.try_pop(element)) {
            if (stats && streak == 0) {
                start = __rdtsc();
            }
            ++streak;
            support::spin_loop_pause();
        }
        if (stats && streak != 0) {
            stats->record(streak, __rdtsc() - start);
        }
        return element;
    }
}

template <BenchMode Mode, class Queue>
inline void throughput_producer(unsigned message_count,
                                Queue* queue,
                                std::atomic<cycles_t>* start_time,
                                support::Barrier* barrier,
                                RetryStats* push_stats = nullptr) {
    barrier->wait();

    cycles_t expected = 0;
    start_time->compare_exchange_strong(expected,
                                        __rdtsc(),
                                        std::memory_order_acq_rel,
                                        std::memory_order_relaxed);

    for (std::uint64_t n = 1, stop = message_count + 1; n <= stop; ++n) {
        queue_push<Mode>(*queue, static_cast<typename Queue::value_type>(n), push_stats);
    }
}

template <BenchMode Mode, class Queue>
inline sum_t throughput_consumer(unsigned message_count,
                                 Queue* queue,
                                 cycles_t* end_time,
                                 RetryStats* pop_stats = nullptr) {
    unsigned const stop = message_count + 1;
    sum_t sum = 0;

    for (;;) {
        std::uint64_t n = queue_pop<Mode>(*queue, pop_stats);
        if (n == stop) {
            break;
        }
        sum += n;
    }

    *end_time = __rdtsc();
    return sum;
}

template <BenchMode Mode, class Queue>
inline cycles_t benchmark_throughput(support::HugePages& hp,
                                     std::array<unsigned, 2> const& cpus,
                                     unsigned message_count,
                                     sum_t* consumer_sum,
                                     RetryProfile* retry_profile = nullptr) {
    support::set_thread_affinity(cpus[1]);
    auto queue = hp.create_unique_ptr<Queue>(support::ContextOf<Queue>{1, 1});
    std::atomic<cycles_t> start_time{0};
    cycles_t end_time = 0;
    support::Barrier barrier;

    support::set_default_thread_affinity(cpus[0]);
    std::thread producer(throughput_producer<Mode, Queue>,
                         message_count,
                         queue.get(),
                         &start_time,
                         &barrier,
                         retry_profile ? &retry_profile->push : nullptr);
    barrier.release(1);
    *consumer_sum = throughput_consumer<Mode>(
        message_count,
        queue.get(),
        &end_time,
        retry_profile ? &retry_profile->pop : nullptr);
    producer.join();

    support::reset_thread_affinity();
    return end_time - start_time.load(std::memory_order_relaxed);
}

} // namespace measure
