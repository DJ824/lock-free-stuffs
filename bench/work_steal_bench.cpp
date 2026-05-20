#include "../include/work_steal_deque.h"
#include "../include/wsq.h"
#include "runtime.h"

#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using Clock = std::chrono::steady_clock;
using Duration = std::chrono::duration<double>;

namespace {

template <typename T, std::size_t LogSize>
using OursBoundedWSQ = WorkStealDeque<T, (std::size_t{1} << LogSize)>;

struct BenchmarkConfig {
    long num_ops{1'000'000};
    int num_thieves{3};
    int rounds{20};
    double min_sample_ms{15.0};
    std::vector<unsigned> pinned_cpus;
};

BenchmarkConfig g_config;

double elapsed_sec(Clock::time_point start, Clock::time_point end) {
    return Duration(end - start).count();
}

static constexpr int LABEL_W = 72;

void pin_current_thread_if_configured(unsigned index) {
    if (index < g_config.pinned_cpus.size()) {
        support::set_thread_affinity(g_config.pinned_cpus[index]);
    }
}

std::vector<unsigned> parse_cpu_list(const char* value) {
    std::vector<unsigned> cpu_ids;
    char const* p = value;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == ',') {
            ++p;
        }
        if (!*p) {
            break;
        }

        char* end = nullptr;
        const unsigned long cpu = std::strtoul(p, &end, 10);
        if (end == p || !end || (*end != '\0' && *end != ',' && *end != ' ' && *end != '\t')) {
            throw std::runtime_error("CPU list must be a comma-separated list of integers.");
        }

        cpu_ids.push_back(static_cast<unsigned>(cpu));
        p = end;
    }

    if (cpu_ids.empty()) {
        throw std::runtime_error("CPU list must contain at least one hardware thread id.");
    }
    return cpu_ids;
}

std::vector<unsigned> default_one_thread_per_core_desc() {
    auto topology = support::get_cpu_topology_info();
    std::sort(topology.begin(), topology.end(), [](const auto& a, const auto& b) {
        return std::tie(a.socket_id, a.core_id, a.hw_thread_id) <
               std::tie(b.socket_id, b.core_id, b.hw_thread_id);
    });

    std::vector<unsigned> selected;
    for (std::size_t i = 0; i < topology.size();) {
        std::size_t j = i + 1;
        while (j < topology.size() &&
               topology[j].socket_id == topology[i].socket_id &&
               topology[j].core_id == topology[i].core_id) {
            ++j;
        }
        selected.push_back(topology[j - 1].hw_thread_id);
        i = j;
    }

    std::sort(selected.begin(), selected.end(), std::greater<unsigned>{});
    return selected;
}

BenchmarkConfig parse_config(int argc, char* argv[]) {
    BenchmarkConfig config;
    std::vector<const char*> positionals;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--cpu-list") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("--cpu-list requires a comma-separated list of hardware thread ids");
            }
            config.pinned_cpus = parse_cpu_list(argv[++i]);
            continue;
        }
        if (std::strcmp(argv[i], "--min-sample-ms") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("--min-sample-ms requires a positive numeric value");
            }
            config.min_sample_ms = std::strtod(argv[++i], nullptr);
            continue;
        }
        positionals.push_back(argv[i]);
    }

    if (positionals.size() > 3) {
        throw std::runtime_error("Usage: work_steal_bench [num_ops] [num_thieves] [rounds] [--min-sample-ms ms] [--cpu-list list]");
    }

    if (positionals.size() >= 1) {
        config.num_ops = std::atol(positionals[0]);
    }
    if (positionals.size() >= 2) {
        config.num_thieves = std::atoi(positionals[1]);
    }
    if (positionals.size() >= 3) {
        config.rounds = std::atoi(positionals[2]);
    }

    if (config.num_ops <= 0 || config.num_thieves <= 0 || config.rounds <= 0 || config.min_sample_ms <= 0.0) {
        throw std::runtime_error("num_ops, num_thieves, rounds, and min_sample_ms must all be positive.");
    }

    if (config.pinned_cpus.empty()) {
        if (const char* env = std::getenv("ATOMIC_QUEUE_CPU_LIST")) {
            config.pinned_cpus = parse_cpu_list(env);
        } else {
            config.pinned_cpus = default_one_thread_per_core_desc();
        }
    }

    const std::size_t required_threads = static_cast<std::size_t>(config.num_thieves + 1);
    if (config.pinned_cpus.size() < required_threads) {
        throw std::runtime_error("Pinned CPU list is smaller than the required owner+thief thread count.");
    }

    return config;
}

void print_result(const char* label, const std::vector<double>& mops) {
    const int rounds = static_cast<int>(mops.size());
    const double mean = std::accumulate(mops.begin(), mops.end(), 0.0) / rounds;
    double variance = 0.0;
    for (double sample : mops) {
        variance += (sample - mean) * (sample - mean);
    }
    const double stddev = rounds > 1 ? std::sqrt(variance / (rounds - 1)) : 0.0;

    std::printf("  %-*s  %9.2f +- %7.2f Mops/s\n", LABEL_W, label, mean, stddev);
    std::fflush(stdout);
}

template <typename BenchFn>
void bench(const char* label, long ops, int rounds, BenchFn fn) {
    std::vector<double> mops;
    mops.reserve(rounds);
    const double min_sample_sec = g_config.min_sample_ms / 1000.0;
    for (int i = 0; i <= rounds; ++i) {
        double elapsed = 0.0;
        std::uint64_t effective_ops = 0;
        do {
            elapsed += fn();
            effective_ops += static_cast<std::uint64_t>(ops);
        } while (elapsed < min_sample_sec);
        if (i > 0) {
            mops.push_back(static_cast<double>(effective_ops) / elapsed / 1e6);
        }
    }
    print_result(label, mops);
}

template <template <typename, std::size_t> class Queue, typename T, std::size_t LogSize>
double run_bounded_push_pop(long count) {
    Queue<T, LogSize> q;
    std::vector<T> data(count, T{});
    const long cap = static_cast<long>(q.capacity());

    const auto start = Clock::now();
    for (long i = 0; i < count;) {
        const long batch = std::min(cap, count - i);
        for (long j = 0; j < batch; ++j) {
            (void)q.try_push(data[i + j]);
        }
        for (long j = 0; j < batch; ++j) {
            (void)q.pop();
        }
        i += batch;
    }
    return elapsed_sec(start, Clock::now());
}

template <template <typename, std::size_t> class Queue, typename T, std::size_t LogSize>
double run_bounded_interleaved(long count) {
    Queue<T, LogSize> q;
    std::vector<T> data(count, T{});

    const auto start = Clock::now();
    for (long i = 0; i < count; ++i) {
        (void)q.try_push(data[i]);
        (void)q.pop();
    }
    return elapsed_sec(start, Clock::now());
}

template <template <typename, std::size_t> class Queue, typename T, std::size_t LogSize>
double run_bounded_burst(long count, int fill_pct) {
    Queue<T, LogSize> q;
    std::vector<T> data(count, T{});
    const long cap = static_cast<long>(q.capacity());
    const long burst = std::max(1L, cap * fill_pct / 100);

    const auto start = Clock::now();
    for (long i = 0; i < count;) {
        const long batch = std::min(burst, count - i);
        for (long j = 0; j < batch; ++j) {
            (void)q.try_push(data[i + j]);
        }
        for (long j = 0; j < batch; ++j) {
            (void)q.pop();
        }
        i += batch;
    }
    return elapsed_sec(start, Clock::now());
}

template <template <typename, std::size_t> class Queue, typename T, std::size_t LogSize>
double run_bounded_bulk(long count) {
    Queue<T, LogSize> q;
    std::vector<T> data(count, T{});
    const long cap = static_cast<long>(q.capacity());
    constexpr long batch_size = 64;

    const auto start = Clock::now();
    for (long i = 0; i < count;) {
        const long window = std::min(cap, count - i);
        for (long j = 0; j + batch_size <= window; j += batch_size) {
            (void)q.try_bulk_push(data.data() + i + j, batch_size);
        }
        while (q.pop()) {
        }
        i += window;
    }
    return elapsed_sec(start, Clock::now());
}

template <template <typename, std::size_t> class Queue, typename T, std::size_t LogSize>
double run_bounded_bulk_var(long count, long batch_size) {
    Queue<T, LogSize> q;
    std::vector<T> data(count, T{});
    const long cap = static_cast<long>(q.capacity());

    const auto start = Clock::now();
    for (long i = 0; i < count;) {
        const long window = std::min(cap, count - i);
        for (long j = 0; j + batch_size <= window; j += batch_size) {
            (void)q.try_bulk_push(data.data() + i + j, static_cast<std::size_t>(batch_size));
        }
        while (q.pop()) {
        }
        i += window;
    }
    return elapsed_sec(start, Clock::now());
}

template <template <typename, std::size_t> class Queue, typename T, std::size_t LogSize>
double run_bounded_concurrent(long count, int thieves) {
    Queue<T, LogSize> q;
    std::vector<T> data(count, T{});
    const long cap = static_cast<long>(q.capacity());
    std::atomic<long> remaining{count};
    std::atomic<bool> started{false};
    std::barrier start_barrier(static_cast<std::ptrdiff_t>(thieves + 1));

    pin_current_thread_if_configured(0);
    double elapsed = 0.0;
    {
        std::vector<std::jthread> thief_threads;
        thief_threads.reserve(thieves);
        for (int i = 0; i < thieves; ++i) {
            thief_threads.emplace_back([&, i] {
                pin_current_thread_if_configured(static_cast<unsigned>(i + 1));
                start_barrier.arrive_and_wait();
                while (remaining.load(std::memory_order_acquire) > 0) {
                    while (!started.load(std::memory_order_acquire) &&
                           remaining.load(std::memory_order_acquire) > 0) {
                        std::this_thread::yield();
                    }
                    while (q.steal()) {
                        remaining.fetch_sub(1, std::memory_order_relaxed);
                    }
                }
            });
        }

        start_barrier.arrive_and_wait();
        const auto start = Clock::now();
        for (long base = 0; base < count; base += cap) {
            const long batch = std::min(cap, count - base);
            for (long j = 0; j < batch; ++j) {
                (void)q.try_push(data[base + j]);
            }
            started.store(true, std::memory_order_release);
            while (q.pop()) {
                remaining.fetch_sub(1, std::memory_order_relaxed);
            }
            started.store(false, std::memory_order_release);
        }
        while (remaining.load(std::memory_order_acquire) > 0) {
            std::this_thread::yield();
        }
        elapsed = elapsed_sec(start, Clock::now());
    }
    return elapsed;
}

template <template <typename, std::size_t> class Queue, typename T, std::size_t LogSize>
double run_bounded_continuous(long count, int thieves) {
    Queue<T, LogSize> q;
    std::vector<T> data(count, T{});

    std::atomic<long> remaining{count};
    std::atomic<bool> done{false};
    std::barrier start_barrier(static_cast<std::ptrdiff_t>(thieves + 1));

    pin_current_thread_if_configured(0);
    double elapsed = 0.0;
    {
        std::vector<std::jthread> thief_threads;
        thief_threads.reserve(thieves);
        for (int i = 0; i < thieves; ++i) {
            thief_threads.emplace_back([&, i] {
                pin_current_thread_if_configured(static_cast<unsigned>(i + 1));
                start_barrier.arrive_and_wait();
                while (!done.load(std::memory_order_acquire) ||
                       remaining.load(std::memory_order_relaxed) > 0) {
                    if (q.steal()) {
                        remaining.fetch_sub(1, std::memory_order_relaxed);
                    } else {
                        std::this_thread::yield();
                    }
                }
            });
        }

        start_barrier.arrive_and_wait();
        const auto start = Clock::now();
        for (long i = 0; i < count;) {
            if (q.try_push(data[i])) {
                ++i;
            } else if (q.pop()) {
                remaining.fetch_sub(1, std::memory_order_relaxed);
            } else {
                std::this_thread::yield();
            }
        }

        while (remaining.load(std::memory_order_relaxed) > 0) {
            if (q.pop()) {
                remaining.fetch_sub(1, std::memory_order_relaxed);
            } else {
                std::this_thread::yield();
            }
        }
        done.store(true, std::memory_order_release);
        elapsed = elapsed_sec(start, Clock::now());
    }
    return elapsed;
}

template <template <typename, std::size_t> class Queue, typename T, std::size_t LogSize>
double run_bounded_sparse(long count, int thieves) {
    const long sparse_count = std::min(count, 10'000L);
    Queue<T, LogSize> q;
    std::vector<T> data(sparse_count, T{});

    std::atomic<long> remaining{sparse_count};
    std::atomic<bool> done{false};
    std::barrier start_barrier(static_cast<std::ptrdiff_t>(thieves + 1));

    pin_current_thread_if_configured(0);
    double elapsed = 0.0;
    {
        std::vector<std::jthread> thief_threads;
        thief_threads.reserve(thieves);
        for (int i = 0; i < thieves; ++i) {
            thief_threads.emplace_back([&, i] {
                pin_current_thread_if_configured(static_cast<unsigned>(i + 1));
                start_barrier.arrive_and_wait();
                std::size_t empty_steals = 0;
                while (!done.load(std::memory_order_acquire) ||
                       remaining.load(std::memory_order_relaxed) > 0) {
                    if (q.steal_with_feedback(empty_steals)) {
                        remaining.fetch_sub(1, std::memory_order_relaxed);
                    } else if (empty_steals > 16) {
                        std::this_thread::yield();
                    }
                }
            });
        }

        start_barrier.arrive_and_wait();
        const auto start = Clock::now();
        for (long i = 0; i < sparse_count; ++i) {
            while (!q.try_push(data[i])) {
                std::this_thread::yield();
            }
            std::this_thread::yield();
        }

        done.store(true, std::memory_order_release);
        while (remaining.load(std::memory_order_acquire) > 0) {
            std::this_thread::yield();
        }
        elapsed = elapsed_sec(start, Clock::now());
    }
    return elapsed;
}

template <template <typename, std::size_t> class Queue, typename T, std::size_t LogSize>
double run_bounded_bulk_concurrent(long count, int thieves) {
    Queue<T, LogSize> q;
    std::vector<T> data(count, T{});
    const long cap = static_cast<long>(q.capacity());
    std::barrier start_barrier(static_cast<std::ptrdiff_t>(thieves + 1));
    std::barrier barrier(static_cast<std::ptrdiff_t>(thieves + 1));

    pin_current_thread_if_configured(0);
    double elapsed = 0.0;
    {
        std::vector<std::jthread> thief_threads;
        thief_threads.reserve(thieves);
        for (int i = 0; i < thieves; ++i) {
            thief_threads.emplace_back([&, i] {
                pin_current_thread_if_configured(static_cast<unsigned>(i + 1));
                start_barrier.arrive_and_wait();
                for (long base = 0; base < count; base += cap) {
                    barrier.arrive_and_wait();
                    while (true) {
                        if (q.steal()) {
                        } else if (q.empty()) {
                            break;
                        } else {
                            std::this_thread::yield();
                        }
                    }
                    barrier.arrive_and_wait();
                }
            });
        }

        start_barrier.arrive_and_wait();
        const auto start = Clock::now();
        for (long base = 0; base < count; base += cap) {
            const long window = std::min(cap, count - base);
            for (long j = 0; j + 64 <= window; j += 64) {
                (void)q.try_bulk_push(data.data() + base + j, 64);
            }
            barrier.arrive_and_wait();
            while (true) {
                if (q.pop()) {
                } else if (q.empty()) {
                    break;
                } else {
                    std::this_thread::yield();
                }
            }
            barrier.arrive_and_wait();
        }
        elapsed = elapsed_sec(start, Clock::now());
    }
    return elapsed;
}

template <template <typename, std::size_t> class Queue, typename T, std::size_t LogSize>
double run_bounded_multi_queue(long count, int thread_count) {
    std::vector<std::unique_ptr<Queue<T, LogSize>>> queues;
    queues.reserve(thread_count);
    for (int i = 0; i < thread_count; ++i) {
        queues.push_back(std::make_unique<Queue<T, LogSize>>());
    }

    std::vector<T> data(count, T{});
    const long per_queue = count / thread_count;
    std::atomic<long> remaining{count};
    std::barrier start_barrier(static_cast<std::ptrdiff_t>(thread_count + 1));

    double elapsed = 0.0;
    {
        std::vector<std::jthread> threads;
        threads.reserve(thread_count);
        for (int qi = 0; qi < thread_count; ++qi) {
            threads.emplace_back([&, qi] {
                pin_current_thread_if_configured(static_cast<unsigned>(qi));
                start_barrier.arrive_and_wait();
                const int victim = (qi + 1) % thread_count;
                const long start_index = static_cast<long>(qi) * per_queue;
                const long end_index = qi == thread_count - 1 ? count : start_index + per_queue;
                long pushed = start_index;

                while (remaining.load(std::memory_order_relaxed) > 0) {
                    if (pushed < end_index) {
                        if (queues[qi]->try_push(data[pushed])) {
                            ++pushed;
                        }
                    }
                    if (queues[qi]->pop()) {
                        remaining.fetch_sub(1, std::memory_order_relaxed);
                    } else if (queues[victim]->steal()) {
                        remaining.fetch_sub(1, std::memory_order_relaxed);
                    } else if (pushed >= end_index) {
                        std::this_thread::yield();
                    }
                }
            });
        }

        start_barrier.arrive_and_wait();
        const auto start = Clock::now();
        while (remaining.load(std::memory_order_acquire) > 0) {
            std::this_thread::yield();
        }
        elapsed = elapsed_sec(start, Clock::now());
    }
    return elapsed;
}

template <template <typename, std::size_t> class Queue, typename T, std::size_t LogSize>
void bench_bounded_impl(const char* impl_name, const char* type_name, long count, int thieves, int rounds) {
    char label[192];
    const int thread_count = thieves + 1;

    std::printf("\n  -- %s<%s, %zu> (capacity=%zu) --\n",
                impl_name,
                type_name,
                LogSize,
                std::size_t{1} << LogSize);
    std::fflush(stdout);

    std::snprintf(label, sizeof(label), "%s<%s,%2zu> [1] push+pop (owner only)", impl_name, type_name, LogSize);
    bench(label, count * 2, rounds, [&] { return run_bounded_push_pop<Queue, T, LogSize>(count); });

    std::snprintf(label, sizeof(label), "%s<%s,%2zu> [2] interleaved push/pop (owner only)", impl_name, type_name, LogSize);
    bench(label, count * 2, rounds, [&] { return run_bounded_interleaved<Queue, T, LogSize>(count); });

    for (int pct : {25, 50, 75}) {
        std::snprintf(label, sizeof(label), "%s<%s,%2zu> [3] burst fill=%d%% (owner only)", impl_name, type_name, LogSize, pct);
        bench(label, count * 2, rounds, [&] { return run_bounded_burst<Queue, T, LogSize>(count, pct); });
    }

    std::snprintf(label, sizeof(label), "%s<%s,%2zu> [4] bulk_push+pop batch=64 (owner only)", impl_name, type_name, LogSize);
    bench(label, count * 2, rounds, [&] { return run_bounded_bulk<Queue, T, LogSize>(count); });

    for (long batch_size : {1L, 16L, 64L, 256L}) {
        std::snprintf(label, sizeof(label), "%s<%s,%2zu> [5] bulk batch=%-3ld (owner only)", impl_name, type_name, LogSize, batch_size);
        bench(label, count * 2, rounds, [&] { return run_bounded_bulk_var<Queue, T, LogSize>(count, batch_size); });
    }

    std::snprintf(label, sizeof(label), "%s<%s,%2zu> [6] concurrent drain (%d thieves)", impl_name, type_name, LogSize, thieves);
    bench(label, count, rounds, [&] { return run_bounded_concurrent<Queue, T, LogSize>(count, thieves); });

    std::snprintf(label, sizeof(label), "%s<%s,%2zu> [7] continuous push+steal (%d thieves)", impl_name, type_name, LogSize, thieves);
    bench(label, count, rounds, [&] { return run_bounded_continuous<Queue, T, LogSize>(count, thieves); });

    std::snprintf(label, sizeof(label), "%s<%s,%2zu> [8] sparse steal_with_feedback (%d thieves)", impl_name, type_name, LogSize, thieves);
    bench(label, count, rounds, [&] { return run_bounded_sparse<Queue, T, LogSize>(count, thieves); });

    std::snprintf(label, sizeof(label), "%s<%s,%2zu> [9] bulk+concurrent drain (%d thieves)", impl_name, type_name, LogSize, thieves);
    bench(label, count, rounds, [&] { return run_bounded_bulk_concurrent<Queue, T, LogSize>(count, thieves); });

    std::snprintf(label, sizeof(label), "%s<%s,%2zu>[10] multi_queue (%d threads)", impl_name, type_name, LogSize, thread_count);
    bench(label, count, rounds, [&] { return run_bounded_multi_queue<Queue, T, LogSize>(count, thread_count); });
}

template <typename T, std::size_t LogSize>
void bench_combination(const char* type_name, long count, int thieves, int rounds) {
    bench_bounded_impl<wsq::BoundedWSQ, T, LogSize>("wsq::BoundedWSQ", type_name, count, thieves, rounds);
    bench_bounded_impl<OursBoundedWSQ, T, LogSize>("WorkStealDeque", type_name, count, thieves, rounds);
}

template <typename T>
void bench_type(const char* type_name, long count, int thieves, int rounds) {
    std::printf("\n======== Type: %s ========\n", type_name);
    std::fflush(stdout);
    bench_combination<T, 8>(type_name, count, thieves, rounds);
    bench_combination<T, 9>(type_name, count, thieves, rounds);
    bench_combination<T, 10>(type_name, count, thieves, rounds);
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        g_config = parse_config(argc, argv);
    } catch (const std::exception& ex) {
        std::fprintf(stderr,
                     "Error: %s\n"
                     "Usage: %s [num_ops] [num_thieves] [rounds] [--min-sample-ms ms] [--cpu-list list]\n"
                     "  num_ops     > 0  (default 1000000)\n"
                     "  num_thieves > 0  (default 3)\n"
                     "  rounds      > 0  (default 20; +1 warm-up not counted)\n"
                     "  --min-sample-ms  > 0  (default 15; repeats each sample until this duration)\n"
                     "  --cpu-list  comma-separated hardware thread ids\n"
                     "  env fallback: ATOMIC_QUEUE_CPU_LIST=15,14,13,12\n",
                     ex.what(),
                     argv[0]);
        return 1;
    }

    support::set_thread_affinity(g_config.pinned_cpus.front());

    const int total_threads = g_config.num_thieves + 1;
    std::printf("Work-Stealing Queue Benchmark\n");
    std::printf("  num_ops    = %ld\n", g_config.num_ops);
    std::printf("  n_thieves  = %d\n", g_config.num_thieves);
    std::printf("  n_threads  = %d\n", total_threads);
    std::printf("  rounds     = %d (+ 1 warm-up)\n", g_config.rounds);
    std::printf("  min_sample = %.1f ms\n", g_config.min_sample_ms);
    std::printf("  hw_threads = %u\n", std::thread::hardware_concurrency());
    std::printf("  pinned_cpus=");
    for (std::size_t i = 0; i < g_config.pinned_cpus.size(); ++i) {
        std::printf("%s%u", i == 0 ? "" : ",", g_config.pinned_cpus[i]);
    }
    std::printf("\n");
    std::printf("  types      = int, int64_t\n");
    std::printf("  log_sizes  = 8, 9, 10\n");
    std::printf("\n  %-*s  %26s\n", LABEL_W, "benchmark", "mean +- stddev Mops/s");
    std::printf("  %s\n", std::string(LABEL_W + 30, '-').c_str());

    bench_type<int>("int", g_config.num_ops, g_config.num_thieves, g_config.rounds);
    bench_type<std::int64_t>("int64_t", g_config.num_ops, g_config.num_thieves, g_config.rounds);

    std::printf("\nDone.\n");
    support::reset_thread_affinity();
    return 0;
}
