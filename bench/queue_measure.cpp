#include "queue_measure.h"

#include "../include/spsc_4.h"

#include "memory.h"
#include "runtime.h"

#include <algorithm>
#include <array>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

using BenchValue = unsigned;
using measure::BenchMode;
using measure::RetryProfile;
using measure::cycles_t;
using measure::sum_t;
using support::HugePageAllocatorBase;
using support::HugePages;

struct BenchConfig {
    std::array<unsigned, 2> cpus{};
    unsigned message_count{2'000'000};
};

template <class T, unsigned Capacity>
struct SpscQueue4Adapter {
    using value_type = T;
    LockFreeQueueStage4<T, Capacity> queue;

    void push(T element) noexcept {
        queue.emplace(element);
    }

    bool try_push(T element) noexcept {
        return queue.try_emplace(element);
    }

    T pop() noexcept {
        T element{};
        queue.pop(element);
        return element;
    }

    bool try_pop(T& element) noexcept {
        return queue.try_pop(element);
    }
};

double to_seconds(cycles_t cycles) {
    return cycles * (1e-9 / support::cpu_base_frequency());
}

void advise_hugeadm_1GB() {
    std::fprintf(stderr,
                 "Warning: Failed to allocate 1GB huge pages. Run "
                 "\"sudo hugeadm --pool-pages-min 1GB:1 --pool-pages-max 1GB:1\".\n");
}

void advise_hugeadm_2MB() {
    std::fprintf(stderr,
                 "Warning: Failed to allocate 2MB huge pages. Run "
                 "\"sudo hugeadm --pool-pages-min 2MB:16 --pool-pages-max 2MB:16\".\n");
}

void check_huge_pages_leaks(char const* name, HugePages& hp) {
    if (!hp.empty()) {
        std::fprintf(stderr, "%s: %zu bytes of HugePages memory leaked.\n", name, hp.used());
        hp.reset();
    }
}

bool use_huge_pages(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--huge-pages") == 0) {
            return true;
        }
    }

    if (char const* env = std::getenv("QUEUE_MEASURE_USE_HUGEPAGES")) {
        return std::strcmp(env, "0") != 0;
    }

    char const* env = std::getenv("QUEUE_BENCH2_USE_HUGEPAGES");
    return env && std::strcmp(env, "0") != 0;
}

HugePages make_benchmark_pages(bool enable_huge_pages) {
    size_t constexpr MB = 1024 * 1024;
    if (!enable_huge_pages) {
        return HugePages(HugePages::PAGE_DEFAULT, 4 * MB);
    }

    try {
        return HugePages(HugePages::PAGE_1GB, 32 * MB);
    }
    catch (std::system_error const&) {
        std::fprintf(stderr,
                     "Warning: Falling back to a smaller default-page benchmark arena.\n");
        return HugePages(HugePages::PAGE_DEFAULT, 4 * MB);
    }
}

unsigned parse_cpu_id(char const* arg, char const* flag) {
    char* end = nullptr;
    unsigned long value = std::strtoul(arg, &end, 10);
    if (!arg[0] || !end || *end != '\0') {
        throw std::runtime_error(std::string(flag) + " expects an unsigned integer cpu id");
    }
    return static_cast<unsigned>(value);
}

unsigned parse_message_count(char const* arg) {
    char* end = nullptr;
    unsigned long value = std::strtoul(arg, &end, 10);
    if (!arg[0] || !end || *end != '\0' || value == 0 ||
        value > std::numeric_limits<unsigned>::max()) {
        throw std::runtime_error("--messages expects a positive unsigned integer");
    }
    return static_cast<unsigned>(value);
}

std::array<unsigned, 2> default_benchmark_cpus(std::vector<unsigned> const& hw_thread_ids) {
    auto has_cpu = [&](unsigned cpu) {
        return std::find(hw_thread_ids.begin(), hw_thread_ids.end(), cpu) != hw_thread_ids.end();
    };

    if (has_cpu(14) && has_cpu(15)) {
        return {14, 15};
    }

    return {hw_thread_ids[0], hw_thread_ids[1]};
}

BenchConfig parse_bench_config(int argc, char** argv, std::vector<unsigned> const& hw_thread_ids) {
    BenchConfig config{default_benchmark_cpus(hw_thread_ids)};

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--cpus") == 0) {
            if (i + 2 >= argc) {
                throw std::runtime_error("--cpus requires two hardware thread ids");
            }
            config.cpus = {
                parse_cpu_id(argv[i + 1], "--cpus"),
                parse_cpu_id(argv[i + 2], "--cpus")
            };
            i += 2;
        }
        else if (std::strcmp(argv[i], "--messages") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("--messages requires a positive integer");
            }
            config.message_count = parse_message_count(argv[i + 1]);
            i += 1;
        }
        else if (std::strcmp(argv[i], "--huge-pages") == 0) {
        }
        else {
            throw std::runtime_error(std::string("Unknown argument: ") + argv[i]);
        }
    }

    return config;
}

template <class Queue>
void run_retry_profile_benchmark(char const* name,
                                 HugePages& hp,
                                 std::array<unsigned, 2> const& cpus,
                                 unsigned message_count) {
    sum_t consumer_sum = 0;
    RetryProfile retry_profile{};
    cycles_t run_time =
        measure::benchmark_throughput<BenchMode::NonBlocking, Queue>(hp, cpus, message_count, &consumer_sum,
                                                                     &retry_profile);
    check_huge_pages_leaks(name, hp);

    sum_t expected_sum = (message_count + 1) / 2.0 * message_count;
    if (consumer_sum != expected_sum) {
        std::fprintf(stderr,
                     "%s: retry profile checksum mismatch: expected %'lld got %'lld.\n",
                     name,
                     expected_sum,
                     consumer_sum);
    }

    double seconds = to_seconds(run_time);
    unsigned msg_per_sec = static_cast<unsigned>(message_count / seconds);
    std::printf("%32s: %'11u msg/sec\n", name, msg_per_sec);
    measure::print_retry_stats("producer try_push", retry_profile.push);
    measure::print_retry_stats("consumer try_pop", retry_profile.pop);
}

} // namespace

int main(int argc, char** argv) {
    std::setlocale(LC_NUMERIC, "");

    auto cpu_topology = support::get_cpu_topology_info();
    if (cpu_topology.size() < 2) {
        throw std::runtime_error("A CPU with at least 2 hardware threads is required.");
    }

    auto hw_thread_ids = support::hw_thread_id(cpu_topology);
    auto config = parse_bench_config(argc, argv, hw_thread_ids);

    bool const enable_huge_pages = use_huge_pages(argc, argv);
    if (enable_huge_pages) {
        HugePages::warn_no_1GB_pages = advise_hugeadm_1GB;
        HugePages::warn_no_2MB_pages = advise_hugeadm_2MB;
    }
    HugePages hp = make_benchmark_pages(enable_huge_pages);
    HugePageAllocatorBase::hp = &hp;

    std::printf("---- Thread pinning enabled: producer cpu=%u, consumer cpu=%u ----\n",
                config.cpus[0],
                config.cpus[1]);
    std::printf("---- Measuring stalls with the nonblocking queue API (%'u messages) ----\n",
                config.message_count);

    unsigned constexpr capacity = 1 << 16;
    run_retry_profile_benchmark<SpscQueue4Adapter<BenchValue, capacity>>(
        "spsc_4", hp, config.cpus, config.message_count);

    return 0;
}
