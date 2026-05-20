#include <boost/lockfree/queue.hpp>
#include <folly/ProducerConsumerQueue.h>
#include <oneapi/tbb/concurrent_queue.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>
#include <x86intrin.h>

#include <atomic_queue/atomic_queue.h>
#include <concurrentqueue/concurrentqueue.h>

#include "../include/joad_q.h"
#include "../include/mpmc_1.h"
#include "../include/mpmc_seq.h"
#include "../include/mpmc_seq_split.h"
#include "memory.h"
#include "runtime.h"
#include "sync.h"

namespace {

using BenchValue = unsigned;
using sum_t = long long;
using cycles_t = std::uint64_t;
using support::Barrier;
using support::ContextOf;
using support::HugePageAllocatorBase;
using support::HugePages;
using support::spin_loop_pause;

double const TSC_TO_SECONDS = 1e-9 / support::cpu_base_frequency();

template <class T>
double to_seconds(T cycles) {
    return cycles * TSC_TO_SECONDS;
}

struct BenchmarkConfig {
    unsigned throughput_messages{1'000'000};
    unsigned throughput_runs{3};
    unsigned ping_pong_messages{100'000};
    unsigned ping_pong_runs{10};
    unsigned min_pairs{1};
    unsigned max_pairs{0};
    bool run_throughput{true};
    bool run_ping_pong{true};
    bool use_huge_pages{false};
};

struct NoToken {
    template <class... Args>
    constexpr NoToken(Args&&...) noexcept {
    }

    template <class Queue, class T>
    void push(Queue& queue, T&& element) noexcept {
        queue.push(std::forward<T>(element));
    }

    template <class Queue>
    auto pop(Queue& queue) noexcept {
        return queue.pop();
    }
};

template <class T>
typename T::Producer producer_of_(int);
template <class T>
NoToken producer_of_(long);
template <class T>
using ProducerOf = decltype(producer_of_<T>(0));

template <class T>
typename T::Consumer consumer_of_(int);
template <class T>
NoToken consumer_of_(long);
template <class T>
using ConsumerOf = decltype(consumer_of_<T>(0));

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

template <unsigned Capacity>
struct Mpmc1Adapter {
    using value_type = BenchValue;
    static constexpr unsigned kMaxPairs = std::numeric_limits<unsigned>::max();

    LockFreeQueueMpmc1<BenchValue, Capacity> queue;

    void push(BenchValue element) noexcept {
        queue.push(element);
    }

    BenchValue pop() noexcept {
        BenchValue element{};
        queue.pop(element);
        return element;
    }
};

template <unsigned Capacity>
struct MpmcSeqAdapter {
    using value_type = BenchValue;
    static constexpr unsigned kMaxPairs = std::numeric_limits<unsigned>::max();

    LockFreeQueueMpmcSeq<BenchValue, Capacity> queue;

    void push(BenchValue element) noexcept {
        queue.push(element);
    }

    BenchValue pop() noexcept {
        BenchValue element{};
        queue.pop(element);
        return element;
    }
};

template <unsigned Capacity>
struct MpmcSeqSplitAdapter {
    using value_type = BenchValue;
    static constexpr unsigned kMaxPairs = std::numeric_limits<unsigned>::max();

    LockFreeQueueMpmcSeqSplit<BenchValue, Capacity> queue;

    void push(BenchValue element) noexcept {
        queue.push(element);
    }

    BenchValue pop() noexcept {
        BenchValue element{};
        queue.pop(element);
        return element;
    }
};

template <unsigned Capacity>
struct JoadMpmcAdapter {
    using value_type = BenchValue;
    static constexpr unsigned kMaxPairs = std::numeric_limits<unsigned>::max();

    jdz::MpmcQueue<BenchValue, Capacity> queue;

    void push(BenchValue element) noexcept {
        queue.emplace(element);
    }

    BenchValue pop() noexcept {
        BenchValue element{};
        queue.pop(element);
        return element;
    }
};

template <unsigned Capacity>
struct BoostMpmcQueueAdapter {
    using value_type = BenchValue;
    static constexpr unsigned kMaxPairs = std::numeric_limits<unsigned>::max();
    static_assert(Capacity > 2, "BoostMpmcQueueAdapter requires Capacity > 2");

    boost::lockfree::queue<BenchValue, boost::lockfree::capacity<Capacity - 2>> queue;

    void push(BenchValue element) noexcept {
        while (!queue.bounded_push(element)) {
            spin_loop_pause();
        }
    }

    BenchValue pop() noexcept {
        BenchValue element{};
        while (!queue.pop(element)) {
            spin_loop_pause();
        }
        return element;
    }
};

template <unsigned Capacity>
struct FollyQueueReferenceAdapter {
    using value_type = BenchValue;
    static constexpr unsigned kMaxPairs = 1;

    folly::ProducerConsumerQueue<BenchValue> queue;

    FollyQueueReferenceAdapter()
        : queue(static_cast<uint32_t>(Capacity)) {
    }

    void push(BenchValue element) noexcept {
        while (!queue.write(element)) {
            spin_loop_pause();
        }
    }

    BenchValue pop() noexcept {
        BenchValue element{};
        while (!queue.read(element)) {
            spin_loop_pause();
        }
        return element;
    }
};

template <unsigned Capacity>
struct MoodyCamelQueueAdapter : moodycamel::ConcurrentQueue<BenchValue> {
    using value_type = BenchValue;
    using ContextType = support::Context;
    using producer_token_t = typename moodycamel::ConcurrentQueue<BenchValue>::producer_token_t;
    using consumer_token_t = typename moodycamel::ConcurrentQueue<BenchValue>::consumer_token_t;
    static constexpr unsigned kMaxPairs = std::numeric_limits<unsigned>::max();

    struct Producer {
        producer_token_t token_;

        explicit Producer(MoodyCamelQueueAdapter& queue) noexcept
            : token_(queue) {
        }

        void push(MoodyCamelQueueAdapter& queue, BenchValue element) noexcept {
            while (!queue.try_enqueue(token_, element)) {
                spin_loop_pause();
            }
        }
    };

    struct Consumer {
        consumer_token_t token_;

        explicit Consumer(MoodyCamelQueueAdapter& queue) noexcept
            : token_(queue) {
        }

        BenchValue pop(MoodyCamelQueueAdapter& queue) noexcept {
            BenchValue element{};
            while (!queue.try_dequeue(token_, element)) {
                spin_loop_pause();
            }
            return element;
        }
    };

    explicit MoodyCamelQueueAdapter(support::Context context)
        : moodycamel::ConcurrentQueue<BenchValue>(Capacity, context.producers, 0) {
    }

    void push(BenchValue element) noexcept {
        while (!this->try_enqueue(element)) {
            spin_loop_pause();
        }
    }

    BenchValue pop() noexcept {
        BenchValue element{};
        while (!this->try_dequeue(element)) {
            spin_loop_pause();
        }
        return element;
    }
};

template <unsigned Capacity>
struct TbbBoundedQueueAdapter {
    using value_type = BenchValue;
    static constexpr unsigned kMaxPairs = std::numeric_limits<unsigned>::max();

    tbb::concurrent_bounded_queue<BenchValue> queue;

    TbbBoundedQueueAdapter() {
        queue.set_capacity(Capacity);
    }

    void push(BenchValue element) noexcept {
        while (!queue.try_push(element)) {
            spin_loop_pause();
        }
    }

    BenchValue pop() noexcept {
        BenchValue element{};
        while (!queue.try_pop(element)) {
            spin_loop_pause();
        }
        return element;
    }
};

template <unsigned Capacity>
struct AtomicQueueB2RetryAdapter {
    using value_type = BenchValue;
    static constexpr unsigned kMaxPairs = std::numeric_limits<unsigned>::max();
    using queue_type = atomic_queue::RetryDecorator<
        atomic_queue::CapacityArgAdaptor<atomic_queue::AtomicQueueB2<BenchValue>, Capacity>>;

    queue_type queue;

    void push(BenchValue element) noexcept {
        queue.push(element);
    }

    BenchValue pop() noexcept {
        return queue.pop();
    }
};

template <unsigned Capacity>
struct AtomicQueueRetryAdapter {
    using value_type = BenchValue;
    static constexpr unsigned kMaxPairs = std::numeric_limits<unsigned>::max();
    using queue_type = atomic_queue::RetryDecorator<
        atomic_queue::AtomicQueue<BenchValue, Capacity, BenchValue{}, true, true, false, false>>;

    queue_type queue;

    void push(BenchValue element) noexcept {
        queue.push(element);
    }

    BenchValue pop() noexcept {
        return queue.pop();
    }
};

template <unsigned Capacity>
struct AtomicQueueOptimistAdapter {
    using value_type = BenchValue;
    static constexpr unsigned kMaxPairs = std::numeric_limits<unsigned>::max();
    using queue_type =
        atomic_queue::AtomicQueue<BenchValue, Capacity, BenchValue{}, true, true, false, false>;

    queue_type queue;

    void push(BenchValue element) noexcept {
        queue.push(element);
    }

    BenchValue pop() noexcept {
        return queue.pop();
    }
};

template <unsigned Capacity>
struct AtomicQueueB2OptimistAdapter {
    using value_type = BenchValue;
    static constexpr unsigned kMaxPairs = std::numeric_limits<unsigned>::max();
    using queue_type = atomic_queue::CapacityArgAdaptor<atomic_queue::AtomicQueueB2<BenchValue>, Capacity>;

    queue_type queue;

    void push(BenchValue element) noexcept {
        queue.push(element);
    }

    BenchValue pop() noexcept {
        return queue.pop();
    }
};

template <class Queue>
void throughput_producer(unsigned message_count,
                         Queue* queue,
                         std::atomic<cycles_t>* start_time,
                         Barrier* barrier) {
    barrier->wait();

    cycles_t expected = 0;
    start_time->compare_exchange_strong(expected,
                                        __rdtsc(),
                                        std::memory_order_acq_rel,
                                        std::memory_order_relaxed);

    ProducerOf<Queue> producer{*queue};
    for (unsigned n = 1, stop = message_count + 1; n <= stop; ++n) {
        producer.push(*queue, static_cast<BenchValue>(n));
    }
}

template <class Queue>
void throughput_consumer_impl(unsigned message_count,
                              Queue* queue,
                              sum_t* consumer_sum,
                              std::atomic<unsigned>* last_consumer,
                              cycles_t* end_time) {
    unsigned const stop = message_count + 1;
    sum_t sum = 0;

    ConsumerOf<Queue> consumer{*queue};
    for (;;) {
        unsigned n = consumer.pop(*queue);
        if (n == stop) {
            break;
        }
        sum += n;
    }

    cycles_t const now = __rdtsc();
    if (last_consumer->fetch_sub(1, std::memory_order_acq_rel) == 1) {
        *end_time = now;
    }

    *consumer_sum = sum;
}

template <class Queue>
void throughput_consumer(unsigned message_count,
                         Queue* queue,
                         sum_t* consumer_sum,
                         std::atomic<unsigned>* last_consumer,
                         cycles_t* end_time,
                         Barrier* barrier) {
    barrier->wait();
    throughput_consumer_impl(message_count, queue, consumer_sum, last_consumer, end_time);
}

template <class Queue>
cycles_t benchmark_throughput(HugePages& hp,
                              std::vector<unsigned> const& hw_thread_ids,
                              unsigned message_count,
                              unsigned thread_pairs,
                              bool alternative_placement,
                              sum_t* consumer_sums) {
    support::set_thread_affinity(hw_thread_ids[thread_pairs * 2 - 1]);

    auto queue = hp.create_unique_ptr<Queue>(ContextOf<Queue>{thread_pairs, thread_pairs});
    std::atomic<cycles_t> start_time{0};
    cycles_t end_time = 0;
    std::atomic<unsigned> last_consumer{thread_pairs};

    Barrier barrier;
    std::vector<std::thread> threads(thread_pairs * 2 - 1);
    unsigned cpu_index = 0;

    if (alternative_placement) {
        for (unsigned i = 0; i < thread_pairs; ++i) {
            support::set_default_thread_affinity(hw_thread_ids[cpu_index++]);
            threads[i] = std::thread(throughput_producer<Queue>,
                                     message_count,
                                     queue.get(),
                                     &start_time,
                                     &barrier);
            if (i != thread_pairs - 1) {
                support::set_default_thread_affinity(hw_thread_ids[cpu_index++]);
                threads[thread_pairs + i] = std::thread(throughput_consumer<Queue>,
                                                        message_count,
                                                        queue.get(),
                                                        consumer_sums + i,
                                                        &last_consumer,
                                                        &end_time,
                                                        &barrier);
            }
        }
    } else {
        for (unsigned i = 0; i < thread_pairs; ++i) {
            support::set_default_thread_affinity(hw_thread_ids[cpu_index++]);
            threads[i] = std::thread(throughput_producer<Queue>,
                                     message_count,
                                     queue.get(),
                                     &start_time,
                                     &barrier);
        }
        for (unsigned i = 0; i < thread_pairs - 1; ++i) {
            support::set_default_thread_affinity(hw_thread_ids[cpu_index++]);
            threads[thread_pairs + i] = std::thread(throughput_consumer<Queue>,
                                                    message_count,
                                                    queue.get(),
                                                    consumer_sums + i,
                                                    &last_consumer,
                                                    &end_time,
                                                    &barrier);
        }
    }

    barrier.release(thread_pairs * 2 - 1);
    throughput_consumer_impl(message_count,
                             queue.get(),
                             consumer_sums + (thread_pairs - 1),
                             &last_consumer,
                             &end_time);

    for (auto& thread : threads) {
        thread.join();
    }

    support::reset_thread_affinity();
    return end_time - start_time.load(std::memory_order_relaxed);
}

unsigned available_pairs(std::vector<unsigned> const& hw_thread_ids) {
    return static_cast<unsigned>(hw_thread_ids.size() / 2);
}

template <class Queue>
unsigned queue_max_pairs(std::vector<unsigned> const& hw_thread_ids,
                         BenchmarkConfig const& config) {
    unsigned const hw_limit = available_pairs(hw_thread_ids);
    unsigned const queue_limit = std::min(hw_limit, Queue::kMaxPairs);
    if (config.max_pairs == 0) {
        return queue_limit;
    }
    return std::min(config.max_pairs, queue_limit);
}

template <class Queue>
void run_throughput_benchmark(char const* name,
                              HugePages& hp,
                              std::vector<unsigned> const& hw_thread_ids,
                              BenchmarkConfig const& config) {
    unsigned const max_pairs = queue_max_pairs<Queue>(hw_thread_ids, config);
    if (max_pairs == 0 || config.min_pairs > max_pairs) {
        return;
    }

    std::vector<sum_t> consumer_sums(max_pairs);

    for (unsigned pairs = config.min_pairs; pairs <= max_pairs; ++pairs) {
        unsigned const messages_per_producer = config.throughput_messages / pairs;
        if (messages_per_producer == 0) {
            throw std::runtime_error("--messages must be at least as large as the maximum pair count");
        }

        sum_t const expected_sum =
            static_cast<sum_t>(messages_per_producer) *
            static_cast<sum_t>(messages_per_producer + 1) / 2;
        double const expected_sum_inv = 1.0 / static_cast<double>(expected_sum);

        for (bool alternative_placement : {false, true}) {
            cycles_t best_time = std::numeric_limits<cycles_t>::max();

            for (unsigned run = 0; run < config.throughput_runs; ++run) {
                cycles_t const time = benchmark_throughput<Queue>(hp,
                                                                  hw_thread_ids,
                                                                  messages_per_producer,
                                                                  pairs,
                                                                  alternative_placement,
                                                                  consumer_sums.data());
                best_time = std::min(best_time, time);
                check_huge_pages_leaks(name, hp);

                sum_t total_sum = 0;
                for (unsigned consumer_index = 0; consumer_index < pairs; ++consumer_index) {
                    sum_t const consumer_sum = consumer_sums[consumer_index];
                    total_sum += consumer_sum;

                    double const fraction = consumer_sum * expected_sum_inv;
                    if (fraction < 0.1) {
                        std::fprintf(stderr,
                                     "%s: pairs=%u: consumer %u received too few messages: %.2lf%% of expected.\n",
                                     name,
                                     pairs,
                                     consumer_index,
                                     100.0 * fraction);
                    }
                }

                sum_t const expected_total = expected_sum * pairs;
                if (sum_t const diff = total_sum - expected_total; diff != 0) {
                    std::fprintf(stderr,
                                 "%s: wrong checksum error: pairs=%u, expected_sum=%'lld, diff=%'lld.\n",
                                 name,
                                 pairs,
                                 expected_total,
                                 diff);
                }
            }

            double const seconds = to_seconds(best_time);
            unsigned const msg_per_sec =
                static_cast<unsigned>(static_cast<double>(messages_per_producer) * pairs / seconds);
            std::printf("%32s,%2u,%c: %'11u msg/sec\n",
                        name,
                        pairs,
                        alternative_placement ? 'i' : 's',
                        msg_per_sec);
        }
    }
}

template <class Queue>
void ping_pong_thread_impl(Queue* q1,
                           Queue* q2,
                           unsigned message_count,
                           cycles_t* time,
                           std::false_type) {
    cycles_t const start = __rdtsc();
    ConsumerOf<Queue> consumer_q1{*q1};
    ProducerOf<Queue> producer_q2{*q2};
    for (unsigned i = 1, received = 0; received < message_count; ++i) {
        received = consumer_q1.pop(*q1);
        producer_q2.push(*q2, static_cast<BenchValue>(i));
    }
    *time = __rdtsc() - start;
}

template <class Queue>
void ping_pong_thread_impl(Queue* q1,
                           Queue* q2,
                           unsigned message_count,
                           cycles_t* time,
                           std::true_type) {
    cycles_t const start = __rdtsc();
    ProducerOf<Queue> producer_q1{*q1};
    ConsumerOf<Queue> consumer_q2{*q2};
    for (unsigned i = 1, received = 0; received < message_count; ++i) {
        producer_q1.push(*q1, static_cast<BenchValue>(i));
        received = consumer_q2.pop(*q2);
    }
    *time = __rdtsc() - start;
}

template <class Queue>
void ping_pong_receiver(Barrier* barrier,
                        Queue* q1,
                        Queue* q2,
                        unsigned message_count,
                        cycles_t* time) {
    barrier->wait();
    ping_pong_thread_impl(q1, q2, message_count, time, std::false_type{});
}

template <class Queue>
void ping_pong_sender(Barrier* barrier,
                      Queue* q1,
                      Queue* q2,
                      unsigned message_count,
                      cycles_t* time) {
    barrier->release(1);
    ping_pong_thread_impl(q1, q2, message_count, time, std::true_type{});
}

template <class Queue>
std::array<cycles_t, 2> ping_pong_benchmark(unsigned message_count,
                                            HugePages& hp,
                                            std::array<unsigned, 2> const& cpus) {
    support::set_thread_affinity(cpus[0]);
    auto q1 = hp.create_unique_ptr<Queue>(ContextOf<Queue>{1, 1});
    auto q2 = hp.create_unique_ptr<Queue>(ContextOf<Queue>{1, 1});
    Barrier barrier;
    std::array<cycles_t, 2> times{};

    support::set_default_thread_affinity(cpus[1]);
    std::thread receiver(ping_pong_receiver<Queue>, &barrier, q1.get(), q2.get(), message_count, &times[0]);
    ping_pong_sender<Queue>(&barrier, q1.get(), q2.get(), message_count, &times[1]);
    receiver.join();

    support::reset_thread_affinity();
    return times;
}

template <class Queue>
void run_ping_pong_benchmark(char const* name,
                             HugePages& hp,
                             std::array<unsigned, 2> const& cpus,
                             BenchmarkConfig const& config) {
    std::array<cycles_t, 2> best_times{
        std::numeric_limits<cycles_t>::max(),
        std::numeric_limits<cycles_t>::max()
    };

    for (unsigned run = 0; run < config.ping_pong_runs; ++run) {
        auto const times = ping_pong_benchmark<Queue>(config.ping_pong_messages, hp, cpus);
        if (best_times[0] + best_times[1] > times[0] + times[1]) {
            best_times = times;
        }
        check_huge_pages_leaks(name, hp);
    }

    double const avg_time = to_seconds((best_times[0] + best_times[1]) / 2);
    double const round_trip_time = avg_time / config.ping_pong_messages;
    std::printf("%32s: %.9f sec/round-trip\n", name, round_trip_time);
}

bool use_huge_pages(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--huge-pages") == 0) {
            return true;
        }
    }

    char const* env = std::getenv("MPMC_BENCH_USE_HUGEPAGES");
    return env && std::strcmp(env, "0") != 0;
}

HugePages make_benchmark_pages(bool enable_huge_pages) {
    size_t constexpr MB = 1024 * 1024;
    if (!enable_huge_pages) {
        return HugePages(HugePages::PAGE_DEFAULT, 32 * MB);
    }

    try {
        return HugePages(HugePages::PAGE_1GB, 32 * MB);
    }
    catch (std::system_error const&) {
        std::fprintf(stderr,
                     "Warning: Falling back to a smaller default-page benchmark arena.\n");
        return HugePages(HugePages::PAGE_DEFAULT, 32 * MB);
    }
}

unsigned parse_unsigned(char const* arg, char const* flag) {
    char* end = nullptr;
    unsigned long value = std::strtoul(arg, &end, 10);
    if (!arg[0] || !end || *end != '\0' ||
        value > std::numeric_limits<unsigned>::max()) {
        throw std::runtime_error(std::string(flag) + " expects an unsigned integer");
    }
    return static_cast<unsigned>(value);
}

std::vector<unsigned> parse_hw_thread_ids(char const* value) {
    std::vector<unsigned> hw_thread_ids;
    char const* p = value;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == ',') {
            ++p;
        }
        if (!*p) {
            break;
        }

        char* end = nullptr;
        errno = 0;
        unsigned long const hw_thread_id = std::strtoul(p, &end, 10);
        if (end == p || errno || hw_thread_id > std::numeric_limits<unsigned>::max()) {
            throw std::runtime_error("Invalid ATOMIC_QUEUE_CPU_LIST value.");
        }

        while (*end == ' ' || *end == '\t') {
            ++end;
        }
        if (*end && *end != ',') {
            throw std::runtime_error("ATOMIC_QUEUE_CPU_LIST must be a comma-separated list of hardware thread ids.");
        }

        hw_thread_ids.push_back(static_cast<unsigned>(hw_thread_id));
        p = end;
    }

    if (hw_thread_ids.empty()) {
        throw std::runtime_error("ATOMIC_QUEUE_CPU_LIST must contain at least 2 hardware thread ids.");
    }

    return hw_thread_ids;
}

std::vector<support::CpuTopologyInfo> select_cpu_topology(
    std::vector<support::CpuTopologyInfo> const& cpu_topology,
    std::vector<unsigned> const& hw_thread_ids) {
    std::vector<support::CpuTopologyInfo> selected_cpu_topology;
    selected_cpu_topology.reserve(hw_thread_ids.size());

    for (unsigned hw_thread_id : hw_thread_ids) {
        auto const it = std::find_if(cpu_topology.begin(), cpu_topology.end(),
                                     [&](support::CpuTopologyInfo const& cpu) {
                                         return cpu.hw_thread_id == hw_thread_id;
                                     });
        if (it == cpu_topology.end()) {
            throw std::runtime_error(
                "ATOMIC_QUEUE_CPU_LIST contains a hardware thread id that is not present in /proc/cpuinfo.");
        }
        selected_cpu_topology.push_back(*it);
    }

    return selected_cpu_topology;
}

BenchmarkConfig parse_config(int argc, char** argv) {
    BenchmarkConfig config;
    config.use_huge_pages = use_huge_pages(argc, argv);

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--messages") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("--messages requires an unsigned integer");
            }
            config.throughput_messages = parse_unsigned(argv[++i], "--messages");
        }
        else if (std::strcmp(argv[i], "--throughput-runs") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("--throughput-runs requires an unsigned integer");
            }
            config.throughput_runs = parse_unsigned(argv[++i], "--throughput-runs");
        }
        else if (std::strcmp(argv[i], "--ping-pong-messages") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("--ping-pong-messages requires an unsigned integer");
            }
            config.ping_pong_messages = parse_unsigned(argv[++i], "--ping-pong-messages");
        }
        else if (std::strcmp(argv[i], "--ping-pong-runs") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("--ping-pong-runs requires an unsigned integer");
            }
            config.ping_pong_runs = parse_unsigned(argv[++i], "--ping-pong-runs");
        }
        else if (std::strcmp(argv[i], "--min-pairs") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("--min-pairs requires an unsigned integer");
            }
            config.min_pairs = parse_unsigned(argv[++i], "--min-pairs");
        }
        else if (std::strcmp(argv[i], "--max-pairs") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("--max-pairs requires an unsigned integer");
            }
            config.max_pairs = parse_unsigned(argv[++i], "--max-pairs");
        }
        else if (std::strcmp(argv[i], "--throughput-only") == 0) {
            config.run_ping_pong = false;
        }
        else if (std::strcmp(argv[i], "--ping-pong-only") == 0) {
            config.run_throughput = false;
        }
        else if (std::strcmp(argv[i], "--huge-pages") == 0) {
        }
        else {
            throw std::runtime_error(std::string("Unknown argument: ") + argv[i]);
        }
    }

    if (!config.run_throughput && !config.run_ping_pong) {
        throw std::runtime_error("At least one benchmark group must be enabled.");
    }
    if (config.min_pairs == 0) {
        throw std::runtime_error("--min-pairs must be at least 1.");
    }
    if (config.max_pairs != 0 && config.max_pairs < config.min_pairs) {
        throw std::runtime_error("--max-pairs must be greater than or equal to --min-pairs.");
    }
    if (config.throughput_runs == 0 || config.ping_pong_runs == 0) {
        throw std::runtime_error("Run counts must be at least 1.");
    }
    if (config.throughput_messages == 0 || config.ping_pong_messages == 0) {
        throw std::runtime_error("Message counts must be at least 1.");
    }

    return config;
}

template <unsigned ThroughputCapacity, unsigned PingPongCapacity>
void run_all_benchmarks(HugePages& hp,
                        std::vector<unsigned> const& hw_thread_ids,
                        BenchmarkConfig const& config) {
    if (config.run_throughput) {
        std::printf("---- Running throughput benchmarks (higher is better) ----\n");
        std::printf("---- Results are reported as queue_name,producer_consumer_pairs,placement ----\n");
        run_throughput_benchmark<Mpmc1Adapter<ThroughputCapacity>>("mpmc_1", hp, hw_thread_ids, config);
        run_throughput_benchmark<MpmcSeqAdapter<ThroughputCapacity>>("mpmc_seq", hp, hw_thread_ids, config);
        run_throughput_benchmark<MpmcSeqSplitAdapter<ThroughputCapacity>>("mpmc_seq_split", hp, hw_thread_ids, config);
        run_throughput_benchmark<JoadMpmcAdapter<ThroughputCapacity>>("jdz::MpmcQueue", hp, hw_thread_ids, config);
        run_throughput_benchmark<BoostMpmcQueueAdapter<ThroughputCapacity>>("boost::lockfree::queue", hp, hw_thread_ids, config);
        run_throughput_benchmark<MoodyCamelQueueAdapter<ThroughputCapacity>>("moodycamel::ConcurrentQueue", hp, hw_thread_ids, config);
        run_throughput_benchmark<TbbBoundedQueueAdapter<ThroughputCapacity>>("tbb::concurrent_bounded_queue", hp, hw_thread_ids, config);
        run_throughput_benchmark<AtomicQueueRetryAdapter<ThroughputCapacity>>("AtomicQueue", hp, hw_thread_ids, config);
        run_throughput_benchmark<AtomicQueueOptimistAdapter<ThroughputCapacity>>("OptimistAtomicQueue", hp, hw_thread_ids, config);
        run_throughput_benchmark<AtomicQueueB2RetryAdapter<ThroughputCapacity>>("AtomicQueueB2", hp, hw_thread_ids, config);
        run_throughput_benchmark<AtomicQueueB2OptimistAdapter<ThroughputCapacity>>("OptimistAtomicQueueB2", hp, hw_thread_ids, config);
        run_throughput_benchmark<FollyQueueReferenceAdapter<ThroughputCapacity>>(
            "folly::ProducerConsumerQueue [1p1c ref]", hp, hw_thread_ids, config);
        std::printf("\n");
    }

    if (config.run_ping_pong) {
        std::printf("---- Running ping-pong benchmarks (lower is better) ----\n");
        std::printf("---- Vendored folly only provides ProducerConsumerQueue, so it is included as a 1p1c reference ----\n");
        std::array<unsigned, 2> const cpus{hw_thread_ids[0], hw_thread_ids[1]};
        run_ping_pong_benchmark<Mpmc1Adapter<PingPongCapacity>>("mpmc_1", hp, cpus, config);
        run_ping_pong_benchmark<MpmcSeqAdapter<PingPongCapacity>>("mpmc_seq", hp, cpus, config);
        run_ping_pong_benchmark<MpmcSeqSplitAdapter<PingPongCapacity>>("mpmc_seq_split", hp, cpus, config);
        run_ping_pong_benchmark<JoadMpmcAdapter<PingPongCapacity>>("jdz::MpmcQueue", hp, cpus, config);
        run_ping_pong_benchmark<BoostMpmcQueueAdapter<PingPongCapacity>>("boost::lockfree::queue", hp, cpus, config);
        run_ping_pong_benchmark<MoodyCamelQueueAdapter<PingPongCapacity>>("moodycamel::ConcurrentQueue", hp, cpus, config);
        run_ping_pong_benchmark<TbbBoundedQueueAdapter<PingPongCapacity>>("tbb::concurrent_bounded_queue", hp, cpus, config);
        run_ping_pong_benchmark<AtomicQueueRetryAdapter<PingPongCapacity>>("AtomicQueue", hp, cpus, config);
        run_ping_pong_benchmark<AtomicQueueOptimistAdapter<PingPongCapacity>>("OptimistAtomicQueue", hp, cpus, config);
        run_ping_pong_benchmark<AtomicQueueB2RetryAdapter<PingPongCapacity>>("AtomicQueueB2", hp, cpus, config);
        run_ping_pong_benchmark<AtomicQueueB2OptimistAdapter<PingPongCapacity>>("OptimistAtomicQueueB2", hp, cpus, config);
        run_ping_pong_benchmark<FollyQueueReferenceAdapter<PingPongCapacity>>(
            "folly::ProducerConsumerQueue [1p1c ref]", hp, cpus, config);
        std::printf("\n");
    }
}

} // namespace

int main(int argc, char** argv) {
    std::setlocale(LC_NUMERIC, "");

    auto config = parse_config(argc, argv);

    auto cpu_topology = support::get_cpu_topology_info();
    if (char const* hw_thread_ids = std::getenv("ATOMIC_QUEUE_CPU_LIST")) {
        cpu_topology = select_cpu_topology(cpu_topology, parse_hw_thread_ids(hw_thread_ids));
        std::fprintf(stderr, "Using hardware threads from ATOMIC_QUEUE_CPU_LIST=%s.\n", hw_thread_ids);
    }
    if (cpu_topology.size() < 2) {
        throw std::runtime_error("A CPU with at least 2 hardware threads is required.");
    }

    auto hw_thread_ids = support::hw_thread_id(cpu_topology);
    if (available_pairs(hw_thread_ids) == 0) {
        throw std::runtime_error("At least one producer/consumer pair is required.");
    }

    if (config.use_huge_pages) {
        HugePages::warn_no_1GB_pages = advise_hugeadm_1GB;
        HugePages::warn_no_2MB_pages = advise_hugeadm_2MB;
    }
    HugePages hp = make_benchmark_pages(config.use_huge_pages);
    HugePageAllocatorBase::hp = &hp;

    unsigned const effective_max_pairs =
        config.max_pairs == 0 ? available_pairs(hw_thread_ids)
                              : std::min(config.max_pairs, available_pairs(hw_thread_ids));

    std::printf("---- Selected hardware threads: %zu (%u producer/consumer pairs max) ----\n",
                hw_thread_ids.size(),
                effective_max_pairs);
    std::printf("---- Throughput messages total: %u, runs: %u ----\n",
                config.throughput_messages,
                config.throughput_runs);
    std::printf("---- Ping-pong messages: %u, runs: %u ----\n",
                config.ping_pong_messages,
                config.ping_pong_runs);

    constexpr unsigned throughput_capacity = 1u << 16;
    constexpr unsigned ping_pong_capacity = 8;
    run_all_benchmarks<throughput_capacity, ping_pong_capacity>(hp, hw_thread_ids, config);

    return 0;
}
