#include <boost/lockfree/spsc_queue.hpp>
#include <folly/ProducerConsumerQueue.h>
#include <oneapi/tbb/concurrent_queue.h>
#include <readerwriterqueue/readerwriterqueue.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <x86intrin.h>

#include <atomic_queue/atomic_queue.h>

#include "../include/joad_q.h"
#include "../include/rigtorp.h"
#include "../include/spsc.h"
#include "../include/spsc_1.h"
#include "../include/spsc_2.h"
#include "../include/spsc_3.h"
#include "../include/spsc_4.h"
#include "../include/spsc_5.h"
#include "../include/spsc_final.h"
#include "../include/test.h"
#include "memory.h"
#include "queue_measure.h"
#include "runtime.h"
#include "sync.h"

namespace {
    using BenchValue = unsigned;
    using measure::BenchMode;
    using measure::cycles_t;
    using measure::sum_t;
    using support::Barrier;
    using support::ContextOf;
    using support::HugePageAllocatorBase;
    using support::HugePages;
    using support::spin_loop_pause;

    struct BenchConfig {
        std::array<unsigned, 2> cpus{};
        BenchMode mode{BenchMode::Blocking};
    };

    double const TSC_TO_SECONDS = 1e-9 / support::cpu_base_frequency();

    template <class T>
    double to_seconds(T cycles) {
        return cycles * TSC_TO_SECONDS;
    }

    char const* bench_mode_name(BenchMode mode) noexcept {
        switch (mode) {
            case BenchMode::Blocking:
                return "blocking";
            case BenchMode::NonBlocking:
                return "nonblocking";
        }
        return "unknown";
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

    template <class T, unsigned Capacity>
    struct SpscQueueAdapter {
        using value_type = T;
        LockFreeQueue<T, Capacity> queue;

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

    template <class T, unsigned Capacity>
    struct SpscQueue1Adapter {
        using value_type = T;
        LockFreeQueue1<T, Capacity> queue;
 
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

    template <class T, unsigned Capacity>
    struct SpscQueue2Adapter {
        using value_type = T;
        LockFreeQueueStage2<T, Capacity> queue;
 
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

    template <class T, unsigned Capacity>
    struct SpscQueue3Adapter {
        using value_type = T;
        LockFreeQueueStage3<T, Capacity> queue;
 
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

    template <class T, unsigned Capacity>
    struct SpscQueue5Adapter {
        using value_type = T;
        LockFreeQueueStage5<T, Capacity> queue;

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

    template <class T, unsigned Capacity>
    struct SpscFinalQueueAdapter {
        using value_type = T;
        LockFreeQueue2<T, Capacity> queue;

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

    template <class T, unsigned Capacity>
    struct DroQueueAdapter {
        using value_type = T;
        static constexpr bool supports_blocking = true;
        static constexpr bool supports_nonblocking = true;
        dro::SPSCQueue<T> queue;

        DroQueueAdapter()
            : queue(Capacity - 1) {
        }

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

    template <class T, unsigned Capacity>
    struct JoadQueueAdapter {
        using value_type = T;
        static constexpr bool supports_blocking = true;
        static constexpr bool supports_nonblocking = true;
        jdz::SpscQueue<T, Capacity> queue;

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

    template <class T, unsigned Capacity>
    struct RigtorpQueueAdapter {
        using value_type = T;
        static constexpr bool supports_blocking = true;
        static constexpr bool supports_nonblocking = true;
        rigtorp::SPSCQueue<T> queue;

        RigtorpQueueAdapter()
            : queue(Capacity) {
        }

        void push(T element) noexcept {
            queue.push(element);
        }

        bool try_push(T element) noexcept {
            return queue.try_push(element);
        }

        T pop() noexcept {
            T* front = nullptr;
            while ((front = queue.front()) == nullptr) {
            }
            T element = *front;
            queue.pop();
            return element;
        }

        bool try_pop(T& element) noexcept {
            T* front = queue.front();
            if (front == nullptr) {
                return false;
            }
            element = *front;
            queue.pop();
            return true;
        }
    };

    template <class T, unsigned Capacity>
    struct BoostQueueAdapter {
        using value_type = T;
        static constexpr bool supports_blocking = false;
        static constexpr bool supports_nonblocking = true;
        boost::lockfree::spsc_queue<T, boost::lockfree::capacity<Capacity>> queue;

        bool try_push(T element) noexcept {
            return queue.push(element);
        }

        bool try_pop(T& element) noexcept {
            return queue.pop(element);
        }
    };

    template <class T, unsigned Capacity>
    struct FollyQueueAdapter {
        using value_type = T;
        static constexpr bool supports_blocking = false;
        static constexpr bool supports_nonblocking = true;
        folly::ProducerConsumerQueue<T> queue;

        FollyQueueAdapter()
            : queue(static_cast<uint32_t>(Capacity)) {
        }

        bool try_push(T element) noexcept {
            return queue.write(element);
        }

        bool try_pop(T& element) noexcept {
            return queue.read(element);
        }
    };

    template <class T>
    struct TbbQueueAdapter {
        using value_type = T;
        static constexpr bool supports_blocking = false;
        static constexpr bool supports_nonblocking = true;
        tbb::concurrent_queue<T> queue;

        bool try_push(T element) noexcept {
            queue.push(element);
            return true;
        }

        bool try_pop(T& element) noexcept {
            return queue.try_pop(element);
        }
    };

    template <class T, unsigned Capacity>
    struct MoodyCamelReaderWriterQueueAdapter {
        using value_type = T;
        static constexpr bool supports_blocking = false;
        static constexpr bool supports_nonblocking = true;
        moodycamel::ReaderWriterQueue<T> queue;

        MoodyCamelReaderWriterQueueAdapter()
            : queue(Capacity) {
        }

        bool try_push(T element) noexcept {
            return queue.try_enqueue(element);
        }

        bool try_pop(T& element) noexcept {
            return queue.try_dequeue(element);
        }
    };

    template <unsigned Capacity>
    struct AtomicQueue2Adapter {
        using value_type = BenchValue;
        static constexpr bool supports_blocking = true;
        static constexpr bool supports_nonblocking = true;
        using queue_type = atomic_queue::AtomicQueue2<BenchValue, Capacity, true, true, false, true>;
        queue_type queue;

        void push(BenchValue element) noexcept {
            queue.push(element);
        }

        bool try_push(BenchValue element) noexcept {
            return queue.try_push(element);
        }

        BenchValue pop() noexcept {
            return queue.pop();
        }

        bool try_pop(BenchValue& element) noexcept {
            return queue.try_pop(element);
        }
    };

    template <BenchMode Mode, class Queue>
    void run_throughput_benchmark(char const* name,
                                  HugePages& hp,
                                  std::array<unsigned, 2> const& cpus) {
        int constexpr runs = 3;
        unsigned constexpr message_count = 10000000;
        sum_t expected_sum = (message_count + 1) / 2.0 * message_count;
        cycles_t best_time = std::numeric_limits<cycles_t>::max();
        sum_t consumer_sum = 0;

        for (int run = 0; run < runs; ++run) {
            sum_t run_consumer_sum = 0;
            cycles_t run_time =
                measure::benchmark_throughput<Mode, Queue>(hp, cpus, message_count, &run_consumer_sum);
            best_time = std::min(best_time, run_time);
            consumer_sum = run_consumer_sum;
            check_huge_pages_leaks(name, hp);

            if (auto diff = run_consumer_sum - expected_sum) {
                std::fprintf(stderr,
                             "%s: wrong checksum error: expected_sum: %'lld, diff: %'lld.\n",
                             name,
                             expected_sum,
                             diff);
            }
        }

        if (consumer_sum != expected_sum) {
            std::fprintf(stderr,
                         "%s: final checksum mismatch: expected %'lld got %'lld.\n",
                         name,
                         expected_sum,
                         consumer_sum);
        }

        double seconds = to_seconds(best_time);
        unsigned msg_per_sec = static_cast<unsigned>(message_count / seconds);
        std::printf("%32s: %'11u msg/sec\n", name, msg_per_sec);
    }

    template <BenchMode RequestedMode, class Queue>
    void run_third_party_throughput_benchmark(char const* name,
                                              HugePages& hp,
                                              std::array<unsigned, 2> const& cpus) {
        if constexpr (RequestedMode == BenchMode::Blocking) {
            if constexpr (Queue::supports_blocking) {
                run_throughput_benchmark<BenchMode::Blocking, Queue>(name, hp, cpus);
            } else {
                run_throughput_benchmark<BenchMode::NonBlocking, Queue>(name, hp, cpus);
            }
        } else {
            if constexpr (Queue::supports_nonblocking) {
                run_throughput_benchmark<BenchMode::NonBlocking, Queue>(name, hp, cpus);
            } else {
                run_throughput_benchmark<BenchMode::Blocking, Queue>(name, hp, cpus);
            }
        }
    }

    template <BenchMode Mode, class Queue>
    void ping_pong_thread_impl(Queue* q1, Queue* q2, unsigned message_count, cycles_t* time, std::false_type) {
        cycles_t start = __rdtsc();
        for (unsigned i = 1, received = 0; received < message_count; ++i) {
            received = measure::queue_pop<Mode>(*q1);
            measure::queue_push<Mode>(*q2, static_cast<typename Queue::value_type>(i));
        }
        *time = __rdtsc() - start;
    }

    template <BenchMode Mode, class Queue>
    void ping_pong_thread_impl(Queue* q1, Queue* q2, unsigned message_count, cycles_t* time, std::true_type) {
        cycles_t start = __rdtsc();
        for (unsigned i = 1, received = 0; received < message_count; ++i) {
            measure::queue_push<Mode>(*q1, static_cast<typename Queue::value_type>(i));
            received = measure::queue_pop<Mode>(*q2);
        }
        *time = __rdtsc() - start;
    }

    template <BenchMode Mode, class Queue>
    void ping_pong_receiver(Barrier* barrier, Queue* q1, Queue* q2, unsigned message_count, cycles_t* time) {
        barrier->wait();
        ping_pong_thread_impl<Mode>(q1, q2, message_count, time, std::false_type{});
    }

    template <BenchMode Mode, class Queue>
    void ping_pong_sender(Barrier* barrier, Queue* q1, Queue* q2, unsigned message_count, cycles_t* time) {
        barrier->release(1);
        ping_pong_thread_impl<Mode>(q1, q2, message_count, time, std::true_type{});
    }

    template <BenchMode Mode, class Queue>
    std::array<cycles_t, 2> ping_pong_benchmark(unsigned message_count,
                                                HugePages& hp,
                                                std::array<unsigned, 2> const& cpus) {
        support::set_thread_affinity(cpus[0]);
        auto q1 = hp.create_unique_ptr<Queue>(ContextOf<Queue>{1, 1});
        auto q2 = hp.create_unique_ptr<Queue>(ContextOf<Queue>{1, 1});
        Barrier barrier;
        std::array<cycles_t, 2> times{};

        support::set_default_thread_affinity(cpus[1]);
        std::thread receiver(ping_pong_receiver<Mode, Queue>, &barrier, q1.get(), q2.get(), message_count, &times[0]);
        ping_pong_sender<Mode, Queue>(&barrier, q1.get(), q2.get(), message_count, &times[1]);
        receiver.join();

        support::reset_thread_affinity();
        return times;
    }

    template <BenchMode Mode, class Queue>
    void run_ping_pong_benchmark(char const* name,
                                 HugePages& hp,
                                 std::array<unsigned, 2> const& cpus) {
        int constexpr runs = 10;
        unsigned constexpr message_count = 100000;
        std::array<cycles_t, 2> best_times{
            std::numeric_limits<cycles_t>::max(),
            std::numeric_limits<cycles_t>::max()
        };

        for (int run = 0; run < runs; ++run) {
            auto times = ping_pong_benchmark<Mode, Queue>(message_count, hp, cpus);
            if (best_times[0] + best_times[1] > times[0] + times[1]) {
                best_times = times;
            }
            check_huge_pages_leaks(name, hp);
        }

        double avg_time = to_seconds((best_times[0] + best_times[1]) / 2);
        double round_trip_time = avg_time / message_count;
        std::printf("%32s: %.9f sec/round-trip\n", name, round_trip_time);
    }

    template <BenchMode RequestedMode, class Queue>
    void run_third_party_ping_pong_benchmark(char const* name,
                                             HugePages& hp,
                                             std::array<unsigned, 2> const& cpus) {
        if constexpr (RequestedMode == BenchMode::Blocking) {
            if constexpr (Queue::supports_blocking) {
                run_ping_pong_benchmark<BenchMode::Blocking, Queue>(name, hp, cpus);
            } else {
                run_ping_pong_benchmark<BenchMode::NonBlocking, Queue>(name, hp, cpus);
            }
        } else {
            if constexpr (Queue::supports_nonblocking) {
                run_ping_pong_benchmark<BenchMode::NonBlocking, Queue>(name, hp, cpus);
            } else {
                run_ping_pong_benchmark<BenchMode::Blocking, Queue>(name, hp, cpus);
            }
        }
    }

    bool use_huge_pages(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--huge-pages") == 0) {
                return true;
            }
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

    BenchMode parse_mode(char const* arg) {
        if (std::strcmp(arg, "blocking") == 0) {
            return BenchMode::Blocking;
        }
        if (std::strcmp(arg, "nonblocking") == 0) {
            return BenchMode::NonBlocking;
        }
        throw std::runtime_error("--mode expects 'blocking' or 'nonblocking'");
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
            else if (std::strcmp(argv[i], "--mode") == 0) {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--mode requires one of: blocking, nonblocking");
                }
                config.mode = parse_mode(argv[i + 1]);
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

    template <BenchMode Mode>
    void run_all_benchmarks(HugePages& hp, std::array<unsigned, 2> const& cpus) {
        unsigned constexpr throughput_capacity = 1 << 16;
        unsigned constexpr ping_pong_capacity = 8;

        std::printf("---- Running throughput benchmarks (higher is better) ----\n");
        run_throughput_benchmark<Mode, SpscQueueAdapter<BenchValue, throughput_capacity>>("spsc", hp, cpus);
        run_throughput_benchmark<Mode, SpscQueue1Adapter<BenchValue, throughput_capacity>>("spsc_1", hp, cpus);
        run_throughput_benchmark<Mode, SpscQueue2Adapter<BenchValue, throughput_capacity>>("spsc_2", hp, cpus);
        run_throughput_benchmark<Mode, SpscQueue3Adapter<BenchValue, throughput_capacity>>("spsc_3", hp, cpus);
        run_throughput_benchmark<Mode, SpscQueue4Adapter<BenchValue, throughput_capacity>>("spsc_4", hp, cpus);
        run_throughput_benchmark<Mode, SpscQueue5Adapter<BenchValue, throughput_capacity>>("spsc_5", hp, cpus);
        run_throughput_benchmark<Mode, SpscFinalQueueAdapter<BenchValue, throughput_capacity>>("spsc_final", hp, cpus);
        run_third_party_throughput_benchmark<Mode, DroQueueAdapter<BenchValue, throughput_capacity>>("dro::SPSCQueue", hp, cpus);
        run_third_party_throughput_benchmark<Mode, JoadQueueAdapter<BenchValue, throughput_capacity>>("jdz::SpscQueue", hp, cpus);
        run_third_party_throughput_benchmark<Mode, RigtorpQueueAdapter<BenchValue, throughput_capacity>>(
            "rigtorp::SPSCQueue", hp, cpus);
        run_third_party_throughput_benchmark<Mode, BoostQueueAdapter<BenchValue, throughput_capacity>>(
            "boost::lockfree::spsc_queue", hp, cpus);
        run_third_party_throughput_benchmark<Mode, FollyQueueAdapter<BenchValue, throughput_capacity>>(
            "folly::ProducerConsumerQueue", hp, cpus);
        run_third_party_throughput_benchmark<Mode, TbbQueueAdapter<BenchValue>>("tbb::concurrent_queue", hp, cpus);
        run_third_party_throughput_benchmark<Mode, MoodyCamelReaderWriterQueueAdapter<BenchValue, throughput_capacity>>(
            "moodycamel::ReaderWriterQueue", hp, cpus);
        run_third_party_throughput_benchmark<Mode, AtomicQueue2Adapter<throughput_capacity>>("AtomicQueue2", hp, cpus);
        std::printf("\n");

        std::printf("---- Running ping-pong benchmarks (lower is better) ----\n");
        run_ping_pong_benchmark<Mode, SpscQueueAdapter<BenchValue, ping_pong_capacity>>("spsc", hp, cpus);
        run_ping_pong_benchmark<Mode, SpscQueue1Adapter<BenchValue, ping_pong_capacity>>("spsc_1", hp, cpus);
        run_ping_pong_benchmark<Mode, SpscQueue2Adapter<BenchValue, ping_pong_capacity>>("spsc_2", hp, cpus);
        run_ping_pong_benchmark<Mode, SpscQueue3Adapter<BenchValue, ping_pong_capacity>>("spsc_3", hp, cpus);
        run_ping_pong_benchmark<Mode, SpscQueue4Adapter<BenchValue, ping_pong_capacity>>("spsc_4", hp, cpus);
        run_ping_pong_benchmark<Mode, SpscQueue5Adapter<BenchValue, ping_pong_capacity>>("spsc_5", hp, cpus);
        run_ping_pong_benchmark<Mode, SpscFinalQueueAdapter<BenchValue, ping_pong_capacity>>("spsc_final", hp, cpus);
        run_third_party_ping_pong_benchmark<Mode, DroQueueAdapter<BenchValue, ping_pong_capacity>>("dro::SPSCQueue", hp, cpus);
        run_third_party_ping_pong_benchmark<Mode, JoadQueueAdapter<BenchValue, ping_pong_capacity>>("jdz::SpscQueue", hp, cpus);
        run_third_party_ping_pong_benchmark<Mode, RigtorpQueueAdapter<BenchValue, ping_pong_capacity>>(
            "rigtorp::SPSCQueue", hp, cpus);
        run_third_party_ping_pong_benchmark<Mode, BoostQueueAdapter<BenchValue, ping_pong_capacity>>(
            "boost::lockfree::spsc_queue", hp, cpus);
        run_third_party_ping_pong_benchmark<Mode, FollyQueueAdapter<BenchValue, ping_pong_capacity>>(
            "folly::ProducerConsumerQueue", hp, cpus);
        run_third_party_ping_pong_benchmark<Mode, TbbQueueAdapter<BenchValue>>("tbb::concurrent_queue", hp, cpus);
        run_third_party_ping_pong_benchmark<Mode, MoodyCamelReaderWriterQueueAdapter<BenchValue, ping_pong_capacity>>(
            "moodycamel::ReaderWriterQueue", hp, cpus);
        run_third_party_ping_pong_benchmark<Mode, AtomicQueue2Adapter<ping_pong_capacity>>("AtomicQueue2", hp, cpus);
        std::printf("\n");
    }
}

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

    std::printf("---- Thread pinning enabled: producer/sender cpu=%u, consumer/receiver cpu=%u ----\n",
                config.cpus[0],
                config.cpus[1]);
    std::printf("---- Queue API mode: %s ----\n", bench_mode_name(config.mode));
    std::printf("---- Third-party queues follow --mode when supported, otherwise fall back to their native API ----\n");
    if (config.mode == BenchMode::Blocking) {
        run_all_benchmarks<BenchMode::Blocking>(hp, config.cpus);
    }
    else {
        run_all_benchmarks<BenchMode::NonBlocking>(hp, config.cpus);
    }

    return 0;
}
