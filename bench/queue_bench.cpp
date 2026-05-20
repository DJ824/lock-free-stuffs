#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <memory>
#include <pthread.h>
#include <sched.h>
#include <cstdlib>
#include <cstdio>
#include <boost/lockfree/spsc_queue.hpp>
#include <atomic_queue/atomic_queue.h>
#include <folly/ProducerConsumerQueue.h>
#include <oneapi/tbb/concurrent_queue.h>
#include <readerwriterqueue/readerwriterqueue.h>

#include "../include/joad_q.h"
#include "../include/spsc.h"
#include "../include/spsc_final.h"
#include "../include/test.h"

enum class BenchBackend {
    Raw,
    Raw2,
    Dro,
    Joad,
    Boost,
    Folly,
    Tbb,
    MoodyCamel,
    Atomic2
};

const char* backend_name(BenchBackend backend) {
    switch (backend) {
    case BenchBackend::Raw:
        return "raw";
    case BenchBackend::Raw2:
        return "raw2";
    case BenchBackend::Dro:
        return "dro";
    case BenchBackend::Joad:
        return "joad";
    case BenchBackend::Boost:
        return "boost";
    case BenchBackend::Folly:
        return "folly";
    case BenchBackend::Tbb:
        return "tbb";
    case BenchBackend::MoodyCamel:
        return "moodycamel";
    case BenchBackend::Atomic2:
        return "atomic2";
    }
    return "unknown";
}

void pinThread(int cpu) {
    if (cpu < 0) {
        return;
    }
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    const int rc = ::pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "pthread_setaffinity_np: " << std::strerror(rc) << " (rc=" << rc << ")\n";
        std::exit(1);
    }
}

int main(int argc, char* argv[]) {
    using BenchValue = uint64_t;

    int producer_cpu = -1;
    int consumer_cpu = -1;

    if (argc >= 3) {
        producer_cpu = std::stoi(argv[1]);
        consumer_cpu = std::stoi(argv[2]);
        std::cout << "Pinning producer to CPU " << producer_cpu
            << " and consumer to CPU " << consumer_cpu << std::endl;
    }

    constexpr size_t QUEUE_SIZE = 8192;
    constexpr size_t NUM_ITERATIONS = 10000000; // 10M
    constexpr int NUM_WARMUP_RUNS = 3;
    constexpr int NUM_RUNS = 5;
    constexpr size_t PAYLOAD_BYTES = sizeof(BenchValue);

    std::cout << "Queue capacity: " << (QUEUE_SIZE - 1) << " elements" << std::endl;
    std::cout << "Operations per test: " << NUM_ITERATIONS << std::endl;
    std::cout << "Warmup runs: " << NUM_WARMUP_RUNS << std::endl;
    std::cout << "Number of test runs: " << NUM_RUNS << std::endl << std::endl;

    std::cout << "Single Producer, Single Consumer Throughput Test" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    const std::array<BenchBackend, 9> backends = {
        BenchBackend::Raw, BenchBackend::Raw2, BenchBackend::Dro, BenchBackend::Joad, BenchBackend::Boost,
        BenchBackend::Folly, BenchBackend::Tbb, BenchBackend::MoodyCamel, BenchBackend::Atomic2
    };

    auto run_throughput = [&](int run, bool report_result = true) {
        auto queue = std::make_unique<LockFreeQueue<BenchValue, QUEUE_SIZE>>();

        auto consumer = std::thread([&] {
            if (consumer_cpu >= 0) {
                pinThread(consumer_cpu);
            }

            BenchValue expected_value = 0;
            for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
                BenchValue result = 0;
                queue->pop(result);
                if (result != expected_value) {
                    std::cerr << "error, expected " << expected_value << " but got "
                              << result << std::endl;
                    exit(1);
                }
                expected_value++;
            }
        });

        if (producer_cpu >= 0) {
            pinThread(producer_cpu);
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
            queue->emplace(static_cast<BenchValue>(i));
        }

        consumer.join();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time)
            .count();

        double throughput_ops_per_ms = (NUM_ITERATIONS * 1000000.0) / duration_ns;
        double latency_ns_per_op = static_cast<double>(duration_ns) / NUM_ITERATIONS;

        if (report_result) {
            std::cout << "    Run " << (run + 1) << ": "
                << std::fixed << std::setprecision(2)
                << throughput_ops_per_ms << " ops/ms, "
                << latency_ns_per_op << " ns/op" << std::endl;
        }
    };

    auto run_throughput_dro = [&](int run, bool report_result = true) {
        dro::SPSCQueue<BenchValue> queue(QUEUE_SIZE - 1);

        auto consumer = std::thread([&] {
            if (consumer_cpu >= 0) {
                pinThread(consumer_cpu);
            }

            BenchValue expected_value = 0;
            for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
                BenchValue result = 0;
                queue.pop(result);
                if (result != expected_value) {
                    std::cerr << "error, expected " << expected_value << " but got "
                              << result << std::endl;
                    exit(1);
                }
                expected_value++;
            }
        });

        if (producer_cpu >= 0) {
            pinThread(producer_cpu);
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
            queue.emplace(static_cast<BenchValue>(i));
        }

        consumer.join();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time)
            .count();

        double throughput_ops_per_ms = (NUM_ITERATIONS * 1000000.0) / duration_ns;
        double latency_ns_per_op = static_cast<double>(duration_ns) / NUM_ITERATIONS;

        if (report_result) {
            std::cout << "    Run " << (run + 1) << ": "
                << std::fixed << std::setprecision(2)
                << throughput_ops_per_ms << " ops/ms, "
                << latency_ns_per_op << " ns/op" << std::endl;
        }
    };

    auto run_throughput_raw2 = [&](int run, bool report_result = true) {
        LockFreeQueue2<BenchValue, QUEUE_SIZE> queue;

        auto consumer = std::thread([&] {
            if (consumer_cpu >= 0) {
                pinThread(consumer_cpu);
            }

            BenchValue expected_value = 0;
            for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
                BenchValue result = 0;
                queue.pop(result);
                if (result != expected_value) {
                    std::cerr << "error, expected " << expected_value << " but got "
                              << result << std::endl;
                    exit(1);
                }
                expected_value++;
            }
        });

        if (producer_cpu >= 0) {
            pinThread(producer_cpu);
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
            queue.emplace(static_cast<BenchValue>(i));
        }

        consumer.join();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time)
            .count();

        double throughput_ops_per_ms = (NUM_ITERATIONS * 1000000.0) / duration_ns;
        double latency_ns_per_op = static_cast<double>(duration_ns) / NUM_ITERATIONS;

        if (report_result) {
            std::cout << "    Run " << (run + 1) << ": "
                << std::fixed << std::setprecision(2)
                << throughput_ops_per_ms << " ops/ms, "
                << latency_ns_per_op << " ns/op" << std::endl;
        }
    };

    auto run_throughput_joad = [&](int run, bool report_result = true) {
        jdz::SpscQueue<BenchValue, QUEUE_SIZE> queue;

        auto consumer = std::thread([&] {
            if (consumer_cpu >= 0) {
                pinThread(consumer_cpu);
            }

            BenchValue expected_value = 0;
            for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
                BenchValue result = 0;
                queue.pop(result);
                if (result != expected_value) {
                    std::cerr << "error, expected " << expected_value << " but got "
                              << result << std::endl;
                    exit(1);
                }
                expected_value++;
            }
        });

        if (producer_cpu >= 0) {
            pinThread(producer_cpu);
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
            queue.emplace(static_cast<BenchValue>(i));
        }

        consumer.join();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time)
            .count();

        double throughput_ops_per_ms = (NUM_ITERATIONS * 1000000.0) / duration_ns;
        double latency_ns_per_op = static_cast<double>(duration_ns) / NUM_ITERATIONS;

        if (report_result) {
            std::cout << "    Run " << (run + 1) << ": "
                << std::fixed << std::setprecision(2)
                << throughput_ops_per_ms << " ops/ms, "
                << latency_ns_per_op << " ns/op" << std::endl;
        }
    };

    auto run_throughput_boost = [&](int run, bool report_result = true) {
        boost::lockfree::spsc_queue<BenchValue, boost::lockfree::capacity<QUEUE_SIZE>> queue;

        auto consumer = std::thread([&] {
            if (consumer_cpu >= 0) {
                pinThread(consumer_cpu);
            }

            BenchValue expected_value = 0;
            for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
                BenchValue result = 0;
                while (!queue.pop(result)) {
                }
                if (result != expected_value) {
                    std::cerr << "error, expected " << expected_value << " but got "
                              << result << std::endl;
                    exit(1);
                }
                expected_value++;
            }
        });

        if (producer_cpu >= 0) {
            pinThread(producer_cpu);
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
            while (!queue.push(static_cast<BenchValue>(i))) {
            }
        }

        consumer.join();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time)
            .count();

        double throughput_ops_per_ms = (NUM_ITERATIONS * 1000000.0) / duration_ns;
        double latency_ns_per_op = static_cast<double>(duration_ns) / NUM_ITERATIONS;

        if (report_result) {
            std::cout << "    Run " << (run + 1) << ": "
                << std::fixed << std::setprecision(2)
                << throughput_ops_per_ms << " ops/ms, "
                << latency_ns_per_op << " ns/op" << std::endl;
        }
    };

    auto run_throughput_folly = [&](int run, bool report_result = true) {
        folly::ProducerConsumerQueue<BenchValue> queue(static_cast<uint32_t>(QUEUE_SIZE));

        auto consumer = std::thread([&] {
            if (consumer_cpu >= 0) {
                pinThread(consumer_cpu);
            }

            BenchValue expected_value = 0;
            for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
                BenchValue result = 0;
                while (!queue.read(result)) {
                }
                if (result != expected_value) {
                    std::cerr << "error, expected " << expected_value << " but got "
                              << result << std::endl;
                    exit(1);
                }
                expected_value++;
            }
        });

        if (producer_cpu >= 0) {
            pinThread(producer_cpu);
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
            while (!queue.write(static_cast<BenchValue>(i))) {
            }
        }

        consumer.join();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time)
            .count();

        double throughput_ops_per_ms = (NUM_ITERATIONS * 1000000.0) / duration_ns;
        double latency_ns_per_op = static_cast<double>(duration_ns) / NUM_ITERATIONS;

        if (report_result) {
            std::cout << "    Run " << (run + 1) << ": "
                << std::fixed << std::setprecision(2)
                << throughput_ops_per_ms << " ops/ms, "
                << latency_ns_per_op << " ns/op" << std::endl;
        }
    };

    auto run_throughput_tbb = [&](int run, bool report_result = true) {
        tbb::concurrent_queue<BenchValue> queue;

        auto consumer = std::thread([&] {
            if (consumer_cpu >= 0) {
                pinThread(consumer_cpu);
            }

            BenchValue expected_value = 0;
            for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
                BenchValue result = 0;
                while (!queue.try_pop(result)) {
                }
                if (result != expected_value) {
                    std::cerr << "error, expected " << expected_value << " but got "
                              << result << std::endl;
                    exit(1);
                }
                expected_value++;
            }
        });

        if (producer_cpu >= 0) {
            pinThread(producer_cpu);
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
            queue.push(static_cast<BenchValue>(i));
        }

        consumer.join();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time)
            .count();

        double throughput_ops_per_ms = (NUM_ITERATIONS * 1000000.0) / duration_ns;
        double latency_ns_per_op = static_cast<double>(duration_ns) / NUM_ITERATIONS;

        if (report_result) {
            std::cout << "    Run " << (run + 1) << ": "
                << std::fixed << std::setprecision(2)
                << throughput_ops_per_ms << " ops/ms, "
                << latency_ns_per_op << " ns/op" << std::endl;
        }
    };

    auto run_throughput_moodycamel = [&](int run, bool report_result = true) {
        moodycamel::ReaderWriterQueue<BenchValue> queue(QUEUE_SIZE);

        auto consumer = std::thread([&] {
            if (consumer_cpu >= 0) {
                pinThread(consumer_cpu);
            }

            BenchValue expected_value = 0;
            for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
                BenchValue result = 0;
                while (!queue.try_dequeue(result)) {
                }
                if (result != expected_value) {
                    std::cerr << "error, expected " << expected_value << " but got "
                              << result << std::endl;
                    exit(1);
                }
                expected_value++;
            }
        });

        if (producer_cpu >= 0) {
            pinThread(producer_cpu);
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
            while (!queue.try_enqueue(static_cast<BenchValue>(i))) {
            }
        }

        consumer.join();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time)
            .count();

        double throughput_ops_per_ms = (NUM_ITERATIONS * 1000000.0) / duration_ns;
        double latency_ns_per_op = static_cast<double>(duration_ns) / NUM_ITERATIONS;

        if (report_result) {
            std::cout << "    Run " << (run + 1) << ": "
                << std::fixed << std::setprecision(2)
                << throughput_ops_per_ms << " ops/ms, "
                << latency_ns_per_op << " ns/op" << std::endl;
        }
    };

    auto run_throughput_atomic2 = [&](int run, bool report_result = true) {
        atomic_queue::AtomicQueue2<BenchValue, QUEUE_SIZE, true, true, false, true> queue;

        auto consumer = std::thread([&] {
            if (consumer_cpu >= 0) {
                pinThread(consumer_cpu);
            }

            BenchValue expected_value = 0;
            for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
                const BenchValue result = queue.pop();
                if (result != expected_value) {
                    std::cerr << "error, expected " << expected_value << " but got "
                              << result << std::endl;
                    exit(1);
                }
                expected_value++;
            }
        });

        if (producer_cpu >= 0) {
            pinThread(producer_cpu);
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
            queue.push(static_cast<BenchValue>(i));
        }

        consumer.join();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time)
            .count();

        double throughput_ops_per_ms = (NUM_ITERATIONS * 1000000.0) / duration_ns;
        double latency_ns_per_op = static_cast<double>(duration_ns) / NUM_ITERATIONS;

        if (report_result) {
            std::cout << "    Run " << (run + 1) << ": "
                << std::fixed << std::setprecision(2)
                << throughput_ops_per_ms << " ops/ms, "
                << latency_ns_per_op << " ns/op" << std::endl;
        }
    };

    auto run_throughput_for_backend = [&](BenchBackend backend, int run, bool report_result = true) {
        switch (backend) {
        case BenchBackend::Raw:
            run_throughput(run, report_result);
            break;
        case BenchBackend::Raw2:
            run_throughput_raw2(run, report_result);
            break;
        case BenchBackend::Dro:
            run_throughput_dro(run, report_result);
            break;
        case BenchBackend::Joad:
            run_throughput_joad(run, report_result);
            break;
        case BenchBackend::Boost:
            run_throughput_boost(run, report_result);
            break;
        case BenchBackend::Folly:
            run_throughput_folly(run, report_result);
            break;
        case BenchBackend::Tbb:
            run_throughput_tbb(run, report_result);
            break;
        case BenchBackend::MoodyCamel:
            run_throughput_moodycamel(run, report_result);
            break;
        case BenchBackend::Atomic2:
            run_throughput_atomic2(run, report_result);
            break;
        }
    };

    std::cout << "Payload: " << PAYLOAD_BYTES << " bytes (uint64_t)" << std::endl;
    for (BenchBackend backend : backends) {
        std::cout << "  Backend: " << backend_name(backend) << std::endl;
        for (int run = 0; run < NUM_WARMUP_RUNS; ++run) {
            run_throughput_for_backend(backend, run, false);
        }
        for (int run = 0; run < NUM_RUNS; ++run) {
            run_throughput_for_backend(backend, run);
        }
        std::cout << std::endl;
    }

    std::cout << "Round-Trip Latency Test" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    auto run_rtt = [&](int run, bool report_result = true) {
        auto ping_queue = std::make_unique<LockFreeQueue<BenchValue, QUEUE_SIZE>>();
        auto pong_queue = std::make_unique<LockFreeQueue<BenchValue, QUEUE_SIZE>>();

        auto worker = std::thread([&] {
            if (consumer_cpu >= 0) {
                pinThread(consumer_cpu);
            }

            for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
                BenchValue req = 0;
                ping_queue->pop(req);
                pong_queue->emplace(req);
            }
        });

        if (producer_cpu >= 0) {
            pinThread(producer_cpu);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
            ping_queue->emplace(static_cast<BenchValue>(i));

            BenchValue resp = 0;
            pong_queue->pop(resp);
            if (resp != static_cast<BenchValue>(i)) {
                std::cerr << "Error: Expected " << i << " but got " << resp << std::endl;
                exit(1);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time)
            .count();

        worker.join();

        double rtt_ns = static_cast<double>(duration_ns) / NUM_ITERATIONS;

        if (report_result) {
            std::cout << "    Run " << (run + 1) << ": "
                << std::fixed << std::setprecision(2)
                << rtt_ns << " ns (RTT)" << std::endl;
        }
    };

    auto run_rtt_dro = [&](int run, bool report_result = true) {
        dro::SPSCQueue<BenchValue> ping_queue(QUEUE_SIZE - 1);
        dro::SPSCQueue<BenchValue> pong_queue(QUEUE_SIZE - 1);

        auto worker = std::thread([&] {
            if (consumer_cpu >= 0) {
                pinThread(consumer_cpu);
            }

            for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
                BenchValue req = 0;
                ping_queue.pop(req);
                pong_queue.emplace(req);
            }
        });

        if (producer_cpu >= 0) {
            pinThread(producer_cpu);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
            ping_queue.emplace(static_cast<BenchValue>(i));

            BenchValue resp = 0;
            pong_queue.pop(resp);
            if (resp != static_cast<BenchValue>(i)) {
                std::cerr << "Error: Expected " << i << " but got " << resp << std::endl;
                exit(1);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time)
            .count();

        worker.join();

        double rtt_ns = static_cast<double>(duration_ns) / NUM_ITERATIONS;

        if (report_result) {
            std::cout << "    Run " << (run + 1) << ": "
                << std::fixed << std::setprecision(2)
                << rtt_ns << " ns (RTT)" << std::endl;
        }
    };

    auto run_rtt_raw2 = [&](int run, bool report_result = true) {
        LockFreeQueue2<BenchValue, QUEUE_SIZE> ping_queue;
        LockFreeQueue2<BenchValue, QUEUE_SIZE> pong_queue;

        auto worker = std::thread([&] {
            if (consumer_cpu >= 0) {
                pinThread(consumer_cpu);
            }

            for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
                BenchValue req = 0;
                ping_queue.pop(req);
                pong_queue.emplace(req);
            }
        });

        if (producer_cpu >= 0) {
            pinThread(producer_cpu);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
            ping_queue.emplace(static_cast<BenchValue>(i));

            BenchValue resp = 0;
            pong_queue.pop(resp);
            if (resp != static_cast<BenchValue>(i)) {
                std::cerr << "Error: Expected " << i << " but got " << resp << std::endl;
                exit(1);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time)
            .count();

        worker.join();

        double rtt_ns = static_cast<double>(duration_ns) / NUM_ITERATIONS;

        if (report_result) {
            std::cout << "    Run " << (run + 1) << ": "
                << std::fixed << std::setprecision(2)
                << rtt_ns << " ns (RTT)" << std::endl;
        }
    };

    auto run_rtt_joad = [&](int run, bool report_result = true) {
        jdz::SpscQueue<BenchValue, QUEUE_SIZE> ping_queue;
        jdz::SpscQueue<BenchValue, QUEUE_SIZE> pong_queue;

        auto worker = std::thread([&] {
            if (consumer_cpu >= 0) {
                pinThread(consumer_cpu);
            }

            for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
                BenchValue req = 0;
                ping_queue.pop(req);
                pong_queue.emplace(req);
            }
        });

        if (producer_cpu >= 0) {
            pinThread(producer_cpu);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
            ping_queue.emplace(static_cast<BenchValue>(i));

            BenchValue resp = 0;
            pong_queue.pop(resp);
            if (resp != static_cast<BenchValue>(i)) {
                std::cerr << "Error: Expected " << i << " but got " << resp << std::endl;
                exit(1);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time)
            .count();

        worker.join();

        double rtt_ns = static_cast<double>(duration_ns) / NUM_ITERATIONS;

        if (report_result) {
            std::cout << "    Run " << (run + 1) << ": "
                << std::fixed << std::setprecision(2)
                << rtt_ns << " ns (RTT)" << std::endl;
        }
    };

    auto run_rtt_boost = [&](int run, bool report_result = true) {
        boost::lockfree::spsc_queue<BenchValue, boost::lockfree::capacity<QUEUE_SIZE>> ping_queue;
        boost::lockfree::spsc_queue<BenchValue, boost::lockfree::capacity<QUEUE_SIZE>> pong_queue;

        auto worker = std::thread([&] {
            if (consumer_cpu >= 0) {
                pinThread(consumer_cpu);
            }

            for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
                BenchValue req = 0;
                while (!ping_queue.pop(req)) {
                }
                while (!pong_queue.push(req)) {
                }
            }
        });

        if (producer_cpu >= 0) {
            pinThread(producer_cpu);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
            while (!ping_queue.push(static_cast<BenchValue>(i))) {
            }

            BenchValue resp = 0;
            while (!pong_queue.pop(resp)) {
            }
            if (resp != static_cast<BenchValue>(i)) {
                std::cerr << "Error: Expected " << i << " but got " << resp << std::endl;
                exit(1);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time)
            .count();

        worker.join();

        double rtt_ns = static_cast<double>(duration_ns) / NUM_ITERATIONS;

        if (report_result) {
            std::cout << "    Run " << (run + 1) << ": "
                << std::fixed << std::setprecision(2)
                << rtt_ns << " ns (RTT)" << std::endl;
        }
    };

    auto run_rtt_folly = [&](int run, bool report_result = true) {
        folly::ProducerConsumerQueue<BenchValue> ping_queue(static_cast<uint32_t>(QUEUE_SIZE));
        folly::ProducerConsumerQueue<BenchValue> pong_queue(static_cast<uint32_t>(QUEUE_SIZE));

        auto worker = std::thread([&] {
            if (consumer_cpu >= 0) {
                pinThread(consumer_cpu);
            }

            for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
                BenchValue req = 0;
                while (!ping_queue.read(req)) {
                }
                while (!pong_queue.write(req)) {
                }
            }
        });

        if (producer_cpu >= 0) {
            pinThread(producer_cpu);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
            while (!ping_queue.write(static_cast<BenchValue>(i))) {
            }

            BenchValue resp = 0;
            while (!pong_queue.read(resp)) {
            }
            if (resp != static_cast<BenchValue>(i)) {
                std::cerr << "Error: Expected " << i << " but got " << resp << std::endl;
                exit(1);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time)
            .count();

        worker.join();

        double rtt_ns = static_cast<double>(duration_ns) / NUM_ITERATIONS;

        if (report_result) {
            std::cout << "    Run " << (run + 1) << ": "
                << std::fixed << std::setprecision(2)
                << rtt_ns << " ns (RTT)" << std::endl;
        }
    };

    auto run_rtt_tbb = [&](int run, bool report_result = true) {
        tbb::concurrent_queue<BenchValue> ping_queue;
        tbb::concurrent_queue<BenchValue> pong_queue;

        auto worker = std::thread([&] {
            if (consumer_cpu >= 0) {
                pinThread(consumer_cpu);
            }

            for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
                BenchValue req = 0;
                while (!ping_queue.try_pop(req)) {
                }
                pong_queue.push(req);
            }
        });

        if (producer_cpu >= 0) {
            pinThread(producer_cpu);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
            ping_queue.push(static_cast<BenchValue>(i));

            BenchValue resp = 0;
            while (!pong_queue.try_pop(resp)) {
            }
            if (resp != static_cast<BenchValue>(i)) {
                std::cerr << "Error: Expected " << i << " but got " << resp << std::endl;
                exit(1);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time)
            .count();

        worker.join();

        double rtt_ns = static_cast<double>(duration_ns) / NUM_ITERATIONS;

        if (report_result) {
            std::cout << "    Run " << (run + 1) << ": "
                << std::fixed << std::setprecision(2)
                << rtt_ns << " ns (RTT)" << std::endl;
        }
    };

    auto run_rtt_moodycamel = [&](int run, bool report_result = true) {
        moodycamel::ReaderWriterQueue<BenchValue> ping_queue(QUEUE_SIZE);
        moodycamel::ReaderWriterQueue<BenchValue> pong_queue(QUEUE_SIZE);

        auto worker = std::thread([&] {
            if (consumer_cpu >= 0) {
                pinThread(consumer_cpu);
            }

            for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
                BenchValue req = 0;
                while (!ping_queue.try_dequeue(req)) {
                }
                while (!pong_queue.try_enqueue(req)) {
                }
            }
        });

        if (producer_cpu >= 0) {
            pinThread(producer_cpu);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
            while (!ping_queue.try_enqueue(static_cast<BenchValue>(i))) {
            }

            BenchValue resp = 0;
            while (!pong_queue.try_dequeue(resp)) {
            }
            if (resp != static_cast<BenchValue>(i)) {
                std::cerr << "Error: Expected " << i << " but got " << resp << std::endl;
                exit(1);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time)
            .count();

        worker.join();

        double rtt_ns = static_cast<double>(duration_ns) / NUM_ITERATIONS;

        if (report_result) {
            std::cout << "    Run " << (run + 1) << ": "
                << std::fixed << std::setprecision(2)
                << rtt_ns << " ns (RTT)" << std::endl;
        }
    };

    auto run_rtt_atomic2 = [&](int run, bool report_result = true) {
        atomic_queue::AtomicQueue2<BenchValue, QUEUE_SIZE, true, true, false, true> ping_queue;
        atomic_queue::AtomicQueue2<BenchValue, QUEUE_SIZE, true, true, false, true> pong_queue;

        auto worker = std::thread([&] {
            if (consumer_cpu >= 0) {
                pinThread(consumer_cpu);
            }

            for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
                const BenchValue req = ping_queue.pop();
                pong_queue.push(req);
            }
        });

        if (producer_cpu >= 0) {
            pinThread(producer_cpu);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
            ping_queue.push(static_cast<BenchValue>(i));

            const BenchValue resp = pong_queue.pop();
            if (resp != static_cast<BenchValue>(i)) {
                std::cerr << "Error: Expected " << i << " but got " << resp << std::endl;
                exit(1);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time)
            .count();

        worker.join();

        double rtt_ns = static_cast<double>(duration_ns) / NUM_ITERATIONS;

        if (report_result) {
            std::cout << "    Run " << (run + 1) << ": "
                << std::fixed << std::setprecision(2)
                << rtt_ns << " ns (RTT)" << std::endl;
        }
    };

    auto run_rtt_for_backend = [&](BenchBackend backend, int run, bool report_result = true) {
        switch (backend) {
        case BenchBackend::Raw:
            run_rtt(run, report_result);
            break;
        case BenchBackend::Raw2:
            run_rtt_raw2(run, report_result);
            break;
        case BenchBackend::Dro:
            run_rtt_dro(run, report_result);
            break;
        case BenchBackend::Joad:
            run_rtt_joad(run, report_result);
            break;
        case BenchBackend::Boost:
            run_rtt_boost(run, report_result);
            break;
        case BenchBackend::Folly:
            run_rtt_folly(run, report_result);
            break;
        case BenchBackend::Tbb:
            run_rtt_tbb(run, report_result);
            break;
        case BenchBackend::MoodyCamel:
            run_rtt_moodycamel(run, report_result);
            break;
        case BenchBackend::Atomic2:
            run_rtt_atomic2(run, report_result);
            break;
        }
    };

    std::cout << "Payload: " << PAYLOAD_BYTES << " bytes (uint64_t)" << std::endl;
    for (BenchBackend backend : backends) {
        std::cout << "  Backend: " << backend_name(backend) << std::endl;
        for (int run = 0; run < NUM_WARMUP_RUNS; ++run) {
            run_rtt_for_backend(backend, run, false);
        }
        for (int run = 0; run < NUM_RUNS; ++run) {
            run_rtt_for_backend(backend, run);
        }
        std::cout << std::endl;
    }
    return 0;
}
