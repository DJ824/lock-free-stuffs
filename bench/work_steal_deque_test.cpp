#include "../include/work_steal_deque.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <string_view>
#include <thread>
#include <vector>

namespace {
    int g_total = 0;
    int g_passed = 0;
    int g_failed = 0;

#define CHECK(expr)                                                                \
    do {                                                                           \
        ++g_total;                                                                 \
        if (expr) {                                                                \
            ++g_passed;                                                            \
        } else {                                                                   \
            ++g_failed;                                                            \
            std::fprintf(stderr, "  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #expr); \
        }                                                                          \
    } while (0)

    void run_suite(std::string_view name, const std::function<void()>& fn) {
        std::printf("[suite] %s\n", name.data());
        std::fflush(stdout);
        fn();
    }

    bool check_all_seen_exactly(const std::vector<std::atomic<int>>& seen,
                                int expected_count,
                                const char* context) {
        bool ok = true;
        for (std::size_t i = 0; i < seen.size(); ++i) {
            const int value = seen[i].load(std::memory_order_relaxed);
            if (value != expected_count) {
                std::fprintf(stderr,
                             "  FAIL  [%s] item[%zu] seen %d times (expected %d)\n",
                             context,
                             i,
                             value,
                             expected_count);
                ok = false;
            }
        }
        return ok;
    }

    template <template <typename, std::size_t> class QueueTemplate, std::size_t LogSize>
    using PointerDeque = QueueTemplate<int*, (std::size_t{1} << LogSize)>;

    template <template <typename, std::size_t> class QueueTemplate>
    using StressDeque = PointerDeque<QueueTemplate, 17>;

    template <template <typename, std::size_t> class QueueTemplate>
    void test_empty_boundary_safety() {
        run_suite("empty-boundary: pop/steal on empty do not corrupt state", [] {
            PointerDeque<QueueTemplate, 3> deque;
            for (int i = 0; i < 1000; ++i) {
                CHECK(deque.pop() == nullptr);
                CHECK(deque.steal() == nullptr);
            }

            int values[8] = {1, 2, 3, 4, 5, 6, 7, 8};
            for (auto& value : values) {
                CHECK(deque.try_push(&value));
            }

            CHECK(deque.size() == 8);
            CHECK(!deque.try_push(&values[0]));
            for (int i = 7; i >= 0; --i) {
                CHECK(deque.pop() == &values[i]);
            }
            CHECK(deque.empty());
        });
    }

    template <template <typename, std::size_t> class QueueTemplate>
    void test_oscillate_boundary() {
        run_suite("boundary: oscillate push/pop at size 0<->1", [] {
            constexpr int iterations = 500'000;
            PointerDeque<QueueTemplate, 3> deque;
            std::vector<int> data(iterations);
            std::iota(data.begin(), data.end(), 0);

            bool ok = true;
            for (int i = 0; i < iterations; ++i) {
                CHECK(deque.try_push(&data[i]));
                if (deque.pop() != &data[i]) {
                    ok = false;
                }
            }

            CHECK(ok);
            CHECK(deque.empty());
        });
    }

    template <template <typename, std::size_t> class QueueTemplate>
    void test_pop_then_fill() {
        run_suite("boundary: pop-on-empty then fill-to-capacity", [] {
            PointerDeque<QueueTemplate, 2> deque;
            CHECK(deque.pop() == nullptr);
            CHECK(deque.pop() == nullptr);

            int values[4] = {10, 20, 30, 40};
            CHECK(deque.try_push(&values[0]));
            CHECK(deque.try_push(&values[1]));
            CHECK(deque.try_push(&values[2]));
            CHECK(deque.try_push(&values[3]));
            CHECK(deque.size() == 4);
            CHECK(!deque.try_push(&values[0]));

            CHECK(deque.pop() == &values[3]);
            CHECK(deque.pop() == &values[2]);
            CHECK(deque.pop() == &values[1]);
            CHECK(deque.pop() == &values[0]);
            CHECK(deque.empty());
        });
    }

    template <template <typename, std::size_t> class QueueTemplate>
    void test_multicycle_drain() {
        run_suite("boundary: multi-cycle fill+drain correctness", [] {
            constexpr int cycles = 200;
            constexpr int batch = 64;

            PointerDeque<QueueTemplate, 7> deque;
            std::vector<int> data(batch);
            std::iota(data.begin(), data.end(), 0);
            std::vector<int*> ptrs(batch);
            for (int i = 0; i < batch; ++i) {
                ptrs[i] = &data[i];
            }

            for (int cycle = 0; cycle < cycles; ++cycle) {
                const std::size_t pushed = deque.try_bulk_push(ptrs.data(), batch);
                for (std::size_t i = pushed; i < batch; ++i) {
                    CHECK(deque.try_push(ptrs[i]));
                }

                CHECK(deque.size() == batch);
                if ((cycle & 1) == 0) {
                    for (int i = batch - 1; i >= 0; --i) {
                        CHECK(deque.pop() == &data[i]);
                    }
                } else {
                    for (int i = 0; i < batch; ++i) {
                        CHECK(deque.steal() == &data[i]);
                    }
                }
                CHECK(deque.empty());
                CHECK(deque.size() == 0);
            }
        });
    }

    template <template <typename, std::size_t> class QueueTemplate>
    void test_try_methods_output_contract() {
        run_suite("single-thread: failed try_* leaves output untouched", [] {
            PointerDeque<QueueTemplate, 3> deque;
            int sentinel = 1234;
            int value = 99;
            int* out = &sentinel;

            CHECK(!deque.try_pop_bottom(out));
            CHECK(out == &sentinel);
            CHECK(!deque.try_steal_top(out));
            CHECK(out == &sentinel);

            CHECK(deque.try_push(&value));
            CHECK(deque.try_pop_bottom(out));
            CHECK(out == &value);

            out = &sentinel;
            CHECK(!deque.try_pop_bottom(out));
            CHECK(out == &sentinel);
            CHECK(!deque.try_steal_top(out));
            CHECK(out == &sentinel);
        });
    }

    template <template <typename, std::size_t> class QueueTemplate>
    void test_lifo_fifo_ordering() {
        run_suite("single-thread: strict LIFO pop / FIFO steal ordering", [] {
            PointerDeque<QueueTemplate, 4> deque;
            int values[16];
            for (int i = 0; i < 16; ++i) {
                values[i] = i;
                CHECK(deque.try_push(&values[i]));
            }

            for (int i = 15; i >= 8; --i) {
                CHECK(deque.pop() == &values[i]);
            }
            for (int i = 0; i < 8; ++i) {
                CHECK(deque.steal() == &values[i]);
            }
            CHECK(deque.empty());
        });
    }

    template <template <typename, std::size_t> class QueueTemplate>
    void test_bulk_push_correctness() {
        run_suite("single-thread: try_bulk_push partial/full semantics", [] {
            PointerDeque<QueueTemplate, 3> deque;
            int data[20];
            int* ptrs[20];
            for (int i = 0; i < 20; ++i) {
                data[i] = i;
                ptrs[i] = &data[i];
            }

            CHECK(deque.try_bulk_push(ptrs, 5) == 5);
            CHECK(deque.size() == 5);
            CHECK(deque.try_bulk_push(ptrs + 5, 5) == 3);
            CHECK(deque.size() == 8);
            CHECK(deque.try_bulk_push(ptrs, 1) == 0);

            int* expected[] = {
                ptrs[7], ptrs[6], ptrs[5], ptrs[4],
                ptrs[3], ptrs[2], ptrs[1], ptrs[0]
            };
            for (auto* item : expected) {
                CHECK(deque.pop() == item);
            }
            CHECK(deque.empty());
        });
    }

    template <template <typename, std::size_t> class QueueTemplate>
    void stress_push_pop_steal(int n_items, int n_thieves) {
        char name[128];
        std::snprintf(name,
                      sizeof(name),
                      "stress: push+pop+steal  items=%d thieves=%d",
                      n_items,
                      n_thieves);

        run_suite(name, [&] {
            auto deque = std::make_unique<StressDeque<QueueTemplate>>();
            std::vector<int> data(n_items);
            std::iota(data.begin(), data.end(), 0);
            std::vector<std::atomic<int>> seen(n_items);
            for (auto& entry : seen) {
                entry.store(0, std::memory_order_relaxed);
            }

            std::atomic<bool> done{false};
            std::barrier sync(static_cast<std::ptrdiff_t>(n_thieves + 1));

            {
                std::vector<std::jthread> thieves;
                thieves.reserve(n_thieves);
                for (int i = 0; i < n_thieves; ++i) {
                    thieves.emplace_back([&] {
                        sync.arrive_and_wait();
                        while (true) {
                            int* item = deque->steal();
                            if (item != nullptr) {
                                seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
                            } else if (done.load(std::memory_order_acquire) && deque->empty()) {
                                break;
                            } else {
                                std::this_thread::yield();
                            }
                        }
                    });
                }

                sync.arrive_and_wait();
                int index = 0;
                while (index < n_items) {
                    const int batch = std::min(64, n_items - index);
                    for (int i = 0; i < batch; ++i) {
                        while (!deque->try_push(&data[index])) {
                            std::this_thread::yield();
                        }
                        ++index;
                    }

                    for (int i = 0; i < batch / 2; ++i) {
                        int* item = deque->pop();
                        if (item != nullptr) {
                            seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }

                while (int* item = deque->pop()) {
                    seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
                }
                done.store(true, std::memory_order_release);
            }

            CHECK(check_all_seen_exactly(seen, 1, name));
        });
    }

    template <template <typename, std::size_t> class QueueTemplate>
    void stress_last_item_race(int rounds) {
        char name[128];
        std::snprintf(name, sizeof(name), "stress: last-item race  rounds=%d", rounds);

        run_suite(name, [&] {
            PointerDeque<QueueTemplate, 8> deque;
            std::vector<int> data(rounds);
            std::iota(data.begin(), data.end(), 0);
            std::vector<std::atomic<int>> seen(rounds);
            for (auto& entry : seen) {
                entry.store(0, std::memory_order_relaxed);
            }

            std::atomic<bool> done{false};

            {
                std::jthread thief([&] {
                    while (true) {
                        int* item = deque.steal();
                        if (item != nullptr) {
                            seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
                        } else if (done.load(std::memory_order_acquire) && deque.empty()) {
                            break;
                        }
                    }
                });

                for (int round = 0; round < rounds; ++round) {
                    while (!deque.try_push(&data[round])) {
                    }
                    int* item = deque.pop();
                    if (item != nullptr) {
                        seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
                    }
                }
                done.store(true, std::memory_order_release);
            }

            CHECK(check_all_seen_exactly(seen, 1, name));
        });
    }

    template <template <typename, std::size_t> class QueueTemplate>
    void stress_multiround(int n_rounds, int n_per_round, int n_thieves) {
        char name[128];
        std::snprintf(name,
                      sizeof(name),
                      "stress: multi-round  rounds=%d per_round=%d thieves=%d",
                      n_rounds,
                      n_per_round,
                      n_thieves);

        run_suite(name, [&] {
            auto deque = std::make_unique<StressDeque<QueueTemplate>>();
            std::vector<int> data(n_per_round);
            std::iota(data.begin(), data.end(), 0);
            std::vector<std::atomic<int>> seen(n_per_round);

            for (int round = 0; round < n_rounds; ++round) {
                for (auto& entry : seen) {
                    entry.store(0, std::memory_order_relaxed);
                }

                std::atomic<bool> round_done{false};
                std::barrier sync(static_cast<std::ptrdiff_t>(n_thieves + 1));

                {
                    std::vector<std::jthread> thieves;
                    thieves.reserve(n_thieves);
                    for (int i = 0; i < n_thieves; ++i) {
                        thieves.emplace_back([&] {
                            sync.arrive_and_wait();
                            while (true) {
                                int* item = deque->steal();
                                if (item != nullptr) {
                                    seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
                                } else if (round_done.load(std::memory_order_acquire) && deque->empty()) {
                                    break;
                                } else {
                                    std::this_thread::yield();
                                }
                            }
                        });
                    }

                    sync.arrive_and_wait();
                    for (auto& value : data) {
                        while (!deque->try_push(&value)) {
                            std::this_thread::yield();
                        }
                    }

                    while (int* item = deque->pop()) {
                        seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
                    }
                    round_done.store(true, std::memory_order_release);
                }

                const bool ok = check_all_seen_exactly(seen, 1, name);
                CHECK(ok);
                if (!ok) {
                    break;
                }
            }
        });
    }

    template <template <typename, std::size_t> class QueueTemplate>
    void stress_random_interleaving(int n_items, int n_thieves, std::uint64_t seed) {
        char name[160];
        std::snprintf(name,
                      sizeof(name),
                      "stress: random interleaving  items=%d thieves=%d seed=%llu",
                      n_items,
                      n_thieves,
                      static_cast<unsigned long long>(seed));

        run_suite(name, [&] {
            auto deque = std::make_unique<StressDeque<QueueTemplate>>();
            std::vector<int> data(n_items);
            std::iota(data.begin(), data.end(), 0);
            std::vector<int*> ptrs(n_items);
            for (int i = 0; i < n_items; ++i) {
                ptrs[i] = &data[i];
            }

            std::vector<std::atomic<int>> seen(n_items);
            for (auto& entry : seen) {
                entry.store(0, std::memory_order_relaxed);
            }

            std::atomic<int> remaining{n_items};
            std::atomic<bool> all_pushed{false};
            std::barrier sync(static_cast<std::ptrdiff_t>(n_thieves + 1));

            {
                std::vector<std::jthread> thieves;
                thieves.reserve(n_thieves);
                for (int i = 0; i < n_thieves; ++i) {
                    thieves.emplace_back([&] {
                        sync.arrive_and_wait();
                        while (true) {
                            int* item = deque->steal();
                            if (item != nullptr) {
                                seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
                                remaining.fetch_sub(1, std::memory_order_relaxed);
                            } else if (all_pushed.load(std::memory_order_acquire) &&
                                       remaining.load(std::memory_order_relaxed) == 0) {
                                break;
                            } else {
                                std::this_thread::yield();
                            }
                        }
                    });
                }

                sync.arrive_and_wait();
                std::mt19937_64 rng(seed);
                int pushed = 0;

                while (pushed < n_items || remaining.load(std::memory_order_relaxed) > 0) {
                    const int choice = static_cast<int>(rng() % 3);
                    if (choice == 0 && pushed < n_items) {
                        if (deque->try_push(&data[pushed])) {
                            ++pushed;
                        } else if (int* item = deque->pop()) {
                            seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
                            remaining.fetch_sub(1, std::memory_order_relaxed);
                        } else {
                            std::this_thread::yield();
                        }
                    } else if (choice == 1 && pushed < n_items) {
                        const int requested = std::min(static_cast<int>(rng() % 15) + 2, n_items - pushed);
                        const std::size_t did_push =
                            deque->try_bulk_push(ptrs.data() + pushed, static_cast<std::size_t>(requested));
                        pushed += static_cast<int>(did_push);
                        if (did_push == 0) {
                            if (int* item = deque->pop()) {
                                seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
                                remaining.fetch_sub(1, std::memory_order_relaxed);
                            } else {
                                std::this_thread::yield();
                            }
                        }
                    } else {
                        if (int* item = deque->pop()) {
                            seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
                            remaining.fetch_sub(1, std::memory_order_relaxed);
                        } else {
                            std::this_thread::yield();
                        }
                    }
                }

                all_pushed.store(true, std::memory_order_release);
                while (remaining.load(std::memory_order_relaxed) > 0) {
                    if (int* item = deque->pop()) {
                        seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
                        remaining.fetch_sub(1, std::memory_order_relaxed);
                    } else {
                        std::this_thread::yield();
                    }
                }
            }

            CHECK(check_all_seen_exactly(seen, 1, name));
        });
    }

    template <template <typename, std::size_t> class QueueTemplate>
    void stress_high_contention_steal(int n_items, int n_thieves) {
        char name[128];
        std::snprintf(name,
                      sizeof(name),
                      "stress: high-contention steal  items=%d thieves=%d",
                      n_items,
                      n_thieves);

        run_suite(name, [&] {
            PointerDeque<QueueTemplate, 8> deque;
            std::vector<int> data(n_items);
            std::iota(data.begin(), data.end(), 0);
            std::vector<std::atomic<int>> seen(n_items);
            for (auto& entry : seen) {
                entry.store(0, std::memory_order_relaxed);
            }

            std::atomic<bool> done{false};
            std::barrier sync(static_cast<std::ptrdiff_t>(n_thieves + 1));

            {
                std::vector<std::jthread> thieves;
                thieves.reserve(n_thieves);
                for (int i = 0; i < n_thieves; ++i) {
                    thieves.emplace_back([&] {
                        sync.arrive_and_wait();
                        while (true) {
                            int* item = deque.steal();
                            if (item != nullptr) {
                                seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
                            } else if (done.load(std::memory_order_acquire) && deque.empty()) {
                                break;
                            }
                        }
                    });
                }

                sync.arrive_and_wait();
                for (int i = 0; i < n_items; ++i) {
                    while (!deque.try_push(&data[i])) {
                        std::this_thread::yield();
                    }
                }

                while (int* item = deque.pop()) {
                    seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
                }
                done.store(true, std::memory_order_release);
            }

            CHECK(check_all_seen_exactly(seen, 1, name));
        });
    }

    template <template <typename, std::size_t> class QueueTemplate>
    void stress_steal_with_feedback(int n_items, int n_thieves) {
        char name[128];
        std::snprintf(name,
                      sizeof(name),
                      "stress: steal_with_feedback  items=%d thieves=%d",
                      n_items,
                      n_thieves);

        run_suite(name, [&] {
            auto deque = std::make_unique<StressDeque<QueueTemplate>>();
            std::vector<int> data(n_items);
            std::iota(data.begin(), data.end(), 0);
            std::vector<std::atomic<int>> seen(n_items);
            for (auto& entry : seen) {
                entry.store(0, std::memory_order_relaxed);
            }

            std::atomic<bool> done{false};
            std::barrier sync(static_cast<std::ptrdiff_t>(n_thieves + 1));

            {
                std::vector<std::jthread> thieves;
                thieves.reserve(n_thieves);
                for (int i = 0; i < n_thieves; ++i) {
                    thieves.emplace_back([&] {
                        sync.arrive_and_wait();
                        std::size_t empty_steals = 0;
                        while (true) {
                            int* item = deque->steal_with_feedback(empty_steals);
                            if (item != nullptr) {
                                seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
                            } else if (done.load(std::memory_order_acquire) && deque->empty()) {
                                break;
                            } else {
                                std::this_thread::yield();
                            }
                        }
                    });
                }

                sync.arrive_and_wait();
                for (auto& value : data) {
                    while (!deque->try_push(&value)) {
                        std::this_thread::yield();
                    }
                }

                while (int* item = deque->pop()) {
                    seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
                }
                done.store(true, std::memory_order_release);
            }

            CHECK(check_all_seen_exactly(seen, 1, name));
        });
    }

    template <template <typename, std::size_t> class QueueTemplate>
    void run_all_tests_for_impl(const char* impl_name, int scale, int n_thieves) {
        std::printf("== %s ==\n", impl_name);

        std::printf("-- Section 1: Empty-boundary correctness --\n");
        test_empty_boundary_safety<QueueTemplate>();
        test_oscillate_boundary<QueueTemplate>();
        test_pop_then_fill<QueueTemplate>();
        test_multicycle_drain<QueueTemplate>();

        std::printf("\n-- Section 2: Single-threaded correctness --\n");
        test_try_methods_output_contract<QueueTemplate>();
        test_lifo_fifo_ordering<QueueTemplate>();
        test_bulk_push_correctness<QueueTemplate>();

        std::printf("\n-- Section 3: Concurrent stress (scale=%d, thieves=%d) --\n",
                    scale,
                    n_thieves);

        const int n_items = 200'000 * scale;
        const int rounds = 20 * scale;
        const int last_item_rounds = 100'000 * scale;

        stress_push_pop_steal<QueueTemplate>(n_items, n_thieves);
        stress_last_item_race<QueueTemplate>(last_item_rounds);
        stress_multiround<QueueTemplate>(rounds, n_items / rounds, n_thieves);
        stress_random_interleaving<QueueTemplate>(n_items, n_thieves, 0xdeadbeefcafe1234ULL);
        stress_random_interleaving<QueueTemplate>(n_items, n_thieves, 0x0123456789abcdefULL);
        stress_high_contention_steal<QueueTemplate>(50'000 * scale, n_thieves);
        stress_steal_with_feedback<QueueTemplate>(n_items, n_thieves);

        std::printf("\n");
    }
} // namespace

int main(int argc, char* argv[]) {
    int scale = 1;
    int n_thieves = 0;

    if (argc >= 2) {
        scale = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        n_thieves = std::atoi(argv[2]);
    }

    if (scale < 1) {
        scale = 1;
    }
    if (n_thieves < 0) {
        n_thieves = 0;
    }

    const unsigned hw_threads_raw = std::thread::hardware_concurrency();
    const int hw_threads = hw_threads_raw == 0 ? 1 : static_cast<int>(hw_threads_raw);
    if (n_thieves == 0) {
        n_thieves = std::max(1, std::min(hw_threads - 1, 16));
    }

    std::printf("WorkStealDeque Unit Tests\n");
    std::printf("  scale=%d  n_thieves=%d  hw_threads=%u\n\n",
                scale,
                n_thieves,
                hw_threads_raw);

    run_all_tests_for_impl<WorkStealDeque>("WorkStealDeque", scale, n_thieves);

    std::printf("%d / %d tests passed", g_passed, g_total);
    if (g_failed != 0) {
        std::printf("  (%d FAILED)\n", g_failed);
        return EXIT_FAILURE;
    }

    std::printf("  -- all OK\n");
    return EXIT_SUCCESS;
}
