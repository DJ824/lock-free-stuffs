#include "../include/mpmc_seq.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

namespace {

void test_try_api_and_fifo_order() {
    LockFreeQueueMpmcSeq<std::uint64_t, 8> queue;
    std::uint64_t value = 0;

    assert(!queue.try_pop(value));
    for (std::uint64_t i = 1; i <= queue.capacity(); ++i) {
        assert(queue.try_emplace(i));
    }
    assert(!queue.try_emplace(999));

    for (std::uint64_t i = 1; i <= queue.capacity(); ++i) {
        assert(queue.try_pop(value));
        assert(value == i);
    }
    assert(!queue.try_pop(value));
}

void test_first_blocking_emplace_on_empty_queue() {
    LockFreeQueueMpmcSeq<std::uint64_t, 8> queue;
    std::atomic<bool> producer_done{false};

    std::thread producer([&] {
        queue.emplace(42);
        producer_done.store(true, std::memory_order_release);
    });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!producer_done.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }

    if (!producer_done.load(std::memory_order_acquire)) {
        std::cerr << "timeout: mpmc_seq producer failed to enqueue into an empty queue\n";
        std::abort();
    }

    std::uint64_t value = 0;
    queue.pop(value);
    assert(value == 42);

    producer.join();
}

void test_mpmc_stress() {
    constexpr size_t producer_count = 4;
    constexpr size_t consumer_count = 4;
    constexpr size_t items_per_producer = 20'000;
    constexpr size_t total_items = producer_count * items_per_producer;

    LockFreeQueueMpmcSeq<std::uint64_t, 1024> queue;
    std::unique_ptr<std::atomic<unsigned>[]> seen(new std::atomic<unsigned>[total_items + 1]);
    for (size_t i = 0; i <= total_items; ++i) {
        seen[i].store(0, std::memory_order_relaxed);
    }

    std::atomic<size_t> producers_done{0};
    std::atomic<size_t> consumed{0};
    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;

    consumers.reserve(consumer_count);
    for (size_t consumer_id = 0; consumer_id < consumer_count; ++consumer_id) {
        consumers.emplace_back([&] {
            for (;;) {
                std::uint64_t value = 0;
                if (queue.try_pop(value)) {
                    assert(value >= 1 && value <= total_items);
                    seen[value].fetch_add(1, std::memory_order_relaxed);
                    consumed.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

                if (producers_done.load(std::memory_order_acquire) == producer_count &&
                    consumed.load(std::memory_order_acquire) == total_items) {
                    break;
                }

                std::this_thread::yield();
            }
        });
    }

    producers.reserve(producer_count);
    for (size_t producer_id = 0; producer_id < producer_count; ++producer_id) {
        producers.emplace_back([&, producer_id] {
            const std::uint64_t base = static_cast<std::uint64_t>(producer_id * items_per_producer);
            for (size_t i = 0; i < items_per_producer; ++i) {
                queue.emplace(base + static_cast<std::uint64_t>(i) + 1);
            }
            producers_done.fetch_add(1, std::memory_order_release);
        });
    }

    for (auto& producer : producers) {
        producer.join();
    }

    for (auto& consumer : consumers) {
        consumer.join();
    }

    assert(consumed.load(std::memory_order_acquire) == total_items);
    assert(queue.empty());
    for (size_t i = 1; i <= total_items; ++i) {
        assert(seen[i].load(std::memory_order_relaxed) == 1);
    }
}

} // namespace

int main() {
    test_try_api_and_fifo_order();
    test_first_blocking_emplace_on_empty_queue();
    test_mpmc_stress();
    std::cout << "mpmc_seq tests passed\n";
    return 0;
}
