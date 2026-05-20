#include "../include/spsc_4.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>

namespace {

void test_first_blocking_emplace_on_empty_queue() {
    LockFreeQueueStage4<std::uint64_t, 8> queue;
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
        std::cerr << "timeout: spsc_4 producer failed to enqueue into an empty queue\n";
        std::abort();
    }

    std::uint64_t value = 0;
    queue.pop(value);
    assert(value == 42);

    producer.join();
}

} // namespace

int main() {
    test_first_blocking_emplace_on_empty_queue();
    std::cout << "spsc_4 tests passed\n";
    return 0;
}
