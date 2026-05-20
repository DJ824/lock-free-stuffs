#pragma once

#include <cstdint>

struct BlockingSpinStats {
    std::uint64_t producer_wait_loops{0};
    std::uint64_t consumer_wait_loops{0};
    std::uint64_t producer_index_refreshes{0};
    std::uint64_t consumer_index_refreshes{0};
    std::uint64_t producer_pause_calls{0};
    std::uint64_t consumer_pause_calls{0};
    std::uint64_t producer_pause_instructions{0};
    std::uint64_t consumer_pause_instructions{0};
};
