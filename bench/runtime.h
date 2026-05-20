#pragma once

#include <vector>

namespace support {

double cpu_base_frequency();

struct CpuTopologyInfo {
    unsigned socket_id;
    unsigned core_id;
    unsigned hw_thread_id;
};

std::vector<CpuTopologyInfo> get_cpu_topology_info();
std::vector<unsigned> hw_thread_id(std::vector<CpuTopologyInfo> const& topology);

void set_thread_affinity(unsigned hw_thread_id);
void reset_thread_affinity();
void set_default_thread_affinity(unsigned hw_thread_id);

} // namespace support
