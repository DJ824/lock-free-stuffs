#include "runtime.h"

#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>

#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <x86intrin.h>
#endif

namespace {

double cpu_base_frequency_from_model_name() {
    std::regex const re("model name\\s*:[^@]+@\\s*([0-9.]+)\\s*GHz");
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::smatch m;
    for (std::string line; getline(cpuinfo, line);) {
        if (regex_match(line, m, re)) {
            return std::stod(m[1]);
        }
    }
    return 0.0;
}

double calibrate_tsc_frequency_ghz() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    using clock = std::chrono::steady_clock;

    auto sample = [] {
        auto const start_time = clock::now();
        auto const start_cycles = __rdtsc();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        auto const end_cycles = __rdtsc();
        auto const end_time = clock::now();

        double const elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
        if (elapsed_seconds <= 0.0) {
            return 0.0;
        }
        return static_cast<double>(end_cycles - start_cycles) * 1e-9 / elapsed_seconds;
    };

    std::array<double, 3> samples{sample(), sample(), sample()};
    std::sort(samples.begin(), samples.end());
    return samples[1];
#else
    return 1.0;
#endif
}

void set_thread_affinity_(cpu_set_t const& cpuset) {
    pthread_t thread = ::pthread_self();
    if (int err = ::pthread_setaffinity_np(thread, sizeof cpuset, &cpuset)) {
        throw std::system_error(err, std::system_category(), "pthread_setaffinity_np");
    }
}

int default_thread_affinity = -1;
auto const real_pthread_create =
    reinterpret_cast<decltype(&pthread_create)>(::dlsym(RTLD_NEXT, "pthread_create"));

} // namespace

namespace support {

double cpu_base_frequency() {
    static double const frequency_ghz = [] {
        if (double const from_model_name = cpu_base_frequency_from_model_name();
            from_model_name > 0.0) {
            return from_model_name;
        }
        return calibrate_tsc_frequency_ghz();
    }();
    return frequency_ghz;
}

std::vector<CpuTopologyInfo> get_cpu_topology_info() {
    std::vector<CpuTopologyInfo> cpus;

    unsigned constexpr kMembers = 3;
    using MemberPtr = unsigned CpuTopologyInfo::*;
    MemberPtr const member_ptrs[kMembers] = {
        &CpuTopologyInfo::socket_id,
        &CpuTopologyInfo::core_id,
        &CpuTopologyInfo::hw_thread_id,
    };
    std::regex const res[kMembers] = {
        std::regex("physical id\\s+:\\s+([0-9]+)"),
        std::regex("core id\\s+:\\s+([0-9]+)"),
        std::regex("processor\\s+:\\s+([0-9]+)"),
    };

    std::ifstream cpuinfo("/proc/cpuinfo");
    std::smatch m;
    CpuTopologyInfo element{};
    unsigned valid_members = 0;
    for (std::string line; getline(cpuinfo, line);) {
        for (unsigned i = 0, mask = 1; i < kMembers; ++i, mask <<= 1) {
            if ((valid_members & mask) != 0 || !regex_match(line, m, res[i])) {
                continue;
            }
            (element.*member_ptrs[i]) = std::stoul(m[1]);
            valid_members |= mask;
            if (valid_members == ((1u << kMembers) - 1)) {
                cpus.push_back(element);
                valid_members = 0;
            }
            break;
        }
    }

    if (std::thread::hardware_concurrency() != cpus.size()) {
        throw std::runtime_error("get_cpu_topology_info() invariant broken.");
    }

    std::sort(cpus.begin(), cpus.end(), [](auto const& a, auto const& b) {
        return a.hw_thread_id < b.hw_thread_id;
    });
    return cpus;
}

std::vector<unsigned> hw_thread_id(std::vector<CpuTopologyInfo> const& topology) {
    std::vector<unsigned> ids(topology.size());
    for (size_t i = 0; i < topology.size(); ++i) {
        ids[i] = topology[i].hw_thread_id;
    }
    return ids;
}

void set_thread_affinity(unsigned hw_thread_id_value) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(hw_thread_id_value, &cpuset);
    set_thread_affinity_(cpuset);
}

void reset_thread_affinity() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (unsigned i = 0, j = std::thread::hardware_concurrency(); i < j; ++i) {
        CPU_SET(i, &cpuset);
    }
    set_thread_affinity_(cpuset);
}

void set_default_thread_affinity(unsigned hw_thread_id_value) {
    default_thread_affinity = static_cast<int>(hw_thread_id_value);
}

} // namespace support

int pthread_create(pthread_t* newthread,
                   pthread_attr_t const* attr,
                   void* (*start_routine)(void*),
                   void* arg) {
    using namespace support;

    if (!real_pthread_create) {
        std::abort();
    }

    pthread_attr_t attr2;
    pthread_attr_t* pattr = const_cast<pthread_attr_t*>(attr);
    if (default_thread_affinity >= 0) {
        if (!pattr) {
            if (::pthread_attr_init(&attr2)) {
                std::abort();
            }
            pattr = &attr2;
        }
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(default_thread_affinity, &cpuset);
        if (::pthread_attr_setaffinity_np(pattr, sizeof cpuset, &cpuset)) {
            std::abort();
        }
    }

    int r = real_pthread_create(newthread, pattr, start_routine, arg);

    if (pattr == &attr2) {
        ::pthread_attr_destroy(&attr2);
    }

    return r;
}
