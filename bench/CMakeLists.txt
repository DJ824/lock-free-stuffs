if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()

find_package(Threads REQUIRED)
find_package(Boost REQUIRED)

set(THIRD_PARTY_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/third_party")
set(ATOMIC_QUEUE_ROOT "${THIRD_PARTY_INCLUDE_DIR}/atomic_queue")
set(ATOMIC_QUEUE_INCLUDE_DIR "${ATOMIC_QUEUE_ROOT}/include")
set(ATOMIC_QUEUE_SRC_DIR "${ATOMIC_QUEUE_ROOT}/src")
set(LOCAL_TBB_ROOT "${THIRD_PARTY_INCLUDE_DIR}/tbb-root/usr")
set(LOCAL_TBB_INCLUDE_DIR "${LOCAL_TBB_ROOT}/include")
set(LOCAL_TBB_LIB_DIR "${LOCAL_TBB_ROOT}/lib64")
set(LOCAL_TBB_LIBRARY "${LOCAL_TBB_LIB_DIR}/libtbb.so")
set(CMAKE_CXX_FLAGS_RELEASE
        "-O3 -g -march=native -flto -ffast-math -funroll-loops -mavx2 -fno-omit-frame-pointer -falign-functions=64 -falign-loops=64")

foreach(required_header IN ITEMS
        "${THIRD_PARTY_INCLUDE_DIR}/folly/ProducerConsumerQueue.h"
        "${THIRD_PARTY_INCLUDE_DIR}/folly/concurrency/CacheLocality.h"
        "${THIRD_PARTY_INCLUDE_DIR}/concurrentqueue/concurrentqueue.h"
        "${THIRD_PARTY_INCLUDE_DIR}/readerwriterqueue/readerwriterqueue.h"
        "${THIRD_PARTY_INCLUDE_DIR}/readerwriterqueue/atomicops.h"
        "${LOCAL_TBB_INCLUDE_DIR}/oneapi/tbb/concurrent_queue.h")
    if(NOT EXISTS "${required_header}")
        message(FATAL_ERROR "Missing vendored dependency header: ${required_header}")
    endif()
endforeach()

if(NOT EXISTS "${LOCAL_TBB_LIBRARY}")
    message(FATAL_ERROR "Missing vendored TBB library: ${LOCAL_TBB_LIBRARY}")
endif()


add_executable(mpmc_1_test mpmc_1_test.cpp)
add_executable(mpmc_seq_test mpmc_seq_test.cpp)
add_executable(work_steal_deque_test work_steal_deque_test.cpp)
add_executable(work_steal_bench
        work_steal_bench.cpp
        runtime.cpp)
add_executable(mpmc_bench
        mpmc_bench.cpp
        memory.cpp
        runtime.cpp)
add_executable(spsc_4_test spsc_4_test.cpp)
add_executable(queue_bench queue_bench.cpp)
add_executable(queue_bench2
        queue_bench2.cpp
        memory.cpp
        runtime.cpp)
add_executable(queue_measure
        queue_measure.cpp
        memory.cpp
        runtime.cpp)
add_executable(atomic_queue_benchmarks
        ${ATOMIC_QUEUE_SRC_DIR}/benchmarks.cc
        ${ATOMIC_QUEUE_SRC_DIR}/cpu_base_frequency.cc
        ${ATOMIC_QUEUE_SRC_DIR}/huge_pages.cc)

foreach(target IN ITEMS mpmc_1_test mpmc_seq_test work_steal_deque_test work_steal_bench mpmc_bench spsc_4_test queue_bench queue_bench2 queue_measure atomic_queue_benchmarks)
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(${target} PRIVATE ${ATOMIC_QUEUE_INCLUDE_DIR})
    target_link_libraries(${target} PRIVATE Threads::Threads)
endforeach()

target_include_directories(mpmc_bench PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(mpmc_bench PRIVATE ${THIRD_PARTY_INCLUDE_DIR})
target_include_directories(mpmc_bench PRIVATE ${LOCAL_TBB_INCLUDE_DIR})
target_link_libraries(mpmc_bench PRIVATE ${LOCAL_TBB_LIBRARY} dl)
set_target_properties(mpmc_bench PROPERTIES BUILD_RPATH "${LOCAL_TBB_LIB_DIR}")

target_include_directories(queue_bench PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(queue_bench PRIVATE ${THIRD_PARTY_INCLUDE_DIR})
target_include_directories(queue_bench PRIVATE ${LOCAL_TBB_INCLUDE_DIR})
target_link_libraries(queue_bench PRIVATE ${LOCAL_TBB_LIBRARY})
set_target_properties(queue_bench PROPERTIES BUILD_RPATH "${LOCAL_TBB_LIB_DIR}")

target_include_directories(queue_bench2 PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(queue_bench2 PRIVATE ${THIRD_PARTY_INCLUDE_DIR})
target_include_directories(queue_bench2 PRIVATE ${LOCAL_TBB_INCLUDE_DIR})
target_link_libraries(queue_bench2 PRIVATE ${LOCAL_TBB_LIBRARY} dl)
set_target_properties(queue_bench2 PROPERTIES BUILD_RPATH "${LOCAL_TBB_LIB_DIR}")

target_link_libraries(queue_measure PRIVATE dl)
target_link_libraries(work_steal_bench PRIVATE dl)

target_include_directories(atomic_queue_benchmarks PRIVATE ${ATOMIC_QUEUE_SRC_DIR})
target_include_directories(atomic_queue_benchmarks PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(atomic_queue_benchmarks PRIVATE ${THIRD_PARTY_INCLUDE_DIR})
target_include_directories(atomic_queue_benchmarks PRIVATE ${LOCAL_TBB_INCLUDE_DIR})
target_link_libraries(atomic_queue_benchmarks PRIVATE ${LOCAL_TBB_LIBRARY} dl)
set_target_properties(atomic_queue_benchmarks PROPERTIES BUILD_RPATH "${LOCAL_TBB_LIB_DIR}")
