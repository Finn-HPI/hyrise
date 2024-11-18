#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <span>

#include "operators/join_simd_sort_merge/radix_partitioning.hpp"
#include "operators/join_simd_sort_merge/simd_utils.hpp"
#include "operators/join_simd_sort_merge/util.hpp"
#include "types.hpp"
#include "utils/assert.hpp"

using namespace hyrise;  // NOLINT(build/namespaces)

namespace {
template <class Tp>
inline __attribute__((always_inline)) void do_not_optimize_away(Tp const& value) {
  asm volatile("" : : "r,m"(value) : "memory");  // NOLINT
}

// leaf_size is required to be a multiple of 64 due to alignment assumptions.
template <typename T>
void benchmark(size_t number_of_partitions, std::ofstream& out, bool warmup = false) {
  std::mt19937 gen(42);

  auto num_items = size_t{100'000'000};
  auto items = std::vector<SimdElement>(num_items);

  auto dist = std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint32_t>::max());
  for (auto& item : items) {
    item.key = dist(gen);
    item.index = dist(gen);
  }

  auto radix_partition = radix_partition::RadixPartition<T>(items, number_of_partitions);

  auto temp_mem1 = simd_sort::simd_vector<SimdElement>{};
  auto temp_mem2 = simd_sort::simd_vector<SimdElement>{};

  radix_partition.execute(temp_mem1, temp_mem2);

  do_not_optimize_away(radix_partition.buckets());

  auto [time_histogram, time_init_partitions, time_partition] = radix_partition.performance_info;
  do_not_optimize_away(time_histogram);
  if (warmup) {
    return;
  }
  out << number_of_partitions << "," << time_histogram << "," << time_init_partitions << "," << time_partition
      << std::endl;
}
}  // namespace

int main() {
  std::cout << "Benchmark Radix Partitioning!\n";
  std::string file_name = "results.csv";
  std::ofstream file;
  file.open(file_name, std::ios::app);

  // Check if file opened successfully
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << file_name << '\n';
    return -1;
  }

  file << "num_partitions,time_histogram,time_init,time_partition" << '\n';

  // Start warmup runs.
  benchmark<int64_t>(32, file, true);
  benchmark<int64_t>(64, file, true);
  benchmark<int64_t>(128, file, true);
  // End warmup runs.

  const auto max_fan_out = size_t{16384};
  for (auto fan_out = size_t{32}; fan_out <= max_fan_out; fan_out *= 2) {
    std::cout << "fan_out: " << fan_out << std::endl;
    benchmark<int64_t>(fan_out, file);
  }
  // file.close();
  return 0;
}
