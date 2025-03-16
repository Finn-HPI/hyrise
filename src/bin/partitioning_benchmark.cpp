#include <algorithm>
#include <chrono>
#include <fstream>
#include <hyrise.hpp>
#include <iostream>
#include <random>
#include <span>

#include "operators/join_simd_sort_merge/radix_partitioning_simd_elements.hpp"
#include "operators/join_simd_sort_merge/simd_utils.hpp"
#include "operators/join_simd_sort_merge/util.hpp"
#include "types.hpp"
#include "utils/assert.hpp"

using namespace hyrise;  // NOLINT(build/namespaces)

namespace {
template <class Tp>
[[maybe_unused]] inline __attribute__((always_inline)) void do_not_optimize_away(Tp const& value) {
  asm volatile("" : : "r,m"(value) : "memory");  // NOLINT
}

// leaf_size is required to be a multiple of 64 due to alignment assumptions.
template <typename T>
[[maybe_unused]] void benchmark(std::span<SimdElement> items, size_t number_of_partitions, std::ofstream& out,
                                size_t iterations, bool warmup = false) {
  auto avg_time_histogram = size_t{0};
  auto avg_time_init = size_t{0};
  auto avg_time_partition = size_t{0};

  for (auto iteration = size_t{0}; iteration < iterations; ++iteration) {
    auto radix_partition = radix_partition::RadixPartition<T>(items, number_of_partitions);

    auto temp_mem1 = simd_sort::simd_vector<SimdElement>{};
    auto temp_mem2 = simd_sort::simd_vector<SimdElement>{};

    radix_partition.execute(temp_mem1, temp_mem2);

    do_not_optimize_away(radix_partition.buckets());

    auto [time_histogram, time_init_partitions, time_partition] = radix_partition.performance_info;
    do_not_optimize_away(avg_time_histogram);
    if (warmup) {
      return;
    }
    avg_time_histogram += time_histogram;
    avg_time_init += time_init_partitions;
    avg_time_partition += time_partition;
  }
  avg_time_histogram /= iterations;
  avg_time_init /= iterations;
  avg_time_partition /= iterations;

  out << number_of_partitions << "," << avg_time_histogram << "," << avg_time_init << "," << avg_time_partition << '\n';
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

  std::mt19937 gen(42);

  auto num_items = size_t{100'000'000};
  auto items = std::vector<SimdElement>(num_items);

  auto dist = std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint32_t>::max());
  for (auto& item : items) {
    item.key = dist(gen);
    item.index = dist(gen);
  }

  const auto iterations = 5;

  // Start warmup runs.
  benchmark<int64_t>(items, 32, file, iterations, true);
  benchmark<int64_t>(items, 64, file, iterations, true);
  benchmark<int64_t>(items, 128, file, iterations, true);
  // End warmup runs.

  const auto max_fan_out = size_t{32};
  for (auto fan_out = size_t{32}; fan_out <= max_fan_out; fan_out *= 2) {
    std::cout << "fan_out: " << fan_out << '\n';
    benchmark<int64_t>(items, fan_out, file, iterations);
  }
  file.close();
  return 0;
}
