#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>

#include "operators/join_simd_sort_merge/k_way_merge.hpp"
#include "operators/join_simd_sort_merge/multiway_merging.hpp"
#include "operators/join_simd_sort_merge/radix_partitioning.hpp"
#include "operators/join_simd_sort_merge/simd_utils.hpp"
#include "types.hpp"

using namespace hyrise;  // NOLINT(build/namespaces)

template <class Tp>
inline __attribute__((always_inline)) void do_not_optimize_away(Tp const& value) {
  asm volatile("" : : "r,m"(value) : "memory");  // NOLINT
}

template <typename T>
simd_sort::simd_vector<T> generate_random_vector(size_t vec_size) {
  std::mt19937 gen(42);

  simd_sort::simd_vector<T> random_vector(vec_size);

  if constexpr (std::is_same_v<T, int64_t>) {
    std::uniform_int_distribution<int64_t> value_dist(INT64_MIN, INT64_MAX);
    for (auto& value : random_vector) {
      value = value_dist(gen);
    }
  } else if constexpr (std::is_same_v<T, double>) {
    std::uniform_real_distribution<double> value_dist(-1.0e10, 1.0e10);
    for (auto& value : random_vector) {
      value = value_dist(gen);
    }
  }

  return random_vector;
}

template <typename T>
std::vector<std::unique_ptr<radix_partition::Bucket>> generate_sorted_buckets(
    const size_t num_buckets, std::vector<simd_sort::simd_vector<int64_t>>& bucket_data, const size_t max_size) {
  auto buckets = std::vector<std::unique_ptr<radix_partition::Bucket>>(num_buckets);

  for (auto bucket_index = size_t{0}; bucket_index < num_buckets; ++bucket_index) {
    auto& bucket = buckets[bucket_index];
    bucket = std::make_unique<radix_partition::Bucket>();
    // Allocate bucket data
    bucket_data[bucket_index] = generate_random_vector<int64_t>(max_size);
    bucket->data = reinterpret_cast<SimdElement*>(bucket_data[bucket_index].data());
    bucket->size = bucket_data[bucket_index].size();
    // Sort internal bucket data
    std::sort(bucket->template begin<T>(), bucket->template end<T>());
  }
  return buckets;
}

void clear_cache() {
  auto clear = std::vector<int>();
  clear.resize(size_t{500} * 1000 * 1000, 42);
  const auto clear_cache_size = clear.size();
  for (auto index = size_t{0}; index < clear_cache_size; index++) {
    clear[index] += 1;
  }
  clear.resize(0);
}

constexpr size_t count_per_vector() {
#ifdef __AVX512F__
  return 8;
#else
  return 4;
#endif
}

template <class C, typename SortingType>
void benchmark(size_t scale, size_t leaf_count, std::ofstream& out) {
  const auto base_size = 1'048'576;  // 2^20
  const auto num_items = scale * base_size;

  const auto leaf_size = num_items / leaf_count;

  const auto warmup_runs = 1;
  const auto runs = 4;
  const auto num_runs = runs + warmup_runs;

  std::vector<uint64_t> runtimes;
  runtimes.reserve(runs);

  for (auto run = size_t{0}; run < num_runs; ++run) {
    auto bucket_data = std::vector<simd_sort::simd_vector<int64_t>>(leaf_count);
    auto sorted_buckets = generate_sorted_buckets<SortingType>(leaf_count, bucket_data, leaf_size);

    auto start = std::chrono::high_resolution_clock::now();
    auto merger = C(sorted_buckets);
    do_not_optimize_away(merger.merge());
    auto end = std::chrono::high_resolution_clock::now();

    /* Getting number of nanoseconds as an integer. */
    auto execution_time = duration_cast<std::chrono::milliseconds>(end - start).count();

    if (run < warmup_runs) {
      continue;
    }
    runtimes.push_back(execution_time);
  }

  const auto total_runtime = std::accumulate(runtimes.begin(), runtimes.end(), 0ul);
  auto execution_time = total_runtime / runs;

  std::cout << scale << ", " << leaf_count << ", " << execution_time << std::endl;
  out << scale << "," << execution_time;
}

int main() {
  using DataType = int64_t;
  const auto name = pmr_string{"Merging Test"};
  std::cout << "Benchmark: " << name << "!\n";

  for (auto leaf_count = size_t{8}; leaf_count <= 256; leaf_count *= 2) {
    std::string kway_output = std::to_string(leaf_count) + "_kway_merge.csv";
    std::ofstream kway_out_file;
    kway_out_file.open(kway_output, std::ios::out | std::ios::trunc);

    std::string mway_output = std::to_string(leaf_count) + "_mway_merge.csv";
    std::ofstream mway_out_file;
    mway_out_file.open(mway_output, std::ios::out | std::ios::trunc);

    for (auto scale = size_t{1}; scale <= 256; scale *= 2) {
      std::cout << "KWayMerge: scale = " << scale << std::endl;
      benchmark<k_way_merge::KWayMerge<DataType>, DataType>(scale, leaf_count, kway_out_file);
      std::cout << "MWayMerge: scale = " << scale << std::endl;
      benchmark<multiway_merging::MultiwayMerger<count_per_vector(), DataType>, DataType>(scale, leaf_count,
                                                                                          mway_out_file);
    }
    kway_out_file.close();
    mway_out_file.close();
  }
  return 0;
}
