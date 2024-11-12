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

template <class C>
uint64_t benchmark(size_t count_leaves, size_t leaf_size) {
  using SortingType = C::value_type;
  auto bucket_data = std::vector<simd_sort::simd_vector<int64_t>>(count_leaves);
  auto sorted_buckets = generate_sorted_buckets<SortingType>(count_leaves, bucket_data, leaf_size);

  auto start = std::chrono::high_resolution_clock::now();
  auto merger = C(sorted_buckets);
  do_not_optimize_away(merger.merge());
  auto end = std::chrono::high_resolution_clock::now();

  /* Getting number of milliseconds as an integer. */
  auto ns_int = duration_cast<std::chrono::nanoseconds>(end - start);
  return ns_int.count();
}

int main() {
  const auto name = pmr_string{"Merging Test"};
  std::cout << "Playground: " << name << "!\n";

  const auto max_leaf_size = 1000 * 20;
  const auto increment = 1000;

  auto run = [&]<class C>(std::string_view type) {
    std::cout << "run: " << type << std::endl;
    std::string file_name = std::string(type) + "_results.csv";
    std::ofstream file;
    file.open(file_name, std::ios::app);

    // Check if file opened successfully
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << file_name << std::endl;
      return;
    }

    for (auto leaf_size = size_t{1000}; leaf_size <= max_leaf_size; leaf_size += increment) {
      std::cout << "leaf_size: " << leaf_size << std::endl;
      for (auto leaf_count = size_t{2}; leaf_count <= 256; ++leaf_count) {
        auto execution_time = benchmark<C>(leaf_count, leaf_size);

        file << leaf_count << "," << leaf_size << "," << execution_time << "\n";
      }
      file.flush();
    }
    file.close();
  };

  run.template operator()<k_way_merge::KWayMerge<int64_t>>("kway_merge_int");
#ifdef __AVX512F__
  run.template operator()<multiway_merging::MultiwayMerger<8, int64_t>>("multiway_merge_int");
#else
  run.template operator()<multiway_merging::MultiwayMerger<4, int64_t>>("multiway_merge_int");
#endif

  run.template operator()<k_way_merge::KWayMerge<double>>("kway_merge_double");
#ifdef __AVX512F__
  run.template operator()<multiway_merging::MultiwayMerger<8, double>>("multiway_merge_double");
#else
  run.template operator()<multiway_merging::MultiwayMerger<4, double>>("multiway_merge_double");
#endif
  return 0;
}
