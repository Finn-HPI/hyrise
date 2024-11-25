#include <memory>
#include <random>

#include "benchmark/benchmark.h"

#include "hyrise.hpp"
#include "operators/join_simd_sort_merge/k_way_merge.hpp"
#include "operators/join_simd_sort_merge/multiway_merging.hpp"

namespace {

void clear_cache() {
  auto clear = std::vector<int>();
  clear.resize(size_t{500} * 1000 * 1000, 42);
  const auto clear_cache_size = clear.size();
  for (auto index = size_t{0}; index < clear_cache_size; index++) {
    clear[index] += 1;
  }
  clear.resize(0);
}

}  // namespace

namespace hyrise {

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
std::vector<radix_partition::Bucket> generate_sorted_buckets(const size_t num_buckets,
                                                             std::vector<simd_sort::simd_vector<int64_t>>& bucket_data,
                                                             const size_t max_size) {
  auto buckets = std::vector<radix_partition::Bucket>(num_buckets);

  for (auto bucket_index = size_t{0}; bucket_index < num_buckets; ++bucket_index) {
    auto& bucket = buckets[bucket_index];
    // Allocate bucket data
    bucket_data[bucket_index] = generate_random_vector<int64_t>(max_size);
    bucket.data = reinterpret_cast<SimdElement*>(bucket_data[bucket_index].data());
    bucket.size = bucket_data[bucket_index].size();
    // Sort internal bucket data
    std::sort(bucket.template begin<T>(), bucket.template end<T>());
  }
  return buckets;
}

std::vector<std::unique_ptr<radix_partition::Bucket>> create_bucket_ptrs(
    std::vector<radix_partition::Bucket>& sorted_buckets) {
  const auto buckets_count = sorted_buckets.size();
  auto buckets_ptr = std::vector<std::unique_ptr<radix_partition::Bucket>>(buckets_count);
  for (auto index = size_t{0}; index < buckets_count; ++index) {
    buckets_ptr[index] = std::make_unique<radix_partition::Bucket>(sorted_buckets[index]);
  }
  return buckets_ptr;
}

template <class C>
void bm_merge_impl(benchmark::State& state, std::vector<radix_partition::Bucket>& sorted_buckets) {
  clear_cache();

  auto buckets = create_bucket_ptrs(sorted_buckets);
  auto warm_up = C(buckets);
  benchmark::DoNotOptimize(warm_up.merge());

  for (auto _ : state) {  // NOLINT
    auto buckets = create_bucket_ptrs(sorted_buckets);
    auto merge_imp = C(buckets);
    benchmark::DoNotOptimize(merge_imp.merge());
  }
  Hyrise::reset();
}

template <class C>
void BM_MERGE_SMALL(benchmark::State& state) {  // NOLINT
  constexpr auto BUCKET_SIZE = 1000;
  constexpr auto COUNT_LEAVES = 32;
  using SortingType = C::value_type;

  auto bucket_data = std::vector<simd_sort::simd_vector<int64_t>>(COUNT_LEAVES);
  auto sorted_buckets = generate_sorted_buckets<SortingType>(COUNT_LEAVES, bucket_data, BUCKET_SIZE);

  bm_merge_impl<C>(state, sorted_buckets);
}

template <class C>
void BM_MERGE_MEDIUM(benchmark::State& state) {  // NOLINT
  constexpr auto COUNT_LEAVES = 256;
  constexpr auto BUCKET_SIZE = 100000;

  using SortingType = C::value_type;

  auto bucket_data = std::vector<simd_sort::simd_vector<int64_t>>(COUNT_LEAVES);
  auto sorted_buckets = generate_sorted_buckets<SortingType>(COUNT_LEAVES, bucket_data, BUCKET_SIZE);

  bm_merge_impl<C>(state, sorted_buckets);
}

template <class C>
void BM_MERGE_BIG(benchmark::State& state) {  // NOLINT
  constexpr auto COUNT_LEAVES = 256;
  constexpr auto BUCKET_SIZE = 20'000'000;

  using SortingType = C::value_type;

  auto bucket_data = std::vector<simd_sort::simd_vector<int64_t>>(COUNT_LEAVES);
  auto sorted_buckets = generate_sorted_buckets<SortingType>(COUNT_LEAVES, bucket_data, BUCKET_SIZE);

  bm_merge_impl<C>(state, sorted_buckets);
}

// BENCHMARK_TEMPLATE(BM_MERGE_SMALL, multiway_merging::MultiwayMerger<4, double>, 100);
// BENCHMARK_TEMPLATE(BM_MERGE_SMALL, k_way_merge::KWayMerge<double>);

BENCHMARK_TEMPLATE(BM_MERGE_SMALL, k_way_merge::KWayMerge<int64_t>);
BENCHMARK_TEMPLATE(BM_MERGE_SMALL, multiway_merging::MultiwayMerger<4, int64_t>);
BENCHMARK_TEMPLATE(BM_MERGE_SMALL, multiway_merging::MultiwayMerger<4, double>);

BENCHMARK_TEMPLATE(BM_MERGE_MEDIUM, k_way_merge::KWayMerge<int64_t>);
BENCHMARK_TEMPLATE(BM_MERGE_MEDIUM, multiway_merging::MultiwayMerger<4, int64_t>);
BENCHMARK_TEMPLATE(BM_MERGE_MEDIUM, multiway_merging::MultiwayMerger<4, double>);

// BENCHMARK_TEMPLATE(BM_MERGE_BIG, k_way_merge::KWayMerge<int64_t>);
// BENCHMARK_TEMPLATE(BM_MERGE_BIG, multiway_merging::MultiwayMerger<4, int64_t>);

}  // namespace hyrise
