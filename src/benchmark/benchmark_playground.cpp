#include <benchmark/benchmark.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <span>
#include <zipfian_int_distribution.hpp>

#include <boost/sort/pdqsort/pdqsort.hpp>

#include "operators/join_simd_sort_merge/k_way_merge.hpp"
#include "operators/join_simd_sort_merge/multiway_merging.hpp"
#include "operators/join_simd_sort_merge/multiway_merging_balkesen.hpp"
#include "operators/join_simd_sort_merge/radix_partitioning.hpp"
#include "operators/join_simd_sort_merge/simd_utils.hpp"
#include "operators/join_simd_sort_merge/util.hpp"
#include "types.hpp"

using namespace hyrise;  // NOLINT(build/namespaces)

using Bucket = radix_partition::Bucket;
using SortingType = double;

namespace {
[[maybe_unused]] constexpr std::size_t choose_count_per_vector() {
#if defined(__AVX512F__)
  return 8;
#else
  return 4;
#endif
}

[[maybe_unused]] simd_sort::simd_vector<SimdElement> generate_ordered_tuples(size_t num_tuples) {
  constexpr uint32_t INCRMOD = 100;
  constexpr uint32_t MAX_INT = std::numeric_limits<uint32_t>::max() - INCRMOD;

  std::mt19937 gen(42);
  std::uniform_int_distribution<uint32_t> dist(1, INCRMOD - 1);
  auto result = simd_sort::simd_vector<SimdElement>(num_tuples);

  uint32_t next_key = dist(gen);

  for (auto index = size_t{0}; index < num_tuples; ++index) {
    result[index].key = 0;
    if (next_key < MAX_INT) {
      next_key += dist(gen);
    }
  }
  return result;
}
}  // namespace

constexpr auto CACHE_SIZE = size_t{1024} * 1024 * 1;
constexpr auto CACHE_PER_THREAD = CACHE_SIZE / 1;

// NOLINTNEXTLINE
static void BM_MWAY_MERGE(benchmark::State& state) {
  const auto fan_in = static_cast<size_t>(state.range(0));
  const auto total_size = static_cast<size_t>(state.range(1));
  const auto chunk_size = total_size / fan_in;

  std::cout << "Fan-in=" << fan_in << ", Tuples=" << total_size << '\n';

  // NOLINTNEXTLINE
  for (auto _ : state) {
    state.PauseTiming();
    auto chunks = std::vector<simd_sort::simd_vector<SimdElement>>(fan_in);
    auto buckets = std::vector<Relation>(fan_in);
    auto sorted_bucket_ptrs = std::vector<Relation*>(fan_in);
    for (auto chunk_id = size_t{0}; chunk_id < fan_in; ++chunk_id) {
      chunks[chunk_id] = generate_ordered_tuples(chunk_size);
      buckets[chunk_id].tuples = chunks[chunk_id].data();
      buckets[chunk_id].num_tuples = chunk_size;
      sorted_bucket_ptrs[chunk_id] = &buckets[chunk_id];
    }
    auto merged_output = simd_sort::simd_vector<SimdElement>(chunk_size * fan_in);
    state.ResumeTiming();

    auto start = std::chrono::high_resolution_clock::now();

    auto multiway_merger = multiway_merging::MultiwayMergerBalkesen<choose_count_per_vector(), SortingType>(
        sorted_bucket_ptrs, CACHE_PER_THREAD);
    multiway_merger.merge(merged_output);
    benchmark::DoNotOptimize(merged_output);
    benchmark::ClobberMemory();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    state.SetIterationTime(elapsed_seconds.count());
  }
  state.counters["Fan-in"] = static_cast<double>(fan_in);
  state.counters["Tuples"] = static_cast<double>(total_size);
}

// NOLINTNEXTLINE
[[maybe_unused]] static void BM_KWAY_MERGE(benchmark::State& state) {
  const auto fan_in = static_cast<size_t>(state.range(0));
  const auto total_size = static_cast<size_t>(state.range(1));
  const auto chunk_size = total_size / fan_in;

  std::cout << "Fan-in=" << fan_in << ", Tuples=" << total_size << '\n';

  // NOLINTNEXTLINE
  for (auto _ : state) {
    state.PauseTiming();
    auto chunks = std::vector<simd_sort::simd_vector<SimdElement>>(fan_in);
    auto buckets = std::vector<radix_partition::Bucket>(fan_in);
    auto sorted_bucket_ptrs = std::vector<Bucket*>(fan_in);
    for (auto chunk_id = size_t{0}; chunk_id < fan_in; ++chunk_id) {
      chunks[chunk_id] = generate_ordered_tuples(chunk_size);
      buckets[chunk_id].data = chunks[chunk_id].data();
      buckets[chunk_id].size = chunk_size;
      sorted_bucket_ptrs[chunk_id] = &buckets[chunk_id];
    }
    auto merged_output = simd_sort::simd_vector<SimdElement>(chunk_size * fan_in);
    state.ResumeTiming();

    auto start = std::chrono::high_resolution_clock::now();

    auto kway_merger = k_way_merge::KWayMerge<SortingType>(sorted_bucket_ptrs);
    kway_merger.merge(merged_output);
    benchmark::DoNotOptimize(merged_output);
    benchmark::ClobberMemory();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    state.SetIterationTime(elapsed_seconds.count());
  }
  state.counters["Fan-in"] = static_cast<double>(fan_in);
  state.counters["Tuples"] = static_cast<double>(total_size);
}

constexpr auto NUM_TUPLES = size_t{16} * 1024 * 1024;

BENCHMARK(BM_MWAY_MERGE)->Args({4, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();
BENCHMARK(BM_MWAY_MERGE)->Args({5, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();
BENCHMARK(BM_MWAY_MERGE)->Args({6, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();
BENCHMARK(BM_MWAY_MERGE)->Args({7, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();
BENCHMARK(BM_MWAY_MERGE)->Args({8, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();
BENCHMARK(BM_MWAY_MERGE)->Args({16, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();
BENCHMARK(BM_MWAY_MERGE)->Args({32, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();
BENCHMARK(BM_MWAY_MERGE)->Args({64, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();
BENCHMARK(BM_MWAY_MERGE)->Args({128, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();
BENCHMARK(BM_MWAY_MERGE)->Args({256, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();
BENCHMARK(BM_MWAY_MERGE)->Args({512, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();
BENCHMARK(BM_MWAY_MERGE)->Args({1024, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();
BENCHMARK(BM_MWAY_MERGE)->Args({2048, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();

BENCHMARK(BM_KWAY_MERGE)->Args({4, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();
BENCHMARK(BM_KWAY_MERGE)->Args({8, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();
BENCHMARK(BM_KWAY_MERGE)->Args({16, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();
BENCHMARK(BM_KWAY_MERGE)->Args({32, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();
BENCHMARK(BM_KWAY_MERGE)->Args({64, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();
BENCHMARK(BM_KWAY_MERGE)->Args({128, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();
BENCHMARK(BM_KWAY_MERGE)->Args({256, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();
BENCHMARK(BM_KWAY_MERGE)->Args({512, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();
BENCHMARK(BM_KWAY_MERGE)->Args({1024, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();
BENCHMARK(BM_KWAY_MERGE)->Args({2048, NUM_TUPLES})->Unit(benchmark::kMillisecond)->Iterations(5)->UseManualTime();

BENCHMARK_MAIN();
