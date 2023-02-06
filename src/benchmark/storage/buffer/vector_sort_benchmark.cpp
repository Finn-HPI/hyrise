#include <memory>

#include <boost/container/vector.hpp>
#include <memory>
#include <numeric>
#include <vector>
#include "benchmark/benchmark.h"
#include "storage/buffer/buffer_pool_allocator.hpp"

namespace hyrise {

static void BM_vector_sort_custom(benchmark::State& state) {
  auto count = static_cast<size_t>(state.range(0));
  auto array = boost::container::vector<int32_t, BufferPoolAllocator<int32_t>>(count);
  std::iota(array.begin().get_ptr().operator->(), array.end().get_ptr().operator->(), 1);

  for (auto _ : state) {
    state.PauseTiming();
    std::reverse(array.begin().get_ptr().operator->(), array.end().get_ptr().operator->());
    state.ResumeTiming();

    std::sort(array.begin().get_ptr().operator->(), array.end().get_ptr().operator->());
    benchmark::DoNotOptimize(array.size());
  }
}

template <typename VectorType>
static void BM_vector_sort(benchmark::State& state) {
  auto count = static_cast<size_t>(state.range(0));
  auto array = VectorType(count);
  std::iota(array.begin(), array.end(), 1);

  for (auto _ : state) {
    state.PauseTiming();
    std::reverse(array.begin(), array.end());
    state.ResumeTiming();

    std::sort(array.begin(), array.end());
    benchmark::DoNotOptimize(array.size());
  }
}

BENCHMARK_TEMPLATE(BM_vector_sort, std::vector<int32_t, std::allocator<int32_t>>)
    ->Name("std::sort with std::vector")
    ->Range(8, 8 << 9);
BENCHMARK_TEMPLATE(BM_vector_sort, boost::container::vector<int32_t, std::allocator<int32_t>>)
    ->Name("std::sort with boost::container::vector")
    ->Range(8, 8 << 9);
BENCHMARK_TEMPLATE(BM_vector_sort, boost::container::vector<int32_t, BufferPoolAllocator<int32_t>>)
    ->Name("std::sort with pmr_vector (Buffer Pool)")
    ->Range(8, 8 << 9);
BENCHMARK(BM_vector_sort_custom)->Name("std::sort with pmr_vector (Buffer Pool) and raw pointers")->Range(8, 8 << 9);
}  // namespace hyrise