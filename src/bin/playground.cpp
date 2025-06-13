#include <iostream>
#include <random>

#include <boost/container_hash/hash.hpp>
#include <boost/functional/hash.hpp>
#include <boost/unordered/unordered_flat_set.hpp>

#include "operators/join_simd_sort_merge/field_accessor.hpp"
#include "operators/join_simd_sort_merge/simd_sort.hpp"
#include "operators/join_simd_sort_merge/simd_utils.hpp"
#include "operators/join_simd_sort_merge/util.hpp"

using namespace hyrise;  // NOLINT(build/namespaces)

int main() {
  using T = int64_t;
  constexpr auto BLOCK_SIZE = simd_sort::block_size<T>();
  constexpr auto NUM_BLOCKS = 1027;

  auto vec = simd_sort::simd_vector<T>(NUM_BLOCKS * BLOCK_SIZE);
  auto tmp = simd_sort::simd_vector<T>(NUM_BLOCKS * BLOCK_SIZE);

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<T> dist(0.0, 1000000);

  for (auto& val : vec) {
    val = dist(rng);
  }

  constexpr size_t COUNT_PER_VECTOR = 4;
  auto* input = vec.data();
  auto* output = tmp.data();
  auto num_elements = vec.size();

  simd_sort::sort<COUNT_PER_VECTOR, T, ExecutionStrategy::ParallelMergeSort>(input, output, num_elements);

  return 0;
}
