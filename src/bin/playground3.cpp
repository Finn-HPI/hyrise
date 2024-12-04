#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <span>

#include <boost/range/numeric.hpp>

#include "operators/join_simd_sort_merge/multiway_merging.hpp"
#include "operators/join_simd_sort_merge/multiway_merging_par.hpp"
#include "operators/join_simd_sort_merge/radix_partitioning.hpp"
#include "operators/join_simd_sort_merge/simd_sort.hpp"
#include "operators/join_simd_sort_merge/simd_utils.hpp"
#include "operators/join_simd_sort_merge/util.hpp"
#include "scheduler/node_queue_scheduler.hpp"
#include "types.hpp"
#include "utils/assert.hpp"

using namespace hyrise;  // NOLINT(build/namespaces)

namespace {
template <class Tp>
inline __attribute__((always_inline)) void do_not_optimize_away(Tp const& value) {
  asm volatile("" : : "r,m"(value) : "memory");  // NOLINT
}

template <typename T>
auto uniform_distribution() {
  if constexpr (std::is_integral_v<T>) {
    return std::uniform_int_distribution<T>(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());
  } else if constexpr (std::is_floating_point_v<T>) {
    return std::uniform_real_distribution<T>(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());
  }
}

template <typename T>
void generate_leaf(std::span<SimdElement> range, std::mt19937& gen) {
  auto dist = uniform_distribution<T>();
  // fill with random values of type T
  for (auto& elem : range) {
    auto value = dist(gen);
    elem = *reinterpret_cast<SimdElement*>(&value);
  }
  std::sort(range.begin(), range.end(), [](auto& lhs, auto& rhs) {
    return *reinterpret_cast<T*>(&lhs) < *reinterpret_cast<T*>(&rhs);
  });
}

constexpr size_t count_per_vector() {
#ifdef __AVX512F__
  return 8;
#endif
  return 4;
}

// template <typename T>
// T* merge_using_simd_merge(std::vector<simd_sort::DataChunk<T>>& chunk_list) {
//   auto chunk_count = chunk_list.size();
//   while (chunk_count > 1) {
//     chunk_count = simd_sort::merge_chunk_list<count_per_vector(), T>(chunk_list, chunk_count);
//   }
//   return chunk_list.front().input;
// }

// template <std::size_t count_per_vector, typename T>
// inline std::size_t __attribute__((always_inline)) parallel_merge_chunk_list2(
//     std::vector<simd_sort::DataChunk<T>>& chunk_list, std::size_t chunk_count) {
//   using TwoWayMerge = hyrise::simd_sort::TwoWayMerge<count_per_vector, T>;
//   auto updated_chunk_count = std::size_t{0};
//   const auto last_chunk_index = chunk_count - 1;
//
//   auto merge_tasks = std::vector<std::shared_ptr<AbstractTask>>{};
//
//   for (auto chunk_index = std::size_t{0}; chunk_index < last_chunk_index; chunk_index += 2) {
//     const auto& chunk_info_a = chunk_list[chunk_index];
//     const auto& chunk_info_b = chunk_list[chunk_index + 1];
//
//     merge_tasks.push_back(std::make_shared<JobTask>([&chunk_list, chunk_info_a, chunk_info_b, updated_chunk_count]() {
//       TwoWayMerge::template merge_variable_length<count_per_vector * 4>(
//           chunk_info_a.input, chunk_info_b.input, chunk_info_a.output, chunk_info_a.size, chunk_info_b.size);
//       chunk_list[updated_chunk_count] = {chunk_info_a.output, chunk_info_a.input,
//                                          chunk_info_a.size + chunk_info_b.size};
//     }));
//     ++updated_chunk_count;
//   }
//   // std::cout << "merge_tasks: " << merge_tasks.size() << std::endl;
//   Hyrise::get().scheduler()->schedule_and_wait_for_tasks(merge_tasks);
//
//   // If we had odd many blocks, we have one additional unmerged block for the next iteration.
//   if (chunk_count % 2) {
//     chunk_list[updated_chunk_count] = chunk_list[chunk_count - 1];
//     ++updated_chunk_count;
//   }
//   return updated_chunk_count;
// }

//
// template <typename T>
// T* merge_using_simd_merge_par(std::vector<simd_sort::DataChunk<T>>& chunk_list) {
//   auto chunk_count = chunk_list.size();
//   while (chunk_count > 1) {
//     chunk_count = parallel_merge_chunk_list2<count_per_vector(), T>(chunk_list, chunk_count);
//   }
//   return chunk_list.front().input;
// }

// leaf_size is required to be a multiple of 64 due to alignment assumptions.
template <typename T>
void benchmark(size_t leaf_count, size_t leaf_size, std::ofstream& result) {
  std::cout << "leaf_count: " << leaf_count << ", leaf_size: " << leaf_size << std::endl;

  const auto num_iterations = size_t{4};

  auto times = std::vector<size_t>{};
  times.reserve(num_iterations);

  for (auto it = size_t{0}; it < num_iterations; ++it) {
    std::mt19937 gen(42);

    // Create input and output vector of size leaf_count * leaf_size
    const auto total_size = leaf_count * leaf_size;
    auto input_simd_merge = simd_sort::simd_vector<SimdElement>(total_size);
    auto output_simd_merge = simd_sort::simd_vector<SimdElement>(total_size);

    auto input_multiway_merge = simd_sort::simd_vector<SimdElement>(total_size);
    auto sorted_buckets = std::vector<std::unique_ptr<radix_partition::Bucket>>(leaf_count);

    // auto input_multiway_merge_par = simd_sort::simd_vector<SimdElement>(total_size);
    // auto sorted_buckets_par = std::vector<std::unique_ptr<radix_partition::Bucket>>(leaf_count);
    auto chunk_list = std::vector<simd_sort::DataChunk<T>>(leaf_count);

    for (auto offset = size_t{0}, leaf_index = size_t{0}; offset < total_size; offset += leaf_size, ++leaf_index) {
      auto* input_begin = input_multiway_merge.data() + offset;
      generate_leaf<T>(std::span(input_begin, leaf_size), gen);
      // Create leaf bucket for Multiway Merging.
      sorted_buckets[leaf_index] =
          std::make_unique<radix_partition::Bucket>(input_multiway_merge.data() + offset, leaf_size);

      // sorted_buckets_par[leaf_index] =
      //     std::make_unique<radix_partition::Bucket>(input_multiway_merge_par.data() + offset, leaf_size);

      // Create chunk for SIMD Merging.
      auto& chunk = chunk_list[leaf_index];
      chunk.input = reinterpret_cast<T*>(input_begin);
      chunk.output = reinterpret_cast<T*>(output_simd_merge.data() + offset);
      chunk.size = leaf_size;
      //
      // auto& chunk_par = chunk_list_par[leaf_index];
      // chunk_par.input = reinterpret_cast<T*>(input_simd_par_merge.data() + offset);
      // chunk_par.output = reinterpret_cast<T*>(output_simd_par_merge.data() + offset);
      // chunk_par.size = leaf_size;
    }

    // Copy input for Multiway Merging.
    std::ranges::copy(input_multiway_merge, input_simd_merge.begin());
    DebugAssert(input_simd_merge == input_multiway_merge, "Mismatch between Inputs.");

    // Merge using Mutliway Merge.
    auto start_mutliway_merging = std::chrono::high_resolution_clock::now();

    auto multiway_merger = multiway_merging::MultiwayMerger<count_per_vector(), T>(sorted_buckets);
    auto merged_output_multiway_merge = std::move(multiway_merger.merge());

    auto end_multiway_merging = std::chrono::high_resolution_clock::now();

    DebugAssert(merged_output_multiway_merge.size() == total_size, "Output of Multiway Merging has wrong size.");
    DebugAssert(std::ranges::is_sorted(merged_output_multiway_merge,
                                       [](auto& lhs, auto& rhs) {
                                         return *reinterpret_cast<T*>(&lhs) < *reinterpret_cast<T*>(&rhs);
                                       }),
                "Output of Multiway Merging is not sorted.");

    do_not_optimize_away(merged_output_multiway_merge);

    auto execution_time_mutliway_merging =
        duration_cast<std::chrono::nanoseconds>(end_multiway_merging - start_mutliway_merging).count();

    times.push_back(execution_time_mutliway_merging);
  }
  const auto total_duration = std::accumulate(times.begin(), times.end(), 0ul);
  const auto avg_time = total_duration / num_iterations;
  result << leaf_count << "," << leaf_size << "," << avg_time << std::endl;
}

}  // namespace

int main() {
  const auto world = pmr_string{"Experiment with SIMD Merge and Multiway Merge"};
  std::cout << "Playround: " << world << "!\n";

  // Experiment to determine threshold between SIMD Merge and Mutliway Merging.
  // Input paramters are number of leaves and leaf size.

  std::string file_name = "results.csv";
  std::ofstream file;
  file.open(file_name, std::ios::app);

  // Check if file opened successfully
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << file_name << std::endl;
    return -1;
  }

  const auto max_leaf_size = size_t{2'000'000};
  for (auto leaf_count = size_t{8}; leaf_count <= 128; leaf_count *= 2) {
    for (auto leaf_size = size_t{100'000}; leaf_size <= max_leaf_size; leaf_size += 100'000) {
      benchmark<int64_t>(leaf_count, leaf_size, file);
    }
  }
  file.close();
  return 0;
}
