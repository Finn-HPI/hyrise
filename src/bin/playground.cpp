#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <span>

#include <boost/range/numeric.hpp>

#include "operators/join_simd_sort_merge/merge_path.hpp"
#include "operators/join_simd_sort_merge/simd_sort.hpp"
#include "operators/join_simd_sort_merge/simd_utils.hpp"
#include "operators/join_simd_sort_merge/util.hpp"
#include "scheduler/immediate_execution_scheduler.hpp"
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
#else
  return 4;
#endif
}

template <typename T>
T* merge_using_simd_merge(std::vector<simd_sort::DataChunk<T>>& chunk_list) {
  auto chunk_count = chunk_list.size();
  while (chunk_count > 1) {
    chunk_count = simd_sort::merge_chunk_list<count_per_vector(), T>(chunk_list, chunk_count);
  }
  return chunk_list.front().input;
}

template <std::size_t count_per_vector, typename T>
[[maybe_unused]] inline std::size_t __attribute__((always_inline)) parallel_merge_chunk_list2(
    std::vector<simd_sort::DataChunk<T>>& chunk_list, std::size_t chunk_count) {
  using TwoWayMerge = hyrise::simd_sort::TwoWayMerge<count_per_vector, T>;
  auto updated_chunk_count = std::size_t{0};
  const auto last_chunk_index = chunk_count - 1;

  auto merge_tasks = std::vector<std::shared_ptr<AbstractTask>>{};

  for (auto chunk_index = std::size_t{0}; chunk_index < last_chunk_index; chunk_index += 2) {
    merge_tasks.push_back(std::make_shared<JobTask>([chunk_index, &chunk_list, updated_chunk_count]() {
      const auto& chunk_info_a = chunk_list[chunk_index];
      const auto& chunk_info_b = chunk_list[chunk_index + 1];

      TwoWayMerge::template merge_variable_length<count_per_vector * 4>(
          chunk_info_a.input, chunk_info_b.input, chunk_info_a.output, chunk_info_a.size, chunk_info_b.size);
      chunk_list[updated_chunk_count] = {chunk_info_a.output, chunk_info_a.input,
                                         chunk_info_a.size + chunk_info_b.size};
    }));
    ++updated_chunk_count;
  }
  Hyrise::get().scheduler()->schedule_and_wait_for_tasks(merge_tasks);

  // If we had odd many blocks, we have one additional unmerged block for the next iteration.
  if (chunk_count % 2) {
    chunk_list[updated_chunk_count] = chunk_list[chunk_count - 1];
    ++updated_chunk_count;
  }
  return updated_chunk_count;
}

template <typename T>
[[maybe_unused]] T* merge_using_simd_merge_par(std::vector<simd_sort::DataChunk<T>>& chunk_list) {
  auto chunk_count = chunk_list.size();
  while (chunk_count > 1) {
    chunk_count = parallel_merge_chunk_list2<count_per_vector(), T>(chunk_list, chunk_count);
  }
  return chunk_list.front().input;
}

template <size_t count_per_vector, typename T>
simd_sort::DataChunk<T> merge_recursive(std::span<simd_sort::DataChunk<T>> chunks) {
  if (chunks.empty()) {
    return {nullptr, nullptr, 0};
  }

  if (chunks.size() == 1) {
    return chunks[0];
  }

  const auto half_size = chunks.size() / 2;
  auto lhs = std::span(chunks.begin(), half_size);
  auto rhs = std::span(chunks.begin() + half_size, chunks.end());

  auto tasks = std::vector<std::shared_ptr<AbstractTask>>{};
  tasks.reserve(2);

  auto chunk_info_lhs = simd_sort::DataChunk<T>{};
  auto chunk_info_rhs = simd_sort::DataChunk<T>{};

  tasks.emplace_back(std::make_shared<JobTask>([&]() {
    chunk_info_lhs = std::move(merge_recursive<count_per_vector, T>(lhs));
  }));
  tasks.emplace_back(std::make_shared<JobTask>([&]() {
    chunk_info_rhs = std::move(merge_recursive<count_per_vector, T>(rhs));
  }));

  Hyrise::get().scheduler()->schedule_and_wait_for_tasks(tasks);

  using TwoWayMerge = hyrise::simd_sort::TwoWayMerge<count_per_vector, T>;
  TwoWayMerge::template merge_variable_length<count_per_vector * 4>(
      chunk_info_lhs.input, chunk_info_rhs.input, chunk_info_lhs.output, chunk_info_lhs.size, chunk_info_rhs.size);

  return {chunk_info_lhs.output, chunk_info_lhs.input, chunk_info_lhs.size + chunk_info_rhs.size};
}

template <typename T>
[[maybe_unused]] simd_sort::DataChunk<T> merge_using_merge_path(std::span<simd_sort::DataChunk<T>> chunks, size_t cores,
                                                                size_t level = 1) {
  if (chunks.empty()) {
    return {nullptr, nullptr, 0};
  }

  if (chunks.size() == 1) {
    return chunks[0];
  }

  const auto half_size = chunks.size() / 2;
  auto lhs = std::span(chunks.begin(), half_size);
  auto rhs = std::span(chunks.begin() + half_size, chunks.end());

  auto chunk_info_lhs = simd_sort::DataChunk<T>{};
  auto chunk_info_rhs = simd_sort::DataChunk<T>{};

  chunk_info_lhs = std::move(merge_using_merge_path<T>(lhs, cores, level + 1));
  chunk_info_rhs = std::move(merge_using_merge_path<T>(rhs, cores, level + 1));

  auto input_lhs = std::span(chunk_info_lhs.input, chunk_info_lhs.size);
  auto input_rhs = std::span(chunk_info_rhs.input, chunk_info_lhs.size);
  auto output = std::span(chunk_info_lhs.output, chunk_info_lhs.size + chunk_info_rhs.size);

  const auto num_partitions = cores / level;

  auto merge_path = merge_path::MergePath<count_per_vector(), T>(input_lhs, input_rhs, num_partitions);
  merge_path.merge(output);

  return {chunk_info_lhs.output, chunk_info_lhs.input, chunk_info_lhs.size + chunk_info_rhs.size};
}

template <bool use_merge_path = false, typename T>
T* simd_merge_parallel(std::vector<simd_sort::DataChunk<T>>& chunk_list, size_t core_count) {
  const auto merge_path_end_level = simd_sort::log2_builtin(std::bit_floor(core_count));

  if (!use_merge_path || !merge_path_end_level) {
    const auto final_chunk = std::move(merge_recursive<count_per_vector(), T>(chunk_list));
    return final_chunk.input;
  }

  const auto chunk_count = 1ul << merge_path_end_level;

  auto merged_chunks = std::vector<simd_sort::DataChunk<T>>(chunk_count);

  auto chunk_size =
      static_cast<size_t>(std::ceil(static_cast<double>(chunk_list.size()) / static_cast<double>(chunk_count)));

  // Use normal recursive SIMD merging.
  auto chunk_begin = chunk_list.begin();
  for (auto chunk_index = size_t{0}; chunk_index < chunk_count - 1; ++chunk_index) {
    const auto offset = chunk_index * chunk_size;
    auto chunk_range = std::span(chunk_begin + offset, chunk_size);
    merged_chunks[chunk_index] = std::move(merge_recursive<count_per_vector(), T>(std::move(chunk_range)));
  }
  {
    const auto offset = (chunk_count - 1) * chunk_size;
    auto chunk_range = std::span(chunk_begin + offset, chunk_list.end());
    merged_chunks[chunk_count - 1] = std::move(merge_recursive<count_per_vector(), T>(std::move(chunk_range)));
  }

  std::cout << "merged_chunks: " << merged_chunks.size() << std::endl;

  // Use MergePath to merge remaining chunks.
  const auto final_chunk = std::move(merge_using_merge_path<T>(merged_chunks, core_count));
  return final_chunk.input;
}

// leaf_size is required to be a multiple of 64 due to alignment assumptions.
template <typename T>
void benchmark(size_t leaf_count, size_t leaf_size, std::ofstream& result, size_t cores) {
  std::cout << "leaf_count: " << leaf_count << ", leaf_size: " << leaf_size << std::endl;

  const auto num_warmup_runs = size_t{2};
  const auto num_iterations = size_t{4};

  auto times = std::vector<size_t>{};
  times.reserve(num_iterations);

  auto times_par = std::vector<size_t>{};
  times_par.reserve(num_iterations);

  auto times_par_merge_path = std::vector<size_t>{};
  times_par_merge_path.reserve(num_iterations);

  for (auto it = size_t{0}; it < num_warmup_runs + num_iterations; ++it) {
    std::cout << "iteration: " << it << std::endl;
    std::mt19937 gen(42);

    // Create input and output vector of size leaf_count * leaf_size
    const auto total_size = leaf_count * leaf_size;
    auto input_simd_merge = simd_sort::simd_vector<SimdElement>(total_size);
    auto output_simd_merge = simd_sort::simd_vector<SimdElement>(total_size);

    auto input_simd_merge_par = simd_sort::simd_vector<SimdElement>(total_size);
    auto output_simd_merge_par = simd_sort::simd_vector<SimdElement>(total_size);

    auto input_simd_merge_par_merge_path = simd_sort::simd_vector<SimdElement>(total_size);
    auto output_simd_merge_par_merge_path = simd_sort::simd_vector<SimdElement>(total_size);

    auto chunk_list = std::vector<simd_sort::DataChunk<T>>(leaf_count);
    auto chunk_list_par = std::vector<simd_sort::DataChunk<T>>(leaf_count);
    auto chunk_list_par_merge_path = std::vector<simd_sort::DataChunk<T>>(leaf_count);

    for (auto offset = size_t{0}, leaf_index = size_t{0}; offset < total_size; offset += leaf_size, ++leaf_index) {
      auto* input_begin = input_simd_merge.data() + offset;
      generate_leaf<T>(std::span(input_begin, leaf_size), gen);

      // Create chunk for SIMD Merging.
      auto& chunk = chunk_list[leaf_index];
      chunk.input = reinterpret_cast<T*>(input_begin);
      chunk.output = reinterpret_cast<T*>(output_simd_merge.data() + offset);
      chunk.size = leaf_size;

      auto& chunk_par = chunk_list_par[leaf_index];
      chunk_par.input = reinterpret_cast<T*>(input_simd_merge_par.data() + offset);
      chunk_par.output = reinterpret_cast<T*>(output_simd_merge_par.data() + offset);
      chunk_par.size = leaf_size;

      auto& chunk_par_merge_path = chunk_list_par_merge_path[leaf_index];
      chunk_par_merge_path.input = reinterpret_cast<T*>(input_simd_merge_par_merge_path.data() + offset);
      chunk_par_merge_path.output = reinterpret_cast<T*>(output_simd_merge_par_merge_path.data() + offset);
      chunk_par_merge_path.size = leaf_size;
    }

    std::ranges::copy(input_simd_merge, input_simd_merge_par.begin());
    DebugAssert(input_simd_merge == input_simd_merge_par, "Mismatch between Inputs.");

    std::ranges::copy(input_simd_merge, input_simd_merge_par_merge_path.begin());
    DebugAssert(input_simd_merge == input_simd_merge_par_merge_path, "Mismatch between Inputs.");

    // Merge using SIMD Merge.
    auto start_simd = std::chrono::high_resolution_clock::now();
    auto* output = merge_using_simd_merge(chunk_list);
    auto end_simd = std::chrono::high_resolution_clock::now();

    do_not_optimize_away(output);

    auto execution_time_simd_merge = duration_cast<std::chrono::nanoseconds>(end_simd - start_simd).count();

    // Merge using par SIMD Merge.
    auto start_simd_par = std::chrono::high_resolution_clock::now();
    auto* output_par = simd_merge_parallel<false, T>(chunk_list_par, cores);
    auto end_simd_par = std::chrono::high_resolution_clock::now();

    do_not_optimize_away(output_par);

    auto execution_time_simd_merge_par = duration_cast<std::chrono::nanoseconds>(end_simd_par - start_simd_par).count();

    // Merge using par merge_path SIMD Merge.
    auto start_simd_par_merge_path = std::chrono::high_resolution_clock::now();
    auto* output_par_merge_path = simd_merge_parallel<true, T>(chunk_list_par_merge_path, cores);
    auto end_simd_par_merge_path = std::chrono::high_resolution_clock::now();

    do_not_optimize_away(output_par_merge_path);

    auto execution_time_simd_merge_par_merge_path =
        duration_cast<std::chrono::nanoseconds>(end_simd_par_merge_path - start_simd_par_merge_path).count();

    auto& sorted_data =
        (reinterpret_cast<SimdElement*>(output) == output_simd_merge.data()) ? output_simd_merge : input_simd_merge;
    auto& sorted_data_par = (reinterpret_cast<SimdElement*>(output_par) == output_simd_merge_par.data())
                                ? output_simd_merge_par
                                : input_simd_merge_par;
    auto& sorted_data_merge_merge_path =
        (reinterpret_cast<SimdElement*>(output_par_merge_path) == output_simd_merge_par_merge_path.data())
            ? output_simd_merge_par_merge_path
            : input_simd_merge_par_merge_path;

    Assert(sorted_data == sorted_data_par && sorted_data == sorted_data_merge_merge_path,
           "Merged Outputs must be the same.");

    const auto improvement_par =
        static_cast<double>(execution_time_simd_merge) / static_cast<double>(execution_time_simd_merge_par);

    const auto improvement_par_merge_path =
        static_cast<double>(execution_time_simd_merge) / static_cast<double>(execution_time_simd_merge_par_merge_path);

    std::cout << "cores: " << cores << ", seq: " << execution_time_simd_merge
              << ", par: " << execution_time_simd_merge_par
              << ", par_mpath: " << execution_time_simd_merge_par_merge_path << ", better par: x" << improvement_par
              << ", better par_mpath: x" << improvement_par_merge_path << std::endl;

    if (it < num_warmup_runs) {
      continue;
    }

    times.push_back(execution_time_simd_merge);
    times_par.push_back(execution_time_simd_merge_par);
    times_par_merge_path.push_back(execution_time_simd_merge_par_merge_path);
  }

  const auto total_duration = std::accumulate(times.begin(), times.end(), 0ul);
  const auto avg_time = total_duration / num_iterations;

  const auto total_duration_par = std::accumulate(times_par.begin(), times_par.end(), 0ul);
  const auto avg_time_par = total_duration_par / num_iterations;

  const auto total_duration_par_merge_path =
      std::accumulate(times_par_merge_path.begin(), times_par_merge_path.end(), 0ul);
  const auto avg_time_par_merge_path = total_duration_par_merge_path / num_iterations;

  const auto improvement_par = static_cast<double>(avg_time) / static_cast<double>(avg_time_par);
  const auto improvement_par_merge_path = static_cast<double>(avg_time) / static_cast<double>(avg_time_par_merge_path);

  std::cout << "cores: " << cores << ", seq: " << avg_time << ", par: " << avg_time_par
            << ", par_mpath: " << avg_time_par_merge_path << ", better par: x" << improvement_par
            << ", better par_mpath: x" << improvement_par_merge_path << std::endl;

  result << cores << "," << avg_time << "," << avg_time_par << "," << avg_time_par_merge_path << std::endl;
}

}  // namespace

int main() {
  const auto world = pmr_string{"Experiment with SIMD Merge "};
  std::cout << "Playround: " << world << "!\n";

  std::string file_name = "simd_merging_par.csv";
  std::ofstream file;
  file.open(file_name, std::ios::app);

  // Check if file opened successfully
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << file_name << std::endl;
    return -1;
  }

  file << "cores,seq_time,par_time,par_merge_path_time" << std::endl;

  std::cout << "count_per_vector: " << count_per_vector() << std::endl;

  const auto scale = 256;
  const auto base_num_items = 1'048'576;  // 2^20
  const auto num_items = base_num_items * scale;

  constexpr auto BLOCK_SIZE = hyrise::simd_sort::block_size<int64_t>();
  std::cout << "BLOCK_SIZE: " << BLOCK_SIZE << ", num_items: " << num_items << std::endl;

  const auto leaf_count = num_items / BLOCK_SIZE;

  const auto max_cores = 8;
  for (auto cores = size_t{2}; cores <= max_cores; cores *= 2) {
    Hyrise::get().topology.use_default_topology(cores);
    std::cout << "- Multi-threaded Topology:\n";
    std::cout << Hyrise::get().topology;

    const auto scheduler = std::make_shared<NodeQueueScheduler>();
    Hyrise::get().set_scheduler(scheduler);

    benchmark<int64_t>(leaf_count, BLOCK_SIZE, file, cores);

    Hyrise::get().scheduler()->finish();
  }

  Hyrise::get().set_scheduler(std::make_shared<ImmediateExecutionScheduler>());

  file.close();
  return 0;
}
