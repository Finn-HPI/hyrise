#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>

#include <boost/sort/sort.hpp>

#include "hyrise.hpp"
#include "operators/join_simd_sort_merge/merge_path.hpp"
#include "operators/join_simd_sort_merge/util.hpp"
#include "scheduler/abstract_task.hpp"
#include "scheduler/job_task.hpp"
#include "simd_utils.hpp"
#include "two_way_merge.hpp"

namespace hyrise::simd_sort {

template <typename T>
struct DataChunk {
  T* input;
  T* output;
  std::size_t size;
};

template <std::size_t count_per_vector, std::size_t kernel_size, typename T>
void merge_level(std::size_t level, std::array<T*, 2>& ptrs, std::size_t num_items) {
  auto ptr_index = level & 1u;
  auto* input = ptrs[ptr_index];
  auto* output = ptrs[ptr_index ^ 1u];
  auto* const end = input + num_items;

  const auto input_length = 1u << level;          // = 2^level
  const auto output_length = input_length << 1u;  // = input_length x 2
  using TwoWayMerge = TwoWayMerge<count_per_vector, T>;
  while (input < end) {
    TwoWayMerge::template merge_equal_length<kernel_size>(input, input + input_length, output, input_length);
    input += output_length;
    output += output_length;
  }
}

template <std::size_t count_per_vector, typename T>
inline void __attribute__((always_inline)) sort_chunk(DataChunk<T>& block) {
  const auto num_items = block.size;
  const auto start_level = log2_builtin(count_per_vector);

  auto input_output_pointers = std::array<T*, 2>{};
  auto input_selection_index = start_level & 1u;
  input_output_pointers[input_selection_index] = block.input;
  input_output_pointers[input_selection_index ^ 1u] = block.output;
  {
    using block_t = struct alignas(sizeof(T) * count_per_vector * count_per_vector) {};

    auto* block_start_address = reinterpret_cast<block_t*>(block.input);
    auto* const block_end_address = reinterpret_cast<block_t*>(block.input + num_items);
    using SortingNetwork = SortingNetwork<count_per_vector, T>;
    while (block_start_address < block_end_address) {
      SortingNetwork::sort(reinterpret_cast<T*>(block_start_address), reinterpret_cast<T*>(block_start_address));
      ++block_start_address;
    }
  }
  const auto log_block_size = log2_builtin(num_items);
  const auto stop_level = log_block_size - 2;
  merge_level<count_per_vector, count_per_vector>(start_level, input_output_pointers, num_items);
  merge_level<count_per_vector, count_per_vector * 2>(start_level + 1, input_output_pointers, num_items);

  for (auto level = std::size_t{start_level + 2}; level < stop_level; ++level) {
    merge_level<count_per_vector, count_per_vector * 4>(level, input_output_pointers, num_items);
  }

  auto input_length = 1u << stop_level;
  input_selection_index = stop_level & 1u;
  auto* input = input_output_pointers[input_selection_index];
  auto* output = input_output_pointers[input_selection_index ^ 1u];

  using TwoWayMerge = TwoWayMerge<count_per_vector, T>;
  TwoWayMerge::template merge_equal_length<count_per_vector * 4>(input, input + input_length, output, input_length);
  TwoWayMerge::template merge_equal_length<count_per_vector * 4>(input + (2 * input_length), input + (3 * input_length),
                                                                 output + (2 * input_length), input_length);
  input_length <<= 1u;
  TwoWayMerge::template merge_equal_length<count_per_vector * 4>(output, output + input_length, input, input_length);
  block.input = output;
  block.output = input;
}

template <std::size_t count_per_vector, typename T>
inline std::size_t __attribute__((always_inline)) merge_chunk_list(std::vector<DataChunk<T>>& chunk_list,
                                                                   std::size_t chunk_count) {
  using TwoWayMerge = TwoWayMerge<count_per_vector, T>;
  auto updated_chunk_count = std::size_t{0};
  const auto last_chunk_index = chunk_count - 1;
  for (auto chunk_index = std::size_t{0}; chunk_index < last_chunk_index; chunk_index += 2) {
    const auto& chunk_info_a = chunk_list[chunk_index];
    const auto& chunk_info_b = chunk_list[chunk_index + 1];
    TwoWayMerge::template merge_variable_length<count_per_vector * 4>(
        chunk_info_a.input, chunk_info_b.input, chunk_info_a.output, chunk_info_a.size, chunk_info_b.size);
    chunk_list[updated_chunk_count] = {chunk_info_a.output, chunk_info_a.input, chunk_info_a.size + chunk_info_b.size};
    ++updated_chunk_count;
  }
  // If we had odd many blocks, we have one additional unmerged block for the next iteration.
  if (chunk_count % 2) {
    chunk_list[updated_chunk_count] = chunk_list[chunk_count - 1];
    ++updated_chunk_count;
  }
  return updated_chunk_count;
}

template <std::size_t count_per_vector, typename T>
inline std::size_t __attribute__((always_inline)) parallel_merge_chunk_list(std::vector<DataChunk<T>>& chunk_list,
                                                                            std::size_t chunk_count) {
  using TwoWayMerge = TwoWayMerge<count_per_vector, T>;
  auto updated_chunk_count = std::size_t{0};
  const auto last_chunk_index = chunk_count - 1;

  auto merge_tasks = std::vector<std::shared_ptr<AbstractTask>>{};
  merge_tasks.reserve(last_chunk_index / 2);

  for (auto chunk_index = std::size_t{0}; chunk_index < last_chunk_index; chunk_index += 2) {
    const auto& chunk_info_a = chunk_list[chunk_index];
    const auto& chunk_info_b = chunk_list[chunk_index + 1];

    merge_tasks.push_back(std::make_shared<JobTask>([&chunk_list, chunk_info_a, chunk_info_b, updated_chunk_count]() {
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

template <size_t count_per_vector, typename T>
simd_sort::DataChunk<T> merge_recursive_with_merge_path(std::span<simd_sort::DataChunk<T>> chunks,
                                                        size_t last_skip_level, size_t cores, size_t level = 1) {
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
    chunk_info_lhs =
        std::move(merge_recursive_with_merge_path<count_per_vector, T>(lhs, last_skip_level, cores, level + 1));
  }));
  tasks.emplace_back(std::make_shared<JobTask>([&]() {
    chunk_info_rhs =
        std::move(merge_recursive_with_merge_path<count_per_vector, T>(rhs, last_skip_level, cores, level + 1));
  }));

  Hyrise::get().scheduler()->schedule_and_wait_for_tasks(tasks);

  // Use MergePath.
  if (level <= last_skip_level) {
    auto input_lhs = std::span(chunk_info_lhs.input, chunk_info_lhs.size);
    auto input_rhs = std::span(chunk_info_rhs.input, chunk_info_lhs.size);
    auto output = std::span(chunk_info_lhs.output, chunk_info_lhs.size + chunk_info_rhs.size);

    const auto num_partitions = cores / level;

    auto merge_path = merge_path::MergePath<count_per_vector, T>(input_lhs, input_rhs, num_partitions);
    merge_path.merge(output);

    return {chunk_info_lhs.output, chunk_info_lhs.input, chunk_info_lhs.size + chunk_info_rhs.size};
  }

  using TwoWayMerge = hyrise::simd_sort::TwoWayMerge<count_per_vector, T>;
  TwoWayMerge::template merge_variable_length<count_per_vector * 4>(
      chunk_info_lhs.input, chunk_info_rhs.input, chunk_info_lhs.output, chunk_info_lhs.size, chunk_info_rhs.size);

  return {chunk_info_lhs.output, chunk_info_lhs.input, chunk_info_lhs.size + chunk_info_rhs.size};
}

template <size_t count_per_vector, bool use_merge_path = true, typename T>
T* simd_merge_parallel(std::vector<simd_sort::DataChunk<T>>& chunk_list, size_t core_count) {
  const auto merge_path_end_level = simd_sort::log2_builtin(std::bit_floor(core_count));

  if (!use_merge_path || !merge_path_end_level) {
    const auto final_chunk = std::move(merge_recursive<count_per_vector, T>(chunk_list));
    return final_chunk.input;
  }

  // Use recursive SIMD merging with MergePath on the last levels.
  const auto final_chunk =
      std::move(merge_recursive_with_merge_path<count_per_vector, T>(chunk_list, merge_path_end_level, core_count));
  return final_chunk.input;
}

template <std::size_t count_per_vector, typename T>
inline void __attribute__((always_inline)) sort_incomplete_chunk(DataChunk<T>& chunk) {
  constexpr auto NORMAL_SORT_THRESHOLD_SIZE = 128;
  auto next_possible_smaller_blocksize = block_size<T>() / 2;
  auto num_remaining_items = chunk.size;

  auto find_next_possible_blocksize = [&]() {
    while (next_possible_smaller_blocksize > num_remaining_items) {
      next_possible_smaller_blocksize >>= 1;
    }
    return next_possible_smaller_blocksize;
  };

  auto chunk_list = std::vector<DataChunk<T>>();
  chunk_list.reserve(32);
  auto chunk_count = std::size_t{0};

  auto offset = std::size_t{0};
  next_possible_smaller_blocksize = find_next_possible_blocksize();

  // Split block into simd sortable blocks of smaller size and sort them.
  while (next_possible_smaller_blocksize > NORMAL_SORT_THRESHOLD_SIZE) {
    chunk_list.emplace_back(chunk.input + offset, chunk.output + offset, next_possible_smaller_blocksize);
    auto& chunk_info = chunk_list.back();
    sort_chunk<count_per_vector>(chunk_info);
    std::swap(chunk_info.input, chunk_info.output);

    offset += next_possible_smaller_blocksize;
    num_remaining_items -= next_possible_smaller_blocksize;
    next_possible_smaller_blocksize = find_next_possible_blocksize();
    ++chunk_count;
  }

  // Sort last chunk of remaining items with boost::pdqsort.
  if (num_remaining_items) {
    chunk_list.emplace_back(chunk.input + offset, chunk.output + offset, num_remaining_items);
    auto& chunk_info = chunk_list.back();
    boost::sort::pdqsort(chunk_info.input, chunk_info.input + num_remaining_items);
    ++chunk_count;
  }
  // Merge sorted chunks into one sorted list.
  while (chunk_count > 1) {
    chunk_count = merge_chunk_list<count_per_vector>(chunk_list, chunk_count);
  }
  auto& merged_chunk = chunk_list.front();
  chunk.input = merged_chunk.output;
  chunk.output = merged_chunk.input;
}

template <std::size_t count_per_vector, typename T,
          ExecutionStrategy execution_strategy = ExecutionStrategy::SEQUENTIAL>
void sort(T*& input_ptr, T*& output_ptr, std::size_t element_count) {
  if (element_count <= 0) [[unlikely]] {
    return;
  }
  constexpr auto BLOCK_SIZE = block_size<T>();
  auto* input = input_ptr;
  auto* output = output_ptr;

  // We split our data into blocks of size BLOCK_SIZE and compute the bounds for
  // each block.
  const auto remaining_items = element_count % BLOCK_SIZE;
  auto chunk_count = (element_count / BLOCK_SIZE) + (remaining_items > 0);

  auto chunk_list = std::vector<DataChunk<T>>{};
  chunk_list.reserve(chunk_count);

  for (auto block_index = std::size_t{0}; block_index < chunk_count; ++block_index) {
    const auto offset = block_index * BLOCK_SIZE;
    chunk_list.emplace_back(input + offset, output + offset, BLOCK_SIZE);
  }

  // We then call our local sort routine for each block.
  const auto chunk_count_without_remaining = chunk_count - (remaining_items > 0);
  if constexpr (execution_strategy == ExecutionStrategy::PARALLEL) {
    auto sort_tasks = std::vector<std::shared_ptr<AbstractTask>>{};
    sort_tasks.reserve(chunk_count_without_remaining);

    for (auto chunk_index = std::size_t{0}; chunk_index < chunk_count_without_remaining; ++chunk_index) {
      sort_tasks.push_back(std::make_shared<JobTask>([&, chunk_index]() {
        auto& chunk = chunk_list[chunk_index];
        sort_chunk<count_per_vector>(chunk);
        std::swap(chunk.input, chunk.output);
      }));
    }

    Hyrise::get().scheduler()->schedule_and_wait_for_tasks(sort_tasks);

  } else {
    for (auto chunk_index = std::size_t{0}; chunk_index < chunk_count_without_remaining; ++chunk_index) {
      auto& chunk = chunk_list[chunk_index];
      sort_chunk<count_per_vector>(chunk);
      std::swap(chunk.input, chunk.output);
    }
  }

  if (remaining_items) {
    auto& chunk = chunk_list.back();
    chunk.size = remaining_items;
    sort_incomplete_chunk<count_per_vector>(chunk);
    std::swap(chunk.input, chunk.output);
  }
  // Next we merge all these chunks iteratively to achieve a global sorting.
  const auto log_n = static_cast<std::size_t>(std::ceil(std::log2(element_count)));
  const auto log_block_size = log2_builtin(BLOCK_SIZE);

  for (auto level_index = log_block_size; level_index < log_n; ++level_index) {
    if constexpr (execution_strategy == ExecutionStrategy::PARALLEL) {
      chunk_count = parallel_merge_chunk_list<count_per_vector>(chunk_list, chunk_count);
    } else {
      chunk_count = merge_chunk_list<count_per_vector>(chunk_list, chunk_count);
    }
  }

  auto& merged_chunk = chunk_list.front();
  output_ptr = merged_chunk.input;
  input_ptr = merged_chunk.output;
}
}  // namespace hyrise::simd_sort
