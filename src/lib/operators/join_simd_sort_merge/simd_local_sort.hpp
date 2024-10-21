#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>

#include <boost/sort/sort.hpp>

#include "simd_utils.hpp"
#include "two_way_merge.hpp"

namespace hyrise {

template <typename T>
struct DataChunk {
  T* input;
  T* output;
  std::size_t size;
};

template <std::size_t count_per_register, std::size_t kernel_size, typename T>
void merge_level(std::size_t level, std::array<T*, 2>& ptrs, std::size_t num_items) {
  auto ptr_index = level & 1u;
  auto* input = ptrs[ptr_index];
  auto* output = ptrs[ptr_index ^ 1u];
  auto* const end = input + num_items;

  const auto input_length = 1u << level;          // = 2^level
  const auto output_length = input_length << 1u;  // = input_length x 2
  using TwoWayMerge = TwoWayMerge<count_per_register, T>;
  while (input < end) {
    TwoWayMerge::template merge_equal_length<kernel_size>(input, input + input_length, output, input_length);
    input += output_length;
    output += output_length;
  }
}

template <std::size_t count_per_register, typename T>
inline void __attribute__((always_inline)) simd_sort_chunk(DataChunk<T>& block) {
  const auto num_items = block.size;
  const auto start_level = log2_builtin(count_per_register);

  auto input_output_pointers = std::array<T*, 2>{};
  auto input_selection_index = start_level & 1u;
  input_output_pointers[input_selection_index] = block.input;
  input_output_pointers[input_selection_index ^ 1u] = block.output;
  {
    using block_t = struct alignas(sizeof(T) * count_per_register * count_per_register) {};

    auto* block_start_address = reinterpret_cast<block_t*>(block.input);
    auto* const block_end_address = reinterpret_cast<block_t*>(block.input + num_items);
    using SortingNetwork = SortingNetwork<count_per_register, T>;
    while (block_start_address < block_end_address) {
      SortingNetwork::sort(reinterpret_cast<T*>(block_start_address), reinterpret_cast<T*>(block_start_address));
      ++block_start_address;
    }
  }
  const auto log_block_size = log2_builtin(num_items);
  const auto stop_level = log_block_size - 2;
  merge_level<count_per_register, count_per_register>(start_level, input_output_pointers, num_items);
  merge_level<count_per_register, count_per_register * 2>(start_level + 1, input_output_pointers, num_items);
#pragma unroll
  for (auto level = std::size_t{start_level + 2}; level < stop_level; ++level) {
    merge_level<count_per_register, count_per_register * 4>(level, input_output_pointers, num_items);
  }

  auto input_length = 1u << stop_level;
  input_selection_index = stop_level & 1u;
  auto* input = input_output_pointers[input_selection_index];
  auto* output = input_output_pointers[input_selection_index ^ 1u];

  using TwoWayMerge = TwoWayMerge<count_per_register, T>;
  TwoWayMerge::template merge_equal_length<count_per_register * 4>(input, input + input_length, output, input_length);
  TwoWayMerge::template merge_equal_length<count_per_register * 4>(input + 2 * input_length, input + 3 * input_length,
                                                                   output + 2 * input_length, input_length);
  input_length <<= 1u;
  TwoWayMerge::template merge_equal_length<count_per_register * 4>(output, output + input_length, input, input_length);
  block.input = output;
  block.output = input;
}

template <std::size_t count_per_register, typename T>
inline std::size_t __attribute__((always_inline)) simd_merge_chunk_list(std::vector<DataChunk<T>>& chunk_list,
                                                                        std::size_t chunk_count) {
  using TwoWayMerge = TwoWayMerge<count_per_register, T>;
  auto updated_chunk_count = std::size_t{0};
  const auto last_chunk_index = chunk_count - 1;
  for (auto chunk_index = std::size_t{0}; chunk_index < last_chunk_index; chunk_index += 2) {
    const auto& chunk_info_a = chunk_list[chunk_index];
    const auto& chunk_info_b = chunk_list[chunk_index + 1];
    TwoWayMerge::template merge_variable_length<count_per_register * 4>(
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

template <std::size_t count_per_register, typename T>
inline void __attribute__((always_inline)) simd_sort_incomplete_chunk(DataChunk<T>& chunk) {
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
    simd_sort_chunk<count_per_register>(chunk_info);
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
    chunk_count = simd_merge_chunk_list<count_per_register>(chunk_list, chunk_count);
  }
  auto& merged_chunk = chunk_list.front();
  chunk.input = merged_chunk.output;
  chunk.output = merged_chunk.input;
}

template <std::size_t count_per_register, typename T>
void simd_sort(T*& input_ptr, T*& output_ptr, std::size_t element_count) {
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
  for (auto chunk_index = std::size_t{0}; chunk_index < chunk_count_without_remaining; ++chunk_index) {
    auto& chunk = chunk_list[chunk_index];
    simd_sort_chunk<count_per_register>(chunk);
    std::swap(chunk.input, chunk.output);
  }
  if (remaining_items) {
    auto& chunk = chunk_list.back();
    chunk.size = remaining_items;
    simd_sort_incomplete_chunk<count_per_register>(chunk);
    std::swap(chunk.input, chunk.output);
  }
  // Next we merge all these chunks iteratively to achieve a global sorting.
  const auto log_n = static_cast<std::size_t>(std::ceil(std::log2(element_count)));
  const auto log_block_size = log2_builtin(BLOCK_SIZE);
  for (auto level_index = log_block_size; level_index < log_n; ++level_index) {
    chunk_count = simd_merge_chunk_list<count_per_register>(chunk_list, chunk_count);
  }
  auto& merged_chunk = chunk_list.front();
  output_ptr = merged_chunk.input;
  input_ptr = merged_chunk.output;
}
}  // namespace hyrise
