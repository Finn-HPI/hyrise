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
struct BlockInfo {
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
inline void __attribute__((always_inline))
simd_sort_block(T*& input_ptr, T*& output_ptr, const std::size_t num_items = block_size<T>()) {
  const auto start_level = log2_builtin(count_per_register);

  auto input_output_pointers = std::array<T*, 2>{};
  auto input_selection_index = start_level & 1u;
  input_output_pointers[input_selection_index] = input_ptr;
  input_output_pointers[input_selection_index ^ 1u] = output_ptr;
  {
    using block_t = struct alignas(sizeof(T) * count_per_register * count_per_register) {};

    auto* block_start_address = reinterpret_cast<block_t*>(input_ptr);
    auto* const block_end_address = reinterpret_cast<block_t*>(input_ptr + num_items);
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
  input_ptr = output;
  output_ptr = input;
}

template <std::size_t count_per_register, typename T>
inline void __attribute__((always_inline)) simd_sort_incomplete_block(T*& input_ptr, T*& output_ptr, std::size_t size) {
  constexpr auto NORMAL_SORT_THRESHOLD_SIZE = 128;
  auto next_possible_smaller_blocksize = block_size<T>() / 2;
  auto num_remaining_items = size;

  auto find_next_possible_blocksize = [&]() {
    while (next_possible_smaller_blocksize > num_remaining_items) {
      next_possible_smaller_blocksize >>= 1;
    }
    return next_possible_smaller_blocksize;
  };

  auto chunk_infos = std::vector<BlockInfo<T>>();
  chunk_infos.reserve(32);

  auto offset = std::size_t{0};
  auto chunk_count = std::size_t{0};
  next_possible_smaller_blocksize = find_next_possible_blocksize();

  // Split block into simd sortable blocks of smaller size and sort them.
  while (next_possible_smaller_blocksize > NORMAL_SORT_THRESHOLD_SIZE) {
    chunk_infos.emplace_back(input_ptr + offset, output_ptr + offset, next_possible_smaller_blocksize);
    auto& chunk_info = chunk_infos.back();
    simd_sort_block<count_per_register>(chunk_info.input, chunk_info.output, chunk_info.size);

    std::swap(chunk_info.input, chunk_info.output);

    offset += next_possible_smaller_blocksize;
    num_remaining_items -= next_possible_smaller_blocksize;
    next_possible_smaller_blocksize = find_next_possible_blocksize();
    ++chunk_count;
  }

  // Sort last chunk of remaining items with boost::pdqsort.
  if (num_remaining_items) {
    chunk_infos.emplace_back(input_ptr + offset, output_ptr + offset, num_remaining_items);
    auto& chunk_info = chunk_infos.back();
    boost::sort::pdqsort(chunk_info.input, chunk_info.input + num_remaining_items);
    ++chunk_count;
  }

  // Merge sorted chunks into one sorted list.
  using TwoWayMerge = TwoWayMerge<count_per_register, T>;
  while (chunk_count > 1) {
    auto updated_chunk_count = std::size_t{0};
    const auto last_chunk_index = chunk_count - 1;
    for (auto chunk_index = std::size_t{0}; chunk_index < last_chunk_index; chunk_index += 2) {
      auto& chunk_info_a = chunk_infos[chunk_index];
      auto& chunk_info_b = chunk_infos[chunk_index + 1];
      TwoWayMerge::template merge_variable_length<count_per_register * 4>(
          chunk_info_a.input, chunk_info_b.input, chunk_info_a.output, chunk_info_a.size, chunk_info_b.size);
      chunk_infos[updated_chunk_count] = {chunk_info_a.output, chunk_info_a.input,
                                          chunk_info_a.size + chunk_info_b.size};
      ++updated_chunk_count;
    }
    // If we had odd many blocks, we have one additional unmerged block for the next iteration.
    if (chunk_count % 2) {
      chunk_infos[updated_chunk_count] = chunk_infos[chunk_count - 1];
      ++updated_chunk_count;
    }
    chunk_count = updated_chunk_count;
  }
  auto& last_used_chunk_info = chunk_infos.front();
  output_ptr = last_used_chunk_info.input;
  input_ptr = last_used_chunk_info.output;
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
  auto block_count = (element_count / BLOCK_SIZE) + (remaining_items > 0);

  auto block_infos = std::vector<BlockInfo<T>>{};
  block_infos.reserve(block_count);

  for (auto block_index = std::size_t{0}; block_index < block_count; ++block_index) {
    const auto offset = block_index * BLOCK_SIZE;
    block_infos.emplace_back(input + offset, output + offset, BLOCK_SIZE);
  }

  // We then call our local sort routine for each block.
  const auto block_count_without_remaining = block_count - (remaining_items > 0);
  for (auto block_index = std::size_t{0}; block_index < block_count_without_remaining; ++block_index) {
    auto& block_info = block_infos[block_index];
    simd_sort_block<count_per_register>(block_info.input, block_info.output);
    std::swap(block_info.input, block_info.output);
  }

  if (remaining_items) {
    auto& block_info = block_infos.back();
    block_info.size = remaining_items;
    simd_sort_incomplete_block<count_per_register>(block_info.input, block_info.output, block_info.size);
    std::swap(block_info.input, block_info.output);
  }

  // Next we merge all these chunks iteratively to achieve a global sorting.
  const auto log_n = static_cast<std::size_t>(std::ceil(std::log2(element_count)));
  const auto log_block_size = log2_builtin(BLOCK_SIZE);

  using TwoWayMerge = TwoWayMerge<count_per_register, T>;
  for (auto level_index = log_block_size; level_index < log_n; ++level_index) {
    auto updated_block_count = std::size_t{0};
    for (auto block_index = std::size_t{0}; block_index < block_count - 1; block_index += 2) {
      auto& a_info = block_infos[block_index];
      auto& b_info = block_infos[block_index + 1];
      auto* input_a = a_info.input;
      auto* input_b = b_info.input;
      auto* out = a_info.output;
      const auto a_size = a_info.size;
      const auto b_size = b_info.size;

      TwoWayMerge::template merge_variable_length<count_per_register * 4>(input_a, input_b, out, a_size, b_size);
      block_infos[updated_block_count] = {out, input_a, a_size + b_size};
      ++updated_block_count;
    }
    // If we had odd many blocks, we have one additional unmerged block for the next iteration.
    if (block_count % 2) {
      block_infos[updated_block_count] = block_infos[block_count - 1];
      ++updated_block_count;
    }
    block_count = updated_block_count;
  }
  auto& last_used_block_info = block_infos.front();
  output_ptr = last_used_block_info.input;
  input_ptr = last_used_block_info.output;
}
}  // namespace hyrise
