#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "simd_utils.hpp"
#include "two_way_merge.hpp"

namespace hyrise {

template <typename VecType>
static inline void __attribute__((always_inline)) compare_min_max(VecType& input1, VecType& input2) {
  // NOLINTBEGIN(cppcoreguidelines-pro-type-vararg, hicpp-vararg)
  auto min = __builtin_elementwise_min(input1, input2);
  auto max = __builtin_elementwise_max(input1, input2);
  // NOLINTEND(cppcoreguidelines-pro-type-vararg, hicpp-vararg)
  input1 = min;
  input2 = max;
}

template <size_t elements_per_register, typename T>
struct SortingNetwork {
  static inline void __attribute__((always_inline)) sort(T* /*data*/, T* /*output*/) {
    static_assert(false, "Not implemented.");
  }
};

template <typename T>
struct SortingNetwork<2, T> {
  static inline void __attribute__((always_inline)) sort(T* data, T* output) {
    constexpr auto COUNT_PER_REGISTER = 2;
    constexpr auto REGISTER_SIZE = COUNT_PER_REGISTER * sizeof(T);
    using VecType = Vec<REGISTER_SIZE, T>;

    auto row_0 = load_aligned<VecType>(data);
    auto row_1 = load_aligned<VecType>(data + COUNT_PER_REGISTER);

    // Level 1 comparisons.
    compare_min_max(row_0, row_1);

    // Transpose Matrix
    auto out_1 = __builtin_shufflevector(row_0, row_1, 0, 2);
    auto out_2 = __builtin_shufflevector(row_0, row_1, 1, 3);
    // Write to output
    store_aligned(out_1, output);
    store_aligned(out_2, output + COUNT_PER_REGISTER);
  }
};

template <typename T>
struct SortingNetwork<4, T> {
  static inline void __attribute__((always_inline)) sort(T* data, T* output) {
    constexpr auto COUNT_PER_REGISTER = 4;
    constexpr auto REGISTER_SIZE = COUNT_PER_REGISTER * sizeof(T);
    using VecType = Vec<REGISTER_SIZE, T>;

    auto row_0 = load_aligned<VecType>(data);
    auto row_1 = load_aligned<VecType>(data + COUNT_PER_REGISTER);
    auto row_2 = load_aligned<VecType>(data + 2 * COUNT_PER_REGISTER);
    auto row_3 = load_aligned<VecType>(data + 3 * COUNT_PER_REGISTER);

    // Level 1 comparisons.
    compare_min_max(row_0, row_2);
    compare_min_max(row_1, row_3);
    // Level 2 comparisons.
    compare_min_max(row_0, row_1);
    compare_min_max(row_2, row_3);
    // Level 3 comparisons.
    compare_min_max(row_1, row_2);

    // Transpose Matrix
    auto ab_interleaved_lower_halves = __builtin_shufflevector(row_0, row_1, INTERLEAVE_LOWERS);
    auto ab_interleaved_upper_halves = __builtin_shufflevector(row_0, row_1, INTERLEAVE_UPPERS);
    auto cd_interleaved_lower_halves = __builtin_shufflevector(row_2, row_3, INTERLEAVE_LOWERS);
    auto cd_interleaved_upper_halves = __builtin_shufflevector(row_2, row_3, INTERLEAVE_UPPERS);
    row_0 = __builtin_shufflevector(ab_interleaved_lower_halves, cd_interleaved_lower_halves, LOWER_HALVES);
    row_1 = __builtin_shufflevector(ab_interleaved_lower_halves, cd_interleaved_lower_halves, UPPER_HALVES);
    row_2 = __builtin_shufflevector(ab_interleaved_upper_halves, cd_interleaved_upper_halves, LOWER_HALVES);
    row_3 = __builtin_shufflevector(ab_interleaved_upper_halves, cd_interleaved_upper_halves, UPPER_HALVES);

    // Write to output
    store_aligned(row_0, output);
    store_aligned(row_1, output + COUNT_PER_REGISTER);
    store_aligned(row_2, output + 2 * COUNT_PER_REGISTER);
    store_aligned(row_3, output + 3 * COUNT_PER_REGISTER);
  }
};

// https://stackoverflow.com/questions/35311711/whats-the-right-way-to-compute-integral-base-2-logarithms-at-compile-time
template <typename T>
constexpr size_t cilog2(T val) {
  return (val != 0u) ? 1 + cilog2(val >> 1u) : -1;
}

template <size_t count_per_register, size_t kernel_size, typename T>
void merge_level(size_t level, std::array<T*, 2>& ptrs) {
  constexpr auto BLOCK_SIZE = block_size<T>();
  auto ptr_index = level & 1u;
  auto* input = ptrs[ptr_index];
  auto* output = ptrs[ptr_index ^ 1u];
  auto* const end = input + BLOCK_SIZE;

  const auto input_length = 1u << level;          // = 2^level
  const auto output_length = input_length << 1u;  // = input_length x 2
  using TwoWayMerge = TwoWayMerge<count_per_register, T>;
  while (input < end) {
    TwoWayMerge::template merge_equal_length<kernel_size>(input, input + input_length, output, input_length);
    input += output_length;
    output += output_length;
  }
}

template <size_t count_per_register, typename T>
inline void __attribute__((always_inline)) simd_sort_block(T*& input_ptr, T*& output_ptr) {
  constexpr auto BLOCK_SIZE = block_size<T>();
  constexpr auto START_LEVEL = cilog2(count_per_register);

  auto ptrs = std::array<T*, 2>{};
  auto pointer_index = START_LEVEL & 1u;
  ptrs[pointer_index] = input_ptr;
  ptrs[pointer_index ^ 1u] = output_ptr;
  {
    using block_t = struct alignas(sizeof(T) * count_per_register * count_per_register) {};

    auto* block_start_address = reinterpret_cast<block_t*>(input_ptr);
    auto* const block_end_address = reinterpret_cast<block_t*>(input_ptr + BLOCK_SIZE);
    using SortingNetwork = SortingNetwork<count_per_register, T>;
    while (block_start_address < block_end_address) {
      SortingNetwork::sort(reinterpret_cast<T*>(block_start_address), reinterpret_cast<T*>(block_start_address));
      ++block_start_address;
    }
  }
  constexpr auto LOG_BLOCK_SIZE = cilog2(BLOCK_SIZE);
  constexpr auto STOP_LEVEL = LOG_BLOCK_SIZE - 2;
  merge_level<count_per_register, count_per_register>(START_LEVEL, ptrs);
  merge_level<count_per_register, count_per_register * 2>(START_LEVEL + 1, ptrs);
#pragma unroll
  for (auto level = size_t{START_LEVEL + 2}; level < STOP_LEVEL; ++level) {
    merge_level<count_per_register, count_per_register * 4>(level, ptrs);
  }

  auto input_length = 1u << STOP_LEVEL;
  pointer_index = STOP_LEVEL & 1u;
  auto* input = ptrs[pointer_index];
  auto* output = ptrs[pointer_index ^ 1u];

  using TwoWayMerge = TwoWayMerge<count_per_register, T>;
  TwoWayMerge::template merge_equal_length<count_per_register * 4>(input, input + input_length, output, input_length);
  TwoWayMerge::template merge_equal_length<count_per_register * 4>(input + 2 * input_length, input + 3 * input_length,
                                                                   output + 2 * input_length, input_length);
  input_length <<= 1u;
  TwoWayMerge::template merge_equal_length<count_per_register * 4>(output, output + input_length, input, input_length);
  input_ptr = output;
  output_ptr = input;
}

template <size_t count_per_register, typename T>
inline void __attribute__((always_inline)) simd_sort_incomplete_block(T*& input_ptr, T*& output_ptr) {
  // TODO(finn): Implement.
}

template <typename T>
struct BlockInfo {
  T* input;
  T* output;
  size_t size;
};

template <size_t count_per_register, typename T>
void simd_sort(T*& input_ptr, T*& output_ptr, size_t element_count) {
  if (element_count <= 0) [[unlikely]] {
    return;
  }

  constexpr auto BLOCK_SIZE = block_size<T>();
  auto* input = input_ptr;
  auto* output = output_ptr;

  // We split our data into blocks of size BLOCK_SIZE and compute the bounds for
  // each block.
  auto block_count = element_count / BLOCK_SIZE;
  const auto remaining_items = element_count % BLOCK_SIZE;
  auto block_infos = std::vector<BlockInfo<T>>{};
  block_infos.reserve(block_count + (remaining_items > 0));

  for (auto block_index = size_t{0}; block_index < block_count; ++block_index) {
    const auto offset = block_index * BLOCK_SIZE;
    block_infos.emplace_back(input + offset, output + offset, BLOCK_SIZE);
  }

  // We then call our local sort routine for each block.
  for (auto block_index = size_t{0}; block_index < block_count; ++block_index) {
    auto& block_info = block_infos[block_index];
    simd_sort_block<count_per_register>(block_info.input, block_info.output);
    std::swap(block_info.input, block_info.output);
  }
  if (remaining_items) {
    // auto& block_info = block_infos.back();
    // TODO(finn): Implement simd_sort for remaining items.
    assert(false && "Sort for remaining values not implemented.");
  }
  block_count += remaining_items > 0;

  // Next we merge all these chunks iteratively to achieve a global sorting.
  const auto log_n = static_cast<size_t>(std::ceil(std::log2(element_count)));
  constexpr auto LOG_BLOCK_SIZE = cilog2(BLOCK_SIZE);

  using TwoWayMerge = TwoWayMerge<count_per_register, T>;

  for (auto index = LOG_BLOCK_SIZE; index < log_n; ++index) {
    auto updated_block_count = size_t{0};
    for (auto block_index = size_t{0}; block_index < block_count - 1; block_index += 2) {
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
    if (block_count % 2) {
      block_infos[updated_block_count] = block_infos[block_count - 1];
      ++updated_block_count;
    }
    block_count = updated_block_count;
  }
  output_ptr = block_infos[0].input;
  input_ptr = block_infos[0].output;
}
}  // namespace hyrise
