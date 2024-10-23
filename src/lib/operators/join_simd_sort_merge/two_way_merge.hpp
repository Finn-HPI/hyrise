#pragma once

#include "abstract_two_way_merge.hpp"
#include "simd_utils.hpp"

namespace hyrise::simd_sort {

template <std::size_t count_per_vector, typename T>
class TwoWayMerge : public AbstractTwoWayMerge<count_per_vector, T, TwoWayMerge<count_per_vector, T>> {
  TwoWayMerge() : AbstractTwoWayMerge<count_per_vector, T, TwoWayMerge<count_per_vector, T>>() {
    static_assert(false, "Not implemented.");
  }
};

// NOLINTBEGIN(cppcoreguidelines-pro-type-vararg, hicpp-vararg)

template <typename T>
class TwoWayMerge<2, T> : public AbstractTwoWayMerge<2, T, TwoWayMerge<2, T>> {
  static constexpr auto COUNT_PER_VECTOR = 2;
  static constexpr auto VECTOR_SIZE = COUNT_PER_VECTOR * sizeof(T);
  using VecType = Vec<VECTOR_SIZE, T>;

 public:
  static inline void __attribute__((always_inline)) reverse(VecType& vec) {
    vec = __builtin_shufflevector(vec, vec, 1, 0);
  }

  static inline void __attribute__((always_inline)) merge_network_input_x2(VecType& input_a, VecType& input_b,
                                                                           VecType& out1, VecType& out2) {
    // Level 1
    auto low1 = __builtin_elementwise_min(input_a, input_b);
    auto high1 = __builtin_elementwise_max(input_a, input_b);
    auto permutated_low1 = __builtin_shufflevector(low1, high1, 0, 2);
    auto permutated_high1 = __builtin_shufflevector(low1, high1, 1, 3);
    // Level 2
    auto low2 = __builtin_elementwise_min(permutated_low1, permutated_high1);
    auto high2 = __builtin_elementwise_max(permutated_low1, permutated_high1);
    out1 = __builtin_shufflevector(low2, high2, 0, 2);
    out2 = __builtin_shufflevector(low2, high2, 1, 3);
  }

  static inline void __attribute__((always_inline)) merge_network_input_x4(VecType& in11, VecType& in12, VecType& in21,
                                                                           VecType& in22, VecType& out1, VecType& out2,
                                                                           VecType& out3, VecType& out4) {
    auto l11 = __builtin_elementwise_min(in11, in21);
    auto h11 = __builtin_elementwise_max(in11, in21);
    auto l12 = __builtin_elementwise_min(in12, in22);
    auto h12 = __builtin_elementwise_max(in12, in22);
    merge_network_input_x2(l11, l12, out1, out2);
    merge_network_input_x2(h11, h12, out3, out4);
  }
};

template <typename T>
class TwoWayMerge<4, T> : public AbstractTwoWayMerge<4, T, TwoWayMerge<4, T>> {
  static constexpr auto COUNT_PER_VECTOR = 4;
  static constexpr auto VECTOR_SIZE = COUNT_PER_VECTOR * sizeof(T);
  using VecType = Vec<VECTOR_SIZE, T>;

 public:
  static inline void __attribute__((always_inline)) reverse(VecType& vec) {
    vec = __builtin_shufflevector(vec, vec, 3, 2, 1, 0);
  }

  static inline void __attribute__((always_inline)) merge_network_input_x2(VecType& input_a, VecType& input_b,
                                                                           VecType& out1, VecType& out2) {
    // Level 1
    auto lo1 = __builtin_elementwise_min(input_a, input_b);
    auto hi1 = __builtin_elementwise_max(input_a, input_b);
    auto lo1_perm = __builtin_shufflevector(lo1, hi1, LOWER_HALVES);
    auto hi1_perm = __builtin_shufflevector(lo1, hi1, UPPER_HALVES);
    // Level 2
    auto lo2 = __builtin_elementwise_min(lo1_perm, hi1_perm);
    auto hi2 = __builtin_elementwise_max(lo1_perm, hi1_perm);
    auto lo2_perm = __builtin_shufflevector(lo2, hi2, 0, 4, 2, 6);
    auto hi2_perm = __builtin_shufflevector(lo2, hi2, 1, 5, 3, 7);
    // Level 3
    auto lo3 = __builtin_elementwise_min(lo2_perm, hi2_perm);
    auto hi3 = __builtin_elementwise_max(lo2_perm, hi2_perm);

    out1 = __builtin_shufflevector(lo3, hi3, 0, 4, 1, 5);
    out2 = __builtin_shufflevector(lo3, hi3, 2, 6, 3, 7);
  }

  static inline void __attribute__((always_inline)) merge_network_input_x4(VecType& in11, VecType& in12, VecType& in21,
                                                                           VecType& in22, VecType& out1, VecType& out2,
                                                                           VecType& out3, VecType& out4) {
    // NOLINTBEGIN
    auto l11 = __builtin_elementwise_min(in11, in21);
    auto l12 = __builtin_elementwise_min(in12, in22);
    auto h11 = __builtin_elementwise_max(in11, in21);
    auto h12 = __builtin_elementwise_max(in12, in22);
    // NOLINTEND
    merge_network_input_x2(l11, l12, out1, out2);
    merge_network_input_x2(h11, h12, out3, out4);
  }
};

template <typename T>
class TwoWayMerge<8, T> : public AbstractTwoWayMerge<8, T, TwoWayMerge<8, T>> {
  static constexpr auto COUNT_PER_VECTOR = 8;
  static constexpr auto VECTOR_SIZE = COUNT_PER_VECTOR * sizeof(T);
  using VecType = Vec<VECTOR_SIZE, T>;

 public:
  static inline void __attribute__((always_inline)) reverse(VecType& vec) {
    vec = __builtin_shufflevector(vec, vec, 7, 6, 5, 4, 3, 2, 1, 0);
  }

  static inline void __attribute__((always_inline)) merge_network_input_x2(VecType& input_a, VecType& input_b,
                                                                           VecType& out1, VecType& out2) {
    // Level 1
    auto lo1 = __builtin_elementwise_min(input_a, input_b);
    auto hi1 = __builtin_elementwise_max(input_a, input_b);
    // 0 1 2 3 4 5 6 7   8 9 10 11 12 13 14 15
    auto lo1_perm = __builtin_shufflevector(lo1, hi1, 0, 1, 2, 3, 8, 9, 10, 11);
    auto hi1_perm = __builtin_shufflevector(lo1, hi1, 4, 5, 6, 7, 12, 13, 14, 15);
    // Level 2
    auto lo2 = __builtin_elementwise_min(lo1_perm, hi1_perm);
    auto hi2 = __builtin_elementwise_max(lo1_perm, hi1_perm);
    auto lo2_perm = __builtin_shufflevector(lo2, hi2, 0, 1, 8, 9, 4, 5, 12, 13);
    auto hi2_perm = __builtin_shufflevector(lo2, hi2, 2, 3, 10, 11, 6, 7, 14, 15);
    // Level 3
    auto lo3 = __builtin_elementwise_min(lo2_perm, hi2_perm);
    auto hi3 = __builtin_elementwise_max(lo2_perm, hi2_perm);
    auto lo3_perm = __builtin_shufflevector(lo3, hi3, 0, 8, 2, 10, 4, 12, 6, 14);
    auto hi3_perm = __builtin_shufflevector(lo3, hi3, 1, 9, 3, 11, 5, 13, 7, 15);
    // Level 4
    auto lo4 = __builtin_elementwise_min(lo3_perm, hi3_perm);
    auto hi4 = __builtin_elementwise_max(lo3_perm, hi3_perm);
    out1 = __builtin_shufflevector(lo4, hi4, 0, 8, 1, 9, 2, 10, 3, 11);
    out2 = __builtin_shufflevector(lo4, hi4, 4, 12, 5, 13, 6, 14, 7, 15);
  }

  static inline void __attribute__((always_inline)) merge_network_input_x4(VecType& in11, VecType& in12, VecType& in21,
                                                                           VecType& in22, VecType& out1, VecType& out2,
                                                                           VecType& out3, VecType& out4) {
    auto l11 = __builtin_elementwise_min(in11, in21);
    auto l12 = __builtin_elementwise_min(in12, in22);
    auto h11 = __builtin_elementwise_max(in11, in21);
    auto h12 = __builtin_elementwise_max(in12, in22);
    merge_network_input_x2(l11, l12, out1, out2);
    merge_network_input_x2(h11, h12, out3, out4);
  }
};

// NOLINTEND(cppcoreguidelines-pro-type-vararg, hicpp-vararg)
}  // namespace hyrise::simd_sort
