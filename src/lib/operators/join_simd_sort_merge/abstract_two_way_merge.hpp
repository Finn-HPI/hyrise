#pragma once
#include <cstddef>

#include "simd_utils.hpp"

template <size_t count_per_register, typename T, typename Derived>
class AbstractTwoWayMerge {
  static constexpr auto REGISTER_SIZE = count_per_register * sizeof(T);
  using VecType = Vec<REGISTER_SIZE, T>;

 protected:
  template <size_t kernel_size, typename MulitVecType>
  struct BitonicMergeNetwork {
    static inline void __attribute__((always_inline)) merge(MulitVecType& /*in1*/, MulitVecType& /*in2*/,
                                                            MulitVecType& /*out1*/, MulitVecType& /*out2*/) {
      static_assert(false, "Not implemented.");
    }
  };

 public:
  template <size_t kernel_size>
  static inline void __attribute__((always_inline)) merge_equal_length(T* const a_address, T* const b_address,
                                                                       T* const output_address, const size_t length) {
    using block_t = struct alignas(kernel_size * sizeof(T)) {};

    static constexpr auto REGISTER_COUNT = kernel_size / count_per_register;

    using MultiVecType = MultiVec<REGISTER_COUNT, count_per_register, VecType>;
    using BitonicMergeNetwork = typename Derived::template BitonicMergeNetwork<kernel_size, MultiVecType>;

    auto* a_pointer = reinterpret_cast<block_t*>(a_address);
    auto* b_pointer = reinterpret_cast<block_t*>(b_address);
    auto* const a_end = reinterpret_cast<block_t*>(a_address + length);
    auto* const b_end = reinterpret_cast<block_t*>(b_address + length);

    auto* output_pointer = reinterpret_cast<block_t*>(output_address);
    auto* next_pointer = b_pointer;

    auto a_input = MultiVecType{};
    auto b_input = MultiVecType{};
    auto lower_merge_output = MultiVecType{};
    auto upper_merge_output = MultiVecType{};
    a_input.load(a_address);
    b_input.load(b_address);
    ++a_pointer;
    ++b_pointer;

    BitonicMergeNetwork::merge(a_input, b_input, lower_merge_output, upper_merge_output);
    lower_merge_output.store(reinterpret_cast<T*>(output_pointer));
    ++output_pointer;

    // As long as both A and B are not empty, do fetch and 2x4 merge.
    while (a_pointer < a_end && b_pointer < b_end) {
      choose_next_and_update_pointers<block_t, T>(next_pointer, a_pointer, b_pointer);
      a_input = upper_merge_output;
      b_input.load(reinterpret_cast<T*>(next_pointer));
      BitonicMergeNetwork::merge(a_input, b_input, lower_merge_output, upper_merge_output);
      lower_merge_output.store(reinterpret_cast<T*>(output_pointer));
      ++output_pointer;
    }
    // If A not empty, merge remainder of A.
    while (a_pointer < a_end) {
      a_input.load(reinterpret_cast<T*>(a_pointer));
      b_input = upper_merge_output;
      BitonicMergeNetwork::merge(a_input, b_input, lower_merge_output, upper_merge_output);
      lower_merge_output.store(reinterpret_cast<T*>(output_pointer));
      ++a_pointer;
      ++output_pointer;
    }
    // If B not empty, merge remainder of B.
    while (b_pointer < b_end) {
      a_input = upper_merge_output;
      b_input.load(reinterpret_cast<T*>(b_pointer));
      BitonicMergeNetwork::merge(a_input, b_input, lower_merge_output, upper_merge_output);
      lower_merge_output.store(reinterpret_cast<T*>(output_pointer));
      ++b_pointer;
      ++output_pointer;
    }
    upper_merge_output.store(reinterpret_cast<T*>(output_pointer));
  }

  template <size_t kernel_size>
  static inline void __attribute__((always_inline)) merge_variable_length(T* a_address, T* b_address, T* output_address,
                                                                          const size_t a_length,
                                                                          const size_t b_length) {
    using block_t = struct alignas(kernel_size * sizeof(T)) {};

    static constexpr auto REGISTER_COUNT = kernel_size / count_per_register;
    using MultiVecType = MultiVec<REGISTER_COUNT, count_per_register, VecType>;
    using BitonicMergeNetwork = typename Derived::template BitonicMergeNetwork<kernel_size, MultiVecType>;
    constexpr auto ALIGNMENT_BIT_MASK = get_alignment_bitmask<kernel_size>();

    const auto a_rounded_length = a_length & ALIGNMENT_BIT_MASK;
    const auto b_rounded_length = b_length & ALIGNMENT_BIT_MASK;

    auto a_index = size_t{0};
    auto b_index = size_t{0};

    auto& out = output_address;

    if (a_rounded_length > kernel_size && b_rounded_length > kernel_size) {
      auto* a_pointer = reinterpret_cast<block_t*>(a_address);
      auto* b_pointer = reinterpret_cast<block_t*>(b_address);
      auto* const a_end = reinterpret_cast<block_t*>(a_address + a_length) - 1;
      auto* const b_end = reinterpret_cast<block_t*>(b_address + b_length) - 1;

      auto* output_pointer = reinterpret_cast<block_t*>(out);
      auto* next_pointer = b_pointer;

      auto a_input = MultiVecType{};
      auto b_input = MultiVecType{};
      auto lower_merge_output = MultiVecType{};
      auto upper_merge_output = MultiVecType{};
      a_input.load(reinterpret_cast<T*>(a_pointer));
      b_input.load(reinterpret_cast<T*>(b_pointer));
      ++a_pointer;
      ++b_pointer;

      BitonicMergeNetwork::merge(a_input, b_input, lower_merge_output, upper_merge_output);
      lower_merge_output.store(reinterpret_cast<T*>(output_pointer));
      ++output_pointer;

      // As long as both A and B are not empty, do fetch and 2x4 merge.
      while (a_pointer < a_end && b_pointer < b_end) {
        choose_next_and_update_pointers<block_t, T>(next_pointer, a_pointer, b_pointer);
        a_input = upper_merge_output;
        b_input.load(reinterpret_cast<T*>(next_pointer));
        BitonicMergeNetwork::merge(a_input, b_input, lower_merge_output, upper_merge_output);
        lower_merge_output.store(reinterpret_cast<T*>(output_pointer));
        ++output_pointer;
      }

      const auto last_element_from_merge_output = upper_merge_output.last()[count_per_register - 1];
      if (last_element_from_merge_output <= *reinterpret_cast<T*>(a_pointer)) {
        --a_pointer;
        upper_merge_output.store(reinterpret_cast<T*>(a_pointer));
      } else {
        --b_pointer;
        upper_merge_output.store(reinterpret_cast<T*>(b_pointer));
      }

      a_index = reinterpret_cast<T*>(a_pointer) - a_address;
      b_index = reinterpret_cast<T*>(b_pointer) - b_address;
      a_address = reinterpret_cast<T*>(a_pointer);
      b_address = reinterpret_cast<T*>(b_pointer);
      out = reinterpret_cast<T*>(output_pointer);
    }
    // Serial Merge.
    while (a_index < a_length && b_index < b_length) {
      auto* next = b_address;
      const auto cmp = *a_address < *b_address;
      const auto cmp_neg = !cmp;
      a_index += cmp;
      b_index += cmp_neg;
      next = cmp ? a_address : b_address;
      *out = *next;
      ++out;
      a_address += cmp;
      b_address += cmp_neg;
    }
    while (a_index < a_length) {
      *out = *a_address;
      ++a_index;
      out++;
      ++a_address;
    }
    while (b_index < b_length) {
      *out = *b_address;
      ++b_index;
      out++;
      ++b_address;
    }
  }
};
