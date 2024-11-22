#pragma once
#include <cstddef>

#include "simd_utils.hpp"

namespace hyrise::simd_sort {

constexpr auto MERGE_AB = 2;
constexpr auto MERGE_2AB = 4;
constexpr auto MERGE_4AB = 8;

template <std::size_t count_per_vector, typename T, typename Derived>
class AbstractTwoWayMerge {
  static constexpr auto VECTOR_SIZE = count_per_vector * sizeof(T);
  using VecType = Vec<VECTOR_SIZE, T>;

  // NOLINTBEGIN(cppcoreguidelines-pro-type-vararg, hicpp-vararg)
 public:
  static void __attribute__((always_inline)) merge_2x_base_size(VecType& in11, VecType& in12, VecType& in21,
                                                                VecType& in22, VecType& out1, VecType& out2,
                                                                VecType& out3, VecType& out4) {
    auto l11 = __builtin_elementwise_min(in11, in21);
    auto l12 = __builtin_elementwise_min(in12, in22);
    auto h11 = __builtin_elementwise_max(in11, in21);
    auto h12 = __builtin_elementwise_max(in12, in22);
    Derived::merge_base_size(l11, l12, out1, out2);
    Derived::merge_base_size(h11, h12, out3, out4);
  }

  template <std::size_t input_count, typename MulitVecType>
  struct BitonicMergeNetwork {
    static void __attribute__((always_inline)) merge(MulitVecType& /*in1*/, MulitVecType& /*in2*/,
                                                     MulitVecType& /*out1*/, MulitVecType& /*out2*/) {
      static_assert(false, "Not implemented.");
    }
  };

  template <typename MultiVecType>
  struct BitonicMergeNetwork<MERGE_AB, MultiVecType> {
    static void __attribute__((always_inline)) merge(MultiVecType& in1, MultiVecType& in2, MultiVecType& out1,
                                                     MultiVecType& out2) {
      Derived::reverse(in2.a);
      Derived::merge_base_size(in1.a, in2.a, out1.a, out2.a);
    }
  };

  template <typename MultiVecType>
  struct BitonicMergeNetwork<MERGE_2AB, MultiVecType> {
    static void __attribute__((always_inline)) merge(MultiVecType& in1, MultiVecType& in2, MultiVecType& out1,
                                                     MultiVecType& out2) {
      Derived::reverse(in2.a);
      Derived::reverse(in2.b);
      auto l11 = __builtin_elementwise_min(in1.a, in2.b);
      auto l12 = __builtin_elementwise_min(in1.b, in2.a);
      auto h11 = __builtin_elementwise_max(in1.a, in2.b);
      auto h12 = __builtin_elementwise_max(in1.b, in2.a);
      Derived::merge_base_size(l11, l12, out1.a, out1.b);
      Derived::merge_base_size(h11, h12, out2.a, out2.b);
    }
  };

  template <typename MultiVecType>
  struct BitonicMergeNetwork<MERGE_4AB, MultiVecType> {
    static void __attribute__((always_inline)) merge(MultiVecType& in1, MultiVecType& in2, MultiVecType& out1,
                                                     MultiVecType& out2) {
      Derived::reverse(in2.a);
      Derived::reverse(in2.b);
      Derived::reverse(in2.c);
      Derived::reverse(in2.d);
      auto l01 = __builtin_elementwise_min(in1.a, in2.d);
      auto l02 = __builtin_elementwise_min(in1.b, in2.c);
      auto l03 = __builtin_elementwise_min(in1.c, in2.b);
      auto l04 = __builtin_elementwise_min(in1.d, in2.a);
      auto h01 = __builtin_elementwise_max(in1.a, in2.d);
      auto h02 = __builtin_elementwise_max(in1.b, in2.c);
      auto h03 = __builtin_elementwise_max(in1.c, in2.b);
      auto h04 = __builtin_elementwise_max(in1.d, in2.a);
      merge_2x_base_size(l01, l02, l03, l04, out1.a, out1.b, out1.c, out1.d);
      merge_2x_base_size(h01, h02, h03, h04, out2.a, out2.b, out2.c, out2.d);
    }
  };

  // NOLINTEND(cppcoreguidelines-pro-type-vararg, hicpp-vararg)

  template <std::size_t kernel_size>
  static void __attribute__((always_inline)) merge_equal_length(T* const a_address, T* const b_address,
                                                                T* const output_address, const std::size_t length) {
    using block_t = struct alignas(kernel_size * sizeof(T)) {};

    static constexpr auto VECTOR_COUNT = kernel_size / count_per_vector;
    static constexpr auto MERGE_NETWORK_INPUT_SIZE = VECTOR_COUNT * 2;

    using MultiVecType = MultiVec<VECTOR_COUNT, count_per_vector, VecType>;
    using BitonicMergeNetwork = BitonicMergeNetwork<MERGE_NETWORK_INPUT_SIZE, MultiVecType>;

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

  // Hint: this function has sideeffects on the input of a and b.
  template <std::size_t kernel_size>
  static void __attribute__((always_inline)) merge_variable_length(T* a_address, T* b_address, T* output_address,
                                                                   const std::size_t a_length,
                                                                   const std::size_t b_length) {
    using block_t = struct alignas(kernel_size * sizeof(T)) {};

    static constexpr auto VECTOR_COUNT = kernel_size / count_per_vector;
    static constexpr auto MERGE_NETWORK_INPUT_SIZE = VECTOR_COUNT * 2;

    using MultiVecType = MultiVec<VECTOR_COUNT, count_per_vector, VecType>;
    using BitonicMergeNetwork = BitonicMergeNetwork<MERGE_NETWORK_INPUT_SIZE, MultiVecType>;
    constexpr auto ALIGNMENT_BIT_MASK = get_alignment_bitmask<kernel_size>();

    const auto a_rounded_length = a_length & ALIGNMENT_BIT_MASK;
    const auto b_rounded_length = b_length & ALIGNMENT_BIT_MASK;

    auto a_index = std::size_t{0};
    auto b_index = std::size_t{0};

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

      const auto last_element_from_merge_output = upper_merge_output.last()[count_per_vector - 1];
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
    const auto a_copy_length = a_length - a_index;
    simd_copy<count_per_vector>(out, a_address, a_copy_length);
    out += a_copy_length;
    const auto b_copy_length = b_length - b_index;
    simd_copy<count_per_vector>(out, b_address, b_copy_length);
  }

  template <std::size_t kernel_size>
  static void __attribute__((always_inline)) merge_variable_length_unaligned(T* a_address, T* b_address,
                                                                             T* output_address,
                                                                             const std::size_t a_length,
                                                                             const std::size_t b_length) {
    using block_t = struct alignas(kernel_size * sizeof(T)) {};

    static constexpr auto VECTOR_COUNT = kernel_size / count_per_vector;
    static constexpr auto MERGE_NETWORK_INPUT_SIZE = VECTOR_COUNT * 2;

    using MultiVecType = MultiVec<VECTOR_COUNT, count_per_vector, VecType>;
    using BitonicMergeNetwork = BitonicMergeNetwork<MERGE_NETWORK_INPUT_SIZE, MultiVecType>;
    constexpr auto ALIGNMENT_BIT_MASK = get_alignment_bitmask<kernel_size>();

    const auto a_rounded_length = a_length & ALIGNMENT_BIT_MASK;
    const auto b_rounded_length = b_length & ALIGNMENT_BIT_MASK;

    auto a_index = std::size_t{0};
    auto b_index = std::size_t{0};

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
      a_input.loadu(reinterpret_cast<T*>(a_pointer));
      b_input.loadu(reinterpret_cast<T*>(b_pointer));
      ++a_pointer;
      ++b_pointer;

      BitonicMergeNetwork::merge(a_input, b_input, lower_merge_output, upper_merge_output);
      lower_merge_output.storeu(reinterpret_cast<T*>(output_pointer));
      ++output_pointer;

      // As long as both A and B are not empty, do fetch and 2x4 merge.
      while (a_pointer < a_end && b_pointer < b_end) {
        choose_next_and_update_pointers<block_t, T>(next_pointer, a_pointer, b_pointer);
        a_input = upper_merge_output;
        b_input.loadu(reinterpret_cast<T*>(next_pointer));
        BitonicMergeNetwork::merge(a_input, b_input, lower_merge_output, upper_merge_output);
        lower_merge_output.storeu(reinterpret_cast<T*>(output_pointer));
        ++output_pointer;
      }

      const auto last_element_from_merge_output = upper_merge_output.last()[count_per_vector - 1];
      if (last_element_from_merge_output <= *reinterpret_cast<T*>(a_pointer)) {
        --a_pointer;
        upper_merge_output.storeu(reinterpret_cast<T*>(a_pointer));
      } else {
        --b_pointer;
        upper_merge_output.storeu(reinterpret_cast<T*>(b_pointer));
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
    const auto a_copy_length = a_length - a_index;
    simd_copy<count_per_vector>(out, a_address, a_copy_length);
    out += a_copy_length;
    const auto b_copy_length = b_length - b_index;
    simd_copy<count_per_vector>(out, b_address, b_copy_length);
  }

  template <std::size_t kernel_size>
  static void __attribute__((always_inline)) merge_multiway_merge_nodes(
      T* a_address, T* b_address, T* output_address, std::size_t& count_reads_a, std::size_t& count_reads_b,
      std::size_t& count_writes, const std::size_t a_length, const std::size_t b_length,
      const std::size_t output_size) {
    using block_t = struct alignas(kernel_size * sizeof(T)) {};

    static constexpr auto VECTOR_COUNT = kernel_size / count_per_vector;
    static constexpr auto MERGE_NETWORK_INPUT_SIZE = VECTOR_COUNT * 2;

    using MultiVecType = MultiVec<VECTOR_COUNT, count_per_vector, VecType>;
    using BitonicMergeNetwork = BitonicMergeNetwork<MERGE_NETWORK_INPUT_SIZE, MultiVecType>;
    constexpr auto ALIGNMENT_BIT_MASK = get_alignment_bitmask<kernel_size>();
    constexpr auto SLOT_DECREMENT = VECTOR_COUNT * count_per_vector;

    const auto a_rounded_length = a_length & ALIGNMENT_BIT_MASK;
    const auto b_rounded_length = b_length & ALIGNMENT_BIT_MASK;

    auto number_of_slots = output_size;
    auto remaining_slots = output_size & ~ALIGNMENT_BIT_MASK;
    number_of_slots -= remaining_slots;

    auto a_index = count_reads_a;
    auto b_index = count_reads_b;
    auto output_index = count_writes;

    auto& out = output_address;

    if (number_of_slots && a_rounded_length > kernel_size && b_rounded_length > kernel_size) {
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
      a_input.loadu(reinterpret_cast<T*>(a_pointer));
      b_input.loadu(reinterpret_cast<T*>(b_pointer));
      ++a_pointer;
      ++b_pointer;

      BitonicMergeNetwork::merge(a_input, b_input, lower_merge_output, upper_merge_output);
      lower_merge_output.storeu(reinterpret_cast<T*>(output_pointer));
      ++output_pointer;
      number_of_slots -= SLOT_DECREMENT;

      // As long as both A and B are not empty, do fetch and 2x4 merge.
      while (number_of_slots && a_pointer < a_end && b_pointer < b_end) {
        choose_next_and_update_pointers<block_t, T>(next_pointer, a_pointer, b_pointer);
        a_input = upper_merge_output;
        b_input.loadu(reinterpret_cast<T*>(next_pointer));
        BitonicMergeNetwork::merge(a_input, b_input, lower_merge_output, upper_merge_output);
        lower_merge_output.storeu(reinterpret_cast<T*>(output_pointer));
        ++output_pointer;
        number_of_slots -= SLOT_DECREMENT;
      }

      const auto last_element_from_merge_output = upper_merge_output.last()[count_per_vector - 1];
      if (last_element_from_merge_output <= *reinterpret_cast<T*>(a_pointer)) {
        --a_pointer;
        upper_merge_output.storeu(reinterpret_cast<T*>(a_pointer));
      } else {
        --b_pointer;
        upper_merge_output.storeu(reinterpret_cast<T*>(b_pointer));
      }

      a_index = count_reads_a + (reinterpret_cast<T*>(a_pointer) - a_address);
      b_index = count_reads_b + (reinterpret_cast<T*>(b_pointer) - b_address);
      output_index = count_writes + (reinterpret_cast<T*>(output_pointer) - out);

      a_address = reinterpret_cast<T*>(a_pointer);
      b_address = reinterpret_cast<T*>(b_pointer);
      out = reinterpret_cast<T*>(output_pointer);
    }
    number_of_slots += remaining_slots;

    // Serial merge until all slots are used up.
    while (number_of_slots && a_index < a_length && b_index < b_length) {
      auto* next = b_address;
      const auto cmp = *a_address < *b_address;
      const auto cmp_neg = !cmp;
      a_index += cmp;
      b_index += cmp_neg;
      next = cmp ? a_address : b_address;
      *out = *next;
      ++out;
      ++output_index;
      --number_of_slots;
      a_address += cmp;
      b_address += cmp_neg;
    }
    count_reads_a = a_index;
    count_reads_b = b_index;
    count_writes = output_index;
  }
};
}  // namespace hyrise::simd_sort
