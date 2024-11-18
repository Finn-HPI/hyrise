#pragma once

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include <boost/align/aligned_allocator.hpp>

#ifndef L2_CACHE_SIZE
#define L2_CACHE_SIZE 262144  // Default value (256KiB)
#endif

namespace hyrise::simd_sort {

#define LOWER_HALVES 0, 1, 4, 5
#define UPPER_HALVES 2, 3, 6, 7
#define INTERLEAVE_LOWERS 0, 4, 1, 5
#define INTERLEAVE_UPPERS 2, 6, 3, 7

template <class T, std::size_t alignment = 1>
using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, alignment>>;

template <typename T>
using simd_vector = aligned_vector<T, 64>;

template <typename T>
constexpr std::size_t block_size() {
  return L2_CACHE_SIZE / (2 * sizeof(T));
}

template <std::size_t reg_size, typename T>
  requires(reg_size % sizeof(T) == 0)
using Vec __attribute__((vector_size(reg_size))) = T;

// Loading and Storing SIMD vectors.

template <typename VecType, typename T>
inline __attribute((always_inline)) VecType load_aligned(T* addr) {
  auto vec = VecType{};
  std::memcpy(&vec, addr, sizeof(VecType));
  return std::move(vec);
}

template <typename VecType, typename T>
inline void __attribute((always_inline)) store_aligned(VecType data, T* __restrict output) {
  auto* out_vec = reinterpret_cast<VecType*>(output);
  *out_vec = data;
}

template <typename VecType, typename T>
inline __attribute((always_inline)) VecType load_unaligned(T* addr) {
  using UnalignedVecType __attribute__((aligned(1))) = VecType;
  auto vec = UnalignedVecType{};
  std::memcpy(&vec, addr, sizeof(VecType));
  return vec;
}

template <typename VecType, typename T>
inline void __attribute((always_inline)) store_unaligned(VecType data, T* __restrict output) {
  using UnalignedVecType __attribute__((aligned(1))) = VecType;
  auto* out_vec = reinterpret_cast<UnalignedVecType*>(output);
  *out_vec = data;
}

// Struct for loading & storing  multiple vector registers.

template <std::size_t register_count, std::size_t elements_per_register, typename VecType>
struct MultiVec {
  MultiVec() {
    static_assert(false, "Not implemented");
  }
};

template <std::size_t elements_per_register, typename VecType>
struct MultiVec<1, elements_per_register, VecType> {
  VecType a;

  template <typename T>
  inline void __attribute__((always_inline)) load(T* address) {
    a = load_aligned<VecType>(address);
  }

  template <typename T>
  inline void __attribute__((always_inline)) loadu(T* address) {
    a = load_unaligned<VecType>(address);
  }

  template <typename T>
  inline void __attribute__((always_inline)) store(T* address) {
    store_aligned(a, address);
  }

  template <typename T>
  inline void __attribute__((always_inline)) storeu(T* address) {
    store_unaligned(a, address);
  }

  inline VecType& __attribute__((always_inline)) first() {
    return a;
  }

  inline VecType& __attribute__((always_inline)) last() {
    return a;
  }
};

template <typename VecType, std::size_t elements_per_register>
struct MultiVec<2, elements_per_register, VecType> {
  VecType a;
  VecType b;

  template <typename T>
  inline void __attribute__((always_inline)) load(T* address) {
    a = load_aligned<VecType>(address);
    b = load_aligned<VecType>(address + elements_per_register);
  }

  template <typename T>
  inline void __attribute__((always_inline)) loadu(T* address) {
    a = load_unaligned<VecType>(address);
    b = load_unaligned<VecType>(address + elements_per_register);
  }

  template <typename T>
  inline void __attribute__((always_inline)) store(T* address) {
    store_aligned(a, address);
    store_aligned(b, address + elements_per_register);
  }

  template <typename T>
  inline void __attribute__((always_inline)) storeu(T* address) {
    store_unaligned(a, address);
    store_unaligned(b, address + elements_per_register);
  }

  inline VecType& __attribute__((always_inline)) first() {
    return a;
  }

  inline VecType& __attribute__((always_inline)) last() {
    return b;
  }
};

template <typename VecType, std::size_t elements_per_register>
struct MultiVec<4, elements_per_register, VecType> {
  VecType a;
  VecType b;
  VecType c;
  VecType d;

  template <typename T>
  inline void __attribute__((always_inline)) load(T* address) {
    a = load_aligned<VecType>(address);
    b = load_aligned<VecType>(address + elements_per_register);
    c = load_aligned<VecType>(address + 2 * elements_per_register);
    d = load_aligned<VecType>(address + 3 * elements_per_register);
  }

  template <typename T>
  inline void __attribute__((always_inline)) loadu(T* address) {
    a = load_unaligned<VecType>(address);
    b = load_unaligned<VecType>(address + elements_per_register);
    c = load_unaligned<VecType>(address + 2 * elements_per_register);
    d = load_unaligned<VecType>(address + 3 * elements_per_register);
  }

  template <typename T>
  inline void __attribute__((always_inline)) store(T* address) {
    store_aligned(a, address);
    store_aligned(b, address + elements_per_register);
    store_aligned(c, address + 2 * elements_per_register);
    store_aligned(d, address + 3 * elements_per_register);
  }

  template <typename T>
  inline void __attribute__((always_inline)) storeu(T* address) {
    store_unaligned(a, address);
    store_unaligned(b, address + elements_per_register);
    store_unaligned(c, address + 2 * elements_per_register);
    store_unaligned(d, address + 3 * elements_per_register);
  }

  inline VecType& __attribute__((always_inline)) first() {
    return a;
  }

  inline VecType& __attribute__((always_inline)) last() {
    return d;
  }
};

// Sorting Networks for input sizes 2 and 4.

template <std::size_t elements_per_register, typename T>
struct SortingNetwork {
  SortingNetwork() {
    static_assert(false, "Not implemented.");
  }
};

template <typename VecType>
static inline void __attribute__((always_inline)) compare_min_max(VecType& input1, VecType& input2) {
  // NOLINTBEGIN(cppcoreguidelines-pro-type-vararg, hicpp-vararg)
  auto min = __builtin_elementwise_min(input1, input2);
  auto max = __builtin_elementwise_max(input1, input2);
  // NOLINTEND(cppcoreguidelines-pro-type-vararg, hicpp-vararg)
  input1 = min;
  input2 = max;
}

template <typename T>
struct SortingNetwork<2, T> {
  static inline void __attribute__((always_inline)) sort(T* data, T* output) {
    constexpr auto COUNT_PER_VECTOR = 2;
    constexpr auto VECTOR_SIZE = COUNT_PER_VECTOR * sizeof(T);
    using VecType = Vec<VECTOR_SIZE, T>;

    auto row_0 = load_aligned<VecType>(data);
    auto row_1 = load_aligned<VecType>(data + COUNT_PER_VECTOR);

    // Level 1 comparisons.
    compare_min_max(row_0, row_1);

    // Transpose Matrix
    auto out_1 = __builtin_shufflevector(row_0, row_1, 0, 2);
    auto out_2 = __builtin_shufflevector(row_0, row_1, 1, 3);
    // Write to output
    store_aligned(out_1, output);
    store_aligned(out_2, output + COUNT_PER_VECTOR);
  }
};

template <typename T>
struct SortingNetwork<4, T> {
  static inline void __attribute__((always_inline)) sort(T* data, T* output) {
    constexpr auto COUNT_PER_VECTOR = 4;
    constexpr auto VECTOR_SIZE = COUNT_PER_VECTOR * sizeof(T);
    using VecType = Vec<VECTOR_SIZE, T>;

    auto row_0 = load_aligned<VecType>(data);
    auto row_1 = load_aligned<VecType>(data + COUNT_PER_VECTOR);
    auto row_2 = load_aligned<VecType>(data + 2 * COUNT_PER_VECTOR);
    auto row_3 = load_aligned<VecType>(data + 3 * COUNT_PER_VECTOR);

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
    store_aligned(row_1, output + COUNT_PER_VECTOR);
    store_aligned(row_2, output + 2 * COUNT_PER_VECTOR);
    store_aligned(row_3, output + 3 * COUNT_PER_VECTOR);
  }
};

template <typename T>
struct SortingNetwork<8, T> {
  static inline void __attribute__((always_inline)) sort(T* data, T* output) {
    constexpr auto COUNT_PER_VECTOR = 8;
    constexpr auto VECTOR_SIZE = COUNT_PER_VECTOR * sizeof(T);
    using VecType = Vec<VECTOR_SIZE, T>;

    auto row_0 = load_aligned<VecType>(data);
    auto row_1 = load_aligned<VecType>(data + COUNT_PER_VECTOR);
    auto row_2 = load_aligned<VecType>(data + (2 * COUNT_PER_VECTOR));
    auto row_3 = load_aligned<VecType>(data + (3 * COUNT_PER_VECTOR));
    auto row_4 = load_aligned<VecType>(data + (4 * COUNT_PER_VECTOR));
    auto row_5 = load_aligned<VecType>(data + (5 * COUNT_PER_VECTOR));
    auto row_6 = load_aligned<VecType>(data + (6 * COUNT_PER_VECTOR));
    auto row_7 = load_aligned<VecType>(data + (7 * COUNT_PER_VECTOR));

    // Level 1 comparisons.
    compare_min_max(row_1, row_3);
    compare_min_max(row_0, row_2);
    compare_min_max(row_4, row_6);
    compare_min_max(row_5, row_7);
    // Level 2 comparisons.
    compare_min_max(row_0, row_4);
    compare_min_max(row_1, row_5);
    compare_min_max(row_2, row_6);
    compare_min_max(row_3, row_7);
    // Level 3 comparisons.
    compare_min_max(row_0, row_1);
    compare_min_max(row_2, row_3);
    compare_min_max(row_4, row_5);
    compare_min_max(row_6, row_7);
    // Level 4 comparisons.
    compare_min_max(row_2, row_4);
    compare_min_max(row_3, row_5);
    // Level 5 comparisons.
    compare_min_max(row_1, row_4);
    compare_min_max(row_3, row_6);
    // Level 6 comparisons.
    compare_min_max(row_1, row_2);
    compare_min_max(row_3, row_4);
    compare_min_max(row_5, row_6);

    // Transpose 8x8 Matrix
    // Stage 1.
    auto s1_row_0 = __builtin_shufflevector(row_0, row_1, 0, 8, 2, 10, 4, 12, 6, 14);
    auto s1_row_1 = __builtin_shufflevector(row_0, row_1, 1, 9, 3, 11, 5, 13, 7, 15);
    auto s1_row_2 = __builtin_shufflevector(row_2, row_3, 0, 8, 2, 10, 4, 12, 6, 14);
    auto s1_row_3 = __builtin_shufflevector(row_2, row_3, 1, 9, 3, 11, 5, 13, 7, 15);
    auto s1_row_4 = __builtin_shufflevector(row_4, row_5, 0, 8, 2, 10, 4, 12, 6, 14);
    auto s1_row_5 = __builtin_shufflevector(row_4, row_5, 1, 9, 3, 11, 5, 13, 7, 15);
    auto s1_row_6 = __builtin_shufflevector(row_6, row_7, 0, 8, 2, 10, 4, 12, 6, 14);
    auto s1_row_7 = __builtin_shufflevector(row_6, row_7, 1, 9, 3, 11, 5, 13, 7, 15);
    // Stage 2.
    auto s2_row_0 = __builtin_shufflevector(s1_row_0, s1_row_2, 0, 1, 8, 9, 4, 5, 12, 13);
    auto s2_row_1 = __builtin_shufflevector(s1_row_1, s1_row_3, 0, 1, 8, 9, 4, 5, 12, 13);
    auto s2_row_2 = __builtin_shufflevector(s1_row_0, s1_row_2, 2, 3, 10, 11, 6, 7, 14, 15);
    auto s2_row_3 = __builtin_shufflevector(s1_row_1, s1_row_3, 2, 3, 10, 11, 6, 7, 14, 15);
    auto s2_row_4 = __builtin_shufflevector(s1_row_4, s1_row_6, 0, 1, 8, 9, 4, 5, 12, 13);
    auto s2_row_5 = __builtin_shufflevector(s1_row_5, s1_row_7, 0, 1, 8, 9, 4, 5, 12, 13);
    auto s2_row_6 = __builtin_shufflevector(s1_row_4, s1_row_6, 2, 3, 10, 11, 6, 7, 14, 15);
    auto s2_row_7 = __builtin_shufflevector(s1_row_5, s1_row_7, 2, 3, 10, 11, 6, 7, 14, 15);
    // Stage 3.
    row_0 = __builtin_shufflevector(s2_row_0, s2_row_4, 0, 1, 2, 3, 8, 9, 10, 11);
    row_1 = __builtin_shufflevector(s2_row_1, s2_row_5, 0, 1, 2, 3, 8, 9, 10, 11);
    row_2 = __builtin_shufflevector(s2_row_2, s2_row_6, 0, 1, 2, 3, 8, 9, 10, 11);
    row_3 = __builtin_shufflevector(s2_row_3, s2_row_7, 0, 1, 2, 3, 8, 9, 10, 11);

    row_4 = __builtin_shufflevector(s2_row_0, s2_row_4, 4, 5, 6, 7, 12, 13, 14, 15);
    row_5 = __builtin_shufflevector(s2_row_1, s2_row_5, 4, 5, 6, 7, 12, 13, 14, 15);
    row_6 = __builtin_shufflevector(s2_row_2, s2_row_6, 4, 5, 6, 7, 12, 13, 14, 15);
    row_7 = __builtin_shufflevector(s2_row_3, s2_row_7, 4, 5, 6, 7, 12, 13, 14, 15);

    // Write to output
    store_aligned(row_0, output);
    store_aligned(row_1, output + COUNT_PER_VECTOR);
    store_aligned(row_2, output + (2 * COUNT_PER_VECTOR));
    store_aligned(row_3, output + (3 * COUNT_PER_VECTOR));
    store_aligned(row_4, output + (4 * COUNT_PER_VECTOR));
    store_aligned(row_5, output + (5 * COUNT_PER_VECTOR));
    store_aligned(row_6, output + (6 * COUNT_PER_VECTOR));
    store_aligned(row_7, output + (7 * COUNT_PER_VECTOR));
  }
};

// Collection of utility functions.

template <typename T>
  requires std::is_unsigned_v<T>
inline std::size_t __attribute__((always_inline)) log2_builtin(T val) {
  constexpr int NUM_BITS = sizeof(T) * CHAR_BIT;
  if constexpr (NUM_BITS == 32) {
    return (NUM_BITS - 1) - __builtin_clz(static_cast<unsigned int>(val));
  }
  if constexpr (NUM_BITS == 64) {
    return (NUM_BITS - 1) - __builtin_clzll(static_cast<uint64_t>(val));
  }
  return (NUM_BITS - 1) - __builtin_clz(static_cast<unsigned int>(val));
}

template <typename T, std::size_t alignment = 1>
inline __attribute((always_inline)) bool is_simd_aligned(const T* addr) {
  return reinterpret_cast<std::uintptr_t>(addr) % alignment == 0;
}

template <typename BlockType, typename T>
inline void __attribute__((always_inline)) choose_next_and_update_pointers(BlockType*& next, BlockType*& a_ptr,
                                                                           BlockType*& b_ptr) {
  const int8_t cmp = *reinterpret_cast<T*>(a_ptr) < *reinterpret_cast<T*>(b_ptr);
  next = cmp ? a_ptr : b_ptr;
  a_ptr += cmp;
  b_ptr += !cmp;
}

template <std::size_t kernel_size>
constexpr std::size_t get_alignment_bitmask() {
  return ~(kernel_size - 1);
}

template <std::size_t count_per_vector, typename T>
inline void __attribute__((always_inline)) simd_copy(T* dest, T* src, std::size_t size) {
  if (size == 0) {
    return;
  }
  using VecType = Vec<count_per_vector * sizeof(T), T>;

  auto simd_copy_size = size & ~(count_per_vector - 1);
  size -= simd_copy_size;

  while (simd_copy_size) {
    auto vec = load_unaligned<VecType>(src);
    store_unaligned<VecType>(vec, dest);
    simd_copy_size -= count_per_vector;
    src += count_per_vector;
    dest += count_per_vector;
  }

  while (size) {
    *dest = *src;
    ++dest;
    ++src;
    --size;
  }
}

};  // namespace hyrise::simd_sort
