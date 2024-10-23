#pragma once

#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

#include <boost/align/aligned_allocator.hpp>

#ifndef L2_CACHE_SIZE
#define L2_CACHE_SIZE 262144  // Default value (256KiB)
#endif

#define LOWER_HALVES 0, 1, 4, 5
#define UPPER_HALVES 2, 3, 6, 7
#define INTERLEAVE_LOWERS 0, 4, 1, 5
#define INTERLEAVE_UPPERS 2, 6, 3, 7

namespace hyrise::simd_sort {

template <class T, std::size_t alignment = 1>
using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, alignment>>;

template <typename T>
using simd_vector = aligned_vector<T, 64>;

template <typename T>
constexpr std::size_t block_size() {
  return L2_CACHE_SIZE / (2 * sizeof(T));
}

template <std::size_t reg_size, typename T>
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
  inline void __attribute__((always_inline)) store(T* address) {
    store_aligned(a, address);
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
  inline void __attribute__((always_inline)) store(T* address) {
    store_aligned(a, address);
    store_aligned(b, address + elements_per_register);
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
  inline void __attribute__((always_inline)) store(T* address) {
    store_aligned(a, address);
    store_aligned(b, address + elements_per_register);
    store_aligned(c, address + 2 * elements_per_register);
    store_aligned(d, address + 3 * elements_per_register);
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
    auto row_2 = load_aligned<VecType>(data + 2 * COUNT_PER_VECTOR);
    auto row_3 = load_aligned<VecType>(data + 3 * COUNT_PER_VECTOR);
    auto row_4 = load_aligned<VecType>(data + 4 * COUNT_PER_VECTOR);
    auto row_5 = load_aligned<VecType>(data + 5 * COUNT_PER_VECTOR);
    auto row_6 = load_aligned<VecType>(data + 6 * COUNT_PER_VECTOR);
    auto row_7 = load_aligned<VecType>(data + 7 * COUNT_PER_VECTOR);

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
    //                                                         0     1    2    3    4    5    6    7

    // __m512i temp0 = _mm512_unpacklo_epi64(row0, row1);  // {a00, a10, a01, a11, a02, a12, a03, a13}
    // __m512i temp1 = _mm512_unpackhi_epi64(row0, row1);  // {a04, a14, a05, a15, a06, a16, a07, a17}
    //                                                           8   9    10   11   12   13   14    15
    // __m512i temp2 = _mm512_unpacklo_epi64(row2, row3);  // {a20, a30, a21, a31, a22, a32, a23, a33}
    // __m512i temp3 = _mm512_unpackhi_epi64(row2, row3);  // {a24, a34, a25, a35, a26, a36, a27, a37}
    // __m512i temp4 = _mm512_unpacklo_epi64(row4, row5);  // {a40, a50, a41, a51, a42, a52, a43, a53}
    // __m512i temp5 = _mm512_unpackhi_epi64(row4, row5);  // {a44, a54, a45, a55, a46, a56, a47, a57}
    // __m512i temp6 = _mm512_unpacklo_epi64(row6, row7);  // {a60, a70, a61, a71, a62, a72, a63, a73}
    // __m512i temp7 = _mm512_unpackhi_epi64(row6, row7);  // {a64, a74, a65, a75, a66, a76, a67, a77}
    auto interleaved_low_01 = __builtin_shufflevector(row_0, row_1, 0, 8, 1, 9, 2, 10, 3, 11);
    auto interleaved_high_01 = __builtin_shufflevector(row_0, row_1, 4, 12, 5, 13, 6, 14, 7, 15);
    auto interleaved_low_23 = __builtin_shufflevector(row_2, row_3, 0, 8, 1, 9, 2, 10, 3, 11);
    auto interleaved_high_23 = __builtin_shufflevector(row_2, row_3, 4, 12, 5, 13, 6, 14, 7, 15);
    auto interleaved_low_45 = __builtin_shufflevector(row_4, row_5, 0, 8, 1, 9, 2, 10, 3, 11);
    auto interleaved_high_45 = __builtin_shufflevector(row_4, row_5, 4, 12, 5, 13, 6, 14, 7, 15);
    auto interleaved_low_67 = __builtin_shufflevector(row_6, row_7, 0, 8, 1, 9, 2, 10, 3, 11);
    auto interleaved_high_67 = __builtin_shufflevector(row_6, row_7, 4, 12, 5, 13, 6, 14, 7, 15);

    // __m512i t0 = _mm512_shuffle_i64x2(temp0, temp2, 0x44);  // {a00, a10, a20, a30, a01, a11, a21, a31}
    // __m512i t1 = _mm512_shuffle_i64x2(temp1, temp3, 0x44);  // {a04, a14, a24, a34, a05, a15, a25, a35}
    // __m512i t2 = _mm512_shuffle_i64x2(temp0, temp2, 0xEE);  // {a02, a12, a22, a32, a03, a13, a23, a33}
    // __m512i t3 = _mm512_shuffle_i64x2(temp1, temp3, 0xEE);  // {a06, a16, a26, a36, a07, a17, a27, a37}
    // __m512i t4 = _mm512_shuffle_i64x2(temp4, temp6, 0x44);  // {a40, a50, a60, a70, a41, a51, a61, a71}
    // __m512i t5 = _mm512_shuffle_i64x2(temp5, temp7, 0x44);  // {a44, a54, a64, a74, a45, a55, a65, a75}
    // __m512i t6 = _mm512_shuffle_i64x2(temp4, temp6, 0xEE);  // {a42, a52, a62, a72, a43, a53, a63, a73}
    // __m512i t7 = _mm512_shuffle_i64x2(temp5, temp7, 0xEE);  // {a46, a56, a66, a76, a47, a57, a67, a77}

    auto temp_0 = __builtin_shufflevector(interleaved_low_01, interleaved_low_23, 0, 1, 8, 9, 2, 3, 10, 11);
    auto temp_1 = __builtin_shufflevector(interleaved_high_01, interleaved_high_23, 0, 1, 8, 9, 2, 3, 10, 11);
    auto temp_2 = __builtin_shufflevector(interleaved_low_01, interleaved_low_23, 4, 5, 12, 13, 6, 7, 14, 15);
    auto temp_3 = __builtin_shufflevector(interleaved_high_01, interleaved_high_23, 4, 5, 12, 13, 6, 7, 14, 15);
    auto temp_4 = __builtin_shufflevector(interleaved_low_45, interleaved_low_67, 0, 1, 8, 9, 2, 3, 10, 11);
    auto temp_5 = __builtin_shufflevector(interleaved_high_45, interleaved_high_67, 0, 1, 8, 9, 2, 3, 10, 11);
    auto temp_6 = __builtin_shufflevector(interleaved_low_45, interleaved_low_67, 4, 5, 12, 13, 6, 7, 14, 15);
    auto temp_7 = __builtin_shufflevector(interleaved_high_45, interleaved_high_67, 4, 5, 12, 13, 6, 7, 14, 15);
    //
    // row_0 = _mm512_unpacklo_epi64(t0, t4);  // {a00, a10, a20, a30, a40, a50, a60, a70}
    // row_1 = _mm512_unpackhi_epi64(t0, t4);  // {a01, a11, a21, a31, a41, a51, a61, a71}
    // row_2 = _mm512_unpacklo_epi64(t2, t6);  // {a02, a12, a22, a32, a42, a52, a62, a72}
    // row_3 = _mm512_unpackhi_epi64(t2, t6);  // {a03, a13, a23, a33, a43, a53, a63, a73}
    // row_4 = _mm512_unpacklo_epi64(t1, t5);  // {a04, a14, a24, a34, a44, a54, a64, a74}
    // row_5 = _mm512_unpackhi_epi64(t1, t5);  // {a05, a15, a25, a35, a45, a55, a65, a75}
    // row_6 = _mm512_unpacklo_epi64(t3, t7);  // {a06, a16, a26, a36, a46, a56, a66, a76}
    // row_7 = _mm512_unpackhi_epi64(t3, t7);  // {a07, a17, a27, a37, a47, a57, a67, a77}

    row_0 = __builtin_shufflevector(temp_0, temp_4, 0, 1, 8, 9, 2, 3, 10, 11);
    row_1 = __builtin_shufflevector(temp_0, temp_4, 4, 5, 12, 13, 6, 7, 14, 15);
    row_2 = __builtin_shufflevector(temp_2, temp_6, 0, 1, 8, 9, 2, 3, 10, 11);
    row_3 = __builtin_shufflevector(temp_2, temp_6, 4, 5, 12, 13, 6, 7, 14, 15);
    row_4 = __builtin_shufflevector(temp_1, temp_5, 0, 1, 8, 9, 2, 3, 10, 11);
    row_5 = __builtin_shufflevector(temp_1, temp_5, 4, 5, 12, 13, 6, 7, 14, 15);
    row_6 = __builtin_shufflevector(temp_3, temp_7, 0, 1, 8, 9, 2, 3, 10, 11);
    row_7 = __builtin_shufflevector(temp_3, temp_7, 4, 5, 12, 13, 6, 7, 14, 15);

    // Write to output
    store_aligned(row_0, output);
    store_aligned(row_1, output + COUNT_PER_VECTOR);
    store_aligned(row_2, output + 2 * COUNT_PER_VECTOR);
    store_aligned(row_3, output + 3 * COUNT_PER_VECTOR);
    store_aligned(row_4, output + 4 * COUNT_PER_VECTOR);
    store_aligned(row_5, output + 5 * COUNT_PER_VECTOR);
    store_aligned(row_6, output + 6 * COUNT_PER_VECTOR);
    store_aligned(row_7, output + 7 * COUNT_PER_VECTOR);
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
inline void __attribute__((always_inline))
choose_next_and_update_pointers(BlockType*& next, BlockType*& a_ptr, BlockType*& b_ptr) {
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
