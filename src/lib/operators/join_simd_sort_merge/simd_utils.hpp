#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

namespace hyrise {

template <class T, std::size_t alignment = 1>
using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, alignment>>;

template <typename T>
using simd_vector = aligned_vector<T, 64>;

template <typename T>
constexpr size_t block_size() {
  return L2_CACHE_SIZE / (2 * sizeof(T));
}

template <size_t reg_size, typename T>
using Vec __attribute__((vector_size(reg_size))) = T;

// Loading and Storing SIMD registers.

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

template <size_t register_count, size_t elements_per_register, typename VecType>
struct MultiVec {
  MultiVec() {
    static_assert(false, "Not implemented");
  }
};

template <size_t elements_per_register, typename VecType>
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

template <typename VecType, size_t elements_per_register>
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

template <typename VecType, size_t elements_per_register>
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

template <size_t elements_per_register, typename T>
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

// Collection of utility functions.

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

template <size_t kernel_size>
constexpr size_t get_alignment_bitmask() {
  return ~(kernel_size - 1);
}

};  // namespace hyrise
