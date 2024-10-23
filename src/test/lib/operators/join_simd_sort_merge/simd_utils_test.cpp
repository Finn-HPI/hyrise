#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "base_test.hpp"
#include "operators/join_simd_sort_merge/simd_utils.hpp"

namespace hyrise::simd_sort {

using data_type_list = testing::Types<float, int, uint32_t, double, int64_t, uint64_t>;

template <class>
class SimdUtilsTest : public BaseTest {};

TYPED_TEST_SUITE(SimdUtilsTest, data_type_list);

TYPED_TEST(SimdUtilsTest, CreateAlignedData) {
  constexpr auto NUM_ITEMS = 1'000'000;
  constexpr auto ALIGNMENT = 64;  // Biggest alignment needed (AVX-512).
  const auto simd_aligned_vector = simd_vector<TypeParam>(NUM_ITEMS);
  EXPECT_TRUE((is_simd_aligned<TypeParam, ALIGNMENT>(simd_aligned_vector.data())));
}

TYPED_TEST(SimdUtilsTest, LoadAndStoreAligned) {
  const auto input = simd_vector<TypeParam>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  {
    constexpr auto COUNT_PER_VECTOR = 2;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;  // Vector of 2 64-bit elements.
    auto output = simd_vector<TypeParam>(COUNT_PER_VECTOR);
    const auto result = simd_vector<TypeParam>{4, 6};

    auto vec_a = load_aligned<Vec>(input.data());
    auto vec_b = load_aligned<Vec>(input.data() + COUNT_PER_VECTOR);
    auto sum_vec = vec_a + vec_b;
    store_aligned<Vec>(sum_vec, output.data());
    EXPECT_EQ(output, result);
  }
  {
    constexpr auto COUNT_PER_VECTOR = 4;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;  // Vector of 4 64-bit elements.
    auto output = simd_vector<TypeParam>(COUNT_PER_VECTOR);
    const auto result = simd_vector<TypeParam>{6, 8, 10, 12};

    auto vec_a = load_aligned<Vec>(input.data());
    auto vec_b = load_aligned<Vec>(input.data() + COUNT_PER_VECTOR);
    auto sum_vec = vec_a + vec_b;
    store_aligned<Vec>(sum_vec, output.data());
    EXPECT_EQ(output, result);
  }
#ifdef __AVX512F__
  {
    constexpr auto COUNT_PER_VECTOR = 8;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;  // Vector of 4 64-bit elements.
    auto output = simd_vector<TypeParam>(COUNT_PER_VECTOR);
    const auto result = simd_vector<TypeParam>{10, 12, 14, 16, 18, 20, 22, 24};

    auto vec_a = load_aligned<Vec>(input.data());
    auto vec_b = load_aligned<Vec>(input.data() + COUNT_PER_VECTOR);
    auto sum_vec = vec_a + vec_b;
    store_aligned<Vec>(sum_vec, output.data());
    EXPECT_EQ(output, result);
  }
#endif
}

TYPED_TEST(SimdUtilsTest, LoadAndStoreUnaligned) {
  const auto input = simd_vector<TypeParam>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  {
    constexpr auto COUNT_PER_VECTOR = 2;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;  // Vector of 2 64-bit elements.
    auto output = simd_vector<TypeParam>(COUNT_PER_VECTOR);
    const auto result = simd_vector<TypeParam>{4, 6};

    auto vec_a = load_unaligned<Vec>(input.data());
    auto vec_b = load_unaligned<Vec>(input.data() + COUNT_PER_VECTOR);
    auto sum_vec = vec_a + vec_b;
    store_unaligned<Vec>(sum_vec, output.data());
    EXPECT_EQ(output, result);
  }
  {
    constexpr auto COUNT_PER_VECTOR = 4;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;  // Vector of 4 64-bit elements.
    auto output = simd_vector<TypeParam>(COUNT_PER_VECTOR);
    const auto result = simd_vector<TypeParam>{6, 8, 10, 12};

    auto vec_a = load_unaligned<Vec>(input.data());
    auto vec_b = load_unaligned<Vec>(input.data() + COUNT_PER_VECTOR);
    auto sum_vec = vec_a + vec_b;
    store_unaligned<Vec>(sum_vec, output.data());
    EXPECT_EQ(output, result);
  }
#ifdef __AVX512F__
  {
    constexpr auto COUNT_PER_VECTOR = 8;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;  // Vector of 4 64-bit elements.
    auto output = simd_vector<TypeParam>(COUNT_PER_VECTOR);
    const auto result = simd_vector<TypeParam>{10, 12, 14, 16, 18, 20, 22, 24};

    auto vec_a = load_unaligned<Vec>(input.data());
    auto vec_b = load_unaligned<Vec>(input.data() + COUNT_PER_VECTOR);
    auto sum_vec = vec_a + vec_b;
    store_unaligned<Vec>(sum_vec, output.data());
    EXPECT_EQ(output, result);
  }
#endif
}

TYPED_TEST(SimdUtilsTest, SimdCopy) {
  const auto test_copy = []<std::size_t count_per_vector>() {
    for (auto size = std::size_t{1}; size <= 256; ++size) {
      auto input = simd_vector<TypeParam>(size);
      std::iota(input.begin(), input.end(), 0);
      auto output = simd_vector<TypeParam>(size);
      simd_copy<count_per_vector>(output.data(), input.data(), size);
      EXPECT_EQ(input.size(), output.size());
      for (auto index = std::size_t{0}; index < size; ++index) {
        EXPECT_EQ(input[index], output[index]);
      }
    }
  };
  test_copy.template operator()<2>();
  test_copy.template operator()<4>();
#ifdef __AVX512F__
  test_copy.template operator()<8>();
#endif
}

TYPED_TEST(SimdUtilsTest, SortBlockSize) {
  EXPECT_EQ(block_size<TypeParam>() * sizeof(TypeParam), L2_CACHE_SIZE / 2);
}

TYPED_TEST(SimdUtilsTest, MultiVec) {
  using Vec2 = Vec<2 * sizeof(TypeParam), TypeParam>;  // Vector of 2 64-bit elements.
  using Vec4 = Vec<4 * sizeof(TypeParam), TypeParam>;  // Vector of 4 64-bit elements.

  auto input = simd_vector<TypeParam>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                      17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
  auto output2 = simd_vector<TypeParam>(2);
  auto output4 = simd_vector<TypeParam>(4);
  auto output8 = simd_vector<TypeParam>(8);
  auto output16 = simd_vector<TypeParam>(16);
  auto output32 = simd_vector<TypeParam>(32);

  auto result1x2 = simd_vector<TypeParam>{1, 2};
  auto result1x4 = simd_vector<TypeParam>{1, 2, 3, 4};
  auto result2x2 = simd_vector<TypeParam>{1, 2, 3, 4};
  auto result2x4 = simd_vector<TypeParam>{1, 2, 3, 4, 5, 6, 7, 8};
  auto result4x2 = simd_vector<TypeParam>{1, 2, 3, 4, 5, 6, 7, 8};
  auto result4x4 = simd_vector<TypeParam>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  using SingleMultiVec2 = MultiVec<1, 2, Vec2>;
  using SingleMultiVec4 = MultiVec<1, 4, Vec4>;
  using DoubleMultiVec2 = MultiVec<2, 2, Vec2>;
  using DoubleMultiVec4 = MultiVec<2, 4, Vec4>;
  using QuadMultiVec2 = MultiVec<4, 2, Vec2>;
  using QuadMultiVec4 = MultiVec<4, 4, Vec4>;

  {
    auto multi_vec = SingleMultiVec2{};
    multi_vec.load(input.data());
    EXPECT_EQ(multi_vec.first()[0], 1);
    EXPECT_EQ(multi_vec.last()[0], 1);
    multi_vec.store(output2.data());
    EXPECT_EQ(output2, result1x2);
  }
  {
    auto multi_vec = SingleMultiVec4{};
    multi_vec.load(input.data());
    EXPECT_EQ(multi_vec.first()[0], 1);
    EXPECT_EQ(multi_vec.last()[0], 1);
    multi_vec.store(output4.data());
    EXPECT_EQ(output4, result1x4);
  }
  {
    auto multi_vec = DoubleMultiVec2{};
    multi_vec.load(input.data());
    EXPECT_EQ(multi_vec.first()[0], 1);
    EXPECT_EQ(multi_vec.last()[0], 3);
    multi_vec.store(output4.data());
    EXPECT_EQ(output4, result2x2);
  }
  {
    auto multi_vec = DoubleMultiVec4{};
    multi_vec.load(input.data());
    EXPECT_EQ(multi_vec.first()[0], 1);
    EXPECT_EQ(multi_vec.last()[0], 5);
    multi_vec.store(output8.data());
    EXPECT_EQ(output8, result2x4);
  }
  {
    auto multi_vec = QuadMultiVec2{};
    multi_vec.load(input.data());
    EXPECT_EQ(multi_vec.first()[0], 1);
    EXPECT_EQ(multi_vec.last()[0], 7);
    multi_vec.store(output8.data());
    EXPECT_EQ(output8, result4x2);
  }
  {
    auto multi_vec = QuadMultiVec4{};
    multi_vec.load(input.data());
    EXPECT_EQ(multi_vec.first()[0], 1);
    EXPECT_EQ(multi_vec.last()[0], 13);
    multi_vec.store(output16.data());
    EXPECT_EQ(output16, result4x4);
  }
#ifdef __AVX512F__
  using Vec8 = Vec<8 * sizeof(TypeParam), TypeParam>;  // Vector of 8 64-bit elements.
  using SingleMultiVec8 = MultiVec<1, 8, Vec8>;
  using DoubleMultiVec8 = MultiVec<2, 8, Vec8>;
  using QuadMultiVec8 = MultiVec<4, 8, Vec8>;
  auto result4x8 = input;
  {
    auto multi_vec = SingleMultiVec8{};
    multi_vec.load(input.data());
    EXPECT_EQ(multi_vec.first()[0], 1);
    EXPECT_EQ(multi_vec.last()[0], 1);
    multi_vec.store(output8.data());
    EXPECT_EQ(output8, result2x4);
  }

  {
    auto multi_vec = DoubleMultiVec8{};
    multi_vec.load(input.data());
    EXPECT_EQ(multi_vec.first()[0], 1);
    EXPECT_EQ(multi_vec.last()[0], 9);
    multi_vec.store(output16.data());
    EXPECT_EQ(output16, result4x4);
  }
  {
    auto multi_vec = QuadMultiVec8{};
    multi_vec.load(input.data());
    EXPECT_EQ(multi_vec.first()[0], 1);
    EXPECT_EQ(multi_vec.last()[0], 25);
    multi_vec.store(output32.data());
    EXPECT_EQ(output32, result4x8);
  }

#endif
}

TYPED_TEST(SimdUtilsTest, ChooseNextAndUpdatePointers) {
  {
    auto input_a = std::vector<TypeParam>{1, 2};
    auto input_b = std::vector<TypeParam>{3, 4};

    auto* a_pointer = input_a.data();
    auto* b_pointer = input_b.data();
    auto* next = input_b.data();
    choose_next_and_update_pointers<TypeParam, TypeParam>(next, a_pointer, b_pointer);
    EXPECT_EQ(*a_pointer, input_a[1]);
    EXPECT_EQ(*b_pointer, input_b.front());
    EXPECT_EQ(next, input_a.data());
  }
  {
    auto input_a = std::vector<TypeParam>{1, 2};
    auto input_b = std::vector<TypeParam>{1, 4};

    auto* a_pointer = input_a.data();
    auto* b_pointer = input_b.data();
    auto* next = input_b.data();
    choose_next_and_update_pointers<TypeParam, TypeParam>(next, a_pointer, b_pointer);
    EXPECT_EQ(*a_pointer, input_a.front());
    EXPECT_EQ(*b_pointer, input_b[1]);
    EXPECT_EQ(next, input_b.data());
  }
  {
    auto input_a = std::vector<TypeParam>{3, 4};
    auto input_b = std::vector<TypeParam>{1, 2};

    auto* a_pointer = input_a.data();
    auto* b_pointer = input_b.data();
    auto* next = input_b.data();
    choose_next_and_update_pointers<TypeParam, TypeParam>(next, a_pointer, b_pointer);
    EXPECT_EQ(*a_pointer, input_a.front());
    EXPECT_EQ(*b_pointer, input_b[1]);
    EXPECT_EQ(next, input_b.data());
  }
}

TYPED_TEST(SimdUtilsTest, GetAlignmentBitmask) {
  // Bitmasks are used to round to a multiple of the kernel_size.
  EXPECT_EQ((get_alignment_bitmask<2>()), ~std::size_t{1});
  EXPECT_EQ((get_alignment_bitmask<4>()), ~std::size_t{3});
  EXPECT_EQ((get_alignment_bitmask<8>()), ~std::size_t{7});
  EXPECT_EQ((get_alignment_bitmask<16>()), ~std::size_t{15});
}

TYPED_TEST(SimdUtilsTest, Log2Builtin) {
  for (auto value = std::size_t{2}; value < block_size<TypeParam>(); ++value) {
    EXPECT_EQ(log2_builtin(value), static_cast<std::size_t>(std::log2(value)));
  }
}

TYPED_TEST(SimdUtilsTest, SortingNetwork) {
  // Tests for 2x2 sorting network.
  {
    // clang-format off
    auto input = simd_vector<TypeParam>{
      4, 3,
      1, 6
    };
    const auto result = simd_vector<TypeParam>{
      1, 4,
      3, 6
    };
    // clang-format on
    auto output = simd_vector<TypeParam>(4);
    SortingNetwork<2, TypeParam>::sort(input.data(), output.data());
    EXPECT_EQ(output, result);
  }
  {
    // clang-format off
    auto input = simd_vector<TypeParam>{
      1, 5,
      1, 2
    };
    const auto result = simd_vector<TypeParam>{
      1, 1,
      2, 5
    };
    // clang-format on
    auto output = simd_vector<TypeParam>(4);
    SortingNetwork<2, TypeParam>::sort(input.data(), output.data());
    EXPECT_EQ(output, result);
  }
  // Tests for 4x4 sorting network.
  {
    // clang-format off
    auto input = simd_vector<TypeParam>{
      4, 8, 12, 15,
      16, 3, 7, 11,
      14, 15, 2, 6,
      10, 13, 14, 1
    };
    const auto result = simd_vector<TypeParam>{
      4, 10, 14, 16,
      3, 8, 13, 15,
      2, 7, 12, 14,
      1, 6, 11, 15
    };
    // clang-format on
    auto output = simd_vector<TypeParam>(16);

    SortingNetwork<4, TypeParam>::sort(input.data(), output.data());
    EXPECT_EQ(output, result);
  }
#ifdef __AVX512F__
  // Tests for 8x8 sorting network.
  {
    // clang-format off
    auto input = simd_vector<TypeParam>{
      45, 32, 12, 58, 9, 24, 67, 13,
      88, 54, 76, 2, 35, 21, 63, 42,
      6, 18, 27, 70, 33, 14, 55, 48,
      92, 82, 4, 20, 51, 73, 60, 30,
      40, 99, 7, 16, 78, 26, 8, 64,
      28, 91, 5, 37, 44, 3, 31, 11,
      75, 56, 81, 19, 52, 23, 69, 49,
      90, 80, 17, 66, 47, 29, 34, 59
    };
    const auto result = simd_vector<TypeParam>{
      6, 28, 40, 45, 75, 88, 90, 92,
      18, 32, 54, 56, 80, 82, 91, 99,
      4,  5,  7, 12, 17, 27, 76, 81,
      2, 16, 19, 20, 37, 58, 66, 70,
      9, 33, 35, 44, 47, 51, 52, 78,
      3, 14, 21, 23, 24, 26, 29, 73,
      8, 31, 34, 55, 60, 63, 67, 69,
      11, 13, 30, 42, 48, 49, 59, 64
    };
    // clang-format on
    const auto result2 =
        simd_vector<TypeParam>{6,  18, 4,  2,  9,  3,  8,  11, 28, 32, 5,  16, 33, 14, 31, 13, 40, 54, 7,  19, 35, 21,
                               34, 30, 45, 56, 12, 20, 44, 23, 55, 42, 75, 80, 17, 37, 47, 24, 60, 48, 88, 82, 27, 58,
                               51, 26, 63, 49, 90, 91, 76, 66, 52, 29, 67, 59, 92, 99, 81, 70, 78, 73, 69, 64};
    auto output = simd_vector<TypeParam>(64);

    SortingNetwork<8, TypeParam>::sort(input.data(), output.data());
    EXPECT_EQ(output, result);
  }
#endif
}

}  // namespace hyrise::simd_sort
