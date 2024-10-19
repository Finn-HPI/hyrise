#include <gtest/gtest.h>

#include <vector>

#include "base_test.hpp"
#include "operators/join_simd_sort_merge/simd_utils.hpp"

namespace hyrise {

using data_type_list = testing::Types<double, int64_t, uint64_t>;

template <class>
class SIMDSortTest : public BaseTest {};

TYPED_TEST_SUITE(SIMDSortTest, data_type_list);

TYPED_TEST(SIMDSortTest, CreateAlignedData) {
  constexpr auto NUM_ITEMS = 1'000'000;
  constexpr auto ALIGNMENT = 64;  // Biggest alignment needed (AVX-512).
  const auto simd_aligned_vector = simd_vector<TypeParam>(NUM_ITEMS);
  EXPECT_TRUE((is_simd_aligned<TypeParam, ALIGNMENT>(simd_aligned_vector.data())));
}

TYPED_TEST(SIMDSortTest, LoadAndStoreAligned) {
  const auto input = simd_vector<TypeParam>{1, 2, 3, 4, 5, 6, 7, 8};
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
}

TYPED_TEST(SIMDSortTest, LoadAndStoreUnaligned) {
  const auto input = simd_vector<TypeParam>{1, 2, 3, 4, 5, 6, 7, 8};
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
}

TYPED_TEST(SIMDSortTest, SortBlockSize) {
  EXPECT_EQ(block_size<TypeParam>() * sizeof(TypeParam), L2_CACHE_SIZE / 2);
}

TYPED_TEST(SIMDSortTest, MultiVec) {
  using Vec2 = Vec<16, TypeParam>;  // Vector of 2 64-bit elements.
  using Vec4 = Vec<32, TypeParam>;  // Vector of 4 64-bit elements.

  auto input = simd_vector<TypeParam>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  auto output2 = simd_vector<TypeParam>(2);
  auto output4 = simd_vector<TypeParam>(4);
  auto output8 = simd_vector<TypeParam>(8);
  auto output16 = simd_vector<TypeParam>(16);

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
}

TYPED_TEST(SIMDSortTest, ChooseNextAndUpdatePointers) {
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

}  // namespace hyrise
