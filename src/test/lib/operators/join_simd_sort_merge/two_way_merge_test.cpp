#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

#include "base_test.hpp"
#include "operators/join_simd_sort_merge/two_way_merge.hpp"

namespace hyrise::simd_sort {

using data_type_list = testing::Types<float, int, uint32_t, double, int64_t, uint64_t>;

template <class>
class SimdTwoWayMergeTest : public BaseTest {};

TYPED_TEST_SUITE(SimdTwoWayMergeTest, data_type_list);

TYPED_TEST(SimdTwoWayMergeTest, Reverse) {
  {
    constexpr auto COUNT_PER_VECTOR = 2;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;
    using TwoWayMergeT = TwoWayMerge<COUNT_PER_VECTOR, TypeParam>;

    auto two_way_merge = TwoWayMergeT{};
    auto vec = Vec{2, 1};
    two_way_merge.reverse(vec);
    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[1], 2);
  }
  {
    constexpr auto COUNT_PER_VECTOR = 4;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;
    using TwoWayMergeT = TwoWayMerge<COUNT_PER_VECTOR, TypeParam>;

    auto two_way_merge = TwoWayMergeT{};
    auto vec = Vec{4, 3, 2, 1};
    two_way_merge.reverse(vec);
    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[1], 2);
    EXPECT_EQ(vec[2], 3);
    EXPECT_EQ(vec[3], 4);
  }
#ifdef __AVX512F__
  {
    constexpr auto COUNT_PER_VECTOR = 8;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;
    using TwoWayMergeT = TwoWayMerge<COUNT_PER_VECTOR, TypeParam>;

    auto two_way_merge = TwoWayMergeT{};
    auto vec = Vec{8, 7, 6, 5, 4, 3, 2, 1};
    two_way_merge.reverse(vec);
    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[1], 2);
    EXPECT_EQ(vec[2], 3);
    EXPECT_EQ(vec[3], 4);
    EXPECT_EQ(vec[4], 5);
    EXPECT_EQ(vec[5], 6);
    EXPECT_EQ(vec[6], 7);
    EXPECT_EQ(vec[7], 8);
  }

#endif
}

TYPED_TEST(SimdTwoWayMergeTest, MergeNetwokInputSize2) {
  auto test = []<typename TwoWayMergeT, typename MultiVec>(simd_vector<TypeParam>& data) {
    auto two_way_merge = TwoWayMergeT{};
    auto result = data;
    std::ranges::sort(result);

    auto input_ab = MultiVec{};
    auto merge_output = MultiVec{};
    input_ab.load(data.data());

    two_way_merge.merge_network_input_x2(input_ab.a, input_ab.b, merge_output.a, merge_output.b);

    auto output = simd_vector<TypeParam>(data.size());
    merge_output.store(output.data());

    EXPECT_EQ(output, result);
  };
  {
    constexpr auto COUNT_PER_VECTOR = 2;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;
    using MultiVec = MultiVec<2, COUNT_PER_VECTOR, Vec>;
    using TwoWayMergeT = TwoWayMerge<COUNT_PER_VECTOR, TypeParam>;

    auto data = simd_vector<TypeParam>{1, 5, 9, 1};
    test.template operator()<TwoWayMergeT, MultiVec>(data);
  }
  {
    constexpr auto COUNT_PER_VECTOR = 4;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;
    using MultiVec = MultiVec<2, COUNT_PER_VECTOR, Vec>;
    using TwoWayMergeT = TwoWayMerge<COUNT_PER_VECTOR, TypeParam>;

    auto data = simd_vector<TypeParam>{1, 4, 6, 7, 5, 4, 3, 2};
    test.template operator()<TwoWayMergeT, MultiVec>(data);
  }
#ifdef __AVX512F__
  {
    constexpr auto COUNT_PER_VECTOR = 8;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;
    using MultiVec = MultiVec<2, COUNT_PER_VECTOR, Vec>;
    using TwoWayMergeT = TwoWayMerge<COUNT_PER_VECTOR, TypeParam>;

    auto data = simd_vector<TypeParam>{1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 6, 4, 4, 3, 2, 1};
    test.template operator()<TwoWayMergeT, MultiVec>(data);
  }
#endif
}

TYPED_TEST(SimdTwoWayMergeTest, MergeNetwokInputSize4) {
  constexpr auto INPUT_SIZE = 4;
  auto test = []<typename TwoWayMergeT, typename MultiVec>(simd_vector<TypeParam>& data) {
    auto two_way_merge = TwoWayMergeT{};
    auto result = data;
    std::ranges::sort(result);

    auto input_ab = MultiVec{};
    auto merge_output = MultiVec{};
    input_ab.load(data.data());

    two_way_merge.merge_network_input_x4(input_ab.a, input_ab.b, input_ab.c, input_ab.d, merge_output.a, merge_output.b,
                                         merge_output.c, merge_output.d);

    auto output = simd_vector<TypeParam>(data.size());
    merge_output.store(output.data());

    EXPECT_EQ(output, result);
  };
  {
    constexpr auto COUNT_PER_VECTOR = 2;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;
    using MultiVec = MultiVec<INPUT_SIZE, COUNT_PER_VECTOR, Vec>;
    using TwoWayMergeT = TwoWayMerge<COUNT_PER_VECTOR, TypeParam>;
    // clang-format off
    // The input is split into a smaller and a bigger half.
    auto data = simd_vector<TypeParam>{
      1, 4,  // sorted ascending
      3, 2,  // sorted descending
      7, 8,  // sorted ascending
      6, 5   // sorted descending
    };
    // clang-format on
    test.template operator()<TwoWayMergeT, MultiVec>(data);
  }
  {
    constexpr auto COUNT_PER_VECTOR = 4;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;
    using MultiVec = MultiVec<INPUT_SIZE, COUNT_PER_VECTOR, Vec>;
    using TwoWayMergeT = TwoWayMerge<COUNT_PER_VECTOR, TypeParam>;
    // clang-format off
    // The input is split into a smaller and a bigger half.
    auto data = simd_vector<TypeParam>{
      1, 2, 3, 4,   // sorted ascending
      4, 3, 2, 1,   // sorted descending
      7, 8, 9, 10,  // sorted ascending
      6, 5, 5, 4    // sorted descending
    };
    // clang-format on
    test.template operator()<TwoWayMergeT, MultiVec>(data);
  }
#ifdef __AVX512F__
  {
    constexpr auto COUNT_PER_VECTOR = 8;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;
    using MultiVec = MultiVec<INPUT_SIZE, COUNT_PER_VECTOR, Vec>;
    using TwoWayMergeT = TwoWayMerge<COUNT_PER_VECTOR, TypeParam>;
    // clang-format off
    // The input is split into a smaller and a bigger half.
    auto data = simd_vector<TypeParam>{
      1, 2, 3, 4, 5, 6, 7, 8,   // sorted ascending
      8, 7, 6, 5, 4, 3, 2, 1,   // sorted descending
      8, 8, 16, 17, 18, 19, 20, 21,  // sorted ascending
      32, 21, 20, 16, 15, 14, 9, 8    // sorted descending
    };
    // clang-format on
    test.template operator()<TwoWayMergeT, MultiVec>(data);
  }
#endif
}

template <typename TwoWayMergeT, typename MultiVec, std::size_t merge_input_count, typename T>
void test_bitonic_merge_network(simd_vector<T>& a_data, simd_vector<T>& b_data, std::size_t write_offset) {
  auto result = simd_vector<T>{};
  std::ranges::merge(a_data, b_data, std::back_inserter(result));

  auto input_a = MultiVec{};
  input_a.load(a_data.data());
  auto input_b = MultiVec{};
  input_b.load(b_data.data());
  auto lower_output = MultiVec{};
  auto upper_output = MultiVec{};

  TwoWayMergeT::template BitonicMergeNetwork<merge_input_count, MultiVec>::merge(input_a, input_b, lower_output,
                                                                                 upper_output);
  auto output = simd_vector<T>(a_data.size() + b_data.size());
  lower_output.store(output.data());
  upper_output.store(output.data() + write_offset);
  EXPECT_EQ(output, result);
}

TYPED_TEST(SimdTwoWayMergeTest, BitonicMergeNetworkMergeAB) {
  {
    constexpr auto COUNT_PER_VECTOR = 2;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;
    using MultiVec = MultiVec<1, COUNT_PER_VECTOR, Vec>;
    using TwoWayMergeT = TwoWayMerge<COUNT_PER_VECTOR, TypeParam>;

    auto a_data = simd_vector<TypeParam>{1, 5};
    auto b_data = simd_vector<TypeParam>{1, 9};

    test_bitonic_merge_network<TwoWayMergeT, MultiVec, MERGE_AB>(a_data, b_data, COUNT_PER_VECTOR);
  }
  {
    constexpr auto COUNT_PER_VECTOR = 4;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;
    using MultiVec = MultiVec<1, COUNT_PER_VECTOR, Vec>;
    using TwoWayMergeT = TwoWayMerge<COUNT_PER_VECTOR, TypeParam>;

    auto a_data = simd_vector<TypeParam>{1, 3, 5, 6};
    auto b_data = simd_vector<TypeParam>{2, 3, 4, 7};

    test_bitonic_merge_network<TwoWayMergeT, MultiVec, MERGE_AB>(a_data, b_data, COUNT_PER_VECTOR);
  }
#ifdef __AVX512F__
  {
    constexpr auto COUNT_PER_VECTOR = 8;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;
    using MultiVec = MultiVec<1, COUNT_PER_VECTOR, Vec>;
    using TwoWayMergeT = TwoWayMerge<COUNT_PER_VECTOR, TypeParam>;

    auto a_data = simd_vector<TypeParam>{1, 3, 5, 6, 8, 9, 10, 12};
    auto b_data = simd_vector<TypeParam>{2, 3, 4, 7, 8, 10, 11, 13};

    test_bitonic_merge_network<TwoWayMergeT, MultiVec, MERGE_AB>(a_data, b_data, COUNT_PER_VECTOR);
  }
#endif
}

TYPED_TEST(SimdTwoWayMergeTest, BitonicMergeNetworkMerge2AB) {
  {
    constexpr auto COUNT_PER_VECTOR = 2;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;
    using MultiVec = MultiVec<2, COUNT_PER_VECTOR, Vec>;
    using TwoWayMergeT = TwoWayMerge<COUNT_PER_VECTOR, TypeParam>;

    auto a_data = simd_vector<TypeParam>{1, 3, 6, 9};
    auto b_data = simd_vector<TypeParam>{2, 3, 5, 7};

    test_bitonic_merge_network<TwoWayMergeT, MultiVec, MERGE_2AB>(a_data, b_data, 2 * COUNT_PER_VECTOR);
  }
  {
    constexpr auto COUNT_PER_VECTOR = 4;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;
    using MultiVec = MultiVec<2, COUNT_PER_VECTOR, Vec>;
    using TwoWayMergeT = TwoWayMerge<COUNT_PER_VECTOR, TypeParam>;

    auto a_data = simd_vector<TypeParam>{1, 2, 3, 4, 5, 6, 7, 8};
    auto b_data = simd_vector<TypeParam>{2, 3, 3, 4, 5, 5, 9, 10};

    test_bitonic_merge_network<TwoWayMergeT, MultiVec, MERGE_2AB>(a_data, b_data, 2 * COUNT_PER_VECTOR);
  }
#ifdef __AVX512F__
  {
    constexpr auto COUNT_PER_VECTOR = 8;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;
    using MultiVec = MultiVec<2, COUNT_PER_VECTOR, Vec>;
    using TwoWayMergeT = TwoWayMerge<COUNT_PER_VECTOR, TypeParam>;

    auto a_data = simd_vector<TypeParam>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    auto b_data = simd_vector<TypeParam>{2, 3, 3, 4, 5, 5, 9, 10, 11, 12, 13, 13, 14, 18, 20, 21};

    test_bitonic_merge_network<TwoWayMergeT, MultiVec, MERGE_2AB>(a_data, b_data, 2 * COUNT_PER_VECTOR);
  }
#endif
}

TYPED_TEST(SimdTwoWayMergeTest, BitonicMergeNetworkMerge4AB) {
  {
    constexpr auto COUNT_PER_VECTOR = 2;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;
    using MultiVec = MultiVec<4, COUNT_PER_VECTOR, Vec>;
    using TwoWayMergeT = TwoWayMerge<COUNT_PER_VECTOR, TypeParam>;

    auto a_data = simd_vector<TypeParam>{1, 2, 3, 4, 5, 6, 7, 8};
    auto b_data = simd_vector<TypeParam>{2, 3, 3, 4, 5, 5, 9, 10};

    test_bitonic_merge_network<TwoWayMergeT, MultiVec, MERGE_4AB>(a_data, b_data, 4 * COUNT_PER_VECTOR);
  }
  {
    constexpr auto COUNT_PER_VECTOR = 4;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;
    using MultiVec = MultiVec<4, COUNT_PER_VECTOR, Vec>;
    using TwoWayMergeT = TwoWayMerge<COUNT_PER_VECTOR, TypeParam>;

    auto a_data = simd_vector<TypeParam>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    auto b_data = simd_vector<TypeParam>{2, 3, 3, 5, 6, 8, 8, 10, 11, 13, 16, 18, 19, 20, 21, 22};

    test_bitonic_merge_network<TwoWayMergeT, MultiVec, MERGE_4AB>(a_data, b_data, 4 * COUNT_PER_VECTOR);
  }
#ifdef __AVX512F__
  {
    constexpr auto COUNT_PER_VECTOR = 8;
    using Vec = Vec<COUNT_PER_VECTOR * sizeof(TypeParam), TypeParam>;
    using MultiVec = MultiVec<4, COUNT_PER_VECTOR, Vec>;
    using TwoWayMergeT = TwoWayMerge<COUNT_PER_VECTOR, TypeParam>;

    auto a_data = simd_vector<TypeParam>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    auto b_data = simd_vector<TypeParam>{2,  3,  3,  5,  6,  8,  8,  10, 11, 13, 16, 18, 19, 20, 21, 22,
                                         23, 24, 24, 25, 26, 28, 30, 31, 32, 33, 36, 38, 40, 41, 42, 43};

    test_bitonic_merge_network<TwoWayMergeT, MultiVec, MERGE_4AB>(a_data, b_data, 4 * COUNT_PER_VECTOR);
  }
#endif
}

TYPED_TEST(SimdTwoWayMergeTest, MergeEqLength) {
  auto test_merge_eq_length = []<std::size_t count_per_vector>() {
    using TwoWayMerge = TwoWayMerge<count_per_vector, TypeParam>;

    constexpr auto START_LENGTH = 4 * count_per_vector;
    constexpr auto MAX_LENGTH = 128 * count_per_vector;
    for (auto length = START_LENGTH; length <= MAX_LENGTH; length = length * 2) {
      auto a_data = simd_vector<TypeParam>(length);
      auto b_data = simd_vector<TypeParam>(length);
      std::iota(a_data.begin(), a_data.end(), 0);
      std::iota(b_data.begin(), b_data.end(), a_data[length / 2]);

      auto expected_result = simd_vector<TypeParam>{};
      std::ranges::merge(a_data, b_data, std::back_inserter(expected_result));
      {
        auto result = simd_vector<TypeParam>(length * 2);
        EXPECT_NE(result, expected_result);
        TwoWayMerge::template merge_equal_length<count_per_vector>(a_data.data(), b_data.data(), result.data(), length);
        EXPECT_EQ(result, expected_result);
      }
      {
        auto result = simd_vector<TypeParam>(length * 2);
        EXPECT_NE(result, expected_result);
        TwoWayMerge::template merge_equal_length<2 * count_per_vector>(a_data.data(), b_data.data(), result.data(),
                                                                       length);
        EXPECT_EQ(result, expected_result);
      }
      {
        auto result = simd_vector<TypeParam>(length * 2);
        EXPECT_NE(result, expected_result);
        TwoWayMerge::template merge_equal_length<4 * count_per_vector>(a_data.data(), b_data.data(), result.data(),
                                                                       length);
        EXPECT_EQ(result, expected_result);
      }
    }
  };
  test_merge_eq_length.template operator()<2>();
  test_merge_eq_length.template operator()<4>();
#ifdef __AVX512F__
  test_merge_eq_length.template operator()<8>();
#endif
}

TYPED_TEST(SimdTwoWayMergeTest, MergeVariableLength) {
  auto test_merge_var_length = []<std::size_t count_per_vector>() {
    using TwoWayMerge = TwoWayMerge<count_per_vector, TypeParam>;

    constexpr auto START_LENGTH = 4 * count_per_vector;
    constexpr auto MAX_LENGTH = 128 * count_per_vector;
    for (auto length = START_LENGTH; length <= MAX_LENGTH; length = length * 2) {
      auto a_data = simd_vector<TypeParam>(length);
      auto b_data = simd_vector<TypeParam>(length + 3 * count_per_vector);
      std::iota(a_data.begin(), a_data.end(), 0);
      std::iota(b_data.begin(), b_data.end(), a_data[length / 2]);

      const auto sum_of_lengths = a_data.size() + b_data.size();
      auto expected_result = simd_vector<TypeParam>{};
      std::ranges::merge(a_data, b_data, std::back_inserter(expected_result));
      {
        auto a_data_copy = a_data;
        auto b_data_copy = b_data;
        auto result = simd_vector<TypeParam>(sum_of_lengths);
        EXPECT_NE(result, expected_result);
        TwoWayMerge::template merge_variable_length<count_per_vector>(
            a_data_copy.data(), b_data_copy.data(), result.data(), a_data_copy.size(), b_data_copy.size());
        EXPECT_EQ(result, expected_result);
      }
      {
        auto a_data_copy = a_data;
        auto b_data_copy = b_data;
        auto result = simd_vector<TypeParam>(sum_of_lengths);
        EXPECT_NE(result, expected_result);
        TwoWayMerge::template merge_variable_length<2 * count_per_vector>(
            a_data_copy.data(), b_data_copy.data(), result.data(), a_data_copy.size(), b_data_copy.size());
        EXPECT_EQ(result, expected_result);
      }
      {
        auto a_data_copy = a_data;
        auto b_data_copy = b_data;
        auto result = simd_vector<TypeParam>(sum_of_lengths);
        EXPECT_NE(result, expected_result);
        TwoWayMerge::template merge_variable_length<4 * count_per_vector>(
            a_data_copy.data(), b_data_copy.data(), result.data(), a_data_copy.size(), b_data_copy.size());
        EXPECT_EQ(result, expected_result);
      }
    }
  };
  test_merge_var_length.template operator()<2>();
  test_merge_var_length.template operator()<4>();
#ifdef __AVX512F__
  test_merge_var_length.template operator()<8>();
#endif
}

}  // namespace hyrise::simd_sort
