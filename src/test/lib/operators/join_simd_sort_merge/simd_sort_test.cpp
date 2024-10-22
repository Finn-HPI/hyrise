#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

#include "base_test.hpp"
#include "operators/join_simd_sort_merge/simd_sort.hpp"

namespace hyrise::simd_sort {

using data_type_list = testing::Types<double, int64_t, uint64_t>;

template <class>
class SimdLocalSortTest : public BaseTest {};

TYPED_TEST_SUITE(SimdLocalSortTest, data_type_list);

TYPED_TEST(SimdLocalSortTest, SortBlock) {
  constexpr auto BLOCK_SIZE = block_size<TypeParam>();
  auto test_sort_block = []<std::size_t count_per_vector>() {
    auto input = simd_vector<TypeParam>(BLOCK_SIZE);
    auto temporal_storage = simd_vector<TypeParam>(BLOCK_SIZE);
    std::iota(input.begin(), input.end(), 0);
    std::ranges::reverse(input);
    EXPECT_FALSE(std::ranges::is_sorted(input));

    auto expected_output = input;
    std::ranges::sort(expected_output);
    EXPECT_TRUE(std::ranges::is_sorted(expected_output));

    auto chunk = DataChunk{input.data(), temporal_storage.data(), BLOCK_SIZE};
    simd_sort_chunk<count_per_vector>(chunk);
    auto& sorted_data = (chunk.output == temporal_storage.data()) ? temporal_storage : input;

    EXPECT_TRUE(std::ranges::is_sorted(sorted_data));
    EXPECT_EQ(sorted_data, expected_output);
  };
  test_sort_block.template operator()<2>();
  test_sort_block.template operator()<4>();
}

TYPED_TEST(SimdLocalSortTest, SortIncompleteBlock) {
  constexpr auto BLOCK_SIZE = block_size<TypeParam>();
  auto test_sort_block = []<std::size_t count_per_vector>(const double scale) {
    const auto num_items = static_cast<std::size_t>(static_cast<double>(BLOCK_SIZE) * scale);
    auto input = simd_vector<TypeParam>(num_items);
    auto temporal_storage = simd_vector<TypeParam>(num_items);
    std::iota(input.begin(), input.end(), 0);
    std::ranges::reverse(input);
    EXPECT_FALSE(std::ranges::is_sorted(input));

    auto expected_output = input;
    std::ranges::sort(expected_output);
    EXPECT_TRUE(std::ranges::is_sorted(expected_output));

    auto chunk = DataChunk{input.data(), temporal_storage.data(), num_items};
    simd_sort_incomplete_chunk<count_per_vector>(chunk);
    auto& sorted_data = (chunk.output == temporal_storage.data()) ? temporal_storage : input;

    EXPECT_TRUE(std::ranges::is_sorted(sorted_data));
    EXPECT_EQ(sorted_data, expected_output);
  };
  constexpr auto NUM_FRACTIONS = 80;
  constexpr auto BASE_FACTOR = double{1} / static_cast<double>(NUM_FRACTIONS);
  for (auto scale_factor = std::size_t{1}; scale_factor < NUM_FRACTIONS; ++scale_factor) {
    const auto scale = BASE_FACTOR * static_cast<double>(scale_factor);
    test_sort_block.template operator()<2>(scale);
    test_sort_block.template operator()<4>(scale);
  }
}

TYPED_TEST(SimdLocalSortTest, SortComplete) {
  constexpr auto BLOCK_SIZE = block_size<TypeParam>();

  auto test_sort = []<std::size_t count_per_vector>(double scale) {
    const auto num_items = static_cast<std::size_t>(scale * BLOCK_SIZE);
    auto input = simd_vector<TypeParam>(num_items);
    auto temporal_storage = simd_vector<TypeParam>(num_items);
    std::iota(input.begin(), input.end(), 0);
    std::ranges::reverse(input);
    EXPECT_FALSE(std::ranges::is_sorted(input));

    auto expected_output = input;
    std::ranges::sort(expected_output);
    EXPECT_TRUE(std::ranges::is_sorted(expected_output));
    EXPECT_FALSE(std::ranges::is_sorted(input));

    auto* input_ptr = input.data();
    auto* output_ptr = temporal_storage.data();
    simd_sort<count_per_vector>(input_ptr, output_ptr, num_items);

    auto& sorted_data = (output_ptr == temporal_storage.data()) ? temporal_storage : input;
    EXPECT_EQ(sorted_data.size(), num_items);
    EXPECT_TRUE(std::ranges::is_sorted(sorted_data));
    EXPECT_EQ(sorted_data, expected_output);
  };
  constexpr auto MAX_SCALE = 32;
  constexpr auto NUM_FRACTIONS = 10;
  for (auto scale = std::size_t{1}; scale < MAX_SCALE; scale *= 2) {
    //  Test with integer multiples of BLOCK_SIZE.
    test_sort.template operator()<2>(static_cast<double>(scale));
    test_sort.template operator()<4>(static_cast<double>(scale));
    // Test with fractional multiples of BlOCK_SIZE.
    constexpr auto BASE_FACTOR = double{1} / static_cast<double>(NUM_FRACTIONS);
    for (auto scale_factor = std::size_t{1}; scale_factor < NUM_FRACTIONS; ++scale_factor) {
      const auto fractional_scale_summand = BASE_FACTOR * static_cast<double>(scale_factor);
      test_sort.template operator()<2>(static_cast<double>(scale) + fractional_scale_summand);
      test_sort.template operator()<4>(static_cast<double>(scale) + fractional_scale_summand);
    }
  }
}
}  // namespace hyrise::simd_sort
