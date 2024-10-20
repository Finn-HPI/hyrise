#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

#include "base_test.hpp"
#include "operators/join_simd_sort_merge/simd_local_sort.hpp"
#include "operators/join_simd_sort_merge/simd_utils.hpp"

namespace hyrise {

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

    auto sorted_input = input;
    std::ranges::sort(sorted_input);
    EXPECT_TRUE(std::ranges::is_sorted(sorted_input));

    auto* input_ptr = input.data();
    auto* output_ptr = temporal_storage.data();
    simd_sort_block<count_per_vector>(input_ptr, output_ptr);
    auto& sorted_data = (output_ptr == temporal_storage.data()) ? temporal_storage : input;
    EXPECT_TRUE(std::ranges::is_sorted(sorted_data));
    EXPECT_EQ(sorted_data, sorted_input);
  };
  test_sort_block.template operator()<2>();
  test_sort_block.template operator()<4>();
}

TYPED_TEST(SimdLocalSortTest, SortMultipleBlocks) {
  constexpr auto BLOCK_SIZE = block_size<TypeParam>();

  auto test_sort = []<std::size_t count_per_vector>(std::size_t scale) {
    const auto num_items = scale * BLOCK_SIZE;
    auto input = simd_vector<TypeParam>(num_items);
    auto temporal_storage = simd_vector<TypeParam>(num_items);
    std::iota(input.begin(), input.end(), 0);
    std::ranges::reverse(input);
    EXPECT_FALSE(std::ranges::is_sorted(input));

    auto sorted_input = input;
    std::ranges::sort(sorted_input);
    EXPECT_TRUE(std::ranges::is_sorted(sorted_input));

    auto* input_ptr = input.data();
    auto* output_ptr = temporal_storage.data();
    simd_sort<count_per_vector>(input_ptr, output_ptr, num_items);
    auto& sorted_data = (output_ptr == temporal_storage.data()) ? temporal_storage : input;
    EXPECT_TRUE(std::ranges::is_sorted(sorted_data));
    EXPECT_EQ(sorted_data, sorted_input);
  };
  constexpr auto MAX_SCALE = 64;
  for (auto scale = std::size_t{1}; scale < MAX_SCALE; scale *= 2) {
    test_sort.template operator()<2>(scale);
    test_sort.template operator()<4>(scale);
  }
}

TYPED_TEST(SimdLocalSortTest, ConstantLog2) {
  constexpr auto MAX_VALUE = block_size<TypeParam>();
  for (auto value = std::size_t{2}; value < MAX_VALUE; ++value) {
    EXPECT_EQ(cilog2(value), static_cast<std::size_t>(std::log2(value)));
  }
}

}  // namespace hyrise
