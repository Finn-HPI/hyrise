#include <random>

#include "base_test.hpp"
#include "operators/join_simd_sort_merge/multiway_merging.hpp"
#include "operators/join_simd_sort_merge/radix_partitioning.hpp"
#include "operators/join_simd_sort_merge/simd_utils.hpp"

namespace hyrise {

class MultiwayMergeTest : public BaseTest {};

template <typename T>
simd_sort::simd_vector<T> generate_random_vector(size_t max_size) {
  std::random_device rnd;
  std::mt19937 gen(rnd());

  // Determine the vector length as a random value between 70% and 100% of max_size
  std::uniform_int_distribution<size_t> size_dist(static_cast<size_t>(static_cast<double>(max_size) * 0.7), max_size);
  size_t vec_size = size_dist(gen);

  simd_sort::simd_vector<T> random_vector(vec_size);

  if constexpr (std::is_same_v<T, int64_t>) {
    std::uniform_int_distribution<int64_t> value_dist(INT64_MIN, INT64_MAX);
    for (auto& value : random_vector) {
      value = value_dist(gen);
    }
  } else if constexpr (std::is_same_v<T, double>) {
    std::uniform_real_distribution<double> value_dist(-1.0e10, 1.0e10);
    for (auto& value : random_vector) {
      value = value_dist(gen);
    }
  }

  return random_vector;
}

template <typename T>
std::vector<radix_partition::Bucket> generate_sorted_buckets(const size_t num_buckets,
                                                             std::vector<simd_sort::simd_vector<T>>& bucket_data,
                                                             const size_t max_size) {
  auto buckets = std::vector<radix_partition::Bucket>(num_buckets);

  for (auto bucket_index = size_t{0}; bucket_index < num_buckets; ++bucket_index) {
    auto& bucket = buckets[bucket_index];
    // Allocate bucket data
    bucket_data[bucket_index] = std::move(generate_random_vector<T>(max_size));
    bucket.data = reinterpret_cast<SimdElement*>(bucket_data[bucket_index].data());
    bucket.size = bucket_data[bucket_index].size();
    // Sort internal bucket data
    std::sort(bucket.template begin<int64_t>(), bucket.template end<int64_t>());
    EXPECT_TRUE(std::is_sorted(bucket.template begin<T>(), bucket.template end<T>()));
  }
  return buckets;
}

template <typename T>
simd_sort::simd_vector<SimdElement> merge_sorted_buckets_naive(std::vector<radix_partition::Bucket>& sorted_buckets) {
  if (sorted_buckets.empty())
    return {};

  auto result = simd_sort::simd_vector<T>{};
  std::ranges::copy(sorted_buckets[0].template begin<T>(), sorted_buckets[0].template end<T>(),
                    std::back_inserter(result));

  for (size_t i = 1; i < sorted_buckets.size(); ++i) {
    simd_sort::simd_vector<T> temp_result;
    std::merge(result.begin(), result.end(), sorted_buckets[i].template begin<T>(), sorted_buckets[i].template end<T>(),
               std::back_inserter(temp_result));
    result = std::move(temp_result);
  }

  const auto output_size = result.size();
  auto final_result = simd_sort::simd_vector<SimdElement>(output_size);
  for (auto index = size_t{0}; index < output_size; ++index) {
    final_result[index] = *reinterpret_cast<SimdElement*>(&result[index]);
  }
  return final_result;
}

TEST_F(MultiwayMergeTest, LeafCountNumberOfTwo) {
  using SortingType = int64_t;
  constexpr auto MAX_COUNT = 256;
  constexpr auto COUNT_PER_VECTOR = 4;
  constexpr auto MAX_BUCKET_SIZE = 4000;

  for (auto count_leaves = size_t{4}; count_leaves <= MAX_COUNT; count_leaves *= 2) {
    std::cout << "Test merging with sorted bucket count: " << count_leaves << std::endl;

    auto bucket_data = std::vector<simd_sort::simd_vector<SortingType>>(count_leaves);
    auto sorted_buckets = generate_sorted_buckets<SortingType>(count_leaves, bucket_data, MAX_BUCKET_SIZE);

    // Naive merging.
    std::cout << "Run naive merging" << std::endl;

    auto merged_output = merge_sorted_buckets_naive<SortingType>(sorted_buckets);
    EXPECT_TRUE(std::is_sorted(merged_output.begin(), merged_output.end(), [](auto& left, auto& right) {
      return *reinterpret_cast<SortingType*>(&left) < *reinterpret_cast<SortingType*>(&right);
    }));

    // Multiway merging.
    std::cout << "Run multiway merging" << std::endl;

    auto sorted_bucket_ptrs = std::vector<std::unique_ptr<radix_partition::Bucket>>();
    sorted_bucket_ptrs.reserve(sorted_buckets.size());
    for (auto& bucket : sorted_buckets) {
      sorted_bucket_ptrs.push_back(std::make_unique<radix_partition::Bucket>(bucket));
    }
    auto multiway_merger = multiway_merging::MutliwayMerger<COUNT_PER_VECTOR, SortingType>(sorted_bucket_ptrs);
    auto multiway_merged_output = multiway_merger.merge();

    EXPECT_TRUE(
        std::is_sorted(multiway_merged_output.begin(), multiway_merged_output.end(), [](auto& left, auto& right) {
          return *reinterpret_cast<SortingType*>(&left) < *reinterpret_cast<SortingType*>(&right);
        }));

    EXPECT_EQ(multiway_merged_output, merged_output);
  }
}

TEST_F(MultiwayMergeTest, LeafCountNotNumberOfTwo) {
  using SortingType = int64_t;
  constexpr auto COUNT_PER_VECTOR = 4;
  constexpr auto MAX_BUCKET_SIZE = 4000;

  auto bucket_counts = std::vector<size_t>{5, 7, 14, 15, 30, 75, 90, 127, 160, 200, 212, 245};

  for (auto count_leaves : bucket_counts) {
    std::cout << "Test merging with sorted bucket count: " << count_leaves << std::endl;

    auto bucket_data = std::vector<simd_sort::simd_vector<SortingType>>(count_leaves);
    auto sorted_buckets = generate_sorted_buckets<SortingType>(count_leaves, bucket_data, MAX_BUCKET_SIZE);

    // Naive merging.
    std::cout << "Run naive merging" << std::endl;

    auto merged_output = merge_sorted_buckets_naive<SortingType>(sorted_buckets);
    EXPECT_TRUE(std::is_sorted(merged_output.begin(), merged_output.end(), [](auto& left, auto& right) {
      return *reinterpret_cast<SortingType*>(&left) < *reinterpret_cast<SortingType*>(&right);
    }));

    // Multiway merging.
    std::cout << "Run multiway merging" << std::endl;

    auto sorted_bucket_ptrs = std::vector<std::unique_ptr<radix_partition::Bucket>>();
    sorted_bucket_ptrs.reserve(sorted_buckets.size());
    for (auto& bucket : sorted_buckets) {
      sorted_bucket_ptrs.push_back(std::make_unique<radix_partition::Bucket>(bucket));
    }
    auto multiway_merger = multiway_merging::MutliwayMerger<COUNT_PER_VECTOR, SortingType>(sorted_bucket_ptrs);
    auto multiway_merged_output = multiway_merger.merge();

    EXPECT_TRUE(
        std::is_sorted(multiway_merged_output.begin(), multiway_merged_output.end(), [](auto& left, auto& right) {
          return *reinterpret_cast<SortingType*>(&left) < *reinterpret_cast<SortingType*>(&right);
        }));

    EXPECT_EQ(multiway_merged_output, merged_output);
  }
}

}  // namespace hyrise
