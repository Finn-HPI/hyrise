#pragma once

#include "operators/join_simd_sort_merge/radix_partitioning.hpp"

namespace hyrise::k_way_merge {
template <typename T>
class KWayMerge {
  using Bucket = radix_partition::Bucket;

 public:
  using value_type = T;

  explicit KWayMerge(std::vector<std::unique_ptr<Bucket>>& sorted_buckets)
      : _sorted_buckets(std::move(sorted_buckets)) {}

  simd_sort::simd_vector<SimdElement> merge() {
    if (_sorted_buckets.empty()) {
      return {};
    }

    auto leaf_nodes = std::vector<std::pair<T*, T*>>(_sorted_buckets.size());
    std::transform(_sorted_buckets.begin(), _sorted_buckets.end(), leaf_nodes.begin(), [](const auto& bucket) {
      return std::make_pair(bucket->template begin<T>(), bucket->template end<T>());
    });
    // Remove empty leaves.
    leaf_nodes.erase(std::remove_if(leaf_nodes.begin(), leaf_nodes.end(),
                                    [](const auto& leaf) {
                                      return leaf.first == leaf.second;
                                    }),
                     leaf_nodes.end());

    auto total_output_size =
        std::accumulate(_sorted_buckets.begin(), _sorted_buckets.end(), size_t{0}, [](size_t sum, const auto& bucket) {
          return sum + bucket->size;
        });

    auto output = simd_sort::simd_vector<SimdElement>();
    output.reserve(total_output_size);

    auto cmp = [](const auto& lhs, const auto& rhs) {
      return *(lhs.first) > *(rhs.first);
    };

    std::ranges::make_heap(leaf_nodes, cmp);
    while (!leaf_nodes.empty()) {
      std::ranges::pop_heap(leaf_nodes, cmp);
      auto [current_iterator, end] = leaf_nodes.back();
      leaf_nodes.pop_back();

      output.push_back(*reinterpret_cast<SimdElement*>(current_iterator));

      if (++current_iterator != end) {
        leaf_nodes.emplace_back(current_iterator, end);
        std::ranges::push_heap(leaf_nodes, cmp);
      }
    }

    DebugAssert(std::is_sorted(output.begin(), output.end(),
                               [](auto& lhs, auto& rhs) {
                                 return *reinterpret_cast<T*>(&lhs) < *reinterpret_cast<T*>(&rhs);
                               }),
                "Merged output is not sorted.");

    return output;
  }

 private:
  std::vector<std::unique_ptr<Bucket>> _sorted_buckets;
};
}  // namespace hyrise::k_way_merge
