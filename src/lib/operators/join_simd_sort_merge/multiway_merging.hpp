#pragma once

#include "operators/join_simd_sort_merge/circular_buffer.hpp"
#include "operators/join_simd_sort_merge/radix_partitioning.hpp"
#include "operators/join_simd_sort_merge/two_way_merge.hpp"
#include "operators/join_simd_sort_merge/util.hpp"

namespace hyrise::multiway_merging {

template <size_t count_per_vector, typename T>
class MultiwayMerger {
  using Bucket = radix_partition::Bucket;
  using CircularBuffer = circular_buffer::CircularBuffer;
  using TwoWayMerge = simd_sort::TwoWayMerge<count_per_vector, T>;

 public:
  explicit MultiwayMerger(std::vector<std::unique_ptr<Bucket>>& sorted_buckets)
      : _leaf_count(std::bit_ceil(sorted_buckets.size())),
        _sorted_buckets(std::move(sorted_buckets)),
        _nodes(2 * _leaf_count),
        _done(_nodes.size()),
        _total_output_size(std::accumulate(_sorted_buckets.begin(), _sorted_buckets.end(), size_t{0},
                                           [](size_t sum, const auto& bucket) {
                                             return sum + bucket->size;
                                           })) {
    _initialize();
  }

  simd_sort::simd_vector<SimdElement> merge() {
    // Begin handling of edge cases.
    if (_sorted_buckets.empty()) {
      return {};
    }

    auto merged_output = simd_sort::simd_vector<SimdElement>(_total_output_size);

    if (_sorted_buckets.size() == ONE_REMAINING) {
      std::ranges::copy(_sorted_buckets[0]->elements(), merged_output.begin());
      return merged_output;
    }

    if (_leaf_count == TWO_REMAINING) {
      auto& left = _sorted_buckets[0];
      auto& right = _sorted_buckets[1];
      TwoWayMerge::template merge_variable_length<count_per_vector * 4>(
          left->template begin<T>(), right->template begin<T>(), reinterpret_cast<T*>(merged_output.data()), left->size,
          right->size);

      DebugAssert(std::is_sorted(merged_output.begin(), merged_output.end(),
                                 [](auto& left, auto& right) {
                                   return *reinterpret_cast<T*>(&left) < *reinterpret_cast<T*>(&right);
                                 }),
                  "Merged output is not sorted.");

      return merged_output;
    }

    // End handling of edge cases.

    _execute(merged_output.data());

    DebugAssert(std::is_sorted(merged_output.begin(), merged_output.end(),
                               [](auto& left, auto& right) {
                                 return *reinterpret_cast<T*>(&left) < *reinterpret_cast<T*>(&right);
                               }),
                "Merged output is not sorted.");

    return merged_output;
  }

 private:
  void _initialize() {
    const auto num_buckets = _sorted_buckets.size();
    const auto num_nodes = _nodes.size();
    const auto first_leaf_index = NodeIndex{_leaf_count};

    // In case we have rounded up _leaf_count to the next power of two, we can set the additional leaf nodes as done.
    for (auto node_index = first_leaf_index + num_buckets; node_index < num_nodes; ++node_index) {
      _done[node_index] = true;
    }

    // Initialize done values for the inner nodes and the root node.
    auto count_non_done_inner_nodes = size_t{0};
    for (auto node_index = first_leaf_index - 1; node_index >= ROOT; --node_index) {
      auto left_child_index = _left_child(node_index);
      auto right_child_index = _right_child(node_index);
      auto& left = _nodes[left_child_index];
      auto& right = _nodes[right_child_index];

      _done[node_index] = (_done[left_child_index] && _done[right_child_index]) && (left.empty() && right.empty());
      count_non_done_inner_nodes += static_cast<size_t>(!_done[node_index]);
    }

    if (!_done[ROOT]) {
      --count_non_done_inner_nodes;
    }

    if (count_non_done_inner_nodes == 0) {
      return;
    }

    // Setup buffers for innner nodes.
    constexpr auto CACHE_USAGE = 0.8;
    constexpr auto AVAILABLE_L2_CACHE = static_cast<size_t>(L2_CACHE_SIZE * CACHE_USAGE);

    _buffer_size = (AVAILABLE_L2_CACHE / sizeof(SimdElement)) / count_non_done_inner_nodes;
    _read_threshold = _buffer_size / 2;

    _fifo_buffer.resize(count_non_done_inner_nodes * _buffer_size);
    auto buffer_index = size_t{0};
    for (auto node_index = first_leaf_index - 1; node_index > ROOT; --node_index) {
      if (_done[node_index]) {
        continue;
      }
      _nodes[node_index].set_buffer(_fifo_buffer.data() + buffer_index * _buffer_size);
      ++buffer_index;
    }
  }

  void _execute(SimdElement* output) {
    while (!_finished) {
      _finished = true;
      _load_from_leaves();
      _execute_ready_nodes(output);
    }
  }

  void _load_from_leaves() {
    const auto first_leaf_index = NodeIndex{_leaf_count};
    const auto num_nodes = _nodes.size();

    static auto empty_bucket = Bucket{nullptr, 0};

    for (auto leaf_index = first_leaf_index; leaf_index < num_nodes; leaf_index += 2) {
      auto node_index = _parent(leaf_index);
      if (_done[node_index] || _nodes[node_index].fill_count() >= _read_threshold) {
        continue;
      }
      const auto left_bucket_index = leaf_index - first_leaf_index;
      const auto right_bucket_index = left_bucket_index + 1;

      auto& left_bucket = _done[leaf_index] ? empty_bucket : *_sorted_buckets[left_bucket_index];
      auto& right_bucket = _done[leaf_index + 1] ? empty_bucket : *_sorted_buckets[right_bucket_index];

      auto& buffer = _nodes[node_index];

      const auto num_items_read = _load_and_merge_from_leaves(buffer, left_bucket, right_bucket);

      DebugAssert(buffer.debug_is_sorted<T>(_buffer_size), "After merging from leaves, the buffer is not sorted.");

      _done[node_index] = num_items_read == 0 || (left_bucket.empty() && right_bucket.empty());
      _finished &= _done[node_index];
    }
  }

  void _execute_ready_nodes(SimdElement*& output) {
    const auto first_relevant_inner_node = _parent(_leaf_count - 1);
    // After loading the first level of inner-nodes with data from the leaves, we check all remaining inner-nodes
    // from bottom to top if they are ready to be executed. If that is the case we execute them.
    for (auto node_index = first_relevant_inner_node; node_index > ROOT; --node_index) {
      auto left_child_index = _left_child(node_index);
      auto right_child_index = _right_child(node_index);

      auto& buffer = _nodes[node_index];
      auto& left_child_buffer = _nodes[left_child_index];
      auto& right_child_buffer = _nodes[right_child_index];

      if (_done[node_index]) {
        continue;
      }

      const auto children_done = _done[left_child_index] || _done[right_child_index];
      if ((children_done || buffer.fill_count() < _read_threshold) && buffer.fill_count() < _buffer_size) {
        if (children_done ||
            (right_child_buffer.fill_count() >= _read_threshold && left_child_buffer.fill_count() >= _read_threshold)) {
          DebugAssert(left_child_buffer.debug_is_sorted<T>(_buffer_size), "Left child buffer is not sorted.");
          DebugAssert(right_child_buffer.debug_is_sorted<T>(_buffer_size), "Right child buffer is not sorted.");

          _merge_children_into_node(buffer, left_child_buffer, right_child_buffer, _done[left_child_index],
                                    _done[right_child_index]);
          DebugAssert(buffer.debug_is_sorted<T>(_buffer_size),
                      "After merging from inner child nodes, the buffer is not sorted.");
        }
        _done[node_index] = (_done[left_child_index] && _done[right_child_index]) &&
                            (left_child_buffer.empty() && right_child_buffer.empty());
      }
      _finished &= _done[node_index];
    }

    // Finally we do a final merge of the left and right child of the ROOT and write the merged elemets to output.
    const auto left_index = _left_child(ROOT);
    const auto right_index = _right_child(ROOT);

#if HYRISE_DEBUG
    auto* output_before_merging = output;
    DebugAssert(_nodes[left_index].debug_is_sorted<T>(_buffer_size), "Left child buffer is not sorted.");
    DebugAssert(_nodes[right_index].debug_is_sorted<T>(_buffer_size), "Right child buffer is not sorted.");

    _merge_children_into_output(output, _nodes[left_index], _nodes[right_index], _done[left_index], _done[right_index]);

    auto written_output = std::span(output_before_merging, output);
    DebugAssert(std::is_sorted(written_output.begin(), written_output.end(),
                               [](auto& left, auto& right) {
                                 return *reinterpret_cast<T*>(&left) < *reinterpret_cast<T*>(&right);
                               }),
                "Newly written output is not sorted.");
#else
    _merge_children_into_output(output, _nodes[left_index], _nodes[right_index], _done[left_index], _done[right_index]);
#endif
  }

  size_t _load_and_merge_from_leaves(CircularBuffer& buffer, Bucket& left_bucket, Bucket& right_bucket) {
    if (left_bucket.empty() && right_bucket.empty()) {
      return 0;
    }
    auto total_number_of_writes = size_t{0};

    auto merge_buckets_into_buffer = [&](std::span<SimdElement> output) {
      auto count_reads_left = size_t{0};
      auto count_reads_right = size_t{0};
      auto count_writes = size_t{0};

      TwoWayMerge::template merge_multiway_merge_nodes<count_per_vector * 4>(
          reinterpret_cast<T*>(left_bucket.data), reinterpret_cast<T*>(right_bucket.data),
          reinterpret_cast<T*>(output.data()), count_reads_left, count_reads_right, count_writes, left_bucket.size,
          right_bucket.size, output.size());

      left_bucket.retrieve_elements(count_reads_left);
      right_bucket.retrieve_elements(count_reads_right);
      total_number_of_writes += count_writes;

      return count_writes;
    };
    buffer.write(_buffer_size, merge_buckets_into_buffer);

    if (buffer.fill_count() == _buffer_size) {
      return total_number_of_writes;
    }

    auto read_and_write_remaining = [&](Bucket& bucket) {
      auto write_func = [&](std::span<SimdElement> output) {
        const auto number_of_writes = std::min(output.size(), bucket.size);
        auto input = bucket.elements().subspan(0, number_of_writes);
        output = output.subspan(0, number_of_writes);

        std::ranges::copy(input, output.begin());
        bucket.retrieve_elements(number_of_writes);

        total_number_of_writes += number_of_writes;
        return number_of_writes;
      };
      buffer.write(_buffer_size, write_func);
    };

    if (!left_bucket.empty()) {
      read_and_write_remaining(left_bucket);
    } else {
      read_and_write_remaining(right_bucket);
    }

    DebugAssert(buffer.fill_count() == _buffer_size || (left_bucket.empty() && right_bucket.empty()),
                "Leaves still have elements, but buffer is not fully filled.");

    return total_number_of_writes;
  }

  static inline std::pair<size_t, size_t> _merge_children(std::span<SimdElement> left_input,
                                                          std::span<SimdElement> right_input,
                                                          std::span<SimdElement> output) {
    auto count_reads_left = size_t{0};
    auto count_reads_right = size_t{0};
    auto count_writes = size_t{0};

    TwoWayMerge::template merge_multiway_merge_nodes<count_per_vector * 4>(
        reinterpret_cast<T*>(left_input.data()), reinterpret_cast<T*>(right_input.data()),
        reinterpret_cast<T*>(output.data()), count_reads_left, count_reads_right, count_writes, left_input.size(),
        right_input.size(), output.size());

    return {count_reads_left, count_reads_right};
  }

  template <typename Output>
  bool _merge_other_if_one_done(Output& output, CircularBuffer& left, CircularBuffer& right, bool left_child_done,
                                bool right_child_done, bool is_write_to_output [[maybe_unused]] = false) {
    if (right_child_done && right.empty()) {
      if (!left_child_done && !left.empty()) {
        _write_to_destination(output, left);
        DebugAssert(left.empty() || !is_write_to_output, "Left buffer should be empty if written to final output.");
        return true;
      }
    } else if (left_child_done && left.empty()) {
      if (!right_child_done && !right.empty()) {
        _write_to_destination(output, right);
        DebugAssert(right.empty() || !is_write_to_output, "Right buffer should be empty if written to final output.");
        return true;
      }
    }
    return false;
  }

  template <typename Output>
  void _write_elements_from_remaining_bucket(Output& output, CircularBuffer& left, CircularBuffer& right) {
    auto write_elements = [](std::span<SimdElement> src, std::span<SimdElement> destination) {
      std::ranges::copy(src, destination.begin());
    };
    if (!left.empty()) {
      left.read_and_write_to(output, _buffer_size, write_elements);
    } else {
      right.read_and_write_to(output, _buffer_size, write_elements);
    }
  }

  void _merge_children_into_node(CircularBuffer& buffer, CircularBuffer& left, CircularBuffer& right,
                                 bool left_child_done, bool right_child_done) {
    if (_merge_other_if_one_done(buffer, left, right, left_child_done, right_child_done)) {
      return;
    }

    while (buffer.fill_count() < _buffer_size && (!left.empty() && !right.empty())) {
      left.merge_and_write(right, buffer, _buffer_size, _merge_children);
    }

    DebugAssert((left.empty() || right.empty()) || buffer.fill_count() == _buffer_size,
                "After merging one child buffer has to be empty or the node buffer full.");

    const bool both_children_done = right_child_done && left_child_done;
    if (!both_children_done || buffer.fill_count() == _buffer_size) {
      return;
    }

    _write_elements_from_remaining_bucket(buffer, left, right);
  }

  void _merge_children_into_output(SimdElement*& output, CircularBuffer& left, CircularBuffer& right,
                                   bool left_child_done, bool right_child_done) {
    if (_merge_other_if_one_done(output, left, right, left_child_done, right_child_done, true)) {
      return;
    }

    while (!left.empty() && !right.empty()) {
      left.merge_and_write(right, output, _buffer_size, _merge_children);
    }

    DebugAssert(left.empty() || right.empty(), "After merging one child buffer has to be empty.");

    const bool both_children_done = right_child_done && left_child_done;
    if (!both_children_done) {
      return;
    }

    _write_elements_from_remaining_bucket(output, left, right);
  }

  template <typename Output>
  void _write_to_destination(Output& destination, CircularBuffer& source) {
    source.read_and_write_to(destination, _buffer_size,
                             [](std::span<SimdElement> src, std::span<SimdElement> destination) {
                               std::ranges::copy(src, destination.begin());
                             });
  }

  void _copy_buffer(CircularBuffer& destination, CircularBuffer& source) {
    source.read_and_write_to(destination, _buffer_size,
                             [](std::span<SimdElement> src, std::span<SimdElement> destination) {
                               std::ranges::copy(src, destination.begin());
                             });
  }

  void _write_buffer_to_output(SimdElement*& output, CircularBuffer& source) {
    source.read_and_write_to(output, _buffer_size, [](std::span<SimdElement> src, std::span<SimdElement> destination) {
      std::ranges::copy(src, destination.begin());
    });
  }

  using NodeIndex = size_t;

  // clang-format off
  
  static constexpr NodeIndex ROOT = 1;
  static constexpr auto ONE_REMAINING = 1;
  static constexpr auto TWO_REMAINING = 2;

  static NodeIndex _parent(NodeIndex node) { return node / 2; }
  static NodeIndex _left_child(NodeIndex node) { return 2 * node; }
  static NodeIndex _right_child(NodeIndex node) { return 2 * node + 1; }

  // clang-format on

  size_t _leaf_count;
  std::vector<std::unique_ptr<Bucket>> _sorted_buckets;
  std::vector<CircularBuffer> _nodes;
  std::vector<bool> _done;
  size_t _total_output_size;

  bool _finished = false;

  size_t _buffer_size{};
  size_t _read_threshold{};

  simd_sort::simd_vector<SimdElement> _fifo_buffer;
};
}  // namespace hyrise::multiway_merging
