#pragma once

#include "operators/join_simd_sort_merge/radix_partitioning.hpp"
#include "operators/join_simd_sort_merge/two_way_merge.hpp"
#include "operators/join_simd_sort_merge/util.hpp"

namespace hyrise::multiway_merging {

struct BufferChunk {
  std::size_t start;
  std::size_t end;

  std::size_t number_of_slots() const {
    return end - start;
  }
};

class CircularBuffer {
 public:
  void write(auto&& write_func, const size_t buffer_size) {
    auto first_chunk = BufferChunk{_head, (_tail > _head) ? _tail : buffer_size};
    auto number_of_write_slots = first_chunk.number_of_slots();

    if (number_of_write_slots) {
      auto written_slots = write_func(first_chunk, _buffer + first_chunk.start);
      _update_tail(written_slots, buffer_size);
      number_of_write_slots -= written_slots;
    }

    auto second_chunk = BufferChunk{0, (_tail > _head) ? 0 : _tail};
    if (number_of_write_slots == 0 && second_chunk.number_of_slots()) {
      auto written_slots = write_func(second_chunk, _buffer);
      _update_tail(written_slots, buffer_size);
    }
  }

  void write2(const size_t buffer_size, auto&& write_func) {
    auto first_chunk = BufferChunk{_head, (_tail > _head) ? _tail : buffer_size};
    auto number_of_write_slots = first_chunk.number_of_slots();

    if (number_of_write_slots) {
      auto written_slots = write_func(std::span(_buffer + first_chunk.start, first_chunk.number_of_slots()));
      _update_tail(written_slots, buffer_size);
      number_of_write_slots -= written_slots;
    }

    auto second_chunk = BufferChunk{0, (_tail > _head) ? 0 : _tail};
    if (number_of_write_slots == 0 && second_chunk.number_of_slots()) {
      auto written_slots = write_func(std::span(_buffer, second_chunk.number_of_slots()));
      _update_tail(written_slots, buffer_size);
    }
  }

  void read(auto&& read_func, const size_t buffer_size) {
    auto read_chunk = BufferChunk{_tail, (_tail < _head) ? _head : buffer_size};
    if (!read_chunk.number_of_slots()) {
      return;
    }
    auto number_of_read_slots = read_func(read_chunk, _buffer + read_chunk.start);
    _update_head(number_of_read_slots, buffer_size);
  }

  void read_and_write_to(CircularBuffer& write_buffer, const size_t buffer_size, auto&& read_write_func) {
    while (!empty() && write_buffer.fill_count() < buffer_size) {
      auto read_func = [&](BufferChunk& read_chunk, SimdElement* input) {
        auto slots_read = size_t{0};
        auto write_func = [&](BufferChunk& write_chunk, SimdElement* output) {
          const auto slots_written = std::min(read_chunk.number_of_slots(), write_chunk.number_of_slots());
          read_write_func(std::span(input, slots_written), std::span(output, slots_written));
          slots_read += slots_written;
          return slots_written;
        };
        write_buffer.write(write_func, buffer_size);
        return slots_read;
      };
      read(read_func, buffer_size);
    }
    DebugAssert(empty() || write_buffer.fill_count() == buffer_size,
                "Either source has to be empty or desination full.");
  }

  void read_and_write_to(SimdElement*& output, const size_t buffer_size, auto&& read_write_func) {
    while (!empty()) {
      auto read_func = [&](BufferChunk& read_chunk, SimdElement* input) {
        const auto number_of_slots = read_chunk.number_of_slots();
        read_write_func(std::span(input, number_of_slots), std::span(output, number_of_slots));
        output += number_of_slots;
        return number_of_slots;
      };
      read(read_func, buffer_size);
    }
    DebugAssert(empty(), "Source has to be empty.");
  }

  void merge_and_write(CircularBuffer& inner_read_buffer, CircularBuffer& write_buffer, const size_t buffer_size,
                       auto&& merge_func) {
    auto read_left = [&](BufferChunk& left_chunk, SimdElement* left_input) {
      auto slots_read_left = size_t{0};
      auto read_right = [&](BufferChunk& right_chunk, SimdElement* right_input) {
        auto slots_read_right = size_t{0};
        auto write_func = [&](BufferChunk& write_chunk, SimdElement* output) {
          auto [count_reads_left, count_reads_right] = merge_func(std::span(left_input, left_chunk.number_of_slots()),
                                                                  std::span(right_input, right_chunk.number_of_slots()),
                                                                  std::span(output, write_chunk.number_of_slots()));
          slots_read_left += count_reads_left;
          slots_read_right += count_reads_right;
          return count_reads_left + count_reads_right;
        };
        write_buffer.write(write_func, buffer_size);
        return slots_read_right;
      };
      inner_read_buffer.read(read_right, buffer_size);
      return slots_read_left;
    };
    read(read_left, buffer_size);
  }

  void merge_and_write(CircularBuffer& inner_read_buffer, SimdElement*& output, const size_t buffer_size,
                       auto&& merge_func) {
    auto read_left = [&](BufferChunk& left_chunk, SimdElement* left_input) {
      auto slots_read_left = size_t{0};
      auto read_right = [&](BufferChunk& right_chunk, SimdElement* right_input) {
        auto slots_read_right = size_t{0};
        const auto max_output_size = left_chunk.number_of_slots() + right_chunk.number_of_slots();
        auto [count_reads_left, count_reads_right] =
            merge_func(std::span(left_input, left_chunk.number_of_slots()),
                       std::span(right_input, right_chunk.number_of_slots()), std::span(output, max_output_size));
        output += count_reads_left + count_reads_right;
        slots_read_left += count_reads_left;
        slots_read_right += count_reads_right;
        return slots_read_right;
      };
      inner_read_buffer.read(read_right, buffer_size);
      return slots_read_left;
    };
    read(read_left, buffer_size);
  }

  void set_buffer(SimdElement* init_buffer) {
    _buffer = init_buffer;
  }

  size_t fill_count() const {
    return _fill_count;
  }

  bool empty() const {
    return _fill_count == 0;
  }

 private:
  void _update_tail(size_t number_of_written_slots, const size_t buffer_size) {
    _fill_count += number_of_written_slots;
    _head += number_of_written_slots;
    if (_head == buffer_size) {
      _head = 0;
    }
  }

  void _update_head(size_t number_of_read_slots, const size_t buffer_size) {
    _fill_count -= number_of_read_slots;
    _tail += number_of_read_slots;
    if (_tail == buffer_size) {
      _tail = 0;
    }
  }

  SimdElement* _buffer{};
  std::size_t _head = 0;
  std::size_t _tail = 0;
  std::size_t _fill_count = 0;
};

template <std::size_t count_per_vector, typename T>
class MutliwayMerger {
  using Bucket = radix_partition::Bucket;
  using TwoWayMerge = simd_sort::TwoWayMerge<count_per_vector, T>;

 public:
  explicit MutliwayMerger(std::vector<std::unique_ptr<Bucket>>& sorted_buckets)
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
    auto merged_output = simd_sort::simd_vector<SimdElement>(_total_output_size);
    auto* output = merged_output.data();
    while (!_finished) {
      _finished = true;
      _load_data_from_leaves();
      _execute_ready_nodes(output);
    }
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
    for (auto node_index = first_leaf_index - 1; node_index >= ROOT; --node_index) {
      auto left_child_index = _left_child(node_index);
      auto right_child_index = _right_child(node_index);
      auto& left = _nodes[left_child_index];
      auto& right = _nodes[right_child_index];

      _done[node_index] = (_done[left_child_index] && _done[right_child_index]) && (left.empty() && right.empty());
    }

    // Initialize buffer for innner nodes.
    // TODO(finn): only set buffer for non done inner nodes and root.
    const auto count_inner_nodes = NodeIndex{first_leaf_index - 1 - ROOT};
    _fifo_buffer.resize(count_inner_nodes * _buffer_size);
    auto index = size_t{0};
    for (auto node_index = first_leaf_index - 1; node_index > ROOT; --node_index, ++index) {
      _nodes[node_index].set_buffer(_fifo_buffer.data() + index * _buffer_size);
    }
  }

  void _load_data_from_leaves() {
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

      auto& lhs_bucket = _done[leaf_index] ? empty_bucket : *_sorted_buckets[left_bucket_index];
      auto& rhs_bucket = _done[leaf_index + 1] ? empty_bucket : *_sorted_buckets[right_bucket_index];

      auto& buffer = _nodes[node_index];

      const auto num_items_read = _load_and_merge_from_leaves(buffer, lhs_bucket, rhs_bucket);

      _done[node_index] = num_items_read == 0 || (lhs_bucket.empty() && rhs_bucket.empty());
      _finished &= _done[node_index];
    }
  }

  void _execute_ready_nodes(SimdElement*& output [[maybe_unused]]) {
    const auto first_relevant_inner_node = _parent(_leaf_count - 1);
    for (auto node_index = first_relevant_inner_node; node_index > ROOT; --node_index) {
      auto left_child_index = _left_child(node_index);
      auto right_child_index = _right_child(node_index);

      auto& node_buffer = _nodes[node_index];
      auto& left_child_buffer = _nodes[left_child_index];
      auto& right_child_buffer = _nodes[right_child_index];

      if (_done[node_index]) {
        continue;
      }

      const auto children_done = _done[left_child_index] || _done[right_child_index];

      if ((children_done || node_buffer.fill_count() < _read_threshold) && node_buffer.fill_count() < _buffer_size) {
        if (children_done ||
            (right_child_buffer.fill_count() >= _read_threshold && left_child_buffer.fill_count() >= _read_threshold)) {
          _merge_children_into_node(node_buffer, left_child_buffer, right_child_buffer, _done[left_child_index],
                                    _done[right_child_index]);
        }
        _done[node_index] = (_done[left_child_index] && _done[right_child_index]) &&
                            (left_child_buffer.empty() && right_child_buffer.empty());
      }

      _finished &= _done[node_index];
    }
    const auto left_index = _left_child(ROOT);
    const auto right_index = _right_child(ROOT);
    _merge_children_into_output(output, _nodes[left_index], _nodes[right_index], _done[left_index], _done[right_index]);
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
    buffer.write2(_buffer_size, merge_buckets_into_buffer);

    if (buffer.fill_count() == _buffer_size) {
      return total_number_of_writes;
    }

    auto read_and_write_remaining = [&](Bucket& bucket) {
      auto write_func = [&](std::span<SimdElement> output) {
        const auto number_of_writes = std::min(output.size(), bucket.size);
        output = output.subspan(0, number_of_writes);
        auto* input = bucket.data;
        for (auto& element : output) {
          element = *input;
          ++input;
        }
        bucket.retrieve_elements(number_of_writes);
        total_number_of_writes += number_of_writes;
        return number_of_writes;
      };
      buffer.write2(_buffer_size, write_func);
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

  void _merge_children_into_node(CircularBuffer& node_buffer, CircularBuffer& left, CircularBuffer& right,
                                 bool left_child_done, bool right_child_done) {
    if (right_child_done && right.empty()) {
      if (!left_child_done && !left.empty()) {
        _copy_buffer(node_buffer, left);
        return;
      }
    } else if (left_child_done && left.empty()) {
      if (!right_child_done && !right.empty()) {
        _copy_buffer(node_buffer, right);
        return;
      }
    }

    while (node_buffer.fill_count() < _buffer_size && (!left.empty() && !right.empty())) {
      left.merge_and_write(right, node_buffer, _buffer_size, _merge_children);
    }

    const bool both_children_done = right_child_done && left_child_done;
    if (!both_children_done || node_buffer.fill_count() == _buffer_size) {
      return;
    }

    auto write_elements = [](std::span<SimdElement> src, std::span<SimdElement> destination) {
      std::ranges::copy(src, destination.begin());
    };

    if (!left.empty()) {
      left.read_and_write_to(node_buffer, _buffer_size, write_elements);
    } else {
      right.read_and_write_to(node_buffer, _buffer_size, write_elements);
    }
  }

  void _merge_children_into_output(SimdElement*& output, CircularBuffer& left, CircularBuffer& right,
                                   bool left_child_done, bool right_child_done) {
    if (right_child_done && right.empty()) {
      if (!left_child_done && !left.empty()) {
        _write_buffer_to_output(output, left);
        DebugAssert(left.empty(), "Left buffer should be empty.");
        return;
      }
    } else if (left_child_done && left.empty()) {
      if (!right_child_done && !right.empty()) {
        _write_buffer_to_output(output, right);
        DebugAssert(right.empty(), "Right buffer should be empty.");
        return;
      }
    }

    while (!left.empty() && !right.empty()) {
      left.merge_and_write(right, output, _buffer_size, _merge_children);
    }

    const bool both_children_done = right_child_done && left_child_done;
    if (!both_children_done) {
      return;
    }

    auto write_elements = [](std::span<SimdElement> src, std::span<SimdElement> destination) {
      std::ranges::copy(src, destination.begin());
    };

    if (!left.empty()) {
      left.read_and_write_to(output, _buffer_size, write_elements);
    } else {
      right.read_and_write_to(output, _buffer_size, write_elements);
    }
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

  static NodeIndex _parent(NodeIndex node) { return node / 2; }
  static NodeIndex _left_child(NodeIndex node) { return 2 * node; }
  static NodeIndex _right_child(NodeIndex node) { return 2 * node + 1; }

  // clang-format on
  std::size_t _read_threshold = 800;
  std::size_t _buffer_size = 1600;

  bool _finished = false;
  size_t _leaf_count;
  std::vector<std::unique_ptr<Bucket>> _sorted_buckets;
  std::vector<CircularBuffer> _nodes;
  std::vector<bool> _done;
  simd_sort::simd_vector<SimdElement> _fifo_buffer;
  size_t _total_output_size;
};
}  // namespace hyrise::multiway_merging
