#pragma once

#include "operators/join_simd_sort_merge/radix_partitioning.hpp"
#include "operators/join_simd_sort_merge/two_way_merge.hpp"
#include "operators/join_simd_sort_merge/util.hpp"

namespace hyrise {

struct BufferChunk {
  std::size_t start;
  std::size_t end;

  std::size_t number_of_slots() const {
    return end - start;
  }
};

struct CircularBuffer {
  SimdElement* buffer{};
  std::size_t head = 0;
  std::size_t tail = 0;
  std::size_t fill_count = 0;

  void write_into_buffer(auto&& write_func, const size_t buffer_size) {
    auto first_chunk = BufferChunk{tail, (head > tail) ? head : buffer_size};
    auto number_of_write_slots = first_chunk.number_of_slots();

    if (number_of_write_slots) {
      auto written_slots = write_func(first_chunk, buffer + first_chunk.start);
      _update_tail(written_slots, buffer_size);
      number_of_write_slots -= written_slots;
    }

    auto second_chunk = BufferChunk{0, (head > tail) ? 0 : head};
    if (number_of_write_slots == 0 && second_chunk.number_of_slots()) {
      auto written_slots = write_func(second_chunk, buffer);
      _update_tail(written_slots, buffer_size);
    }
  }

  void read_from_buffer(auto&& read_func, const size_t buffer_size) {
    auto read_chunk = BufferChunk{head, (head < tail) ? tail : buffer_size};
    if (!read_chunk.number_of_slots()) {
      return;
    }
    auto number_of_read_slots = read_func(read_chunk, buffer + read_chunk.start);
    _update_head(number_of_read_slots, buffer_size);
  }

 private:
  void _update_tail(size_t number_of_written_slots, const size_t buffer_size) {
    fill_count += number_of_written_slots;
    tail += number_of_written_slots;
    if (tail == buffer_size) {
      tail = 0;
    }
  }

  void _update_head(size_t number_of_read_slots, const size_t buffer_size) {
    fill_count -= number_of_read_slots;
    head += number_of_read_slots;
    if (head == buffer_size) {
      head = 0;
    }
  }
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
        _done(_nodes.size()) {
    const auto num_buckets = _sorted_buckets.size();
    const auto first_leaf_index = _leaf_count;

    const auto num_nodes = _nodes.size();
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

      _done[node_index] =
          (_done[left_child_index] && _done[right_child_index]) && (left.fill_count == 0 && right.fill_count == 0);
    }
    // Initialize buffer for innner nodes.
    const auto count_inner_nodes = first_leaf_index - 1 - ROOT;
    _fifo_buffer.resize(count_inner_nodes * _buffer_size);
    auto index = size_t{0};
    for (auto node_index = first_leaf_index - 1; node_index > ROOT; --node_index, ++index) {
      _nodes[node_index].buffer = _fifo_buffer.data() + index * _buffer_size;
    }

    _total_output_size =
        std::accumulate(_sorted_buckets.begin(), _sorted_buckets.end(), size_t{0}, [](size_t sum, const auto& bucket) {
          return sum + bucket->size;
        });
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
  size_t _load_and_merge_from_leaves(CircularBuffer& buffer, Bucket& left_bucket, Bucket& right_bucket) {
    auto* a_input = left_bucket.data;
    auto* b_input = right_bucket.data;
    const auto a_length = left_bucket.size;
    const auto b_length = right_bucket.size;

    auto a_start = std::size_t{0};
    auto b_start = std::size_t{0};

    auto merge_data_into_buffer = [&](BufferChunk& chunk, SimdElement* output) {
      const auto old_a_start = a_start;
      const auto old_b_start = b_start;
      const auto old_start_index = chunk.start;

      TwoWayMerge::template merge_multiway_merge_nodes<count_per_vector * 4>(
          reinterpret_cast<T*>(a_input), reinterpret_cast<T*>(b_input), reinterpret_cast<T*>(output), a_start, b_start,
          chunk.start, a_length, b_length, chunk.number_of_slots());

      a_input += a_start - old_a_start;
      b_input += b_start - old_b_start;
      return chunk.start - old_start_index;
    };

    buffer.write_into_buffer(merge_data_into_buffer, _buffer_size);

    if (buffer.fill_count < _buffer_size) {
      auto read_and_write_remaining = [&](SimdElement*& input, std::size_t& input_start, std::size_t input_length) {
        auto write_func = [&](BufferChunk& chunk [[maybe_unused]], SimdElement* output) {
          auto num_written_slots = size_t{0};
          while (input_start < input_length) {
            *output = *input;
            ++input_start;
            ++input;
            ++output;
            ++num_written_slots;
          }
          return num_written_slots;
        };

        buffer.write_into_buffer(write_func, _buffer_size);
      };
      read_and_write_remaining(a_input, a_start, a_length);
      read_and_write_remaining(b_input, b_start, b_length);
    }

    left_bucket.data = a_input;
    left_bucket.size -= a_start;
    right_bucket.data = b_input;
    right_bucket.size -= b_start;

    DebugAssert(buffer.fill_count == _buffer_size || (left_bucket.empty() && right_bucket.empty()),
                "Leaves still have elements, but buffer is not fully filled.");

    return a_start + b_start;
  }

  void _load_data_from_leaves() {
    const auto first_leaf_index = _leaf_count;
    const auto num_nodes = _nodes.size();

    static auto empty_bucket = Bucket{nullptr, 0};

    for (auto leaf_index = first_leaf_index; leaf_index < num_nodes; leaf_index += 2) {
      auto node_index = _parent(leaf_index);
      if (_done[node_index] || _nodes[node_index].fill_count >= _read_threshold) {
        continue;
      }
      const auto left_bucket_index = leaf_index - first_leaf_index;
      const auto right_bucket_index = left_bucket_index + 1;

      auto& lhs_bucket = _done[leaf_index] ? empty_bucket : *_sorted_buckets[left_bucket_index];
      auto& rhs_bucket = _done[leaf_index + 1] ? empty_bucket : *_sorted_buckets[right_bucket_index];

      auto& buffer = _nodes[node_index];

      const auto num_items_read = _load_and_merge_from_leaves(buffer, lhs_bucket, rhs_bucket);

      _done[node_index] = num_items_read == 0 || (lhs_bucket.empty() || rhs_bucket.empty());
      _finished &= _done[node_index];
    }
  }

  void _copy_buffer(CircularBuffer& destination, CircularBuffer& source) {
    while (source.fill_count > 0 && destination.fill_count < _buffer_size) {
      auto read_func = [&](BufferChunk& read_chunk, SimdElement* input) {
        auto number_of_slots = size_t{0};
        auto write_func = [&](BufferChunk& write_chunk, SimdElement* output) {
          number_of_slots = std::min(read_chunk.number_of_slots(), write_chunk.number_of_slots());
          for (auto slot_number = size_t{0}; slot_number < number_of_slots; ++slot_number) {
            *output = *input;
            ++output;
            ++input;
          }
          return number_of_slots;
        };
        destination.write_into_buffer(write_func, _buffer_size);
        return number_of_slots;
      };
      source.read_from_buffer(read_func, _buffer_size);
    }
    DebugAssert(!source.fill_count || destination.fill_count == _buffer_size,
                "Either source has to be empty or desination full.");
  }

  void _write_buffer_to_output(SimdElement*& output, CircularBuffer& source) {
    while (source.fill_count > 0) {
      auto read_func = [&](BufferChunk& read_chunk, SimdElement* input) {
        auto number_of_slots = read_chunk.number_of_slots();
        for (auto slot = size_t{0}; slot < number_of_slots; ++slot) {
          *output = *input;
          ++input;
          ++output;
        }
        return number_of_slots;
      };
      source.read_from_buffer(read_func, _buffer_size);
    }
    DebugAssert(!source.fill_count, "Source has to be empty after writing to output.");
  }

  void _merge_children_into_output(SimdElement*& output, CircularBuffer& left, CircularBuffer& right,
                                   bool left_child_done, bool right_child_done) {
    if (right_child_done && right.fill_count == 0) {
      if (!left_child_done && left.fill_count > 0) {
        _write_buffer_to_output(output, left);
        DebugAssert(!left.fill_count, "Left buffer should be empty.");
        return;
      }
    } else if (left_child_done && left.fill_count == 0) {
      if (!right_child_done && right.fill_count > 0) {
        _write_buffer_to_output(output, right);
        DebugAssert(!right.fill_count, "Right buffer should be empty.");
        return;
      }
    }

    bool both_children_done = right_child_done && left_child_done;

    while (left.fill_count > 0 && right.fill_count > 0) {
      auto left_count = size_t{0};
      auto right_count = size_t{0};

      auto read_left = [&](BufferChunk& left_chunk, SimdElement* left_input) {
        auto read_right = [&](BufferChunk& right_chunk, SimdElement* right_input) {
          const auto old_left_start = left_chunk.start;
          const auto old_right_start = right_chunk.start;

          const auto maximum_needed_slots = left_chunk.number_of_slots() + right_chunk.number_of_slots();
          auto written_slots = size_t{0};

          TwoWayMerge::template merge_multiway_merge_nodes<count_per_vector * 4>(
              reinterpret_cast<T*>(left_input), reinterpret_cast<T*>(right_input), reinterpret_cast<T*>(output),
              left_chunk.start, right_chunk.start, written_slots, left_chunk.end, right_chunk.end,
              maximum_needed_slots);

          output += written_slots;

          left_count += left_chunk.start - old_left_start;
          right_count += right_chunk.start - old_right_start;
          return right_count;
        };
        right.read_from_buffer(read_right, _buffer_size);
        return left_count;
      };
      left.read_from_buffer(read_left, _buffer_size);
    }

    if (!both_children_done) {
      return;
    }

    auto read_and_write_remaining = [&](CircularBuffer& input_buffer) {
      while (input_buffer.fill_count > 0) {
        auto read_func = [&](BufferChunk& read_chunk, SimdElement* input) {
          const auto number_of_slots = read_chunk.number_of_slots();
          for (auto slot = size_t{0}; slot < number_of_slots; ++slot) {
            *output = *input;
            ++input;
            ++output;
          }
          return number_of_slots;
        };
        input_buffer.read_from_buffer(read_func, _buffer_size);
      }
      DebugAssert(!input_buffer.fill_count, "Read-buffer should be empty.");
    };

    if (left.fill_count > 0) {
      read_and_write_remaining(left);
    } else {
      read_and_write_remaining(right);
    }
  }

  void _merge_children_into_node(CircularBuffer& node_buffer, CircularBuffer& left, CircularBuffer& right,
                                 bool left_child_done, bool right_child_done) {
    if (right_child_done && right.fill_count == 0) {
      if (!left_child_done && left.fill_count > 0) {
        _copy_buffer(node_buffer, left);
        return;
      }
    } else if (left_child_done && left.fill_count == 0) {
      if (!right_child_done && right.fill_count > 0) {
        _copy_buffer(node_buffer, right);
        return;
      }
    }

    bool both_children_done = right_child_done && left_child_done;

    while (node_buffer.fill_count < _buffer_size && (left.fill_count > 0 && right.fill_count > 0)) {
      auto left_count = size_t{0};
      auto right_count = size_t{0};

      auto read_left = [&](BufferChunk& left_chunk, SimdElement* left_input) {
        auto read_right = [&](BufferChunk& right_chunk, SimdElement* right_input) {
          auto write_func = [&](BufferChunk& write_chunk, SimdElement* output) {
            const auto old_left_start = left_chunk.start;
            const auto old_right_start = right_chunk.start;

            TwoWayMerge::template merge_multiway_merge_nodes<count_per_vector * 4>(
                reinterpret_cast<T*>(left_input), reinterpret_cast<T*>(right_input), reinterpret_cast<T*>(output),
                left_chunk.start, right_chunk.start, write_chunk.start, left_chunk.end, right_chunk.end,
                write_chunk.number_of_slots());

            left_count += left_chunk.start - old_left_start;
            right_count += right_chunk.start - old_right_start;
            return left_count + right_count;
          };
          node_buffer.write_into_buffer(write_func, _buffer_size);
          return right_count;
        };
        right.read_from_buffer(read_right, _buffer_size);
        return left_count;
      };
      left.read_from_buffer(read_left, _buffer_size);
    }

    if (!both_children_done || node_buffer.fill_count == _buffer_size) {
      return;
    }

    auto read_and_write_remaining = [&](CircularBuffer& input_buffer) {
      while (node_buffer.fill_count < _buffer_size && input_buffer.fill_count > 0) {
        auto read_func = [&](BufferChunk& read_chunk, SimdElement* input) {
          auto read_slots = size_t{0};
          auto write_func = [&](BufferChunk& write_chunk, SimdElement* output) {
            auto old_start_index = write_chunk.start;
            while (read_chunk.start < read_chunk.end && write_chunk.start < write_chunk.end) {
              *output = *input;
              ++input;
              ++output;
              ++read_chunk.start;
              ++write_chunk.start;
            }
            const auto written_slots = write_chunk.start - old_start_index;
            read_slots += written_slots;
            return written_slots;
          };
          node_buffer.write_into_buffer(write_func, _buffer_size);
          return read_slots;
        };
        input_buffer.read_from_buffer(read_func, _buffer_size);
      }
    };

    if (left.fill_count > 0) {
      read_and_write_remaining(left);
    } else {
      read_and_write_remaining(right);
    }
  }

  void _execute_ready_nodes(SimdElement*& output [[maybe_unused]]) {
    const auto first_relevant_inner_node = _parent(_leaf_count - 1);
    for (NodeIndex node_index = first_relevant_inner_node; node_index > ROOT; --node_index) {
      auto left_child_index = _left_child(node_index);
      auto right_child_index = _right_child(node_index);

      auto& buffer = _nodes[node_index];
      auto& left = _nodes[left_child_index];
      auto& right = _nodes[right_child_index];

      if (_done[node_index]) {
        continue;
      }

      const auto children_done = _done[left_child_index] || _done[right_child_index];

      if ((children_done || buffer.fill_count < _read_threshold) && buffer.fill_count < _buffer_size) {
        if (children_done || (right.fill_count >= _read_threshold && left.fill_count >= _read_threshold)) {
          _merge_children_into_node(buffer, left, right, _done[left_child_index], _done[right_child_index]);
        }
        _done[node_index] =
            (_done[left_child_index] && _done[right_child_index]) && (left.fill_count == 0 && right.fill_count == 0);
      }

      _finished &= _done[node_index];
    }
    auto left_index = _left_child(ROOT);
    auto right_index = _right_child(ROOT);
    _merge_children_into_output(output, _nodes[left_index], _nodes[right_index], _done[left_index], _done[right_index]);
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
}  // namespace hyrise
