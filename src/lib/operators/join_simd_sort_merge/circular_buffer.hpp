#pragma once

#include <operators/join_simd_sort_merge/util.hpp>
#include <span>

namespace hyrise::circular_buffer {

class CircularBuffer {
 public:
  inline void write(const size_t buffer_size, auto&& write_func);

  inline void read(const size_t buffer_size, auto&& read_func);

  inline void read_and_write_to(CircularBuffer& write_buffer, const size_t buffer_size, auto&& read_write_func);

  inline void read_and_write_to(SimdElement*& output, const size_t buffer_size, auto&& read_write_func);

  inline void merge_and_write(CircularBuffer& inner_read_buffer, CircularBuffer& write_buffer, const size_t buffer_size,
                              auto&& merge_func);

  inline void merge_and_write(CircularBuffer& inner_read_buffer, SimdElement*& output, const size_t buffer_size,
                              auto&& merge_func);

  // Helper function to determine if buffer content is sorted by SimdElement.key.
  template <typename T>
  inline bool debug_is_sorted(const size_t buffer_size);

  inline size_t fill_count() const {
    return _fill_count;
  }

  inline bool empty() const {
    return _fill_count == 0;
  }

  void set_buffer(SimdElement* init_buffer) {
    _buffer = init_buffer;
  }

 private:
  struct BufferChunk {
    size_t start;
    size_t end;

    size_t size() const {
      return end - start;
    }
  };

  void _update_head(size_t number_of_written_slots, const size_t buffer_size);
  void _update_tail(size_t number_of_read_slots, const size_t buffer_size);

  SimdElement* _buffer{};
  size_t _head = 0;
  size_t _tail = 0;
  size_t _fill_count = 0;
};

inline void CircularBuffer::write(const size_t buffer_size, auto&& write_func) {
  if (_tail == _head && _fill_count == buffer_size) {
    return;
  }
  DebugAssert(_fill_count < buffer_size, "Wrote to full buffer");
  auto first_chunk = BufferChunk{_head, (_tail > _head) ? _tail : buffer_size};

  if (!first_chunk.size()) {
    auto written_slots = write_func(std::span(_buffer + first_chunk.start, first_chunk.size()));
    DebugAssert(written_slots <= first_chunk.size(), "Wrote more slots than available.");
    _update_head(written_slots, buffer_size);
  }

  if (_fill_count < buffer_size && _head == 0 && _head < _tail) {
    auto second_chunk = BufferChunk{0, _tail};
    auto written_slots = write_func(std::span(_buffer, second_chunk.size()));
    DebugAssert(written_slots <= second_chunk.size(), "Wrote more slots than available.");
    _update_head(written_slots, buffer_size);
  }
}

inline void CircularBuffer::read(const size_t buffer_size, auto&& read_func) {
  if (_tail == _head && empty()) {
    return;
  }
  auto read_chunk = BufferChunk{_tail, (_tail < _head) ? _head : buffer_size};
  if (!read_chunk.size()) {
    return;
  }
  auto number_of_read_slots = read_func(std::span(_buffer + read_chunk.start, read_chunk.size()));
  DebugAssert(number_of_read_slots <= read_chunk.size(), "Read more than available.");
  _update_tail(number_of_read_slots, buffer_size);
}

inline void CircularBuffer::read_and_write_to(CircularBuffer& write_buffer, const size_t buffer_size,
                                              auto&& read_write_func) {
  while (!empty() && write_buffer.fill_count() < buffer_size) {
    auto read_func = [&](std::span<SimdElement> input) {
      auto slots_read = size_t{0};
      auto write_func = [&](std::span<SimdElement> output) {
        const auto slots_written = std::min(input.size(), output.size());
        read_write_func(input.subspan(0, slots_written), output.subspan(0, slots_written));

        slots_read += slots_written;
        input = input.subspan(slots_written);

        return slots_written;
      };
      write_buffer.write(buffer_size, write_func);
      return slots_read;
    };
    read(buffer_size, read_func);
  }

  DebugAssert(empty() || write_buffer.fill_count() == buffer_size, "Either source has to be empty or desination full.");
}

inline void CircularBuffer::read_and_write_to(SimdElement*& output, const size_t buffer_size, auto&& read_write_func) {
  while (!empty()) {
    auto read_func = [&](std::span<SimdElement> input) {
      const auto number_of_slots = input.size();
      read_write_func(input, std::span(output, number_of_slots));

      output += number_of_slots;

      return number_of_slots;
    };
    read(buffer_size, read_func);
  }

  DebugAssert(empty(), "Source has to be empty.");
}

inline void CircularBuffer::merge_and_write(CircularBuffer& inner_read_buffer, CircularBuffer& write_buffer,
                                            const size_t buffer_size, auto&& merge_func) {
  auto read_left = [&](std::span<SimdElement> left_input) {
    auto slots_read_left = size_t{0};
    auto read_right = [&](std::span<SimdElement> right_input) {
      auto slots_read_right = size_t{0};
      auto write_func = [&](std::span<SimdElement> output) {
        auto [count_reads_left, count_reads_right] = merge_func(left_input, right_input, output);

        slots_read_left += count_reads_left;
        slots_read_right += count_reads_right;

        left_input = left_input.subspan(count_reads_left);
        right_input = right_input.subspan(count_reads_right);

        return count_reads_left + count_reads_right;
      };
      write_buffer.write(buffer_size, write_func);
      return slots_read_right;
    };
    inner_read_buffer.read(buffer_size, read_right);
    return slots_read_left;
  };
  read(buffer_size, read_left);

  DebugAssert(write_buffer.fill_count() <= buffer_size, "Buffer fill_count exceeds maximum size.");
}

inline void CircularBuffer::merge_and_write(CircularBuffer& inner_read_buffer, SimdElement*& output,
                                            const size_t buffer_size, auto&& merge_func) {
  auto read_left = [&](std::span<SimdElement> left_input) {
    auto slots_read_left = size_t{0};
    auto read_right = [&](std::span<SimdElement> right_input) {
      auto slots_read_right = size_t{0};
      const auto max_output_size = left_input.size() + right_input.size();

      auto [count_reads_left, count_reads_right] =
          merge_func(left_input, right_input, std::span(output, max_output_size));

      left_input = left_input.subspan(count_reads_left);
      right_input = right_input.subspan(count_reads_right);

      output += count_reads_left + count_reads_right;
      slots_read_left += count_reads_left;
      slots_read_right += count_reads_right;

      return slots_read_right;
    };
    inner_read_buffer.read(buffer_size, read_right);
    return slots_read_left;
  };
  read(buffer_size, read_left);
}

inline void CircularBuffer::_update_head(size_t number_of_written_slots, const size_t buffer_size) {
  DebugAssert(_fill_count + number_of_written_slots <= buffer_size, "Number of written slots exceeds fill count");
  _fill_count += number_of_written_slots;
  _head += number_of_written_slots;

  DebugAssert(_head <= buffer_size, "Head is out of buffer_size.");

  if (_head == buffer_size) {
    _head = 0;
  }
}

inline void CircularBuffer::_update_tail(size_t number_of_read_slots, const size_t buffer_size) {
  DebugAssert(number_of_read_slots <= _fill_count, "Number of read slots exceeds fill count.");

  _fill_count -= number_of_read_slots;
  _tail += number_of_read_slots;

  DebugAssert(_tail <= buffer_size, "Tail is out of buffer_size.");

  if (_tail == buffer_size) {
    _tail = 0;
  }
}

template <typename T>
inline bool CircularBuffer::debug_is_sorted(const size_t buffer_size) {
  if (empty()) {
    return true;
  }
  auto first_chunk = BufferChunk{_tail, (_tail < _head) ? _head : buffer_size};
  auto second_chunk = BufferChunk{0, (_tail < _head) ? 0 : _head};

  auto last_element = std::numeric_limits<T>::min();

  auto is_sorted = [&](BufferChunk& chunk) {
    if (chunk.size() == 0) {
      return true;
    }
    auto elements = std::span(_buffer + chunk.start, chunk.size());
    if (*reinterpret_cast<T*>(&elements.front()) < last_element) {
      return false;
    }
    last_element = *reinterpret_cast<T*>(&elements.back());
    return std::is_sorted(elements.begin(), elements.end(), [](auto& left, auto& right) {
      return *reinterpret_cast<T*>(&left) < *reinterpret_cast<T*>(&right);
    });
  };

  auto sorted = true;
  sorted &= is_sorted(first_chunk);
  sorted &= is_sorted(second_chunk);
  return sorted;
}

}  // namespace hyrise::circular_buffer
