#pragma once

#include <algorithm>
#include <memory>
#include <span>
#include <vector>

#include "hyrise.hpp"
#include "operators/join_simd_sort_merge/two_way_merge.hpp"
#include "scheduler/abstract_task.hpp"
#include "scheduler/job_task.hpp"

namespace hyrise::merge_path {
template <size_t count_per_vector, typename T>
class MergePath {
  using IntersectionPoint = std::pair<int64_t, int64_t>;
  using TwoWayMerge = simd_sort::TwoWayMerge<count_per_vector, T>;

 public:
  MergePath(std::span<T> input_a, std::span<T> input_b, const size_t partition_count)
      : _input_a(std::move(input_a)),
        _input_b(std::move(input_b)),
        _partition_count(partition_count),
        _diagonal_intersection_points(partition_count + 1) {}

  void merge(std::span<T> output) {
    const auto count_a = _input_a.size();
    const auto count_b = _input_b.size();

    if (_partition_count <= 1) {
      _merge_section_aligned(_input_a, _input_b, output);
      return;
    }

    const auto combined_size = count_a + count_b;
    _partition_count = std::min(_partition_count, combined_size);

    auto intersection_points = std::vector<std::pair<int64_t, int64_t>>(_partition_count + 1);

    intersection_points[0] = {-1, -1};

    auto tasks = std::vector<std::shared_ptr<AbstractTask>>{};
    tasks.reserve(_partition_count);

    for (auto partition_index = size_t{0}; partition_index < _partition_count; ++partition_index) {
      tasks.push_back(std::make_shared<JobTask>([&, partition_index]() {
        intersection_points[partition_index + 1] =
            std::move(_find_intersection_point(partition_index, _input_a, _input_b));
      }));
    }

    Hyrise::get().scheduler()->schedule_and_wait_for_tasks(tasks);

    tasks.clear();

    for (auto partition_index = size_t{0}; partition_index < _partition_count; ++partition_index) {
      tasks.push_back(std::make_shared<JobTask>([&, partition_index]() {
        auto [a_end_prev, b_end_prev] = intersection_points[partition_index];
        auto [a_end, b_end] = intersection_points[partition_index + 1];

        auto a_start = a_end_prev + 1;
        auto b_start = b_end_prev + 1;

        auto length_a_section = a_start <= a_end ? a_end + 1 - a_start : 0;
        auto length_b_section = b_start <= b_end ? b_end + 1 - b_start : 0;

        const auto output_offset =
            static_cast<size_t>(static_cast<double>(partition_index) * static_cast<double>(combined_size) /
                                static_cast<double>(_partition_count));

        auto a_section = std::span<T>(_input_a.begin() + a_start, length_a_section);
        auto b_section = std::span<T>(_input_b.begin() + b_start, length_b_section);

        auto out_section = std::span(output.begin() + output_offset, length_a_section + length_b_section);

        _merge_section_unaligned(std::move(a_section), std::move(b_section), std::move(out_section));
      }));
    }

    Hyrise::get().scheduler()->schedule_and_wait_for_tasks(tasks);
  }

 private:
  std::span<T> _input_a;

  std::span<T> _input_b;
  size_t _partition_count;

  std::vector<IntersectionPoint> _diagonal_intersection_points;

  void _merge_section_aligned(std::span<T> input_a, std::span<T> input_b, std::span<T> output) {
    TwoWayMerge::template merge_variable_length<count_per_vector * 4>(input_a.data(), input_b.data(), output.data(),
                                                                      input_a.size(), input_b.size());
  }

  void _merge_section_unaligned(std::span<T> input_a, std::span<T> input_b, std::span<T> output) {
    TwoWayMerge::template merge_variable_length_unaligned<count_per_vector * 4>(
        input_a.data(), input_b.data(), output.data(), input_a.size(), input_b.size());
  }

  IntersectionPoint _find_intersection_point(size_t thread_index, std::span<T> input_a, std::span<T> input_b) {
    const auto count_a = static_cast<int64_t>(input_a.size());
    const auto count_b = static_cast<int64_t>(input_b.size());

    if (thread_index + 1 == _partition_count) {
      return {count_a - 1, count_b - 1};
    }

    const auto diagonal =
        static_cast<int64_t>((static_cast<double>(thread_index + 1) * static_cast<double>(count_a + count_b) /
                              static_cast<double>(_partition_count)) -
                             1);

    int64_t a_top = diagonal > count_a ? count_a : diagonal;
    int64_t b_top = diagonal > count_a ? diagonal - count_a : int64_t{0};
    int64_t a_bottom = b_top;

    auto a_index = int64_t{0};
    auto b_index = int64_t{0};

    while (true) {
      const auto offset = std::abs(a_top - a_bottom) / 2;
      a_index = a_top - offset;
      b_index = b_top + offset;
      if (a_index >= 0 && b_index <= count_b &&
          (a_index == count_a || b_index == 0 || input_a[a_index] > input_b[b_index - 1])) {
        if (b_index == count_b || a_index == 0 || input_a[a_index - 1] <= input_b[b_index]) {
          if (a_index < count_a && (b_index == count_b || input_a[a_index] <= input_b[b_index])) {
            return {a_index, b_index - 1};
          }
          return {a_index - 1, b_index};
        }
        a_top = a_index - 1;
        b_top = b_index + 1;
      } else {
        a_bottom = a_index + 1;
      }
    }
    return {};
  }
};
}  // namespace hyrise::merge_path
