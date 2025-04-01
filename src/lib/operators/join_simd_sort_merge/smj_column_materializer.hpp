#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "hyrise.hpp"
#include "operators/join_hash/join_hash_steps.hpp"
#include "operators/join_simd_sort_merge/util.hpp"
#include "resolve_type.hpp"
#include "scheduler/job_task.hpp"
#include "storage/create_iterable_from_segment.hpp"
#include "storage/dictionary_segment.hpp"
#include "storage/segment_iterate.hpp"
#include "storage/vector_compression/resolve_compressed_vector_type.hpp"
#include "types.hpp"

namespace hyrise {

template <typename T>
// NOLINTNEXTLINE
struct MaterializedValue {
  union {
    RowID row_id;
    SimdElement element{};
  };

  T value;
};

template <typename T>
using MaterializedSegment =
    std::conditional_t<std::is_trivially_destructible_v<T>, uninitialized_vector<MaterializedValue<T>>,
                       std::vector<MaterializedValue<T>>>;

template <typename T>
using MaterializedSegmentList = std::vector<MaterializedSegment<T>>;

// Materializes a column and sorts it if requested.
template <typename T>
class SMJColumnMaterializer {
 public:
  explicit SMJColumnMaterializer(bool materialize_null, std::size_t job_spawn_threshold = 500)
      : _materialize_null{materialize_null}, _job_spawn_threshold(job_spawn_threshold) {}

  // For sufficiently large chunks (number of rows > JOB_SPAWN_THRESHOLD), the materialization is parallelized. Returns
  // the materialized segments and a list of null row ids if _materialize_null is true.
  template <bool use_bloom_filter>
  std::tuple<MaterializedSegmentList<T>, RowIDPosList, ChunkID, ChunkOffset, T, T> materialize(
      const std::shared_ptr<const Table>& input, const ColumnID column_id,
      [[maybe_unused]] uint64_t& total_filter_count, BloomFilter& output_bloom_filter,
      const BloomFilter& input_bloom_filter = ALL_TRUE_BLOOM_FILTER) {
    const auto chunk_count = input->chunk_count();
    const std::hash<T> hash_function;

    if constexpr (use_bloom_filter) {
      output_bloom_filter.resize(BLOOM_FILTER_SIZE);
    }
    auto output_bloom_filter_mutex = std::mutex{};

    auto output = MaterializedSegmentList<T>(chunk_count);

    auto null_rows_per_chunk = std::vector<RowIDPosList>(chunk_count);

    auto max_chunk_size = ChunkOffset{0};

    auto jobs = std::vector<std::shared_ptr<AbstractTask>>{};
    for (auto chunk_id = ChunkID{0}; chunk_id < chunk_count; ++chunk_id) {
      const auto& chunk = input->get_chunk(chunk_id);
      if (!chunk) {
        continue;
      }

      Assert(chunk, "Physically deleted chunk should not reach this point, see get_chunk / #1686.");
      const auto chunk_size = chunk->size();
      max_chunk_size = std::max(chunk_size, max_chunk_size);

      auto materialize_job = [&, chunk, chunk_id] {
        auto local_output_bloom_filter = BloomFilter{};
        std::reference_wrapper<BloomFilter> used_output_bloom_filter = output_bloom_filter;

        // Skip chunks that were physically deleted.
        if (!chunk) {
          return;
        }

        if constexpr (use_bloom_filter) {
          if (Hyrise::get().is_multi_threaded()) {
            local_output_bloom_filter = BloomFilter(BLOOM_FILTER_SIZE, false);
            used_output_bloom_filter = local_output_bloom_filter;
          }

          const auto& segment = input->get_chunk(chunk_id)->get_segment(column_id);
          auto filter_count = uint64_t{0};
          if (_materialize_null) {
            output[chunk_id] = std::move(_materialize_segment_with_bloom_filter<true>(
                segment, chunk_id, null_rows_per_chunk[chunk_id], used_output_bloom_filter, input_bloom_filter,
                hash_function, chunk->size(), filter_count));
          } else {
            output[chunk_id] = std::move(_materialize_segment_with_bloom_filter<false>(
                segment, chunk_id, null_rows_per_chunk[chunk_id], used_output_bloom_filter, input_bloom_filter,
                hash_function, chunk->size(), filter_count));
          }

          if (Hyrise::get().is_multi_threaded()) {
            const auto lock = std::lock_guard<std::mutex>{output_bloom_filter_mutex};
            output_bloom_filter |= local_output_bloom_filter;
            // total_filter_count += filter_count;
          }
          // else {
          //   total_filter_count += filter_count;
          // }

        } else {
          const auto& segment = input->get_chunk(chunk_id)->get_segment(column_id);
          if (_materialize_null) {
            output[chunk_id] = std::move(_materialize_segment<true>(segment, chunk_id, null_rows_per_chunk[chunk_id]));

          } else {
            output[chunk_id] = std::move(_materialize_segment<false>(segment, chunk_id, null_rows_per_chunk[chunk_id]));
          }
        }
      };

      if (chunk_size > _job_spawn_threshold) {
        jobs.push_back(std::make_shared<JobTask>(materialize_job));
      } else {
        materialize_job();
      }
    }

    Hyrise::get().scheduler()->schedule_and_wait_for_tasks(jobs);

    auto null_row_count = size_t{0};
    for (const auto& null_rows : null_rows_per_chunk) {
      null_row_count += null_rows.size();
    }
    auto null_rows = RowIDPosList{};
    null_rows.reserve(null_row_count);

    for (auto chunk_id = ChunkID{0}; chunk_id < chunk_count; ++chunk_id) {
      const auto& chunk_null_rows = null_rows_per_chunk[chunk_id];
      null_rows.insert(null_rows.end(), chunk_null_rows.begin(), chunk_null_rows.end());
    }

    return {std::move(output), std::move(null_rows), chunk_count, max_chunk_size, T{}, T{}};
  }

 private:
  template <bool keep_null>
  MaterializedSegment<T> _materialize_segment_with_bloom_filter(
      const std::shared_ptr<AbstractSegment>& segment, const ChunkID chunk_id,
      [[maybe_unused]] RowIDPosList& null_rows_output, std::reference_wrapper<BloomFilter>& used_output_bloom_filter,
      const BloomFilter& input_bloom_filter, const std::hash<T>& hash_function, ChunkOffset num_rows,
      [[maybe_unused]] uint64_t& filter_count) {
    auto elements = MaterializedSegment<T>{};
    elements.resize(num_rows);
    auto elements_iter = elements.begin();

    segment_iterate<T>(*segment, [&](const auto& position) {
      if (position.is_null()) {
        if constexpr (keep_null) {
          null_rows_output.emplace_back(chunk_id, position.chunk_offset());
        }
      } else {
        auto& value = position.value();
        const Hash hashed_value = hash_function(static_cast<T>(value));

        if (input_bloom_filter[hashed_value & BLOOM_FILTER_MASK]) {
          used_output_bloom_filter.get()[hashed_value & BLOOM_FILTER_MASK] = true;
          *elements_iter = MaterializedValue<T>{{RowID(chunk_id, position.chunk_offset())}, value};
          ++elements_iter;
        }
        // else {
        //   ++filter_count;
        // }
      }
    });

    elements.resize(std::distance(elements.begin(), elements_iter));
    return elements;
  }

  template <bool keep_null>
  MaterializedSegment<T> _materialize_segment(const std::shared_ptr<AbstractSegment>& segment, const ChunkID chunk_id,
                                              RowIDPosList& null_rows_output) {
    const auto num_rows = segment->size();

    auto output = MaterializedSegment<T>{};
    output.resize(num_rows);

    auto elements_iter = output.begin();

    segment_iterate<T>(*segment, [&](const auto& position) {
      if (position.is_null()) {
        if constexpr (keep_null) {
          null_rows_output.emplace_back(chunk_id, position.chunk_offset());
        }
      } else {
        *elements_iter = MaterializedValue<T>{{RowID(chunk_id, position.chunk_offset())}, position.value()};
        ++elements_iter;
      }
    });

    output.resize(std::distance(output.begin(), elements_iter));
    return output;
  }

  bool _materialize_null;
  std::size_t _job_spawn_threshold;
};

template <>
class SMJColumnMaterializer<int64_t> {
 public:
  explicit SMJColumnMaterializer<int64_t>(bool materialize_null, std::size_t job_spawn_threshold = 500)
      : _materialize_null{materialize_null}, _job_spawn_threshold(job_spawn_threshold) {}

  // For sufficiently large chunks (number of rows > JOB_SPAWN_THRESHOLD), the materialization is parallelized. Returns
  // the materialized segments and a list of null row ids if _materialize_null is true.
  template <bool use_bloom_filter>
  std::tuple<MaterializedSegmentList<int64_t>, RowIDPosList, ChunkID, ChunkOffset, int64_t, int64_t> materialize(
      const std::shared_ptr<const Table>& input, const ColumnID column_id,
      [[maybe_unused]] uint64_t& total_filter_count, BloomFilter& output_bloom_filter,
      const BloomFilter& input_bloom_filter = ALL_TRUE_BLOOM_FILTER) {
    const auto chunk_count = input->chunk_count();

    auto output = MaterializedSegmentList<int64_t>(chunk_count);
    auto min_max_values = std::vector<std::pair<int64_t, int64_t>>(chunk_count);

    auto null_rows_per_chunk = std::vector<RowIDPosList>(chunk_count);

    auto max_chunk_size = ChunkOffset{0};

    auto jobs = std::vector<std::shared_ptr<AbstractTask>>{};
    for (auto chunk_id = ChunkID{0}; chunk_id < chunk_count; ++chunk_id) {
      const auto& chunk = input->get_chunk(chunk_id);
      Assert(chunk, "Physically deleted chunk should not reach this point, see get_chunk / #1686.");
      const auto chunk_size = chunk->size();
      max_chunk_size = std::max(chunk_size, max_chunk_size);

      auto materialize_job = [&, chunk_id] {
        const auto& segment = input->get_chunk(chunk_id)->get_segment(column_id);
        std::tie(output[chunk_id], min_max_values[chunk_id]) = _materialize_segment(
            segment, chunk_id, null_rows_per_chunk[chunk_id], output_bloom_filter, input_bloom_filter);
      };

      if (chunk_size > _job_spawn_threshold) {
        jobs.push_back(std::make_shared<JobTask>(materialize_job));
      } else {
        materialize_job();
      }
    }

    Hyrise::get().scheduler()->schedule_and_wait_for_tasks(jobs);

    auto min = std::numeric_limits<int64_t>::max();
    auto max = std::numeric_limits<int64_t>::lowest();

    for (auto [segment_min, segment_max] : min_max_values) {
      min = segment_min < min ? segment_min : min;
      max = segment_max > max ? segment_max : max;
    }

    auto null_row_count = size_t{0};
    for (const auto& null_rows : null_rows_per_chunk) {
      null_row_count += null_rows.size();
    }
    auto null_rows = RowIDPosList{};
    null_rows.reserve(null_row_count);

    for (auto chunk_id = ChunkID{0}; chunk_id < chunk_count; ++chunk_id) {
      const auto& chunk_null_rows = null_rows_per_chunk[chunk_id];
      null_rows.insert(null_rows.end(), chunk_null_rows.begin(), chunk_null_rows.end());
    }

    return {std::move(output), std::move(null_rows), chunk_count, max_chunk_size, min, max};
  }

 private:
  std::pair<MaterializedSegment<int64_t>, std::pair<int64_t, int64_t>> _materialize_segment(
      const std::shared_ptr<AbstractSegment>& segment, const ChunkID chunk_id, RowIDPosList& null_rows_output,
      [[maybe_unused]] std::reference_wrapper<BloomFilter> used_output_bloom_filter,
      [[maybe_unused]] const BloomFilter& input_bloom_filter) {
    const auto num_rows = segment->size();

    auto output = MaterializedSegment<int64_t>{};
    output.resize(num_rows);

    auto elements_iter = output.begin();

    auto min = std::numeric_limits<int64_t>::max();
    auto max = std::numeric_limits<int64_t>::lowest();

    segment_iterate<int64_t>(*segment, [&](const auto& position) {
      if (position.is_null()) {
        if (_materialize_null) {
          null_rows_output.emplace_back(chunk_id, position.chunk_offset());
        }
      } else {
        auto& value = position.value();

        *elements_iter = MaterializedValue<int64_t>{{RowID(chunk_id, position.chunk_offset())}, value};
        ++elements_iter;
        min = value < min ? value : min;
        max = value > max ? value : max;
      }
    });

    output.resize(std::distance(output.begin(), elements_iter));
    return {output, {min, max}};
  }

  bool _materialize_null;
  std::size_t _job_spawn_threshold;
};

}  // namespace hyrise
