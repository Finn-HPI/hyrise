#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "hyrise.hpp"
#include "resolve_type.hpp"
#include "scheduler/job_task.hpp"
#include "storage/create_iterable_from_segment.hpp"
#include "storage/dictionary_segment.hpp"
#include "storage/segment_iterate.hpp"
#include "storage/vector_compression/resolve_compressed_vector_type.hpp"
#include "types.hpp"

namespace hyrise {

template <typename T>
struct MaterializedValue {
  MaterializedValue() = default;

  MaterializedValue(RowID row, T init_value) : row_id{row}, value{init_value} {}

  MaterializedValue(ChunkID chunk_id, ChunkOffset chunk_offset, T init_value)
      : row_id{chunk_id, chunk_offset}, value{init_value} {}

  RowID row_id;
  T value;
};

template <typename T>
using MaterializedSegment = std::vector<MaterializedValue<T>>;

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
  std::tuple<MaterializedSegmentList<T>, RowIDPosList, T, T> materialize(const std::shared_ptr<const Table>& input,
                                                                         const ColumnID column_id) {
    const auto chunk_count = input->chunk_count();

    auto output = MaterializedSegmentList<T>(chunk_count);
    auto min_max_values = std::vector<std::pair<T, T>>(chunk_count);

    auto null_rows_per_chunk = std::vector<RowIDPosList>(chunk_count);

    auto jobs = std::vector<std::shared_ptr<AbstractTask>>{};
    for (auto chunk_id = ChunkID{0}; chunk_id < chunk_count; ++chunk_id) {
      const auto& chunk = input->get_chunk(chunk_id);
      Assert(chunk, "Physically deleted chunk should not reach this point, see get_chunk / #1686.");
      const auto chunk_size = chunk->size();

      auto materialize_job = [&, chunk_id] {
        const auto& segment = input->get_chunk(chunk_id)->get_segment(column_id);
        std::tie(output[chunk_id], min_max_values[chunk_id]) =
            std::move(_materialize_segment(segment, chunk_id, null_rows_per_chunk[chunk_id]));
      };

      if (chunk_size > _job_spawn_threshold) {
        jobs.push_back(std::make_shared<JobTask>(materialize_job));
      } else {
        materialize_job();
      }
    }

    Hyrise::get().scheduler()->schedule_and_wait_for_tasks(jobs);

    auto min = std::numeric_limits<T>::max();
    auto max = std::numeric_limits<T>::lowest();

    if constexpr (std::is_same_v<T, int64_t>) {
      for (auto [segment_min, segment_max] : min_max_values) {
        if (segment_max > max) {
          max = segment_max;
        }
        if (segment_min < min) {
          min = segment_min;
        }
      }
    } else {
      max = std::numeric_limits<T>::max();
      min = std::numeric_limits<T>::lowest();
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

    return {std::move(output), std::move(null_rows), min, max};
  }

 private:
  std::pair<MaterializedSegment<T>, std::pair<T, T>> _materialize_segment(
      const std::shared_ptr<AbstractSegment>& segment, const ChunkID chunk_id, RowIDPosList& null_rows_output) {
    auto output = MaterializedSegment<T>{};
    output.reserve(segment->size());

    auto min = std::numeric_limits<T>::max();
    auto max = std::numeric_limits<T>::lowest();

    segment_iterate<T>(*segment, [&](const auto& position) {
      if (position.is_null()) {
        if (_materialize_null) {
          null_rows_output.emplace_back(chunk_id, position.chunk_offset());
        }
      } else {
        auto& value = position.value();
        output.emplace_back(chunk_id, position.chunk_offset(), value);
        if constexpr (std::is_same_v<T, int64_t>) {
          if (value < min) {
            min = value;
          }
          if (value > max) {
            max = value;
          }
        }
      }
    });

    return {output, {min, max}};
  }

  bool _materialize_null;
  std::size_t _job_spawn_threshold;
};

}  // namespace hyrise
