#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/sort/sort.hpp>

#include "hyrise.hpp"
#include "scheduler/job_task.hpp"
#include "storage/segment_iterate.hpp"
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

// Materializes a column.
template <typename T>
class SMJColumnMaterializer {
 public:
  explicit SMJColumnMaterializer(std::size_t job_spawn_threshold = 500) : _job_spawn_threshold(job_spawn_threshold) {}

  // For sufficiently large chunks (number of rows > JOB_SPAWN_THRESHOLD), the materialization is parallelized. Returns
  // the materialized segments and the min and max value of the column.
  std::tuple<MaterializedSegmentList<T>, T, T, RowIDPosList> materialize(const std::shared_ptr<const Table>& input,
                                                                         const ColumnID column_id,
                                                                         const bool materialize_null) {
    const auto chunk_count = input->chunk_count();

    auto output = MaterializedSegmentList<T>(chunk_count);
    auto min_max_per_chunk = std::vector<std::pair<T, T>>(chunk_count);

    auto null_rows_per_chunk = std::vector<RowIDPosList>(chunk_count);

    auto jobs = std::vector<std::shared_ptr<AbstractTask>>{};
    for (auto chunk_id = ChunkID{0}; chunk_id < chunk_count; ++chunk_id) {
      const auto& chunk = input->get_chunk(chunk_id);
      Assert(chunk, "Physically deleted chunk should not reach this point, see get_chunk / #1686.");
      const auto chunk_size = chunk->size();

      auto materialize_job = [&, chunk_id] {
        const auto& segment = input->get_chunk(chunk_id)->get_segment(column_id);
        auto [materialized_segment, min_value, max_value] =
            _materialize_segment(segment, chunk_id, null_rows_per_chunk[chunk_id], materialize_null);
        output[chunk_id] = std::move(materialized_segment);
        min_max_per_chunk[chunk_id] = {min_value, max_value};
      };

      if (chunk_size > _job_spawn_threshold) {
        jobs.push_back(std::make_shared<JobTask>(materialize_job));
      } else {
        materialize_job();
      }
    }

    auto min_value = std::numeric_limits<T>::max();
    auto max_value = std::numeric_limits<T>::lowest();

    for (auto chunk_id = ChunkID{0}; chunk_id < chunk_count; ++chunk_id) {
      auto& [chunk_min, chunk_max] = min_max_per_chunk[chunk_id];
      min_value = min_value <= chunk_min ? min_value : chunk_min;
      max_value = max_value >= chunk_max ? max_value : chunk_max;
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

    return {std::move(output), min_value, max_value, std::move(null_rows)};
  }

 private:
  std::tuple<MaterializedSegment<T>, T, T> _materialize_segment(const std::shared_ptr<AbstractSegment>& segment,
                                                                const ChunkID chunk_id, RowIDPosList& null_rows_output,
                                                                const bool materialize_null) {
    auto output = MaterializedSegment<T>{};
    output.reserve(segment->size());

    auto min_value = std::numeric_limits<T>::max();
    auto max_value = std::numeric_limits<T>::min();

    segment_iterate<T>(*segment, [&](const auto& position) {
      if (position.is_null()) {
        if (materialize_null) {
          null_rows_output.emplace_back(chunk_id, position.chunk_offset());
        }
        return;
      }
      const auto value = position.value();
      min_value = min_value <= value ? min_value : value;
      max_value = max_value >= value ? max_value : value;
      output.emplace_back(chunk_id, position.chunk_offset(), value);
    });

    return {output, min_value, max_value};
  }

  std::size_t _job_spawn_threshold;
};

}  // namespace hyrise
