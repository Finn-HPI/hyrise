#include "generate_pruning_statistics.hpp"

#include <atomic>
#include <iostream>
#include <thread>
#include <unordered_set>

#include "operators/table_wrapper.hpp"
#include "resolve_type.hpp"
#include "statistics/statistics_objects/distinct_value_count.hpp"
#include "statistics/statistics_objects/equal_distinct_count_histogram.hpp"
#include "statistics/statistics_objects/generic_histogram_builder.hpp"
#include "statistics/statistics_objects/null_value_ratio_statistics.hpp"
#include "statistics/table_statistics.hpp"
#include "storage/create_iterable_from_segment.hpp"
#include "storage/table.hpp"

namespace hyrise {

void generate_chunk_pruning_statistics(const std::shared_ptr<Chunk>& chunk) {
  if (chunk->pruning_statistics()) {
    // Pruning statistics should be stable no matter what encoding or sort order is used. Hence, when they are present
    // they are up to date and we can skip the recreation.
    return;
  }

  auto chunk_statistics = ChunkPruningStatistics{chunk->column_count()};

  for (auto column_id = ColumnID{0}; column_id < chunk->column_count(); ++column_id) {
    const auto segment = chunk->get_segment(column_id);

    resolve_data_and_segment_type(*segment, [&](auto type, auto& typed_segment) {
      using SegmentType = std::decay_t<decltype(typed_segment)>;
      using ColumnDataType = typename decltype(type)::type;

      const auto segment_statistics = std::make_shared<AttributeStatistics<ColumnDataType>>();

      if constexpr (std::is_same_v<SegmentType, DictionarySegment<ColumnDataType>>) {
        // we can use the fact that dictionary segments have an accessor for the dictionary
        const auto& dictionary = *typed_segment.dictionary();
        create_pruning_statistics_for_segment(*segment_statistics, dictionary);
      } else {
        // if we have a generic segment we create the dictionary ourselves
        auto iterable = create_iterable_from_segment<ColumnDataType>(typed_segment);
        std::unordered_set<ColumnDataType> values;
        iterable.for_each([&](const auto& value) {
          // we are only interested in non-null values
          if (!value.is_null()) {
            values.insert(value.value());
          }
        });
        pmr_vector<ColumnDataType> dictionary{values.cbegin(), values.cend()};
        std::sort(dictionary.begin(), dictionary.end());
        create_pruning_statistics_for_segment(*segment_statistics, dictionary);
      }

      chunk_statistics[column_id] = segment_statistics;
    });
  }

  chunk->set_pruning_statistics(chunk_statistics);
}

void generate_chunk_pruning_statistics(const std::shared_ptr<Table>& table) {
  const auto chunk_count = table->chunk_count();
  for (auto chunk_id = ChunkID{0}; chunk_id < chunk_count; ++chunk_id) {
    const auto chunk = table->get_chunk(chunk_id);

    if (!chunk || chunk->is_mutable()) {
      continue;
    }

    generate_chunk_pruning_statistics(chunk);
  }
}

}  // namespace hyrise
