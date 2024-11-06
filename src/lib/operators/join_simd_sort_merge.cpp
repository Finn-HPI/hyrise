#include "join_simd_sort_merge.hpp"

#include <boost/container_hash/hash.hpp>
#include <boost/functional/hash.hpp>
#include <boost/unordered/unordered_flat_set.hpp>

#include "operators/join_helper/join_output_writing.hpp"
#include "operators/join_simd_sort_merge/multiway_merging.hpp"
#include "operators/join_simd_sort_merge/radix_partitioning.hpp"
#include "operators/join_simd_sort_merge/simd_sort.hpp"
#include "operators/multi_predicate_join/multi_predicate_join_evaluator.hpp"
#include "utils/timer.hpp"

#if defined(__x86_64__)
#include "immintrin.h"
#endif

#include <algorithm>
#include <iterator>
#include <limits>
#include <span>
#include <utility>

#include "operators/join_simd_sort_merge/column_materializer.hpp"
#include "types.hpp"
#include "utils/assert.hpp"

namespace {
template <typename T>
concept FourByteType = (sizeof(T) == 4);

template <typename T>
struct Data32BitCompression {
  static inline uint32_t compress(T& value [[maybe_unused]], const T& min_value [[maybe_unused]],
                                  const T& max_value [[maybe_unused]]) {
    static_assert(false, "Not implemented");
  }
};

template <FourByteType T>
struct Data32BitCompression<T> {
  static inline uint32_t compress(T value, const T& min_value [[maybe_unused]], const T& max_value [[maybe_unused]]) {
    return std::bit_cast<uint32_t>(value);
  }
};

template <>
struct Data32BitCompression<double> {
  static inline uint32_t compress(double& value, const double& /*min_value*/, const double& /*max_value*/) {
    auto unsigned_value = std::bit_cast<uint64_t>(value);
    const auto high = static_cast<uint32_t>(unsigned_value >> 32u);
    const auto low = static_cast<uint32_t>(unsigned_value);

    std::size_t hash = 0;
    boost::hash_combine(hash, high);
    boost::hash_combine(hash, low);
    return static_cast<uint32_t>(hash);
  }
};

template <>
struct Data32BitCompression<int64_t> {
  static inline uint32_t compress(int64_t& value, const int64_t& min_value, const int64_t& max_value) {
    static constexpr auto MAX_ALLOWED_DIFFERENCE = std::numeric_limits<uint32_t>::max();
    if (max_value - min_value <= MAX_ALLOWED_DIFFERENCE) {
      return Data32BitCompression<uint32_t>::compress(value - min_value, 0, max_value - min_value);
    }
    auto unsigned_value = static_cast<uint64_t>(value);
    const auto high = static_cast<uint32_t>(unsigned_value >> 32u);
    const auto low = static_cast<uint32_t>(unsigned_value);

    std::size_t hash = 0;
    boost::hash_combine(hash, high);
    boost::hash_combine(hash, low);
    return static_cast<uint32_t>(hash);
  }
};

template <>
struct Data32BitCompression<hyrise::pmr_string> {
  static inline uint32_t compress(hyrise::pmr_string& value, const hyrise::pmr_string& min_value [[maybe_unused]],
                                  const hyrise::pmr_string& max_value [[maybe_unused]]) {
    auto key = uint32_t{0};
    const auto string_length = value.length();
    for (auto index = std::size_t{0}; index < string_length; index++) {
      key = ((key << 5u) + key) ^ static_cast<uint32_t>(value[index]);
    }
    return key;
  }
};

}  // namespace

namespace hyrise {

using radix_partition::Bucket;
using radix_partition::RadixPartition;

bool JoinSimdSortMerge::supports(const JoinConfiguration config) {
  return config.predicate_condition == PredicateCondition::Equals && config.left_data_type == config.right_data_type &&
         (config.join_mode == JoinMode::Inner || config.join_mode == JoinMode::Left ||
          config.join_mode == JoinMode::Right || config.join_mode == JoinMode::FullOuter);
}

JoinSimdSortMerge::JoinSimdSortMerge(const std::shared_ptr<const AbstractOperator>& left,
                                     const std::shared_ptr<const AbstractOperator>& right, const JoinMode mode,
                                     const OperatorJoinPredicate& primary_predicate,
                                     const std::vector<OperatorJoinPredicate>& secondary_predicates)
    : AbstractJoinOperator(OperatorType::JoinSortMerge, left, right, mode, primary_predicate, secondary_predicates,
                           std::make_unique<OperatorPerformanceData<OperatorSteps>>()) {}

std::shared_ptr<AbstractOperator> JoinSimdSortMerge::_on_deep_copy(
    const std::shared_ptr<AbstractOperator>& copied_left_input,
    const std::shared_ptr<AbstractOperator>& copied_right_input,
    std::unordered_map<const AbstractOperator*, std::shared_ptr<AbstractOperator>>& /*copied_ops*/) const {
  return std::make_shared<JoinSimdSortMerge>(copied_left_input, copied_right_input, _mode, _primary_predicate,
                                             _secondary_predicates);
}

void JoinSimdSortMerge::_on_set_parameters(const std::unordered_map<ParameterID, AllTypeVariant>& parameters) {}

std::shared_ptr<const Table> JoinSimdSortMerge::_on_execute() {
  Assert(supports({_mode, _primary_predicate.predicate_condition,
                   left_input_table()->column_data_type(_primary_predicate.column_ids.first),
                   right_input_table()->column_data_type(_primary_predicate.column_ids.second),
                   !_secondary_predicates.empty(), left_input_table()->type(), right_input_table()->type()}),
         "JoinSimdSortMerge does not support these parameters.");

  std::shared_ptr<const Table> left_input_table_ptr = _left_input->get_output();
  std::shared_ptr<const Table> right_input_table_ptr = _right_input->get_output();

  // Check column types
  const auto& left_column_type = left_input_table()->column_data_type(_primary_predicate.column_ids.first);
  DebugAssert(left_column_type == right_input_table()->column_data_type(_primary_predicate.column_ids.second),
              "Left and right column types do not match. The simd sort merge join requires matching column types.");

  // Create implementation to compute the join result
  resolve_data_type(left_column_type, [&](const auto type) {
    using ColumnDataType = typename decltype(type)::type;
    _impl = std::make_unique<JoinSimdSortMergeImpl<ColumnDataType>>(
        *this, left_input_table_ptr, right_input_table_ptr, _primary_predicate.column_ids.first,
        _primary_predicate.column_ids.second, _primary_predicate.predicate_condition, _mode, _secondary_predicates,
        dynamic_cast<OperatorPerformanceData<JoinSimdSortMerge::OperatorSteps>&>(*performance_data));
  });

  return _impl->_on_execute();
}

void JoinSimdSortMerge::_on_cleanup() {
  _impl.reset();
}

const std::string& JoinSimdSortMerge::name() const {
  static const auto name = std::string{"JoinSimdSortMerge"};
  return name;
}

constexpr std::size_t choose_count_per_vector() {
#if defined(__AVX512F__)
  return 8;
#else
  return 4;
#endif
}

template <typename SimdSortType, typename ColumnType>
RadixPartition<ColumnType> construct_and_sort_partitions(std::span<SimdElement> elements) {
  auto radix_partitioner = RadixPartition<ColumnType>(elements);
  radix_partitioner.execute();

  const auto num_partitions = RadixPartition<ColumnType>::num_partitions();
  for (auto partition_index = std::size_t{0}; partition_index < num_partitions; ++partition_index) {
    auto& bucket = radix_partitioner.bucket(partition_index);
    if (!bucket.size) {
      continue;
    }
    const auto count_per_vector = choose_count_per_vector();
    auto* input_pointer = bucket.template begin<SimdSortType>();
    auto* output_pointer = radix_partitioner.template get_working_memory<SimdSortType>(partition_index);
    DebugAssert((simd_sort::is_simd_aligned<SimdSortType, 64>(input_pointer)), "Input not cache aligned.");
    DebugAssert((simd_sort::is_simd_aligned<SimdSortType, 64>(output_pointer)), "Output not cache aligned.");

    simd_sort::sort<count_per_vector>(input_pointer, output_pointer, bucket.size);
    bucket.data = reinterpret_cast<SimdElement*>(output_pointer);
  }

  return radix_partitioner;
}

template <typename SortingType, typename RadixPartition>
simd_sort::simd_vector<SimdElement> merge_sorted_buckets(PerThread<RadixPartition>& partitions,
                                                         std::size_t bucket_index) {
  // From each partition take the bucket at bucket_index and merge them into one sorted list.
  auto sorted_buckets = std::vector<std::unique_ptr<Bucket>>{};
  sorted_buckets.reserve(partitions.size());

  for (auto& partition : partitions) {
    auto& bucket = partition.bucket(bucket_index);
    sorted_buckets.push_back(std::make_unique<Bucket>(bucket));
  }

  auto multiway_merger = multiway_merging::MultiwayMerger<choose_count_per_vector(), SortingType>(sorted_buckets);
  return std::move(multiway_merger.merge());
}

// template <typename SortingType, typename RadixPartition>
// simd_sort::simd_vector<SimdElement> merge_sorted_buckets(PerThread<RadixPartition>& partitions,
//                                                          std::size_t bucket_index) {
//   // From each partition take the bucket at bucket_index and merge them into one sorted list.
//   auto temp_a = simd_sort::simd_vector<SortingType>{};
//   auto temp_b = simd_sort::simd_vector<SortingType>{};
//   auto& merged_last = temp_a;
//   auto& output = temp_b;
//   for (auto& partition : partitions) {
//     auto& bucket = partition.bucket(bucket_index);
//     std::merge(merged_last.begin(), merged_last.end(), bucket.template begin<SortingType>(),
//                bucket.template end<SortingType>(), std::back_inserter(output));
//     std::swap(merged_last, output);
//     output.clear();
//   }
//
//   DebugAssert(std::is_sorted(merged_last.begin(), merged_last.end()), "Merged output is not sorted.");
//
//   auto final_output = simd_sort::simd_vector<SimdElement>();
//   final_output.reserve(merged_last.size());
//   for (auto& element : merged_last) {
//     final_output.push_back(std::bit_cast<SimdElement>(element));
//   }
//
//   return final_output;
// }

template <typename ColumnType>
class JoinSimdSortMerge::JoinSimdSortMergeImpl : public AbstractReadOnlyOperatorImpl {
 public:
  JoinSimdSortMergeImpl(JoinSimdSortMerge& sort_merge_join, const std::shared_ptr<const Table>& left_input_table,
                        const std::shared_ptr<const Table>& right_input_table, ColumnID left_column_id,
                        ColumnID right_column_id, const PredicateCondition op, JoinMode mode,
                        const std::vector<OperatorJoinPredicate>& secondary_join_predicates,
                        OperatorPerformanceData<JoinSimdSortMerge::OperatorSteps>& performance_data)
      : _sort_merge_join{sort_merge_join},
        _left_input_table{left_input_table},
        _right_input_table{right_input_table},
        _performance{performance_data},
        _primary_left_column_id{left_column_id},
        _primary_right_column_id{right_column_id},
        _primary_predicate_condition{op},
        _mode{mode},
        _secondary_join_predicates{secondary_join_predicates} {
    _output_pos_lists_left.resize(radix_partition::PARTITION_SIZE);
    _output_pos_lists_right.resize(radix_partition::PARTITION_SIZE);
  }

 protected:
  // NOLINTBEGIN(cppcoreguidelines-avoid-const-or-ref-data-members)
  JoinSimdSortMerge& _sort_merge_join;
  const std::shared_ptr<const Table> _left_input_table;
  const std::shared_ptr<const Table> _right_input_table;

  OperatorPerformanceData<JoinSimdSortMerge::OperatorSteps>& _performance;

  const ColumnID _primary_left_column_id;
  const ColumnID _primary_right_column_id;

  const PredicateCondition _primary_predicate_condition;
  const JoinMode _mode;

  std::vector<MaterializedValue<ColumnType>> _materialized_values_left;
  std::vector<MaterializedValue<ColumnType>> _materialized_values_right;

  // Contains the null value row ids if a join column is an outer join column.
  RowIDPosList _null_rows_left;
  RowIDPosList _null_rows_right;

  SimdElementList _simd_elements_left;
  SimdElementList _simd_elements_right;

  PerHash<simd_sort::simd_vector<SimdElement>> _sorted_per_hash_left;
  PerHash<simd_sort::simd_vector<SimdElement>> _sorted_per_hash_right;

  const std::vector<OperatorJoinPredicate>& _secondary_join_predicates;

  // Contains the output row ids for each cluster.
  std::vector<RowIDPosList> _output_pos_lists_left;
  std::vector<RowIDPosList> _output_pos_lists_right;

  // NOLINTEND(cppcoreguidelines-avoid-const-or-ref-data-members)

  enum class CompareResult { Less, Greater, Equal };

  struct RowHasher {
    size_t operator()(const RowID& row) const {
      auto seed = size_t{0};
      boost::hash_combine(seed, row.chunk_id);
      boost::hash_combine(seed, row.chunk_offset);
      return seed;
    }
  };

  using RowHashSet = boost::unordered_flat_set<RowID, RowHasher>;

  struct PotentialMatchRange {
    PotentialMatchRange(std::size_t init_start_index, std::size_t init_end_index, std::span<SimdElement> init_elements)
        : start_index(init_start_index), end_index(init_end_index), elements(init_elements) {}

    std::size_t start_index;
    std::size_t end_index;
    std::span<SimdElement> elements;

    // Executes the given action for every row id of the table in this range.
    void for_every_row_id(const std::vector<MaterializedValue<ColumnType>>& materialized_values, auto&& action) const {
      for (auto index = start_index; index < end_index; ++index) {
        const auto& element = elements[index];
        const auto& materialized_value = materialized_values[element.index];
        if constexpr (requires { action(materialized_value.row_id, materialized_value.value); }) {
          action(materialized_value.row_id, materialized_value.value);
        } else {
          action(materialized_value.row_id);
        }
      }
    }

    void find_matches_with_range(const PotentialMatchRange& other_range,
                                 const std::vector<MaterializedValue<ColumnType>>& materialized_values_left,
                                 const std::vector<MaterializedValue<ColumnType>>& materialized_values_right,
                                 auto&& action) const {
      this->for_every_row_id(materialized_values_left, [&](const RowID left_row_id, const ColumnType& value_left) {
        other_range.for_every_row_id(materialized_values_right,
                                     [&](const RowID right_row_id, const ColumnType& value_right) {
                                       if (value_left != value_right) {
                                         return;
                                       }
                                       action(left_row_id, right_row_id);
                                     });
      });
    }
  };

  std::size_t _equal_value_range_size(std::size_t start_index, simd_sort::simd_vector<SimdElement>& elements) {
    if (start_index >= elements.size()) {
      return 0;
    }
    auto begin = elements.begin();
    std::advance(begin, start_index);
    const auto run_value = begin->key;

    constexpr auto LINEAR_SEARCH_ITEMS = std::size_t{128};
    auto end = begin + LINEAR_SEARCH_ITEMS;
    if (start_index + LINEAR_SEARCH_ITEMS >= elements.size()) {
      // Set end of linear search to end of input vector if we would overshoot otherwise.
      end = elements.end();
    }

    const auto linear_search_result = std::find_if(begin, end, [&](const auto& simd_element) {
      return simd_element.key > run_value;
    });

    if (linear_search_result != end) {
      // Match found within the linearly scanned part.
      return std::distance(begin, linear_search_result);
    }

    if (linear_search_result == elements.end()) {
      // We did not find a larger value in the linearly scanned part and it spanned until the end of the input vector.
      // That means all values up to the end are part of the run.
      return std::distance(begin, end);
    }

    // Binary search in case the run did not end within the linearly scanned part.
    const auto binary_search_result = std::upper_bound(end, elements.end(), *end, [](const auto& lhs, const auto& rhs) {
      return lhs.key < rhs.key;
    });
    return std::distance(begin, binary_search_result);
  }

  void _emit_combination(std::size_t bucket_index, RowID left_row_id, RowID right_row_id) {
    _output_pos_lists_left[bucket_index].push_back(left_row_id);
    _output_pos_lists_right[bucket_index].push_back(right_row_id);
  }

  // Only for multi predicated inner joins.
  // Emits all the combinations of row ids from the left table range and the right table range to the join output
  // where the secondary predicates are satisfied.
  void _emit_combinations_multi_predicated_inner(const std::size_t bucket_index, const PotentialMatchRange left_range,
                                                 const PotentialMatchRange right_range,
                                                 MultiPredicateJoinEvaluator& multi_predicate_join_evaluator
                                                 [[maybe_unused]]) {
    left_range.find_matches_with_range(
        right_range, _materialized_values_left, _materialized_values_right,
        [&](const RowID left_row_id, const RowID right_row_id) {
          if (multi_predicate_join_evaluator.satisfies_all_predicates(left_row_id, right_row_id)) {
            _emit_combination(bucket_index, left_row_id, right_row_id);
          }
        });
  }

  // Only for multi predicated left outer joins.
  // Emits all the combinations of row ids from the left table range and the right table range to the join output
  // where the secondary predicates are satisfied.
  // For a left row id without a match, the combination [left row id|NULL row id] is emitted.
  void _emit_combinations_multi_predicated_left_outer(const std::size_t bucket_index,
                                                      const PotentialMatchRange& left_range,
                                                      const PotentialMatchRange& right_range,
                                                      MultiPredicateJoinEvaluator& multi_predicate_join_evaluator) {
    DebugAssert(_primary_predicate_condition == PredicateCondition::Equals, "Primary predicate has to be Equals.");

    left_range.for_every_row_id(_materialized_values_left, [&](const RowID left_row_id, const ColumnType& left_value) {
      auto left_row_id_matched = false;
      right_range.for_every_row_id(
          _materialized_values_right, [&](const RowID right_row_id, const ColumnType& right_value) {
            if (multi_predicate_join_evaluator.satisfies_all_predicates(left_row_id, right_row_id) &&
                left_value == right_value) {
              _emit_combination(bucket_index, left_row_id, right_row_id);
              left_row_id_matched = true;
            }
          });
      if (!left_row_id_matched) {
        _emit_combination(bucket_index, left_row_id, NULL_ROW_ID);
      }
    });
  }

  // Only for multi predicated right outer joins.
  // Emits all the combinations of row ids from the left table range and the right table range to the join output
  // where the secondary predicates are satisfied.
  // For a right row id without a match, the combination [NULL row id|right row id] is emitted.
  void _emit_combinations_multi_predicated_right_outer(const std::size_t bucket_index,
                                                       const PotentialMatchRange& left_range,
                                                       const PotentialMatchRange& right_range,
                                                       MultiPredicateJoinEvaluator& multi_predicate_join_evaluator) {
    DebugAssert(_primary_predicate_condition == PredicateCondition::Equals, "Primary predicate has to be Equals.");

    right_range.for_every_row_id(
        _materialized_values_right, [&](const RowID right_row_id, const ColumnType& right_value) {
          auto right_row_id_matched = false;
          left_range.for_every_row_id(
              _materialized_values_left, [&](const RowID left_row_id, const ColumnType& left_value) {
                if (multi_predicate_join_evaluator.satisfies_all_predicates(left_row_id, right_row_id) &&
                    left_value == right_value) {
                  _emit_combination(bucket_index, left_row_id, right_row_id);
                  right_row_id_matched = true;
                }
              });
          if (!right_row_id_matched) {
            _emit_combination(bucket_index, NULL_ROW_ID, right_row_id);
          }
        });
  }

  // Only for multi-predicate full outer joins.
  // Emits all the combinations of row ids from the left table range and the right table range to the join output
  // where the secondary predicates are satisfied.
  // For a left row id without a match, the combination [right row id|NULL row id] is emitted.
  // For a right row id without a match, the combination [NULL row id|right row id] is emitted.
  void _emit_combinations_multi_predicated_full_outer(const std::size_t bucket_index,
                                                      const PotentialMatchRange& left_range,
                                                      const PotentialMatchRange& right_range,
                                                      MultiPredicateJoinEvaluator& multi_predicate_join_evaluator) {
    DebugAssert(_primary_predicate_condition == PredicateCondition::Equals, "Primary predicate has to be Equals.");
    auto matched_right_row_ids = RowHashSet{};

    left_range.for_every_row_id(_materialized_values_left, [&](const RowID left_row_id, const ColumnType& left_value) {
      auto left_row_id_matched = false;
      right_range.for_every_row_id(
          _materialized_values_right, [&](const RowID right_row_id, const ColumnType& right_value) {
            if (multi_predicate_join_evaluator.satisfies_all_predicates(left_row_id, right_row_id) &&
                left_value == right_value) {
              _emit_combination(bucket_index, left_row_id, right_row_id);
              left_row_id_matched = true;
              matched_right_row_ids.insert(right_row_id);
            }
          });
      if (!left_row_id_matched) {
        _emit_combination(bucket_index, left_row_id, NULL_ROW_ID);
      }
    });

    // Add null value combinations for right row ids that have no match.
    right_range.for_every_row_id(_materialized_values_right, [&](RowID right_row_id) {
      // Right_row_ids_with_match has no key `right_row_id`.
      if (!matched_right_row_ids.contains(right_row_id)) {
        _emit_combination(bucket_index, NULL_ROW_ID, right_row_id);
      }
    });
  }

  // Emits all the combinations of row ids from the left table range and the right table range to the join output
  // where also the secondary predicates are satisfied.
  void _emit_qualified_combinations(const std::size_t bucket_index, const PotentialMatchRange& left_range,
                                    const PotentialMatchRange& right_range,
                                    std::optional<MultiPredicateJoinEvaluator>& multi_predicate_join_evaluator) {
    if (multi_predicate_join_evaluator) {
      if (_mode == JoinMode::Inner) {
        _emit_combinations_multi_predicated_inner(bucket_index, left_range, right_range,
                                                  multi_predicate_join_evaluator.value());
      } else if (_mode == JoinMode::Left) {
        _emit_combinations_multi_predicated_left_outer(bucket_index, left_range, right_range,
                                                       multi_predicate_join_evaluator.value());
      } else if (_mode == JoinMode::Right) {
        _emit_combinations_multi_predicated_right_outer(bucket_index, left_range, right_range,
                                                        multi_predicate_join_evaluator.value());
      } else if (_mode == JoinMode::FullOuter) {
        _emit_combinations_multi_predicated_full_outer(bucket_index, left_range, right_range,
                                                       multi_predicate_join_evaluator.value());
      }
    } else {
      // no secondary join predicates
      left_range.find_matches_with_range(right_range, _materialized_values_left, _materialized_values_right,
                                         [&](const RowID left_row_id, const RowID right_row_id) {
                                           _emit_combination(bucket_index, left_row_id, right_row_id);
                                         });
    }
  }

  // Emits all combinations of row ids from the left table range and a NULL value on the right side
  // (regarding the primary predicate) to the join output.
  void _emit_right_primary_null_combinations(const std::size_t bucket_index, const PotentialMatchRange& left_range) {
    left_range.for_every_row_id(_materialized_values_left, [&](RowID left_row_id) {
      _emit_combination(bucket_index, left_row_id, NULL_ROW_ID);
    });
  }

  // Emits all combinations of row ids from the right table range and a NULL value on the left side
  // (regarding the primary predicate) to the join output.
  void _emit_left_primary_null_combinations(const std::size_t bucket_index, const PotentialMatchRange& right_range) {
    right_range.for_every_row_id(_materialized_values_right, [&](RowID right_row_id) {
      _emit_combination(bucket_index, NULL_ROW_ID, right_row_id);
    });
  }

  void _find_matches_in_ranges(const PotentialMatchRange& left_range, const PotentialMatchRange& right_range,
                               const CompareResult compare_result,
                               std::optional<MultiPredicateJoinEvaluator>& multi_predicate_join_evaluator,
                               const std::size_t bucket_index) {
    DebugAssert(_primary_predicate_condition == PredicateCondition::Equals,
                "Primary predicate condition has to be EQUALS!");
    if (compare_result == CompareResult::Equal) {
      _emit_qualified_combinations(bucket_index, left_range, right_range, multi_predicate_join_evaluator);
    } else if (compare_result == CompareResult::Less) {
      if (_mode == JoinMode::Left || _mode == JoinMode::FullOuter) {
        _emit_right_primary_null_combinations(bucket_index, left_range);
      }
    } else if (compare_result == CompareResult::Greater) {
      if (_mode == JoinMode::Right || _mode == JoinMode::FullOuter) {
        _emit_left_primary_null_combinations(bucket_index, right_range);
      }
    }
  }

  // Compares two values and creates a comparison result.
  template <typename T>
  CompareResult _compare(const T& left, const T& right) {
    if (left < right) {
      return CompareResult::Less;
    }

    if (left == right) {
      return CompareResult::Equal;
    }

    return CompareResult::Greater;
  }

  // Currently we only support Inner, Equi-Joins.
  void _join_per_hash(std::size_t bucket_index, simd_sort::simd_vector<SimdElement>& left_elements,
                      simd_sort::simd_vector<SimdElement>& right_elements) {
    auto multi_predicate_join_evaluator = std::optional<MultiPredicateJoinEvaluator>{};
    if (!_secondary_join_predicates.empty()) {
      multi_predicate_join_evaluator.emplace(*_sort_merge_join._left_input->get_output(),
                                             *_sort_merge_join.right_input()->get_output(), _mode,
                                             _secondary_join_predicates);
    }

    auto left_run_start = size_t{0};
    auto right_run_start = size_t{0};

    auto left_run_end = left_run_start + _equal_value_range_size(left_run_start, left_elements);
    auto right_run_end = right_run_start + _equal_value_range_size(right_run_start, right_elements);

    const auto left_size = left_elements.size();
    const auto right_size = right_elements.size();

    while (left_run_start < left_size && right_run_start < right_size) {
      const auto& left_value = left_elements[left_run_start].key;
      const auto& right_value = right_elements[right_run_start].key;

      const auto compare_result = _compare(left_value, right_value);

      const auto left_range = PotentialMatchRange(left_run_start, left_run_end, left_elements);
      const auto right_range = PotentialMatchRange(right_run_start, right_run_end, right_elements);

      _find_matches_in_ranges(left_range, right_range, compare_result, multi_predicate_join_evaluator, bucket_index);

      // Advance to the next run on the smaller side or both if equal.
      switch (compare_result) {
        case CompareResult::Equal:
          // Advance both runs.
          left_run_start = left_run_end;
          right_run_start = right_run_end;
          left_run_end = left_run_start + _equal_value_range_size(left_run_start, left_elements);
          right_run_end = right_run_start + _equal_value_range_size(right_run_start, right_elements);
          break;
        case CompareResult::Less:
          // Advance the left run.
          left_run_start = left_run_end;
          left_run_end = left_run_start + _equal_value_range_size(left_run_start, left_elements);
          break;
        case CompareResult::Greater:
          // Advance the right run.
          right_run_start = right_run_end;
          right_run_end = right_run_start + _equal_value_range_size(right_run_start, right_elements);
          break;
        default:
          throw std::logic_error("Unknown CompareResult.");
      }
    }

    const auto left_remainder = PotentialMatchRange(left_run_start, left_size, left_elements);
    const auto right_remainder = PotentialMatchRange(right_run_start, right_size, right_elements);
    if (left_run_start < left_size) {
      _find_matches_in_ranges(left_remainder, right_remainder, CompareResult::Less, multi_predicate_join_evaluator,
                              bucket_index);
    } else if (right_run_start < right_size) {
      _find_matches_in_ranges(left_remainder, right_remainder, CompareResult::Greater, multi_predicate_join_evaluator,
                              bucket_index);
    }
  }

  void _perform_join() {
    spawn_and_wait_per_hash([this](std::size_t bucket_index) {
      _join_per_hash(bucket_index, _sorted_per_hash_left[bucket_index], _sorted_per_hash_right[bucket_index]);
    });
  }

  template <typename SortingType, JoinSimdSortMerge::OperatorSteps partition_and_sort_step,
            JoinSimdSortMerge::OperatorSteps multiway_merge_step>
  PerHash<simd_sort::simd_vector<SimdElement>> _sort_relation(SimdElementList& simd_elements) {
    auto timer = Timer{};

    const auto num_items = simd_elements.size();
    const auto base_size_per_thread = num_items / THREAD_COUNT;
    const auto remainder = num_items % THREAD_COUNT;

    using RadixPartition = RadixPartition<ColumnType>;

    auto thread_partitions = PerThread<RadixPartition>{};
    auto thread_inputs = PerThread<std::span<SimdElement>>{};

    // Split simd_elements into THREAD_COUNT many sections.
    auto* input_start_address = simd_elements.data();
    for (auto thread_index = std::size_t{0}, offset = std::size_t{0}; thread_index < THREAD_COUNT; ++thread_index) {
      auto current_range_size = base_size_per_thread + (thread_index < remainder ? 1 : 0);
      thread_inputs[thread_index] = {input_start_address + offset, input_start_address + offset + current_range_size};
      offset += current_range_size;
    }

    spawn_and_wait_per_thread(thread_inputs, [&thread_partitions](auto& elements, std::size_t thread_index) {
      thread_partitions[thread_index] = construct_and_sort_partitions<SortingType, ColumnType>(elements);
    });

    _performance.set_step_runtime(partition_and_sort_step, timer.lap());

#if HYRISE_DEBUG
    for (auto thread_index = 0u; thread_index < THREAD_COUNT; ++thread_index) {
      const auto& buckets = thread_partitions[thread_index].buckets();
      for (auto bucket : buckets) {
        // Check that each bucket is sorted according to SortingType and the key of the SimdElement;
        DebugAssert(std::is_sorted(bucket.template begin<SortingType>(), bucket.template end<SortingType>()),
                    "Partition was not sorted correctly.");
      }
    }
#endif

    auto merged_outputs = PerHash<simd_sort::simd_vector<SimdElement>>{};
    spawn_and_wait_per_hash(merged_outputs, [&thread_partitions](auto& merged_output, std::size_t bucket_index) {
      merged_output = merge_sorted_buckets<SortingType>(thread_partitions, bucket_index);
    });

    _performance.set_step_runtime(multiway_merge_step, timer.lap());

    return merged_outputs;
  }

  template <typename T, JoinSimdSortMerge::OperatorSteps materialize_step,
            JoinSimdSortMerge::OperatorSteps transform_step>
  void _materialize_column_and_transform_to_simd_format(const std::shared_ptr<const Table> table,
                                                        const ColumnID column_id,
                                                        std::vector<MaterializedValue<T>>& materialized_values,
                                                        SimdElementList& simd_element_list, RowIDPosList& null_values,
                                                        const bool materialize_null) {
    auto timer = Timer{};
    auto left_column_materializer = SMJColumnMaterializer<T>(JoinSimdSortMerge::JOB_SPAWN_THRESHOLD);
    auto [materialized_segments, min_value, max_value, null_rows] =
        left_column_materializer.materialize(table, column_id, materialize_null);
    null_values = std::move(null_rows);

    _performance.set_step_runtime(materialize_step, timer.lap());

    simd_element_list.reserve(materialized_segments.size());
    auto index = size_t{0};
    for (auto& segment : materialized_segments) {
      for (auto& materialized_value : segment) {
        DebugAssert(index <= std::numeric_limits<uint32_t>::max(), "Index has to fit into 32 bits. ");
        const auto sorting_key = Data32BitCompression<T>::compress(materialized_value.value, min_value, max_value);
        simd_element_list.emplace_back(static_cast<uint32_t>(index), sorting_key);
        materialized_values.push_back(std::move(materialized_value));
        ++index;
      }
    }
    _performance.set_step_runtime(transform_step, timer.lap());
  }

 public:
  std::shared_ptr<const Table> _on_execute() override {
#if HYRISE_DEBUG
    std::cout << "Execute JoinSimdSortMerge" << std::endl;
#endif

    const auto include_null_left = (_mode == JoinMode::Left || _mode == JoinMode::FullOuter);
    const auto include_null_right = (_mode == JoinMode::Right || _mode == JoinMode::FullOuter);

    using enum OperatorSteps;

    _materialize_column_and_transform_to_simd_format<ColumnType, LeftSideMaterialize, LeftSideTransform>(
        _left_input_table, _primary_left_column_id, _materialized_values_left, _simd_elements_left, _null_rows_left,
        include_null_left);

    _materialize_column_and_transform_to_simd_format<ColumnType, RightSideMaterialize, RightSideTransform>(
        _right_input_table, _primary_right_column_id, _materialized_values_right, _simd_elements_right,
        _null_rows_right, include_null_right);

    _sorted_per_hash_left = std::move(
        _sort_relation<SortingType, LeftSidePartitionAndSortBuckets, LeftSideMultiwayMerging>(_simd_elements_left));

    _sorted_per_hash_right = std::move(
        _sort_relation<SortingType, RightSidePartitionAndSortBuckets, RightSideMultiwayMerging>(_simd_elements_right));

    auto timer = Timer{};

    _perform_join();

    _performance.set_step_runtime(FindJoinPartner, timer.lap());

    if (include_null_left || include_null_right) {
      auto null_output_left = RowIDPosList();
      auto null_output_right = RowIDPosList();

      // Add the outer join rows which had a null value in their join column.
      if (include_null_left) {
        null_output_left.insert(null_output_left.end(), _null_rows_left.begin(), _null_rows_left.end());
        null_output_right.insert(null_output_right.end(), _null_rows_left.size(), NULL_ROW_ID);
      }
      if (include_null_right) {
        null_output_left.insert(null_output_left.end(), _null_rows_right.size(), NULL_ROW_ID);
        null_output_right.insert(null_output_right.end(), _null_rows_right.begin(), _null_rows_right.end());
      }

      DebugAssert(null_output_left.size() == null_output_right.size(),
                  "Null position lists are expected to be of equal length.");
      if (!null_output_left.empty()) {
        _output_pos_lists_left.push_back(std::move(null_output_left));
        _output_pos_lists_right.push_back(std::move(null_output_right));
      }
    }

    // _performance.set_step_runtime(OperatorSteps::FindJoinPartner, timer.lap());

    const auto create_left_side_pos_lists_by_segment = (_left_input_table->type() == TableType::References);
    const auto create_right_side_pos_lists_by_segment = (_right_input_table->type() == TableType::References);

    // A sort merge join's input can be heavily pre-filtered or the join results in very few matches. In contrast to
    // the hash join, we do not (for now) merge small partitions to keep the sorted chunk guarantees, which could be
    // exploited by subsequent operators.
    constexpr auto ALLOW_PARTITION_MERGE = false;
    auto output_chunks =
        write_output_chunks(_output_pos_lists_left, _output_pos_lists_right, _left_input_table, _right_input_table,
                            create_left_side_pos_lists_by_segment, create_right_side_pos_lists_by_segment,
                            OutputColumnOrder::LeftFirstRightSecond, ALLOW_PARTITION_MERGE);

    const ColumnID left_join_column = _sort_merge_join._primary_predicate.column_ids.first;
    const ColumnID right_join_column = static_cast<ColumnID>(_sort_merge_join.left_input_table()->column_count() +
                                                             _sort_merge_join._primary_predicate.column_ids.second);

    for (auto& chunk : output_chunks) {
      if (_sort_merge_join._primary_predicate.predicate_condition == PredicateCondition::Equals &&
          _mode == JoinMode::Inner) {
        chunk->set_immutable();
      }
    }

    _performance.set_step_runtime(OperatorSteps::OutputWriting, timer.lap());

    auto result_table = _sort_merge_join._build_output_table(std::move(output_chunks));

    if (_mode != JoinMode::Left && _mode != JoinMode::Right && _mode != JoinMode::FullOuter &&
        _sort_merge_join._primary_predicate.predicate_condition == PredicateCondition::Equals) {
      // Table clustering is not defined for columns storing NULL values. Additionally, clustering is not given for
      // non-equal predicates.
      result_table->set_value_clustered_by({left_join_column, right_join_column});
    }
    return result_table;
  }
};

}  // namespace hyrise
