#include "join_simd_sort_merge.hpp"

#include <boost/container_hash/hash.hpp>
#include <boost/functional/hash.hpp>
#include <boost/unordered/unordered_flat_set.hpp>

#include "operators/join_helper/join_output_writing.hpp"
#include "operators/join_simd_sort_merge/k_way_merge.hpp"
#include "operators/join_simd_sort_merge/multiway_merging.hpp"
#include "operators/join_simd_sort_merge/radix_partitioning.hpp"
#include "operators/join_simd_sort_merge/simd_sort.hpp"
#include "operators/join_simd_sort_merge/simd_utils.hpp"
#include "operators/join_sort_merge/column_materializer.hpp"
#include "operators/multi_predicate_join/multi_predicate_join_evaluator.hpp"
#include "utils/timer.hpp"

#if defined(__x86_64__)
#include "immintrin.h"
#endif

#include <algorithm>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <utility>

// #include "operators/join_simd_sort_merge/column_materializer.hpp"
#include "types.hpp"
#include "utils/assert.hpp"

namespace {
template <typename T>
concept FourByteType = (sizeof(T) == 4);

template <typename Callable, typename T1, typename T2>
concept RequiresTwoParameters = requires(Callable callable) {
  { callable(std::declval<T1>(), std::declval<T2>()) };
};

template <typename T>
struct Data32BitCompression {
  static uint32_t compress(T& value [[maybe_unused]], const T& min_value [[maybe_unused]],
                           const T& max_value [[maybe_unused]]) {
    static_assert(false, "Not implemented");
  }
};

template <FourByteType T>
struct Data32BitCompression<T> {
  static uint32_t compress(T value, const T& min_value [[maybe_unused]], const T& max_value [[maybe_unused]]) {
    return std::bit_cast<uint32_t>(value);
  }
};

template <>
struct Data32BitCompression<double> {
  static uint32_t compress(double& value, const double& /*min_value*/, const double& /*max_value*/) {
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
  static uint32_t compress(int64_t& value, const int64_t& min_value, const int64_t& max_value) {
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
  static uint32_t compress(hyrise::pmr_string& value, const hyrise::pmr_string& min_value [[maybe_unused]],
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

  // auto multiway_merger = multiway_merging::MultiwayMerger<choose_count_per_vector(), SortingType>(sorted_buckets);
  // return multiway_merger.merge();
  auto k_way_merger = k_way_merge::KWayMerge<SortingType>(sorted_buckets);
  return k_way_merger.merge();
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
        _cluster_count(_determine_number_of_clusters()),
        _secondary_join_predicates{secondary_join_predicates} {
    _output_pos_lists_left.resize(_cluster_count);
    _output_pos_lists_right.resize(_cluster_count);
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

  size_t _cluster_count;

  std::vector<ColumnType> _materialized_values_left;
  std::vector<ColumnType> _materialized_values_right;
  RowIDPosList _row_ids_left;
  RowIDPosList _row_ids_right;

  std::vector<simd_sort::simd_vector<ColumnType>> _sorted_values_left;
  std::vector<simd_sort::simd_vector<ColumnType>> _sorted_values_right;
  std::vector<simd_sort::simd_vector<RowID>> _sorted_row_ids_left;
  std::vector<simd_sort::simd_vector<RowID>> _sorted_row_ids_right;

  // Contains the null value row ids if a join column is an outer join column.
  RowIDPosList _null_rows_left;
  RowIDPosList _null_rows_right;

  SimdElementList _simd_elements_left;
  SimdElementList _simd_elements_right;

  SimdElementList _partition_storage_left;
  SimdElementList _partition_storage_right;

  SimdElementList _working_memory_left;
  SimdElementList _working_memory_right;

  std::vector<std::span<SimdElement>> _sorted_per_hash_left;
  std::vector<std::span<SimdElement>> _sorted_per_hash_right;

  const std::vector<OperatorJoinPredicate>& _secondary_join_predicates;

  // Contains the output row ids for each cluster.
  std::vector<RowIDPosList> _output_pos_lists_left;
  std::vector<RowIDPosList> _output_pos_lists_right;

  // NOLINTEND(cppcoreguidelines-avoid-const-or-ref-data-members)

  enum class CompareResult : std::uint8_t { Less, Greater, Equal };

  static constexpr auto IS_LOSSLESS_COMPRESSION = sizeof(ColumnType) == 4 && !std::is_same_v<ColumnType, pmr_string>;

  struct RowHasher {
    size_t operator()(const RowID& row) const {
      auto seed = size_t{0};
      boost::hash_combine(seed, row.chunk_id);
      boost::hash_combine(seed, row.chunk_offset);
      return seed;
    }
  };

  using RowHashSet = boost::unordered_flat_set<RowID, RowHasher>;

  // Determines the number of clusters to be used for the join. The number of clusters must be a power of two.
  size_t _determine_number_of_clusters() {
    // We try to have a partition size of roughly 256 KB to limit out-of-cache sorting and increase parallelism. This
    // value has been determined by an array of benchmarks and should be revisited for larger changes to the operator.
    // Ideally, it would incorporate hardware knowledge such as the actual L2 cache size of the current system.
    constexpr auto MAX_SORT_ITEMS_COUNT = 1'048'576;  // 4'194'304;
    const size_t cluster_count_left = _sort_merge_join.left_input_table()->row_count() / MAX_SORT_ITEMS_COUNT;
    const size_t cluster_count_right = _sort_merge_join.right_input_table()->row_count() / MAX_SORT_ITEMS_COUNT;

    // Return the next smaller power of two for the larger of the two cluster counts. Do not use more than 2^8 clusters
    // as TLB misses during clustering become too expensive (see "An Experimental Comparison of Thirteen Relational
    // Equi-Joins in Main Memory" by Schuh et al.).
    return static_cast<size_t>(std::pow(
        2, std::min(8.0, std::floor(std::log2(std::max({size_t{1}, cluster_count_left, cluster_count_right}))))));
  }

  struct PotentialMatchRange {
    PotentialMatchRange(std::size_t init_start_index, std::size_t init_end_index, std::span<SimdElement> init_elements,
                        std::span<ColumnType> init_values, std::span<RowID> init_row_ids)
        : start_index(init_start_index),
          end_index(init_end_index),
          elements(init_elements),
          values(init_values.data() + init_start_index, init_values.data() + init_end_index),
          row_ids(init_row_ids.data() + init_start_index, init_row_ids.data() + init_end_index) {}

    std::size_t start_index;
    std::size_t end_index;
    std::span<SimdElement> elements;
    std::span<ColumnType> values;
    std::span<RowID> row_ids;

   public:
    // Executes the given action for every row id of the table in this range.
    void for_every_row_id(auto&& action) const {
      const auto num_items = row_ids.size();
      for (auto index = size_t{0}; index < num_items; ++index) {
        DebugAssert(values.size() > index || IS_LOSSLESS_COMPRESSION, "Values has broken size.");
        const auto& row_id = row_ids[index];
        if constexpr (requires { action(row_id); }) {
          action(row_id);
        } else {
          const auto& value = IS_LOSSLESS_COMPRESSION ? ColumnType{} : values[index];
          action(row_id, value);
        }
      }
    }

    void find_matches_with_range(const PotentialMatchRange& other_range, auto&& action) const {
      // Handle float and int32_t values.
      if constexpr (IS_LOSSLESS_COMPRESSION) {
        this->for_every_row_id([&](const RowID& left_row_id) {
          other_range.for_every_row_id([&](const RowID& right_row_id) {
            action(left_row_id, right_row_id);
          });
        });
        return;
      }
      // Handle pmr_string values.
      this->for_every_row_id([&](const RowID& left_row_id, const ColumnType& value_left) {
        other_range.for_every_row_id([&](const RowID& right_row_id, const ColumnType& value_right) {
          if (value_left != value_right) {
            return;
          }
          action(left_row_id, right_row_id);
        });
      });
    }
  };

  std::size_t _equal_value_range_size(std::size_t start_index, std::span<SimdElement>& elements) {
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
                                                 MultiPredicateJoinEvaluator& multi_predicate_join_evaluator) {
    left_range.find_matches_with_range(right_range, [&](const RowID& left_row_id, const RowID& right_row_id) {
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

    left_range.for_every_row_id([&](const RowID left_row_id, const ColumnType& left_value) {
      auto left_row_id_matched = false;
      right_range.for_every_row_id([&](const RowID right_row_id, const ColumnType& right_value) {
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

    right_range.for_every_row_id([&](const RowID right_row_id, const ColumnType& right_value) {
      auto right_row_id_matched = false;
      left_range.for_every_row_id([&](const RowID left_row_id, const ColumnType& left_value) {
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

    left_range.for_every_row_id([&](const RowID left_row_id, const ColumnType& left_value) {
      auto left_row_id_matched = false;
      right_range.for_every_row_id([&](const RowID right_row_id, const ColumnType& right_value) {
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
    right_range.for_every_row_id([&](RowID right_row_id) {
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
      left_range.find_matches_with_range(right_range, [&](const RowID& left_row_id, const RowID& right_row_id) {
        _emit_combination(bucket_index, left_row_id, right_row_id);
      });
    }
  }

  // Emits all combinations of row ids from the left table range and a NULL value on the right side
  // (regarding the primary predicate) to the join output.
  void _emit_right_primary_null_combinations(const std::size_t bucket_index, const PotentialMatchRange& left_range) {
    left_range.for_every_row_id([&](RowID left_row_id) {
      _emit_combination(bucket_index, left_row_id, NULL_ROW_ID);
    });
  }

  // Emits all combinations of row ids from the right table range and a NULL value on the left side
  // (regarding the primary predicate) to the join output.
  void _emit_left_primary_null_combinations(const std::size_t bucket_index, const PotentialMatchRange& right_range) {
    right_range.for_every_row_id([&](RowID right_row_id) {
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
  void _join_per_hash(std::size_t bucket_index, std::span<SimdElement> left_elements,
                      std::span<SimdElement> right_elements, std::span<ColumnType> left_values,
                      std::span<ColumnType> right_values, std::span<RowID> left_row_ids,
                      std::span<RowID> right_row_ids) {
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

      const auto left_range =
          PotentialMatchRange(left_run_start, left_run_end, left_elements, left_values, left_row_ids);
      const auto right_range =
          PotentialMatchRange(right_run_start, right_run_end, right_elements, right_values, right_row_ids);

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

    const auto left_remainder =
        PotentialMatchRange(left_run_start, left_size, left_elements, left_values, left_row_ids);
    const auto right_remainder =
        PotentialMatchRange(right_run_start, right_size, right_elements, right_values, right_row_ids);
    if (left_run_start < left_size) {
      _find_matches_in_ranges(left_remainder, right_remainder, CompareResult::Less, multi_predicate_join_evaluator,
                              bucket_index);
    } else if (right_run_start < right_size) {
      _find_matches_in_ranges(left_remainder, right_remainder, CompareResult::Greater, multi_predicate_join_evaluator,
                              bucket_index);
    }
  }

  void _perform_join() {
    auto jobs = std::vector<std::shared_ptr<AbstractTask>>{};
    for (auto cluster_index = size_t{0}; cluster_index < _cluster_count; ++cluster_index) {
      const auto merge_row_count =
          _sorted_per_hash_left[cluster_index].size() + _sorted_per_hash_right[cluster_index].size();
      if (merge_row_count > JOB_SPAWN_THRESHOLD) {
        jobs.push_back(std::make_shared<JobTask>([cluster_index, this]() {
          _join_per_hash(cluster_index, _sorted_per_hash_left[cluster_index], _sorted_per_hash_right[cluster_index],
                         _sorted_values_left[cluster_index], _sorted_values_right[cluster_index],
                         _sorted_row_ids_left[cluster_index], _sorted_row_ids_right[cluster_index]);
        }));
      } else {
        _join_per_hash(cluster_index, _sorted_per_hash_left[cluster_index], _sorted_per_hash_right[cluster_index],
                       _sorted_values_left[cluster_index], _sorted_values_right[cluster_index],
                       _sorted_row_ids_left[cluster_index], _sorted_row_ids_right[cluster_index]);
      }
    }
    Hyrise::get().scheduler()->schedule_and_wait_for_tasks(jobs);
  }

  template <typename SortingType, OperatorSteps partition_step, OperatorSteps sort_buckets_step>
  std::vector<std::span<SimdElement>> _sort_relation(SimdElementList& simd_elements, SimdElementList& partition_storage,
                                                     SimdElementList& working_memory) {
    auto timer = Timer{};

    auto radix_partition = RadixPartition<ColumnType>(simd_elements, _cluster_count);
    radix_partition.execute(partition_storage, working_memory);

    _performance.set_step_runtime(partition_step, timer.lap());

    auto sort_bucket = [&radix_partition, &working_memory](size_t bucket_index) {
      auto& bucket = radix_partition.bucket(bucket_index);
      if (!bucket.size) {
        return;
      }
      const auto count_per_vector = choose_count_per_vector();
      auto* input_pointer = bucket.template begin<SortingType>();
      auto* output_pointer = radix_partition.template get_working_memory<SortingType>(bucket_index, working_memory);

      DebugAssert((simd_sort::is_simd_aligned<SortingType, 64>(input_pointer)), "Input not cache aligned.");
      DebugAssert((simd_sort::is_simd_aligned<SortingType, 64>(output_pointer)), "Output not cache aligned.");

      simd_sort::sort<count_per_vector>(input_pointer, output_pointer, bucket.size);
      bucket.data = reinterpret_cast<SimdElement*>(output_pointer);
    };

    auto jobs = std::vector<std::shared_ptr<AbstractTask>>{};
    for (auto bucket_index = size_t{0}; bucket_index < _cluster_count; ++bucket_index) {
      auto& bucket = radix_partition.bucket(bucket_index);
      if (bucket.size > JOB_SPAWN_THRESHOLD) {
        jobs.push_back(std::make_shared<JobTask>([&sort_bucket, bucket_index]() {
          sort_bucket(bucket_index);
        }));
      } else {
        sort_bucket(bucket_index);
      }
    }

    Hyrise::get().scheduler()->schedule_and_wait_for_tasks(jobs);

    auto sorted_clusters = std::vector<std::span<SimdElement>>{};
    sorted_clusters.reserve(_cluster_count);

    for (auto bucket : radix_partition.buckets()) {
      // Check that each bucket is sorted according to SortingType and the key of the SimdElement;
      DebugAssert(std::is_sorted(bucket.template begin<SortingType>(), bucket.template end<SortingType>()),
                  "Partition was not sorted correctly.");

      sorted_clusters.push_back(std::span(bucket.template begin<SimdElement>(), bucket.template end<SimdElement>()));
    }

    _performance.set_step_runtime(sort_buckets_step, timer.lap());

    return sorted_clusters;
  }

  template <typename T, JoinSimdSortMerge::OperatorSteps materialize_step,
            JoinSimdSortMerge::OperatorSteps transform_step>
  void _materialize_column_and_transform_to_simd_format(const std::shared_ptr<const Table> table,
                                                        const ColumnID column_id, std::vector<T>& materialized_values,
                                                        RowIDPosList& row_ids, SimdElementList& simd_element_list,
                                                        RowIDPosList& null_values, const bool materialize_null) {
    auto timer = Timer{};
    auto left_column_materializer = ColumnMaterializer<T>(false, materialize_null);
    auto [materialized_segments, null_rows, samples] = left_column_materializer.materialize(table, column_id);

    // auto left_column_materializer = SMJColumnMaterializer<T>(JoinSimdSortMerge::JOB_SPAWN_THRESHOLD);
    // auto [materialized_segments, min_value, max_value, null_rows] =
    //     left_column_materializer.materialize(table, column_id, materialize_null);
    null_values = std::move(null_rows);

    _performance.set_step_runtime(materialize_step, timer.lap());

    simd_element_list.reserve(materialized_segments.size());

    auto min_value = std::numeric_limits<T>::lowest();
    auto max_value = std::numeric_limits<T>::max();

    auto index = size_t{0};
    for (auto& segment : materialized_segments) {
      for (auto& materialized_value : segment) {
        const auto sorting_key = Data32BitCompression<T>::compress(materialized_value.value, min_value, max_value);
        simd_element_list.emplace_back(static_cast<uint32_t>(index), sorting_key);
        materialized_values.push_back(std::move(materialized_value.value));
        row_ids.push_back(std::move(materialized_value.row_id));
        ++index;
      }
    }
    DebugAssert(index <= std::numeric_limits<uint32_t>::max(), "Index has to fit into 32 bits. ");

    _performance.set_step_runtime(transform_step, timer.lap());
  }

  void _materialized_row_ids_and_values(const std::span<SimdElement> sorted_elements, const RowIDPosList& row_ids,
                                        simd_sort::simd_vector<RowID>& output_row_ids,
                                        const std::span<ColumnType> values [[maybe_unused]],
                                        simd_sort::simd_vector<ColumnType>& output_values [[maybe_unused]]) {
#ifdef __AVX512F__
    static constexpr auto GATHER_SIZE = 8;
#else
    static constexpr auto GATHER_SIZE = 4;
#endif
    constexpr auto BATCH_SIZE = GATHER_SIZE * 8;

    constexpr auto ALIGNMENT_BIT_MASK = simd_sort::get_alignment_bitmask<GATHER_SIZE>();
    const auto num_items = sorted_elements.size();
    const auto aligned_size = num_items & ALIGNMENT_BIT_MASK;
    const auto remaining_size = num_items - aligned_size;

    output_row_ids.resize(num_items);
    if constexpr (!IS_LOSSLESS_COMPRESSION) {
      output_values.resize(num_items);
    }

    auto gather_indices = [&](const uint64_t* __restrict index_begin) {
      auto indices = std::array<uint32_t, GATHER_SIZE>{};

#pragma clang loop vectorize(enable) interleave(enable) unroll(disable)
      for (auto index = size_t{0}; index < GATHER_SIZE; ++index) {
        indices[index] = ((index_begin[index]) << 32u) >> 32u;
      }
      return indices;
    };

    auto gather_row_ids = [&](const uint32_t* __restrict index_begin, const RowID* __restrict row_ids_data,
                              RowID* __restrict row_ids_output) {
      auto loaded_data = std::array<RowID, GATHER_SIZE>{};
#pragma clang loop vectorize(enable) interleave(enable) unroll(disable)
      for (auto index = size_t{0}; index < GATHER_SIZE; ++index) {
        loaded_data[index] = row_ids_data[index_begin[index]];
      }
      __builtin_memcpy_inline(__builtin_assume_aligned(row_ids_output, BATCH_SIZE), loaded_data.data(), BATCH_SIZE);
    };

    // Gathers both RowIDs and values (only double and int64_t, strings are handled separately).
    auto gather_row_ids_and_values = [&](const uint32_t* __restrict index_begin, const RowID* __restrict row_ids_data,
                                         RowID* __restrict row_ids_output, const ColumnType* __restrict values,
                                         const ColumnType* __restrict values_output) {
      auto loaded_data = std::array<RowID, GATHER_SIZE>{};
      auto loaded_values = std::array<ColumnType, GATHER_SIZE>{};

#pragma clang loop vectorize(enable) interleave(enable) unroll(disable)
      for (auto index = size_t{0}; index < GATHER_SIZE; ++index) {
        const auto& rid = index_begin[index];
        loaded_data[index] = row_ids_data[rid];
        loaded_values[index] = values[rid];
      }
      __builtin_memcpy_inline(__builtin_assume_aligned(row_ids_output, BATCH_SIZE), loaded_data.data(), BATCH_SIZE);
      __builtin_memcpy_inline(__builtin_assume_aligned(values_output, BATCH_SIZE), loaded_values.data(), BATCH_SIZE);
    };

    // Gather RowIDs and values in batches of size GATHER_SIZE.
    // By applying auto vectorization vectorized gather instructions should be used.
    for (auto index = size_t{0}; index < aligned_size; index += GATHER_SIZE) {
      auto indices = std::move(gather_indices(reinterpret_cast<uint64_t*>(&sorted_elements[index])));
      const auto offset = index;
      if constexpr (IS_LOSSLESS_COMPRESSION) {  // ColumnType is float or int32_t.
        gather_row_ids(indices.data(), row_ids.data(), output_row_ids.data() + offset);
      } else if (!std::is_same_v<ColumnType, pmr_string> &&
                 !std::is_same_v<ColumnType, int32_t>) {  // ColumnType is double or int64_t.
        gather_row_ids_and_values(indices.data(), row_ids.data(), output_row_ids.data() + offset, values.data(),
                                  output_values.data() + offset);
      } else {  // ColumnType is pmr_string.
        gather_row_ids(indices.data(), row_ids.data(), output_row_ids.data() + offset);
        for (auto offset = size_t{0}; offset < GATHER_SIZE; ++offset) {
          output_values[index + offset] = std::move(values[indices[offset]]);
        }
      }
    }

    // Gather remaining RowIDs and values.
    if (remaining_size > 0) {
      for (auto index = aligned_size; index < num_items; ++index) {
        auto rid = sorted_elements[index].index;
        output_row_ids[index] = row_ids[rid];

        if constexpr (!IS_LOSSLESS_COMPRESSION) {
          output_values[index] = std::move(values[rid]);
        }
      }
    }
  }

  void _gather_row_ids_and_values_according_to_sorted_simd_data(
      std::vector<std::span<SimdElement>>& sorted_elements_per_hash, std::vector<ColumnType>& values,
      RowIDPosList& row_ids, std::vector<simd_sort::simd_vector<ColumnType>>& sorted_values,
      std::vector<simd_sort::simd_vector<RowID>>& sorted_row_ids) {
    auto jobs = std::vector<std::shared_ptr<AbstractTask>>{};

    auto total_size = sorted_elements_per_hash.size();
    sorted_values.resize(total_size);
    sorted_row_ids.resize(total_size);

    for (auto cluster_index = size_t{0}; cluster_index < _cluster_count; ++cluster_index) {
      const auto element_count = sorted_elements_per_hash[cluster_index].size();
      if (element_count > JOB_SPAWN_THRESHOLD) {
        jobs.push_back(std::make_shared<JobTask>([&, this, cluster_index]() {
          _materialized_row_ids_and_values(sorted_elements_per_hash[cluster_index], row_ids,
                                           sorted_row_ids[cluster_index], values, sorted_values[cluster_index]);
        }));
      } else {
        _materialized_row_ids_and_values(sorted_elements_per_hash[cluster_index], row_ids,
                                         sorted_row_ids[cluster_index], values, sorted_values[cluster_index]);
      }
    }

    Hyrise::get().scheduler()->schedule_and_wait_for_tasks(jobs);
  }

 public:
  std::shared_ptr<const Table> _on_execute() override {
    if constexpr (HYRISE_DEBUG) {
      std::cout << "Execute JoinSimdSortMerge: cluster_count: " << _cluster_count << '\n';
    }

    const auto include_null_left = (_mode == JoinMode::Left || _mode == JoinMode::FullOuter);
    const auto include_null_right = (_mode == JoinMode::Right || _mode == JoinMode::FullOuter);

    using enum OperatorSteps;

    _materialize_column_and_transform_to_simd_format<ColumnType, LeftSideMaterialize, LeftSideTransform>(
        _left_input_table, _primary_left_column_id, _materialized_values_left, _row_ids_left, _simd_elements_left,
        _null_rows_left, include_null_left);

    _materialize_column_and_transform_to_simd_format<ColumnType, RightSideMaterialize, RightSideTransform>(
        _right_input_table, _primary_right_column_id, _materialized_values_right, _row_ids_right, _simd_elements_right,
        _null_rows_right, include_null_right);

    _sorted_per_hash_left = std::move(_sort_relation<SortingType, LeftSidePartition, LeftSideSortBuckets>(
        _simd_elements_left, _partition_storage_left, _working_memory_left));

    _sorted_per_hash_right = std::move(_sort_relation<SortingType, RightSidePartition, RightSideSortBuckets>(
        _simd_elements_right, _partition_storage_right, _working_memory_right));

    auto timer = Timer{};

    _gather_row_ids_and_values_according_to_sorted_simd_data(_sorted_per_hash_left, _materialized_values_left,
                                                             _row_ids_left, _sorted_values_left, _sorted_row_ids_left);

    _gather_row_ids_and_values_according_to_sorted_simd_data(_sorted_per_hash_right, _materialized_values_right,
                                                             _row_ids_right, _sorted_values_right,
                                                             _sorted_row_ids_right);

    _perform_join();

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

    _performance.set_step_runtime(FindJoinPartner, timer.lap());

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
