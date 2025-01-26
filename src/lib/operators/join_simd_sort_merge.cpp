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
#include "operators/join_simd_sort_merge/smj_column_materializer.hpp"
// #include "operators/join_sort_merge/column_materializer.hpp"
#include "operators/multi_predicate_join/multi_predicate_join_evaluator.hpp"
#include "utils/timer.hpp"

#if defined(__x86_64__)
#include "immintrin.h"
#endif

#include <algorithm>
#include <fstream>
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
      return Data32BitCompression<uint32_t>::compress(static_cast<uint32_t>(value - min_value), 0,
                                                      static_cast<uint32_t>(max_value - min_value));
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
        _num_cpus{Hyrise::get().topology.num_cpus()},
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

  size_t _num_cpus;
  size_t _cluster_count{1};

  std::vector<simd_sort::simd_vector<ColumnType>> _sorted_values_left;
  std::vector<simd_sort::simd_vector<ColumnType>> _sorted_values_right;

  uint32_t _chunk_offset_bits_left{};
  uint32_t _chunk_offset_bits_right{};

  // Contains the null value row ids if a join column is an outer join column.
  RowIDPosList _null_rows_left;
  RowIDPosList _null_rows_right;

  SimdElementList _simd_elements_left;
  SimdElementList _simd_elements_right;

  std::vector<SimdElementList> _sorted_per_hash_left;
  std::vector<SimdElementList> _sorted_per_hash_right;

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

  static uint32_t __attribute__((always_inline)) _pack_row_id(RowID& row_id, uint32_t chunk_offset_bits) {
    return (row_id.chunk_id << chunk_offset_bits) | row_id.chunk_offset;
  }

  static RowID _unpack_row_id(uint32_t packed_row_id, uint32_t chunk_offset_bits) {
    uint32_t chunk_offset_mask = (1u << chunk_offset_bits) - 1;
    auto chunk_offset = static_cast<ChunkOffset>(packed_row_id & chunk_offset_mask);
    auto chunk_id = static_cast<ChunkID>(packed_row_id >> chunk_offset_bits);
    return {chunk_id, chunk_offset};
  }

  struct PotentialMatchRange {
    PotentialMatchRange(std::size_t init_start_index, std::size_t init_end_index, std::span<SimdElement> init_elements,
                        std::span<ColumnType> init_values, uint32_t init_chunk_offset_bits)
        : start_index(init_start_index),
          end_index(init_end_index),
          elements(init_elements.data() + init_start_index, init_elements.data() + init_end_index),
          values(init_values.data() + init_start_index, init_values.data() + init_end_index),
          chunk_offset_bits{init_chunk_offset_bits} {}

    std::size_t start_index;
    std::size_t end_index;
    std::span<SimdElement> elements;
    std::span<ColumnType> values;
    uint32_t chunk_offset_bits;

   public:
    // Executes the given action for every row id of the table in this range.
    void for_every_row_id(auto&& action) const {
      const auto num_items = elements.size();
      for (auto index = size_t{0}; index < num_items; ++index) {
        DebugAssert(values.size() > index || IS_LOSSLESS_COMPRESSION, "Values has broken size.");
        const auto& row_id = _unpack_row_id(elements[index].index, chunk_offset_bits);
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
                      std::span<ColumnType> right_values) {
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
          PotentialMatchRange(left_run_start, left_run_end, left_elements, left_values, _chunk_offset_bits_left);
      const auto right_range =
          PotentialMatchRange(right_run_start, right_run_end, right_elements, right_values, _chunk_offset_bits_right);

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
        PotentialMatchRange(left_run_start, left_size, left_elements, left_values, _chunk_offset_bits_left);
    const auto right_remainder =
        PotentialMatchRange(right_run_start, right_size, right_elements, right_values, _chunk_offset_bits_right);
    if (left_run_start < left_size) {
      _find_matches_in_ranges(left_remainder, right_remainder, CompareResult::Less, multi_predicate_join_evaluator,
                              bucket_index);
    } else if (right_run_start < right_size) {
      _find_matches_in_ranges(left_remainder, right_remainder, CompareResult::Greater, multi_predicate_join_evaluator,
                              bucket_index);
    }
  }

  // void _join(SimdElementList& left, SimdElementList& right) {
  //   // NOLINTBEGIN
  //   size_t i = 0;
  //   size_t j = 0;
  //
  //   while (i < left.size() && j < right.size()) {
  //     if (left[i].key == right[j].key) {
  //       // Match found, iterate through duplicates in Column2
  //       size_t start_j = j;
  //       while (start_j < right.size() && right[start_j].key == left[i].key) {
  //         auto row_id_left = _unpack_row_id_left(left[i].index);
  //         auto row_id_right = _unpack_row_id_right(right[start_j].index);
  //         _emit_combination(0, row_id_left, row_id_right);
  //
  //         start_j++;
  //       }
  //       i++;  // Move to the next element in Column1
  //     } else if (left[i].key < right[j].key) {
  //       i++;  // Move the pointer in column1
  //     } else {
  //       j++;  // Move the pointer in column2
  //     }
  //   }
  //   // NOLINTEND
  // }

  void _perform_join() {
    auto jobs = std::vector<std::shared_ptr<AbstractTask>>{};
    for (auto cluster_index = size_t{0}; cluster_index < _cluster_count; ++cluster_index) {
      const auto merge_row_count =
          _sorted_per_hash_left[cluster_index].size() + _sorted_per_hash_right[cluster_index].size();
      if (merge_row_count > JOB_SPAWN_THRESHOLD) {
        jobs.push_back(std::make_shared<JobTask>([cluster_index, this]() {
          _join_per_hash(cluster_index, _sorted_per_hash_left[cluster_index], _sorted_per_hash_right[cluster_index],
                         _sorted_values_left[cluster_index], _sorted_values_right[cluster_index]);
        }));
      } else {
        _join_per_hash(cluster_index, _sorted_per_hash_left[cluster_index], _sorted_per_hash_right[cluster_index],
                       _sorted_values_left[cluster_index], _sorted_values_right[cluster_index]);
      }
    }
    Hyrise::get().scheduler()->schedule_and_wait_for_tasks(jobs);
  }

  std::vector<std::span<SimdElement>> _split_vector_into_spans(SimdElementList& vec, size_t parts) {
    size_t total_size = vec.size();
    size_t chunk_size = total_size / parts;
    size_t remainder = total_size % parts;

    std::vector<std::span<SimdElement>> chunks;
    size_t start = 0;

    for (size_t i = 0; i < parts; ++i) {
      size_t end = start + chunk_size + (i < remainder ? 1 : 0);  // Distribute the remainder.
      chunks.emplace_back(vec.data() + start, end - start);
      start = end;
    }
    return chunks;
  }

  template <typename SortingType, OperatorSteps partition_step, OperatorSteps sort_buckets_step>
  std::vector<SimdElementList> _sort_relation(SimdElementList& simd_elements) {
    auto timer = Timer{};
    [[maybe_unused]] constexpr auto MIN_PARTITION_ELEMENTS = 1048576;

    // const auto chunk_count = std::max(size_t{1}, static_cast<size_t>(simd_elements.size() / MIN_PARTITION_ELEMENTS));
    const auto chunk_count = 1;
    auto chunks = std::move(_split_vector_into_spans(simd_elements, chunk_count));

    auto partition_storage = std::vector<SimdElementList>(chunk_count);
    auto working_memory = std::vector<SimdElementList>(chunk_count);

    auto sort_bucket = [](size_t bucket_index, RadixPartition<ColumnType>& radix_partition,
                          SimdElementList& chunk_working_memory) {
      auto& bucket = radix_partition.bucket(bucket_index);
      if (bucket.empty()) {
        return;
      }
      const auto count_per_vector = choose_count_per_vector();
      auto* input_pointer = bucket.template begin<SortingType>();
      auto* output_pointer =
          radix_partition.template get_working_memory<SortingType>(bucket_index, chunk_working_memory);

      DebugAssert((simd_sort::is_simd_aligned<SortingType, 64>(input_pointer)), "Input not cache aligned.");
      DebugAssert((simd_sort::is_simd_aligned<SortingType, 64>(output_pointer)), "Output not cache aligned.");

      simd_sort::sort<count_per_vector, SortingType>(input_pointer, output_pointer, bucket.size);
      bucket.data = reinterpret_cast<SimdElement*>(output_pointer);
    };

    auto chunk_partitions = std::vector<RadixPartition<ColumnType>>{};
    chunk_partitions.reserve(chunk_count);

    auto partition_and_sort_chunk = [&](size_t chunk_index) {
      // First, we partition the chunk
      auto& chunk_working_memory = working_memory[chunk_index];
      auto& radix_partition = chunk_partitions[chunk_index];
      radix_partition.execute(partition_storage[chunk_index], chunk_working_memory);

      for (auto bucket_index = size_t{0}; bucket_index < _cluster_count; ++bucket_index) {
        sort_bucket(bucket_index, radix_partition, chunk_working_memory);
      }
    };

    auto jobs = std::vector<std::shared_ptr<AbstractTask>>{};
    for (auto chunk_index = size_t{0}; chunk_index < chunk_count; ++chunk_index) {
      auto& chunk = chunks[chunk_index];
      chunk_partitions.emplace_back(chunk, _cluster_count);
      jobs.push_back(std::make_shared<JobTask>([&partition_and_sort_chunk, chunk_index]() {
        partition_and_sort_chunk(chunk_index);
      }));
    }
    Hyrise::get().scheduler()->schedule_and_wait_for_tasks(jobs);

    _performance.set_step_runtime(partition_step, timer.lap());

    // After each chunk has partitioned and sorted all buckets, we use multiway merging to merge
    // all buckets of the same id of all chunks using multiway merging.

    auto sorted_clusters = std::vector<SimdElementList>(_cluster_count);

    auto multiway_merge_buckets = [&](size_t bucket_index) {
      auto sorted_buckets = std::vector<Bucket*>{};
      sorted_buckets.reserve(chunk_count);
      for (auto& partition : chunk_partitions) {
        if (partition.bucket(bucket_index).empty()) {
          continue;
        }
        sorted_buckets.push_back(&partition.bucket(bucket_index));
      }
      auto multiway_merger = multiway_merging::MultiwayMerger<choose_count_per_vector(), SortingType>(sorted_buckets);
      multiway_merger.merge(sorted_clusters[bucket_index]);
    };

    jobs.clear();
    for (auto bucket_index = size_t{0}; bucket_index < _cluster_count; ++bucket_index) {
      jobs.push_back(std::make_shared<JobTask>([&multiway_merge_buckets, bucket_index] {
        multiway_merge_buckets(bucket_index);
      }));
    }
    Hyrise::get().scheduler()->schedule_and_wait_for_tasks(jobs);

    return sorted_clusters;
  }

  template <typename T, JoinSimdSortMerge::OperatorSteps materialize_step>
  std::pair<T, T> _materialize_column(const std::shared_ptr<const Table> table, const ColumnID column_id,
                                      MaterializedSegmentList<T>& materialized_segment_list, RowIDPosList& null_values,
                                      const bool materialize_null, uint32_t& chunk_offset_bits) {
    auto timer = Timer{};
    auto left_column_materializer = SMJColumnMaterializer<T>(materialize_null);
    auto [materialized_segments, null_rows, chunk_count, max_chunk_size, min, max] =
        std::move(left_column_materializer.materialize(table, column_id));

    auto chunk_id_bits = static_cast<uint32_t>(std::ceil(std::log2(static_cast<double>(chunk_count))));
    chunk_offset_bits = static_cast<uint32_t>(std::ceil(std::log2(static_cast<double>(max_chunk_size))));

    // std::cout << chunk_count << ": " << chunk_id_bits << ", " << max_chunk_size << ": " << chunk_offset_bits
    //           << std::endl;

    Assert(chunk_id_bits + chunk_offset_bits <= 32, "RowIDs can't be compressed to 32-bits.");

    null_values = std::move(null_rows);
    materialized_segment_list = std::move(materialized_segments);
    _performance.set_step_runtime(materialize_step, timer.lap());

    return {min, max};
  }

  //
  // uint32_t __attribute__((always_inline)) _pack_row_id_right(RowID& row_id) {
  //   return (row_id.chunk_id << _chunk_offset_bits_right) | row_id.chunk_offset;
  // }
  //
  // RowID _unpack_row_id_right(uint32_t packed_row_id) {
  //   uint32_t chunk_offset_mask = (1u << _chunk_offset_bits_right) - 1;
  //   auto chunk_offset = static_cast<ChunkOffset>(packed_row_id & chunk_offset_mask);
  //   auto chunk_id = static_cast<ChunkID>(packed_row_id >> _chunk_offset_bits_right);
  //   return {chunk_id, chunk_offset};
  // }

  // static RowID _unpack_row_id(uint32_t packed_row_id) {
  //   ChunkID chunk_id = static_cast<ChunkID>((packed_row_id >> 16u) & 0xFFFFu);     // Extract the 16 MSBs
  //   ChunkOffset chunk_offset = static_cast<ChunkOffset>(packed_row_id & 0xFFFFu);  // Extract the 16 LSBs
  //   return {chunk_id, chunk_offset};
  // }

  template <typename T, JoinSimdSortMerge::OperatorSteps transform_step>
  void _transform_to_simd_format(MaterializedSegmentList<T>& materialized_segments, SimdElementList& simd_element_list,
                                 auto&& pack_row_id, T min_value, T max_value) {
    auto timer = Timer{};

    auto total_size = std::accumulate(materialized_segments.begin(), materialized_segments.end(), size_t{0},
                                      [](size_t sum, auto& segment) {
                                        return std::move(sum) + segment.size();
                                      });

    // std::cout << "total_size: " << total_size << std::endl;

    DebugAssert(total_size <= std::numeric_limits<uint32_t>::max(), "Index has to fit into 32 bits. ");
    simd_element_list.resize(total_size);

    auto transform_segment = [&](const size_t start_index, std::span<MaterializedValue<T>> segment) {
      auto index = start_index;
      const auto segment_size = segment.size();
      for (auto segment_index = size_t{0}; segment_index < segment_size; ++segment_index) {
        auto& materialized_value = segment[segment_index];
        auto& simd_element = simd_element_list[index + segment_index];
        const auto sorting_key = Data32BitCompression<T>::compress(materialized_value.value, min_value, max_value);
        simd_element = SimdElement{pack_row_id(materialized_value.row_id), sorting_key};
      }
    };

    auto jobs = std::vector<std::shared_ptr<AbstractTask>>{};

    auto index = size_t{0};
    for (auto& segment : materialized_segments) {
      if (segment.size() > JOB_SPAWN_THRESHOLD) {
        jobs.push_back(std::make_shared<JobTask>([&, index]() {
          transform_segment(index, segment);
        }));
      } else {
        transform_segment(index, segment);
      }
      index += segment.size();
    }
    Hyrise::get().scheduler()->schedule_and_wait_for_tasks(jobs);

    _performance.set_step_runtime(transform_step, timer.lap());
  }

  void _materialize_values(const std::span<SimdElement> sorted_elements,
                           MaterializedSegmentList<ColumnType>& materialized_segments,
                           simd_sort::simd_vector<ColumnType>& output_values, auto&& unpack_row_id) {
    if constexpr (IS_LOSSLESS_COMPRESSION) {
      return;
    }

    const auto num_elements = sorted_elements.size();
    output_values.resize(num_elements);

    for (auto i = size_t{0}; i < num_elements; ++i) {
      auto row_id = unpack_row_id(sorted_elements[i].index);
      output_values[i] = materialized_segments[row_id.chunk_id][row_id.chunk_offset].value;
    }
  }

  void _gather_values_according_to_sorted_simd_data(std::vector<SimdElementList>& sorted_elements_per_hash,
                                                    MaterializedSegmentList<ColumnType>& materialized_segments,
                                                    std::vector<simd_sort::simd_vector<ColumnType>>& sorted_values,
                                                    auto&& unpack_row_id) {
    auto total_size = sorted_elements_per_hash.size();
    sorted_values.resize(total_size);

    if constexpr (IS_LOSSLESS_COMPRESSION) {
      return;
    }

    auto jobs = std::vector<std::shared_ptr<AbstractTask>>{};

    for (auto cluster_index = size_t{0}; cluster_index < _cluster_count; ++cluster_index) {
      const auto element_count = sorted_elements_per_hash[cluster_index].size();
      if (element_count > JOB_SPAWN_THRESHOLD) {
        jobs.push_back(std::make_shared<JobTask>([&, this, cluster_index]() {
          _materialize_values(sorted_elements_per_hash[cluster_index], materialized_segments,
                              sorted_values[cluster_index], unpack_row_id);
        }));
      } else {
        _materialize_values(sorted_elements_per_hash[cluster_index], materialized_segments,
                            sorted_values[cluster_index], unpack_row_id);
      }
    }

    Hyrise::get().scheduler()->schedule_and_wait_for_tasks(jobs);
  }

  std::shared_ptr<const Table> _on_execute() override {
    if constexpr (HYRISE_DEBUG) {
      std::cout << "Execute JoinSimdSortMerge: L2-Cache = " << L2_SIZE << '\n';
      // std::cout << "float: " << std::is_same_v<ColumnType, float> << " int32: "
      //           << std::is_same_v<ColumnType, int32_t> << '\n';
      // std::cout << "type size: " << sizeof(ColumnType) << "mode: " << _mode << std::endl;
      // std::cout << "secondary_join_predicates: " << _secondary_join_predicates.size() << std::endl;
      // std::cout << _left_input_table->row_count() << " " << _right_input_table->row_count() << std::endl;
    }

    Assert(_left_input_table->chunk_count() <= 65535, "Left chunk_count to big");
    Assert(_right_input_table->chunk_count() <= 65535, "Right chunk_count to big");

    const auto include_null_left = (_mode == JoinMode::Left || _mode == JoinMode::FullOuter);
    const auto include_null_right = (_mode == JoinMode::Right || _mode == JoinMode::FullOuter);

    using enum OperatorSteps;

    auto materialized_segments_left = MaterializedSegmentList<ColumnType>{};
    auto materialized_segments_right = MaterializedSegmentList<ColumnType>{};

    const auto [min_left, max_left] = _materialize_column<ColumnType, LeftSideMaterialize>(
        _left_input_table, _primary_left_column_id, materialized_segments_left, _null_rows_left, include_null_left,
        _chunk_offset_bits_left);

    const auto [min_right, max_right] = _materialize_column<ColumnType, RightSideMaterialize>(
        _right_input_table, _primary_right_column_id, materialized_segments_right, _null_rows_right, include_null_right,
        _chunk_offset_bits_right);

    const auto min_value = std::min(min_left, min_right);
    const auto max_value = std::max(max_left, max_right);

    auto pack_left_row_id = [this](RowID& row_id) {
      return _pack_row_id(row_id, _chunk_offset_bits_left);
    };
    auto pack_right_row_id = [this](RowID& row_id) {
      return _pack_row_id(row_id, _chunk_offset_bits_right);
    };
    auto unpack_left_row_id = [this](size_t packed) {
      return _unpack_row_id(packed, _chunk_offset_bits_left);
    };
    auto unpack_right_row_id = [this](size_t packed) {
      return _unpack_row_id(packed, _chunk_offset_bits_right);
    };

    _transform_to_simd_format<ColumnType, LeftSideTransform>(materialized_segments_left, _simd_elements_left,
                                                             pack_left_row_id, min_value, max_value);

    _transform_to_simd_format<ColumnType, RightSideTransform>(materialized_segments_right, _simd_elements_right,
                                                              pack_right_row_id, min_value, max_value);

    _sorted_per_hash_left =
        std::move(_sort_relation<SortingType, LeftSidePartition, LeftSideSortBuckets>(_simd_elements_left));

    _sorted_per_hash_right =
        std::move(_sort_relation<SortingType, RightSidePartition, RightSideSortBuckets>(_simd_elements_right));

    auto timer = Timer{};

    _gather_values_according_to_sorted_simd_data(_sorted_per_hash_left, materialized_segments_left, _sorted_values_left,
                                                 unpack_left_row_id);
    _gather_values_according_to_sorted_simd_data(_sorted_per_hash_right, materialized_segments_right,
                                                 _sorted_values_right, unpack_right_row_id);
    _performance.set_step_runtime(GatherRowIds, timer.lap());

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
    constexpr auto ALLOW_PARTITION_MERGE = true;
    auto output_chunks =
        write_output_chunks(_output_pos_lists_left, _output_pos_lists_right, _left_input_table, _right_input_table,
                            create_left_side_pos_lists_by_segment, create_right_side_pos_lists_by_segment,
                            OutputColumnOrder::LeftFirstRightSecond, ALLOW_PARTITION_MERGE);

    // const ColumnID left_join_column = _sort_merge_join._primary_predicate.column_ids.first;
    // const ColumnID right_join_column = static_cast<ColumnID>(_sort_merge_join.left_input_table()->column_count() +
    //                                                          _sort_merge_join._primary_predicate.column_ids.second);

    for (auto& chunk : output_chunks) {
      if (_sort_merge_join._primary_predicate.predicate_condition == PredicateCondition::Equals &&
          _mode == JoinMode::Inner) {
        chunk->set_immutable();
      }
    }

    _performance.set_step_runtime(OperatorSteps::OutputWriting, timer.lap());

    auto result_table = _sort_merge_join._build_output_table(std::move(output_chunks));

    // if (_mode != JoinMode::Left && _mode != JoinMode::Right && _mode != JoinMode::FullOuter &&
    //     _sort_merge_join._primary_predicate.predicate_condition == PredicateCondition::Equals) {
    //   // Table clustering is not defined for columns storing NULL values. Additionally, clustering is not given for
    //   // non-equal predicates.
    //   result_table->set_value_clustered_by({left_join_column, right_join_column});
    // }
    return result_table;
  }
};

}  // namespace hyrise
