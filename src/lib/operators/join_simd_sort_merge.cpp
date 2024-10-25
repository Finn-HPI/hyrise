#include "join_simd_sort_merge.hpp"

#include <boost/functional/hash.hpp>

#include "operators/join_simd_sort_merge/radix_partitioning.hpp"
#include "operators/join_simd_sort_merge/simd_sort.hpp"

#if defined(__x86_64__)
#include "immintrin.h"
#endif

#include <span>

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

using Partition = radix_partition::Partition;
using RadixPartition = radix_partition::RadixPartition;

bool JoinSimdSortMerge::supports(const JoinConfiguration config) {
  return config.predicate_condition == PredicateCondition::Equals && config.left_data_type == config.right_data_type &&
         config.join_mode == JoinMode::Inner;
}

JoinSimdSortMerge::JoinSimdSortMerge(const std::shared_ptr<const AbstractOperator>& left,
                                     const std::shared_ptr<const AbstractOperator>& right, const JoinMode mode,
                                     const OperatorJoinPredicate& primary_predicate,
                                     const std::vector<OperatorJoinPredicate>& secondary_predicates)
    : AbstractJoinOperator(OperatorType::JoinSortMerge, left, right, mode, primary_predicate, secondary_predicates) {}

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
        _primary_predicate.column_ids.second, _primary_predicate.predicate_condition, _mode, _secondary_predicates);
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

template <typename T>
void materialize_column_and_transform_to_simd_format(const std::shared_ptr<const Table> table, const ColumnID column_id,
                                                     MaterializedSegmentList<T>& materialized_segments,
                                                     SimdElementList& simd_element_list) {
  auto left_column_materializer = SMJColumnMaterializer<T>(JoinSimdSortMerge::JOB_SPAWN_THRESHOLD);
  auto [_materialized_segments, min_value, max_value] = left_column_materializer.materialize(table, column_id);

  materialized_segments = _materialized_segments;

  simd_element_list.reserve(materialized_segments.size());

  auto index = size_t{0};
  for (auto& segment : materialized_segments) {
    for (auto& materialized_value : segment) {
      DebugAssert(index <= std::numeric_limits<uint32_t>::max(), "Index has to fit into 32 bits. ");
      const auto sorting_key = Data32BitCompression<T>::compress(materialized_value.value, min_value, max_value);
      simd_element_list.emplace_back(sorting_key, static_cast<uint32_t>(index));
      ++index;
    }
  }
}

constexpr std::size_t choose_count_per_vector() {
#if defined(__AVX512F__)
  return 8;
#else
  return 4;
#endif
}

template <typename SimdSortType>
RadixPartition construct_and_sort_partitions(std::size_t thread_index [[maybe_unused]],
                                             std::span<SimdElement> elements) {
  auto radix_partitioner = RadixPartition(elements);
  radix_partitioner.execute();

  const auto count_per_vector = choose_count_per_vector();
  for (auto& partition : radix_partitioner.partitions()) {
    if (!partition.size) {
      continue;
    }
    auto* input_pointer = partition.begin<SimdSortType>();
    // auto* output_pointer = reinterpret_cast<SimdSortType*>(sort_working_memory.data() + partition.original_data_offset);
    auto* output_pointer = radix_partitioner.get_working_memory<SimdSortType>(partition);

    simd_sort::sort<count_per_vector>(input_pointer, output_pointer, partition.size);

    partition.data = reinterpret_cast<SimdElement*>(output_pointer);
  }

  return radix_partitioner;
};

template <typename SortingType>
void sort_relation(SimdElementList& simd_elements) {
  const auto num_items = simd_elements.size();
  const auto base_size_per_thread = num_items / THREAD_COUNT;
  const auto remainder = num_items % THREAD_COUNT;

  auto thread_partitions = std::array<RadixPartition, THREAD_COUNT>{};
  auto thread_inputs = std::array<std::span<SimdElement>, THREAD_COUNT>{};

  // Split simd_elements into THREAD_COUNT many sections.
  auto* input_start_address = simd_elements.data();
  for (auto thread_index = std::size_t{0}, offset = std::size_t{0}; thread_index < THREAD_COUNT; ++thread_index) {
    auto current_range_size = base_size_per_thread + (thread_index < remainder ? 1 : 0);
    thread_inputs[thread_index] = {input_start_address + offset, input_start_address + offset + current_range_size};
    offset += current_range_size;
  }

  // Each section is assigned to a thread, which when radix partitions the input and sorts each partition.
  const auto num_threads = THREAD_COUNT;
  spawn_and_wait_per_thread(thread_inputs, [&thread_partitions](auto& elements, std::size_t thread_index) {
    thread_partitions[thread_index] = construct_and_sort_partitions<SortingType>(thread_index, elements);
  });

  // DebugAssert that each partition is sorted correctly.
  for (auto thread_index = 0u; thread_index < num_threads; ++thread_index) {
    const auto& partitions = thread_partitions[thread_index].partitions();
    for (auto partition : partitions) {
      DebugAssert(std::is_sorted(partition.begin<SortingType>(), partition.end<SortingType>()),
                  "Partition was not sorted");
    }
  }
  for (auto thread_index = 0u; thread_index < num_threads; ++thread_index) {
    std::cout << "========================================================" << std::endl;
    std::cout << "thread: " << thread_index << std::endl;
    auto& radix_partitions = thread_partitions[thread_index].partitions();
    for (std::size_t i = 0; i < radix_partition::PARTITION_SIZE; ++i) {
      if (radix_partitions[i].size == 0u) {
        continue;
      }
      std::cout << "p: " << i << std::endl;
      for (auto& element : radix_partitions[i].elements()) {
        std::cout << element << std::endl;
      }
    }
  }
}

template <typename T>
class JoinSimdSortMerge::JoinSimdSortMergeImpl : public AbstractReadOnlyOperatorImpl {
 public:
  JoinSimdSortMergeImpl(JoinSimdSortMerge& sort_merge_join, const std::shared_ptr<const Table>& left_input_table,
                        const std::shared_ptr<const Table>& right_input_table, ColumnID left_column_id,
                        ColumnID right_column_id, const PredicateCondition op, JoinMode mode,
                        const std::vector<OperatorJoinPredicate>& secondary_join_predicates)
      : _sort_merge_join{sort_merge_join},
        _left_input_table{left_input_table},
        _right_input_table{right_input_table},
        _primary_left_column_id{left_column_id},
        _primary_right_column_id{right_column_id},
        _primary_predicate_condition{op},
        _mode{mode},
        _secondary_join_predicates{secondary_join_predicates} {}

 protected:
  // NOLINTBEGIN(cppcoreguidelines-avoid-const-or-ref-data-members)
  JoinSimdSortMerge& _sort_merge_join;
  const std::shared_ptr<const Table> _left_input_table;
  const std::shared_ptr<const Table> _right_input_table;
  const ColumnID _primary_left_column_id;
  const ColumnID _primary_right_column_id;

  const PredicateCondition _primary_predicate_condition;
  const JoinMode _mode;

  MaterializedSegmentList<T> _materialized_segments_left;
  MaterializedSegmentList<T> _materialized_segments_right;
  SimdElementList _simd_elements_left;
  SimdElementList _simd_elements_right;

  SimdElementList _partitioned_element_left;
  SimdElementList _partitioned_element_right;

  const std::vector<OperatorJoinPredicate>& _secondary_join_predicates;
  // NOLINTEND(cppcoreguidelines-avoid-const-or-ref-data-members)

 public:
  std::shared_ptr<const Table> _on_execute() override {
    materialize_column_and_transform_to_simd_format<T>(_left_input_table, _primary_left_column_id,
                                                       _materialized_segments_left, _simd_elements_left);

    materialize_column_and_transform_to_simd_format<T>(_right_input_table, _primary_right_column_id,
                                                       _materialized_segments_right, _simd_elements_right);

    using SortingType = uint64_t;
    sort_relation<SortingType>(_simd_elements_left);
    sort_relation<SortingType>(_simd_elements_right);

    return nullptr;
  }
};

// template <>
// class JoinSimdSortMerge::JoinSimdSortMergeImpl<pmr_string> : public AbstractReadOnlyOperatorImpl {
//  public:
//   JoinSimdSortMergeImpl(JoinSimdSortMerge& sort_merge_join, const std::shared_ptr<const Table>& left_input_table,
//                         const std::shared_ptr<const Table>& right_input_table, ColumnID left_column_id,
//                         ColumnID right_column_id, const PredicateCondition op, JoinMode mode,
//                         const std::vector<OperatorJoinPredicate>& secondary_join_predicates)
//       : _sort_merge_join{sort_merge_join},
//         _left_input_table{left_input_table},
//         _right_input_table{right_input_table},
//         _primary_left_column_id{left_column_id},
//         _primary_right_column_id{right_column_id},
//         _primary_predicate_condition{op},
//         _mode{mode},
//         _secondary_join_predicates{secondary_join_predicates} {}
//
//  protected:
//   // NOLINTBEGIN(cppcoreguidelines-avoid-const-or-ref-data-members)
//   JoinSimdSortMerge& _sort_merge_join;
//   const std::shared_ptr<const Table> _left_input_table;
//   const std::shared_ptr<const Table> _right_input_table;
//   const ColumnID _primary_left_column_id;
//   const ColumnID _primary_right_column_id;
//
//   const PredicateCondition _primary_predicate_condition;
//   const JoinMode _mode;
//
//   const std::vector<OperatorJoinPredicate>& _secondary_join_predicates;
//   // NOLINTEND(cppcoreguidelines-avoid-const-or-ref-data-members)
//
//  public:
//   std::shared_ptr<const Table> _on_execute() override {
//     std::cout << "execute with pmr_string :)" << std::endl;
//     return nullptr;
//   }
// };

}  // namespace hyrise
