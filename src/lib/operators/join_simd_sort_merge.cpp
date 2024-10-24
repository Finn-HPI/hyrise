#include "join_simd_sort_merge.hpp"

#include <boost/functional/hash.hpp>

#if defined(__x86_64__)
#include "immintrin.h"
#endif

#include <span>

#include "operators/join_simd_sort_merge/column_materializer.hpp"
#include "types.hpp"
#include "utils/assert.hpp"

namespace hyrise {

constexpr auto BUCKET_BITS = uint8_t{8};
constexpr auto BUCKET_MASK = std::size_t{(1u << BUCKET_BITS) - 1};
constexpr auto BUCKET_COUNT = uint32_t{1u << BUCKET_BITS};
constexpr auto CACHE_LINE_SIZE = std::size_t{64};
constexpr auto TUPLES_PER_CACHELINE = CACHE_LINE_SIZE / 8;

using CacheLine = union {
  struct {
    std::array<SimdElement, TUPLES_PER_CACHELINE> values;
  } tuples;

  struct {
    std::array<SimdElement, TUPLES_PER_CACHELINE - 1> values;
    std::size_t output_offset;
  } data;
};

struct Partition {
  SimdElement* data;
  std::size_t size;

  std::span<SimdElement> elements() const {
    return {data, data + size};
  }
};

using BucketData = struct {
  std::size_t count;
  std::size_t cache_aligend_count;
};

using HistogramData = struct {
  std::array<BucketData, BUCKET_COUNT> histogram;
  std::size_t cache_aligned_size;
};

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
struct Data32BitCompression<pmr_string> {
  static inline uint32_t compress(pmr_string& value, const pmr_string& min_value [[maybe_unused]],
                                  const pmr_string& max_value [[maybe_unused]]) {
    auto key = uint32_t{0};
    const auto string_length = value.length();
    for (auto index = std::size_t{0}; index < string_length; index++) {
      key = ((key << 5u) + key) ^ static_cast<uint32_t>(value[index]);
    }
    return key;
  }
};

template <typename T>
void materialize_column_and_transform_to_simd_format(const std::shared_ptr<const Table> table, const ColumnID column_id,
                                                     MaterializedSegmentList<T>& materialized_segments [[maybe_unused]],
                                                     SimdElementList& simd_element_list [[maybe_unused]]) {
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

constexpr std::size_t align_to_cacheline(std::size_t value) {
  return (value + TUPLES_PER_CACHELINE - 1) & ~(TUPLES_PER_CACHELINE - 1);
}

void store_cacheline(auto* destination, auto* source) {
#if defined(__AVX512F__)
  auto* destination_address = reinterpret_cast<__m512i*>(destination);
  auto vec = _mm512_load_si512(reinterpret_cast<__m512i*>(source));
  _mm512_stream_si512(destination_address, vec);
#elif defined(__AVX2__) || defined(__AVX__)
  auto* first_destination_address = reinterpret_cast<__m256i*>(destination);
  auto vec1 = _mm256_load_si256(reinterpret_cast<__m256i*>(source));
  auto* second_destination_address = first_destination_address + 1;
  auto vec2 = _mm256_load_si256(reinterpret_cast<__m256i*>(source) + 1);
  _mm256_stream_si256(first_destination_address, vec1);
  _mm256_stream_si256(second_destination_address, vec2);
#else
  std::memcpy(destination, source, 64);
#endif
}

HistogramData compute_histogram(SimdElementList& simd_elements) {
  auto histogram_data = HistogramData{};
  auto& histogram = histogram_data.histogram;

  for (auto& element : simd_elements) {
    ++histogram[element.key & BUCKET_MASK].count;
  }
  auto cache_aligned_output_size = std::size_t{0};
  for (auto bucket_index = std::size_t{0}; bucket_index < BUCKET_COUNT; ++bucket_index) {
    __builtin_prefetch(&histogram[bucket_index], 1, 0);
    histogram[bucket_index].cache_aligend_count = align_to_cacheline(histogram[bucket_index].count);
    cache_aligned_output_size += histogram[bucket_index].cache_aligend_count;
  }

  histogram_data.cache_aligned_size = cache_aligned_output_size;
  return histogram_data;
}

std::vector<Partition> radix_partition(SimdElementList& simd_elements, HistogramData& histogram_data,
                                       simd_sort::simd_vector<SimdElement>& output) {
  auto partitions = std::vector<Partition>(BUCKET_COUNT);
  auto histogram = histogram_data.histogram;
  output.resize(histogram_data.cache_aligned_size);

  auto* output_start_address = output.data();

  auto buffers = simd_sort::simd_vector<CacheLine>(BUCKET_COUNT);  // Cacheline aligned (64-bit).
  buffers[0].data.output_offset = 0;
  partitions[0].data = output_start_address;
  partitions[0].size = histogram[0].count;
  for (auto bucket_index = std::size_t{1}; bucket_index < BUCKET_COUNT; ++bucket_index) {
    __builtin_prefetch(&buffers[bucket_index], 1, 0);
    buffers[bucket_index].data.output_offset =
        buffers[bucket_index - 1].data.output_offset + histogram[bucket_index - 1].cache_aligend_count;
    partitions[bucket_index].data = output_start_address + buffers[bucket_index].data.output_offset;
    partitions[bucket_index].size = histogram[bucket_index].count;
  }

  for (auto& element : simd_elements) {
    const auto bucket_index = element.key & BUCKET_MASK;
    __builtin_prefetch(&buffers[bucket_index], 1, 0);
    auto& buffer = buffers[bucket_index];
    auto slot = buffer.data.output_offset;
    auto local_offset = slot & (TUPLES_PER_CACHELINE - 1);

    buffer.tuples.values[local_offset] = element;
    if (local_offset == TUPLES_PER_CACHELINE - 1) {
      auto* destination = output_start_address + slot - (TUPLES_PER_CACHELINE - 1);
      auto* source = &buffer;
      store_cacheline(destination, source);
    }
    buffer.data.output_offset = slot + 1;
  }

  for (auto bucket_index = std::size_t{0}; bucket_index < BUCKET_COUNT; ++bucket_index) {
    __builtin_prefetch(&buffers[bucket_index], 1, 0);
    auto& buffer = buffers[bucket_index];
    auto slot = buffer.data.output_offset;
    auto local_offset = slot & (TUPLES_PER_CACHELINE - 1);
    if (local_offset > 0) {
      auto* output_address = output_start_address + slot - local_offset;
      std::memcpy(output_address, &buffer, local_offset * 8);
    }
  }

  return partitions;
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
    auto histogram_data_left = compute_histogram(_simd_elements_left);
    auto partitions_left = radix_partition(_simd_elements_left, histogram_data_left, _partitioned_element_left);

    // materialize_column_and_transform_to_simd_format<T>(_right_input_table, _primary_right_column_id,
    //                                                    _materialized_segments_right, _simd_elements_right);
    // auto histogram_data_right = compute_histogram(_simd_elements_right);
    // auto partitions_right = radix_partition(_simd_elements_right, histogram_data_right, _partitioned_element_right);

    auto partition_index = std::size_t{0};
    for (auto& partition : partitions_left) {
      if (partition.size == 0) {
        ++partition_index;
        continue;
      }
      for (auto& element : partition.elements()) {
        std::cout << partition_index << ": " << element << std::endl;
      }
      ++partition_index;
    }

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
