#include "join_simd_sort_merge.hpp"

#include <boost/functional/hash.hpp>

#include "operators/join_simd_sort_merge/simd_sort.hpp"

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

template <typename T>
using PerHash = std::array<T, BUCKET_COUNT>;

constexpr auto CORE_COUNT = 8;

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
  SimdElement* data_start_address;
  std::size_t original_data_offset;
  std::size_t size;

  template <typename T>
  T* begin() const {
    return reinterpret_cast<T*>(data_start_address);
  }

  template <typename T>
  T* end() const {
    return reinterpret_cast<T*>(data_start_address + size);
  }

  std::span<SimdElement> elements() const {
    return {data_start_address, data_start_address + size};
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

HistogramData compute_histogram(std::span<SimdElement> simd_elements) {
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

std::vector<Partition> radix_partition(std::span<SimdElement> simd_elements, HistogramData& histogram_data,
                                       simd_sort::simd_vector<SimdElement>& output) {
  auto partitions = std::vector<Partition>(BUCKET_COUNT);
  auto histogram = histogram_data.histogram;
  output.resize(histogram_data.cache_aligned_size);

  auto* output_start_address = output.data();

  auto buffers = simd_sort::simd_vector<CacheLine>(BUCKET_COUNT);  // Cacheline aligned (64-bit).
  buffers[0].data.output_offset = 0;
  partitions[0].original_data_offset = 0;
  partitions[0].size = histogram[0].count;
  partitions[0].data_start_address = output_start_address;

  for (auto bucket_index = std::size_t{1}; bucket_index < BUCKET_COUNT; ++bucket_index) {
    // __builtin_prefetch(&buffers[bucket_index], 1, 0);
    buffers[bucket_index].data.output_offset =
        buffers[bucket_index - 1].data.output_offset + histogram[bucket_index - 1].cache_aligend_count;
    auto& partition = partitions[bucket_index];
    partition.original_data_offset = buffers[bucket_index].data.output_offset;
    partition.size = histogram[bucket_index].count;
    partition.data_start_address = output_start_address + buffers[bucket_index].data.output_offset;

    // std::cout << "bucket_index: " << bucket_index << "offset: " << partition.data_offset << std::endl;
  }

  for (auto& element : simd_elements) {
    const auto bucket_index = element.key & BUCKET_MASK;
    // __builtin_prefetch(&buffers[bucket_index], 1, 0);
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
    // __builtin_prefetch(&buffers[bucket_index], 1, 0);
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

constexpr std::size_t choose_count_per_vector() {
  return 4;
}

template <typename SimdSortType>
std::vector<Partition> partition_and_sort_partitions(std::size_t thread_index [[maybe_unused]],
                                                     std::span<SimdElement> elements,
                                                     simd_sort::simd_vector<SimdElement>& partition_output,
                                                     simd_sort::simd_vector<SimdElement>& sort_working_memory
                                                     [[maybe_unused]]) {
  auto histogram_data = compute_histogram(elements);
  auto partitions = radix_partition(elements, histogram_data, partition_output);

  const auto count_per_vector = choose_count_per_vector();
  for (auto& partition : partitions) {
    if (!partition.size) {
      continue;
    }
    auto* begin = partition.begin<SimdSortType>();
    auto* cmp = reinterpret_cast<SimdSortType*>(sort_working_memory.data() + partition.original_data_offset);
    auto* working_mem = reinterpret_cast<SimdSortType*>(sort_working_memory.data() + partition.original_data_offset);

    simd_sort::sort<count_per_vector>(begin, working_mem, partition.size);

    if (working_mem == cmp) {
      partition.data_start_address = reinterpret_cast<SimdElement*>(cmp);
    }
  }

  return partitions;
};

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
    const auto num_items = _simd_elements_left.size();
    const auto base_size_per_thread = num_items / CORE_COUNT;
    const auto remainder = num_items % CORE_COUNT;

    auto thread_partition_outputs = std::array<simd_sort::simd_vector<SimdElement>, CORE_COUNT>{};
    auto thread_sort_outputs = std::array<simd_sort::simd_vector<SimdElement>, CORE_COUNT>{};
    auto thread_partition_bounds = std::array<std::vector<Partition>, CORE_COUNT>{};
    auto thread_inputs = std::array<std::span<SimdElement>, CORE_COUNT>{};

    auto* input_start_address = _simd_elements_left.data();
    for (auto thread_index = std::size_t{0}, offset = std::size_t{0}; thread_index < CORE_COUNT; ++thread_index) {
      auto current_range_size = base_size_per_thread + (thread_index < remainder ? 1 : 0);
      thread_inputs[thread_index] = {input_start_address + offset, input_start_address + offset + current_range_size};
      offset += current_range_size;
    }

    const auto num_threads = CORE_COUNT;
    auto tasks = std::vector<std::shared_ptr<AbstractTask>>{};
    tasks.reserve(num_threads);
    for (auto thread_index = 0u; thread_index < num_threads; ++thread_index) {
      tasks.emplace_back(std::make_shared<JobTask>(
          [thread_index, &thread_inputs, &thread_partition_outputs, &thread_sort_outputs, &thread_partition_bounds]() {
            thread_partition_bounds[thread_index] = partition_and_sort_partitions<int64_t>(
                thread_index, thread_inputs[thread_index], thread_partition_outputs[thread_index],
                thread_sort_outputs[thread_index]);
          }));
    }
    Hyrise::get().scheduler()->schedule_and_wait_for_tasks(tasks);

    for (auto thread_index = 0u; thread_index < num_threads; ++thread_index) {
      std::cout << "========================================================" << std::endl;
      std::cout << "thread: " << thread_index << std::endl;
      auto& partition_bounds = thread_partition_bounds[thread_index];
      for (std::size_t i = 0; i < BUCKET_COUNT; ++i) {
        if (partition_bounds[i].size == 0u) {
          continue;
        }
        std::cout << "p: " << i << std::endl;
        for (auto& element : partition_bounds[i].elements()) {
          std::cout << element << std::endl;
        }
      }
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
