#pragma once

#include <algorithm>
#include <span>
#include <utility>
#include <vector>

#include "operators/join_simd_sort_merge/radix_partitioning.hpp"
#include "operators/join_simd_sort_merge/simd_utils.hpp"
#include "util.hpp"

namespace hyrise::radix_partition {

template <typename T>
using cache_aligned_vector = simd_sort::simd_vector<T>;

template <typename ColumnType>
struct RadixPartitionBalkesen {
 public:
  explicit RadixPartitionBalkesen(Relation* input, Relation* tmp_output, const std::span<Relation> output_buckets,
                                  size_t cluster_count)
      : _partition_size(cluster_count),
        _bitshift_count{32u - simd_sort::log2_builtin(cluster_count)},
        _radix_mask{(1u << simd_sort::log2_builtin(cluster_count)) - 1},
        _has_data(true),
        _input{input},
        _tmp_output{tmp_output},
        _output_buckets{output_buckets} {}

  RadixPartitionBalkesen() = default;

  // Needed for benchmark.
  std::tuple<size_t, size_t, size_t> performance_info;

 protected:
  using CacheLine = union {
    struct {
      std::array<SimdElement, BUFFER_SIZE> values;
    } tuples;

    struct {
      std::array<SimdElement, BUFFER_SIZE - 1> values;
      std::size_t output_offset;
    } data;
  };

  struct HistogramData {
    explicit HistogramData(size_t partition_size) : histogram(partition_size), cache_aligned_counts(partition_size) {}

    std::vector<size_t> histogram;
    std::vector<size_t> cache_aligned_counts;
    std::size_t cache_aligned_size{};
  };

  size_t _partition_size{};
  size_t _bitshift_count{};
  uint32_t _radix_mask{};
  bool _has_data = false;
  bool _executed = false;

  Relation* _input{};
  Relation* _tmp_output{};
  std::span<Relation> _output_buckets;

  size_t _bucket_index(uint32_t key) {
    // return key >> _bitshift_count; // MSB
    return key & _radix_mask;  // LSB
  }

  HistogramData _compute_histogram() {
    auto histogram_data = HistogramData(_partition_size);
    auto& histogram = histogram_data.histogram;
    auto& cache_aligned_counts = histogram_data.cache_aligned_counts;

    auto* elements = _input->tuples;
    const auto num_elements = _input->num_tuples;

    for (auto index = size_t{0}; index < num_elements; ++index) {
      auto& element = *(elements + index);
      const auto bucket_index = _bucket_index(element.key);
      __builtin_prefetch(histogram.data() + bucket_index, 1, 3);
      ++histogram[bucket_index];
    }

    auto cache_aligned_output_size = std::size_t{0};
    for (auto bucket_index = std::size_t{0}; bucket_index < _partition_size; ++bucket_index) {
      cache_aligned_counts[bucket_index] = align_to_cacheline(histogram[bucket_index]);
      cache_aligned_output_size += cache_aligned_counts[bucket_index];
    }

    histogram_data.cache_aligned_size = cache_aligned_output_size;
    return histogram_data;
  }

  void __attribute__((always_inline)) _store_cacheline(auto* destination, auto* source) {
    auto nontemporal_store_vec = []<typename VecType>(auto* src, auto* dest) {
      auto cache_line_vec = __builtin_nontemporal_load(src);
      __builtin_nontemporal_store(cache_line_vec, dest);
    };
#if defined(__AVX512F__)
    using Vec = simd_sort::Vec<64, int64_t>;  // 512-bit Vector.
    nontemporal_store_vec.template operator()<Vec>(reinterpret_cast<Vec*>(source), reinterpret_cast<Vec*>(destination));
#elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__)
    using Vec = simd_sort::Vec<128, int64_t>;  // 1024-bit Vector.
    nontemporal_store_vec.template operator()<Vec>(reinterpret_cast<Vec*>(source), reinterpret_cast<Vec*>(destination));
#else
    using Vec = simd_sort::Vec<32, int64_t>;  // 256-bit Vector.
    nontemporal_store_vec.template operator()<Vec>(reinterpret_cast<Vec*>(source), reinterpret_cast<Vec*>(destination));
    nontemporal_store_vec.template operator()<Vec>(reinterpret_cast<Vec*>(source) + 1,
                                                   reinterpret_cast<Vec*>(destination) + 1);
#endif
  }

 public:
  void execute() {
    DebugAssert(_has_data, "No input data to partition.");
    DebugAssert(!_executed, "RadixPartition execute can only be called once.");

    // NOLINTNEXTLINE
    size_t time_histogram, time_init_buffer, time_partition_elements;

    //TODO(finn): Implement partition size 1.
    // if (_partition_size == 1) {
    //   _partitions.resize(_partition_size);
    //   _partiton_offsets.resize(_partition_size);
    //
    //   const auto cluster_size = _elements.size();
    //   storage_memory.reserve(cluster_size);
    //
    //   std::ranges::copy(_elements, storage_memory.begin());
    //
    //   _partiton_offsets[0] = 0;
    //   auto& bucket = _partitions[0];
    //   bucket.data = storage_memory.data();
    //   bucket.size = cluster_size;
    //   _executed = true;
    //   return;
    // }

    auto start_compute_histogram = std::chrono::high_resolution_clock::now();

    auto histogram_data = std::move(_compute_histogram());
    auto& histogram = histogram_data.histogram;
    auto& cache_aligned_counts = histogram_data.cache_aligned_counts;

    auto end_compute_histogram = std::chrono::high_resolution_clock::now();
    time_histogram = duration_cast<std::chrono::milliseconds>(end_compute_histogram - start_compute_histogram).count();

    auto start_init_buffer = std::chrono::high_resolution_clock::now();

    auto* output_start_address = _tmp_output->tuples;

    auto buffers = cache_aligned_vector<CacheLine>(_partition_size);  // Cacheline aligned (64-bit).
    buffers[0].data.output_offset = 0;

    _output_buckets[0].num_tuples = histogram[0];
    _output_buckets[0].tuples = output_start_address;

    for (auto bucket_index = std::size_t{1}; bucket_index < _partition_size; ++bucket_index) {
      buffers[bucket_index].data.output_offset =
          buffers[bucket_index - 1].data.output_offset + cache_aligned_counts[bucket_index - 1];

      auto& partition = _output_buckets[bucket_index];
      partition.num_tuples = histogram[bucket_index];
      partition.tuples = output_start_address + buffers[bucket_index].data.output_offset;
    }

    auto end_init_buffer = std::chrono::high_resolution_clock::now();
    time_init_buffer = duration_cast<std::chrono::milliseconds>(end_init_buffer - start_init_buffer).count();

    auto start_partitioning = std::chrono::high_resolution_clock::now();

    auto* elements = _input->tuples;
    const auto num_elements = _input->num_tuples;

    for (auto index = size_t{0}; index < num_elements; ++index) {
      auto& element = *(elements + index);

      const auto bucket_index = _bucket_index(element.key);

      __builtin_prefetch(buffers.data() + bucket_index, 1, 3);
      auto& buffer = buffers[bucket_index];
      auto slot = buffer.data.output_offset;
      auto local_offset = slot & (BUFFER_SIZE - 1);

      buffer.tuples.values[local_offset] = element;
      if (local_offset == BUFFER_SIZE - 1) {
        auto* destination = output_start_address + slot - (BUFFER_SIZE - 1);
        auto* source = reinterpret_cast<SimdElement*>(&buffer);
        for (auto cache_line_index = size_t{0}; cache_line_index < NUM_CACHE_LINES; ++cache_line_index) {
          const auto offset = cache_line_index * TUPLES_PER_CACHELINE;
          _store_cacheline(destination + offset, source + offset);
        }
      }
      buffer.data.output_offset = slot + 1;
    }

    for (auto bucket_index = std::size_t{0}; bucket_index < _partition_size; ++bucket_index) {
      __builtin_prefetch(buffers.data() + bucket_index, 1, 3);
      auto& buffer = buffers[bucket_index];
      auto slot = buffer.data.output_offset;
      auto local_offset = slot & (BUFFER_SIZE - 1);
      if (local_offset > 0) {
        auto* output_address = output_start_address + slot - local_offset;
        std::memcpy(output_address, &buffer, local_offset * 8);
      }
    }
    auto end_partitioning = std::chrono::high_resolution_clock::now();
    time_partition_elements = duration_cast<std::chrono::milliseconds>(end_partitioning - start_partitioning).count();

    performance_info = {time_histogram, time_init_buffer, time_partition_elements};
    _executed = true;
  }

  std::size_t num_partitions() const {
    return _partition_size;
  }

  // Bucket& bucket(std::size_t index) {
  //   DebugAssert(_executed, "Do not call before execute.");
  //   DebugAssert(index >= 0 && index < num_partitions(), "Invalid partition index.");
  //   return _partitions[index];
  // }
  //
  // std::vector<Bucket>& buckets() {
  //   DebugAssert(_executed, "Do not call before execute.");
  //   return _partitions;
  // }

  // template <typename T>
  // T* get_working_memory(size_t partition_index, simd_sort::simd_vector<SimdElement>& working_memory) {
  //   DebugAssert(_executed, "Do not call before execute.");
  //   DebugAssert(partition_index >= 0 && partition_index < num_partitions(), "Invalid partition index.");
  //   DebugAssert(_partiton_offsets[partition_index] % 8 == 0, "Offset has to be cache_aligned.");
  //   return reinterpret_cast<T*>(working_memory.data() + _partiton_offsets[partition_index]);
  // }
};

}  // namespace hyrise::radix_partition
