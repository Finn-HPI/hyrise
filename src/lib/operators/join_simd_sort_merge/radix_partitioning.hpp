#pragma once

#include <algorithm>
#include <span>
#include <utility>
#include <vector>

#include "operators/join_simd_sort_merge/simd_utils.hpp"
#include "util.hpp"

namespace hyrise::radix_partition {

struct Bucket {
  SimdElement* data;
  std::size_t size;

  template <typename T>
  T* begin() const {
    return reinterpret_cast<T*>(data);
  }

  template <typename T>
  T* end() const {
    return reinterpret_cast<T*>(data + size);
  }

  bool empty() const {
    return size == 0;
  }

  std::span<SimdElement> elements() const {
    return {data, size};
  }

  void retrieve_elements(size_t count) {
    DebugAssert(count <= size, "Can not retrieve more elements than currently in Bucket.");
    size -= count;
    data += count;
  }
};

template <typename T>
using cache_aligned_vector = simd_sort::simd_vector<T>;

template <typename ColumnType>
struct RadixPartition {
 public:
  explicit RadixPartition(const std::span<SimdElement> elements, size_t cluster_count)
      : _partition_size(cluster_count),
        _bitshift_count{32u - simd_sort::log2_builtin(cluster_count)},
        _has_data(true),
        _elements(elements) {}

  RadixPartition() = default;

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

  struct BucketData {
    std::size_t count;
    std::size_t cache_aligend_count;
  };

  struct HistogramData {
    explicit HistogramData(size_t partition_size) : histogram(partition_size) {}

    std::vector<BucketData> histogram;
    std::size_t cache_aligned_size{};
  };

  size_t _partition_size{};
  size_t _bitshift_count{};
  bool _has_data = false;
  bool _executed = false;
  std::span<SimdElement> _elements;
  std::vector<Bucket> _partitions;
  std::vector<std::size_t> _partiton_offsets;

  static constexpr std::size_t _align_to_cacheline(std::size_t value) {
    return (value + BUFFER_SIZE - 1) & ~(BUFFER_SIZE - 1);
  }

  size_t _bucket_index(uint32_t key) {
    return key >> _bitshift_count;
  }

  HistogramData _compute_histogram() {
    auto histogram_data = HistogramData(_partition_size);
    auto& histogram = histogram_data.histogram;

    for (auto& element : _elements) {
      const auto bucket_index = _bucket_index(element.key);
      __builtin_prefetch(histogram.data() + bucket_index, 1, 3);
      ++histogram[bucket_index].count;
    }
    auto cache_aligned_output_size = std::size_t{0};
    for (auto bucket_index = std::size_t{0}; bucket_index < _partition_size; ++bucket_index) {
      histogram[bucket_index].cache_aligend_count = _align_to_cacheline(histogram[bucket_index].count);
      cache_aligned_output_size += histogram[bucket_index].cache_aligend_count;
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
  void set_element(std::span<SimdElement> elements) {
    _has_data = true;
    _elements = elements;
  }

  void execute(simd_sort::simd_vector<SimdElement>& storage_memory,
               simd_sort::simd_vector<SimdElement>& working_memory) {
    DebugAssert(_has_data, "No input data to partition.");
    DebugAssert(!_executed, "RadixPartition execute can only be called once.");

    // NOLINTNEXTLINE
    size_t time_histogram, time_init_buffer, time_partition_elements;

    if (_partition_size == 1) {
      _partitions.resize(_partition_size);
      _partiton_offsets.resize(_partition_size);

      const auto cluster_size = _elements.size();
      storage_memory.resize(cluster_size);
      working_memory.resize(cluster_size);

      std::ranges::copy(_elements, storage_memory.begin());
      _partiton_offsets[0] = 0;
      auto& bucket = _partitions[0];
      bucket.data = storage_memory.data();
      bucket.size = cluster_size;
      _executed = true;
      return;
    }

    auto start_compute_histogram = std::chrono::high_resolution_clock::now();
    auto histogram_data = std::move(_compute_histogram());
    auto& histogram = histogram_data.histogram;

    auto end_compute_histogram = std::chrono::high_resolution_clock::now();
    time_histogram = duration_cast<std::chrono::milliseconds>(end_compute_histogram - start_compute_histogram).count();

    auto start_init_buffer = std::chrono::high_resolution_clock::now();

    _partitions.resize(_partition_size);
    _partiton_offsets.resize(_partition_size);

    storage_memory.resize(histogram_data.cache_aligned_size);
    working_memory.resize(histogram_data.cache_aligned_size);

    auto* output_start_address = storage_memory.data();

    auto buffers = cache_aligned_vector<CacheLine>(_partition_size);  // Cacheline aligned (64-bit).
    buffers[0].data.output_offset = 0;
    _partitions[0].size = histogram[0].count;
    _partitions[0].data = output_start_address;
    _partiton_offsets[0] = 0;

    for (auto bucket_index = std::size_t{1}; bucket_index < _partition_size; ++bucket_index) {
      buffers[bucket_index].data.output_offset =
          buffers[bucket_index - 1].data.output_offset + histogram[bucket_index - 1].cache_aligend_count;

      auto& partition = _partitions[bucket_index];
      _partiton_offsets[bucket_index] = buffers[bucket_index].data.output_offset;
      partition.size = histogram[bucket_index].count;
      partition.data = output_start_address + buffers[bucket_index].data.output_offset;
    }

    auto end_init_buffer = std::chrono::high_resolution_clock::now();
    time_init_buffer = duration_cast<std::chrono::milliseconds>(end_init_buffer - start_init_buffer).count();

    auto start_partitioning = std::chrono::high_resolution_clock::now();
    for (auto& element : _elements) {
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
        // std::memcpy(destination, source, 8 * BUFFER_SIZE);
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

  Bucket& bucket(std::size_t index) {
    DebugAssert(_executed, "Do not call before execute.");
    DebugAssert(index >= 0 && index < num_partitions(), "Invalid partition index.");
    return _partitions[index];
  }

  std::vector<Bucket>& buckets() {
    DebugAssert(_executed, "Do not call before execute.");
    return _partitions;
  }

  template <typename T>
  T* get_working_memory(size_t partition_index, simd_sort::simd_vector<SimdElement>& working_memory) {
    DebugAssert(_executed, "Do not call before execute.");
    DebugAssert(partition_index >= 0 && partition_index < num_partitions(), "Invalid partition index.");
    DebugAssert(_partiton_offsets[partition_index] % 8 == 0, "Offset has to be cache_aligned.");
    return reinterpret_cast<T*>(working_memory.data() + _partiton_offsets[partition_index]);
  }
};

}  // namespace hyrise::radix_partition
