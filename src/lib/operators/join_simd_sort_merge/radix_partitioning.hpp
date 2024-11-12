#pragma once

#include <span>
#include <vector>

#include "operators/join_simd_sort_merge/simd_utils.hpp"
#include "util.hpp"

#if defined(__x86_64__)
#include "immintrin.h"
#endif

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
      : _partition_size(cluster_count), _hash_mask(cluster_count - 1), _has_data(true), _elements(elements) {}

  RadixPartition() = default;

 protected:
  using CacheLine = union {
    struct {
      std::array<SimdElement, TUPLES_PER_CACHELINE> values;
    } tuples;

    struct {
      std::array<SimdElement, TUPLES_PER_CACHELINE - 1> values;
      std::size_t output_offset;
    } data;
  };

  using BucketData = struct {
    std::size_t count;
    std::size_t cache_aligend_count;
  };

  struct HistogramData {
    explicit HistogramData(size_t partition_size) : histogram(partition_size) {}

    std::vector<BucketData> histogram;
    std::size_t cache_aligned_size{};
  };

  size_t _partition_size{};
  size_t _hash_mask{};
  bool _has_data = false;
  bool _executed = false;
  std::span<SimdElement> _elements;
  std::vector<Bucket> _partitions;
  std::vector<std::size_t> _partiton_offsets;

  static constexpr std::size_t _align_to_cacheline(std::size_t value) {
    return (value + TUPLES_PER_CACHELINE - 1) & ~(TUPLES_PER_CACHELINE - 1);
  }

  template <typename T>
  inline std::size_t _bucket_index(T key, std::size_t seed = 41) {
    boost::hash_combine(seed, key);
    return seed & _hash_mask;
  }

  HistogramData _compute_histogram() {
    auto histogram_data = HistogramData(_partition_size);
    auto& histogram = histogram_data.histogram;

    for (auto& element : _elements) {
      ++histogram[_bucket_index(element.key)].count;
    }
    auto cache_aligned_output_size = std::size_t{0};
    for (auto bucket_index = std::size_t{0}; bucket_index < _partition_size; ++bucket_index) {
      __builtin_prefetch(&histogram[bucket_index], 1, 0);
      histogram[bucket_index].cache_aligend_count = _align_to_cacheline(histogram[bucket_index].count);
      cache_aligned_output_size += histogram[bucket_index].cache_aligend_count;
    }

    histogram_data.cache_aligned_size = cache_aligned_output_size;
    return histogram_data;
  }

  void _store_cacheline(auto* destination, auto* source) {
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

 public:
  void set_element(std::span<SimdElement> elements) {
    _has_data = true;
    _elements = elements;
  }

  void execute(simd_sort::simd_vector<SimdElement>& storage_memory,
               simd_sort::simd_vector<SimdElement>& working_memory) {
    DebugAssert(_has_data, "No input data to partition.");
    DebugAssert(!_executed, "RadixPartition execute can only be called once.");
    _partitions.resize(_partition_size);
    _partiton_offsets.resize(_partition_size);

    if (_partition_size == 1) {
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

    auto histogram_data = std::move(_compute_histogram());
    auto histogram = histogram_data.histogram;

    storage_memory.resize(histogram_data.cache_aligned_size);
    working_memory.resize(histogram_data.cache_aligned_size);

    auto* output_start_address = storage_memory.data();

    auto buffers = cache_aligned_vector<CacheLine>(_partition_size);  // Cacheline aligned (64-bit).
    buffers[0].data.output_offset = 0;
    _partitions[0].size = histogram[0].count;
    _partitions[0].data = output_start_address;
    _partiton_offsets[0] = 0;

    for (auto bucket_index = std::size_t{1}; bucket_index < _partition_size; ++bucket_index) {
      __builtin_prefetch(&buffers[bucket_index - 1], 1, 0);
      __builtin_prefetch(&buffers[bucket_index], 1, 0);

      buffers[bucket_index].data.output_offset =
          buffers[bucket_index - 1].data.output_offset + histogram[bucket_index - 1].cache_aligend_count;

      auto& partition = _partitions[bucket_index];
      _partiton_offsets[bucket_index] = buffers[bucket_index].data.output_offset;
      partition.size = histogram[bucket_index].count;
      partition.data = output_start_address + buffers[bucket_index].data.output_offset;
    }

    for (auto& element : _elements) {
      const auto bucket_index = _bucket_index(element.key);
      __builtin_prefetch(&buffers[bucket_index], 1, 0);
      auto& buffer = buffers[bucket_index];
      auto slot = buffer.data.output_offset;
      auto local_offset = slot & (TUPLES_PER_CACHELINE - 1);

      buffer.tuples.values[local_offset] = element;
      if (local_offset == TUPLES_PER_CACHELINE - 1) {
        auto* destination = output_start_address + slot - (TUPLES_PER_CACHELINE - 1);
        auto* source = &buffer;
        _store_cacheline(destination, source);
      }
      buffer.data.output_offset = slot + 1;
    }

    for (auto bucket_index = std::size_t{0}; bucket_index < _partition_size; ++bucket_index) {
      __builtin_prefetch(&buffers[bucket_index], 1, 0);
      auto& buffer = buffers[bucket_index];
      auto slot = buffer.data.output_offset;
      auto local_offset = slot & (TUPLES_PER_CACHELINE - 1);
      if (local_offset > 0) {
        auto* output_address = output_start_address + slot - local_offset;
        std::memcpy(output_address, &buffer, local_offset * 8);
      }
    }
    _executed = true;
  }

  std::size_t num_partitions() const {
    return _partition_size;
  }

  // std::size_t get_cache_aligned_size() const {
  //   DebugAssert(_executed, "Do not call before execute.");
  //   return _partitioned_output.size();
  // }

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
