#pragma once

#include "operators/join_simd_sort_merge/simd_utils.hpp"
#include "util.hpp"

#if defined(__x86_64__)
#include "immintrin.h"
#endif

namespace hyrise::radix_partition {

constexpr auto RADIX_BITS = uint8_t{8};
constexpr auto HASH_MASK = std::size_t{(1u << RADIX_BITS) - 1};
constexpr auto PARTITION_SIZE = uint32_t{1u << RADIX_BITS};
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
  std::size_t original_data_offset;
  std::size_t size;

  template <typename T>
  T* begin() const {
    return reinterpret_cast<T*>(data);
  }

  template <typename T>
  T* end() const {
    return reinterpret_cast<T*>(data + size);
  }

  std::span<SimdElement> elements() const {
    return {data, data + size};
  }
};

using BucketData = struct {
  std::size_t count;
  std::size_t cache_aligend_count;
};

using HistogramData = struct {
  std::array<BucketData, PARTITION_SIZE> histogram;
  std::size_t cache_aligned_size;
};

template <typename T>
using cache_aligned_vector = simd_sort::simd_vector<T>;

struct RadixPartition {
 public:
  explicit RadixPartition(const std::span<SimdElement> elements) : _has_data(true), _elements(elements) {}

  RadixPartition() = default;

 protected:
  bool _has_data = false;
  bool _executed = false;
  std::span<SimdElement> _elements;
  std::vector<Partition> _partitions;
  cache_aligned_vector<SimdElement> _partitioned_output;
  cache_aligned_vector<SimdElement> _working_memory;

  static constexpr std::size_t _align_to_cacheline(std::size_t value) {
    return (value + TUPLES_PER_CACHELINE - 1) & ~(TUPLES_PER_CACHELINE - 1);
  }

  HistogramData _compute_histogram() {
    auto histogram_data = HistogramData{};
    auto& histogram = histogram_data.histogram;

    for (auto& element : _elements) {
      ++histogram[element.key & HASH_MASK].count;
    }
    auto cache_aligned_output_size = std::size_t{0};
    for (auto bucket_index = std::size_t{0}; bucket_index < PARTITION_SIZE; ++bucket_index) {
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

  void execute() {
    DebugAssert(_has_data, "No input data to partition.");
    DebugAssert(!_executed, "RadixPartition execute can only be called once.");
    _partitions.resize(PARTITION_SIZE);

    auto histogram_data = _compute_histogram();
    auto histogram = histogram_data.histogram;
    _partitioned_output.resize(histogram_data.cache_aligned_size);

    auto* output_start_address = _partitioned_output.data();

    auto buffers = simd_sort::simd_vector<CacheLine>(PARTITION_SIZE);  // Cacheline aligned (64-bit).
    buffers[0].data.output_offset = 0;
    _partitions[0].original_data_offset = 0;
    _partitions[0].size = histogram[0].count;
    _partitions[0].data = output_start_address;

    for (auto bucket_index = std::size_t{1}; bucket_index < PARTITION_SIZE; ++bucket_index) {
      __builtin_prefetch(&buffers[bucket_index], 1, 0);
      buffers[bucket_index].data.output_offset =
          buffers[bucket_index - 1].data.output_offset + histogram[bucket_index - 1].cache_aligend_count;

      auto& partition = _partitions[bucket_index];
      partition.original_data_offset = buffers[bucket_index].data.output_offset;
      partition.size = histogram[bucket_index].count;
      partition.data = output_start_address + buffers[bucket_index].data.output_offset;
    }

    for (auto& element : _elements) {
      const auto bucket_index = element.key & HASH_MASK;
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

    for (auto bucket_index = std::size_t{0}; bucket_index < PARTITION_SIZE; ++bucket_index) {
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

  static constexpr std::size_t num_partitions() {
    return PARTITION_SIZE;
  }

  std::size_t get_cache_aligned_size() {
    DebugAssert(_executed, "Do not call before execute.");
    return _partitioned_output.size();
  }

  Partition& get_partition(std::size_t index) {
    DebugAssert(_executed, "Do not call before execute.");
    DebugAssert(index >= 0 && index < num_partitions(), "Invalid partition index.");
    return _partitions[index];
  }

  std::vector<Partition>& partitions() {
    DebugAssert(_executed, "Do not call before execute.");
    return _partitions;
  }

  template <typename T>
  T* get_working_memory(Partition& partition) {
    return reinterpret_cast<T*>(_working_memory.data() + partition.original_data_offset);
  }
};

}  // namespace hyrise::radix_partition
