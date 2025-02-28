#include "partition.hpp"

#include <immintrin.h> /* AVX intrinsics */
#include <math.h>      /* log2() */
#include <stdio.h>
#include <stdlib.h> /* calloc() */
#include <string.h> /* memset() */

#include "operators/join_simd_sort_merge/simd_utils.hpp"
/* #include "iacaMarks.h" */

#ifdef PERF_COUNTERS
#include "perf_counters.h" /* PCM_x */
#endif

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

#ifdef __GNUC__
#define UNUSED __attribute__((unused))
#else
#define UNUSED
#endif

/** Number of tuples that fits into a single cache line */
#define TUPLESPERCACHELINE (CACHE_LINE_SIZE / sizeof(SimdElement))

/** Modulo hash function using bitmask and shift */
#define HASH_BIT_MODULO(K, MASK, NBITS) (((K - 1) & MASK) >> NBITS)

/** Align a pointer to given size */
#define ALIGNPTR(PTR, SZ) (((uintptr_t)PTR + (SZ - 1)) & ~(SZ - 1))

/** Align N to number of tuples that is a multiple of cache lines */
#define ALIGN_NUMTUPLES(N) ((N + TUPLESPERCACHELINE - 1) & ~(TUPLESPERCACHELINE - 1))

using namespace hyrise;

/** Data structure for representing a single cache line */
using cacheline_t = union {
  struct {
    std::array<SimdElement, TUPLESPERCACHELINE> tuples;
  } tuples;

  struct {
    std::array<SimdElement, TUPLESPERCACHELINE - 1> tuples;
    uint32_t slot;
  } data;
};

/**
 * Makes a non-temporal write of 64 bytes from src to dst.
 * Uses vectorized non-temporal stores if available, falls
 * back to assignment copy.
 *
 * @param dst
 * @param src
 *
 * @return
 */
static inline void store_nontemp_64B(void* dst, void* src) {
#ifdef __AVX__
  __m256i* d1 = (__m256i*)dst;
  __m256i s1 = *((__m256i*)src);
  __m256i* d2 = d1 + 1;
  __m256i s2 = *(((__m256i*)src) + 1);

  _mm256_stream_si256(d1, s1);
  _mm256_stream_si256(d2, s2);
#elif defined(__SSE2__)

  register __m128i* d1 = (__m128i*)dst;
  register __m128i* d2 = d1 + 1;
  register __m128i* d3 = d1 + 2;
  register __m128i* d4 = d1 + 3;
  register __m128i s1 = *(__m128i*)src;
  register __m128i s2 = *((__m128i*)src + 1);
  register __m128i s3 = *((__m128i*)src + 2);
  register __m128i s4 = *((__m128i*)src + 3);

  _mm_stream_si128(d1, s1);
  _mm_stream_si128(d2, s2);
  _mm_stream_si128(d3, s3);
  _mm_stream_si128(d4, s4);

#else
  /* just copy with assignment */
  *(cacheline_t*)dst = *(cacheline_t*)src;

#endif
}

void radix_cluster_optimized(Relation* __restrict out_rel, Relation* __restrict in_rel, int32_t* __restrict hist,
                             size_t shift_bits, size_t nbits) {
  uint32_t index{};
  uint32_t offset = 0;
  const uint32_t mod = ((1u << nbits) - 1u) << shift_bits;
  const uint32_t fan_out = 1u << nbits;
  const uint32_t ntuples = in_rel->num_tuples;

  SimdElement* input = in_rel->tuples;
  SimdElement* output = out_rel->tuples;

  auto dst = simd_sort::simd_vector<uint32_t>(fan_out);
  auto buffer = simd_sort::simd_vector<cacheline_t>(fan_out);

  /* count tuples per cluster */
  for (index = 0; index < ntuples; index++) {
    uint32_t idx = HASH_BIT_MODULO(input->key, mod, shift_bits);
    hist[idx]++;
    input++;
  }

  for (index = 0; index < fan_out; index++) {
    buffer[index].data.slot = 0;
  }

  /* determine the start and end of each cluster depending on the counts. */
  for (index = 0; index < fan_out; index++) {
    dst[index] = offset;
    /* for aligning partition-outputs to cacheline: */
    offset += ALIGN_NUMTUPLES(hist[index]);
  }

  input = in_rel->tuples;
  /* copy tuples to their corresponding clusters at appropriate offsets */
  for (index = 0; index < ntuples; index++) {
    uint32_t idx = HASH_BIT_MODULO(input->key, mod, shift_bits);
    /* store in the cache-resident buffer first */
    uint32_t slot = buffer[idx].data.slot;
    auto* tup = reinterpret_cast<SimdElement*>(buffer.data() + idx);
    tup[slot] = *input;
    input++;
    slot++;

    if (slot == TUPLESPERCACHELINE) {
      store_nontemp_64B((output + dst[idx]), (buffer.data() + idx));
      slot = 0;
      dst[idx] += TUPLESPERCACHELINE;
    }
    buffer[idx].data.slot = slot;
  }

  /* flush the remainder tuples in the buffer */
  for (index = 0; index < fan_out; index++) {
    uint32_t num = buffer[index].data.slot;
    if (num > 0) {
      SimdElement* dest = output + dst[index];

      for (uint32_t j = 0; j < num; j++) {
        dest[j] = buffer[index].data.tuples[j];
      }
    }
  }
}

void partition_relation_optimized(std::span<Relation> partitions, Relation* input, Relation* output, uint32_t nbits,
                                  uint32_t shiftbits) {
  int index{};
  uint32_t offset = 0;
  const int fan_out = static_cast<int>(1u << nbits);

  auto hist = simd_sort::simd_vector<int32_t>(fan_out + 16);
  int32_t* hist_aligned = hist.data();

  radix_cluster_optimized(output, input, hist_aligned, shiftbits, nbits);

  for (index = 0; index < fan_out; index++) {
    Relation& part = partitions[index];
    part.num_tuples = hist_aligned[index];
    part.tuples = output->tuples + offset;

    offset += ALIGN_NUMTUPLES(hist_aligned[index]);
  }
}
