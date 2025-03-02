#include <algorithm>
#include <barrier>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <vector>

#include "barrier.hpp"
#include "cpu_mapping.hpp"
#include "lock.hpp"
#include "operators/join_simd_sort_merge/multiway_merging_balkesen.hpp"
#include "operators/join_simd_sort_merge/radix_partitioning_balkesen.hpp"
#include "operators/join_simd_sort_merge/simd_sort.hpp"
#include "operators/join_simd_sort_merge/simd_utils.hpp"
#include "operators/join_simd_sort_merge/util.hpp"
#include "partition.hpp"
#include "rdtsc.hpp"
#include "scheduler/immediate_execution_scheduler.hpp"
#include "scheduler/node_queue_scheduler.hpp"
#include "types.hpp"

using namespace hyrise;  // NOLINT(build/namespaces)

namespace {

constexpr size_t cache_line_padding(size_t partition_fan_out) {
  constexpr auto CACHE_LINE_SIZE = 64;
  return (partition_fan_out * CACHE_LINE_SIZE / sizeof(SimdElement));
}

constexpr size_t relation_padding(size_t thread_count, size_t partition_fan_out) {
  return thread_count * cache_line_padding(partition_fan_out);
}

}  // namespace

// NOLINTBEGIN
#define RAND_RANGE(N) ((double)rand() / ((double)RAND_MAX + 1) * (N))
#define RAND_RANGE48(N, STATE) ((double)nrand48(STATE) / ((double)RAND_MAX + 1) * (N))

static int seeded = 0;
static unsigned int seed_value;

void seed_generator(unsigned int seed) {
  srand(seed);
  seed_value = seed;
  seeded = 1;
}

/** Check wheter seeded, if not seed the generator with current time */
static void check_seed() {
  if (!seeded) {
    seed_value = time(NULL);
    srand(seed_value);
    seeded = 1;
  }
}

struct create_arg_t {
  Relation rel;
  int64_t firstkey;
  int64_t maxid;
  Relation* fullrel;
  volatile void* locks;
  pthread_barrier_t* barrier;
};

typedef struct create_arg_t create_arg_t;

void* random_unique_gen_thread(void* args) {
  create_arg_t* arg = (create_arg_t*)args;
  Relation* rel = &arg->rel;
  int64_t firstkey = arg->firstkey;
  int64_t maxid = arg->maxid;
  uint64_t i;

  uint32_t randstart = 5; /* rand() % 1000; */

  /* for randomly seeding nrand48() */
  unsigned short state[3] = {0, 0, 0};
  unsigned int seed = time(NULL) + *(unsigned int*)pthread_self();
  memcpy(state, &seed, sizeof(seed));

  for (i = 0; i < rel->num_tuples; i++) {
    rel->tuples[i].key = firstkey;
    rel->tuples[i].index = randstart + i;

    if (firstkey == maxid)
      firstkey = 0;

    firstkey++;
  }

  /* randomly shuffle elements */
  /* knuth_shuffle48(rel, state); */

  /* wait at a barrier until all threads finish initializing data */
  int rv;
  BARRIER_ARRIVE(arg->barrier, rv)

  /* parallel synchronized knuth-shuffle */
  volatile char* locks = (volatile char*)(arg->locks);
  Relation* fullrel = arg->fullrel;

  uint64_t rel_offset_in_full = rel->tuples - fullrel->tuples;
  uint64_t k = rel_offset_in_full + rel->num_tuples - 1;
  for (i = rel->num_tuples - 1; i > 0; i--, k--) {
    int64_t j = static_cast<int64_t>(((double)nrand48(state) / ((double)2147483647 + 1) * static_cast<double>(k)));
    lock(locks + k); /* lock this rel-idx=i, fullrel-idx=k */
    lock(locks + j); /* lock full rel-idx=j */

    uint32_t tmp = fullrel->tuples[k].key;
    fullrel->tuples[k].key = fullrel->tuples[j].key;
    fullrel->tuples[j].key = tmp;

    unlock(locks + j);
    unlock(locks + k);
  }

  return 0;
}

void create_relation(Relation* relation, uint64_t num_tuples, uint32_t nthreads, uint64_t maxid) {
  int rv;
  uint32_t i;
  uint64_t offset = 0;

  check_seed();

  // relation->num_tuples = num_tuples;
  //
  // if (!relation->tuples) {
  //   perror("memory must be allocated first");
  //   return -1;
  // }

  auto args = std::vector<create_arg_t>(nthreads);
  auto tid = std::vector<pthread_t>(nthreads);
  cpu_set_t set;
  pthread_attr_t attr;
  pthread_barrier_t barrier;

  unsigned int pagesize;
  unsigned int npages;
  unsigned int npages_perthr;
  uint64_t ntuples_perthr;
  uint64_t ntuples_lastthr;

  pagesize = getpagesize();
  npages = (num_tuples * sizeof(SimdElement)) / pagesize + 1;
  npages_perthr = npages / nthreads;
  ntuples_perthr = npages_perthr * (pagesize / sizeof(SimdElement));

  if (npages_perthr == 0)
    ntuples_perthr = num_tuples / nthreads;

  ntuples_lastthr = num_tuples - ntuples_perthr * (nthreads - 1);

  pthread_attr_init(&attr);

  rv = pthread_barrier_init(&barrier, NULL, nthreads);
  if (rv != 0) {
    printf("[ERROR] Couldn't create the barrier\n");
    exit(EXIT_FAILURE);
  }

  volatile void* locks = (volatile void*)calloc(num_tuples, sizeof(char));

  for (i = 0; i < nthreads; i++) {
    int cpu_idx = get_cpu_id(i);

    CPU_ZERO(&set);
    CPU_SET(cpu_idx, &set);
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);

    args[i].firstkey = (offset + 1) % maxid;
    args[i].maxid = maxid;
    args[i].rel.tuples = relation->tuples + offset;
    args[i].rel.num_tuples = (i == nthreads - 1) ? ntuples_lastthr : ntuples_perthr;

    args[i].fullrel = relation;
    args[i].locks = locks;
    args[i].barrier = &barrier;

    offset += ntuples_perthr;

    rv = pthread_create(&tid[i], &attr, random_unique_gen_thread, (void*)&args[i]);
    if (rv) {
      fprintf(stderr, "[ERROR] pthread_create() return code is %d\n", rv);
      exit(-1);
    }
  }
  for (i = 0; i < nthreads; i++) {
    pthread_join(tid[i], NULL);
  }

  free((char*)locks);
  pthread_barrier_destroy(&barrier);
}

// NOLINTEND

auto CLUSTER_COUNT = size_t{128};
auto L3_CACHE_SIZE = size_t{20} * 1024 * 1024;
using ColumnType = uint32_t;

struct RelationPair {
  Relation relation_r;
  Relation relation_s;
};

struct ThreadInfo {
  SimdElement* rel_r{};
  SimdElement* rel_s{};

  SimdElement* tmp_part_r{};
  SimdElement* tmp_part_s{};

  SimdElement* tmp_sort_r{};
  SimdElement* tmp_sort_s{};

  size_t num_r{};
  size_t num_s{};

  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::chrono::time_point<std::chrono::high_resolution_clock> end;

  size_t tid{};
  std::vector<simd_sort::simd_vector<RelationPair>>* thread_chunks{};
  size_t results{};
  uint64_t part{}, sort{}, merge{}, join{};
};

namespace {
using namespace hyrise::simd_sort;
using namespace hyrise::radix_partition;
using namespace hyrise::multiway_merging;

[[maybe_unused]] bool is_sorted_helper(int64_t* items, uint64_t nitems) {
  uint32_t curr = 0;
  uint64_t index{};
  bool warned = false;
  auto* tuples = reinterpret_cast<SimdElement*>(items);
  for (index = 0; index < nitems; index++) {
    if (tuples[index].key == curr) {
      if (!warned) {
        warned = true;
        // std::cout << "[WARN] Equal items, still ok... item[" << index << "].key=" << tuples[index].key << std::endl;
      }
    } else if (tuples[index].key < curr) {
      std::cout << "[ERROR] item[" << index << "].key=" << tuples[index].key << " is less than item[" << (index - 1)
                << "].key=" << curr << std::endl;
      return false;
    }

    curr = tuples[index].key;
  }

  return true;
}

[[maybe_unused]] void partition_phase(std::span<Relation> buckets_r, std::span<Relation> buckets_s,
                                      ThreadInfo& thread_info) {
  auto chunk_r = Relation{.tuples = thread_info.rel_r, .num_tuples = thread_info.num_r};
  auto chunk_s = Relation{.tuples = thread_info.rel_s, .num_tuples = thread_info.num_s};

  auto tmp_chunk_r = Relation{.tuples = thread_info.tmp_part_r, .num_tuples = thread_info.num_r};
  auto tmp_chunk_s = Relation{.tuples = thread_info.tmp_part_s, .num_tuples = thread_info.num_s};

  {
    auto radix_partition_r = RadixPartitionBalkesen<ColumnType>(&chunk_r, &tmp_chunk_r, buckets_r, CLUSTER_COUNT);
    radix_partition_r.execute();
  }
  {
    auto radix_partition_s = RadixPartitionBalkesen<ColumnType>(&chunk_s, &tmp_chunk_s, buckets_s, CLUSTER_COUNT);
    radix_partition_s.execute();
  }
}

[[maybe_unused]] void partition_phase2(std::span<Relation> buckets_r, std::span<Relation> buckets_s,
                                       ThreadInfo& thread_info, size_t num_threads) {
  const auto nradixbits = static_cast<int>(log2(static_cast<double>(CLUSTER_COUNT)));

  auto chunk_r = Relation{.tuples = thread_info.rel_r, .num_tuples = thread_info.num_r};
  auto chunk_s = Relation{.tuples = thread_info.rel_s, .num_tuples = thread_info.num_s};

  auto tmp_chunk_r = Relation{.tuples = thread_info.tmp_part_r, .num_tuples = thread_info.num_r};
  auto tmp_chunk_s = Relation{.tuples = thread_info.tmp_part_s, .num_tuples = thread_info.num_s};

  int bitshift = static_cast<int>(ceil(log2(static_cast<double>(chunk_r.num_tuples * num_threads))) - 1);
  bitshift = bitshift - nradixbits - 1;

  partition_relation_optimized(buckets_r, &chunk_r, &tmp_chunk_r, nradixbits, bitshift);
  partition_relation_optimized(buckets_s, &chunk_s, &tmp_chunk_s, nradixbits, bitshift);
}

[[maybe_unused]] constexpr std::size_t choose_count_per_vector() {
#if defined(__AVX512F__)
  return 8;
#else
  return 4;
#endif
}

using SortingType = int64_t;

SortingType compare_value(uint32_t key) {
  return std::bit_cast<SortingType>(static_cast<uint64_t>(key) << 32u);
}

std::size_t equal_value_range_size(std::size_t start_index, std::span<SimdElement>& elements) {
  if (start_index >= elements.size()) {
    return 0;
  }
  auto begin = elements.begin();
  std::advance(begin, start_index);
  const auto run_value = compare_value(begin->key);

  constexpr auto LINEAR_SEARCH_ITEMS = std::size_t{128};
  auto end = begin + LINEAR_SEARCH_ITEMS;
  if (start_index + LINEAR_SEARCH_ITEMS >= elements.size()) {
    // Set end of linear search to end of input vector if we would overshoot otherwise.
    end = elements.end();
  }

  const auto linear_search_result = std::find_if(begin, end, [&](const auto& simd_element) {
    return compare_value(simd_element.key) > run_value;
  });

  if (linear_search_result != end) {
    // Match found within the linearly scanned part.
    return std::distance(begin, linear_search_result);
  }

  if (linear_search_result == elements.end() || compare_value(end->key) > run_value) {
    // We did not find a larger value in the linearly scanned part and it spanned until the end of the input vector.
    // That means all values up to the end are part of the run.
    return std::distance(begin, end);
  }

  // Binary search in case the run did not end within the linearly scanned part.
  const auto binary_search_result = std::upper_bound(end, elements.end(), *end, [](const auto& lhs, const auto& rhs) {
    return compare_value(lhs.key) < compare_value(rhs.key);
  });
  return std::distance(begin, binary_search_result);
}

enum class CompareResult : std::uint8_t { Less, Greater, Equal };

template <typename T>
CompareResult compare(const T& left, const T& right) {
  if (left < right) {
    return CompareResult::Less;
  }

  if (left == right) {
    return CompareResult::Equal;
  }

  return CompareResult::Greater;
}

struct PotentialMatchRange {
  PotentialMatchRange(std::size_t init_start_index, std::size_t init_end_index, std::span<SimdElement> init_elements)
      : start_index(init_start_index),
        end_index(init_end_index),
        elements(init_elements.data() + init_start_index, init_elements.data() + init_end_index)

  {}

  std::size_t start_index;
  std::size_t end_index;
  std::span<SimdElement> elements;

 public:
  void for_every_row_id(auto&& action) const {
    const auto num_items = elements.size();
    for (auto index = size_t{0}; index < num_items; ++index) {
      action();
    }
  }

  void find_matches_with_range(const PotentialMatchRange& other_range, auto&& action) const {
    // Handle float and int32_t values.
    this->for_every_row_id([&]() {
      other_range.for_every_row_id([&]() {
        action();
      });
    });
  }
};

void find_matches_in_ranges(const PotentialMatchRange& left_range, const PotentialMatchRange& right_range,
                            const CompareResult compare_result, size_t& matches) {
  if (compare_result == CompareResult::Equal) {
    left_range.find_matches_with_range(right_range, [&]() {
      ++matches;
    });
  }
}

size_t join_per_hash(std::span<SimdElement> left_elements, std::span<SimdElement> right_elements) {
  auto left_run_start = size_t{0};
  auto right_run_start = size_t{0};

  auto matches = size_t{0};

  auto left_run_end = left_run_start + equal_value_range_size(left_run_start, left_elements);
  auto right_run_end = right_run_start + equal_value_range_size(right_run_start, right_elements);

  const auto left_size = left_elements.size();
  const auto right_size = right_elements.size();

  while (left_run_start < left_size && right_run_start < right_size) {
    const auto left_value = compare_value(left_elements[left_run_start].key);
    const auto right_value = compare_value(right_elements[right_run_start].key);

    const auto compare_result = compare(left_value, right_value);

    [[maybe_unused]] const auto left_range = PotentialMatchRange(left_run_start, left_run_end, left_elements);
    [[maybe_unused]] const auto right_range = PotentialMatchRange(right_run_start, right_run_end, right_elements);

    find_matches_in_ranges(left_range, right_range, compare_result, matches);

    // Advance to the next run on the smaller side or both if equal.
    switch (compare_result) {
      case CompareResult::Equal:
        // Advance both runs.
        left_run_start = left_run_end;
        right_run_start = right_run_end;
        left_run_end = left_run_start + equal_value_range_size(left_run_start, left_elements);
        right_run_end = right_run_start + equal_value_range_size(right_run_start, right_elements);
        break;
      case CompareResult::Less:
        // Advance the left run.
        left_run_start = left_run_end;
        left_run_end = left_run_start + equal_value_range_size(left_run_start, left_elements);
        break;
      case CompareResult::Greater:
        // Advance the right run.
        right_run_start = right_run_end;
        right_run_end = right_run_start + equal_value_range_size(right_run_start, right_elements);
        break;
      default:
        throw std::logic_error("Unknown CompareResult.");
    }
  }
  return matches;
}

[[maybe_unused]] void sorting_phase(std::span<Relation> buckets_r, std::span<Relation> buckets_s,
                                    ThreadInfo& thread_info) {
  const auto tid = thread_info.tid;
  [[maybe_unused]] const auto count_per_vector = choose_count_per_vector();

  auto& thread_chunks = *(thread_info.thread_chunks);

  thread_chunks[tid].resize(CLUSTER_COUNT);

  // Sort local R buckets.
  auto offset = size_t{0};
  auto* sort_outupt_begin = thread_info.tmp_sort_r + (tid * cache_line_padding(CLUSTER_COUNT));
  for (auto bucket_index = size_t{0}; bucket_index < CLUSTER_COUNT; ++bucket_index) {
    auto num_tuples_in_bucket = buckets_r[bucket_index].num_tuples;

    auto* input_pointer = reinterpret_cast<SortingType*>(buckets_r[bucket_index].tuples);
    auto* output_pointer = reinterpret_cast<SortingType*>(sort_outupt_begin + offset);
    offset += align_to_cacheline(num_tuples_in_bucket);

    DebugAssert((simd_sort::is_simd_aligned<SortingType, 64>(input_pointer)), "Input not cache aligned.");
    DebugAssert((simd_sort::is_simd_aligned<SortingType, 64>(output_pointer)), "Output not cache aligned.");

    simd_sort::sort<count_per_vector, SortingType, ExecutionStrategy::SEQUENTIAL>(input_pointer, output_pointer,
                                                                                  num_tuples_in_bucket);

    // if (!is_sorted_helper(output_pointer, num_tuples_in_bucket)) {
    //   std::cout << "===> " << tid << "-thread -> R is NOT sorted, size = " << num_tuples_in_bucket << std::endl;
    // }
    thread_chunks[tid][bucket_index].relation_r.tuples = reinterpret_cast<SimdElement*>(output_pointer);
    thread_chunks[tid][bucket_index].relation_r.num_tuples = num_tuples_in_bucket;
  }

  offset = 0;
  sort_outupt_begin = thread_info.tmp_sort_s + (tid * cache_line_padding(CLUSTER_COUNT));
  for (auto bucket_index = size_t{0}; bucket_index < CLUSTER_COUNT; ++bucket_index) {
    auto num_tuples_in_bucket = buckets_s[bucket_index].num_tuples;
    auto* input_pointer = reinterpret_cast<SortingType*>(buckets_s[bucket_index].tuples);
    auto* output_pointer = reinterpret_cast<SortingType*>(sort_outupt_begin + offset);
    offset += align_to_cacheline(num_tuples_in_bucket);

    simd_sort::sort<count_per_vector, SortingType, ExecutionStrategy::SEQUENTIAL>(input_pointer, output_pointer,
                                                                                  num_tuples_in_bucket);
    // if (!is_sorted_helper(output_pointer, num_tuples_in_bucket)) {
    //   std::cout << "===> " << tid << "-thread -> R is NOT sorted, size = " << num_tuples_in_bucket << std::endl;
    // }

    thread_chunks[tid][bucket_index].relation_s.tuples = reinterpret_cast<SimdElement*>(output_pointer);
    thread_chunks[tid][bucket_index].relation_s.num_tuples = num_tuples_in_bucket;
  }
}

[[maybe_unused]] void mwaymerge_phase(simd_vector<SimdElement>& merged_tuples_r,
                                      simd_vector<SimdElement>& merged_tuples_s, Relation& merged_r, Relation& merged_s,
                                      ThreadInfo& thread_info, size_t thread_count) {
  const auto curr_tid = thread_info.tid;
  const auto bucket_ids_per_thread = CLUSTER_COUNT / thread_count;
  const auto start_bucket_id = curr_tid * bucket_ids_per_thread;
  const auto end_bucket_id = start_bucket_id + bucket_ids_per_thread;

  auto output_size_r = size_t{0};
  auto output_size_s = size_t{0};

  auto parts_r = std::vector<Relation*>(CLUSTER_COUNT);
  auto parts_s = std::vector<Relation*>(CLUSTER_COUNT);
  uint32_t part_index = 0;
  for (auto bucket_index = start_bucket_id; bucket_index < end_bucket_id; bucket_index++) {
    for (auto thread_index = size_t{0}; thread_index < thread_count; thread_index++) {
      auto tid = (curr_tid + thread_index) % thread_count;
      auto& relation_pair = (*(thread_info.thread_chunks))[tid][bucket_index];

      parts_r[part_index] = &relation_pair.relation_r;
      parts_s[part_index] = &relation_pair.relation_s;
      part_index++;

      output_size_r += relation_pair.relation_r.num_tuples;
      output_size_s += relation_pair.relation_s.num_tuples;
    }
  }

  // std::cout << "output_size_r: " << output_size_r << ", output_size_s: " << output_size_s << std::endl;

  merged_tuples_r.resize(output_size_r);
  merged_tuples_s.resize(output_size_s);
  const auto buffer_size = L3_CACHE_SIZE / thread_count;
  auto mway_merge_r = MultiwayMergerBalkesen<choose_count_per_vector(), SortingType>(parts_r, buffer_size);
  auto mway_merge_s = MultiwayMergerBalkesen<choose_count_per_vector(), SortingType>(parts_s, buffer_size);
  mway_merge_r.merge(merged_tuples_r);
  mway_merge_s.merge(merged_tuples_s);

  merged_r.tuples = merged_tuples_r.data();
  merged_r.num_tuples = output_size_r;
  merged_s.tuples = merged_tuples_s.data();
  merged_s.num_tuples = output_size_s;
}

[[maybe_unused]] void join_phase(Relation& merged_r, Relation& merged_s, ThreadInfo& thread_info) {
  const auto results =
      join_per_hash(std::span(merged_r.tuples, merged_r.num_tuples), std::span(merged_s.tuples, merged_s.num_tuples));
  thread_info.results = results;
}

void join_thread(ThreadInfo& thread_info, std::barrier<>& sync_point [[maybe_unused]],
                 size_t thread_count [[maybe_unused]]) {
  const auto tid = thread_info.tid;
  sync_point.arrive_and_wait();
  if (tid == 0) {
    thread_info.start = std::chrono::high_resolution_clock::now();
    start_timer(&thread_info.part);
    start_timer(&thread_info.sort);
    start_timer(&thread_info.merge);
    start_timer(&thread_info.join);
  }

  // Phase 1: Partition
  auto buckets = simd_vector<Relation>(CLUSTER_COUNT * 2);
  auto r_buckets = std::span(buckets.data(), CLUSTER_COUNT);
  auto s_buckets = std::span(buckets.data() + CLUSTER_COUNT, CLUSTER_COUNT);
  partition_phase(r_buckets, s_buckets, thread_info);
  // partition_phase2(r_buckets, s_buckets, thread_info, thread_count);

  sync_point.arrive_and_wait();
  if (tid == 0) {
    stop_timer(&thread_info.part);
  }

  // Phase 2: Sorting local partitions
  sorting_phase(r_buckets, s_buckets, thread_info);
  sync_point.arrive_and_wait();
  if (tid == 0) {
    stop_timer(&thread_info.sort);
  }

  // Phase 3: Multiway merging
  auto merged_tuples_r = simd_vector<SimdElement>{};
  auto merged_tuples_s = simd_vector<SimdElement>{};

  auto merged_r = Relation{};
  auto merged_s = Relation{};

  mwaymerge_phase(merged_tuples_r, merged_tuples_s, merged_r, merged_s, thread_info, thread_count);
  sync_point.arrive_and_wait();
  if (tid == 0) {
    stop_timer(&thread_info.merge);
  }

  // Phase 4: Join
  join_phase(merged_r, merged_s, thread_info);
  sync_point.arrive_and_wait();

  if (tid == 0) {
    stop_timer(&thread_info.join);
    thread_info.end = std::chrono::high_resolution_clock::now();
  }
}

size_t simd_sort_merge_join(Relation* relation_r, Relation* relation_s) {
  const auto thread_count = Hyrise::get().topology.num_cpus();

  // Allocate temporary memory for radix partitions.
  auto temp_partition_r =
      simd_vector<SimdElement>(relation_r->num_tuples + relation_padding(thread_count, CLUSTER_COUNT));
  auto temp_partition_s =
      simd_vector<SimdElement>(relation_s->num_tuples + relation_padding(thread_count, CLUSTER_COUNT));
  // Allocate temporary memory for sorting.
  auto temp_sorting_r =
      simd_vector<SimdElement>(relation_r->num_tuples + relation_padding(thread_count, CLUSTER_COUNT));
  auto temp_sorting_s =
      simd_vector<SimdElement>(relation_s->num_tuples + relation_padding(thread_count, CLUSTER_COUNT));

  auto thread_chunks = std::vector<simd_vector<RelationPair>>(thread_count);

  auto count_per_thread = std::array<size_t, 2>{};
  count_per_thread[0] = relation_r->num_tuples / thread_count;
  count_per_thread[1] = relation_s->num_tuples / thread_count;

  auto thread_infos = std::vector<ThreadInfo>(thread_count);

  // auto threads = std::vector<std::shared_ptr<AbstractTask>>{};
  // threads.reserve(thread_count);
  std::vector<std::jthread> threads;
  threads.reserve(thread_count);

  auto sync_point = std::barrier(static_cast<std::ptrdiff_t>(thread_count));

  for (auto thread_id = size_t{0}; thread_id < thread_count; ++thread_id) {
    auto& info = thread_infos[thread_id];
    info.tid = thread_id;
    info.rel_r = relation_r->tuples + thread_id * count_per_thread[0];
    info.rel_s = relation_s->tuples + thread_id * count_per_thread[1];
    info.tmp_part_r = temp_partition_r.data() + thread_id * (count_per_thread[0] + cache_line_padding(CLUSTER_COUNT));
    info.tmp_part_s = temp_partition_s.data() + thread_id * (count_per_thread[1] + cache_line_padding(CLUSTER_COUNT));

    info.tmp_sort_r = temp_sorting_r.data() + thread_id * (count_per_thread[0]);
    info.tmp_sort_s = temp_sorting_s.data() + thread_id * (count_per_thread[1]);

    info.num_r = (thread_id == (thread_count - 1)) ? (relation_r->num_tuples - (thread_id * count_per_thread[0]))
                                                   : count_per_thread[0];
    info.num_s = (thread_id == (thread_count - 1)) ? (relation_s->num_tuples - (thread_id * count_per_thread[1]))
                                                   : count_per_thread[1];

    info.thread_chunks = &thread_chunks;

    auto work = [&, thread_id]() {
      join_thread(thread_infos[thread_id], sync_point, thread_count);
    };
    threads.emplace_back(work);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  const auto total_results =
      std::accumulate(thread_infos.begin(), thread_infos.end(), size_t{0}, [](size_t sum, auto& info) {
        return sum + info.results;
      });

  auto& info = thread_infos[0];

  auto us_time = std::chrono::duration_cast<std::chrono::microseconds>(info.end - info.start);

  const auto total = info.join;
  std::cout << "Total, Partitioning, Sort, Merge, Join" << "\n";
  std::cout << total << ", " << info.part << ", " << info.sort << ", " << info.merge << ", " << info.join << '\n';
  std::cout << "Per-stage: " << info.part << ", " << (info.sort - info.part) << ", " << (info.merge - info.sort) << ", "
            << (info.join - info.merge) << '\n';

  std::cout << "TOTAL-TIME-USECS: " << us_time << '\n';
  std::cout << "[INFO] Results: " << total_results << '\n';

  return 0;
}

}  // namespace

int main(int argc [[maybe_unused]], char** argv) {
  const uint64_t r_size = {1'600'000'000};
  const uint64_t s_size = uint64_t{1'600'000'000} * atoi(argv[1]);
  // const auto r_size = 16000000;
  // const auto s_size = 16000000;

  const size_t num_threads = atoi(argv[2]);
  CLUSTER_COUNT = num_threads;
  L3_CACHE_SIZE = atoi(argv[3]);

  std::cout << "Create R relation." << '\n';

  auto r_tuples = simd_sort::simd_vector<SimdElement>(r_size + relation_padding(num_threads, CLUSTER_COUNT));
  auto r_relation = Relation{.tuples = r_tuples.data(), .num_tuples = r_size};
  seed_generator(12345);
  create_relation(&r_relation, r_size, num_threads, r_size);

  std::cout << "R created." << '\n';

  std::cout << "Create S relation." << '\n';

  auto s_tuples = simd_sort::simd_vector<SimdElement>(s_size + relation_padding(num_threads, CLUSTER_COUNT));
  auto s_relation = Relation{.tuples = s_tuples.data(), .num_tuples = s_size};
  seed_generator(54321);
  create_relation(&s_relation, s_size, num_threads, r_size);

  std::cout << "S created." << '\n';

  Hyrise::get().topology.use_default_topology(num_threads);
  const auto scheduler = std::make_shared<NodeQueueScheduler>();
  Hyrise::get().set_scheduler(scheduler);

  std::cout << "Run SMJ" << '\n';
  simd_sort_merge_join(&r_relation, &s_relation);

  Hyrise::get().set_scheduler(std::make_shared<ImmediateExecutionScheduler>());

  return 0;
}
