#include <algorithm>
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
#include "operators/join_simd_sort_merge/multiway_merging.hpp"
#include "operators/join_simd_sort_merge/radix_partitioning_balkesen.hpp"
#include "operators/join_simd_sort_merge/simd_sort.hpp"
#include "operators/join_simd_sort_merge/simd_utils.hpp"
#include "operators/join_simd_sort_merge/util.hpp"
#include "scheduler/immediate_execution_scheduler.hpp"
#include "scheduler/node_queue_scheduler.hpp"
#include "types.hpp"

using namespace hyrise;  // NOLINT(build/namespaces)

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

struct relation_t {
  SimdElement* tuples;
  uint64_t num_tuples;
};

struct create_arg_t {
  relation_t rel;
  int64_t firstkey;
  int64_t maxid;
  relation_t* fullrel;
  volatile void* locks;
  pthread_barrier_t* barrier;
};

typedef struct create_arg_t create_arg_t;

void* random_unique_gen_thread(void* args) {
  create_arg_t* arg = (create_arg_t*)args;
  relation_t* rel = &arg->rel;
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
  relation_t* fullrel = arg->fullrel;

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

void create_relation(relation_t* relation, uint64_t num_tuples, uint32_t nthreads, uint64_t maxid) {
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

using ColumnType = uint32_t;
using SortingType = int64_t;

static constexpr auto JOB_SPAWN_THRESHOLD = 500;
static constexpr auto CLUSTER_COUNT = 256;

namespace {

using SimdElementList = simd_sort::simd_vector<SimdElement>;
using namespace hyrise::simd_sort;
using namespace hyrise::radix_partition;

constexpr std::size_t choose_count_per_vector() {
#if defined(__AVX512F__)
  return 8;
#else
  return 4;
#endif
}

template <typename T>
std::vector<std::span<T>> split_into_chunks(std::span<T> vec, size_t chunk_count) {
  std::vector<std::span<T>> chunks;
  if (chunk_count == 0 || vec.empty())
    return chunks;

  size_t chunk_size = vec.size() / chunk_count;
  size_t remainder = vec.size() % chunk_count;
  size_t offset = 0;

  for (size_t i = 0; i < chunk_count; ++i) {
    size_t current_chunk_size = chunk_size + (i < remainder ? 1 : 0);
    chunks.emplace_back(vec.data() + offset, current_chunk_size);
    offset += current_chunk_size;
  }

  return chunks;
}

std::vector<SimdElementList> sort_relation(std::span<SimdElement> relation) {
  const auto num_cpus = Hyrise::get().topology.num_cpus();

  constexpr auto MIN_PARTITION_ELEMENTS = 1048576;
  const auto element_count = relation.size();

  const auto chunk_count = (element_count <= MIN_PARTITION_ELEMENTS) ? 1 : num_cpus;
  auto chunks = split_into_chunks(relation, chunk_count);

  auto partition_storage = std::vector<SimdElementList>(chunk_count);
  auto working_memory = std::vector<SimdElementList>(chunk_count);

  auto sort_bucket = [chunk_count](size_t bucket_index, RadixPartitionBalkesen<ColumnType>& radix_partition,
                                   SimdElementList& chunk_working_memory) {
    auto& bucket = radix_partition.bucket(bucket_index);
    if (bucket.empty()) {
      return;
    }
    const auto count_per_vector = choose_count_per_vector();
    auto* input_pointer = bucket.template begin<SortingType>();
    auto* output_pointer = radix_partition.template get_working_memory<SortingType>(bucket_index, chunk_working_memory);

    DebugAssert((simd_sort::is_simd_aligned<SortingType, 64>(input_pointer)), "Input not cache aligned.");
    DebugAssert((simd_sort::is_simd_aligned<SortingType, 64>(output_pointer)), "Output not cache aligned.");

    if (chunk_count == 1) {
      simd_sort::sort<count_per_vector, SortingType, ExecutionStrategy::PARALLEL>(input_pointer, output_pointer,
                                                                                  bucket.size);
    } else {
      simd_sort::sort<count_per_vector, SortingType, ExecutionStrategy::SEQUENTIAL>(input_pointer, output_pointer,
                                                                                    bucket.size);
    }

    bucket.data = reinterpret_cast<SimdElement*>(output_pointer);
  };

  auto chunk_partitions = std::vector<RadixPartitionBalkesen<ColumnType>>{};
  chunk_partitions.reserve(chunk_count);

  auto partition_and_sort_chunk = [&](size_t chunk_index) {
    // First, we partition the chunk
    auto& chunk_working_memory = working_memory[chunk_index];
    auto& radix_partition = chunk_partitions[chunk_index];
    radix_partition.execute(partition_storage[chunk_index], chunk_working_memory);

    auto sort_tasks = std::vector<std::shared_ptr<AbstractTask>>{};
    sort_tasks.reserve(CLUSTER_COUNT);
    for (auto bucket_index = size_t{0}; bucket_index < CLUSTER_COUNT; ++bucket_index) {
      if (radix_partition.bucket(bucket_index).size > JOB_SPAWN_THRESHOLD) {
        sort_tasks.emplace_back(std::make_shared<JobTask>([&, bucket_index]() {
          sort_bucket(bucket_index, radix_partition, chunk_working_memory);
        }));

      } else {
        sort_bucket(bucket_index, radix_partition, chunk_working_memory);
      }
    }
    Hyrise::get().scheduler()->schedule_and_wait_for_tasks(sort_tasks);
  };

  auto jobs = std::vector<std::shared_ptr<AbstractTask>>{};
  for (auto chunk_index = size_t{0}; chunk_index < chunk_count; ++chunk_index) {
    auto& chunk = chunks[chunk_index];
    chunk_partitions.emplace_back(chunk, CLUSTER_COUNT);
    jobs.push_back(std::make_shared<JobTask>([&partition_and_sort_chunk, chunk_index]() {
      partition_and_sort_chunk(chunk_index);
    }));
  }
  Hyrise::get().scheduler()->schedule_and_wait_for_tasks(jobs);

  auto sorted_clusters = std::vector<SimdElementList>(CLUSTER_COUNT);

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
  for (auto bucket_index = size_t{0}; bucket_index < CLUSTER_COUNT; ++bucket_index) {
    jobs.push_back(std::make_shared<JobTask>([&multiway_merge_buckets, bucket_index] {
      multiway_merge_buckets(bucket_index);
    }));
  }
  Hyrise::get().scheduler()->schedule_and_wait_for_tasks(jobs);

  return sorted_clusters;
}

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

size_t perform_join(std::vector<SimdElementList>& sorted_clusters_left,
                    std::vector<SimdElementList>& sorted_clusters_right) {
  auto matches = size_t{0};
  auto matches_mutex = std::mutex{};

  auto jobs = std::vector<std::shared_ptr<AbstractTask>>{};
  for (auto cluster_index = size_t{0}; cluster_index < CLUSTER_COUNT; ++cluster_index) {
    const auto merge_row_count =
        sorted_clusters_left[cluster_index].size() + sorted_clusters_right[cluster_index].size();
    if (merge_row_count > JOB_SPAWN_THRESHOLD) {
      jobs.push_back(std::make_shared<JobTask>([&, cluster_index]() {
        auto partial_matches = join_per_hash(sorted_clusters_left[cluster_index], sorted_clusters_right[cluster_index]);
        const auto lock = std::lock_guard<std::mutex>(matches_mutex);
        matches += partial_matches;
      }));
    } else {
      matches += join_per_hash(sorted_clusters_left[cluster_index], sorted_clusters_right[cluster_index]);
    }
  }
  Hyrise::get().scheduler()->schedule_and_wait_for_tasks(jobs);
  return matches;
}

size_t simd_sort_merge_join(relation_t* relation_r, relation_t* relation_s) {
  auto sorted_clusters_left = sort_relation(std::span(relation_r->tuples, relation_r->num_tuples));
  auto sorted_clusters_right = sort_relation(std::span(relation_s->tuples, relation_s->num_tuples));
  auto matches = perform_join(sorted_clusters_left, sorted_clusters_right);
  return matches;
}

}  // namespace

int main() {
  const auto r_size = 1024;
  const auto s_size = 1024;
  const auto num_threads = 16;

  std::cout << "Create R rleation." << std::endl;

  auto r_tuples = simd_sort::simd_vector<SimdElement>(r_size);
  auto r_relation = relation_t{.tuples = r_tuples.data(), .num_tuples = r_size};
  seed_generator(12345);
  create_relation(&r_relation, r_size, num_threads, r_size);

  std::cout << "R created." << std::endl;

  std::cout << "Create S rleation." << std::endl;

  auto s_tuples = simd_sort::simd_vector<SimdElement>(s_size);
  auto s_relation = relation_t{.tuples = s_tuples.data(), .num_tuples = s_size};
  seed_generator(54321);
  create_relation(&s_relation, s_size, num_threads, r_size);

  std::cout << "S created." << std::endl;

  Hyrise::get().topology.use_default_topology(num_threads);
  const auto scheduler = std::make_shared<NodeQueueScheduler>();
  Hyrise::get().set_scheduler(scheduler);

  auto start = std::chrono::high_resolution_clock::now();
  auto matches = simd_sort_merge_join(&r_relation, &s_relation);
  auto end = std::chrono::high_resolution_clock::now();
  auto join_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "TOTAL-TIME-USECS: " << join_time << std::endl;

  Hyrise::get().set_scheduler(std::make_shared<ImmediateExecutionScheduler>());
  std::cout << "Results = " << matches << "." << std::endl;

  return 0;
}
