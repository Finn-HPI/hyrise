#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <span>

#include <boost/range/numeric.hpp>
#include <boost/sort/pdqsort/pdqsort.hpp>

#include "hyrise.hpp"
#include "operators/join_simd_sort_merge/k_way_merge.hpp"
#include "operators/join_simd_sort_merge/multiway_merging.hpp"
#include "operators/join_simd_sort_merge/radix_partitioning.hpp"
#include "operators/join_simd_sort_merge/simd_sort.hpp"
#include "operators/join_simd_sort_merge/simd_utils.hpp"
#include "operators/join_simd_sort_merge/util.hpp"
#include "scheduler/immediate_execution_scheduler.hpp"
#include "scheduler/node_queue_scheduler.hpp"
#include "types.hpp"

using namespace hyrise;  // NOLINT(build/namespaces)

namespace {
template <class Tp>
[[maybe_unused]] inline __attribute__((always_inline)) void do_not_optimize_away(Tp const& value) {
  asm volatile("" : : "r,m"(value) : "memory");  // NOLINT
}

[[maybe_unused]] constexpr std::size_t choose_count_per_vector() {
#if defined(__AVX512F__)
  return 8;
#else
  return 4;
#endif
}

template <typename T>
radix_partition::RadixPartition<T> sort_chunk(std::span<SimdElement> elements, const size_t num_buckets) {
  auto partition = radix_partition::RadixPartition<T>(elements, num_buckets);
  auto partitioning_data = simd_sort::simd_vector<SimdElement>{};
  auto working_data = simd_sort::simd_vector<SimdElement>{};

  partition.execute(partitioning_data, working_data);

  for (auto bucket_index = size_t{0}; bucket_index < num_buckets; ++bucket_index) {
    auto& bucket = partition.bucket(bucket_index);
    if (!bucket.size) {
      continue;
    }
    const auto count_per_vector = choose_count_per_vector();
    auto* input_pointer = bucket.template begin<T>();
    auto* output_pointer = partition.template get_working_memory<T>(bucket_index, working_data);

    simd_sort::sort<count_per_vector, T, ExecutionStrategy::PARALLEL>(input_pointer, output_pointer, bucket.size);
    bucket.data = reinterpret_cast<SimdElement*>(output_pointer);
  }
  return partition;
}

template <typename T>
void benchmark(size_t scale, size_t cores, std::ofstream& out) {
  std::mt19937 gen(42);
  auto dist = std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint32_t>::max());


  auto base_size = 1'048'576;
  Hyrise::get().topology.use_default_topology(cores);
  const auto scheduler = std::make_shared<NodeQueueScheduler>();
  Hyrise::get().set_scheduler(scheduler);

  auto elements = simd_sort::simd_vector<SimdElement>(base_size * scale);
  auto rid = size_t{0};
  for (auto& element : elements) {
    element.key = dist(gen);
    element.index = rid;
    ++rid;
  }

  std::cout << "element count: " << elements.size() << std::endl;

  const auto warmup_runs = 1;
  const auto runs = 6;
  const auto total_runs = warmup_runs + runs;

  for (auto it = size_t{0}; it < total_runs; ++it) {
    auto start_sort = std::chrono::high_resolution_clock::now();
    auto chunks = std::vector<std::span<SimdElement>>(cores);
    const auto chunk_size = elements.size() / cores;
    for (auto chunk_index = size_t{0}; chunk_index < cores; ++chunk_index) {
      chunks[chunk_index] = std::span<SimdElement>(elements.data() + (chunk_index * chunk_size), chunk_size);
    }

    auto tasks = std::vector<std::shared_ptr<AbstractTask>>{};
    tasks.reserve(cores);

    auto partitions = std::vector<radix_partition::RadixPartition<T>>(cores);

    auto index = size_t{0};
    for (auto& chunk : chunks) {
      tasks.push_back(std::make_shared<JobTask>([chunk, index, cores, &partitions]() {
        partitions[index] = std::move(sort_chunk<T>(chunk, cores));
      }));
      ++index;
    }

    Hyrise::get().scheduler()->schedule_and_wait_for_tasks(tasks);

    // Merge individual buckets with same index.
    auto bucket_merge_tasks = std::vector<std::shared_ptr<AbstractTask>>{};
    bucket_merge_tasks.reserve(cores);
    for (auto bucket_index = size_t{0}; bucket_index < cores; ++bucket_index) {
      bucket_merge_tasks.push_back(std::make_shared<JobTask>([&, bucket_index]() {
        auto sorted_buckets = std::vector<std::unique_ptr<radix_partition::Bucket>>{};
        sorted_buckets.reserve(partitions.size());
	for (auto& partition : partitions) {
            auto& bucket = partition.bucket(bucket_index);
            sorted_buckets.push_back(std::make_unique<radix_partition::Bucket>(bucket));
        }
	// auto merger =  k_way_merge::KWayMerge<T>(sorted_buckets);
        auto merger =  multiway_merging::MultiwayMerger<choose_count_per_vector(), T>(sorted_buckets);
	//auto merger2 = multiway_merging::MultiwayMerger<choose_count_per_vector(), T>(std::span(sorted_buckets.begin() + sorted_buckets.size() /2, sorted_buckets/2));
        do_not_optimize_away(merger.merge());
      }));
    }

    Hyrise::get().scheduler()->schedule_and_wait_for_tasks(bucket_merge_tasks);

    auto stop_sort = std::chrono::high_resolution_clock::now();
    if (it < warmup_runs) {
      continue;
    }

    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(stop_sort - start_sort).count();

  auto me_s = (static_cast<double>(elements.size()) / (static_cast<double>(time) / 1'000)) / 1'000'000;

  std::cout <<"scale: " << scale << ", time: " << time << " (ms), ME/s: " << me_s << std::endl;
  out << scale << "," << it << "," << time << "," << me_s << std::endl;

  }
}

}  // namespace

int main() {
  auto cores = size_t{64};
  std::string file_name = std::to_string(cores) + "_throughput_test.csv";
  std::ofstream out;
  out.open(file_name, std::ios::out | std::ios::trunc);

  out << "scale,run,time,me_s" << std::endl;
  for (auto scale = size_t{1024}; scale <= 1024; scale *=2) {
	  benchmark<double>(scale, cores, out);
  }

  Hyrise::get().set_scheduler(std::make_shared<ImmediateExecutionScheduler>());

  return 0;
}

