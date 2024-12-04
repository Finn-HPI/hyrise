#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <span>

#include <boost/range/numeric.hpp>

#include "operators/join_simd_sort_merge/merge_path.hpp"
#include "operators/join_simd_sort_merge/simd_sort.hpp"
#include "operators/join_simd_sort_merge/simd_utils.hpp"
#include "operators/join_simd_sort_merge/util.hpp"
#include "scheduler/immediate_execution_scheduler.hpp"
#include "scheduler/node_queue_scheduler.hpp"
#include "types.hpp"
#include "utils/assert.hpp"

using namespace hyrise;  // NOLINT(build/namespaces)

namespace {
template <class Tp>
inline __attribute__((always_inline)) void do_not_optimize_away(Tp const& value) {
  asm volatile("" : : "r,m"(value) : "memory");  // NOLINT
}

template <typename T>
auto uniform_distribution() {
  if constexpr (std::is_integral_v<T>) {
    return std::uniform_int_distribution<T>(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());
  } else if constexpr (std::is_floating_point_v<T>) {
    return std::uniform_real_distribution<T>(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());
  }
}

template <typename T>
void generate_leaf(std::span<SimdElement> range, std::mt19937& gen) {
  auto dist = uniform_distribution<T>();
  // fill with random values of type T
  for (auto& elem : range) {
    auto value = dist(gen);
    elem = *reinterpret_cast<SimdElement*>(&value);
  }
  std::sort(range.begin(), range.end(), [](auto& lhs, auto& rhs) {
    return *reinterpret_cast<T*>(&lhs) < *reinterpret_cast<T*>(&rhs);
  });
}

constexpr size_t count_per_vector() {
#ifdef __AVX512F__
  return 8;
#else
  return 4;
#endif
}

template <typename T>
T* merge_using_simd_merge(std::vector<simd_sort::DataChunk<T>>& chunk_list) {
  auto chunk_count = chunk_list.size();
  while (chunk_count > 1) {
    chunk_count = simd_sort::merge_chunk_list<count_per_vector(), T>(chunk_list, chunk_count);
  }
  return chunk_list.front().input;
}

enum class Option : std::uint8_t { Seq, Par, ParMergePath };

// leaf_size is required to be a multiple of 64 due to alignment assumptions.
template <typename T, Option option>
void benchmark(size_t scale, std::ofstream& result, size_t cores = 1) {
  const auto base_num_items = 1'048'576;  // 2^20
  constexpr auto BLOCK_SIZE = hyrise::simd_sort::block_size<T>();

  const auto num_items = scale * base_num_items;

  const auto leaf_count = num_items / BLOCK_SIZE;
  const auto leaf_size = BLOCK_SIZE;

  const auto num_warmup_runs = size_t{2};
  const auto num_iterations = size_t{4};

  auto times = std::vector<size_t>{};
  times.reserve(num_iterations);

  for (auto it = size_t{0}; it < num_warmup_runs + num_iterations; ++it) {
    std::cout << "iteration: " << it << std::endl;
    std::mt19937 gen(42);

    // Create input and output vector of size leaf_count * leaf_size
    const auto total_size = leaf_count * leaf_size;
    auto input_simd_merge = simd_sort::simd_vector<SimdElement>(total_size);
    auto output_simd_merge = simd_sort::simd_vector<SimdElement>(total_size);

    auto chunk_list = std::vector<simd_sort::DataChunk<T>>(leaf_count);

    for (auto offset = size_t{0}, leaf_index = size_t{0}; offset < total_size; offset += leaf_size, ++leaf_index) {
      auto* input_begin = input_simd_merge.data() + offset;
      generate_leaf<T>(std::span(input_begin, leaf_size), gen);

      // Create chunk for SIMD Merging.
      auto& chunk = chunk_list[leaf_index];
      chunk.input = reinterpret_cast<T*>(input_begin);
      chunk.output = reinterpret_cast<T*>(output_simd_merge.data() + offset);
      chunk.size = leaf_size;
    }

    auto execution_time = size_t{};
    if constexpr (option == Option::Seq) {
      auto start_simd = std::chrono::high_resolution_clock::now();
      auto* output = merge_using_simd_merge(chunk_list);
      auto end_simd = std::chrono::high_resolution_clock::now();
      do_not_optimize_away(output);
      execution_time = duration_cast<std::chrono::nanoseconds>(end_simd - start_simd).count();

    } else if constexpr (option == Option::Par) {
      auto start_simd = std::chrono::high_resolution_clock::now();
      auto* output = simd_merge_parallel<count_per_vector(), false, T>(chunk_list, cores);
      auto end_simd = std::chrono::high_resolution_clock::now();
      do_not_optimize_away(output);
      execution_time = duration_cast<std::chrono::nanoseconds>(end_simd - start_simd).count();

    } else {
      auto start_simd = std::chrono::high_resolution_clock::now();
      auto* output = simd_merge_parallel<count_per_vector(), true, T>(chunk_list, cores);
      auto end_simd = std::chrono::high_resolution_clock::now();
      do_not_optimize_away(output);
      execution_time = duration_cast<std::chrono::nanoseconds>(end_simd - start_simd).count();
    }

    if (it < num_warmup_runs) {
      continue;
    }

    times.push_back(execution_time);
  }

  const auto total_duration = std::accumulate(times.begin(), times.end(), 0ul);
  const auto avg_time = total_duration / num_iterations;

  result << scale << "," << avg_time << std::endl;
}

void run_sequential() {
  std::string file_name = "seq.csv";
  std::ofstream file;
  file.open(file_name, std::ios::app);

  // Check if file opened successfully
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << file_name << '\n';
    return;
  }

  file << "scale,time" << '\n';
  for (auto scale = size_t{1}; scale <= 256; scale *= 2) {
    benchmark<int64_t, Option::Seq>(scale, file);
  }
  file.close();
}

}  // namespace

int main() {
  const auto world = pmr_string{"Experiment with SIMD Merge "};
  std::cout << "Playround: " << world << "!\n";

  run_sequential();

  std::cout << "count_per_vector: " << count_per_vector() << std::endl;

  const auto max_cores = 8;
  for (auto cores = size_t{2}; cores <= max_cores; cores *= 2) {
    std::string file_name_par = std::to_string(cores) + "_par.csv";
    std::ofstream file_par;
    file_par.open(file_name_par, std::ios::app);

    std::string file_name_par_mp = std::to_string(cores) + "_par_mp.csv";
    std::ofstream file_par_mp;
    file_par.open(file_name_par_mp, std::ios::app);

    // Check if file opened successfully
    if (!file_par.is_open()) {
      std::cerr << "Error: Could not open file " << file_name_par << '\n';
      return -1;
    }

    // Check if file opened successfully
    if (!file_par_mp.is_open()) {
      std::cerr << "Error: Could not open file " << file_name_par_mp << '\n';
      return -1;
    }

    file_par << "scale,time" << std::endl;
    file_par_mp << "scale,time" << std::endl;

    Hyrise::get().topology.use_default_topology(cores);
    std::cout << "- Multi-threaded Topology:\n";
    std::cout << Hyrise::get().topology;

    const auto scheduler = std::make_shared<NodeQueueScheduler>();
    Hyrise::get().set_scheduler(scheduler);

    for (auto scale = size_t{1}; scale <= 256; scale *= 2) {
      benchmark<int64_t, Option::Par>(scale, file_par, cores);
      benchmark<int64_t, Option::ParMergePath>(scale, file_par, cores);
    }

    Hyrise::get().scheduler()->finish();
    file_par.close();
    file_par_mp.close();
  }

  Hyrise::get().set_scheduler(std::make_shared<ImmediateExecutionScheduler>());

  return 0;
}
