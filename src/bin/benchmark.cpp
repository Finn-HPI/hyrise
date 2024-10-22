#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <utils/assert.hpp>

#include "cxxopts.hpp"

#include "operators/join_simd_sort_merge/simd_sort.hpp"

template <typename T>
auto get_uniform_distribution(T min, T max) {
  if constexpr (std::is_same_v<T, double>) {
    return std::uniform_real_distribution<double>(min, max);
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return std::uniform_int_distribution<int64_t>(min, max);
  } else {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, int64_t>, "Unsupported type for uniform distribution");
  }
}

template <std::size_t count_per_vector, typename KeyType>
void benchmark(const std::size_t scale, const std::size_t num_warumup_runs, const std::size_t num_runs,
               std::ofstream& out) {
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  using hyrise::simd_sort::simd_sort;
  using hyrise::simd_sort::simd_vector;

  std::cout << "Running benchmark for scale: " << scale << " ..." << std::endl;

  std::random_device rnd;
  std::mt19937 gen(rnd());
  auto dis = get_uniform_distribution<KeyType>(0, std::numeric_limits<KeyType>::max());

  const auto base_num_items = 1'048'576;  // 2^20
  const auto num_items = base_num_items * scale;

  std::cout << "num_items: " << num_items << std::endl;

  auto data = simd_vector<KeyType>(num_items);
  auto data_std_sort = simd_vector<KeyType>(num_items);
  auto data_simd_sort = simd_vector<KeyType>(num_items);
  auto output_simd_sort = simd_vector<KeyType>(num_items);

  for (auto& val : data) {
    val = dis(gen);
  }
  std::cout << "start execution" << std::endl;
  Assert(!std::ranges::is_sorted(data), "Data has to be 32-byte aligend.");

  std::vector<uint64_t> runtimes_std_sort;
  std::vector<uint64_t> runtimes_simd_sort;
  runtimes_std_sort.reserve(num_runs);
  runtimes_simd_sort.reserve(num_runs);

  auto* input_ptr = data_simd_sort.data();
  auto* output_ptr = output_simd_sort.data();

  const auto num_total_runs = num_warumup_runs + num_runs;
  for (std::size_t run_index = 0; run_index < num_total_runs; ++run_index) {
    std::cout << "run: " << run_index << std::endl;
    for (auto index = std::size_t{0}; index < num_items; ++index) {
      data_std_sort[index] = data[index];
      data_simd_sort[index] = data[index];
    }
    std::cout << "start std::sort" << std::endl;

    for (volatile auto& elem : data_std_sort) {
      (void)elem;  // Prevents unused variable warning
                   // Access element to bring it into cache
    }
    //////////////////////////////
    /// START TIMING std::sort ///
    //////////////////////////////

    auto start_std_sort = std::chrono::steady_clock::now();
    std::sort(data_std_sort.begin(), data_std_sort.end());
    auto end_std_sort = std::chrono::steady_clock::now();

    // NOLINTNEXTLINE
    asm volatile("" : : "r"(data_std_sort.data()) : "memory");  // Ensures std::sort is not optimized away

    /////////////////////////////
    /// END TIMING std::sort  ///
    /////////////////////////////

    const uint64_t runtime_std_sort =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_std_sort - start_std_sort).count();

    input_ptr = data_simd_sort.data();
    output_ptr = output_simd_sort.data();

    std::cout << "start simd_sort" << std::endl;

    for (volatile auto& elem : data_std_sort) {
      (void)elem;  // Prevents unused variable warning
                   // Access element to bring it into cache
    }

    //////////////////////////////
    /// START TIMING simd_sort ///
    //////////////////////////////

    auto start_simd_sort = std::chrono::steady_clock::now();
    simd_sort<count_per_vector>(input_ptr, output_ptr, num_items);
    auto end_simd_sort = std::chrono::steady_clock::now();

    /////////////////////////////
    /// END TIMING simd_sort  ///
    /////////////////////////////

    const uint64_t runtime_simd_sort =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_simd_sort - start_simd_sort).count();

    auto& sorted_data = (output_ptr == output_simd_sort.data()) ? output_simd_sort : data_simd_sort;
    Assert(std::ranges::is_sorted(sorted_data), "The simd_sort did not produce a sorted result");
    Assert(sorted_data == data_std_sort, "Ouput of simd_sort is not the same as std::sort.");

    if (run_index < num_warumup_runs) {
      // Skip warm-up runs
      continue;
    }
    runtimes_std_sort.push_back(runtime_std_sort);
    runtimes_simd_sort.push_back(runtime_simd_sort);
  }

  const auto total_duration_std_sort = std::accumulate(runtimes_std_sort.begin(), runtimes_std_sort.end(), 0ul);
  const double avg_duration_std_sort =
      static_cast<double>(total_duration_std_sort) / static_cast<double>(runtimes_std_sort.size());

  const auto total_duration_simd_sort = std::accumulate(runtimes_simd_sort.begin(), runtimes_simd_sort.end(), 0ul);
  const double avg_duration_simd_sort =
      static_cast<double>(total_duration_simd_sort) / static_cast<double>(runtimes_simd_sort.size());
  const auto speed_up = static_cast<double>(avg_duration_std_sort) / static_cast<double>(avg_duration_simd_sort);
  out << scale << "," << avg_duration_std_sort << "," << avg_duration_simd_sort << "," << speed_up << std::endl;
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("SIMDSort", "A single-threaded simd_sort benchmark.");
  // clang-format off
  options.add_options()
  ("c,cpr", "element count per simd vector", cxxopts::value<std::size_t>()->default_value("4"))
  ("t,dt", "element data type", cxxopts::value<std::string>()->default_value("double"))
  ("w,warmup", "number of warmup runs", cxxopts::value<std::size_t>()->default_value("1"))
  ("r,runs", "number of runs", cxxopts::value<std::size_t>()->default_value("5"))
  ("o,output", "Output file name", cxxopts::value<std::string>()->default_value("benchmark.csv"))
  ("h,help", "Print usage");
  // clang-format on
  auto result = options.parse(argc, argv);
  if (result.count("help") != 0u) {
    std::cout << options.help() << std::endl;
    return 0;
  }
  const auto count_per_vector = result["cpr"].as<std::size_t>();  // 64-bit elements with AVX2.
  const auto output_path = result["output"].as<std::string>();
  std::cout << "[Configuration] cpr: " << count_per_vector << std::endl;
  std::cout << "L2_CACHE_SIZE: " << L2_CACHE_SIZE << std::endl;

  auto output_file = std::ofstream(output_path);
  if (!output_file.is_open()) {
    std::cerr << "Error: Could not open the file!" << std::endl;
    return 1;
  }
  const auto key_type = result["dt"].as<std::string>();
  const auto num_warumup_runs = result["warmup"].as<std::size_t>();
  const auto num_runs = result["runs"].as<std::size_t>();

  if (count_per_vector == 4) {
    if (key_type == "double") {
      for (auto scale = std::size_t{1}; scale <= 256; scale *= 2) {
        benchmark<4, double>(scale, num_warumup_runs, num_runs, output_file);
      }
    } else if (key_type == "int64_t") {
      for (auto scale = std::size_t{1}; scale <= 256; scale *= 2) {
        benchmark<4, int64_t>(scale, num_warumup_runs, num_runs, output_file);
      }
    }
  } else if (count_per_vector == 2) {
    if (key_type == "double") {
      for (auto scale = std::size_t{1}; scale <= 256; scale *= 2) {
        benchmark<2, double>(scale, num_warumup_runs, num_runs, output_file);
      }
    } else if (key_type == "int64_t") {
      for (auto scale = std::size_t{1}; scale <= 256; scale *= 2) {
        benchmark<2, int64_t>(scale, num_warumup_runs, num_runs, output_file);
      }
    }
  } else {
    assert(false && "benchmark not implemented");
  }

  return 0;
}
