#include <benchmark/benchmark.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <span>
#include <zipfian_int_distribution.hpp>

#include <boost/sort/pdqsort/pdqsort.hpp>

#include "operators/join_simd_sort_merge/simd_utils.hpp"
#include "operators/join_simd_sort_merge/util.hpp"
#include "types.hpp"

using namespace hyrise;  // NOLINT(build/namespaces)

using SortingType = double;

namespace {

enum class CompareResult : std::uint8_t { Less, Greater, Equal };

struct PotentialMatchRange {
  PotentialMatchRange(std::size_t init_start_index, std::size_t init_end_index, std::span<SimdElement> init_elements)

      : start_index(init_start_index),
        end_index(init_end_index),
        elements(init_elements.data() + init_start_index, init_elements.data() + init_end_index) {}

  std::size_t start_index;
  std::size_t end_index;
  std::span<SimdElement> elements;

 public:
  // Executes the given action for every row id of the table in this range.
  void for_every_row_id(auto&& action) const {
    const auto num_items = elements.size();
    for (auto index = size_t{0}; index < num_items; ++index) {
      const auto row_id = static_cast<size_t>(elements[index].index);
      action(row_id);
    }
  }

  void find_matches_with_range(const PotentialMatchRange& other_range, auto&& action) const {
    this->for_every_row_id([&](const size_t left_row_id) {
      other_range.for_every_row_id([&](const size_t right_row_id) {
        action(left_row_id, right_row_id);
      });
    });
  }
};

template <typename T>
inline CompareResult __attribute__((always_inline)) compare(const T& left, const T& right) {
  if (left < right) {
    return CompareResult::Less;
  }

  if (left == right) {
    return CompareResult::Equal;
  }

  return CompareResult::Greater;
}

inline SortingType __attribute__((always_inline)) compare_value(uint32_t key) {
  return std::bit_cast<SortingType>(static_cast<uint64_t>(key) << 32u);
}

[[maybe_unused]] std::size_t __attribute__((always_inline)) align_to_cacheline(std::size_t value) {
  return (value + radix_partition::TUPLES_PER_CACHELINE - 1) & ~(radix_partition::TUPLES_PER_CACHELINE - 1);
}

[[maybe_unused]] std::pair<std::size_t, std::size_t> equal_value_range_size_binary_search(
    std::size_t start_index, std::span<SimdElement>& elements) {
  if (start_index >= elements.size()) {
    return {0, 0};
  }
  auto begin = elements.begin();
  std::advance(begin, start_index);

  auto start = begin;
  std::ranges::advance(start, 1, elements.end());

  auto comparisons = size_t{0};

  // Binary search in case the run did not end within the linearly scanned part.
  const auto binary_search_result =
      std::upper_bound(start, elements.end(), *begin, [&](const auto& lhs, const auto& rhs) {
        ++comparisons;
        return compare_value(lhs.key) < compare_value(rhs.key);
      });
  return {std::distance(begin, binary_search_result), comparisons};
}

template <size_t search_items>
[[maybe_unused]] std::pair<std::size_t, std::size_t> equal_value_range_size(std::size_t start_index,
                                                                            std::span<SimdElement>& elements) {
  if (start_index >= elements.size()) {
    return {0, 0};
  }
  auto begin = elements.begin();
  std::advance(begin, start_index);
  const auto run_value = compare_value(begin->key);

  auto start = begin;
  std::ranges::advance(start, 1, elements.end());

  constexpr auto LINEAR_SEARCH_ITEMS = search_items;

  auto end = start + LINEAR_SEARCH_ITEMS;
  if (start_index + LINEAR_SEARCH_ITEMS >= elements.size()) {
    // Set end of linear search to end of input vector if we would overshoot otherwise.
    end = elements.end();
  }

  auto comparisons = size_t{0};

  const auto linear_search_result = std::find_if(start, end, [&](const auto& simd_element) {
    ++comparisons;
    return compare_value(simd_element.key) > run_value;
  });

  if (linear_search_result != end) {
    return {std::distance(begin, linear_search_result), comparisons};
  }

  if (linear_search_result == elements.end() || compare_value(end->key) > run_value) {
    return {std::distance(begin, end), comparisons};
  }
  // Binary search in case the run did not end within the linearly scanned part.
  const auto binary_search_result = std::upper_bound(end, elements.end(), *end, [&](const auto& lhs, const auto& rhs) {
    ++comparisons;
    return compare_value(lhs.key) < compare_value(rhs.key);
  });

  return {std::distance(begin, binary_search_result), comparisons};
}

[[maybe_unused]] std::pair<std::size_t, std::size_t> equal_value_range_size_experimental_search(
    std::size_t start_index, std::span<SimdElement>& elements) {
  if (start_index >= elements.size()) {
    return {0, 0};
  }
  auto begin = elements.begin();
  const auto end = elements.end();

  std::advance(begin, start_index);
  const auto run_value = compare_value(begin->key);

  auto next = begin;
  std::ranges::advance(next, 1, end);

  if (compare_value(next->key) > run_value) {
    return {1, 1};
  }

  auto prev = next;

  auto jump = std::iterator_traits<simd_sort::simd_vector<SimdElement>::iterator>::difference_type{1};
  std::ranges::advance(next, jump, end);

  auto comparisons = size_t{0};

  while (next != end) {
    ++comparisons;
    if (compare_value(next->key) > run_value) {
      break;
    }
    prev = next;
    std::ranges::advance(next, jump, end);
    jump *= 2;
  }

  const auto start = prev;
  std::ranges::advance(prev, 1, end);

  if (next == end) {
    const auto binary_search_result = std::upper_bound(prev, end, *start, [&](const auto& lhs, const auto& rhs) {
      ++comparisons;
      return compare_value(lhs.key) < compare_value(rhs.key);
    });
    return {std::distance(begin, binary_search_result), comparisons};
  }

  std::ranges::advance(next, 1, end);

  const auto binary_search_result = std::upper_bound(prev, next, *start, [&](const auto& lhs, const auto& rhs) {
    ++comparisons;
    return compare_value(lhs.key) < compare_value(rhs.key);
  });
  return {std::distance(begin, binary_search_result), comparisons};
}

enum class KeyRangeAlgo : uint8_t { LinBin8 = 0, LinBin64 = 1, LinBin128 = 2, Bin = 3, Exp = 4 };

template <KeyRangeAlgo algo_type>
std::pair<size_t, size_t> join(std::span<SimdElement> left_elements, std::span<SimdElement> right_elements) {
  auto num_output_tuples = size_t{0};

  auto left_run_start = size_t{0};
  auto right_run_start = size_t{0};

  auto comparisons = size_t{0};

  auto advance = [&](size_t start, std::span<SimdElement> elements) {
    using enum KeyRangeAlgo;
    if constexpr (algo_type == LinBin8) {
      auto [dist, cmp] = equal_value_range_size<8>(start, elements);
      comparisons += cmp;
      return dist;
    }
    if constexpr (algo_type == LinBin64) {
      auto [dist, cmp] = equal_value_range_size<64>(start, elements);

      comparisons += cmp;
      return dist;
    }

    if constexpr (algo_type == LinBin128) {
      auto [dist, cmp] = equal_value_range_size<128>(start, elements);
      comparisons += cmp;
      return dist;
    }

    if constexpr (algo_type == Bin) {
      auto [dist, cmp] = equal_value_range_size_binary_search(start, elements);
      comparisons += cmp;

      return dist;
    }
    if constexpr (algo_type == Exp) {
      auto [dist, cmp] = equal_value_range_size_experimental_search(start, elements);
      comparisons += cmp;
      return dist;
    }
  };

  auto left_run_end = left_run_start + advance(left_run_start, left_elements);
  auto right_run_end = right_run_start + advance(right_run_start, right_elements);

  const auto left_size = left_elements.size();
  const auto right_size = right_elements.size();

  while (left_run_start < left_size && right_run_start < right_size) {
    const auto left_value = compare_value(left_elements[left_run_start].key);
    const auto right_value = compare_value(right_elements[right_run_start].key);

    const auto compare_result = compare(left_value, right_value);

    [[maybe_unused]] const auto left_range = PotentialMatchRange(left_run_start, left_run_end, left_elements);
    [[maybe_unused]] const auto right_range = PotentialMatchRange(right_run_start, right_run_end, right_elements);

    // Emit all combinations of left and right range.

    if (compare_result == CompareResult::Equal) {
      // auto count = size_t{0};
      left_range.find_matches_with_range(
          right_range, [&]([[maybe_unused]] const size_t left_row_id, [[maybe_unused]] const size_t right_row_id) {
            // ++count;
            ++num_output_tuples;
          });
      // std::cout << "count: " << count << '\n';
    }
    // Advance to the next run on the smaller side or both if equal.
    switch (compare_result) {
      case CompareResult::Equal:
        // Advance both runs.
        left_run_start = left_run_end;
        right_run_start = right_run_end;
        left_run_end = left_run_start + advance(left_run_start, left_elements);
        right_run_end = right_run_start + advance(right_run_start, right_elements);
        break;
      case CompareResult::Less:
        // Advance the left run.
        left_run_start = left_run_end;
        left_run_end = left_run_start + advance(left_run_start, left_elements);
        break;
      case CompareResult::Greater:
        // Advance the right run.
        right_run_start = right_run_end;
        right_run_end = right_run_start + advance(right_run_start, right_elements);
        break;
      default:
        throw std::logic_error("Unknown CompareResult.");
    }
  }
  return {num_output_tuples, comparisons};
}

void generate_data(simd_sort::simd_vector<SimdElement>& relation_r, simd_sort::simd_vector<SimdElement>& relation_s,
                   size_t size_r, size_t size_s, [[maybe_unused]] double overlap, [[maybe_unused]] std::mt19937& rng,
                   [[maybe_unused]] double skew, [[maybe_unused]] size_t range_size) {
  // std::uniform_int_distribution<uint32_t> dist(0, size_r-1);
  // auto dist = zipfian_int_distribution<uint32_t>(0, size_r - 1, skew);

  relation_r.clear();
  relation_r.reserve(size_r);
  auto key = uint32_t{0};
  for (auto index = uint32_t{0}; index < size_s; ++index) {
    relation_r.push_back({index, key});
    ++key;
  }

  key = 0;
  relation_s.clear();
  relation_s.reserve(size_s);
  for (auto index = uint32_t{0}; index < size_s; ++index) {
    relation_s.push_back({index, key});
    if (index > 0 && index % range_size == 0) {
      ++key;
    }
  }

  // relation_r.clear();
  // relation_r.reserve(size_r);
  // for (auto index = uint32_t{0}; index < size_s; ++index) {
  //   relation_r.push_back({index, index});
  // }
  //
  // relation_s.clear();
  // relation_s.reserve(size_s);
  // for (auto index = uint32_t{0}; index < size_s; ++index) {
  //   relation_s.push_back({index, dist(rng)});
  // }

  auto key_comp = [](SimdElement& lhs, SimdElement& rhs) {
    return *reinterpret_cast<SortingType*>(&lhs) < *reinterpret_cast<SortingType*>(&rhs);
  };

  boost::sort::pdqsort(relation_s.begin(), relation_s.end(), key_comp);
  boost::sort::pdqsort(relation_r.begin(), relation_r.end(), key_comp);
}
}  // namespace

static void BM_JoinLinBin8(benchmark::State& state) {
  const auto size_r = static_cast<size_t>(state.range(0));
  const auto size_s = static_cast<size_t>(state.range(1));
  // const auto overlap = static_cast<double>(state.range(2)) / 100.0;  // passed as percent
  const auto range_size = static_cast<size_t>(state.range(2));

  const auto overlap = double{1};
  const auto skew = static_cast<double>(state.range(2)) / 100.0;

  std::mt19937 rng(42);
  simd_sort::simd_vector<SimdElement> relation_r;
  simd_sort::simd_vector<SimdElement> relation_s;

  auto total_matches = double{0};
  auto total_comparison = double{0};

  // state.PauseTiming();  // Don't include data generation
  generate_data(relation_r, relation_s, size_r, size_s, overlap, rng, skew, range_size);
  // state.ResumeTiming();

  const auto warmup = 2;
  for (auto index = 0u; index < warmup; ++index) {
    auto [matches, cmp] = join<KeyRangeAlgo::LinBin8>(relation_r, relation_s);
    benchmark::DoNotOptimize(matches);
    benchmark::DoNotOptimize(cmp);
    benchmark::ClobberMemory();
  }

  // NOLINTNEXTLINE
  for (auto _ : state) {
    auto [matches, cmp] = join<KeyRangeAlgo::LinBin8>(relation_r, relation_s);

    // std::cout << "matches: " << matches << '\n';
    benchmark::DoNotOptimize(matches);
    benchmark::DoNotOptimize(cmp);

    benchmark::ClobberMemory();

    total_matches += static_cast<double>(matches);
    total_comparison += static_cast<double>(cmp);
  }
  total_matches /= static_cast<double>(state.iterations());
  total_comparison /= static_cast<double>(state.iterations());
  state.counters["Matches"] = total_matches;
  state.counters["Comparisons"] = total_comparison;
}

static void BM_JoinLinBin64(benchmark::State& state) {
  const auto size_r = static_cast<size_t>(state.range(0));
  const auto size_s = static_cast<size_t>(state.range(1));
  // const auto overlap = static_cast<double>(state.range(2)) / 100.0;  // passed as percent
  const auto range_size = static_cast<size_t>(state.range(2));

  const auto overlap = double{1};
  const auto skew = static_cast<double>(state.range(2)) / 100.0;

  std::mt19937 rng(42);
  simd_sort::simd_vector<SimdElement> relation_r;
  simd_sort::simd_vector<SimdElement> relation_s;

  auto total_matches = double{0};
  auto total_comparison = double{0};

  // state.PauseTiming();  // Don't include data generation
  generate_data(relation_r, relation_s, size_r, size_s, overlap, rng, skew, range_size);
  // state.ResumeTiming();

  const auto warmup = 2;
  for (auto index = 0u; index < warmup; ++index) {
    auto [matches, cmp] = join<KeyRangeAlgo::LinBin64>(relation_r, relation_s);
    benchmark::DoNotOptimize(matches);
    benchmark::DoNotOptimize(cmp);
    benchmark::ClobberMemory();
  }

  // NOLINTNEXTLINE
  for (auto _ : state) {
    auto [matches, cmp] = join<KeyRangeAlgo::LinBin64>(relation_r, relation_s);

    // std::cout << "matches: " << matches << '\n';
    benchmark::DoNotOptimize(matches);
    benchmark::DoNotOptimize(cmp);

    benchmark::ClobberMemory();

    total_matches += static_cast<double>(matches);
    total_comparison += static_cast<double>(cmp);
  }
  total_matches /= static_cast<double>(state.iterations());
  total_comparison /= static_cast<double>(state.iterations());
  state.counters["Matches"] = total_matches;
  state.counters["Comparisons"] = total_comparison;
}

static void BM_JoinLinBin128(benchmark::State& state) {
  const auto size_r = static_cast<size_t>(state.range(0));
  const auto size_s = static_cast<size_t>(state.range(1));
  // const auto overlap = static_cast<double>(state.range(2)) / 100.0;  // passed as percent
  const auto range_size = static_cast<size_t>(state.range(2));

  const auto overlap = double{1};
  const auto skew = static_cast<double>(state.range(2)) / 100.0;

  std::mt19937 rng(42);
  simd_sort::simd_vector<SimdElement> relation_r;
  simd_sort::simd_vector<SimdElement> relation_s;

  auto total_matches = double{0};
  auto total_comparison = double{0};

  // state.PauseTiming();  // Don't include data generation
  generate_data(relation_r, relation_s, size_r, size_s, overlap, rng, skew, range_size);
  // state.ResumeTiming();

  const auto warmup = 2;
  for (auto index = 0u; index < warmup; ++index) {
    auto [matches, cmp] = join<KeyRangeAlgo::LinBin128>(relation_r, relation_s);
    benchmark::DoNotOptimize(matches);
    benchmark::DoNotOptimize(cmp);
    benchmark::ClobberMemory();
  }

  // NOLINTNEXTLINE
  for (auto _ : state) {
    auto [matches, cmp] = join<KeyRangeAlgo::LinBin128>(relation_r, relation_s);

    // std::cout << "matches: " << matches << '\n';
    benchmark::DoNotOptimize(matches);
    benchmark::DoNotOptimize(cmp);

    benchmark::ClobberMemory();

    total_matches += static_cast<double>(matches);
    total_comparison += static_cast<double>(cmp);
  }
  total_matches /= static_cast<double>(state.iterations());
  total_comparison /= static_cast<double>(state.iterations());
  state.counters["Matches"] = total_matches;
  state.counters["Comparisons"] = total_comparison;
}

[[maybe_unused]] static void BM_JoinBin(benchmark::State& state) {
  const auto size_r = static_cast<size_t>(state.range(0));
  const auto size_s = static_cast<size_t>(state.range(1));
  // const auto overlap = static_cast<double>(state.range(2)) / 100.0;  // passed as percent
  const auto range_size = static_cast<size_t>(state.range(2));

  const auto overlap = double{1};
  const auto skew = static_cast<double>(state.range(2)) / 100.0;

  std::mt19937 rng(42);
  simd_sort::simd_vector<SimdElement> relation_r;
  simd_sort::simd_vector<SimdElement> relation_s;

  auto total_matches = double{0};
  auto total_comparison = double{0};

  // state.PauseTiming();  // Don't include data generation
  generate_data(relation_r, relation_s, size_r, size_s, overlap, rng, skew, range_size);
  // state.ResumeTiming();

  const auto warmup = 2;
  for (auto index = 0u; index < warmup; ++index) {
    auto [matches, cmp] = join<KeyRangeAlgo::Bin>(relation_r, relation_s);
    benchmark::DoNotOptimize(matches);
    benchmark::DoNotOptimize(cmp);
    benchmark::ClobberMemory();
  }

  // NOLINTNEXTLINE
  for (auto _ : state) {
    auto [matches, cmp] = join<KeyRangeAlgo::Bin>(relation_r, relation_s);
    // std::cout << "matches: " << matches << '\n';
    benchmark::DoNotOptimize(matches);
    benchmark::DoNotOptimize(cmp);

    benchmark::ClobberMemory();

    total_matches += static_cast<double>(matches);
    total_comparison += static_cast<double>(cmp);
  }
  total_matches /= static_cast<double>(state.iterations());
  total_comparison /= static_cast<double>(state.iterations());
  state.counters["Matches"] = total_matches;
  state.counters["Comparisons"] = total_comparison;
}

[[maybe_unused]] static void BM_JoinExp(benchmark::State& state) {
  const auto size_r = static_cast<size_t>(state.range(0));
  const auto size_s = static_cast<size_t>(state.range(1));
  // const auto overlap = static_cast<double>(state.range(2)) / 100.0;  // passed as percent
  const auto range_size = static_cast<size_t>(state.range(2));

  const auto overlap = double{1};
  const auto skew = static_cast<double>(state.range(2)) / 100.0;

  std::mt19937 rng(42);
  simd_sort::simd_vector<SimdElement> relation_r;
  simd_sort::simd_vector<SimdElement> relation_s;

  auto total_matches = double{0};
  auto total_comparison = double{0};

  // state.PauseTiming();  // Don't include data generation
  generate_data(relation_r, relation_s, size_r, size_s, overlap, rng, skew, range_size);
  // state.ResumeTiming();

  const auto warmup = 2;
  for (auto index = 0u; index < warmup; ++index) {
    auto [matches, cmp] = join<KeyRangeAlgo::Exp>(relation_r, relation_s);
    benchmark::DoNotOptimize(matches);
    benchmark::DoNotOptimize(cmp);
    benchmark::ClobberMemory();
  }

  // NOLINTNEXTLINE
  for (auto _ : state) {
    auto [matches, cmp] = join<KeyRangeAlgo::Exp>(relation_r, relation_s);
    // std::cout << "matches: " << matches << '\n';
    benchmark::DoNotOptimize(matches);
    benchmark::DoNotOptimize(cmp);

    benchmark::ClobberMemory();

    total_matches += static_cast<double>(matches);
    total_comparison += static_cast<double>(cmp);
  }
  total_matches /= static_cast<double>(state.iterations());
  total_comparison /= static_cast<double>(state.iterations());
  state.counters["Matches"] = total_matches;
  state.counters["Comparisons"] = total_comparison;
}

constexpr auto SIZE_ONE = 1 * 1'048'576;
constexpr auto SIZE_TWO = 10 * 1'048'576;
constexpr auto SIZE_THREE = 50 * 1'048'576;

BENCHMARK(BM_JoinLinBin8)
    ->Args({SIZE_ONE, SIZE_ONE, 1})
    ->Args({SIZE_ONE, SIZE_ONE, 2})
    ->Args({SIZE_ONE, SIZE_ONE, 4})
    ->Args({SIZE_ONE, SIZE_ONE, 8})
    ->Args({SIZE_ONE, SIZE_ONE, 16})
    ->Args({SIZE_ONE, SIZE_ONE, 32})
    ->Args({SIZE_ONE, SIZE_ONE, 64})
    ->Args({SIZE_ONE, SIZE_ONE, 128})
    ->Args({SIZE_ONE, SIZE_ONE, 256})
    ->Args({SIZE_ONE, SIZE_ONE, 512})
    ->Args({SIZE_ONE, SIZE_ONE, 1024})
    ->Args({SIZE_ONE, SIZE_ONE, 2048})
    ->Args({SIZE_TWO, SIZE_TWO, 1})
    ->Args({SIZE_TWO, SIZE_TWO, 2})
    ->Args({SIZE_TWO, SIZE_TWO, 4})
    ->Args({SIZE_TWO, SIZE_TWO, 8})
    ->Args({SIZE_TWO, SIZE_TWO, 16})
    ->Args({SIZE_TWO, SIZE_TWO, 32})
    ->Args({SIZE_TWO, SIZE_TWO, 64})
    ->Args({SIZE_TWO, SIZE_TWO, 128})
    ->Args({SIZE_TWO, SIZE_TWO, 256})
    ->Args({SIZE_TWO, SIZE_TWO, 512})
    ->Args({SIZE_TWO, SIZE_TWO, 1024})
    ->Args({SIZE_TWO, SIZE_TWO, 2048})
    ->Args({SIZE_THREE, SIZE_THREE, 1})
    ->Args({SIZE_THREE, SIZE_THREE, 2})
    ->Args({SIZE_THREE, SIZE_THREE, 4})
    ->Args({SIZE_THREE, SIZE_THREE, 8})
    ->Args({SIZE_THREE, SIZE_THREE, 16})
    ->Args({SIZE_THREE, SIZE_THREE, 32})
    ->Args({SIZE_THREE, SIZE_THREE, 64})
    ->Args({SIZE_THREE, SIZE_THREE, 128})
    ->Args({SIZE_THREE, SIZE_THREE, 256})
    ->Args({SIZE_THREE, SIZE_THREE, 512})
    ->Args({SIZE_THREE, SIZE_THREE, 1024})
    ->Args({SIZE_THREE, SIZE_THREE, 2048})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(5);

BENCHMARK(BM_JoinLinBin64)
    ->Args({SIZE_ONE, SIZE_ONE, 1})
    ->Args({SIZE_ONE, SIZE_ONE, 2})
    ->Args({SIZE_ONE, SIZE_ONE, 4})
    ->Args({SIZE_ONE, SIZE_ONE, 8})
    ->Args({SIZE_ONE, SIZE_ONE, 16})
    ->Args({SIZE_ONE, SIZE_ONE, 32})
    ->Args({SIZE_ONE, SIZE_ONE, 64})
    ->Args({SIZE_ONE, SIZE_ONE, 128})
    ->Args({SIZE_ONE, SIZE_ONE, 256})
    ->Args({SIZE_ONE, SIZE_ONE, 512})
    ->Args({SIZE_ONE, SIZE_ONE, 1024})
    ->Args({SIZE_ONE, SIZE_ONE, 2048})
    ->Args({SIZE_TWO, SIZE_TWO, 1})
    ->Args({SIZE_TWO, SIZE_TWO, 2})
    ->Args({SIZE_TWO, SIZE_TWO, 4})
    ->Args({SIZE_TWO, SIZE_TWO, 8})
    ->Args({SIZE_TWO, SIZE_TWO, 16})
    ->Args({SIZE_TWO, SIZE_TWO, 32})
    ->Args({SIZE_TWO, SIZE_TWO, 64})
    ->Args({SIZE_TWO, SIZE_TWO, 128})
    ->Args({SIZE_TWO, SIZE_TWO, 256})
    ->Args({SIZE_TWO, SIZE_TWO, 512})
    ->Args({SIZE_TWO, SIZE_TWO, 1024})
    ->Args({SIZE_TWO, SIZE_TWO, 2048})
    ->Args({SIZE_THREE, SIZE_THREE, 1})
    ->Args({SIZE_THREE, SIZE_THREE, 2})
    ->Args({SIZE_THREE, SIZE_THREE, 4})
    ->Args({SIZE_THREE, SIZE_THREE, 8})
    ->Args({SIZE_THREE, SIZE_THREE, 16})
    ->Args({SIZE_THREE, SIZE_THREE, 32})
    ->Args({SIZE_THREE, SIZE_THREE, 64})
    ->Args({SIZE_THREE, SIZE_THREE, 128})
    ->Args({SIZE_THREE, SIZE_THREE, 256})
    ->Args({SIZE_THREE, SIZE_THREE, 512})
    ->Args({SIZE_THREE, SIZE_THREE, 1024})
    ->Args({SIZE_THREE, SIZE_THREE, 2048})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(5);

BENCHMARK(BM_JoinLinBin128)
    ->Args({SIZE_ONE, SIZE_ONE, 1})
    ->Args({SIZE_ONE, SIZE_ONE, 2})
    ->Args({SIZE_ONE, SIZE_ONE, 4})
    ->Args({SIZE_ONE, SIZE_ONE, 8})
    ->Args({SIZE_ONE, SIZE_ONE, 16})
    ->Args({SIZE_ONE, SIZE_ONE, 32})
    ->Args({SIZE_ONE, SIZE_ONE, 64})
    ->Args({SIZE_ONE, SIZE_ONE, 128})
    ->Args({SIZE_ONE, SIZE_ONE, 256})
    ->Args({SIZE_ONE, SIZE_ONE, 512})
    ->Args({SIZE_ONE, SIZE_ONE, 1024})
    ->Args({SIZE_ONE, SIZE_ONE, 2048})
    ->Args({SIZE_TWO, SIZE_TWO, 1})
    ->Args({SIZE_TWO, SIZE_TWO, 2})
    ->Args({SIZE_TWO, SIZE_TWO, 4})
    ->Args({SIZE_TWO, SIZE_TWO, 8})
    ->Args({SIZE_TWO, SIZE_TWO, 16})
    ->Args({SIZE_TWO, SIZE_TWO, 32})
    ->Args({SIZE_TWO, SIZE_TWO, 64})
    ->Args({SIZE_TWO, SIZE_TWO, 128})
    ->Args({SIZE_TWO, SIZE_TWO, 256})
    ->Args({SIZE_TWO, SIZE_TWO, 512})
    ->Args({SIZE_TWO, SIZE_TWO, 1024})
    ->Args({SIZE_TWO, SIZE_TWO, 2048})
    ->Args({SIZE_THREE, SIZE_THREE, 1})
    ->Args({SIZE_THREE, SIZE_THREE, 2})
    ->Args({SIZE_THREE, SIZE_THREE, 4})
    ->Args({SIZE_THREE, SIZE_THREE, 8})
    ->Args({SIZE_THREE, SIZE_THREE, 16})
    ->Args({SIZE_THREE, SIZE_THREE, 32})
    ->Args({SIZE_THREE, SIZE_THREE, 64})
    ->Args({SIZE_THREE, SIZE_THREE, 128})
    ->Args({SIZE_THREE, SIZE_THREE, 256})
    ->Args({SIZE_THREE, SIZE_THREE, 512})
    ->Args({SIZE_THREE, SIZE_THREE, 1024})
    ->Args({SIZE_THREE, SIZE_THREE, 2048})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(5);

BENCHMARK(BM_JoinBin)
    ->Args({SIZE_ONE, SIZE_ONE, 1})
    ->Args({SIZE_ONE, SIZE_ONE, 2})
    ->Args({SIZE_ONE, SIZE_ONE, 4})
    ->Args({SIZE_ONE, SIZE_ONE, 8})
    ->Args({SIZE_ONE, SIZE_ONE, 16})
    ->Args({SIZE_ONE, SIZE_ONE, 32})
    ->Args({SIZE_ONE, SIZE_ONE, 64})
    ->Args({SIZE_ONE, SIZE_ONE, 128})
    ->Args({SIZE_ONE, SIZE_ONE, 256})
    ->Args({SIZE_ONE, SIZE_ONE, 512})
    ->Args({SIZE_ONE, SIZE_ONE, 1024})
    ->Args({SIZE_ONE, SIZE_ONE, 2048})
    ->Args({SIZE_TWO, SIZE_TWO, 1})
    ->Args({SIZE_TWO, SIZE_TWO, 2})
    ->Args({SIZE_TWO, SIZE_TWO, 4})
    ->Args({SIZE_TWO, SIZE_TWO, 8})
    ->Args({SIZE_TWO, SIZE_TWO, 16})
    ->Args({SIZE_TWO, SIZE_TWO, 32})
    ->Args({SIZE_TWO, SIZE_TWO, 64})
    ->Args({SIZE_TWO, SIZE_TWO, 128})
    ->Args({SIZE_TWO, SIZE_TWO, 256})
    ->Args({SIZE_TWO, SIZE_TWO, 512})
    ->Args({SIZE_TWO, SIZE_TWO, 1024})
    ->Args({SIZE_TWO, SIZE_TWO, 2048})
    ->Args({SIZE_THREE, SIZE_THREE, 1})
    ->Args({SIZE_THREE, SIZE_THREE, 2})
    ->Args({SIZE_THREE, SIZE_THREE, 4})
    ->Args({SIZE_THREE, SIZE_THREE, 8})
    ->Args({SIZE_THREE, SIZE_THREE, 16})
    ->Args({SIZE_THREE, SIZE_THREE, 32})
    ->Args({SIZE_THREE, SIZE_THREE, 64})
    ->Args({SIZE_THREE, SIZE_THREE, 128})
    ->Args({SIZE_THREE, SIZE_THREE, 256})
    ->Args({SIZE_THREE, SIZE_THREE, 512})
    ->Args({SIZE_THREE, SIZE_THREE, 1024})
    ->Args({SIZE_THREE, SIZE_THREE, 2048})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(5);

BENCHMARK(BM_JoinExp)
    ->Args({SIZE_ONE, SIZE_ONE, 1})
    ->Args({SIZE_ONE, SIZE_ONE, 2})
    ->Args({SIZE_ONE, SIZE_ONE, 4})
    ->Args({SIZE_ONE, SIZE_ONE, 8})
    ->Args({SIZE_ONE, SIZE_ONE, 16})
    ->Args({SIZE_ONE, SIZE_ONE, 32})
    ->Args({SIZE_ONE, SIZE_ONE, 64})
    ->Args({SIZE_ONE, SIZE_ONE, 128})
    ->Args({SIZE_ONE, SIZE_ONE, 256})
    ->Args({SIZE_ONE, SIZE_ONE, 512})
    ->Args({SIZE_ONE, SIZE_ONE, 1024})
    ->Args({SIZE_ONE, SIZE_ONE, 2048})
    ->Args({SIZE_TWO, SIZE_TWO, 1})
    ->Args({SIZE_TWO, SIZE_TWO, 2})
    ->Args({SIZE_TWO, SIZE_TWO, 4})
    ->Args({SIZE_TWO, SIZE_TWO, 8})
    ->Args({SIZE_TWO, SIZE_TWO, 16})
    ->Args({SIZE_TWO, SIZE_TWO, 32})
    ->Args({SIZE_TWO, SIZE_TWO, 64})
    ->Args({SIZE_TWO, SIZE_TWO, 128})
    ->Args({SIZE_TWO, SIZE_TWO, 256})
    ->Args({SIZE_TWO, SIZE_TWO, 512})
    ->Args({SIZE_TWO, SIZE_TWO, 1024})
    ->Args({SIZE_TWO, SIZE_TWO, 2048})
    ->Args({SIZE_THREE, SIZE_THREE, 1})
    ->Args({SIZE_THREE, SIZE_THREE, 2})
    ->Args({SIZE_THREE, SIZE_THREE, 4})
    ->Args({SIZE_THREE, SIZE_THREE, 8})
    ->Args({SIZE_THREE, SIZE_THREE, 16})
    ->Args({SIZE_THREE, SIZE_THREE, 32})
    ->Args({SIZE_THREE, SIZE_THREE, 64})
    ->Args({SIZE_THREE, SIZE_THREE, 128})
    ->Args({SIZE_THREE, SIZE_THREE, 256})
    ->Args({SIZE_THREE, SIZE_THREE, 512})
    ->Args({SIZE_THREE, SIZE_THREE, 1024})
    ->Args({SIZE_THREE, SIZE_THREE, 2048})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(5);

// BENCHMARK(BM_JoinLinBin8)
//     ->Args({SIZE_ONE, SIZE_ONE, 1})
//     ->Args({SIZE_ONE, SIZE_ONE, 20})
//     ->Args({SIZE_ONE, SIZE_ONE, 30})
//     ->Args({SIZE_ONE, SIZE_ONE, 40})
//     ->Args({SIZE_ONE, SIZE_ONE, 50})
//     ->Args({SIZE_ONE, SIZE_ONE, 60})
//     ->Args({SIZE_ONE, SIZE_ONE, 70})
//     ->Args({SIZE_ONE, SIZE_ONE, 80})
//     ->Args({SIZE_ONE, SIZE_ONE, 90})
//     ->Args({SIZE_ONE, SIZE_ONE, 99})
//     ->Args({SIZE_TWO, SIZE_TWO, 1})
//     ->Args({SIZE_TWO, SIZE_TWO, 20})
//     ->Args({SIZE_TWO, SIZE_TWO, 30})
//     ->Args({SIZE_TWO, SIZE_TWO, 40})
//     ->Args({SIZE_TWO, SIZE_TWO, 50})
//     ->Args({SIZE_TWO, SIZE_TWO, 60})
//     ->Args({SIZE_TWO, SIZE_TWO, 70})
//     ->Args({SIZE_TWO, SIZE_TWO, 80})
//     ->Args({SIZE_TWO, SIZE_TWO, 90})
//     ->Args({SIZE_TWO, SIZE_TWO, 99})
//     ->Args({SIZE_THREE, SIZE_THREE, 1})
//     ->Args({SIZE_THREE, SIZE_THREE, 20})
//     ->Args({SIZE_THREE, SIZE_THREE, 30})
//     ->Args({SIZE_THREE, SIZE_THREE, 40})
//     ->Args({SIZE_THREE, SIZE_THREE, 50})
//     ->Args({SIZE_THREE, SIZE_THREE, 60})
//     ->Args({SIZE_THREE, SIZE_THREE, 70})
//     ->Args({SIZE_THREE, SIZE_THREE, 80})
//     ->Args({SIZE_THREE, SIZE_THREE, 90})
//     ->Args({SIZE_THREE, SIZE_THREE, 99})
//     ->Unit(benchmark::kMillisecond)
//     ->Iterations(5);
//
// BENCHMARK(BM_JoinLinBin64)
//     ->Args({SIZE_ONE, SIZE_ONE, 1})
//     ->Args({SIZE_ONE, SIZE_ONE, 20})
//     ->Args({SIZE_ONE, SIZE_ONE, 30})
//     ->Args({SIZE_ONE, SIZE_ONE, 40})
//     ->Args({SIZE_ONE, SIZE_ONE, 50})
//     ->Args({SIZE_ONE, SIZE_ONE, 60})
//     ->Args({SIZE_ONE, SIZE_ONE, 70})
//     ->Args({SIZE_ONE, SIZE_ONE, 80})
//     ->Args({SIZE_ONE, SIZE_ONE, 90})
//     ->Args({SIZE_ONE, SIZE_ONE, 99})
//     ->Args({SIZE_TWO, SIZE_TWO, 1})
//     ->Args({SIZE_TWO, SIZE_TWO, 20})
//     ->Args({SIZE_TWO, SIZE_TWO, 30})
//     ->Args({SIZE_TWO, SIZE_TWO, 40})
//     ->Args({SIZE_TWO, SIZE_TWO, 50})
//     ->Args({SIZE_TWO, SIZE_TWO, 60})
//     ->Args({SIZE_TWO, SIZE_TWO, 70})
//     ->Args({SIZE_TWO, SIZE_TWO, 80})
//     ->Args({SIZE_TWO, SIZE_TWO, 90})
//     ->Args({SIZE_TWO, SIZE_TWO, 99})
//     ->Args({SIZE_THREE, SIZE_THREE, 1})
//     ->Args({SIZE_THREE, SIZE_THREE, 20})
//     ->Args({SIZE_THREE, SIZE_THREE, 30})
//     ->Args({SIZE_THREE, SIZE_THREE, 40})
//     ->Args({SIZE_THREE, SIZE_THREE, 50})
//     ->Args({SIZE_THREE, SIZE_THREE, 60})
//     ->Args({SIZE_THREE, SIZE_THREE, 70})
//     ->Args({SIZE_THREE, SIZE_THREE, 80})
//     ->Args({SIZE_THREE, SIZE_THREE, 90})
//     ->Args({SIZE_THREE, SIZE_THREE, 99})
//     ->Unit(benchmark::kMillisecond)
//     ->Iterations(5);
//
// BENCHMARK(BM_JoinLinBin128)
//     ->Args({SIZE_ONE, SIZE_ONE, 1})
//     ->Args({SIZE_ONE, SIZE_ONE, 20})
//     ->Args({SIZE_ONE, SIZE_ONE, 30})
//     ->Args({SIZE_ONE, SIZE_ONE, 40})
//     ->Args({SIZE_ONE, SIZE_ONE, 50})
//     ->Args({SIZE_ONE, SIZE_ONE, 60})
//     ->Args({SIZE_ONE, SIZE_ONE, 70})
//     ->Args({SIZE_ONE, SIZE_ONE, 80})
//     ->Args({SIZE_ONE, SIZE_ONE, 90})
//     ->Args({SIZE_ONE, SIZE_ONE, 99})
//     ->Args({SIZE_TWO, SIZE_TWO, 1})
//     ->Args({SIZE_TWO, SIZE_TWO, 20})
//     ->Args({SIZE_TWO, SIZE_TWO, 30})
//     ->Args({SIZE_TWO, SIZE_TWO, 40})
//     ->Args({SIZE_TWO, SIZE_TWO, 50})
//     ->Args({SIZE_TWO, SIZE_TWO, 60})
//     ->Args({SIZE_TWO, SIZE_TWO, 70})
//     ->Args({SIZE_TWO, SIZE_TWO, 80})
//     ->Args({SIZE_TWO, SIZE_TWO, 90})
//     ->Args({SIZE_TWO, SIZE_TWO, 99})
//     ->Args({SIZE_THREE, SIZE_THREE, 1})
//     ->Args({SIZE_THREE, SIZE_THREE, 20})
//     ->Args({SIZE_THREE, SIZE_THREE, 30})
//     ->Args({SIZE_THREE, SIZE_THREE, 40})
//     ->Args({SIZE_THREE, SIZE_THREE, 50})
//     ->Args({SIZE_THREE, SIZE_THREE, 60})
//     ->Args({SIZE_THREE, SIZE_THREE, 70})
//     ->Args({SIZE_THREE, SIZE_THREE, 80})
//     ->Args({SIZE_THREE, SIZE_THREE, 90})
//     ->Args({SIZE_THREE, SIZE_THREE, 99})
//     ->Unit(benchmark::kMillisecond)
//     ->Iterations(5);
//
// BENCHMARK(BM_JoinBin)
//     ->Args({SIZE_ONE, SIZE_ONE, 1})
//     ->Args({SIZE_ONE, SIZE_ONE, 20})
//     ->Args({SIZE_ONE, SIZE_ONE, 30})
//     ->Args({SIZE_ONE, SIZE_ONE, 40})
//     ->Args({SIZE_ONE, SIZE_ONE, 50})
//     ->Args({SIZE_ONE, SIZE_ONE, 60})
//     ->Args({SIZE_ONE, SIZE_ONE, 70})
//     ->Args({SIZE_ONE, SIZE_ONE, 80})
//     ->Args({SIZE_ONE, SIZE_ONE, 90})
//     ->Args({SIZE_ONE, SIZE_ONE, 99})
//     ->Args({SIZE_TWO, SIZE_TWO, 1})
//     ->Args({SIZE_TWO, SIZE_TWO, 20})
//     ->Args({SIZE_TWO, SIZE_TWO, 30})
//     ->Args({SIZE_TWO, SIZE_TWO, 40})
//     ->Args({SIZE_TWO, SIZE_TWO, 50})
//     ->Args({SIZE_TWO, SIZE_TWO, 60})
//     ->Args({SIZE_TWO, SIZE_TWO, 70})
//     ->Args({SIZE_TWO, SIZE_TWO, 80})
//     ->Args({SIZE_TWO, SIZE_TWO, 90})
//     ->Args({SIZE_TWO, SIZE_TWO, 99})
//     ->Args({SIZE_THREE, SIZE_THREE, 1})
//     ->Args({SIZE_THREE, SIZE_THREE, 20})
//     ->Args({SIZE_THREE, SIZE_THREE, 30})
//     ->Args({SIZE_THREE, SIZE_THREE, 40})
//     ->Args({SIZE_THREE, SIZE_THREE, 50})
//     ->Args({SIZE_THREE, SIZE_THREE, 60})
//     ->Args({SIZE_THREE, SIZE_THREE, 70})
//     ->Args({SIZE_THREE, SIZE_THREE, 80})
//     ->Args({SIZE_THREE, SIZE_THREE, 90})
//     ->Args({SIZE_THREE, SIZE_THREE, 99})
//     ->Unit(benchmark::kMillisecond)
//     ->Iterations(5);
//
// BENCHMARK(BM_JoinExp)
//     ->Args({SIZE_ONE, SIZE_ONE, 1})
//     ->Args({SIZE_ONE, SIZE_ONE, 20})
//     ->Args({SIZE_ONE, SIZE_ONE, 30})
//     ->Args({SIZE_ONE, SIZE_ONE, 40})
//     ->Args({SIZE_ONE, SIZE_ONE, 50})
//     ->Args({SIZE_ONE, SIZE_ONE, 60})
//     ->Args({SIZE_ONE, SIZE_ONE, 70})
//     ->Args({SIZE_ONE, SIZE_ONE, 80})
//     ->Args({SIZE_ONE, SIZE_ONE, 90})
//     ->Args({SIZE_ONE, SIZE_ONE, 99})
//     ->Args({SIZE_TWO, SIZE_TWO, 1})
//     ->Args({SIZE_TWO, SIZE_TWO, 20})
//     ->Args({SIZE_TWO, SIZE_TWO, 30})
//     ->Args({SIZE_TWO, SIZE_TWO, 40})
//     ->Args({SIZE_TWO, SIZE_TWO, 50})
//     ->Args({SIZE_TWO, SIZE_TWO, 60})
//     ->Args({SIZE_TWO, SIZE_TWO, 70})
//     ->Args({SIZE_TWO, SIZE_TWO, 80})
//     ->Args({SIZE_TWO, SIZE_TWO, 90})
//     ->Args({SIZE_TWO, SIZE_TWO, 99})
//     ->Args({SIZE_THREE, SIZE_THREE, 1})
//     ->Args({SIZE_THREE, SIZE_THREE, 20})
//     ->Args({SIZE_THREE, SIZE_THREE, 30})
//     ->Args({SIZE_THREE, SIZE_THREE, 40})
//     ->Args({SIZE_THREE, SIZE_THREE, 50})
//     ->Args({SIZE_THREE, SIZE_THREE, 60})
//     ->Args({SIZE_THREE, SIZE_THREE, 70})
//     ->Args({SIZE_THREE, SIZE_THREE, 80})
//     ->Args({SIZE_THREE, SIZE_THREE, 90})
//     ->Args({SIZE_THREE, SIZE_THREE, 99})
//     ->Unit(benchmark::kMillisecond)
//     ->Iterations(5);

BENCHMARK_MAIN();