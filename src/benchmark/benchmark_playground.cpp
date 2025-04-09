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

[[maybe_unused]] std::size_t equal_value_range_size(std::size_t start_index, std::span<SimdElement>& elements) {
  if (start_index >= elements.size()) {
    return 0;
  }
  auto begin = elements.begin();
  std::advance(begin, start_index);
  const auto run_value = compare_value(begin->key);

  constexpr auto LINEAR_SEARCH_ITEMS = std::size_t{64};

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

[[maybe_unused]] std::size_t equal_value_range_size_binary_search(std::size_t start_index,
                                                                  std::span<SimdElement>& elements) {
  if (start_index >= elements.size()) {
    return 0;
  }
  auto begin = elements.begin();
  std::advance(begin, start_index);

  // Binary search in case the run did not end within the linearly scanned part.
  const auto binary_search_result =
      std::upper_bound(begin, elements.end(), *begin, [](const auto& lhs, const auto& rhs) {
        return compare_value(lhs.key) < compare_value(rhs.key);
      });
  return std::distance(begin, binary_search_result);
}

[[maybe_unused]] std::size_t equal_value_range_size_experimental_search(std::size_t start_index,
                                                                        std::span<SimdElement>& elements) {
  if (start_index >= elements.size()) {
    return 0;
  }
  auto begin = elements.begin();
  std::advance(begin, start_index);

  // while (begin <)

  // Binary search in case the run did not end within the linearly scanned part.
  const auto binary_search_result =
      std::upper_bound(begin, elements.end(), *begin, [](const auto& lhs, const auto& rhs) {
        return compare_value(lhs.key) < compare_value(rhs.key);
      });
  return std::distance(begin, binary_search_result);
}

size_t join(std::span<SimdElement> left_elements, std::span<SimdElement> right_elements) {
  auto num_output_tuples = size_t{0};

  auto left_run_start = size_t{0};
  auto right_run_start = size_t{0};

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

    // Emit all combinations of left and right range.
    if (compare_result == CompareResult::Equal) {
      auto count = size_t{0};
      left_range.find_matches_with_range(
          right_range, [&]([[maybe_unused]] const size_t left_row_id, [[maybe_unused]] const size_t right_row_id) {
            ++count;
            ++num_output_tuples;
          });
      // if (count > 128) {
      std::cout << "count: " << count << '\n';
      // }
    }
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
  return num_output_tuples;
}

void generate_data(simd_sort::simd_vector<SimdElement>& relation_r, simd_sort::simd_vector<SimdElement>& relation_s,
                   size_t size_r, size_t size_s, double overlap, std::mt19937& rng, [[maybe_unused]] double skew) {
  // std::uniform_int_distribution<uint32_t> dist(0, std::numeric_limits<uint32_t>::max());
  constexpr auto RANGE_SIZE = 250;
  std::uniform_int_distribution<uint32_t> dist(0, (size_r / RANGE_SIZE) - 1);
  // auto dist = zipfian_int_distribution<uint32_t>(0, std::numeric_limits<uint32_t>::max(), skew);

  relation_r.clear();
  relation_r.reserve(size_s);
  for (uint32_t i = 0; i < size_s; ++i) {
    relation_r.push_back({i, dist(rng)});
  }

  std::vector<uint32_t> r_keys;
  r_keys.reserve(size_s);
  for (const auto& elem : relation_r) {
    r_keys.push_back(elem.key);
  }

  // std::shuffle(r_keys.begin(), r_keys.end(), rng);

  relation_s.clear();
  relation_s.reserve(size_r);
  const auto overlap_count = static_cast<size_t>(static_cast<double>(size_r) * overlap);
  for (size_t i = 0; i < overlap_count; ++i) {
    relation_s.push_back({static_cast<uint32_t>(i), r_keys[i]});
  }

  std::unordered_set<uint32_t> s_key_set(r_keys.begin(), r_keys.end());
  while (relation_s.size() < size_r) {
    uint32_t key = dist(rng);
    if (!s_key_set.contains(key)) {
      relation_s.push_back({static_cast<uint32_t>(relation_s.size()), key});
    }
  }

  auto key_comp = [](SimdElement& lhs, SimdElement& rhs) {
    return *reinterpret_cast<SortingType*>(&lhs) < *reinterpret_cast<SortingType*>(&rhs);
  };

  boost::sort::pdqsort(relation_s.begin(), relation_s.end(), key_comp);
  boost::sort::pdqsort(relation_r.begin(), relation_r.end(), key_comp);
}
}  // namespace

static void BM_FinalJoin(benchmark::State& state) {
  const auto size_r = static_cast<size_t>(state.range(0));
  const auto size_s = static_cast<size_t>(state.range(1));
  const auto overlap = static_cast<double>(state.range(2)) / 100.0;  // passed as percent
  // const auto skew = static_cast<double>(state.range(3));
  const auto skew = 0;

  std::mt19937 rng(42);
  simd_sort::simd_vector<SimdElement> relation_r;
  simd_sort::simd_vector<SimdElement> relation_s;

  // NOLINTNEXTLINE
  for (auto _ : state) {
    state.PauseTiming();  // Don't include data generation
    generate_data(relation_r, relation_s, size_r, size_s, overlap, rng, skew);
    state.ResumeTiming();

    size_t matches = join(relation_r, relation_s);
    // std::cout << "matches: " << matches << '\n';
    benchmark::DoNotOptimize(matches);
    benchmark::ClobberMemory();
  }

  // NOLINTNEXTLINE
  // state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * (size_r + size_s));
}

BENCHMARK(BM_FinalJoin)
    ->Args({1u << 20u, 1u << 20u, 0})    // 0% overlap
    ->Args({1u << 20u, 1u << 20u, 50})   // 50% overlap
    ->Args({1u << 20u, 1u << 20u, 100})  // 50% overlap

    // ->Args({128000000, 128000000, 100})  // 100% overlap
    ->Unit(benchmark::kMillisecond)
    ->Iterations(5);

BENCHMARK_MAIN();
