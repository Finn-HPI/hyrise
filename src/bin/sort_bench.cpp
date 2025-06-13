#include <cstdlib>

#include "operators/join_simd_sort_merge/simd_sort.hpp"
#include "operators/join_simd_sort_merge/simd_utils.hpp"
#include "operators/join_simd_sort_merge/util.hpp"
#include "types.hpp"
#include "utils/assert.hpp"

using namespace hyrise;  // NOLINT(build/namespaces)

namespace {
bool seeded = false;
unsigned int seed_value;

constexpr auto DUMMY_SIZE = (size_t{24} * 1024 * 1024) / sizeof(SimdElement);

int64_t rand_range(size_t num) {
  // NOLINTNEXTLINE
  return static_cast<int64_t>(static_cast<double>(rand()) / (static_cast<double>(RAND_MAX) + 1) *
                              static_cast<double>(num));
}

void seed_generator(unsigned int seed) {
  srand(seed);
  seed_value = seed;
  seeded = true;
}

void check_seed() {
  if (!seeded) {
    seed_value = time(nullptr);
    srand(seed_value);
    seeded = true;
  }
}

void knuth_shuffle(SimdElement* tuples, size_t num_tuples) {
  size_t index{};
  for (index = num_tuples - 1; index > 0; index--) {
    int64_t swap_index = rand_range(index);
    uint32_t tmp = tuples[index].key;
    tuples[index].key = tuples[swap_index].key;
    tuples[swap_index].key = tmp;
  }
}

void random_unique_gen(SimdElement* tuples, size_t num_tuples) {
  uint64_t index{};

  for (index = 0; index < num_tuples; index++) {
    tuples[index].key = (index + 1);
  }

  /* randomly shuffle elements */
  knuth_shuffle(tuples, num_tuples);
}

int create_relation_pk(SimdElement* tuples, size_t num_tuples) {
  check_seed();

  random_unique_gen(tuples, num_tuples);

  return 0;
}

[[maybe_unused]] constexpr std::size_t choose_count_per_vector() {
#if defined(__AVX512F__)
  return 8;
#else
  return 4;
#endif
}

using SortingType = double;

void bench_simd_sort(size_t num_tuples) {
  auto dummybuffer = simd_sort::simd_vector<SimdElement>(DUMMY_SIZE);
  auto tuples = simd_sort::simd_vector<SimdElement>(num_tuples);
  auto out = simd_sort::simd_vector<SimdElement>(num_tuples);
  seed_generator(12345);
  create_relation_pk(tuples.data(), num_tuples);

  uint32_t garbage = 0xdeadbeef;
  for (unsigned int i = 0; i < DUMMY_SIZE; i++) {
    garbage += dummybuffer[i].key;
    dummybuffer[i].key = garbage;
  }

  simd_sort::simd_vector<SimdElement> copy{};
  std::ranges::copy(tuples.begin(), tuples.end(), std::back_inserter(copy));
  std::sort(copy.begin(), copy.end(), [](auto& lhs, auto& rhs) {
    return *reinterpret_cast<SortingType*>(&lhs) < *reinterpret_cast<SortingType*>(&rhs);
  });

  auto sorted_copy = std::span(reinterpret_cast<SortingType*>(copy.data()), num_tuples);

  auto* input = reinterpret_cast<SortingType*>(tuples.data());
  auto* output = reinterpret_cast<SortingType*>(out.data());

  const auto count_per_vector = choose_count_per_vector();

  auto start = std::chrono::high_resolution_clock::now();
  simd_sort::sort<count_per_vector, SortingType, ExecutionStrategy::SEQUENTIAL>(input, output, num_tuples);
  auto end = std::chrono::high_resolution_clock::now();

  auto us_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "NUM-TUPLES: " << num_tuples << '\n';
  std::cout << "TOTAL-TIME-USECS:" << us_time << '\n';
  std::cout << "TUPLES-PER-SECOND: " << std::fixed << std::setprecision(4)
            << (static_cast<double>(num_tuples) / (static_cast<double>(us_time) / 1'000'000)) << '\n';

  auto sorted = std::span(output, num_tuples);
  for (auto i = num_tuples - 51; i < num_tuples; ++i) {
    std::cout << sorted[i] << " ";
  }

  for (auto i = 0u; i < num_tuples; ++i) {
    Assert(sorted[i] == sorted_copy[i], "does not match");
  }
  Assert(std::is_sorted(sorted.begin(), sorted.end()), "Not sorted");
}

}  // namespace

int main(int argc [[maybe_unused]], char** argv) {
  size_t num_tuples = 1'048'576;
  // NOLINTNEXTLINE
  num_tuples *= atoi(argv[1]);
  bench_simd_sort(num_tuples);
  return 0;
}
