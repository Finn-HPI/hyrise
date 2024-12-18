#include <memory>

#include "benchmark/benchmark.h"

#include "hyrise.hpp"
#include "operators/join_hash.hpp"
#include "operators/join_index.hpp"
#include "operators/join_nested_loop.hpp"
#include "operators/join_simd_sort_merge.hpp"
#include "operators/join_sort_merge.hpp"
#include "operators/table_wrapper.hpp"
#include "storage/chunk.hpp"
#include "storage/index/adaptive_radix_tree/adaptive_radix_tree_index.hpp"
#include "synthetic_table_generator.hpp"
#include "types.hpp"
#include "visualization/abstract_visualizer.hpp"
#include "visualization/pqp_visualizer.hpp"

namespace {
constexpr auto NUMBER_OF_CHUNKS = size_t{50};

// These numbers were arbitrarily chosen to form a representative group of JoinBenchmarks
// that run in a tolerable amount of time
constexpr auto TABLE_SIZE_SMALL = size_t{1'000};
constexpr auto TABLE_SIZE_MEDIUM = size_t{100'000};
constexpr auto TABLE_SIZE_BIG = size_t{10'000'000};

void clear_cache() {
  auto clear = std::vector<int>();
  clear.resize(500 * 1000 * 1000, 42);
  const auto clear_cache_size = clear.size();
  for (auto index = size_t{0}; index < clear_cache_size; index++) {
    clear[index] += 1;
  }
  clear.resize(0);
}
}  // namespace

namespace hyrise {

std::shared_ptr<TableWrapper> generate_table(const size_t number_of_rows) {
  auto table_generator = std::make_shared<SyntheticTableGenerator>();

  const auto chunk_size = static_cast<ChunkOffset>(number_of_rows / NUMBER_OF_CHUNKS);
  Assert(chunk_size > 0, "The chunk size is 0 or less, cannot generate such a table.");

  auto table =
      table_generator->generate_table(1ul, number_of_rows, chunk_size, SegmentEncodingSpec{EncodingType::Dictionary});

  const auto chunk_count = table->chunk_count();
  for (ChunkID chunk_id{0}; chunk_id < chunk_count; ++chunk_id) {
    const auto chunk = table->get_chunk(chunk_id);
    Assert(chunk, "Physically deleted chunk should not reach this point, see get_chunk / #1686.");

    for (auto column_id = ColumnID{0}; column_id < chunk->column_count(); ++column_id) {
      chunk->create_index<AdaptiveRadixTreeIndex>(std::vector<ColumnID>{column_id});
    }
  }

  auto table_wrapper = std::make_shared<TableWrapper>(table);
  table_wrapper->never_clear_output();
  table_wrapper->execute();

  return table_wrapper;
}

void visualize(const std::string& prefix, std::shared_ptr<AbstractOperator> join) {
  auto graphviz_config = GraphvizConfig{};
  graphviz_config.format = "svg";
  const auto& pqps = std::vector<std::shared_ptr<AbstractOperator>>{std::move(join)};
  PQPVisualizer{graphviz_config, {}, {}, {}}.visualize(pqps, prefix + "-PQP.svg");
}

template <class C>
void bm_join_impl(benchmark::State& state, std::shared_ptr<TableWrapper> table_wrapper_left,
                  std::shared_ptr<TableWrapper> table_wrapper_right, std::string& prefix [[maybe_unused]]) {
  clear_cache();

  auto warm_up = std::make_shared<C>(table_wrapper_left, table_wrapper_right, JoinMode::Inner,
                                     OperatorJoinPredicate{{ColumnID{0}, ColumnID{0}}, PredicateCondition::Equals});
  warm_up->execute();
  auto index = 0;
  auto name = prefix + "_" + warm_up->name();
  for (auto _ : state) {  // NOLINT
    auto join = std::make_shared<C>(table_wrapper_left, table_wrapper_right, JoinMode::Inner,
                                    OperatorJoinPredicate{{ColumnID{0}, ColumnID{0}}, PredicateCondition::Equals});
    join->execute();
    auto final_prefix = name + "_" + std::to_string(index) + "_";
    visualize(final_prefix, join);
    ++index;
  }

  Hyrise::reset();
}

template <class C>
void BM_Join_SmallAndSmall(benchmark::State& state) {  // NOLINT 1,000 x 1,000
  auto table_wrapper_left = generate_table(TABLE_SIZE_SMALL);
  auto table_wrapper_right = generate_table(TABLE_SIZE_SMALL);

  auto prefix = std::string("small_and_small");
  bm_join_impl<C>(state, table_wrapper_left, table_wrapper_right, prefix);
}

template <class C>
void BM_Join_SmallAndBig(benchmark::State& state) {  // NOLINT 1,000 x 10,000,000
  auto table_wrapper_left = generate_table(TABLE_SIZE_SMALL);
  auto table_wrapper_right = generate_table(TABLE_SIZE_BIG);

  auto prefix = std::string("small_and_big");
  bm_join_impl<C>(state, table_wrapper_left, table_wrapper_right, prefix);
}

template <class C>
void BM_Join_MediumAndMedium(benchmark::State& state) {  // NOLINT 100,000 x 100,000
  auto table_wrapper_left = generate_table(TABLE_SIZE_MEDIUM);
  auto table_wrapper_right = generate_table(TABLE_SIZE_MEDIUM);

  auto prefix = std::string("med_and_med");
  bm_join_impl<C>(state, table_wrapper_left, table_wrapper_right, prefix);
}

BENCHMARK_TEMPLATE(BM_Join_SmallAndSmall, JoinNestedLoop);

// BENCHMARK_TEMPLATE(BM_Join_SmallAndSmall, JoinIndex);
// BENCHMARK_TEMPLATE(BM_Join_SmallAndBig, JoinIndex);
// BENCHMARK_TEMPLATE(BM_Join_MediumAndMedium, JoinIndex);

BENCHMARK_TEMPLATE(BM_Join_SmallAndSmall, JoinHash);
BENCHMARK_TEMPLATE(BM_Join_SmallAndBig, JoinHash);
BENCHMARK_TEMPLATE(BM_Join_MediumAndMedium, JoinHash);

BENCHMARK_TEMPLATE(BM_Join_SmallAndSmall, JoinSortMerge);
BENCHMARK_TEMPLATE(BM_Join_SmallAndBig, JoinSortMerge);
BENCHMARK_TEMPLATE(BM_Join_MediumAndMedium, JoinSortMerge);

BENCHMARK_TEMPLATE(BM_Join_SmallAndSmall, JoinSimdSortMerge);
BENCHMARK_TEMPLATE(BM_Join_SmallAndBig, JoinSimdSortMerge);
BENCHMARK_TEMPLATE(BM_Join_MediumAndMedium, JoinSimdSortMerge);

}  // namespace hyrise
