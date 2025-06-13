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
#include "storage/segment_iterate.hpp"
#include "synthetic_table_generator.hpp"
#include "types.hpp"
#include "visualization/abstract_visualizer.hpp"
#include "visualization/pqp_visualizer.hpp"

namespace {

// These numbers were arbitrarily chosen to form a representative group of JoinBenchmarks
// that run in a tolerable amount of time
constexpr auto BASE_SIZE = size_t{1'048'576};

[[maybe_unused]] void clear_cache() {
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

std::shared_ptr<TableWrapper> generate_table(const size_t number_of_rows, const size_t max_row_id) {
  auto table_generator = std::make_shared<SyntheticTableGenerator>();

  const auto chunk_size = static_cast<ChunkOffset>(65535);
  Assert(chunk_size > 0, "The chunk size is 0 or less, cannot generate such a table.");

  const auto column_specification =
      ColumnSpecification{{ColumnDataDistribution::make_uniform_config(0.0, static_cast<int>(max_row_id))},
                          DataType::Int,
                          SegmentEncodingSpec{EncodingType::Dictionary}};

  auto table = hyrise::SyntheticTableGenerator::generate_table({1ul, column_specification}, number_of_rows, chunk_size,
                                                               UseMvcc::No);

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

std::shared_ptr<TableWrapper> generate_key_table(const size_t number_of_rows) {
  auto table_generator = std::make_shared<SyntheticTableGenerator>();

  const auto chunk_size = static_cast<ChunkOffset>(65535);
  Assert(chunk_size > 0, "The chunk size is 0 or less, cannot generate such a table.");

  const auto column_specification =
      ColumnSpecification{{ColumnDataDistribution::make_key_config(0.0, static_cast<int>(number_of_rows) - 1)},
                          DataType::Int,
                          SegmentEncodingSpec{EncodingType::Dictionary}};

  auto table = hyrise::SyntheticTableGenerator::generate_table({1ul, column_specification}, number_of_rows, chunk_size,
                                                               UseMvcc::No);

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
[[maybe_unused]] void BM_Join_impl(benchmark::State& state, std::shared_ptr<TableWrapper> table_wrapper_left,
                                   std::shared_ptr<TableWrapper> table_wrapper_right,
                                   std::string& prefix [[maybe_unused]]) {
  clear_cache();

  auto warm_up = std::make_shared<C>(table_wrapper_left, table_wrapper_right, JoinMode::Inner,
                                     OperatorJoinPredicate{{ColumnID{0}, ColumnID{0}}, PredicateCondition::Equals});
  warm_up->execute();
  for (auto _ : state) {  // NOLINT
    auto join = std::make_shared<C>(table_wrapper_left, table_wrapper_right, JoinMode::Inner,
                                    OperatorJoinPredicate{{ColumnID{0}, ColumnID{0}}, PredicateCondition::Equals});
    join->execute();
  }

  Hyrise::reset();
}

template <typename C>
void BM_Join_Small(benchmark::State& state) {  // NOLINT M x 256M
  const auto factor = static_cast<size_t>(state.range(0));
  const auto num_left = static_cast<size_t>(BASE_SIZE / 1024);

  auto table_wrapper_left = generate_key_table(num_left);
  auto table_wrapper_right = generate_table(factor * num_left, num_left - 1);

  auto prefix = std::string("");
  BM_Join_impl<C>(state, table_wrapper_left, table_wrapper_right, prefix);
}

template <typename C>
void BM_Join_Medium(benchmark::State& state) {  // NOLINT M x 256M
  const auto factor = static_cast<size_t>(state.range(0));
  const auto num_left = static_cast<size_t>(100 * (BASE_SIZE / 1024));

  auto table_wrapper_left = generate_key_table(num_left);
  auto table_wrapper_right = generate_table(factor * num_left, num_left - 1);

  auto prefix = std::string("");
  BM_Join_impl<C>(state, table_wrapper_left, table_wrapper_right, prefix);
}

template <typename C>
void BM_Join(benchmark::State& state) {  // NOLINT 16M x 256M
  const auto scale_left = static_cast<size_t>(state.range(0));
  const auto factor = static_cast<size_t>(state.range(1));

  const auto num_left = scale_left * BASE_SIZE;
  auto table_wrapper_left = generate_key_table(num_left);
  auto table_wrapper_right = generate_table(scale_left * factor * BASE_SIZE, num_left - 1);

  auto prefix = std::string("");
  BM_Join_impl<C>(state, table_wrapper_left, table_wrapper_right, prefix);
}

BENCHMARK_TEMPLATE(BM_Join_Small, JoinHash)->Args({1});
BENCHMARK_TEMPLATE(BM_Join_Small, JoinHash)->Args({2});
BENCHMARK_TEMPLATE(BM_Join_Small, JoinHash)->Args({4});
BENCHMARK_TEMPLATE(BM_Join_Small, JoinHash)->Args({8});
BENCHMARK_TEMPLATE(BM_Join_Small, JoinHash)->Args({16});
BENCHMARK_TEMPLATE(BM_Join_Small, JoinSortMerge)->Args({1});
BENCHMARK_TEMPLATE(BM_Join_Small, JoinSortMerge)->Args({2});
BENCHMARK_TEMPLATE(BM_Join_Small, JoinSortMerge)->Args({4});
BENCHMARK_TEMPLATE(BM_Join_Small, JoinSortMerge)->Args({8});
BENCHMARK_TEMPLATE(BM_Join_Small, JoinSortMerge)->Args({16});
BENCHMARK_TEMPLATE(BM_Join_Small, JoinSortMerge)->Args({32});
BENCHMARK_TEMPLATE(BM_Join_Small, JoinSimdSortMerge)->Args({1});
BENCHMARK_TEMPLATE(BM_Join_Small, JoinSimdSortMerge)->Args({2});
BENCHMARK_TEMPLATE(BM_Join_Small, JoinSimdSortMerge)->Args({4});
BENCHMARK_TEMPLATE(BM_Join_Small, JoinSimdSortMerge)->Args({8});
BENCHMARK_TEMPLATE(BM_Join_Small, JoinSimdSortMerge)->Args({16});
BENCHMARK_TEMPLATE(BM_Join_Small, JoinSimdSortMerge)->Args({32});

BENCHMARK_TEMPLATE(BM_Join_Medium, JoinHash)->Args({1});
BENCHMARK_TEMPLATE(BM_Join_Medium, JoinHash)->Args({2});
BENCHMARK_TEMPLATE(BM_Join_Medium, JoinHash)->Args({4});
BENCHMARK_TEMPLATE(BM_Join_Medium, JoinHash)->Args({8});
BENCHMARK_TEMPLATE(BM_Join_Medium, JoinHash)->Args({16});
BENCHMARK_TEMPLATE(BM_Join_Medium, JoinHash)->Args({32});
BENCHMARK_TEMPLATE(BM_Join_Medium, JoinSortMerge)->Args({1});
BENCHMARK_TEMPLATE(BM_Join_Medium, JoinSortMerge)->Args({2});
BENCHMARK_TEMPLATE(BM_Join_Medium, JoinSortMerge)->Args({4});
BENCHMARK_TEMPLATE(BM_Join_Medium, JoinSortMerge)->Args({8});
BENCHMARK_TEMPLATE(BM_Join_Medium, JoinSortMerge)->Args({16});
BENCHMARK_TEMPLATE(BM_Join_Medium, JoinSortMerge)->Args({32});
BENCHMARK_TEMPLATE(BM_Join_Medium, JoinSimdSortMerge)->Args({1});
BENCHMARK_TEMPLATE(BM_Join_Medium, JoinSimdSortMerge)->Args({2});
BENCHMARK_TEMPLATE(BM_Join_Medium, JoinSimdSortMerge)->Args({4});
BENCHMARK_TEMPLATE(BM_Join_Medium, JoinSimdSortMerge)->Args({8});
BENCHMARK_TEMPLATE(BM_Join_Medium, JoinSimdSortMerge)->Args({16});
BENCHMARK_TEMPLATE(BM_Join_Medium, JoinSimdSortMerge)->Args({32});

BENCHMARK_TEMPLATE(BM_Join, JoinHash)->Args({1, 1});
BENCHMARK_TEMPLATE(BM_Join, JoinHash)->Args({1, 2});
BENCHMARK_TEMPLATE(BM_Join, JoinHash)->Args({1, 4});
BENCHMARK_TEMPLATE(BM_Join, JoinHash)->Args({1, 8});
BENCHMARK_TEMPLATE(BM_Join, JoinHash)->Args({1, 16});
BENCHMARK_TEMPLATE(BM_Join, JoinHash)->Args({1, 32});
BENCHMARK_TEMPLATE(BM_Join, JoinSortMerge)->Args({1, 1});
BENCHMARK_TEMPLATE(BM_Join, JoinSortMerge)->Args({1, 2});
BENCHMARK_TEMPLATE(BM_Join, JoinSortMerge)->Args({1, 4});
BENCHMARK_TEMPLATE(BM_Join, JoinSortMerge)->Args({1, 8});
BENCHMARK_TEMPLATE(BM_Join, JoinSortMerge)->Args({1, 16});
BENCHMARK_TEMPLATE(BM_Join, JoinSortMerge)->Args({1, 32});
BENCHMARK_TEMPLATE(BM_Join, JoinSimdSortMerge)->Args({1, 1});
BENCHMARK_TEMPLATE(BM_Join, JoinSimdSortMerge)->Args({1, 2});
BENCHMARK_TEMPLATE(BM_Join, JoinSimdSortMerge)->Args({1, 4});
BENCHMARK_TEMPLATE(BM_Join, JoinSimdSortMerge)->Args({1, 8});
BENCHMARK_TEMPLATE(BM_Join, JoinSimdSortMerge)->Args({1, 16});
BENCHMARK_TEMPLATE(BM_Join, JoinSimdSortMerge)->Args({1, 32});

BENCHMARK_TEMPLATE(BM_Join, JoinHash)->Args({8, 1});
BENCHMARK_TEMPLATE(BM_Join, JoinHash)->Args({8, 2});
BENCHMARK_TEMPLATE(BM_Join, JoinHash)->Args({8, 4});
BENCHMARK_TEMPLATE(BM_Join, JoinHash)->Args({8, 8});
BENCHMARK_TEMPLATE(BM_Join, JoinHash)->Args({8, 16});
BENCHMARK_TEMPLATE(BM_Join, JoinHash)->Args({8, 32});
BENCHMARK_TEMPLATE(BM_Join, JoinSortMerge)->Args({8, 1});
BENCHMARK_TEMPLATE(BM_Join, JoinSortMerge)->Args({8, 2});
BENCHMARK_TEMPLATE(BM_Join, JoinSortMerge)->Args({8, 4});
BENCHMARK_TEMPLATE(BM_Join, JoinSortMerge)->Args({8, 8});
BENCHMARK_TEMPLATE(BM_Join, JoinSortMerge)->Args({8, 16});
BENCHMARK_TEMPLATE(BM_Join, JoinSortMerge)->Args({8, 32});
BENCHMARK_TEMPLATE(BM_Join, JoinSimdSortMerge)->Args({8, 1});
BENCHMARK_TEMPLATE(BM_Join, JoinSimdSortMerge)->Args({8, 2});
BENCHMARK_TEMPLATE(BM_Join, JoinSimdSortMerge)->Args({8, 4});
BENCHMARK_TEMPLATE(BM_Join, JoinSimdSortMerge)->Args({8, 8});
BENCHMARK_TEMPLATE(BM_Join, JoinSimdSortMerge)->Args({8, 16});
BENCHMARK_TEMPLATE(BM_Join, JoinSimdSortMerge)->Args({8, 32});

BENCHMARK_TEMPLATE(BM_Join, JoinHash)->Args({16, 1});
BENCHMARK_TEMPLATE(BM_Join, JoinHash)->Args({16, 2});
BENCHMARK_TEMPLATE(BM_Join, JoinHash)->Args({16, 4});
BENCHMARK_TEMPLATE(BM_Join, JoinHash)->Args({16, 8});
BENCHMARK_TEMPLATE(BM_Join, JoinHash)->Args({16, 16});
BENCHMARK_TEMPLATE(BM_Join, JoinHash)->Args({16, 32});
BENCHMARK_TEMPLATE(BM_Join, JoinSortMerge)->Args({16, 1});
BENCHMARK_TEMPLATE(BM_Join, JoinSortMerge)->Args({16, 2});
BENCHMARK_TEMPLATE(BM_Join, JoinSortMerge)->Args({16, 4});
BENCHMARK_TEMPLATE(BM_Join, JoinSortMerge)->Args({16, 8});
BENCHMARK_TEMPLATE(BM_Join, JoinSortMerge)->Args({16, 16});
BENCHMARK_TEMPLATE(BM_Join, JoinSortMerge)->Args({16, 32});
BENCHMARK_TEMPLATE(BM_Join, JoinSimdSortMerge)->Args({16, 1});
BENCHMARK_TEMPLATE(BM_Join, JoinSimdSortMerge)->Args({16, 2});
BENCHMARK_TEMPLATE(BM_Join, JoinSimdSortMerge)->Args({16, 4});
BENCHMARK_TEMPLATE(BM_Join, JoinSimdSortMerge)->Args({16, 8});
BENCHMARK_TEMPLATE(BM_Join, JoinSimdSortMerge)->Args({16, 16});
BENCHMARK_TEMPLATE(BM_Join, JoinSimdSortMerge)->Args({16, 32});

}  // namespace hyrise
