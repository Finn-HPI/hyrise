#include <memory>
#include <new>
#include <vector>

#include <random>
#include "benchmark/benchmark.h"
#include "buffer_benchmark_utils.hpp"
#include "hyrise.hpp"
#include "storage/buffer/buffer_manager.hpp"
#include "storage/buffer/jemalloc_resource.hpp"
#include "storage/buffer/zipfian_int_distribution.hpp"

namespace hyrise {

/**
 * Bechmark idea:
 * Sacle read ops with difedderne page size
 * Scale read ops with single page size and different DRAM size ratios
 * Use zipfian skews to test different hit and miss rates for single pahe size and diffeent dram size ratios
 * 
*/
template <YCSBTableAccessPattern AccessPattern, MigrationPolicy policy = LazyMigrationPolicy>
void BM_ycsb(benchmark::State& state) {
  constexpr auto GB = 1024 * 1024 * 1024;
  constexpr static auto PAGE_SIZE_TYPE = MIN_PAGE_SIZE_TYPE;
  constexpr static auto DEFAULT_DRAM_BUFFER_POOL_SIZE = 0.5 * GB;

  constexpr static auto NUM_OPERATIONS = 1000;
  constexpr static size_t DATABASE_SIZE = 1UL * GB;

  static YCSBTable table;
  static YCSBOperations operations;
  static boost::container::pmr::memory_resource* memory_resource;

  if (state.thread_index() == 0) {
    auto config = BufferManager::Config::from_env();
    config.dram_buffer_pool_size = DEFAULT_DRAM_BUFFER_POOL_SIZE;
    config.numa_buffer_pool_size = DEFAULT_DRAM_BUFFER_POOL_SIZE;
    config.memory_node = NumaMemoryNode{2};
    config.migration_policy = policy;

    Hyrise::get().buffer_manager = BufferManager(config);
#ifdef __APPLE__
    memory_resource = &Hyrise::get().buffer_manager;
#elif __linux__
    JemallocMemoryResource::get().reset();
    memory_resource = &JemallocMemoryResource::get();
#endif

    table = generate_ycsb_table<DATABASE_SIZE>(memory_resource);
    operations = generate_ycsb_operations<AccessPattern, NUM_OPERATIONS>(table.size(), 0.99);
    warmup(table, Hyrise::get().buffer_manager);
  }

  run_ycsb(state, table, operations, Hyrise::get().buffer_manager);
}

#define CONFIGURE_BENCHMARK(AccessPattern, Policy)                  \
  BENCHMARK(BM_ycsb<YCSBTableAccessPattern::AccessPattern, Policy>) \
      ->ThreadRange(1, 48)                                          \
      ->Iterations(1)                                               \
      ->Repetitions(1)                                              \
      ->UseRealTime()                                               \
      ->Name("BM_ycsb/" #AccessPattern "/" #Policy);

CONFIGURE_BENCHMARK(ReadHeavy, LazyMigrationPolicy)
CONFIGURE_BENCHMARK(Balanced, LazyMigrationPolicy)
CONFIGURE_BENCHMARK(WriteHeavy, LazyMigrationPolicy)

}  // namespace hyrise
