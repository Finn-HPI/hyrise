// Include before Fail() is defined
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Woverflow"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wattributes"
#pragma GCC diagnostic pop

#include <iostream>
#include <fstream>

#include "expression/expression_functional.hpp"
#include "hyrise.hpp"
#include "operators/print.hpp"
#include "operators/sort.hpp"
#include "operators/table_wrapper.hpp"
#include "optimizer/optimizer.hpp"
#include "optimizer/strategy/chunk_pruning_rule.hpp"
#include "optimizer/strategy/column_pruning_rule.hpp"
#include "optimizer/strategy/index_scan_rule.hpp"
#include "logical_query_plan/logical_plan_root_node.hpp"
#include "logical_query_plan/lqp_translator.hpp"
#include "logical_query_plan/predicate_node.hpp"
#include "logical_query_plan/stored_table_node.hpp"
#include "sql/sql_pipeline_builder.hpp"
#include "sql/sql_pipeline_statement.hpp"
#include "storage/chunk_encoder.hpp"
#include "storage/index/group_key/composite_group_key_index.hpp"
#include "storage/index/group_key/group_key_index.hpp"
#include "utils/load_table.hpp"
#include "utils/format_duration.hpp"
#include "utils/timer.hpp"
#include "visualization/pqp_visualizer.hpp"

#include "memory/dram_memory_resource.hpp"
#include "memory/umap_memory_resource.hpp"

#include "umap/RegionManager.hpp"
#include "umap/Buffer.hpp"

using namespace opossum::expression_functional;  // NOLINT

using namespace opossum;  // NOLINT

// Import

//constexpr auto TBL_FILE = "../../data/10mio_pings_no_id_int.tbl";
constexpr auto TBL_FILE = "../../data/400mio_pings_no_id_int.tbl";
constexpr auto WORKLOAD_FILE = "../../data/workload.csv";
constexpr auto CONFIG_PATH = "../../data/config";
constexpr auto CHUNK_SIZE = size_t{40'000'000};
//constexpr auto CHUNK_SIZE = size_t{10};
constexpr auto TABLE_NAME = "PING";

constexpr auto BUFFER_EVICTION_TABLE_NAME = "BUFFERTABLE";
constexpr auto BUFFER_EVICTION_CHUNK_SIZE = size_t{400'000'000};

constexpr auto SORT_MODE = SortMode::Ascending;
constexpr auto EXECUTION_COUNT = 100;

//Chunk encodings copied from ping data micro benchmark 
const auto CHUNK_ENCODINGS = std::vector{
  SegmentEncodingSpec{EncodingType::Dictionary},
  SegmentEncodingSpec{EncodingType::Unencoded},
  SegmentEncodingSpec{EncodingType::LZ4},
  SegmentEncodingSpec{EncodingType::RunLength},
  SegmentEncodingSpec{EncodingType::FrameOfReference, VectorCompressionType::SimdBp128}
};

// multi column index candidates
const std::vector<std::vector<int>> MULTI_COLUMN_INDEXES {{0, 1}, {1, 0}, {0, 4}, {4, 0}, {0, 3}, {3, 0}, {1, 2}, {2, 1}, {1, 3}, {3, 1}, 
                                                          {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0},
                                                          {1, 2, 3}, {1, 3, 2}, {2, 1, 3}, {2, 3, 1}, {3, 1, 2}, {3, 2, 1},
                                                          {1, 2, 3, 4}, {1, 3, 2, 4}, {2, 1, 3, 4}, {2, 3, 1, 4}, {3, 1, 2, 4}, {3, 2, 1, 4},
                                                          {4, 2, 3, 1}, {4, 3, 2, 1}, {2, 4, 3, 1}, {2, 3, 4, 1}, {3, 4, 2, 1}, {3, 2, 4, 1},
                                                          {1, 4, 3, 2}, {1, 3, 4, 2}, {4, 1, 3, 2}, {4, 3, 1, 2}, {3, 1, 4, 2}, {3, 4, 1, 2},
                                                          {1, 2, 4, 3}, {1, 4, 2, 3}, {2, 1, 4, 3}, {2, 4, 1, 3}, {4, 1, 2, 3}, {4, 2, 1, 3} 
                                                         };

std::map<std::pair<std::vector<int>, ChunkID>, std::shared_ptr<AbstractIndex>> multi_indexes;

// Export 

constexpr auto MEMORY_CONSUMPTION_FILE = "../../out/config_results/memory_consumption.csv";
constexpr auto PERFORMANCE_FILE = "../../out/config_results/performance.csv";

// returns a vector with all lines of the file
std::vector<std::vector<std::string>> read_file(const std::string file) {
  std::ifstream f(file);
  std::string line;
  std::vector<std::vector<std::string>> file_values;

  std::string header;
  std::getline(f, header);

  while (std::getline(f, line)){
    std::vector<std::string> line_values;
    std::istringstream linestream(line);
    std::string value;

    while (std::getline(linestream, value, ',')){
     line_values.push_back(value);
    }

    file_values.push_back(line_values);
  }

  return file_values;  
} 

// returns all segmenst of a chunk 
Segments get_segments_of_chunk(const std::shared_ptr<const Table>& input_table, ChunkID chunk_id){
  Segments segments{};
  for (auto column_id = ColumnID{0}; column_id < input_table->column_count(); ++column_id) {
    segments.emplace_back(input_table->get_chunk(chunk_id)->get_segment(column_id));
  }
  return segments;
} 

// returns a vector of indexed chunks for a given column
std::vector<ChunkID> get_indexed_chunk_ids(const std::shared_ptr<const Table>& table, const ColumnID column_id){
  std::vector<ChunkID> indexed_chunk_ids = {};
  const auto chunk_count = table->chunk_count();
  // Iterate over chunks to check if for the given column an index exists on a segment 
  for (auto chunk_id = ChunkID{0}; chunk_id < chunk_count; ++chunk_id) {
    const auto& index = table->get_chunk(chunk_id)->get_index(SegmentIndexType::GroupKey, std::vector<ColumnID>{column_id});
    if (index){
      indexed_chunk_ids.emplace_back(chunk_id);
    }
  }
  return indexed_chunk_ids;
} 

/**
 * This function takes a file path to a CSV file containing the workload and returns the queries in form of LQP-based
 * queries together with their frequency. The reason to use LQPs is that we can later selectively apply optimizer rules
 * (most importantly the chunk pruning rule).
 * Alternatives are SQL-based queries, which either do not use the optimizer (no chunk pruning) or the complete
 * optimizer (we lose control over the predicate order), or creating PQPs (we would need to manually prune chunks).
 */
std::vector<std::shared_ptr<AbstractLQPNode>> load_queries_from_csv(const std::string workload_file) {
  std::vector<std::shared_ptr<AbstractLQPNode>> output_queries;

  const auto csv_lines = read_file(workload_file);

  auto previous_query_id = int64_t{-1};
  auto previous_predicate_selectivity = 17.0;
  std::shared_ptr<AbstractLQPNode> current_node;
  
  const auto& table = Hyrise::get().storage_manager.get_table(TABLE_NAME);
  const auto column_names = table->column_names();

  // Create initial node 
  auto stored_table_node = StoredTableNode::make(TABLE_NAME);
  std::shared_ptr<AbstractLQPNode> previous_node = stored_table_node;

  // Get values from workload csv file
  for (const auto& csv_line : csv_lines) {
    const auto query_id = std::stol(csv_line[0]);
    const auto scan_id = std::stol(csv_line[1]);
    const auto predicate_str = csv_line[2];
    const auto column_id = ColumnID{static_cast<uint16_t>(std::stoi(csv_line[3]))};
    const auto predicate_selectivity = stof(csv_line[4]);
    const auto search_value_0 = stoi(csv_line[7]);
    const auto search_value_1 = stoi(csv_line[8]);

    Assert(query_id >= previous_query_id,
           "Queries are expected to be sorted ascendingly by: query ID ascendingly.");
    Assert(query_id == previous_query_id || scan_id == 0,
           "Queries are expected to start with scan id 0");
    Assert(query_id != previous_query_id || predicate_selectivity >= previous_predicate_selectivity,
           "Queries are expected to be sorted ascendingly by: query ID ascendingly & selectivity descendingly.");

    // If query id has changed store current node in output queries and create new initial node 
    if (query_id > 0 && query_id != previous_query_id) {
      output_queries.emplace_back(current_node);
      stored_table_node = StoredTableNode::make(TABLE_NAME);
      previous_node = stored_table_node;
    }

    const auto lqp_column = stored_table_node->get_column(column_names[column_id]);
    if (predicate_str == "Between") {
      Assert(search_value_1 > -1, "Between predicate with missing second search value.");
      current_node = PredicateNode::make(between_inclusive_(lqp_column, search_value_0, search_value_1), previous_node);
    } else if (predicate_str == "LessThanEquals") {
      current_node = PredicateNode::make(less_than_equals_(lqp_column, search_value_0), previous_node);
    }

    // Set scan type to index scan if the scan is the first scan of a query and at least one segment 
    // of the scan column has an index 
    // Partial index configurations are handeled by LQPTranslater line 215 -231
    if (scan_id == 0 && !get_indexed_chunk_ids(table, column_id).empty()) {
      auto index_node = std::dynamic_pointer_cast<PredicateNode>(current_node);
      index_node->scan_type = ScanType::IndexScan;
    }

    previous_query_id = query_id;
    previous_predicate_selectivity = predicate_selectivity;
    previous_node = current_node;
  }
  output_queries.emplace_back(current_node);  // Store last query

  return output_queries;
}


void load_buffer_eviction_table(UmapMemoryResource* umap_resource) {
  std::cout << "Loading buffer eviction table ... ";
  
  // Create new table to evict buffer during query executions
  const auto buffer_table = load_table(TBL_FILE, BUFFER_EVICTION_CHUNK_SIZE);

  // move all segments on SSD
  for (auto chunk_id = ChunkID{0}; chunk_id < buffer_table->chunk_count(); ++chunk_id) {
    auto chunk = std::make_shared<Chunk>(get_segments_of_chunk(buffer_table, chunk_id));
    for (ColumnID column_id = ColumnID{0}; column_id < chunk->column_count(); ++column_id) {
      const auto segment = chunk->get_segment(column_id);

      auto resource = umap_resource;
      auto allocator = PolymorphicAllocator<void>{resource};

      const auto migrated_segment = segment->copy_using_allocator(allocator);
      buffer_table->get_chunk(chunk_id)->replace_segment(column_id, migrated_segment);
    }
  }
  
  // add table to storage manager
  auto& storage_manager = Hyrise::get().storage_manager;
  storage_manager.add_table(BUFFER_EVICTION_TABLE_NAME, buffer_table);
  std::cout << "done " << std::endl;
}

/**
 * This function executes a table scan on the timestamp column of the buffer eviction table to pollute
 * the umap buffer
 */
void execute_buffer_eviction_query() {
  // Create initial node 
  auto stored_table_node = StoredTableNode::make(BUFFER_EVICTION_TABLE_NAME);
  auto node = PredicateNode::make(less_than_equals_(stored_table_node->get_column("timestamp"), 1548975682), stored_table_node);

  const auto root_node = LogicalPlanRootNode::make(std::move(node));

  const auto optimized_node = root_node->left_input();
  root_node->set_left_input(nullptr);

  const auto pqp = LQPTranslator{}.translate_node(optimized_node);

  std::cout << "Query Executed: " << Umap::RegionManager::getInstance().get_buffer_h() << std::endl;

  // Execute pgp 
  const auto tasks = OperatorTask::make_tasks_from_operator(pqp);
  Hyrise::get().scheduler()->schedule_and_wait_for_tasks(tasks);

  std::cout << "Inter Query Executed: " << Umap::RegionManager::getInstance().get_buffer_h() << std::endl;
}

void flush_umap_buffer() {
  auto& umap_region = Umap::RegionManager::getInstance();
  std::cout << "buffer status: " << Umap::RegionManager::getInstance().get_buffer_h() << std::endl;

  // Umap evict region
  //auto const tiering_dev = getenv("TIERING_DEV"); 
  //auto umap_region_descriptor = umap_region.containing_region(tiering_dev);
  //umap_region.get_buffer_h()->evict_region(umap_region_descriptor);
  
  // evict all  
  umap_region.get_evict_manager()->EvictAll();

  // flush umap buffer
  umap_region.flush_buffer();

  std::cout << "evicted buffer status: "<< Umap::RegionManager::getInstance().get_buffer_h() << std::endl;
}

/**
 * Creates a svg file taht visualizes the physical query plan 
 */
void visualize_pqps(const std::vector<std::shared_ptr<AbstractOperator>> pqps, const std::string conf_name, const std::string query_id) {
  GraphvizConfig graphviz_config;
  graphviz_config.format = "svg";
  std::string path = conf_name + "_" + query_id + "_pqp.svg";
  PQPVisualizer{graphviz_config, {}, {}, {}}.visualize(pqps, path);
}

/**
 * Takes a pair of an LQP-based query and the frequency, partially optimizes the query (only chunk and column pruning
 * for now), translates the query, and executes the query (single-threaded).
 */
float partially_optimize_translate_and_execute_query(const std::shared_ptr<AbstractLQPNode>& lqp_query, const std::string conf_name, const std::string query_id) {

  //std::cout << "LQP Query" << std::endl;
  //std::cout << *lqp_query << std::endl;

  // Run chunk and column pruning rules. Kept it quite simple for now. Take a look at Optimizer::optimize() in case
  // problems occur. The following code is taken from optimizer.cpp. In case the new root is confusing to you, take a
  // look there.
  const auto root_node = LogicalPlanRootNode::make(std::move(lqp_query));

  const auto chunk_pruning_rule = ChunkPruningRule();
  chunk_pruning_rule.apply_to_plan(root_node);

  const auto column_pruning_rule = ColumnPruningRule();
  column_pruning_rule.apply_to_plan(root_node);

  const auto index_scan_rule = IndexScanRule();
  index_scan_rule.apply_to_plan(root_node);

  // Remove LogicalPlanRootNode
  const auto optimized_node = root_node->left_input();
  root_node->set_left_input(nullptr);

  std::vector<size_t> runtimes;
  for (auto count = size_t{0}; count < EXECUTION_COUNT; ++count) {

    //std::cout << "Start: " << Umap::RegionManager::getInstance().get_buffer_h() << std::endl;

    //flush_umap_buffer();

    Timer timer;
    const auto pqp = LQPTranslator{}.translate_node(optimized_node);

    //std::cout << "PQP Query" << std::endl;
    //std::cout << *pqp << std::endl;

    const auto tasks = OperatorTask::make_tasks_from_operator(pqp);

    Hyrise::get().scheduler()->schedule_and_wait_for_tasks(tasks);
    const auto runtime = static_cast<size_t>(timer.lap().count());
    runtimes.push_back(runtime);

    //visualize_pqps({pqp}, conf_name, query_id);
    
    //execute_buffer_eviction_query();
  }

  auto runtime_accumulated = size_t{0};
  for (const auto runtime : runtimes) {
    runtime_accumulated += runtime;
  }

  const auto avg_accumulated_runtime = static_cast<float>(runtime_accumulated) / static_cast<float>(runtimes.size());
  //std::cout << std::fixed << "Execution took in average " << avg_accumulated_runtime << " ns" << std::endl;

  return avg_accumulated_runtime;
}

int main() {
  auto& storage_manager = Hyrise::get().storage_manager;

  std::ofstream memory_consumption_csv_file(MEMORY_CONSUMPTION_FILE);
  std::ofstream performance_csv_file(PERFORMANCE_FILE);

  memory_consumption_csv_file << "CONFIG_NAME, MEMORY_CONSUMPTION,INDEX,TABLE\n";
  performance_csv_file << "CONFIG_NAME,QUERY_ID,EXECUTION_TIME\n";

  Assert(getenv("UMAP_BUFSIZE") != NULL, "Environment variable UMAP_BUFSIZE should be set for umap to work as expected");
  Assert(getenv("UMAP_PAGESIZE") != NULL, "Environment variable UMAP_PAGESIZE should be set for umap to work as expected");
  Assert(getenv("TIERING_DEV") != NULL, "Environment variable TIERING_DEV should be set to specify the device");

  // Create new ploymorphic resource
  static auto global_umap_resource = new UmapMemoryResource("global");
  (void) global_umap_resource;

  // Load buffer eviction table 
  //load_buffer_eviction_table(global_umap_resource);

  for (const auto& entry : std::filesystem::directory_iterator(CONFIG_PATH)) {
    const auto conf_path = entry.path();
    const auto conf_name = conf_path.stem();
    const auto filename = conf_path.filename().string();

    // check that file name is csv file
    if (filename.find(".csv") == std::string::npos) {
      std::cout << "Skipping " << conf_path << std::endl;
      continue;
    }

    // Create a new PING table that is used for the actual benchmark configuration
    std::cout << "Loading PING table ... ";
    Timer load_timer;
    const auto table = load_table(TBL_FILE, CHUNK_SIZE);
    std::cout << "done (" << format_duration(load_timer.lap()) << ")" << std::endl;
    
    // Load configuration from csv file
    const auto conf = read_file(conf_path);

    // Apply specified configuration schema
    std::cout << "Preparing table (encoding, sorting, ...) with given configuration: " << conf_name << " ... ";
    Timer preparation_timer;

    const auto sorted_table = std::make_shared<Table>(table->column_definitions(), 
      TableType::Data, CHUNK_SIZE, UseMvcc::No);
    const auto chunk_count = table->chunk_count();

    auto index_memory_consumption = size_t{0};
    auto conf_line_count = 0;
    for (auto chunk_id = ChunkID{0}; chunk_id < chunk_count; ++chunk_id) {
      const auto conf_chunk_id = ChunkID{static_cast<uint16_t>(std::stoi(conf[conf_line_count][0]))};
      const auto conf_chunk_sort_column_id = ColumnID{static_cast<uint16_t>(std::stoi(conf[conf_line_count][3]))};
      Assert(chunk_id == conf_chunk_id,
           "Expected chunk id does not match to chunk id in configuration file");
      
      // Sort 

      // Create single chunk table
      auto chunk = std::make_shared<Chunk>(get_segments_of_chunk(table, chunk_id));
      std::vector<std::shared_ptr<Chunk>> single_chunk_vector = {chunk};
      auto single_chunk_table = std::make_shared<Table>(table->column_definitions(), TableType::Data, std::move(single_chunk_vector), UseMvcc::No);

      auto table_wrapper = std::make_shared<TableWrapper>(single_chunk_table);
      table_wrapper->execute();

      std::shared_ptr<Chunk> added_chunk;

      // Sort single chunk table
      if (conf_chunk_sort_column_id < single_chunk_table->column_count()) {
        auto sort = std::make_shared<Sort>(
          table_wrapper, std::vector<SortColumnDefinition>{
            SortColumnDefinition{conf_chunk_sort_column_id, SORT_MODE}},CHUNK_SIZE, Sort::ForceMaterialization::Yes);
        sort->execute();
        const auto sorted_single_chunk_table = sort->get_output();

        // Add sorted chunk to sorted table
        // Note: we do not care about MVCC at all at the moment
        sorted_table->append_chunk(get_segments_of_chunk(sorted_single_chunk_table, ChunkID{0}));
        added_chunk = sorted_table->get_chunk(chunk_id);
        added_chunk->finalize();
        // Set order by for chunk 
        added_chunk->set_individually_sorted_by(SortColumnDefinition(conf_chunk_sort_column_id, SORT_MODE));
      } else {
        // append unsorted chunk to sorted table 
        sorted_table->append_chunk(get_segments_of_chunk(single_chunk_table, ChunkID{0}));
        added_chunk = sorted_table->get_chunk(chunk_id);
        added_chunk->finalize();
      }
     
      // Encode segments of sorted single chunk table

      for (ColumnID column_id = ColumnID{0}; column_id < added_chunk->column_count(); ++column_id) {
        const auto conf_column_id = ColumnID{static_cast<uint16_t>(std::stoi(conf[conf_line_count][1]))};
        Assert(column_id == conf_column_id,
           "Expected column id does not match column id in configuration file");

        const auto conf_segment_sort_column_id = ColumnID{static_cast<uint16_t>(std::stoi(conf[conf_line_count][3]))};
        Assert(conf_chunk_sort_column_id == conf_segment_sort_column_id,
           "Different sort configurations for a single chunk in configuration file");

        //Encode segment with specified encoding 
        const auto encoding_id = static_cast<uint16_t>(std::stoi(conf[conf_line_count][2]));
        const auto encoding = CHUNK_ENCODINGS[encoding_id];
        const auto segment = added_chunk->get_segment(column_id);

        Assert(encoding_id < CHUNK_ENCODINGS.size(), 
          "Undefined encoding specified in configuration file");

        const auto encoded_segment = ChunkEncoder::encode_segment(segment, segment->data_type(), encoding);

        // Move segments of sorted single chunk table to defined storage medium
        const auto storage_id = static_cast<uint16_t>(std::stoi(conf[conf_line_count][5]));
        //auto allocator = PolymorphicAllocator<void>{};

        if (storage_id > 0) {
           auto resource = global_umap_resource;
           auto allocator = PolymorphicAllocator<void>{resource};

           const auto migrated_segment = encoded_segment->copy_using_allocator(allocator);
           sorted_table->get_chunk(chunk_id)->replace_segment(column_id, migrated_segment);
           //std::cout << "Segment (" << chunk_id << "," << column_id << ")" << std::endl;
        } else {
          auto allocator = PolymorphicAllocator<void>{};
          const auto migrated_segment = encoded_segment->copy_using_allocator(allocator);
          added_chunk->replace_segment(column_id, migrated_segment);
        }

        //Store index columns 

        const auto index_conf = static_cast<uint16_t>(std::stoi(conf[conf_line_count][4]));
        // Create single column index 
        if (index_conf == 1) {
          Assert(encoding_id == 0, "Tried to set index on a not dictionary encoded segment");
          const auto added_index = added_chunk->create_index<GroupKeyIndex>(std::vector<ColumnID>{column_id});
          index_memory_consumption += added_index->memory_consumption();

          if (storage_id > 0) {
            auto resource = global_umap_resource;
            auto allocator = PolymorphicAllocator<void>{resource};

            const auto  migrated_index = added_index->copy_using_allocator(allocator);
            sorted_table->get_chunk(chunk_id)->replace_index(added_index, migrated_index);
            //std::cout << "Index (" << chunk_id << "," << column_id << ")" << std::endl;
          }

        // Create multi column index
        if (index_conf > 1) {
          Assert(encoding_id == 0, "Tried to set index on a not dictionary encoded segment");
          const auto multi_column_index_id = index_conf - 2;

          // Check if index was already created 
          if (multi_indexes.count({MULTI_COLUMN_INDEXES[multi_column_index_id], chunk_id}) == 0) {
            auto column_ids = std::vector<ColumnID>{};
            
            for (const auto& index_column : MULTI_COLUMN_INDEXES[multi_column_index_id]) {
              column_ids.emplace_back(index_column);
            }

            const auto& index = added_chunk->create_index<CompositeGroupKeyIndex>(column_ids);
            index_memory_consumption += added_index->memory_consumption();

            multi_indexes.insert({{MULTI_COLUMN_INDEXES[multi_column_index_id], chunk_id}, index});

            if (storage_id > 0) {
              auto resource = global_umap_resource;
              auto allocator = PolymorphicAllocator<void>{resource};

              const auto  migrated_index = added_index->copy_using_allocator(allocator);
              sorted_table->get_chunk(chunk_id)->replace_index(added_index, migrated_index);
              //std::cout << "Index (" << chunk_id << "," << column_id << ")" << std::endl;
            }
          }
        }
      }

      ++conf_line_count;
      }
    }

    //Print::print(sorted_table);

    std::cout << " done (" << format_duration(preparation_timer.lap()) << ")" << std::endl;

    storage_manager.add_table(TABLE_NAME, sorted_table);

    // Write memory usage of indexes and table to memory consumption csv file 
    auto mem_usage = sorted_table->memory_usage(MemoryUsageCalculationMode::Full) + index_memory_consumption;
    memory_consumption_csv_file << conf_name << "," << mem_usage << "," << index_memory_consumption << "," << sorted_table->memory_usage(MemoryUsageCalculationMode::Full) <<"\n";

    // We load queries here, as the construction of the queries needs the existing actual table
    const auto queries = load_queries_from_csv(WORKLOAD_FILE);

    std::cout << "Execute benchmark queries ... ";
    auto query_id = size_t{0};
    for (auto const& query : queries) {
      const auto query_runtime = partially_optimize_translate_and_execute_query(query, conf_name, std::to_string(query_id));
      performance_csv_file << conf_name << "," << query_id << "," << query_runtime << "\n";
      query_id += 1;
    }
    std::cout << " done" << std::endl;
    
    storage_manager.drop_table(TABLE_NAME);
  }

  memory_consumption_csv_file.close();
  performance_csv_file.close();

}
