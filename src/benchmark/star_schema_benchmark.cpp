#include <iostream>

#include <boost/algorithm/string.hpp>

#include "benchmark_runner.hpp"
#include "cli_config_parser.hpp"
#include "file_based_benchmark_item_runner.hpp"
#include "hyrise.hpp"
#include "ssb/ssb_table_generator.hpp"

/**
 * The Star Schema Benchmark was introduced by O'Neil et al. "The Star Schema Benchmark and Augmented Fact Table
 * Indexing". It is based on an adapted an normalized TPC-H dataset. The data is generated by the provided data
 * generator, the queries are directly taken from the specification.
 *
 * For further details, see:
 *     - https://doi.org/10.1007/978-3-642-10424-4_17
 *     - https://www.cs.umb.edu/~poneil/StarSchemaB.PDF
 */

using namespace hyrise;  // NOLINT(build/namespaces)

int main(int argc, char* argv[]) {
  auto cli_options = BenchmarkRunner::get_basic_cli_options("Hyrise Star Schema Benchmark");

  // clang-format off
  cli_options.add_options()
    ("s,scale", "Database scale factor (10.0 ~ 10 GB)", cxxopts::value<float>()->default_value("10"))
    ("q,queries", "Specify queries to run (comma-separated query ids, e.g. \"--queries 1.1,1.3,3.2\"), default is all", cxxopts::value<std::string>()->default_value("all"));  // NOLINT(whitespace/line_length)
  // clang-format on

  // Parse command line args
  const auto cli_parse_result = cli_options.parse(argc, argv);

  if (CLIConfigParser::print_help_if_requested(cli_options, cli_parse_result)) {
    return 0;
  }

  const auto queries_str = cli_parse_result["queries"].as<std::string>();
  const auto scale_factor = cli_parse_result["scale"].as<float>();
  const auto config = CLIConfigParser::parse_cli_options(cli_parse_result);

  auto query_subset = std::optional<std::unordered_set<std::string>>{};
  if (queries_str == "all") {
    std::cout << "- Running all queries\n";
  } else {
    std::cout << "- Running subset of queries: " << queries_str << '\n';

    // "a, b, c, d" -> ["a", " b", " c", " d"]
    auto query_subset_untrimmed = std::vector<std::string>{};
    boost::algorithm::split(query_subset_untrimmed, queries_str, boost::is_any_of(","));

    // ["a", " b", " c", " d"] -> ["a", "b", "c", "d"]
    query_subset.emplace();
    for (const auto& query_name : query_subset_untrimmed) {
      query_subset->emplace(boost::trim_copy(query_name));
    }
  }

  auto context = BenchmarkRunner::create_context(*config);

  std::cout << "- SSB scale factor is " << scale_factor << '\n';
  context.emplace("scale_factor", scale_factor);
  // We cannot verify the results for larger scale factors (SFs) since SQLite overflows integers for aggregation
  // results. We could use dedicated result sets in these cases similar to TPC-DS. However, we need to generate these
  // result sets using a trustworthy DBMS, such as Postgres. We decided against this approach for two reasons. First,
  // we do not consider this worth the effort for now. Second, it does not solve the issue of requiring specific SFs:
  // The result set would only be valid for the SF it was generated on. Thus, we simply limit verification to rather
  // small SFs.
  // We empirically figured out that errors do not occur for SF <= 0.1 (0.11 to account for float comparison).
  Assert(!config->verify || scale_factor < 0.11,
         "SSB result verification is only supported fo scale factors <= 0.1 (--scale 0.1).");

  // Different from the TPC-H benchmark, where the table and query generators are immediately embedded in Hyrise, the
  // SSB implementation calls those generators externally. This is because we would get linking conflicts if we were
  // to include both generators.

  // Try to find dbgen binary.
  const auto executable_path = std::filesystem::canonical(std::string{argv[0]}).remove_filename();
  const auto ssb_dbgen_path = executable_path / "third_party/ssb-dbgen";
  Assert(std::filesystem::exists(ssb_dbgen_path / "dbgen"),
         std::string{"SSB dbgen not found at "} + ssb_dbgen_path.c_str());
  const auto query_path = executable_path / "../resources/benchmark/ssb/queries";
  const auto csv_meta_path = executable_path / "../resources/benchmark/ssb/schema";

  // Create the ssb_data directory (if needed) and generate the ssb_data/sf-... path.
  auto ssb_data_path_str = std::stringstream{};
  ssb_data_path_str << "ssb_data/sf-" << std::noshowpoint << scale_factor;
  std::filesystem::create_directories(ssb_data_path_str.str());
  // Success of create_directories is guaranteed by the call to fs::canonical, which fails on invalid paths.
  const auto ssb_data_path = std::filesystem::canonical(ssb_data_path_str.str());

  std::cout << "- Using SSB dbgen from " << ssb_dbgen_path << '\n';
  std::cout << "- Storing SSB tables in " << ssb_data_path << '\n';

  // Create the table generator and item runner.
  auto table_generator =
      std::make_unique<SSBTableGenerator>(ssb_dbgen_path, csv_meta_path, ssb_data_path, scale_factor, config);

  auto benchmark_item_runner = std::make_unique<FileBasedBenchmarkItemRunner>(
      config, query_path, std::unordered_set<std::string>{}, query_subset);

  auto benchmark_runner =
      std::make_shared<BenchmarkRunner>(*config, std::move(benchmark_item_runner), std::move(table_generator), context);
  Hyrise::get().benchmark_runner = benchmark_runner;

  benchmark_runner->run();
}
