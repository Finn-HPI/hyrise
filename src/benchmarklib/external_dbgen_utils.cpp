#include "external_dbgen_utils.hpp"

#include <filesystem>
#include <utility>

#include "abstract_table_generator.hpp"
#include "utils/timer.hpp"

namespace hyrise {

void generate_csv_tables_with_external_dbgen(const std::string& dbgen_path, const std::vector<std::string>& table_names,
                                             const std::string& csv_meta_path, const std::string& tables_path,
                                             const float scale_factor, const std::string& additional_cli_args) {
  // NOLINTBEGIN(concurrency-mt-unsafe)
  // clang-tidy complains that system() is not thread-safe. We ignore this warning as we expect that users will not call
  // table generator executables in parallel.

  // Check if table data has already been generated (and converted to .bin by the FileBasedTableGenerator).
  if (!std::filesystem::exists(tables_path + "customer.bin")) {
    auto timer = Timer{};
    std::cout << "- Creating table data by calling external dbgen" << std::flush;

    std::filesystem::create_directory(tables_path);
    Assert(std::filesystem::exists(tables_path), "Creating tables folder failed");

    {
      // Call dbgen.
      auto cmd = std::stringstream{};
      // `2>` in a string seems to break Sublime Text's formatter, so it's split into two strings.
      cmd << "cd " << tables_path << " && " << dbgen_path << "/dbgen -f -s " << scale_factor << " "
          << additional_cli_args << " "
          << " -b " << dbgen_path << "/dists.dss >/dev/null 2"
          << ">/dev/null";
      auto ret = system(cmd.str().c_str());
      Assert(!ret, "Calling dbgen failed");
    }

    for (const auto& table_name : table_names) {
      // Rename tbl files generated by dbgen to csv so that the correct importer is used.
      std::filesystem::rename(tables_path + table_name + ".tbl", tables_path + table_name + ".csv");

      // Remove the trailing separator from each line as the CsvParser does not like them.
      {
        // sed on Mac requires a space between -i and '', on Linux it doesn't like it...
#ifdef __APPLE__
        const auto* const sed_inplace = "-i ''";
#else
        const auto* const sed_inplace = "-i''";
#endif

        auto cmd = std::stringstream{};
        cmd << "sed -Ee 's/\\|$//' " << sed_inplace << " " << tables_path << table_name << ".csv";
        const auto ret = system(cmd.str().c_str());
        Assert(!ret, "Removing trailing separators using sed failed");
      }

      // std::filesystem::copy does not seem to work. We could use symlinks here, but those would make reading the file
      // via ifstream more complicated.
      {
        auto cmd = std::stringstream{};
        cmd << "cp  " << csv_meta_path << "/" << table_name << ".csv.json " << tables_path << table_name << ".csv.json";
        const auto ret = system(cmd.str().c_str());
        Assert(!ret, "Copying csv.json files failed");
      }
    }

    std::cout << " (" << timer.lap_formatted() << ")" << std::endl;
  }
  // NOLINTEND(concurrency-mt-unsafe)
}

void remove_csv_tables(const std::string& tables_path) {
  if (std::filesystem::exists(tables_path + "customer.csv")) {
    auto cmd = std::stringstream{};
    cmd << "rm " << tables_path << "*.csv*";
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    const auto ret = system(cmd.str().c_str());
    Assert(!ret, "Removing csv/csv.json files failed");
  }
}

}  // namespace hyrise
