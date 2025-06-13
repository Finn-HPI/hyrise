#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "operators/operator_join_predicate.hpp"
#include "resolve_type.hpp"
#include "storage/base_segment_accessor.hpp"
#include "storage/table.hpp"
#include "type_comparison.hpp"
#include "types.hpp"
#include "utils/assert.hpp"

namespace hyrise {

template <typename T>
class FieldAccessor {
 public:
  FieldAccessor(const Table& table, const ColumnID column_id) {
    _create_accesors(table, column_id);
  }

  std::optional<T> value(const RowID& row_id) {
    return _accessors[row_id.chunk_id]->access(row_id.chunk_offset);
  }

 private:
  void _create_accesors(const Table& table, const ColumnID column_id) {
    _accessors.resize(table.chunk_count());
    const auto chunk_count = table.chunk_count();
    for (auto chunk_id = ChunkID{0}; chunk_id < chunk_count; ++chunk_id) {
      const auto chunk = table.get_chunk(chunk_id);
      Assert(chunk, "Physically deleted chunk should not reach this point, see get_chunk / #1686.");

      const auto& segment = chunk->get_segment(column_id);
      _accessors[chunk_id] = create_segment_accessor<T>(segment);
    }
  }

  std::vector<std::unique_ptr<AbstractSegmentAccessor<T>>> _accessors;
};

}  // namespace hyrise
