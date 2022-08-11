#pragma once

#include <unordered_set>
#include <utility>

#include "partial_hash_index_impl.hpp"
#include "storage/index/abstract_table_index.hpp"

namespace hyrise {

class PartialHashIndexTest;

/**
 * Represents a table index using the tsl::sparse_map hash map where all hashed values are mapped to the RowIDs of their
 * occurrences in the original Chunks. This allows for faster lookup time while index build times are generally
 * increased compared to chunk-based indexes.
 * It can be constructed for a set of chunks of a table column and can later be modified by adding additional chunks
 * or removing already indexed chunks.
 */
class PartialHashIndex : public AbstractTableIndex {
  friend PartialHashIndexTest;

 public:
  PartialHashIndex() = delete;
  PartialHashIndex(const std::vector<std::pair<ChunkID, std::shared_ptr<Chunk>>>& chunks_to_index, const ColumnID);
  PartialHashIndex(const DataType data_type, const ColumnID);

  /**
   * Adds the given chunks to this index. If a chunk is already indexed, it is not indexed again.
   *
   * @return The number of added chunks.
   */
  size_t insert_entries(const std::vector<std::pair<ChunkID, std::shared_ptr<Chunk>>>&);

  /**
   * Removes the given chunks from this index. If a chunk is not indexed, nothing will happen.
   *
   * @return The number of removed chunks.
   */
  size_t remove_entries(const std::vector<ChunkID>&);

 protected:
  Iterator _cbegin() const final;
  Iterator _cend() const final;
  Iterator _null_cbegin() const final;
  Iterator _null_cend() const final;
  size_t _memory_usage() const final;

  IteratorPair _range_equals(const AllTypeVariant& value) const final;
  std::pair<IteratorPair, IteratorPair> _range_not_equals(const AllTypeVariant& value) const final;

  bool _is_index_for(const ColumnID column_id) const final;
  std::unordered_set<ChunkID> _get_indexed_chunk_ids() const final;

 private:
  const ColumnID _column_id;
  std::shared_ptr<BasePartialHashIndexImpl> _impl;
};

}  // namespace hyrise
