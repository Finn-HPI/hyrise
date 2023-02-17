#include "stored_table_node.hpp"

#include "expression/expression_functional.hpp"
#include "expression/expression_utils.hpp"
#include "expression/lqp_column_expression.hpp"
#include "hyrise.hpp"
#include "lqp_utils.hpp"
#include "statistics/table_statistics.hpp"
#include "storage/index/chunk_index_statistics.hpp"
#include "storage/storage_manager.hpp"
#include "storage/table.hpp"
#include "utils/assert.hpp"
#include "utils/column_pruning_utils.hpp"

namespace {

using namespace hyrise;  // NOLINT(build/namespaces)

bool contains_any_column_id(const std::vector<ColumnID>& search_columns, const std::vector<ColumnID>& columns) {
  return std::any_of(search_columns.cbegin(), search_columns.cend(), [&](const auto& search_column_id) {
    return std::find(columns.cbegin(), columns.cend(), search_column_id) != columns.cend();
  });
}

std::vector<std::shared_ptr<AbstractExpression>> find_expressions(
    const std::vector<ColumnID>& column_ids,
    const std::vector<std::shared_ptr<AbstractExpression>>& output_expressions) {
  auto expressions = std::vector<std::shared_ptr<AbstractExpression>>{};
  expressions.reserve(column_ids.size());

  for (const auto column_id : column_ids) {
    for (const auto& expression : output_expressions) {
      Assert(expression->type == ExpressionType::LQPColumn, "Invalid output expression for StoredTableNode");
      const auto& column_expression = static_cast<LQPColumnExpression&>(*expression);
      if (column_expression.original_column_id == column_id) {
        expressions.emplace_back(expression);
        break;
      }
    }
  }

  Assert(expressions.size() == column_ids.size(), "Could not resolve all column IDs to expressions");
  return expressions;
}

}  // namespace

namespace hyrise {

using namespace expression_functional;  // NOLINT(build/namespaces)

StoredTableNode::StoredTableNode(const std::string& init_table_name)
    : AbstractLQPNode(LQPNodeType::StoredTable), table_name(init_table_name) {}

std::shared_ptr<LQPColumnExpression> StoredTableNode::get_column(const std::string& name) const {
  const auto table = Hyrise::get().storage_manager.get_table(table_name);
  const auto column_id = table->column_id_by_name(name);
  return std::make_shared<LQPColumnExpression>(shared_from_this(), column_id);
}

void StoredTableNode::set_pruned_chunk_ids(const std::vector<ChunkID>& pruned_chunk_ids) {
  DebugAssert(std::is_sorted(pruned_chunk_ids.begin(), pruned_chunk_ids.end()), "Expected sorted vector of ChunkIDs");
  DebugAssert(std::adjacent_find(pruned_chunk_ids.begin(), pruned_chunk_ids.end()) == pruned_chunk_ids.end(),
              "Expected vector of unique ChunkIDs");

  _pruned_chunk_ids = pruned_chunk_ids;
}

const std::vector<ChunkID>& StoredTableNode::pruned_chunk_ids() const {
  return _pruned_chunk_ids;
}

void StoredTableNode::set_pruned_column_ids(const std::vector<ColumnID>& pruned_column_ids) {
  DebugAssert(std::is_sorted(pruned_column_ids.begin(), pruned_column_ids.end()),
              "Expected sorted vector of ColumnIDs");
  DebugAssert(std::adjacent_find(pruned_column_ids.begin(), pruned_column_ids.end()) == pruned_column_ids.end(),
              "Expected vector of unique ColumnIDs");

  // It is valid for an LQP to not use any of the table's columns (e.g., SELECT 5 FROM t). We still need to include at
  // least one column in the output of this node, which is used by Table::size() to determine the number of 5's.
  const auto stored_column_count = Hyrise::get().storage_manager.get_table(table_name)->column_count();
  Assert(pruned_column_ids.size() < static_cast<size_t>(stored_column_count), "Cannot exclude all columns from Table.");

  _pruned_column_ids = pruned_column_ids;

  // Rebuilding this lazily the next time `output_expressions()` is called
  _output_expressions.reset();
}

const std::vector<ColumnID>& StoredTableNode::pruned_column_ids() const {
  return _pruned_column_ids;
}

std::string StoredTableNode::description(const DescriptionMode mode) const {
  const auto stored_table = Hyrise::get().storage_manager.get_table(table_name);

  std::ostringstream stream;
  stream << "[StoredTable] Name: '" << table_name << "' pruned: ";
  stream << _pruned_chunk_ids.size() << "/" << stored_table->chunk_count() << " chunk(s), ";
  stream << _pruned_column_ids.size() << "/" << stored_table->column_count() << " column(s)";

  return stream.str();
}

std::vector<std::shared_ptr<AbstractExpression>> StoredTableNode::output_expressions() const {
  // Need to initialize the expressions lazily because (a) they will have a weak_ptr to this node and we can't obtain
  // that in the constructor and (b) because we don't have column pruning information in the constructor
  if (!_output_expressions) {
    const auto table = Hyrise::get().storage_manager.get_table(table_name);

    // Build `_expression` with respect to the `_pruned_column_ids`
    const auto num_unpruned_columns = table->column_count() - _pruned_column_ids.size();
    _output_expressions = std::vector<std::shared_ptr<AbstractExpression>>(num_unpruned_columns);

    auto pruned_column_ids_iter = _pruned_column_ids.begin();
    auto output_column_id = ColumnID{0};
    const auto column_count = table->column_count();
    for (auto stored_column_id = ColumnID{0}; stored_column_id < column_count; ++stored_column_id) {
      // Skip `stored_column_id` if it is in the sorted vector `_pruned_column_ids`
      if (pruned_column_ids_iter != _pruned_column_ids.end() && stored_column_id == *pruned_column_ids_iter) {
        ++pruned_column_ids_iter;
        continue;
      }

      (*_output_expressions)[output_column_id] =
          std::make_shared<LQPColumnExpression>(shared_from_this(), stored_column_id);
      ++output_column_id;
    }
  }

  return *_output_expressions;
}

bool StoredTableNode::is_column_nullable(const ColumnID column_id) const {
  const auto table = Hyrise::get().storage_manager.get_table(table_name);
  return table->column_is_nullable(column_id);
}

UniqueColumnCombinations StoredTableNode::unique_column_combinations() const {
  auto unique_column_combinations = UniqueColumnCombinations{};

  // We create unique column combinations from selected table key constraints.
  const auto& table = Hyrise::get().storage_manager.get_table(table_name);
  const auto& table_key_constraints = table->soft_key_constraints();

  for (const auto& table_key_constraint : table_key_constraints) {
    // Discard key constraints that involve pruned column id(s).
    if (contains_any_column_id(table_key_constraint.columns(), _pruned_column_ids)) {
      continue;
    }

    // Search for expressions representing the key constraint's ColumnIDs.
    const auto& column_expressions = find_column_expressions(*this, table_key_constraint.columns());
    DebugAssert(column_expressions.size() == table_key_constraint.columns().size(),
                "Unexpected count of column expressions.");

    // Create UniqueColumnCombination
    unique_column_combinations.emplace(column_expressions);
  }

  return unique_column_combinations;
}

OrderDependencies StoredTableNode::order_dependencies() const {
  auto order_dependencies = OrderDependencies{};

  // We create order dependencies from table order constraints.
  const auto& table = Hyrise::get().storage_manager.get_table(table_name);
  const auto& table_order_constraints = table->soft_order_constraints();

  for (const auto& table_order_constraint : table_order_constraints) {
    // Discard order constraints that involve pruned column id(s).
    if (contains_any_column_id(table_order_constraint.columns(), _pruned_column_ids) ||
        contains_any_column_id(table_order_constraint.ordered_columns(), _pruned_column_ids)) {
      continue;
    }

    // Search for expressions representing the order constraint's ColumnIDs.
    const auto& output_expressions = this->output_expressions();
    const auto& column_expressions = find_expressions(table_order_constraint.columns(), output_expressions);
    const auto& ordered_column_expressions =
        find_expressions(table_order_constraint.ordered_columns(), output_expressions);

    // Create OrderDependency.
    order_dependencies.emplace(column_expressions, ordered_column_expressions);
  }

  // Construct transitive ODs. For instance, create [a] |-> [c] from [a] |-> [b] and [b] |-> [c].
  build_transitive_od_closure(order_dependencies);

  return order_dependencies;
}

InclusionDependencies StoredTableNode::inclusion_dependencies() const {
  auto inclusion_dependencies = InclusionDependencies{};

  // We create inclusion dependencies from foreign constraints.
  const auto& table = Hyrise::get().storage_manager.get_table(table_name);
  const auto& foreign_key_constraints = table->referenced_foreign_key_constraints();

  for (const auto& foreign_key_constraint : foreign_key_constraints) {
    const auto& referenced_table = foreign_key_constraint.foreign_key_table();
    if (!referenced_table) {
      // Referenced table was deleted, IND is useless.
      continue;
    }

    // Discard inclusion constraints that involve pruned column id(s).
    if (contains_any_column_id(foreign_key_constraint.columns(), _pruned_column_ids)) {
      continue;
    }

    // Search for expressions representing the inclusion constraint's ColumnIDs
    const auto& column_expressions = find_expressions(foreign_key_constraint.columns(), this->output_expressions());

    // Create InclusionDependency
    inclusion_dependencies.emplace(column_expressions, foreign_key_constraint.foreign_key_columns(), referenced_table);
  }

  return inclusion_dependencies;
}

std::vector<ChunkIndexStatistics> StoredTableNode::chunk_indexes_statistics() const {
  DebugAssert(!left_input() && !right_input(), "StoredTableNode must be a leaf");

  const auto table = Hyrise::get().storage_manager.get_table(table_name);
  auto pruned_indexes_statistics = table->chunk_indexes_statistics();

  if (_pruned_column_ids.empty()) {
    return pruned_indexes_statistics;
  }

  const auto column_id_mapping = column_ids_after_pruning(table->column_count(), _pruned_column_ids);

  // Update index statistics
  // Note: The lambda also modifies statistics.column_ids. This is done because a regular for loop runs into issues
  // when remove(iterator) invalidates the iterator.
  // TODO(anyone): Theoretically, we could keep multi-column indexes where only the last column was pruned
  pruned_indexes_statistics.erase(std::remove_if(pruned_indexes_statistics.begin(), pruned_indexes_statistics.end(),
                                                 [&](auto& statistics) {
                                                   for (auto& original_column_id : statistics.column_ids) {
                                                     const auto& updated_column_id =
                                                         column_id_mapping[original_column_id];
                                                     if (!updated_column_id) {
                                                       // Indexed column was pruned - remove index from statistics
                                                       return true;
                                                     }

                                                     // Update column id
                                                     original_column_id = *updated_column_id;
                                                   }
                                                   return false;
                                                 }),
                                  pruned_indexes_statistics.end());

  return pruned_indexes_statistics;
}

size_t StoredTableNode::_on_shallow_hash() const {
  size_t hash{0};
  boost::hash_combine(hash, table_name);
  for (const auto& pruned_chunk_id : _pruned_chunk_ids) {
    boost::hash_combine(hash, static_cast<size_t>(pruned_chunk_id));
  }
  for (const auto& pruned_column_id : _pruned_column_ids) {
    boost::hash_combine(hash, static_cast<size_t>(pruned_column_id));
  }
  return hash;
}

std::shared_ptr<AbstractLQPNode> StoredTableNode::_on_shallow_copy(LQPNodeMapping& node_mapping) const {
  const auto copy = make(table_name);
  copy->set_pruned_chunk_ids(_pruned_chunk_ids);
  copy->set_pruned_column_ids(_pruned_column_ids);
  return copy;
}

bool StoredTableNode::_on_shallow_equals(const AbstractLQPNode& rhs, const LQPNodeMapping& node_mapping) const {
  const auto& stored_table_node = static_cast<const StoredTableNode&>(rhs);
  return table_name == stored_table_node.table_name && _pruned_chunk_ids == stored_table_node._pruned_chunk_ids &&
         _pruned_column_ids == stored_table_node._pruned_column_ids;
}

}  // namespace hyrise
