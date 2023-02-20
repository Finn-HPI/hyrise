#pragma once

#include "dependency_discovery/candidate_strategy/abstract_dependency_candidate_rule.hpp"
#include "dependency_discovery/dependency_candidates.hpp"
#include "expression/abstract_expression.hpp"
#include "logical_query_plan/abstract_lqp_node.hpp"
#include "storage/table.hpp"
#include "types.hpp"
#include "utils/abstract_plugin.hpp"

namespace hyrise {

/**
 *  This plugin implements unary Unique Column Combination (UCC) discovery based on previously executed LQPs. Not all
 *  columns encountered in these LQPs are automatically considered for the UCC validation process. Instead, a column is
 *  only validated/invalidated as UCCs if being a UCC could have helped to optimize their LQP.
 */
class DependencyDiscoveryPlugin : public AbstractPlugin {
 public:
  DependencyDiscoveryPlugin();

  std::string description() const final;

  void start() final;

  void stop() final;

  std::vector<std::pair<PluginFunctionName, PluginFunctionPointer>> provided_user_executable_functions() final;

  std::optional<PreBenchmarkHook> pre_benchmark_hook() final;

 protected:
  friend class DependencyDiscoveryPluginTest;

  /**
   * Takes a snapshot of the current LQP Cache. Iterates through the LQPs and tries to extract sensible columns as can-
   * didates for UCC validation from each of them. A column is added as candidates if being a UCC has the potential to
   * help optimize their respective LQP.
   * 
   * Returns an unordered set of these candidates to be used in the UCC validation function.
   */
  DependencyCandidates _identify_dependency_candidates();

  /**
   * Iterates over the provided set of columns identified as candidates for a uniqueness validation. Validates those
   * that are not already known to be unique.
   */
  static void _validate_dependency_candidates(const DependencyCandidates& dependency_candidates);

 private:
  /**
   * Checks whether individual DictionarySegments contain duplicates. This is an efficient operation as the check is
   * simply comparing the length of the dictionary to that of the attribute vector. This function can therefore be used
   * for an early-out before the more expensive cross-segment uniqueness check.
   */
  template <typename ColumnDataType>
  static bool _dictionary_segments_contain_duplicates(std::shared_ptr<Table> table, ColumnID column_id);

  /**
   * Checks whether the given table contains only unique values by inserting all values into an unordered set. If for
   * any table segment the size of the set increases by less than the number of values in that segment, we know that
   * there must be a duplicate and return false. Otherwise, returns true.
   */
  template <typename ColumnDataType>
  static bool _uniqueness_holds_across_segments(std::shared_ptr<Table> table, ColumnID column_id);

  /**
   * Extracts columns as UCC validation candidates from a join node. Some criteria have to be fulfilled for this to be
   * done:
   *   - The Node may only have one predicate.
   *   - This predicate must have the equals condition. This may be extended in the future to support other conditions.
   *   - The join must be either a semi or an inner join.
   * In addition to the column corresponding to the removable side of the join, the removable input side LQP is iterated
   * and a column used in a PredicateNode may be added as a UCC candidate if the predicate filters the same table that
   * contains the join column.
   */
  static void _dependency_candidates_from_join_node(std::shared_ptr<AbstractLQPNode> node,
                                             DependencyCandidates& dependency_candidates);

  /**
   * Iterates through the LQP underneath the given root node. If a PredicateNode is encountered, checks whether it has a
   * binary predicate checking for `column` = `value`. If the predicate column is from the same table as the passed
   * column_candidate, adds both as UCC candidates.
   */
  static void _dependency_candidates_from_removable_join_input(std::shared_ptr<AbstractLQPNode> root_node,
                                                        std::shared_ptr<LQPColumnExpression> column_candidate,
                                                        DependencyCandidates& dependency_candidates);

  void _add_candidate_rule(std::unique_ptr<AbstractDependencyCandidateRule> rule);

  std::unordered_map<LQPNodeType, std::vector<std::unique_ptr<AbstractDependencyCandidateRule>>> _candidate_rules{};
};

}  // namespace hyrise
