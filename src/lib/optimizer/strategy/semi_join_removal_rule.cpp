#include "semi_join_removal_rule.hpp"

#include <algorithm>

#include "cost_estimation/abstract_cost_estimator.hpp"
#include "expression/between_expression.hpp"
#include "expression/binary_predicate_expression.hpp"
#include "expression/cast_expression.hpp"
#include "expression/expression_utils.hpp"
#include "expression/in_expression.hpp"
#include "expression/is_null_expression.hpp"
#include "expression/list_expression.hpp"
#include "expression/lqp_subquery_expression.hpp"
#include "logical_query_plan/join_node.hpp"
#include "logical_query_plan/lqp_utils.hpp"
#include "logical_query_plan/predicate_node.hpp"
#include "statistics/abstract_cardinality_estimator.hpp"

namespace {
using namespace opossum;  // NOLINT

bool is_value_operand(const std::shared_ptr<AbstractExpression> operand) {
  if (operand->type == ExpressionType::Value) {
    return true;
  } else if (operand->type == ExpressionType::Cast) {
    return static_cast<const CastExpression&>(*operand).argument()->type == ExpressionType::Value;
  }
  return false;
}

/**
 * TODO.. Doc
 * @param predicate
 * @return
 * TODO: Maybe also consider the predicate multiplier defined by the cost model
 */
bool is_expensive_predicate(const std::shared_ptr<AbstractExpression>& predicate) {
  if (const auto binary_predicate = std::dynamic_pointer_cast<BinaryPredicateExpression>(predicate)) {
    // LIKE predicates
    if (binary_predicate->predicate_condition == PredicateCondition::Like ||
        binary_predicate->predicate_condition == PredicateCondition::NotLike) {
      return true;
    }
    // Value-based vs. Non-Value-based predicates
    //  The existence of at least one value operand leads to the efficient ColumnVsValue-TableScanImpl in PQPs.
    //  All other binary predicates require the more expensive ColumnVsColumn- or ExpressionEvaluator-TableScanImpls.
    bool is_column_vs_value_predicate =
        is_value_operand(binary_predicate->left_operand()) || is_value_operand(binary_predicate->right_operand());
    return !is_column_vs_value_predicate;
  }

  if (const auto between_predicate = std::dynamic_pointer_cast<BetweenExpression>(predicate)) {
    // The ColumnBetweenTableScanImpl is chosen when lower and upper bound are specified as values. Otherwise, the
    // expensive ExpressionEvaluator-TableScanImpl is required for evaluation.
    const bool is_column_between_values_predicate =
        is_value_operand(between_predicate->lower_bound()) && is_value_operand(between_predicate->upper_bound());
    return !is_column_between_values_predicate;
  }

  return true;
}

Selectivity calculate_selectivity(const Cardinality in, const Cardinality out) {
  if (out < 1.0f) return 0.0f;
  return out / in;
}

}  // namespace

namespace opossum {

std::string SemiJoinRemovalRule::name() const {
  static const auto name = std::string{"SemiJoinRemovalRule"};
  return name;
}

void SemiJoinRemovalRule::_apply_to_plan_without_subqueries(const std::shared_ptr<AbstractLQPNode>& lqp_root) const {
  Assert(lqp_root->type == LQPNodeType::Root, "SemiJoinRemovalRule needs root to hold onto");

  /**
   * APPROACH
   *  1. Find semi join reduction node
   *  2. Find corresponding original join node and check whether there are expensive operators between the original join
   *     and the semi join reduction node and track removal candidates.
   *  3. Remove semi join reductions
   *
   * REMOVAL BLOCKERS TODO Adjust!
   *  In some cases, semi joins are added on both sides of the join (e.g., TPC-H Q17). In the LQPTranslator, these will
   *  be translated into the same operator. If we remove one of these reductions, we block the reuse of the join result.
   *  To counter these cases, we track semi join reductions that should not be removed. `removal_candidates` holds the
   *  nodes that can be removed unless there is a semantically identical node in `removal_blockers`.
   *  This is slightly ugly, as we have to preempt the behavior of the LQPTranslator. This would be better if we had a
   *  method of identifying plan reuse in the optimizer. However, when we tried this, we found that reuse was close to
   *  impossible to implement correctly in the presence of self-joins.
   *
   *  TODO: Is this implemented?
   *    // Semi Reduction is directly followed by the corresponding join.
   *    if (corresponding_join_by_semi_reduction.at(removal_candidate)->input(corresponding_join_input_side) == removal_candidate) {
   */
  auto removal_candidates = std::vector<std::shared_ptr<AbstractLQPNode>>{};
  auto removal_blockers1 = std::unordered_set<std::shared_ptr<AbstractLQPNode>>{};
  auto removal_blockers2 =
      std::unordered_set<std::shared_ptr<AbstractLQPNode>, LQPNodeSharedPtrHash, LQPNodeSharedPtrEqual>{};

  /**
   * Phase 1: Collect semi join reductions
   *            TODO rename rule if we continue to look at semi join reductions only
   */
  visit_lqp(lqp_root, [&](const auto& node) {
    if (node->type != LQPNodeType::Join) return LQPVisitation::VisitInputs;
    const auto& join_node = static_cast<const JoinNode&>(*node);
    // Check if the current node is a semi join reduction and ... TODO
    if (join_node.is_reducer()) {
      Assert(join_node.join_predicates().size() == 1, "Did not expect multi-predicate semi join reduction.");
      removal_candidates.emplace_back(node);
    }
    return LQPVisitation::VisitInputs;
  });

  if (removal_candidates.empty()) return;

  /**
   * Phase 2: Find corresponding JoinNode & Determine removal blockers.
   */
  const auto estimator = cost_estimator->cardinality_estimator->new_instance();
  estimator->guarantee_bottom_up_construction();

  for (const auto& removal_candidate : removal_candidates) {
    const auto& semi_reduction_node = static_cast<const JoinNode&>(*removal_candidate);
    // TODO skip if we do not have a corresponding join
    visit_lqp_upwards(removal_candidate, [&](const auto& upper_node) {
      // Start with the output(s) of the removal candidate
      if (upper_node == removal_candidate) {
        return LQPUpwardVisitation::VisitOutputs;
      }

      // Removal Blocker: AggregateNode
      //  The estimation for aggregate and column/column scans is bad, so whenever one of these occurs between the semi
      //  join reduction and the original join, do not remove the semi join reduction (i.e., abort the search for an
      //  upper node).
      if (upper_node->type == LQPNodeType::Aggregate) {
        removal_blockers1.emplace(removal_candidate);
        removal_blockers2.emplace(removal_candidate);
        return LQPUpwardVisitation::DoNotVisitOutputs;
      }

      // Removal Blocker: PredicateNode, unless it is not expensive.
      if (upper_node->type == LQPNodeType::Predicate) {
        const auto& upper_predicate_node = static_cast<const PredicateNode&>(*upper_node);
        if (is_expensive_predicate(upper_predicate_node.predicate())) {
          removal_blockers1.emplace(removal_candidate);
          removal_blockers2.emplace(removal_candidate);
          return LQPUpwardVisitation::DoNotVisitOutputs;
        }
      }

      // Skip all other nodes, except for joins.
      if (upper_node->type != LQPNodeType::Join) return LQPUpwardVisitation::VisitOutputs;

      // TODO Add Removal Blocker, if it is not the corresponding join.
      const auto& upper_join_node = static_cast<const JoinNode&>(*upper_node);
      if (upper_join_node != corresponding_join) {
        bool block_removal = [&]() {
          // Any semi join reducer might become removed in the course of this rule. Therefore, they should not block the
          // removal of other semi join reducers.
          if (upper_join_node.is_reducer()) return false;

          // Multi predicate joins and anti joins are always expensive for large numbers of tuples.
          if (upper_join_node.join_predicates().size() > 1 || upper_join_node.join_mode == JoinMode::AntiNullAsFalse ||
              upper_join_node.join_mode == JoinMode::AntiNullAsTrue) {
            return true;
          }

          // We do not want to remove a semi join, if it reduces an upper join's smallest input relation because it
          // usually improves the efficiency of our join implementations. For example, by requiring a smaller hashtable
          // in the JoinHash operator. TODO Benchmark: Remove this block
          const auto cardinality_semi_in = estimator->estimate_cardinality(semi_reduction_node.left_input());
          const auto cardinality_upper_join_in_left = estimator->estimate_cardinality(upper_join_node.left_input());
          const auto cardinality_upper_join_in_right = estimator->estimate_cardinality(upper_join_node.right_input());
          if (cardinality_semi_in < std::max(cardinality_upper_join_in_left, cardinality_upper_join_in_right)) {
            return true;
          }

          // Semi reduction vs. upper semi join
          if (upper_join_node.join_mode == JoinMode::Semi) {
            const auto cardinality_semi_out = estimator->estimate_cardinality(semi_reduction_node.outputs()[0]);
            const auto cardinality_upper_join_out = estimator->estimate_cardinality(upper_join_node.outputs()[0]);
            const auto selectivity_semi = calculate_selectivity(cardinality_semi_in, cardinality_semi_out);
            const auto selectivity_upper_semi =
                calculate_selectivity(cardinality_upper_join_in_left, cardinality_upper_join_out);
            // TODO We want to keep...
            return selectivity_semi < selectivity_upper_semi;
          }

          // Semi Join reduces the upper join's bigger input relation. For efficiency, however, the semi join's
          // right input relation should be smaller than the smallest input relation of the upper join.
          auto min_cardinality_upper_join_in =
              std::min(cardinality_upper_join_in_left, cardinality_upper_join_in_right);
          auto cardinality_semi_reducer = estimator->estimate_cardinality(semi_reduction_node.right_input());
          if (cardinality_semi_reducer < min_cardinality_upper_join_in) return true;

          // Do not block the semi reduction's removal because it does not seem to benefit the latency of the upper join.
          return false;
        }();

        if (!block_removal) return LQPUpwardVisitation::VisitOutputs;

        removal_blockers1.emplace(removal_candidate);
        removal_blockers2.emplace(removal_candidate);
      }

      return LQPUpwardVisitation::DoNotVisitOutputs;
    });
  }

  /**
   * Phase 3: Remove semi join reduction nodes
   */
  for (const auto& removal_candidate : removal_candidates) {
    if (removal_blockers1.contains(removal_candidate) || removal_blockers2.contains(removal_candidate)) continue;
    lqp_remove_node(removal_candidate, AllowRightInput::Yes);
  }
}

}  // namespace opossum
