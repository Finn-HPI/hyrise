#include "predicate_placement_rule.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "cost_estimation/abstract_cost_estimator.hpp"
#include "expression/abstract_expression.hpp"
#include "expression/expression_utils.hpp"
#include "expression/logical_expression.hpp"
#include "expression/lqp_subquery_expression.hpp"
#include "logical_query_plan/abstract_lqp_node.hpp"
#include "logical_query_plan/join_node.hpp"
#include "logical_query_plan/logical_plan_root_node.hpp"
#include "logical_query_plan/lqp_utils.hpp"
#include "logical_query_plan/predicate_node.hpp"
#include "logical_query_plan/union_node.hpp"
#include "operators/operator_scan_predicate.hpp"
#include "statistics/cardinality_estimator.hpp"
#include "types.hpp"
#include "utils/assert.hpp"

namespace hyrise {

std::string PredicatePlacementRule::name() const {
  static const auto name = std::string{"PredicatePlacementRule"};
  return name;
}

void PredicatePlacementRule::_apply_to_plan_without_subqueries(const std::shared_ptr<AbstractLQPNode>& lqp_root) const {
  // The traversal functions require the existence of a root of the LQP, so make sure we have that.
  const auto root_node = lqp_root->type == LQPNodeType::Root ? lqp_root : LogicalPlanRootNode::make(lqp_root);

  const auto estimator = cost_estimator->cardinality_estimator->new_instance();
  estimator->guarantee_bottom_up_construction(lqp_root);
  // Turn off statistics pruning because we untie nodes while estimating cardinalities. Thus, not all required predicate
  // expessions are part of the LQP when the required statistics are populated during the first estimation call.
  estimator->do_not_prune_unused_statistics();

  auto push_down_nodes = std::vector<std::shared_ptr<AbstractLQPNode>>{};
  _push_down_traversal(root_node, LQPInputSide::Left, push_down_nodes, *estimator);

  _pull_up_traversal(root_node, LQPInputSide::Left);
}

void PredicatePlacementRule::_push_down_traversal(const std::shared_ptr<AbstractLQPNode>& current_node,
                                                  const LQPInputSide input_side,
                                                  std::vector<std::shared_ptr<AbstractLQPNode>>& push_down_nodes,
                                                  CardinalityEstimator& estimator) {
  const auto input_node = current_node->input(input_side);
  // Allow calling without checks.
  if (!input_node) {
    Assert(push_down_nodes.empty(), "Expected pushdown nodes to be already inserted.");
    return;
  }

  // A helper method for cases where the input_node does not allow us to proceed.
  const auto handle_barrier = [&]() {
    _insert_nodes(current_node, input_side, push_down_nodes);

    // At this point, all pushdown predicates should have been inserted above the barrier node. In the following, we
    // apply the pushdown traversal logic to the remaining parts of the LQP – including the current barrier node. The
    // latter might also be a predicate eligible for pushdown. If it is, we try to push it down the LQP, and another
    // node might become the "new barrier" (having multiple output nodes).
    const auto& barrier_node = input_node;
    const auto barrier_node_is_pushdown_predicate = [&barrier_node]() {
      if (barrier_node->type == LQPNodeType::Predicate) {
        const auto predicate_node = std::static_pointer_cast<PredicateNode>(barrier_node);
        return !_is_expensive_predicate(predicate_node->predicate());
      }

      if (barrier_node->type == LQPNodeType::Join) {
        const auto& join_node = static_cast<const JoinNode&>(*barrier_node);
        return is_semi_or_anti_join(join_node.join_mode) && join_node.join_predicates().size() == 1;
      }

      return false;
    }();

    auto next_push_down_traversal_root = barrier_node;
    if (barrier_node_is_pushdown_predicate) {
      // barrier_node is a predicate, and we would like to cover it in the next recursion of _push_down_traversal.
      // However, if we simply call _push_down_traversal with barrier_node, the predicate would not become pushed down
      // since _push_down_traversal looks at input nodes only. To overcome this issue, we insert a temporary root node,
      // set barrier_node as an input, and call _push_down_traversal with the temporary root node.
      next_push_down_traversal_root = LogicalPlanRootNode::make();
      lqp_insert_node_above(barrier_node, next_push_down_traversal_root);
    }

    if (next_push_down_traversal_root->left_input()) {
      auto left_push_down_nodes = std::vector<std::shared_ptr<AbstractLQPNode>>{};
      _push_down_traversal(next_push_down_traversal_root, LQPInputSide::Left, left_push_down_nodes, estimator);

      // Check for the left input node first because there cannot be a right input node otherwise.
      if (next_push_down_traversal_root->right_input()) {
        auto right_push_down_nodes = std::vector<std::shared_ptr<AbstractLQPNode>>{};
        _push_down_traversal(next_push_down_traversal_root, LQPInputSide::Right, right_push_down_nodes, estimator);
      }
    }

    // The recursion calls to _push_down_traversal have returned. Therefore, we must remove the temporary root node, we
    // might have inserted previously (see comment above).
    if (next_push_down_traversal_root->type == LQPNodeType::Root && next_push_down_traversal_root->output_count() > 0) {
      lqp_remove_node(next_push_down_traversal_root);
    }
  };

  if (input_node->output_count() > 1) {
    // We cannot push predicates past input_node as doing so would also filter the predicates from the "other" side.
    handle_barrier();
    return;
  }

  // Removes input_node from the current LQP and continues to run _push_down_traversal.
  const auto untie_input_node_and_recurse = [&]() {
    push_down_nodes.emplace_back(input_node);
    lqp_remove_node(input_node, AllowRightInput::Yes);
    _push_down_traversal(current_node, input_side, push_down_nodes, estimator);
  };

  switch (input_node->type) {
    case LQPNodeType::Predicate: {
      const auto predicate_node = std::static_pointer_cast<PredicateNode>(input_node);

      if (!_is_expensive_predicate(predicate_node->predicate())) {
        untie_input_node_and_recurse();
      } else {
        _push_down_traversal(input_node, input_side, push_down_nodes, estimator);
      }
    } break;

    case LQPNodeType::Join: {
      const auto join_node = std::static_pointer_cast<JoinNode>(input_node);

      // We pick up single-predicate semi- and anti-joins on the way and treat them as if they were predicates.
      if (is_semi_or_anti_join(join_node->join_mode) && join_node->join_predicates().size() == 1) {
        // First, we need to recurse into the right side to make sure that it is optimized as well.
        auto right_push_down_nodes = std::vector<std::shared_ptr<AbstractLQPNode>>{};
        _push_down_traversal(input_node, LQPInputSide::Right, right_push_down_nodes, estimator);

        untie_input_node_and_recurse();
        break;
      }

      // Not a semi-/ anti-join. We need to check if we can push the nodes in push_down_nodes past the join or if they
      // need to be inserted here before proceeding.

      // Left empty for non-push-past joins
      auto left_push_down_nodes = std::vector<std::shared_ptr<AbstractLQPNode>>{};
      auto right_push_down_nodes = std::vector<std::shared_ptr<AbstractLQPNode>>{};

      // It is safe to move predicates down past the named joins as doing so does not affect the presence of NULLs.
      if (join_node->join_mode == JoinMode::Inner || join_node->join_mode == JoinMode::Cross) {
        for (const auto& push_down_node : push_down_nodes) {
          const auto move_to_left = _is_evaluable_on_lqp(push_down_node, join_node->left_input());
          const auto move_to_right = _is_evaluable_on_lqp(push_down_node, join_node->right_input());

          if (!move_to_left && !move_to_right) {
            const auto push_down_predicate_node = std::dynamic_pointer_cast<PredicateNode>(push_down_node);
            if (join_node->join_mode == JoinMode::Inner && push_down_predicate_node) {
              // Pre-Join Predicates:
              // The current predicate could not be pushed down to either side. If we cannot push it down, we might be
              // able to create additional predicates that perform some pre-selection before the tuples reach the join.
              // An example can be found in TPC-H query 7, with the predicate
              //   (n1.name = 'DE' AND n2.name = 'FR') OR (n1.name = 'FR' AND n2.n_name = 'DE')
              // We cannot push it to either n1 or n2 as the selected values depend on the result of the other table.
              // However, we can create a predicate (n1.name = 'DE' OR n1.name = 'FR') and reduce the number of tuples
              // that reach the joins from all countries to just two. This behavior is also described in the TPC-H
              // Analyzed paper as "CP4.2b: Join-Dependent Expression Filter Pushdown".
              //
              // Here are the rules that determine whether we can create a pre-join predicate for the tables l or r with
              // predicates that operate on l (l1, l2), r (r1, r2), or are independent of either table (u1, u2). To
              // produce a predicate for a table, it is required that each expression in the disjunction has a predicate
              // for that table:
              //
              // (l1 AND r1) OR (l2)        -> create predicate (l1 OR l2) on left side, everything on right side might
              //                               qualify, so do not create a predicate there
              // (l1 AND r2) OR (l2 AND r1) -> create (l1 OR l2) on left, (r1 OR r2) on right (example from above)
              // (l1 AND u1) OR (r1 AND u2) -> do nothing
              // You will also find these examples in the tests.
              //
              // For now, this rule deals only with inner joins. It might also work for other join types, but the
              // implications of creating a pre-join predicate on the NULL-producing side need to be carefully thought
              // through once the need arises.
              //
              // While the above only decides whether it is possible to create a pre-join predicate, we estimate the
              // selectivity of each individual candidate and compare it to MAX_SELECTIVITY_FOR_PRE_JOIN_PREDICATE.
              // Only if a predicate candidate is selective enough, it is added below the join.
              //
              // NAMING:
              // Input
              // (l1 AND r2) OR (l2 AND r1)
              // ^^^^^^^^^^^    ^^^^^^^^^^^ outer_disjunction holds two (or more) elements from flattening the OR.
              //                            One of these elements is called expression_in_disjunction.
              //  ^^     ^^                 inner_conjunction holds two (or more) elements from flattening the AND.
              //                            One of these elements is called expression_in_conjunction.

              auto left_disjunction = std::vector<std::shared_ptr<AbstractExpression>>{};
              auto right_disjunction = std::vector<std::shared_ptr<AbstractExpression>>{};

              // Tracks whether we had to abort the search for one of the sides as an inner_conjunction was found that
              // did not cover the side.
              auto aborted_left_side = false;
              auto aborted_right_side = false;

              const auto outer_disjunction =
                  flatten_logical_expressions(push_down_predicate_node->predicate(), LogicalOperator::Or);
              for (const auto& expression_in_disjunction : outer_disjunction) {
                // For the current expression_in_disjunction, these hold the PredicateExpressions that need to be true
                // on the left/right side.
                auto left_conjunction = std::vector<std::shared_ptr<AbstractExpression>>{};
                auto right_conjunction = std::vector<std::shared_ptr<AbstractExpression>>{};

                // Fill left/right_conjunction
                const auto inner_conjunction =
                    flatten_logical_expressions(expression_in_disjunction, LogicalOperator::And);
                for (const auto& expression_in_conjunction : inner_conjunction) {
                  const auto evaluable_on_left_side =
                      expression_evaluable_on_lqp(expression_in_conjunction, *join_node->left_input());
                  const auto evaluable_on_right_side =
                      expression_evaluable_on_lqp(expression_in_conjunction, *join_node->right_input());

                  // We can only work with expressions that are specific to one side.
                  if (evaluable_on_left_side && !evaluable_on_right_side && !aborted_left_side) {
                    left_conjunction.emplace_back(expression_in_conjunction);
                  }
                  if (evaluable_on_right_side && !evaluable_on_left_side && !aborted_right_side) {
                    right_conjunction.emplace_back(expression_in_conjunction);
                  }
                }

                if (!left_conjunction.empty()) {
                  // If we have found multiple predicates for the left side, connect them using AND and add them to
                  // the disjunction that will be pushed to the left side:
                  //  Example: `(l1 AND l2 AND r1) OR (l3 AND r2)` is first split into the two conjunctions. When
                  //  looking at the first conjunction, l1 and l2 will end up in left_conjunction. Before it gets added
                  //  to the left_disjunction, it needs to be connected using AND: (l1 AND l2).
                  //  The result for the left_disjunction will be ((l1 AND l2) OR l3)
                  left_disjunction.emplace_back(inflate_logical_expressions(left_conjunction, LogicalOperator::And));
                } else {
                  // If, within the current expression_in_disjunction, we have not found a matching predicate for the
                  // left side, all tuples for the left side qualify and it makes no sense to create a filter.
                  aborted_left_side = true;
                  left_disjunction.clear();
                }
                if (!right_conjunction.empty()) {
                  right_disjunction.emplace_back(inflate_logical_expressions(right_conjunction, LogicalOperator::And));
                } else {
                  aborted_right_side = true;
                  right_disjunction.clear();
                }
              }

              const auto add_disjunction_if_beneficial =
                  [&](const auto& disjunction, const auto& disjunction_input_node, auto& predicate_nodes) {
                    if (disjunction.empty()) {
                      return;
                    }

                    const auto expression = inflate_logical_expressions(disjunction, LogicalOperator::Or);
                    const auto predicate_node = PredicateNode::make(expression, disjunction_input_node);

                    // Determine the selectivity of the predicate if executed on disjunction_input_node.
                    const auto cardinality_in = estimator.estimate_cardinality(disjunction_input_node);
                    const auto cardinality_out = estimator.estimate_cardinality(predicate_node);
                    if (cardinality_out / cardinality_in > MAX_SELECTIVITY_FOR_PRE_JOIN_PREDICATE) {
                      return;
                    }

                    // predicate_node was found to be beneficial. Add it to predicate_nodes so that _insert_nodes will
                    // insert it as low as possible in the left/right input of the join. As predicate_nodes might have
                    // more than one node, remove the input so that _insert_nodes can construct a proper LQP.
                    predicate_node->set_left_input(nullptr);
                    predicate_nodes.emplace_back(predicate_node);
                  };

              add_disjunction_if_beneficial(left_disjunction, join_node->left_input(), left_push_down_nodes);
              add_disjunction_if_beneficial(right_disjunction, join_node->right_input(), right_push_down_nodes);

              // End of the pre-join filter code
            }
            lqp_insert_node(current_node, input_side, push_down_node, AllowRightInput::Yes);
          } else if (move_to_left && move_to_right) {
            // This predicate applies to both the left and the right side. We have not seen this case in the wild yet,
            // it might make more sense to duplicate the predicate and push it down on both sides.
            lqp_insert_node(current_node, input_side, push_down_node, AllowRightInput::Yes);
          } else {
            if (move_to_left) {
              left_push_down_nodes.emplace_back(push_down_node);
            }
            if (move_to_right) {
              right_push_down_nodes.emplace_back(push_down_node);
            }
          }
        }
      } else {
        // We do not push past non-inner/cross joins, place all predicates here.
        _insert_nodes(current_node, input_side, push_down_nodes);
      }

      _push_down_traversal(input_node, LQPInputSide::Left, left_push_down_nodes, estimator);
      _push_down_traversal(input_node, LQPInputSide::Right, right_push_down_nodes, estimator);
    } break;

    case LQPNodeType::Alias:
    case LQPNodeType::Sort:
    case LQPNodeType::Projection: {
      // We can push predicates past these nodes without further consideration.
      _push_down_traversal(input_node, LQPInputSide::Left, push_down_nodes, estimator);
    } break;

    case LQPNodeType::Aggregate: {
      // We can push predicates below the aggregate if they do not depend on an aggregate expression.
      auto aggregate_push_down_nodes = std::vector<std::shared_ptr<AbstractLQPNode>>{};
      for (const auto& push_down_node : push_down_nodes) {
        if (_is_evaluable_on_lqp(push_down_node, input_node->left_input())) {
          aggregate_push_down_nodes.emplace_back(push_down_node);
        } else {
          lqp_insert_node_above(input_node, push_down_node, AllowRightInput::Yes);
        }
      }
      _push_down_traversal(input_node, LQPInputSide::Left, aggregate_push_down_nodes, estimator);
    } break;

    case LQPNodeType::Union: {
      const auto union_node = std::static_pointer_cast<UnionNode>(input_node);
      /**
       * If we have a diamond of predicates where all UnionNode inputs result from the same origin node, the
       * pushdown traversal should continue below the diamond's origin node, if possible.
       *
       *                                        |
       *                                  ____Union_____
       *                                 /              \
       *                      Predicate(a LIKE %man)    |
       *                                |               |
       *                                |     Predicate(a LIKE %woman)
       *                                |               |
       *                                |               |
       *                                \_____Node______/  <---- Diamond's origin node
       *                                        |  <------------ Continue pushdown traversal here, if possible
       *                                        |
       */
      const auto diamond_origin_node = find_diamond_origin_node(union_node);
      if (!diamond_origin_node) {
        handle_barrier();
        return;
      }

      /**
       * In the following, we determine whether the diamond's origin node is used as an input by nodes which are
       * not part of the diamond because we should only filter the predicates of the diamond nodes, not other nodes'
       * predicates. For example:
       *                                           |                                |
       *                                     ____Union_____                   Join(a = x)
       *                                    /              \                     /    \
       *                             ______/               |                     |    |
       *                            /                      |                     |    |
       *                     ____Union_____                |                     |    |
       *                    /              \               |                     |    |
       *                   /               |     Predicate(a LIKE %woman)        |    |
       *        Predicate(a LIKE %man)     |               |                     |    |
       *                  |                |               |                     |    |
       *                  |       Predicate(a LIKE %child) |                     |    |
       *                  |                |               |                     |    |
       *                  |                \______   ______/                     |    |
       *                   \                      \ /                            |    |
       *                    \___________________  | |  __________________________/    |
       *                                        \ | | /                               |
       *                     ---------------->    Node                              Table
       *                    /                      |
       *                   /                      ...
       *       ___________/_________
       *   The diamond's origin node has four outputs, but only three outputs are part of the diamond structure.
       *   Therefore, we do not want to continue the pushdown traversal below the diamond. Because otherwise, we would
       *   incorrectly filter the Join's left input.
       *
       * To identify cases such as above, we check the output count of the diamond's origin node and compare it
       * with the number of UnionNodes in the diamond structure.
       */
      size_t union_node_count = 0;
      visit_lqp(union_node, [&](const auto& diamond_node) {
        if (diamond_node == diamond_origin_node) {
          return LQPVisitation::DoNotVisitInputs;
        }
        if (diamond_node->type == LQPNodeType::Union) {
          union_node_count++;
        }
        return LQPVisitation::VisitInputs;
      });
      if (diamond_origin_node->output_count() != union_node_count + 1) {
        handle_barrier();
        return;
      }

      // Apply predicate pushdown to the diamond's nodes.
      auto left_push_down_nodes = std::vector<std::shared_ptr<AbstractLQPNode>>{};
      auto right_push_down_nodes = std::vector<std::shared_ptr<AbstractLQPNode>>{};
      _push_down_traversal(union_node, LQPInputSide::Left, left_push_down_nodes, estimator);
      _push_down_traversal(union_node, LQPInputSide::Right, right_push_down_nodes, estimator);

      // Continue predicate pushdown below the diamond.
      auto updated_push_down_nodes = std::vector<std::shared_ptr<AbstractLQPNode>>{};
      for (const auto& push_down_node : push_down_nodes) {
        if (_is_evaluable_on_lqp(push_down_node, diamond_origin_node)) {
          // Save for next _push_down_traversal recursion.
          updated_push_down_nodes.emplace_back(push_down_node);
        } else {
          // The diamond is a barrier for push_down_node.
          lqp_insert_node_above(union_node, push_down_node, AllowRightInput::Yes);
        }
      }
      auto temporary_root_node = LogicalPlanRootNode::make();
      lqp_insert_node_above(diamond_origin_node, temporary_root_node);
      _push_down_traversal(temporary_root_node, LQPInputSide::Left, updated_push_down_nodes, estimator);
      lqp_remove_node(temporary_root_node);
    } break;

    default: {
      // All not explicitly handled node types are barriers, and we do not push predicates past them.
      handle_barrier();
    }
  }
}

std::vector<std::shared_ptr<AbstractLQPNode>> PredicatePlacementRule::_pull_up_traversal(
    const std::shared_ptr<AbstractLQPNode>& current_node, const LQPInputSide input_side) {
  if (!current_node) {
    return {};
  }

  const auto input_node = current_node->input(input_side);
  if (!input_node) {
    return {};
  }

  auto candidate_nodes = _pull_up_traversal(current_node->input(input_side), LQPInputSide::Left);
  auto candidate_nodes_tmp = _pull_up_traversal(current_node->input(input_side), LQPInputSide::Right);
  candidate_nodes.insert(candidate_nodes.end(), candidate_nodes_tmp.begin(), candidate_nodes_tmp.end());

  // Expensive PredicateNodes become candidates for a PullUp, but only IFF they have exactly one output connection.
  // If they have more, we cannot move them.
  if (const auto predicate_node = std::dynamic_pointer_cast<PredicateNode>(input_node);
      predicate_node && _is_expensive_predicate(predicate_node->predicate()) && predicate_node->output_count() == 1) {
    candidate_nodes.emplace_back(predicate_node);
    lqp_remove_node(predicate_node);
  }

  if (current_node->output_count() > 1) {
    // No pull up past nodes with more than one output, because if we did, the other outputs would lose the
    // predicate we pulled up.
    _insert_nodes(current_node, input_side, candidate_nodes);
    return {};
  }

  switch (current_node->type) {
    case LQPNodeType::Join: {
      const auto join_node = std::static_pointer_cast<JoinNode>(current_node);

      // It is safe to move predicates down past Inner, Cross, Semi, AntiNullAsTrue and AntiNullAsFalse Joins
      if (join_node->join_mode == JoinMode::Inner || join_node->join_mode == JoinMode::Cross ||
          join_node->join_mode == JoinMode::Semi || join_node->join_mode == JoinMode::AntiNullAsTrue ||
          join_node->join_mode == JoinMode::AntiNullAsFalse) {
        return candidate_nodes;
      }

      _insert_nodes(current_node, input_side, candidate_nodes);
      return {};
    } break;

    case LQPNodeType::Alias:
    case LQPNodeType::Predicate:
      return candidate_nodes;

    case LQPNodeType::Projection: {
      auto pull_up_nodes = std::vector<std::shared_ptr<AbstractLQPNode>>{};
      auto blocked_nodes = std::vector<std::shared_ptr<AbstractLQPNode>>{};

      for (const auto& candidate_node : candidate_nodes) {
        if (_is_evaluable_on_lqp(candidate_node, current_node)) {
          pull_up_nodes.emplace_back(candidate_node);
        } else {
          blocked_nodes.emplace_back(candidate_node);
        }
      }

      _insert_nodes(current_node, input_side, blocked_nodes);
      return pull_up_nodes;
    } break;

    default:
      // No pull up past all other node types.
      _insert_nodes(current_node, input_side, candidate_nodes);
      return {};
  }

  Fail("GCC thinks this is reachable.");
}

void PredicatePlacementRule::_insert_nodes(const std::shared_ptr<AbstractLQPNode>& node, const LQPInputSide input_side,
                                           const std::vector<std::shared_ptr<AbstractLQPNode>>& predicate_nodes) {
  // First node gets inserted on the @param input_side, all others on the left side of their output.
  auto current_node = node;
  auto current_input_side = input_side;

  const auto previous_input_node = node->input(input_side);

  for (const auto& predicate_node : predicate_nodes) {
    current_node->set_input(current_input_side, predicate_node);
    current_node = predicate_node;
    current_input_side = LQPInputSide::Left;
  }

  current_node->set_input(current_input_side, previous_input_node);
}

bool PredicatePlacementRule::_is_expensive_predicate(const std::shared_ptr<AbstractExpression>& predicate) {
  /**
   * We (heuristically) consider a predicate to be expensive if it contains a correlated subquery. Otherwise, we
   * consider it to be cheap
   */
  auto predicate_contains_correlated_subquery = false;
  visit_expression(predicate, [&](const auto& sub_expression) {
    if (const auto subquery_expression = std::dynamic_pointer_cast<LQPSubqueryExpression>(sub_expression);
        subquery_expression && subquery_expression->is_correlated()) {
      predicate_contains_correlated_subquery = true;
      return ExpressionVisitation::DoNotVisitArguments;
    }
    return ExpressionVisitation::VisitArguments;
  });
  return predicate_contains_correlated_subquery;
}

bool PredicatePlacementRule::_is_evaluable_on_lqp(const std::shared_ptr<AbstractLQPNode>& node,
                                                  const std::shared_ptr<AbstractLQPNode>& lqp) {
  switch (node->type) {
    case LQPNodeType::Predicate: {
      const auto& predicate_node = static_cast<PredicateNode&>(*node);
      if (!expression_evaluable_on_lqp(predicate_node.predicate(), *lqp)) {
        return false;
      }

      auto has_uncomputed_aggregate = false;
      const auto predicate = predicate_node.predicate();
      visit_expression(predicate, [&](const auto& expression) {
        if (expression->type == ExpressionType::WindowFunction && !lqp->find_column_id(*expression)) {
          has_uncomputed_aggregate = true;
          return ExpressionVisitation::DoNotVisitArguments;
        }
        return ExpressionVisitation::VisitArguments;
      });
      return !has_uncomputed_aggregate;
    }
    case LQPNodeType::Join: {
      const auto& join_node = static_cast<JoinNode&>(*node);
      for (const auto& join_predicate : join_node.join_predicates()) {
        for (const auto& argument : join_predicate->arguments) {
          if (!lqp->find_column_id(*argument) && !join_node.right_input()->find_column_id(*argument)) {
            return false;
          }
        }
      }
      return true;
    }
    default:
      Fail("Unexpected node type");
  }
}

}  // namespace hyrise
