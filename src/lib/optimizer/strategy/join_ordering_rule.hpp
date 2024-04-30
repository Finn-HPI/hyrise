#pragma once

#include <memory>
#include <string>

#include "abstract_rule.hpp"

namespace hyrise {

class AbstractCostEstimator;

/**
 * A rule that brings join operations into a (supposedly) efficient order. Currently, only the order of inner joins is
 * modified using either the DpCcp algorithm or GreedyOperatorOrdering.
 */
class JoinOrderingRule : public AbstractRule {
 public:
  std::string name() const override;

 protected:
  void _apply_to_plan_without_subqueries(const std::shared_ptr<AbstractLQPNode>& lqp_root) const override;
};

}  // namespace hyrise
