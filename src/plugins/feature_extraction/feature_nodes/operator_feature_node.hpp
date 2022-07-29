#pragma once

#include "abstract_feature_node.hpp"
#include "feature_extraction/feature_nodes/result_table_feature_node.hpp"
#include "operators/abstract_operator.hpp"

namespace opossum {

class OperatorFeatureNode : public AbstractFeatureNode {
 public:
  OperatorFeatureNode(const std::shared_ptr<const AbstractOperator>& op,
                      const std::shared_ptr<AbstractFeatureNode>& left_input,
                      const std::shared_ptr<AbstractFeatureNode>& right_input = nullptr);

  std::shared_ptr<OperatorFeatureNode> from_pqp(const std::shared_ptr<const AbstractOperator>& op,
                                                const std::shared_ptr<Query>& query);

  const std::vector<std::string>& feature_headers() const final;

  static const std::vector<std::string>& headers();

  std::chrono::nanoseconds run_time() const;

  bool is_root_node() const;

  void set_as_root_node(const std::shared_ptr<Query>& query);

  std::shared_ptr<Query> query() const;

  std::shared_ptr<const AbstractOperator> get_operator() const;

  std::shared_ptr<ResultTableFeatureNode> output_table() const;

  const std::vector<std::shared_ptr<AbstractFeatureNode>>& subqueries() const final;

 protected:
  std::shared_ptr<FeatureVector> _on_to_feature_vector() const final;
  size_t _on_shallow_hash() const final;

  std::shared_ptr<const AbstractOperator> _op;
  OperatorType _op_type;
  std::chrono::nanoseconds _run_time;
  std::shared_ptr<Query> _query;

  std::vector<std::shared_ptr<AbstractFeatureNode>> _predicates;
  std::shared_ptr<ResultTableFeatureNode> _output_table;

  bool _is_root_node = false;

  std::vector<std::shared_ptr<AbstractFeatureNode>> _subqueries;
};

}  // namespace opossum