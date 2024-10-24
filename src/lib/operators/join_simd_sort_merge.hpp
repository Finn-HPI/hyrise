#pragma once

#include "abstract_join_operator.hpp"
#include "operator_join_predicate.hpp"
#include "operators/join_simd_sort_merge/simd_utils.hpp"
#include "types.hpp"

struct SimdElement {
  uint32_t key;
  uint32_t index;

  friend std::ostream& operator<<(std::ostream& stream, const SimdElement& element) {
    stream << "SimdElement(" << element.key << "," << element.index << ")";
    return stream;
  }
};

namespace hyrise {
using SimdElementList = simd_sort::simd_vector<SimdElement>;

class JoinSimdSortMerge : public AbstractJoinOperator {
 public:
  static bool supports(const JoinConfiguration config);
  JoinSimdSortMerge(const std::shared_ptr<const AbstractOperator>& left,
                    const std::shared_ptr<const AbstractOperator>& right, const JoinMode mode,
                    const OperatorJoinPredicate& primary_predicate,
                    const std::vector<OperatorJoinPredicate>& secondary_predicates = {});

  const std::string& name() const override;
  static constexpr auto JOB_SPAWN_THRESHOLD = 500;

 protected:
  std::shared_ptr<const Table> _on_execute() override;
  void _on_cleanup() override;
  std::shared_ptr<AbstractOperator> _on_deep_copy(
      const std::shared_ptr<AbstractOperator>& copied_left_input,
      const std::shared_ptr<AbstractOperator>& copied_right_input,
      std::unordered_map<const AbstractOperator*, std::shared_ptr<AbstractOperator>>& /*copied_ops*/) const override;
  void _on_set_parameters(const std::unordered_map<ParameterID, AllTypeVariant>& parameters) override;

  std::unique_ptr<AbstractReadOnlyOperatorImpl> _impl;
  template <typename T>
  class JoinSimdSortMergeImpl;
  template <typename T>
  friend class JoinSimdSortMergeImpl;
};
}  // namespace hyrise
