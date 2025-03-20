#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "abstract_join_operator.hpp"
#include "operator_join_predicate.hpp"
#include "operators/join_simd_sort_merge/simd_utils.hpp"
#include "operators/join_simd_sort_merge/util.hpp"
#include "types.hpp"

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

  enum class OperatorSteps : uint8_t {
    LeftSideMaterialize,
    RightSideMaterialize,
    LeftSideTransform,
    RightSideTransform,
    LeftSidePartition,
    RightSidePartition,
    LeftSideSortBuckets,
    RightSideSortBuckets,
    // LeftSideMultiwayMerging,
    // RightSideMultiwayMerging,
    GatherRowIds,
    FindJoinPartner,
    OutputWriting
  };

  struct PerformanceData : public OperatorPerformanceData<OperatorSteps> {
    void output_to_stream(std::ostream& stream, DescriptionMode description_mode) const override;

    size_t radix_bits{0};
  };

  static constexpr auto CLUSTER_COUNT = 256;

  static constexpr auto JOB_SPAWN_THRESHOLD = 500;

 protected:
// Datatype used for simd sorting (has to be 64 bits).
#if defined(__AVX512F__)
  using SortingType = double;
#elif defined(__AVX2__)
  using SortingType = double;
#elif defined(__powerpc__) || defined(__ppc__) || defined(_ARCH_PPC)
  using SortingType = double;
#elif defined(__arm__) || defined(__aarch64__)
  using SortingType = double;
#else
  using SortingType = double;
#endif

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
