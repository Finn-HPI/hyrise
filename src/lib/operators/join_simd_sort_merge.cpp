#include "join_simd_sort_merge.hpp"

#include "types.hpp"

namespace hyrise {
bool JoinSimdSortMerge::supports(const JoinConfiguration config) {
  return config.predicate_condition == PredicateCondition::Equals && config.left_data_type == config.right_data_type &&
         config.join_mode == JoinMode::Inner;
}

JoinSimdSortMerge::JoinSimdSortMerge(const std::shared_ptr<const AbstractOperator>& left,
                                     const std::shared_ptr<const AbstractOperator>& right, const JoinMode mode,
                                     const OperatorJoinPredicate& primary_predicate,
                                     const std::vector<OperatorJoinPredicate>& secondary_predicates)
    : AbstractJoinOperator(OperatorType::JoinSortMerge, left, right, mode, primary_predicate, secondary_predicates) {}

std::shared_ptr<AbstractOperator> JoinSimdSortMerge::_on_deep_copy(
    const std::shared_ptr<AbstractOperator>& copied_left_input,
    const std::shared_ptr<AbstractOperator>& copied_right_input,
    std::unordered_map<const AbstractOperator*, std::shared_ptr<AbstractOperator>>& /*copied_ops*/) const {
  return std::make_shared<JoinSimdSortMerge>(copied_left_input, copied_right_input, _mode, _primary_predicate,
                                             _secondary_predicates);
}

void JoinSimdSortMerge::_on_set_parameters(const std::unordered_map<ParameterID, AllTypeVariant>& parameters) {}

std::shared_ptr<const Table> JoinSimdSortMerge::_on_execute() {
  Assert(supports({_mode, _primary_predicate.predicate_condition,
                   left_input_table()->column_data_type(_primary_predicate.column_ids.first),
                   right_input_table()->column_data_type(_primary_predicate.column_ids.second),
                   !_secondary_predicates.empty(), left_input_table()->type(), right_input_table()->type()}),
         "JoinSimdSortMerge does not support these parameters.");

  std::shared_ptr<const Table> left_input_table_ptr = _left_input->get_output();
  std::shared_ptr<const Table> right_input_table_ptr = _right_input->get_output();

  // Check column types
  const auto& left_column_type = left_input_table()->column_data_type(_primary_predicate.column_ids.first);
  DebugAssert(left_column_type == right_input_table()->column_data_type(_primary_predicate.column_ids.second),
              "Left and right column types do not match. The simd sort merge join requires matching column types.");

  // Create implementation to compute the join result
  resolve_data_type(left_column_type, [&](const auto type) {
    using ColumnDataType = typename decltype(type)::type;
    _impl = std::make_unique<JoinSimdSortMergeImpl<ColumnDataType>>(
        *this, left_input_table_ptr, right_input_table_ptr, _primary_predicate.column_ids.first,
        _primary_predicate.column_ids.second, _primary_predicate.predicate_condition, _mode, _secondary_predicates);
  });

  return _impl->_on_execute();
}

void JoinSimdSortMerge::_on_cleanup() {
  _impl.reset();
}

const std::string& JoinSimdSortMerge::name() const {
  static const auto name = std::string{"JoinSimdSortMerge"};
  return name;
}

template <typename T>
class JoinSimdSortMerge::JoinSimdSortMergeImpl : public AbstractReadOnlyOperatorImpl {
 public:
  JoinSimdSortMergeImpl(JoinSimdSortMerge& sort_merge_join, const std::shared_ptr<const Table>& left_input_table,
                        const std::shared_ptr<const Table>& right_input_table, ColumnID left_column_id,
                        ColumnID right_column_id, const PredicateCondition op, JoinMode mode,
                        const std::vector<OperatorJoinPredicate>& secondary_join_predicates)
      : _sort_merge_join{sort_merge_join},
        _left_input_table{left_input_table},
        _right_input_table{right_input_table},
        _primary_left_column_id{left_column_id},
        _primary_right_column_id{right_column_id},
        _primary_predicate_condition{op},
        _mode{mode},
        _secondary_join_predicates{secondary_join_predicates} {}

 protected:
  // NOLINTBEGIN(cppcoreguidelines-avoid-const-or-ref-data-members)
  JoinSimdSortMerge& _sort_merge_join;
  const std::shared_ptr<const Table> _left_input_table;
  const std::shared_ptr<const Table> _right_input_table;
  const ColumnID _primary_left_column_id;
  const ColumnID _primary_right_column_id;

  const PredicateCondition _primary_predicate_condition;
  const JoinMode _mode;

  const std::vector<OperatorJoinPredicate>& _secondary_join_predicates;
  // NOLINTEND(cppcoreguidelines-avoid-const-or-ref-data-members)

 public:
  std::shared_ptr<const Table> _on_execute() override {
    std::cout << "execute :)" << std::endl;
    return nullptr;
  }
};

template <>
class JoinSimdSortMerge::JoinSimdSortMergeImpl<pmr_string> : public AbstractReadOnlyOperatorImpl {
 public:
  JoinSimdSortMergeImpl(JoinSimdSortMerge& sort_merge_join, const std::shared_ptr<const Table>& left_input_table,
                        const std::shared_ptr<const Table>& right_input_table, ColumnID left_column_id,
                        ColumnID right_column_id, const PredicateCondition op, JoinMode mode,
                        const std::vector<OperatorJoinPredicate>& secondary_join_predicates)
      : _sort_merge_join{sort_merge_join},
        _left_input_table{left_input_table},
        _right_input_table{right_input_table},
        _primary_left_column_id{left_column_id},
        _primary_right_column_id{right_column_id},
        _primary_predicate_condition{op},
        _mode{mode},
        _secondary_join_predicates{secondary_join_predicates} {}

 protected:
  // NOLINTBEGIN(cppcoreguidelines-avoid-const-or-ref-data-members)
  JoinSimdSortMerge& _sort_merge_join;
  const std::shared_ptr<const Table> _left_input_table;
  const std::shared_ptr<const Table> _right_input_table;
  const ColumnID _primary_left_column_id;
  const ColumnID _primary_right_column_id;

  const PredicateCondition _primary_predicate_condition;
  const JoinMode _mode;

  const std::vector<OperatorJoinPredicate>& _secondary_join_predicates;
  // NOLINTEND(cppcoreguidelines-avoid-const-or-ref-data-members)

 public:
  std::shared_ptr<const Table> _on_execute() override {
    std::cout << "execute with pmr_string :)" << std::endl;
    return nullptr;
  }
};

}  // namespace hyrise
