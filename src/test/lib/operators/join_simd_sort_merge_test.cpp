#include "base_test.hpp"
#include "operators/join_simd_sort_merge.hpp"
#include "operators/projection.hpp"
#include "operators/table_wrapper.hpp"

namespace hyrise {

class OperatorsJoinSimdSortMergeTest : public BaseTest {
 public:
  void SetUp() override {
    const auto dummy_table =
        std::make_shared<Table>(TableColumnDefinitions{{"a", DataType::Int, false}}, TableType::Data);
    dummy_input = std::make_shared<TableWrapper>(dummy_table);
    dummy_input->never_clear_output();
  }

  std::shared_ptr<AbstractOperator> dummy_input;
};

TEST_F(OperatorsJoinSimdSortMergeTest, DescriptionAndName) {
  const auto primary_predicate = OperatorJoinPredicate{{ColumnID{0}, ColumnID{0}}, PredicateCondition::Equals};
  const auto secondary_predicate = OperatorJoinPredicate{{ColumnID{0}, ColumnID{0}}, PredicateCondition::NotEquals};

  const auto join_operator =
      std::make_shared<JoinSimdSortMerge>(dummy_input, dummy_input, JoinMode::Inner, primary_predicate,
                                          std::vector<OperatorJoinPredicate>{secondary_predicate});

  EXPECT_EQ(join_operator->description(DescriptionMode::SingleLine),
            "JoinSimdSortMerge (Inner) Column #0 = Column #0 AND Column #0 != Column #0");
  EXPECT_EQ(join_operator->description(DescriptionMode::MultiLine),
            "JoinSimdSortMerge (Inner)\nColumn #0 = Column #0\nAND Column #0 != Column #0");

  dummy_input->execute();
  EXPECT_EQ(join_operator->description(DescriptionMode::SingleLine), "JoinSimdSortMerge (Inner) a = a AND a != a");
  EXPECT_EQ(join_operator->description(DescriptionMode::MultiLine), "JoinSimdSortMerge (Inner)\na = a\nAND a != a");

  EXPECT_EQ(join_operator->name(), "JoinSimdSortMerge");
}

TEST_F(OperatorsJoinSimdSortMergeTest, DeepCopy) {
  const auto primary_predicate = OperatorJoinPredicate{{ColumnID{0}, ColumnID{0}}, PredicateCondition::Equals};
  const auto join_operator =
      std::make_shared<JoinSimdSortMerge>(dummy_input, dummy_input, JoinMode::Left, primary_predicate);
  const auto abstract_join_operator_copy = join_operator->deep_copy();
  const auto join_operator_copy = std::dynamic_pointer_cast<JoinSimdSortMerge>(join_operator);

  ASSERT_TRUE(join_operator_copy);

  EXPECT_EQ(join_operator_copy->mode(), JoinMode::Left);
  EXPECT_EQ(join_operator_copy->primary_predicate(), primary_predicate);
  EXPECT_NE(join_operator_copy->left_input(), nullptr);
  EXPECT_NE(join_operator_copy->right_input(), nullptr);
}

TEST_F(OperatorsJoinSimdSortMergeTest, StringJoinColumn) {
  const auto test_table = std::make_shared<Table>(
      TableColumnDefinitions{
          {"a", DataType::String, false}, {"b", DataType::String, false}, {"c", DataType::String, false}},
      TableType::Data);

  test_table->append({"1", "2", "3"});
  test_table->append({"2", "1", "4"});
  test_table->append({"1", "2", "5"});

  const auto test_input = std::make_shared<TableWrapper>(test_table);
  test_input->never_clear_output();
  test_input->execute();
  const auto primary_predicate = OperatorJoinPredicate{{ColumnID{0}, ColumnID{1}}, PredicateCondition::Equals};

  // For inner joins, both join columns are clustered
  {
    const auto join_operator =
        std::make_shared<JoinSimdSortMerge>(test_input, test_input, JoinMode::Inner, primary_predicate);
    join_operator->execute();
  }
}

TEST_F(OperatorsJoinSimdSortMergeTest, IntJoinColumn) {
  const auto test_table = std::make_shared<Table>(
      TableColumnDefinitions{{"a", DataType::Int, false}, {"b", DataType::Int, false}, {"c", DataType::Int, false}},
      TableType::Data);

  test_table->append({1, 2, 3});
  test_table->append({2, 1, 4});
  test_table->append({1, 2, 5});
  test_table->append({3, 1, 1});
  test_table->append({3, 1, 1});
  test_table->append({3, 1, 1});
  test_table->append({3, 1, 1});
  test_table->append({3, 1, 1});
  test_table->append({3, 1, 1});
  test_table->append({3, 1, 1});
  test_table->append({3, 1, 1});
  test_table->append({3, 1, 1});

  const auto test_input = std::make_shared<TableWrapper>(test_table);
  test_input->never_clear_output();
  test_input->execute();
  const auto primary_predicate = OperatorJoinPredicate{{ColumnID{0}, ColumnID{1}}, PredicateCondition::Equals};

  // For inner joins, both join columns are clustered
  {
    const auto join_operator =
        std::make_shared<JoinSimdSortMerge>(test_input, test_input, JoinMode::Inner, primary_predicate);
    join_operator->execute();
  }
}

TEST_F(OperatorsJoinSimdSortMergeTest, IntRadixPartitionAndSort) {
  const auto test_table = std::make_shared<Table>(
      TableColumnDefinitions{{"a", DataType::Int, false}, {"b", DataType::Int, false}, {"c", DataType::Int, false}},
      TableType::Data);

  const auto num_items = 2048 * 8;
  for (auto i = int{num_items}; i >= 0; --i) {
    test_table->append({i, i, i});
  }
  // for (auto i = int{0}; i <= num_items; ++i) {
  //   test_table->append({i, i, i});
  // }

  const auto test_input = std::make_shared<TableWrapper>(test_table);
  test_input->never_clear_output();
  test_input->execute();
  const auto primary_predicate = OperatorJoinPredicate{{ColumnID{0}, ColumnID{1}}, PredicateCondition::Equals};

  // For inner joins, both join columns are clustered
  {
    const auto join_operator =
        std::make_shared<JoinSimdSortMerge>(test_input, test_input, JoinMode::Inner, primary_predicate);
    join_operator->execute();
  }
}

TEST_F(OperatorsJoinSimdSortMergeTest, FloatRadixPartitionAndSort) {
  EXPECT_TRUE(true);
  const auto test_table = std::make_shared<Table>(
      TableColumnDefinitions{
          {"a", DataType::Float, false}, {"b", DataType::Float, false}, {"c", DataType::Float, false}},
      TableType::Data);

  const auto num_items = 1024 * 8;
  for (auto index = int{num_items}; index >= 0; --index) {
    auto val = static_cast<float>(index);
    test_table->append({val, val, val});
  }

  // for (auto i = int{0}; i <= num_items; ++i) {
  //   test_table->append({i, i, i});
  // }

  const auto test_input = std::make_shared<TableWrapper>(test_table);
  test_input->never_clear_output();
  test_input->execute();
  const auto primary_predicate = OperatorJoinPredicate{{ColumnID{0}, ColumnID{1}}, PredicateCondition::Equals};

  // For inner joins, both join columns are clustered
  {
    const auto join_operator =
        std::make_shared<JoinSimdSortMerge>(test_input, test_input, JoinMode::Inner, primary_predicate);
    join_operator->execute();
  }
}

}  // namespace hyrise
