#include <memory>

#include "base_test.hpp"

#include "logical_query_plan/dummy_table_node.hpp"
#include "logical_query_plan/lqp_utils.hpp"

namespace hyrise {

class DummyTableNodeTest : public BaseTest {
 protected:
  void SetUp() override {
    _dummy_table_node = DummyTableNode::make();
  }

  std::shared_ptr<DummyTableNode> _dummy_table_node;
};

TEST_F(DummyTableNodeTest, Description) {
  EXPECT_EQ(_dummy_table_node->description(), "[DummyTable]");
}

TEST_F(DummyTableNodeTest, OutputColumnExpressions) {
  EXPECT_EQ(_dummy_table_node->output_expressions().size(), 0u);
}

TEST_F(DummyTableNodeTest, HashingAndEqualityCheck) {
  EXPECT_EQ(*_dummy_table_node, *_dummy_table_node);
  EXPECT_EQ(*_dummy_table_node, *DummyTableNode::make());

  EXPECT_EQ(_dummy_table_node->hash(), _dummy_table_node->hash());
  EXPECT_EQ(_dummy_table_node->hash(), DummyTableNode::make()->hash());
}

TEST_F(DummyTableNodeTest, Copy) {
  EXPECT_EQ(*_dummy_table_node->deep_copy(), *DummyTableNode::make());
}

TEST_F(DummyTableNodeTest, NodeExpressions) {
  ASSERT_EQ(_dummy_table_node->node_expressions.size(), 0u);
}

TEST_F(DummyTableNodeTest, NoUniqueColumnCombinations) {
  // A DummyTableNode is just a wrapper for a single value and should not provide meaningful data dependencies (though a
  // single row is obviously unique).
  EXPECT_TRUE(_dummy_table_node->unique_column_combinations().empty());
}

TEST_F(DummyTableNodeTest, NoOrderDependencies) {
  // A DummyTableNode is just a wrapper for a single value and should not provide meaningful data dependencies.
  EXPECT_TRUE(_dummy_table_node->order_dependencies().empty());
}

TEST_F(DummyTableNodeTest, NoInclusionDependencies) {
  // A DummyTableNode is just a wrapper for a single value and should not provide meaningful data dependencies.
  EXPECT_TRUE(_dummy_table_node->inclusion_dependencies().empty());
}

TEST_F(DummyTableNodeTest, IsColumnNullable) {
  EXPECT_THROW(_dummy_table_node->is_column_nullable(ColumnID{0}), std::logic_error);
}

}  // namespace hyrise
