#include "unary_minus_expression.hpp"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include "all_type_variant.hpp"
#include "expression/abstract_expression.hpp"
#include "operators/abstract_operator.hpp"
#include "utils/assert.hpp"

namespace hyrise {

UnaryMinusExpression::UnaryMinusExpression(const std::shared_ptr<AbstractExpression>& argument)
    : AbstractExpression(ExpressionType::UnaryMinus, {argument}) {
  Assert(argument->data_type() != DataType::String, "Cannot negate strings.");
}

std::shared_ptr<AbstractExpression> UnaryMinusExpression::argument() const {
  return arguments[0];
}

std::shared_ptr<AbstractExpression> UnaryMinusExpression::_on_deep_copy(
    std::unordered_map<const AbstractOperator*, std::shared_ptr<AbstractOperator>>& copied_ops) const {
  return std::make_shared<UnaryMinusExpression>(argument()->deep_copy(copied_ops));
}

std::string UnaryMinusExpression::description(const DescriptionMode mode) const {
  auto stream = std::stringstream{};
  stream << "-" << _enclose_argument(*argument(), mode);
  return stream.str();
}

DataType UnaryMinusExpression::data_type() const {
  return argument()->data_type();
}

bool UnaryMinusExpression::_shallow_equals(const AbstractExpression& expression) const {
  DebugAssert(dynamic_cast<const UnaryMinusExpression*>(&expression),
              "Different expression type should have been caught by AbstractExpression::operator==.");
  return true;
}

}  // namespace hyrise
