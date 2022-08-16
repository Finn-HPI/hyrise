#pragma once

#include <functional>
#include <optional>
#include <type_traits>

#include <boost/range.hpp>
#include <boost/range/join.hpp>

#include "all_type_variant.hpp"
#include "constant_mappings.hpp"
#include "storage/pos_lists/row_id_pos_list.hpp"
#include "types.hpp"

namespace hyrise {

// Generic class which handles the actual scanning of a sorted segment
template <typename IteratorType, typename SearchValueType>
class SortedSegmentSearch {
 public:
  SortedSegmentSearch(IteratorType begin, IteratorType end, const SortMode& sorted_by, const bool nullable,
                      const PredicateCondition& predicate_condition, const SearchValueType& search_value)
      : _begin{begin},
        _end{end},
        _predicate_condition{predicate_condition},
        _first_search_value{search_value},
        _second_search_value{std::nullopt},
        _nullable{nullable},
        _is_ascending{sorted_by == SortMode::Ascending} {}

  // For SortedSegmentBetweenSearch
  SortedSegmentSearch(IteratorType begin, IteratorType end, const SortMode& sorted_by, const bool nullable,
                      const PredicateCondition& predicate_condition, const SearchValueType& left_value,
                      const SearchValueType& right_value)
      : _begin{begin},
        _end{end},
        _predicate_condition{predicate_condition},
        _first_search_value{left_value},
        _second_search_value{right_value},
        _nullable{nullable},
        _is_ascending{sorted_by == SortMode::Ascending} {}

  void scan_sorted_segment(const ChunkID chunk_id, RowIDPosList& matches,
                           const std::shared_ptr<const AbstractPosList>& position_filter) {
    if (_nullable) {
      // Decrease the effective sort range by excluding null values.
      _begin = std::lower_bound(_begin, _end, false,
                                [](const auto& segment_position, const auto& _) { return segment_position.is_null(); });
    }

    if (_predicate_condition == PredicateCondition::NotEquals) {
      _handle_not_equals(chunk_id, matches, position_filter);
      return;
    }

    if (_second_search_value) {
      _set_begin_and_end_positions_for_between_scan();
    } else {
      _set_begin_and_end_positions_for_vs_value_scan();
    }
    _write_rows_to_matches(_begin, _end, chunk_id, matches, position_filter);
  }

  // Flags to indicate whether a shortcut was taken to skip scanning.
  bool no_rows_matching{false};
  bool all_rows_matching{false};

 private:
  /**
   * _get_first_bound and _get_last_bound are used to retrieve the lower and upper bound in a sorted segment but are
   * independent of its sort order. _get_first_bound will always return the bound with the smaller offset and
   * _get_last_bound will return the bigger offset.
   * On a segment sorted in ascending sort order they would work analogously to lower_bound and upper_bound. For
   * descending sort orders, _get_first_bound will actually return an upper bound and _get_last_bound the lower one.
   * However, the first offset will always point to an entry matching the search value, whereas last offset points to
   * the entry behind the last matching one.
   */
  IteratorType _get_first_bound(const SearchValueType& search_value, const IteratorType begin,
                                const IteratorType end) const {
    if (_is_ascending) {
      return std::lower_bound(begin, end, search_value, [](const auto& segment_position, const auto& value) {
        return segment_position.value() < value;
      });
    } else {
      return std::lower_bound(begin, end, search_value, [](const auto& segment_position, const auto& value) {
        return segment_position.value() > value;
      });
    }
  }

  IteratorType _get_last_bound(const SearchValueType& search_value, const IteratorType begin,
                               const IteratorType end) const {
    if (_is_ascending) {
      return std::upper_bound(begin, end, search_value, [](const auto& value, const auto& segment_position) {
        return segment_position.value() > value;
      });
    } else {
      return std::upper_bound(begin, end, search_value, [](const auto& value, const auto& segment_position) {
        return segment_position.value() < value;
      });
    }
  }

  // This function sets the offset(s) which delimit the result set based on the predicate condition and the sort order
  void _set_begin_and_end_positions_for_vs_value_scan() {
    if (_begin == _end) {
      no_rows_matching = true;
      return;
    }

    auto first_value = _begin->value();
    auto last_value = (_end - 1)->value();
    auto predicate_condition = _predicate_condition;

    // If descending: exchange predicate condition and first/last values.
    if (!_is_ascending) {
      std::swap(first_value, last_value);
      predicate_condition = flip_predicate_condition(_predicate_condition);
    }

    // Decide on early out when either everything or nothing matches.
    switch (_predicate_condition) {
      case PredicateCondition::Equals:
        all_rows_matching = first_value == _first_search_value && last_value == _first_search_value;
        no_rows_matching = first_value > _first_search_value || last_value < _first_search_value;
        break;
      case PredicateCondition::GreaterThanEquals:
        all_rows_matching = first_value >= _first_search_value;
        no_rows_matching = last_value < _first_search_value;
        break;
      case PredicateCondition::GreaterThan:
        all_rows_matching = first_value > _first_search_value;
        no_rows_matching = last_value <= _first_search_value;
        break;
      case PredicateCondition::LessThanEquals:
        all_rows_matching = last_value <= _first_search_value;
        no_rows_matching = first_value > _first_search_value;
        break;
      case PredicateCondition::LessThan:
        all_rows_matching = last_value < _first_search_value;
        no_rows_matching = first_value >= _first_search_value;
        break;
      default:
        Fail("Unsupported predicate condition encountered");
    }

    if (no_rows_matching) {
      _begin = _end;
    }

    if (all_rows_matching || no_rows_matching) {
      return;
    }

    switch (predicate_condition) {
      case PredicateCondition::Equals:
        _begin = _get_first_bound(_first_search_value, _begin, _end);
        _end = _get_last_bound(_first_search_value, _begin, _end);
        return;
      case PredicateCondition::GreaterThanEquals:
        _begin = _get_first_bound(_first_search_value, _begin, _end);
        return;
      case PredicateCondition::GreaterThan:
        _begin = _get_last_bound(_first_search_value, _begin, _end);
        return;
      case PredicateCondition::LessThanEquals:
        _end = _get_last_bound(_first_search_value, _begin, _end);
        return;
      case PredicateCondition::LessThan:
        _end = _get_first_bound(_first_search_value, _begin, _end);
        return;
      default:
        Fail("Unsupported predicate condition encountered");
    }
  }

  // This function sets the offset(s) that delimit the result set based on the predicate condition and the sort order
  void _set_begin_and_end_positions_for_between_scan() {
    DebugAssert(_second_search_value, "Second Search Value must be set for between scan");
    if (_begin == _end) {
      no_rows_matching = true;
      return;
    }

    auto first_value = _begin->value();
    auto last_value = (_end - 1)->value();

    auto predicate_condition = _predicate_condition;
    auto first_search_value = _first_search_value;
    auto second_search_value = *_second_search_value;

    // If descending: exchange predicate condition, search values, and first/last values.
    if (!_is_ascending) {
      switch (_predicate_condition) {
        case PredicateCondition::BetweenLowerExclusive:
          predicate_condition = PredicateCondition::BetweenUpperExclusive;
          break;
        case PredicateCondition::BetweenUpperExclusive:
          predicate_condition = PredicateCondition::BetweenLowerExclusive;
          break;
        case PredicateCondition::BetweenInclusive:
        case PredicateCondition::BetweenExclusive:
          break;
        default:
          Fail("Unsupported predicate condition encountered");
      }

      std::swap(first_value, last_value);
      std::swap(first_search_value, second_search_value);
    }

    const auto lower_exclusive = _predicate_condition == PredicateCondition::BetweenExclusive ||
                                 _predicate_condition == PredicateCondition::BetweenLowerExclusive;
    const auto upper_exclusive = _predicate_condition == PredicateCondition::BetweenExclusive ||
                                 _predicate_condition == PredicateCondition::BetweenUpperExclusive;

    // Early out if everything matches.
    const auto first_value_matches =
        lower_exclusive ? first_value > _first_search_value : first_value >= _first_search_value;
    const auto last_value_matches =
        upper_exclusive ? last_value < *_second_search_value : last_value <= *_second_search_value;
    if (first_value_matches && last_value_matches) {
      all_rows_matching = true;
      return;
    }

    // Early out if nothing matches.
    const auto first_value_matches_not =
        upper_exclusive ? first_value >= *_second_search_value : first_value > *_second_search_value;
    const auto last_value_matches_not =
        lower_exclusive ? last_value <= _first_search_value : last_value < _first_search_value;
    if (first_value_matches_not || last_value_matches_not) {
      no_rows_matching = true;
      _begin = _end;
      return;
    }

    // This implementation uses behaviour which resembles std::equal_range's behaviour since it, too, calculates two
    // different bounds. However, equal_range is designed to compare to a single search value, whereas in this case, the
    // upper and lower search value (if given) will differ.
    switch (predicate_condition) {
      case PredicateCondition::BetweenInclusive:
        _begin = _get_first_bound(first_search_value, _begin, _end);
        _end = _get_last_bound(second_search_value, _begin, _end);
        return;
      case PredicateCondition::BetweenLowerExclusive:  // upper inclusive
        _begin = _get_last_bound(first_search_value, _begin, _end);
        _end = _get_last_bound(second_search_value, _begin, _end);
        return;
      case PredicateCondition::BetweenUpperExclusive:
        _begin = _get_first_bound(first_search_value, _begin, _end);
        _end = _get_first_bound(second_search_value, _begin, _end);
        return;
      case PredicateCondition::BetweenExclusive:
        _begin = _get_last_bound(first_search_value, _begin, _end);
        _end = _get_first_bound(second_search_value, _begin, _end);
        return;
      default:
        Fail("Unsupported predicate condition encountered");
    }
  }

  /*
   * NotEquals may result in two matching ranges (one below and one above the search_value) and needs special handling.
   * The function contains four early outs. These are all only for performance reasons and, if removed, would not
   * change the functionality.
   *
   * Note: All comments within this method are written from the point of ranges in ascending sort order.
   */
  void _handle_not_equals(const ChunkID chunk_id, RowIDPosList& matches,
                          const std::shared_ptr<const AbstractPosList>& position_filter) {
    const auto first_bound = _get_first_bound(_first_search_value, _begin, _end);
    if (first_bound == _end) {
      // Neither the search value nor anything greater than it are found. Output the whole range and skip the call to
      // _get_last_bound().
      _write_rows_to_matches(_begin, _end, chunk_id, matches, position_filter);
      return;
    }

    if (first_bound->value() != _first_search_value) {
      // If the first value >= search value is not equal to the search value, then the search value doesn't occur at
      // all. Output the whole range and skip the call to _get_last_bound().
      _write_rows_to_matches(_begin, _end, chunk_id, matches, position_filter);
      return;
    }

    // At this point, first_bound points to the first occurrence of the search value.
    const auto last_bound = _get_last_bound(_first_search_value, _begin, _end);
    if (last_bound == _end) {
      // If no value > search value is found, output everything from start to first occurrence and skip the need for
      // boost::join().
      _write_rows_to_matches(_begin, first_bound, chunk_id, matches, position_filter);
      return;
    }

    if (first_bound == _begin) {
      // If the search value is right at the start, output everything from the first value > search value to end and
      // skip the need for boost::join().
      _write_rows_to_matches(last_bound, _end, chunk_id, matches, position_filter);
      return;
    }

    const auto range = boost::range::join(boost::make_iterator_range(_begin, first_bound),
                                          boost::make_iterator_range(last_bound, _end));
    _write_rows_to_matches(range.begin(), range.end(), chunk_id, matches, position_filter);
  }

  template <typename ResultIteratorType>
  void _write_rows_to_matches(ResultIteratorType begin, ResultIteratorType end, const ChunkID chunk_id,
                              RowIDPosList& matches,
                              const std::shared_ptr<const AbstractPosList>& position_filter) const {
    if (begin == end) {
      return;
    }

    // General note: If the predicate is NotEquals, there might be two ranges that match.
    // These two ranges might have been combined into a single one via boost::join(range_1, range_2).
    // See _handle_not_equals for further details.

    size_t output_idx = matches.size();

    matches.resize(matches.size() + std::distance(begin, end));

    /**
     * If the range of matches consists of continuous ChunkOffsets we can speed up the writing by calculating the
     * offsets based on the first offset instead of calling chunk_offset() for every match. ChunkOffsets in
     * position_filter are not necessarily continuous. The same is true for NotEquals because the result might consist
     * of 2 ranges.
     */
    if (position_filter || _predicate_condition == PredicateCondition::NotEquals) {
      for (; begin != end; ++begin) {
        matches[output_idx++] = RowID{chunk_id, begin->chunk_offset()};
      }
    } else {
      const auto first_offset = begin->chunk_offset();
      const auto distance = std::distance(begin, end);

      for (auto chunk_offset = ChunkOffset{0}; chunk_offset < distance; ++chunk_offset) {
        matches[output_idx++] = RowID{chunk_id, ChunkOffset{first_offset + chunk_offset}};
      }
    }
  }

  // _begin and _end will be modified to match the search range and will be passed to the ResultConsumer, except when
  // handling NotEquals (see _handle_not_equals).
  IteratorType _begin;
  IteratorType _end;
  const PredicateCondition _predicate_condition;
  const SearchValueType _first_search_value;
  const std::optional<SearchValueType> _second_search_value;
  const bool _nullable;
  const bool _is_ascending;
};

}  // namespace hyrise
