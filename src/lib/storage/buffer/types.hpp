#pragma once

#include <limits>
#include "strong_typedef.hpp"
#include "storage/buffer/types.hpp"

// Declare types here to avoid problems with circular dependency
STRONG_TYPEDEF(uint32_t, PageID);
STRONG_TYPEDEF(uint32_t, FrameID);

namespace hyrise {
constexpr PageID INVALID_PAGE_ID{std::numeric_limits<PageID::base_type>::max()};
constexpr FrameID INVALID_FRAME_ID{std::numeric_limits<FrameID::base_type>::max()};

enum class PageSizeType : std::size_t {
  KiB32 = 1 << 15,
  KiB64 = 1 << 16,
  KiB128 = 1 << 17,
  KiB256 = 1 << 18,
};

// Copied from boost::interprocess, because #include <boost/type_traits/add_reference.hpp> was not enough
// I guess, because of "typedef nat &type" that can be used as reference dummy type
struct nat {};

template <typename T>
struct add_reference {
  typedef T& type;
};

template <class T>
struct add_reference<T&> {
  typedef T& type;
};

template <>
struct add_reference<void> {
  typedef nat& type;
};

template <>
struct add_reference<const void> {
  typedef const nat& type;
};

}  // namespace hyrise