#pragma once

#include <boost/container/pmr/memory_resource.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/move/utility.hpp>
#include "storage/buffer/buffer_manager.hpp"
#include "storage/buffer/buffer_pool_allocator_observer.hpp"
#include "storage/buffer/frame.hpp"

#include "utils/assert.hpp"

namespace hyrise {

/**^
 * The BufferPoolAllocator is a custom, polymorphic allocator that uses the BufferManager to allocate and deallocate pages.
 * 
 * TODO: Combine this allocator with scoped allocator to use same page like monotonic buffer resource for strings
*/
template <class T>
class BufferPoolAllocator {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using void_pointer = void*;
  using difference_type = std::ptrdiff_t;
  using reference = typename add_reference<T>::type;
  using const_reference = typename add_reference<const T>::type;

  BufferPoolAllocator() : _memory_resource(get_buffer_manager_memory_resource()) {}

  BufferPoolAllocator(boost::container::pmr::memory_resource* memory_resource) : _memory_resource(memory_resource) {}

  BufferPoolAllocator(boost::container::pmr::memory_resource* memory_resource,
                      std::shared_ptr<BufferPoolAllocatorObserver> observer)
      : _memory_resource(memory_resource), _observer(observer) {}

  BufferPoolAllocator(const BufferPoolAllocator& other) noexcept {
    _memory_resource = other.memory_resource();
    _observer = other.current_observer();
  }

  template <class U>
  BufferPoolAllocator(const BufferPoolAllocator<U>& other) noexcept {
    _memory_resource = other.memory_resource();
    _observer = other.current_observer();
  }

  BufferPoolAllocator& operator=(const BufferPoolAllocator& other) noexcept {
    _memory_resource = other.memory_resource();
    _observer = other.current_observer();
    return *this;
  }

  template <class U>
  bool operator==(const BufferPoolAllocator<U>& other) const noexcept {
    return _memory_resource == other.memory_resource() && _observer.lock() == other.current_observer().lock();
  }

  template <class U>
  bool operator!=(const BufferPoolAllocator<U>& other) const noexcept {
    return _memory_resource != other.memory_resource() || _observer.lock() != other.current_observer().lock();
  }

  [[nodiscard]] pointer allocate(std::size_t n) {
    auto ptr = _memory_resource->allocate(sizeof(value_type) * n, alignof(T));
    if (auto observer = _observer.lock()) {
      const auto page_id = BufferManager::get().find_page(ptr);
      observer->on_allocate(page_id);
    }
    return static_cast<pointer>(ptr);
  }

  void deallocate(const pointer& ptr, std::size_t n) {
    // TODO: Count deallocates for nested resources
    if (auto observer = _observer.lock()) {
      const auto page_id = BufferManager::get().find_page(ptr);
      observer->on_deallocate(page_id);
    }
    _memory_resource->deallocate(static_cast<void_pointer>(ptr), sizeof(value_type) * n, alignof(T));
  }

  boost::container::pmr::memory_resource* memory_resource() const noexcept {
    return _memory_resource;
  }

  BufferPoolAllocator select_on_container_copy_construction() const noexcept {
    return BufferPoolAllocator(_memory_resource, _observer.lock());
  }

  void register_observer(std::shared_ptr<BufferPoolAllocatorObserver> observer) {
    if (!_observer.expired()) {
      Fail("An observer is already registered");
    }
    _observer = observer;
  }

  std::weak_ptr<BufferPoolAllocatorObserver> current_observer() const {
    return _observer;
  }

  pointer address(reference value) const {
    return pointer(boost::addressof(value));
  }

  //!Returns address of non mutable object.
  //!Never throws
  const_pointer address(const_reference value) const {
    return const_pointer(boost::addressof(value));
  }

 private:
  std::weak_ptr<BufferPoolAllocatorObserver> _observer;
  boost::container::pmr::memory_resource* _memory_resource;
};

}  // namespace hyrise