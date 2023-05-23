#pragma once

#include <boost/container/pmr/memory_resource.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/move/utility.hpp>
#include "storage/buffer/buffer_pool_allocator_observer.hpp"
#include "storage/buffer/buffer_ptr.hpp"
#include "storage/buffer/frame.hpp"
#include "storage/buffer/memory_resource.hpp"

#include "utils/assert.hpp"

namespace hyrise {

/**
 * The BufferPoolAllocator is a custom, polymorphic allocator that uses the BufferManager to allocate and deallocate pages.
*/
template <class T>
class BufferPoolAllocator {
 public:
  using value_type = T;
  using pointer = BufferPtr<T>;
  using const_pointer = BufferPtr<const T>;
  using void_pointer = BufferPtr<void>;
  using difference_type = typename pointer::difference_type;

  BufferPoolAllocator() : _memory_resource(get_global_monotonic_buffer_resource()) {}

  BufferPoolAllocator(MemoryResource* memory_resource) : _memory_resource(memory_resource) {}

  BufferPoolAllocator(MemoryResource* memory_resource, std::shared_ptr<BufferPoolAllocatorObserver> observer)
      : _memory_resource(memory_resource), _observer(observer) {}

  BufferPoolAllocator(boost::container::pmr::memory_resource* resource) : _memory_resource(nullptr) {
    Fail("The current BufferPoolAllocator cannot take a boost memory_resource");
  }

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
    auto ptr = pointer(_memory_resource->allocate(sizeof(value_type) * n, alignof(T)), typename pointer::AllocTag{});
    ptr._frame->increase_ref_count();  // TODO: Inject auto ref count in buffer ptr <void>
    if (auto observer = _observer.lock()) {
      const auto frame = ptr.get_frame();
      observer->on_allocate(frame);
    }
    // Manually increase the ref count when passing the pointer to the data structure
    DebugAssert(ptr.get_frame() == nullptr || ptr.get_frame()->is_resident(),
                "Trying to allocate on a non-resident frame");
    return ptr;
  }

  void deallocate(pointer const ptr, std::size_t n) {
    DebugAssert(ptr.get_frame() == nullptr || ptr.get_frame()->is_resident(),
                "Trying to deallocate on a non-resident frame");

    if (auto observer = _observer.lock()) {
      auto frame = ptr.get_frame();
      observer->on_deallocate(frame);
    }
    // TODO: Thread local memory resource might be destroyed (e.g. monotonic)
    _memory_resource->deallocate(static_cast<void_pointer>(ptr), sizeof(value_type) * n, alignof(T));
    ptr._frame->decrease_ref_count();
  }

  MemoryResource* memory_resource() const noexcept {
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

 private:
  std::weak_ptr<BufferPoolAllocatorObserver> _observer;
  MemoryResource* _memory_resource;
};

}  // namespace hyrise