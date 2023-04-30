#pragma once

#include <forward_list>
#include "storage/buffer/types.hpp"

namespace hyrise {

template <typename T>
class BufferPtr;
class BufferManager;

class MemoryResource {
 public:
  virtual ~MemoryResource() = default;
  virtual BufferPtr<void> allocate(const std::size_t bytes, const std::size_t alignment) = 0;
  virtual void deallocate(BufferPtr<void> p, const std::size_t bytes, const std::size_t alignment) = 0;
};

/**
 * The MonotonicBufferResource is a memory resource that allocates memory in pages without deallocation. Its supposed to be fast 
 * and helpful to reduce internal fragmentation for small allocations in pages when using the buffer manager. It was designed to 
 * reduce the memory usage of pmr_string from a pmr_vector. Inspired by std::pmr::monotonic_buffer_resource and adapted some code
 * from boost::container::pmr::monotonic_buffer_resource. It does not contain a release function or a destructor as we assume
 * this to be taken over by the memory ownership model of the buffer manager.
 * 
 * Properties:
 * - Initial page size is 8 KiB.
 * 
 * - Deallocation has no effect.
 * 
 *  - The memory is never released by the memory resource. The ownership of the memory is passed to the BufferPtr. When all references 
 * to a frame/page are gone, the page is released by the BufferManager. 
 * 
 * - When the current page is exhausted, a new page is allocated. The next biggest page size is used until 256 KiB is reached 
 * (thus geometrically increasing). Any new page will be 256 KiB from there on. Larger page sizes are reservered for special cases.
 * 
 * - If the bytes to be allocated fill up more than 80% of a potential page size type, a new page is allocated regardless of the current page. 
 * The existing current page is not touched and kept for the next allocation.
*/
class MonotonicBufferResource : public MemoryResource {
 public:
  // Largest page size to be allocated for small allocations is 256 KiB
  static constexpr PageSizeType MAX_PAGE_SIZE_TYPE = PageSizeType::KiB256;

  // First page size to be allocated for small allocations is 8 KiB
  static constexpr PageSizeType INITIAL_PAGE_SIZE_TYPE = PageSizeType::KiB8;

  MonotonicBufferResource();

  MonotonicBufferResource(MemoryResource* memory_resource, const PageSizeType initial_size = INITIAL_PAGE_SIZE_TYPE);

  BufferPtr<void> allocate(std::size_t bytes, std::size_t alignment);
  void deallocate(BufferPtr<void> ptr, std::size_t bytes, std::size_t alignment);

  /**
    * Check how many bytes are left in the current page including the alignment.
    */
  std::size_t remaining_storage(std::size_t alignment, std::size_t& wasted_due_to_alignment) const noexcept;

  std::size_t remaining_storage(std::size_t alignment = 1u) const noexcept;

 private:
  void increase_next_buffer_at_least_to(std::size_t minimum_size);

  void increase_next_buffer();

  // Allocate memory on the current frame
  BufferPtr<void> allocate_from_current(std::size_t aligner, std::size_t bytes);

  // Check if we can fill a new page with the given bytes at least to NEW_PAGE_FILL_RATIO (e.g. 80%)
  bool fills_page(std::size_t bytes) const;

  // Up to this ratio, the buffer is used, otherwise it is just allocating a new buffer
  static constexpr float NEW_PAGE_FILL_RATIO = 0.8f;

  MemoryResource* _memory_resource;
  std::shared_ptr<SharedFrame> _current_frame;

  std::size_t _current_buffer_pos;
  std::size_t _current_buffer_size;
  std::size_t _next_buffer_size;
};

/**
 * NewDeleteMemoryResource uses classical new-delete calls, but returns a BufferPtr. Inspired by std::pmr::new_delete_resource. 
*/
class NewDeleteMemoryResource : public MemoryResource {
 public:
  BufferPtr<void> allocate(std::size_t bytes, std::size_t alignment);
  void deallocate(BufferPtr<void> ptr, std::size_t bytes, std::size_t alignment);
};

}  // namespace hyrise