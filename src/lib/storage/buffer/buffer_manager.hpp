#pragma once

#include <memory>
#include <tuple>
#include "noncopyable.hpp"
#include "storage/buffer/frame.hpp"
#include "storage/buffer/migration_policy.hpp"
#include "storage/buffer/ssd_region.hpp"
#include "storage/buffer/types.hpp"
#include "storage/buffer/volatile_region.hpp"
#include "utils/pausable_loop_thread.hpp"

namespace hyrise {

template <typename PointedType>
class BufferManagedPtr;

/**
 * TODO
*/
class BufferManager : public Noncopyable {
 public:
  /**
   * Metrics are storing metric data that happens during allocation and access of the buffer manager.
  */
  struct Metrics {
    // The current amount of bytes being allocated
    std::size_t current_bytes_used = 0;

    // The total number of bytes being allocates
    std::size_t total_allocated_bytes = 0;

    // The total number of bytes that is unused when allocating memory on a page. Can be used to calculate internal fragmentation.
    std::size_t total_unused_bytes = 0;

    // The number of allocation
    std::size_t num_allocs = 0;

    // The number of deallocation
    std::size_t num_deallocs = 0;

    // Tracks the number of hits in the page_table
    std::size_t page_table_hits = 0;

    // Tracks the number of hits in the page_table
    std::size_t page_table_misses = 0;

    // Tracks the number of bytes read from SSD
    std::size_t total_bytes_read_from_ssd = 0;

    // Tracks the number of bytes written to SSD
    std::size_t total_bytes_written_to_ssd = 0;

    std::size_t total_pins = 0;

    std::size_t current_pins = 0;

    std::size_t num_ssd_reads = 0;

    std::size_t num_ssd_writes = 0;

    std::size_t num_dram_eviction_queue_purges = 0;

    std::size_t num_dram_eviction_queue_item_purges = 0;

    std::size_t num_dram_eviction_queue_adds = 0;

    std::size_t num_numa_eviction_queue_purges = 0;

    std::size_t num_numa_eviction_queue_item_purges = 0;

    std::size_t num_numa_eviction_queue_adds = 0;

    // TODO: ratio is defined in Spitfire paper. Lower values signfies lower duplication.
    std::size_t dram_numa_inclusivity_ratio = 0;

    // TODO: Internal fragmentation rate over all page sizes (lower is better)
    std::size_t internal_framentation_rate = 0;

    std::size_t num_madvice_free_calls = 0;  //TODO

    // TODO: read and write rates for all paths, dram, numa
  };

  struct Config {
    uint64_t dram_buffer_pool_size = 1UL << 30;  // 1 GB
    uint64_t numa_buffer_pool_size = 1UL << 34;  // 16 GB

    MigrationPolicy migration_policy = EagerMigrationPolicy{};

    uint8_t numa_memory_node = NO_NUMA_MEMORY_NODE;

    std::filesystem::path ssd_path = "~/.hyrise";

    bool enable_eviction_worker = false;

    BufferManagerMode mode = BufferManagerMode::DramSSD;

    static Config from_env();
  };

  BufferManager();

  BufferManager(const Config config);

  ~BufferManager();

  // TODO: Get page should not have page size type as parameter

  /**
   * @brief Get the page object
   * 
   * @param page_id 
   * @return std::unique_ptr<Page> 
   */
  std::byte* get_page(const PageID page_id, const PageSizeType size_type);

  /**
   * @brief Pin a page marks a page unavailable for replacement. It needs to be unpinned before it can be replaced.
   * 
   * @param page_id 
   */
  void pin_page(const PageID page_id, const PageSizeType size_type);

  /**
   * @brief Unpinning a page marks a page available for replacement. This acts as a soft-release without flushing
   * the page back to disk. Calls callback if the pin count is redced to zero.
   * 
   * @param page_id 
   */
  void unpin_page(const PageID page_id, const bool dirty = false);

  /**
   * @brief Get the page id and offset from ptr object. PageID is on its max 
   * if the page there was no page found. TODO: Return Buffer Managed Ptr
   * 
   * @param ptr 
   * @return std::pair<Frame*, std::ptrdiff_t> Pointer  to the frame and the offset in the frame
   */
  std::pair<Frame*, std::ptrdiff_t> unswizzle(const void* ptr);

  /**
   * Allocates pages to fullfil allocation request of the given bytes and alignment 
  */
  BufferManagedPtr<void> allocate(std::size_t bytes, std::size_t align = alignof(std::max_align_t));

  /**
   * Deallocates a pointer and frees the pages.
  */
  void deallocate(BufferManagedPtr<void> ptr, std::size_t bytes, std::size_t align = alignof(std::max_align_t));

  /**
   * @brief Helper function to get the BufferManager singleton. This avoids issues with circular dependencies as the implementation in the .cpp file.
   * 
   * @return BufferManager& 
   */
  static BufferManager& get_global_buffer_manager();  // TODO: Inline?

  /**
   * @brief Returns a snapshot of metrics holding information about allocations, page table hits etc. of the current buffer manager instance.
  */
  Metrics metrics();

  /**
   * @brief Reset the metrics of the buffer manager e.g. when starting a benchmark run.
  */
  void reset_metrics();

  BufferManager& operator=(BufferManager&& other);

  /**
   * Flush all pages to disk and clear the page table. The volatile regions are cleared by purging the eviction queue and unmapping the VM.
  */
  void flush_all_pages();

  /**
   * Get the current number of bytes used by the buffer manager on NUMA.
  */
  std::size_t numa_bytes_used() const;

  /**
   * Get the current number of bytes used by the buffer manager on DRAM.
  */
  std::size_t dram_bytes_used() const;

  /**
   * Reset all data in the internal data structures. TODO: Move to private
  */
  void clear();

 protected:
  friend class Hyrise;
  friend class BufferManagedPtrTest;
  friend class BufferManagerTest;
  friend class BufferPoolAllocatorTest;

 private:
  /**
   * Holds multiple sized buffer pools on either DRAM or NUMA memory.
  */
  struct BufferPools {
    template <typename RemoveFrameCallback>
    Frame* allocate_frame(const PageSizeType size_type, RemoveFrameCallback&& remove_frame_callback);
    void deallocate_frame(Frame* frame);

    // Purge the eviction queue
    void purge_eviction_queue();

    // Add a page to the eviction queue
    void add_to_eviction_queue(Frame* frame);

    // Read a page from disk into the buffer pool
    void read_page(Frame* frame);

    // Write out a page to disk
    void write_page(const Frame* frame);

    void clear();

    BufferPools(const PageType page_type, const size_t pool_size, const bool enable_eviction_worker,
                const std::shared_ptr<SSDRegion> ssd_region, const std::shared_ptr<BufferManager::Metrics> metrics);

    BufferPools& operator=(BufferPools&& other);

    // The maximum number of bytes that can be allocated
    const size_t _total_bytes;

    // The number of bytes that are currently used
    std::atomic_uint64_t _used_bytes;

    std::shared_ptr<SSDRegion> _ssd_region;

    // Memory Regions for each page size type
    std::array<std::unique_ptr<VolatileRegion>, NUM_PAGE_SIZE_TYPES> _buffer_pools;

    // Eviction Queue for pages that are not pinned
    std::shared_ptr<EvictionQueue> _eviction_queue;

    std::unique_ptr<PausableLoopThread> _eviction_worker;

    std::shared_ptr<BufferManager::Metrics> _metrics;

    const PageType _page_type;

    const bool enabled;
  };

  uint32_t get_pin_count(const PageID page_id);
  bool is_dirty(const PageID page_id);

  Frame* get_frame(const PageID page_id, const PageSizeType size_type);
  void remove_frame(Frame* frame);

  // Find a page in the buffer pool
  std::shared_ptr<SharedFrame> find_in_page_table(const PageID page_id);
  std::shared_ptr<SharedFrame> find_or_create_in_page_table(const PageID page_id);

  // Allocate a new page in the buffer pool by allocating a frame. May evict a page from the buffer pool.
  Frame* new_page(const PageSizeType size_type);

  // Remove a page from the buffer pool
  void remove_page(const PageID page_id);

  // Total number of pages currently in the buffer pool
  std::atomic_uint64_t _num_pages;

  // Metrics of buffer manager
  std::shared_ptr<Metrics> _metrics{};

  // Migration Policy to decide when to migrate pages to DRAM or NUMA buffer pools
  MigrationPolicy _migration_policy;

  // Memory Region for pages on SSD
  std::shared_ptr<SSDRegion> _ssd_region;

  // Memory Region for pages in DRAM using multiple volatile region for each page size type
  BufferPools _dram_buffer_pools;

  // Memory Region for pages on a NUMA nodes (preferrably memory only) using multiple volatile region for each page size type
  BufferPools _numa_buffer_pools;

  // Page Table that contains shared frames which are currently in the buffer pool
  PageTable _page_table;
  std::mutex _page_table_mutex;
};

}  // namespace hyrise