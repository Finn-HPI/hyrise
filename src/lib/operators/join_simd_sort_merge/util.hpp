#pragma once

#include <array>
#include <vector>

#include "hyrise.hpp"
#include "scheduler/abstract_task.hpp"
#include "scheduler/job_task.hpp"

namespace hyrise {

namespace radix_partition {

constexpr auto RADIX_BITS = uint8_t{8};
constexpr auto HASH_MASK = std::size_t{(1u << RADIX_BITS) - 1};
constexpr auto PARTITION_SIZE = uint32_t{1u << RADIX_BITS};
constexpr auto CACHE_LINE_SIZE = std::size_t{64};
constexpr auto TUPLES_PER_CACHELINE = CACHE_LINE_SIZE / 8;

}  // namespace radix_partition

constexpr auto THREAD_COUNT = 8;

struct SimdElement {
  uint32_t index;
  uint32_t key;

  friend std::ostream& operator<<(std::ostream& stream, const SimdElement& element) {
    stream << "(" << element.key << "," << element.index << ")";
    return stream;
  }

  bool operator==(const SimdElement& other) const {
    return index == other.index && key == other.key;
  }
};

template <typename T>
using PerThread = std::array<T, THREAD_COUNT>;

template <typename T>
using PerHash = std::array<T, radix_partition::PARTITION_SIZE>;

template <typename T>
void spawn_and_wait_per_thread(PerThread<T>& data, auto&& per_thread_function) {
  auto tasks = std::vector<std::shared_ptr<AbstractTask>>{};
  tasks.reserve(THREAD_COUNT);
  for (auto thread_index = std::size_t{0}; thread_index < THREAD_COUNT; ++thread_index) {
    tasks.emplace_back(std::make_shared<JobTask>([thread_index, &data, &per_thread_function]() {
      if constexpr (requires { per_thread_function(data[thread_index], thread_index); }) {
        per_thread_function(data[thread_index], thread_index);
      } else {
        per_thread_function(data[thread_index]);
      }
    }));
  }

  Hyrise::get().scheduler()->schedule_and_wait_for_tasks(tasks);
}

template <typename T>
void spawn_and_wait_per_hash(PerHash<T>& data, auto&& per_hash_function) {
  auto tasks = std::vector<std::shared_ptr<AbstractTask>>{};
  tasks.reserve(radix_partition::PARTITION_SIZE);
  for (auto bucket_index = std::size_t{0}; bucket_index < radix_partition::PARTITION_SIZE; ++bucket_index) {
    tasks.emplace_back(std::make_shared<JobTask>([bucket_index, &data, &per_hash_function]() {
      if constexpr (requires { per_hash_function(data[bucket_index], bucket_index); }) {
        per_hash_function(data[bucket_index], bucket_index);
      } else {
        per_hash_function(data[bucket_index]);
      }
    }));
  }

  Hyrise::get().scheduler()->schedule_and_wait_for_tasks(tasks);
}

void spawn_and_wait_per_hash(auto&& per_hash_function) {
  auto tasks = std::vector<std::shared_ptr<AbstractTask>>{};
  tasks.reserve(radix_partition::PARTITION_SIZE);
  for (auto bucket_index = std::size_t{0}; bucket_index < radix_partition::PARTITION_SIZE; ++bucket_index) {
    tasks.emplace_back(std::make_shared<JobTask>([bucket_index, &per_hash_function]() {
      if constexpr (requires { per_hash_function(bucket_index); }) {
        per_hash_function(bucket_index);
      } else {
        per_hash_function();
      }
    }));
  }

  Hyrise::get().scheduler()->schedule_and_wait_for_tasks(tasks);
}
}  // namespace hyrise
