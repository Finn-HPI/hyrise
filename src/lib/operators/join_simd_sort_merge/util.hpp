#pragma once

#include <array>
#include <vector>

#include "all_type_variant.hpp"
#include "expression/window_function_expression.hpp"
#include "hyrise.hpp"
#include "scheduler/abstract_task.hpp"
#include "scheduler/job_task.hpp"
#include "types.hpp"

namespace hyrise {

constexpr auto THREAD_COUNT = 8;

struct SimdElement {
  uint32_t key;
  uint32_t index;

  friend std::ostream& operator<<(std::ostream& stream, const SimdElement& element) {
    stream << "SimdElement(" << element.key << "," << element.index << ")";
    return stream;
  }
};

template <typename T>
using PerHash = std::array<T, THREAD_COUNT>;

template <typename T>
void spawn_and_wait_per_thread(PerHash<T>& data, auto&& per_thread_function) {
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
}  // namespace hyrise
