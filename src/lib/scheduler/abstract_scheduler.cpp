#include "abstract_scheduler.hpp"

#include <memory>
#include <vector>

#include "scheduler/abstract_task.hpp"
#include "utils/assert.hpp"

namespace hyrise {

void AbstractScheduler::wait_for_tasks(const std::vector<std::shared_ptr<AbstractTask>>& tasks) {
  if constexpr (HYRISE_DEBUG) {
    for (const auto& task : tasks) {
      Assert(task->is_scheduled(), "In order to wait for a task’s completion, it must have been scheduled first.");
    }
  }

  // In case wait_for_tasks() is called from a task being executed in a worker, let the worker handle the join()-ing,
  // otherwise join right here.
  const auto worker = Worker::get_this_thread_worker();
  if (worker) {
    worker->_wait_for_tasks(tasks);
  } else {
    for (const auto& task : tasks) {
      task->_join();
    }
  }
}

void AbstractScheduler::_group_tasks(const std::vector<std::shared_ptr<AbstractTask>>& tasks) const {
  // Do nothing - grouping tasks is implementation-defined.
}

void AbstractScheduler::schedule_tasks(const std::vector<std::shared_ptr<AbstractTask>>& tasks) {
  for (const auto& task : tasks) {
    task->schedule();
  }
}

void AbstractScheduler::schedule_and_wait_for_tasks(const std::vector<std::shared_ptr<AbstractTask>>& tasks) {
  // _group_tasks(tasks);
  schedule_tasks(tasks);
  wait_for_tasks(tasks);
}

}  // namespace hyrise
