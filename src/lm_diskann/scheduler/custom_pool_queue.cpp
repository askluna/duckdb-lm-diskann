// ==========================================================================
// File: diskann/scheduler/custom_pool_queue.cpp
// ==========================================================================
#include "custom_pool_queue.hpp"

namespace diskann {
namespace scheduler {

void PriorityTaskQueue::push(scheduler::CustomPoolTask task, scheduler::TaskPriority priority) {
	{
		std::lock_guard<std::mutex> lock(queue_mutex_);
		if (priority == scheduler::TaskPriority::HIGH) {
			high_priority_queue_.push(std::move(task));
		} else {
			medium_priority_queue_.push(std::move(task));
		}
	} // Mutex is released here
	cv_.notify_one(); // Notify one waiting worker
}

bool PriorityTaskQueue::wait_pop_high_timed(scheduler::CustomPoolTask &out_task, std::chrono::microseconds timeout) {
	std::unique_lock<std::mutex> lock(queue_mutex_);
	// Wait until predicate is true OR timeout occurs
	if (cv_.wait_for(lock, timeout, [this] { return !high_priority_queue_.empty(); })) {
		// Predicate is true: queue is not empty
		out_task = std::move(high_priority_queue_.front());
		high_priority_queue_.pop();
		return true;
	}
	// Timed out or spurious wakeup with empty queue
	return false;
}

bool PriorityTaskQueue::wait_pop_medium_timed(scheduler::CustomPoolTask &out_task, std::chrono::microseconds timeout) {
	std::unique_lock<std::mutex> lock(queue_mutex_);
	if (cv_.wait_for(lock, timeout, [this] { return !medium_priority_queue_.empty(); })) {
		out_task = std::move(medium_priority_queue_.front());
		medium_priority_queue_.pop();
		return true;
	}
	return false;
}

size_t PriorityTaskQueue::medium_queue_size() const {
	std::lock_guard<std::mutex> lock(queue_mutex_);
	return medium_priority_queue_.size();
}

size_t PriorityTaskQueue::high_queue_size() const {
	std::lock_guard<std::mutex> lock(queue_mutex_);
	return high_priority_queue_.size();
}

void PriorityTaskQueue::notify_all_waiters() {
	cv_.notify_all();
}

} // namespace scheduler
} // namespace diskann