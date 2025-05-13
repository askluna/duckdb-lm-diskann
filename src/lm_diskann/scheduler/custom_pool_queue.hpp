// ==========================================================================
/**
 * @file diskann/scheduler/custom_pool_queue.hpp
 * @brief Declares the PriorityTaskQueue for the custom thread pool,
 *        using standard library queues, mutex, and condition variable.
 */
// ==========================================================================
#pragma once

// #include "duckdb_proxies.hpp" // No longer needed for queue implementation
#include "task_types.hpp"
// #include "concurrentqueue.h" // No longer needed

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>

namespace diskann {
namespace scheduler {

/**
 * @brief Implements a two-priority task queue using std::queue, std::mutex, and std::condition_variable.
 * Used by the BackgroundWorkerPool.
 */
class PriorityTaskQueue {
	public:
	PriorityTaskQueue() = default;
	~PriorityTaskQueue() = default;

	// Disable copy/move
	PriorityTaskQueue(const PriorityTaskQueue &) = delete;
	PriorityTaskQueue &operator=(const PriorityTaskQueue &) = delete;
	PriorityTaskQueue(PriorityTaskQueue &&) = delete;
	PriorityTaskQueue &operator=(PriorityTaskQueue &&) = delete;

	/**
	 * @brief Enqueues a task. Signals waiting threads.
	 * @param task The CustomPoolTask to enqueue.
	 * @param priority The priority level.
	 */
	void push(scheduler::CustomPoolTask task, scheduler::TaskPriority priority);

	/**
	 * @brief Attempts to pop a high-priority task, waiting for a specified timeout.
	 * @param out_task Reference to store the popped task.
	 * @param timeout The maximum duration to wait.
	 * @return True if a task was popped, false if timed out.
	 */
	bool wait_pop_high_timed(scheduler::CustomPoolTask &out_task, std::chrono::microseconds timeout);

	/**
	 * @brief Attempts to pop a medium-priority task, waiting for a specified timeout.
	 * @param out_task Reference to store the popped task.
	 * @param timeout The maximum duration to wait.
	 * @return True if a task was popped, false if timed out.
	 */
	bool wait_pop_medium_timed(scheduler::CustomPoolTask &out_task, std::chrono::microseconds timeout);

	/**
	 * @brief Gets the exact count of tasks in the medium queue.
	 * @return Number of tasks.
	 */
	size_t medium_queue_size() const;

	/**
	 * @brief Gets the exact count of tasks in the high queue.
	 * @return Number of tasks.
	 */
	size_t high_queue_size() const;

	/**
	 * @brief Notifies all waiting threads, typically used during shutdown.
	 */
	void notify_all_waiters();

	private:
	/// @brief Mutex protecting access to the queues.
	/// Mutable to allow locking in const size() methods.
	mutable std::mutex queue_mutex_;

	/// @brief Queue for high priority tasks.
	std::queue<scheduler::CustomPoolTask> high_priority_queue_;

	/// @brief Queue for medium priority tasks.
	std::queue<scheduler::CustomPoolTask> medium_priority_queue_;

	/// @brief Condition variable for worker threads to wait on.
	std::condition_variable cv_;
};

} // namespace scheduler
} // namespace diskann