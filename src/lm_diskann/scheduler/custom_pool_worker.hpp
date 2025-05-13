// ==========================================================================
// File: diskann/custom_pool/custom_pool_worker.hpp
// Description: Declares the BackgroundWorkerPool for the custom thread pool.
// ==========================================================================
#pragma once

#include "custom_pool_queue.hpp" // Correct path
#include "task_types.hpp"        // Was ../common/task_types.hpp

#include <atomic>
#include <chrono> // For timeout values
#include <memory>
#include <thread> // For std::jthread, std::stop_token
#include <vector>

namespace diskann {
namespace scheduler { // Changed from custom_pool

// Forward declare PriorityTaskQueue as it's used in unique_ptr
class PriorityTaskQueue;

/**
 * @brief Manages a pool of worker threads for tasks not requiring DuckDB context.
 * Uses BlockingConcurrentQueue for efficient worker idling.
 */
class BackgroundWorkerPool {
	public:
	explicit BackgroundWorkerPool(size_t num_threads);
	~BackgroundWorkerPool();

	// Disable copy/move
	BackgroundWorkerPool(const BackgroundWorkerPool &) = delete;
	BackgroundWorkerPool &operator=(const BackgroundWorkerPool &) = delete;
	BackgroundWorkerPool(BackgroundWorkerPool &&) = delete;
	BackgroundWorkerPool &operator=(BackgroundWorkerPool &&) = delete;

	void submit_task(scheduler::CustomPoolTask task, scheduler::TaskPriority priority); // common:: -> scheduler::
	void stop_workers();

	private:
	static constexpr int DEFAULT_BURST_ALLOWANCE = 8;
	// Timeout for waiting on queues, can be adjusted.
	// A small timeout allows fairness logic to interleave checks more frequently.
	static constexpr std::chrono::microseconds WORKER_WAIT_TIMEOUT = std::chrono::milliseconds(10);

	void worker_loop(size_t worker_id, std::stop_token stoken);

	std::atomic<bool> stop_flag_;
	std::vector<std::jthread> worker_threads_;
	std::unique_ptr<PriorityTaskQueue> task_queue_; // Type is now local to scheduler namespace
};

} // namespace scheduler
} // namespace diskann