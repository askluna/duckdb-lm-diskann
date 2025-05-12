// ==========================================================================
// File: diskann/custom_pool/custom_pool_worker.cpp
// ==========================================================================
#include "custom_pool_worker.hpp" // Correct path

#include <exception> // For std::exception
#include <iostream>  // For demo output
#include <memory>    // For std::make_unique

namespace diskann {
namespace scheduler { // Changed from custom_pool

BackgroundWorkerPool::BackgroundWorkerPool(size_t num_threads)
    : stop_flag_(false),
      task_queue_(std::make_unique<PriorityTaskQueue>()) { // No namespace needed for PriorityTaskQueue as it's in the
	                                                         // same namespace now
	if (num_threads == 0) {
		// Try to get hardware concurrency, fallback to 1 if not available or 0
		num_threads = std::thread::hardware_concurrency();
		if (num_threads == 0) {
			num_threads = 1;
		}
	}

	std::cout << "[Custom Pool] Starting " << num_threads << " background worker threads (using BlockingQueues)..."
	          << "\n";

	worker_threads_.reserve(num_threads);
	for (size_t i = 0; i < num_threads; ++i) {
		worker_threads_.emplace_back([this, i](std::stop_token stoken) { worker_loop(i, stoken); });
	}
}

BackgroundWorkerPool::~BackgroundWorkerPool() {
	std::cout << "[Custom Pool] Destructor called, stopping workers..." << "\n";
	stop_workers(); // Ensure workers are signaled to stop and joined.
	                // jthreads will automatically join here if not already stopped and joined.
}

void BackgroundWorkerPool::submit_task(scheduler::CustomPoolTask task,
                                       scheduler::TaskPriority priority) { // common:: -> scheduler::
	if (stop_flag_.load(std::memory_order_acquire)) {
		std::cout << "[Custom Pool] Warning: Task submitted after stop requested." << "\n";
		return; // Or handle as an error
	}
	task_queue_->push(std::move(task), priority);
}

void BackgroundWorkerPool::stop_workers() {
	if (!stop_flag_.exchange(true, std::memory_order_acq_rel)) {
		std::cout << "[Custom Pool] Signaling workers to stop..." << "\n";
		// Request stop for all jthreads. They will check their stop_token.
		for (auto &jt : worker_threads_) {
			if (jt.joinable()) { // Check if it's still joinable (might have finished)
				jt.request_stop();
			}
		}
		// The WORKER_WAIT_TIMEOUT in wait_pop_..._timed ensures threads don't block indefinitely.
		// jthread destructor will join, or you can explicitly join here if needed after request_stop.
		// For this example, relying on jthread destructor to join is sufficient after stop_flag and request_stop.
	}
}

void BackgroundWorkerPool::worker_loop(size_t worker_id, std::stop_token stoken) {
	std::cout << "[Custom Pool] Worker thread " << worker_id << " started (Blocking Mode)." << "\n";
	scheduler::CustomPoolTask current_task; // common:: -> scheduler::
	int high_priority_burst_allowance = DEFAULT_BURST_ALLOWANCE;

	while (!stoken.stop_requested()) {
		bool task_popped = false;

		// Try HIGH priority first if allowance remains
		if (high_priority_burst_allowance > 0) {
			if (task_queue_->wait_pop_high_timed(current_task, WORKER_WAIT_TIMEOUT)) {
				task_popped = true;
				high_priority_burst_allowance--;
				// If burst allowance is used up AND there are medium tasks, next iteration might pick medium.
			}
		}

		// If no HIGH task, or allowance used, try MEDIUM
		if (!task_popped) {
			if (task_queue_->wait_pop_medium_timed(current_task, WORKER_WAIT_TIMEOUT)) {
				task_popped = true;
				high_priority_burst_allowance = DEFAULT_BURST_ALLOWANCE; // Reset burst after a medium task
			}
		}

		// If MEDIUM was also empty/timed_out, and we skipped HIGH due to burst, try HIGH again.
		// This ensures HIGH isn't starved if MEDIUM queue is persistently empty.
		if (!task_popped && high_priority_burst_allowance == 0 && task_queue_->high_queue_size() > 0) {
			if (task_queue_->wait_pop_high_timed(current_task, WORKER_WAIT_TIMEOUT)) {
				task_popped = true;
				// No need to reset burst here, as it was 0. Next HIGH task will decrement it from DEFAULT.
			} else {
				// If high priority task timed out, reset allowance to try medium next time fully.
				high_priority_burst_allowance = DEFAULT_BURST_ALLOWANCE;
			}
		}

		if (task_popped) {
			try {
				// std::cout << "[Custom Pool] Worker " << worker_id << " executing task." << "\n";
				current_task.work();
			} catch (const std::exception &e) {
				std::cerr << "[Custom Pool] Worker " << worker_id << " caught exception: " << e.what() << "\n";
			} catch (...) {
				std::cerr << "[Custom Pool] Worker " << worker_id << " caught unknown exception." << "\n";
			}
		}
		// If no task was popped (all timed out), the loop checks stoken.stop_requested().
		// The timed waits in wait_pop prevent busy-waiting.
	}
	std::cout << "[Custom Pool] Worker thread " << worker_id << " stopping." << "\n";
}

} // namespace scheduler
} // namespace diskann