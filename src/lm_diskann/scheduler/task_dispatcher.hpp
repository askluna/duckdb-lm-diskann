// ==========================================================================
// File: diskann/common/task_dispatcher.hpp
// Description: Declares the DiskannTaskDispatcher (for DuckDB Scheduler).
// ==========================================================================
#pragma once

#include "duckdb_proxies.hpp" // Correct path
#include "task_types.hpp"     // Correct path

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread> // For std::jthread

namespace diskann {
namespace scheduler { // Changed from common

/**
 * @brief Manages logical priority queues and dispatches tasks to DuckDB's TaskScheduler.
 */
class DiskannTaskDispatcher {
	public:
	explicit DiskannTaskDispatcher(duckdb::DatabaseInstance &db_instance);
	~DiskannTaskDispatcher();

	// Disable copy/move
	DiskannTaskDispatcher(const DiskannTaskDispatcher &) = delete;
	DiskannTaskDispatcher &operator=(const DiskannTaskDispatcher &) = delete;
	DiskannTaskDispatcher(DiskannTaskDispatcher &&) = delete;
	DiskannTaskDispatcher &operator=(DiskannTaskDispatcher &&) = delete;

	void Start();
	void Stop();
	void SubmitTask(scheduler::DuckDbSchedulerTaskDefinition task_def); // common:: -> scheduler::

	private:
	static const int HIGH_PRIORITY_BURST_LIMIT = 8;
	void dispatcher_loop();

	duckdb::DatabaseInstance &db_;
	// In real build, use std::shared_ptr<duckdb::ProducerToken>
	// For proxy build, maybe just keep it void* or a dummy struct ptr
	std::shared_ptr<duckdb::ProducerToken> producer_token_;

	std::deque<scheduler::DuckDbSchedulerTaskDefinition> high_priority_queue_;   // common:: -> scheduler::
	std::deque<scheduler::DuckDbSchedulerTaskDefinition> medium_priority_queue_; // common:: -> scheduler::
	std::mutex queue_mutex_;
	std::condition_variable cv_;
	std::jthread dispatcher_thread_;
	std::atomic<bool> stop_requested_;
	int high_priority_tasks_processed_since_medium_;
};

} // namespace scheduler
} // namespace diskann