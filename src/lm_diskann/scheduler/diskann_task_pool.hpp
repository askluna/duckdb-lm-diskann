// ==========================================================================
// File: diskann/diskann_task_pool.hpp
// Description: Declares the main task pool manager for the DiskANN extension.
// ==========================================================================
#pragma once

#include "duckdb_proxies.hpp" // Was scheduler/duckdb_proxies.hpp
#include "task_types.hpp"     // Was scheduler/task_types.hpp

#include <cstddef> // for size_t
#include <memory>  // For std::unique_ptr

// Forward declare internal pool types within the correct namespace
namespace diskann {
namespace scheduler {
class BackgroundWorkerPool;  // Was in custom_pool
class DiskannTaskDispatcher; // Was in common
} // namespace scheduler
} // namespace diskann

namespace diskann {

/**
 * @brief Manages both the custom thread pool and the DuckDB scheduler dispatcher.
 * This is the main entry point for background task management in the extension.
 */
class DiskannTaskPool {
	public:
	DiskannTaskPool(duckdb::DatabaseInstance &db_instance, size_t num_custom_threads);
	~DiskannTaskPool();

	// Disable copy/move
	DiskannTaskPool(const DiskannTaskPool &) = delete;
	DiskannTaskPool &operator=(const DiskannTaskPool &) = delete;
	DiskannTaskPool(DiskannTaskPool &&) = delete;
	DiskannTaskPool &operator=(DiskannTaskPool &&) = delete;

	void Start();
	void Stop();
	void SubmitToCustomPool(scheduler::CustomPoolTask task, scheduler::TaskPriority priority);
	void SubmitToDuckDbScheduler(scheduler::DuckDbSchedulerTaskDefinition task_def);

	private:
	// Point to the classes now within the scheduler namespace
	std::unique_ptr<scheduler::BackgroundWorkerPool> custom_pool_manager_;
	std::unique_ptr<scheduler::DiskannTaskDispatcher> duckdb_dispatcher_;
};

} // namespace diskann