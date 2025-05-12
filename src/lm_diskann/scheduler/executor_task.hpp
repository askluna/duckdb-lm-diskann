// ==========================================================================
// File: diskann/common/executor_task.hpp
// Description: Declares the generic ExecutorTask wrapper for tasks
//              submitted to DuckDB's scheduler.
// ==========================================================================
#pragma once

#include "duckdb_proxies.hpp" // Correct path
#include "task_types.hpp"     // Correct path

#include <memory> // For std::shared_ptr

namespace diskann {
namespace scheduler { // Changed from common

/**
 * @brief A concrete duckdb::ExecutorTask that wraps a DuckDbSchedulerTaskDefinition.
 */
class GenericDiskannTask : public duckdb::ExecutorTask {
	public:
	GenericDiskannTask(std::shared_ptr<duckdb::ClientContext> context,
	                   scheduler::DuckDbSchedulerTaskDefinition definition); // common:: -> scheduler::

	// Execute method signature must match DuckDB's ExecutorTask base class.
	// Assuming it takes TaskExecutionMode (adjust if needed based on actual DuckDB version)
	void Execute(duckdb::TaskExecutionMode mode) override;

	private:
	std::shared_ptr<duckdb::ClientContext> context_ptr_;
	scheduler::DuckDbSchedulerTaskDefinition task_definition_; // common:: -> scheduler::
};

} // namespace scheduler
} // namespace diskann