// ==========================================================================
/**
 * @file diskann/scheduler/executor_task.hpp
 * @brief Declares the generic ExecutorTask wrapper for tasks
 *        submitted to DuckDB's scheduler.
 */
// ==========================================================================
#pragma once

// #include "duckdb_proxies.hpp" // No longer needed for ExecutorTask base class
#include "task_types.hpp"

#include <duckdb/main/client_context.hpp>    // For ClientContext shared_ptr
#include <duckdb/parallel/executor_task.hpp> // Actual DuckDB base class
#include <memory>                            // For std::shared_ptr

namespace diskann {
namespace scheduler {

/**
 * @brief A concrete duckdb::Task that wraps a DuckDbSchedulerTaskDefinition.
 * This task is intended to be scheduled and executed by DuckDB's internal task scheduler.
 */
class GenericDiskannTask : public duckdb::Task {
	public:
	/**
	 * @brief Constructs a GenericDiskannTask.
	 * @param context The DuckDB client context for this task.
	 * @param definition The definition of the work to be performed.
	 */
	GenericDiskannTask(std::shared_ptr<duckdb::ClientContext> context,
	                   scheduler::DuckDbSchedulerTaskDefinition definition);

	/**
	 * @brief Executes the task.
	 * This method is called by the DuckDB task scheduler.
	 * @param mode The execution mode (e.g., regular or interrupt).
	 */
	duckdb::TaskExecutionResult Execute(duckdb::TaskExecutionMode mode) override;

	virtual ~GenericDiskannTask() = default;

	private:
	/// @brief Pointer to the client context for this task.
	std::shared_ptr<duckdb::ClientContext> context_ptr_;

	/// @brief The definition of the task to be executed.
	scheduler::DuckDbSchedulerTaskDefinition task_definition_;
};

} // namespace scheduler
} // namespace diskann