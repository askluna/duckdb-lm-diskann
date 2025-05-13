// ==========================================================================
// File: diskann/scheduler/executor_task.cpp
// ==========================================================================
#include "executor_task.hpp" // Correct path

#include <duckdb/common/exception.hpp>    // For duckdb::ErrorData
#include <duckdb/execution/executor.hpp>  // For Executor::Get
#include <duckdb/main/client_context.hpp> // Needed for IsInterrupted check?
#include <duckdb/parallel/task.hpp>       // For TaskExecutionResult
#include <exception>
#include <iostream>

// --- DuckDB Headers (Actual includes needed for implementation) ---
// #include <duckdb/common/exception.hpp> // Now included via duckdb::ErrorData
// #include <duckdb/main/client_context.hpp> // Now included via executor_task.hpp
// #include <duckdb/parallel/executor_task.hpp> // Now included via executor_task.hpp
// --- End DuckDB Headers ---

namespace diskann {
namespace scheduler { // Changed from common

// Constructor: Initializes the base ExecutorTask with the context
GenericDiskannTask::GenericDiskannTask(std::shared_ptr<duckdb::ClientContext> context,
                                       scheduler::DuckDbSchedulerTaskDefinition definition)
    : context_ptr_(context), task_definition_(std::move(definition)) {
}

// Executes the wrapped task logic
duckdb::TaskExecutionResult GenericDiskannTask::Execute(duckdb::TaskExecutionMode mode) {
	try {
		// Handle interruption signal from DuckDB scheduler
		if (context_ptr_->interrupted) { // Changed interruption check
			return duckdb::TaskExecutionResult::TASK_FINISHED;
		}

		// Execute the actual task function
		task_definition_.work(*context_ptr_); // Changed to 'work' and dereference context_ptr_

		return duckdb::TaskExecutionResult::TASK_FINISHED;
	} catch (const duckdb::InterruptException &e) {
		std::cerr << "[GenericDiskannTask] Interrupted: " << e.what() << "\n";
		// Interruption is a normal way to stop tasks, usually not an error to be pushed.
		// Re-throwing or ensuring the scheduler handles it is typical.
		// For now, let's assume TASK_FINISHED is appropriate as the task stops.
		return duckdb::TaskExecutionResult::TASK_FINISHED;
	} catch (const std::exception &e) {
		std::cerr << "[GenericDiskannTask] Exception during execution: " << e.what() << "\n";
		try {
			duckdb::Executor::Get(*context_ptr_).PushError(duckdb::ErrorData(e)); // Changed error handling
		} catch (...) {                                                         /* Ignored */
		}
		return duckdb::TaskExecutionResult::TASK_ERROR;
	} catch (...) {
		std::cerr << "[GenericDiskannTask] Unknown exception during execution." << "\n";
		try {
			duckdb::Executor::Get(*context_ptr_)
			    .PushError(duckdb::ErrorData("Unknown exception in GenericDiskannTask")); // Changed error handling
		} catch (...) {                                                                 /* Ignored */
		}
		return duckdb::TaskExecutionResult::TASK_ERROR;
	}
}

} // namespace scheduler
} // namespace diskann