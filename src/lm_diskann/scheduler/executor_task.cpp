// ==========================================================================
// File: diskann/common/executor_task.cpp
// ==========================================================================
#include "executor_task.hpp" // Correct path

#include <exception>
#include <iostream>

// --- DuckDB Headers (Actual includes needed for implementation) ---
#include <duckdb/common/exception.hpp> // For duckdb::ErrorData
#include <duckdb/main/client_context.hpp>
#include <duckdb/parallel/executor_task.hpp>
// --- End DuckDB Headers ---

namespace diskann {
namespace scheduler { // Changed from common

// Constructor needs to call the base class constructor.
// The base ExecutorTask constructor expects a reference to the Executor, which it gets from the ClientContext.
GenericDiskannTask::GenericDiskannTask(std::shared_ptr<duckdb::ClientContext> context,
                                       scheduler::DuckDbSchedulerTaskDefinition definition)
    : duckdb::ExecutorTask(*context), // Pass the context to the base ExecutorTask constructor
      context_ptr_(context),          // Store the shared_ptr for our use
      task_definition_(std::move(definition)) {
}

void GenericDiskannTask::Execute(duckdb::TaskExecutionMode mode) {
	// DuckDB tasks might be asked to interrupt. Handle this appropriately.
	// Refer to DuckDB's Task documentation for exact handling (e.g., checking context->interrupted)
	if (mode == duckdb::TaskExecutionMode::INTERRUPT) {
		std::cout << "[GenericDiskannTask] Received INTERRUPT mode, skipping execution." << std::endl;
		// Perform any necessary cleanup or state handling for interruption if needed.
		return;
	}

	if (!task_definition_.work || !context_ptr_) {
		std::cerr << "[GenericDiskannTask] Error: Null work function or context. Cannot execute." << std::endl;
		// Depending on DuckDB's task error handling, you might need to signal an error state.
		return;
	}

	try {
		// std::cout << "[GenericDiskannTask] Executing task..." << std::endl;
		task_definition_.work(*context_ptr_); // Execute the stored work function
		                                      // std::cout << "[GenericDiskannTask] Task completed." << std::endl;
	} catch (const std::exception &e) {
		std::cerr << "[GenericDiskannTask] Exception during execution: " << e.what() << std::endl;
		// Propagate the error to the DuckDB executor if possible/necessary.
		// This might involve calling something like executor.PushError(...) if you have access
		// to the Executor instance (ExecutorTask base class provides `executor` member).
		try {
			executor.PushError(duckdb::ErrorData(e));
		} catch (...) { /* Error pushing error? */
		}
	} catch (...) {
		std::cerr << "[GenericDiskannTask] Unknown exception during execution." << std::endl;
		try {
			executor.PushError(duckdb::ErrorData("Unknown exception in GenericDiskannTask"));
		} catch (...) { /* Error pushing error? */
		}
	}
	// DuckDB's ExecutorTask::Execute typically returns a TaskExecutionResult.
	// Since this implementation doesn't have complex state (like blocking),
	// we assume it finishes in one go when called in non-interrupt mode.
	// The base class Execute might handle returning TASK_FINISHED/TASK_ERROR.
	// If specific return values are needed, adjust this function signature and logic.
}

} // namespace scheduler
} // namespace diskann