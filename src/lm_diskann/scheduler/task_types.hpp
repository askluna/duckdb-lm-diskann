// ==========================================================================
// File: diskann/common/task_types.hpp
// Description: Defines common types for task management.
// ==========================================================================
#pragma once

#include <chrono>
#include <functional>

// Forward declare duckdb::ClientContext if not included elsewhere yet
// In real build, get this from <duckdb/main/client_context.hpp>
namespace duckdb {
class ClientContext;
}

namespace diskann {
namespace scheduler {

// Enum to represent task priority levels (used by both mechanisms)
enum class TaskPriority { HIGH, MEDIUM };

// Task definition for the CUSTOM thread pool (Mechanism 1)
// Assumes tasks here generally do NOT need direct DuckDB ClientContext access.
struct CustomPoolTask {
	std::function<void()> work;
	// Add other relevant data if needed (e.g., task id, description)
};

// Task definition for the DUCKDB SCHEDULER dispatcher (Mechanism 2)
// Requires a work function that accepts a ClientContext.
struct DuckDbSchedulerTaskDefinition {
	TaskPriority priority = TaskPriority::MEDIUM;
	std::function<void(duckdb::ClientContext &)> work;
	std::chrono::steady_clock::time_point submission_time;
	// Add specific payload if needed (e.g., using std::variant or std::any)
};

} // namespace scheduler
} // namespace diskann