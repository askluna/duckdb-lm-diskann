// ==========================================================================
// File: diskann/common/duckdb_proxies.hpp
// Description: Conceptual header for DuckDB's internal types and aliases.
//              In a real build, these would come from actual DuckDB headers.
// ==========================================================================
#pragma once

// --- Forward Declarations for DuckDB Types (Conceptual) ---
// These help reduce include dependencies in headers.
// Full definitions will be needed in .cpp files.
namespace duckdb {
class ClientContext;    // From <duckdb/main/client_context.hpp>
class DatabaseInstance; // From <duckdb/main/database_instance.hpp>
class ExecutorTask;     // From <duckdb/parallel/executor_task.hpp>
class ProducerToken;    // From <duckdb/parallel/producer_token.hpp> (or task_scheduler.hpp)
class TaskScheduler;    // From <duckdb/parallel/task_scheduler.hpp>
enum class TaskExecutionMode { REGULAR, INTERRUPT }; // From <duckdb/parallel/task.hpp> (or similar)
} // namespace duckdb
// --- End Forward Declarations ---