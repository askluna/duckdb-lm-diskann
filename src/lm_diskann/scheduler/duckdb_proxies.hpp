// ==========================================================================
/**
 * @file diskann/scheduler/duckdb_proxies.hpp
 * @brief Conceptual header for DuckDB's internal types and aliases.
 *        In a real build, these would come from actual DuckDB headers.
 */
// ==========================================================================
#pragma once

#include <cstdint> // For uint8_t

// --- Forward Declarations for DuckDB Types (Conceptual) ---
// These help reduce include dependencies in headers.
// Full definitions will be needed in .cpp files.
namespace duckdb {

/// @brief Forward declaration from <duckdb/main/client_context.hpp>
class ClientContext;

/// @brief Forward declaration from <duckdb/main/database_instance.hpp>
class DatabaseInstance;

// class ExecutorTask; // Removed: Now included directly in executor_task.hpp

/// @brief Forward declaration from <duckdb/parallel/producer_token.hpp>
class ProducerToken;

/// @brief Forward declaration from <duckdb/parallel/task_scheduler.hpp>
class TaskScheduler;

/// @brief Forward declaration matching DuckDB's underlying type.
/// Actual values might differ; only type consistency needed for proxy.
/// From <duckdb/parallel/task.hpp> (or similar)
enum class TaskExecutionMode : uint8_t;

} // namespace duckdb
// --- End Forward Declarations ---