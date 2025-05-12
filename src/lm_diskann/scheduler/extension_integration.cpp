// ==========================================================================
// File: extension_integration.cpp (Conceptual)
// Description: Shows how DiskannTaskPool integrates with DuckDB.
// NOTE: Contains dummy DuckDB classes for standalone testing.
//       Replace includes and dummy classes with real DuckDB headers for integration.
// ==========================================================================
#include "diskann_task_pool.hpp" // Correct, as it's in the same (scheduler) directory now

#include <atomic>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

// --- Dummy DuckDB Classes for Standalone Compilation (if not using real headers) ---
// --- Replace these with actual DuckDB includes in a real build ---
#ifndef DUCKDB_TYPES_DEFINED_STANDALONE // Guard to prevent redefinition
#define DUCKDB_TYPES_DEFINED_STANDALONE

namespace duckdb {

// Forward Declarations
class TaskScheduler;
class DatabaseInstance;
class ClientContext;
class Task;
class ExecutorTask;
class ProducerToken;
struct ErrorData {
	ErrorData(const std::exception &e) {
	}
	ErrorData(const std::string &s) {
	}
}; // Dummy
enum class TaskExecutionMode { REGULAR, INTERRUPT };

// --- Dummy Task ---
class Task : public std::enable_shared_from_this<Task> {
	public:
	virtual ~Task() = default;
	virtual TaskExecutionResult Execute(TaskExecutionMode mode) = 0;
	ProducerToken *token = nullptr; // Dummy
};
enum class TaskExecutionResult { TASK_FINISHED, TASK_NOT_FINISHED, TASK_ERROR, TASK_BLOCKED };

// --- Dummy Executor ---
class Executor { // Needed by ExecutorTask constructor
	public:
	Executor(ClientContext &context) {};
	void PushError(ErrorData e) {
		std::cerr << "Dummy Executor PushError called." << std::endl;
	}
};

// --- Dummy ExecutorTask ---
class ExecutorTask : public Task {
	public:
	Executor &executor; // Public for simplicity in dummy
	// Dummy constructor - takes context, creates dummy executor
	ExecutorTask(ClientContext &context) : executor_(*(new Executor(context))) {
	}
	virtual ~ExecutorTask() {
		delete &executor_;
	} // Clean up dummy executor
	  // Pure virtual Execute from base Task class still needs implementation in derived classes
	private:
	Executor &executor_;
};

// --- Dummy ProducerToken ---
class ProducerToken {
	public:
	ProducerToken(TaskScheduler &scheduler) {};
};

// --- Dummy TaskScheduler ---
class TaskScheduler {
	private:
	std::queue<std::shared_ptr<Task>> task_queue_;
	std::mutex queue_mutex_;
	std::atomic<bool> has_tasks_ {false};
	std::condition_variable cv_;

	public:
	TaskScheduler() = default; // Add default constructor
	static TaskScheduler &GetScheduler(DatabaseInstance &db) {
		// Simple static instance for dummy
		static TaskScheduler instance;
		return instance;
	}
	std::shared_ptr<ProducerToken> CreateProducer() {
		return std::make_shared<ProducerToken>(*this);
	}
	void ScheduleTask(ProducerToken &token, std::shared_ptr<Task> task) {
		std::lock_guard<std::mutex> lock(queue_mutex_);
		task_queue_.push(std::move(task));
		has_tasks_.store(true, std::memory_order_release);
		cv_.notify_one();
		// Simulate immediate execution for testing
		ExecuteOneTask();
	}
	void ExecuteOneTask() { // Helper for dummy execution
		std::shared_ptr<Task> task_to_run;
		{
			std::lock_guard<std::mutex> lock(queue_mutex_);
			if (!task_queue_.empty()) {
				task_to_run = task_queue_.front();
				task_queue_.pop();
			} else {
				has_tasks_.store(false, std::memory_order_release);
			}
		}
		if (task_to_run) {
			std::cout << "[DuckDB Scheduler (Dummy)] Executing task..." << std::endl;
			try {
				task_to_run->Execute(TaskExecutionMode::REGULAR);
			} catch (const std::exception &e) {
				std::cerr << "[DuckDB Scheduler (Dummy)] Task execution failed: " << e.what() << std::endl;
			} catch (...) {
				std::cerr << "[DuckDB Scheduler (Dummy)] Task execution failed with unknown error." << std::endl;
			}
		} else {
			// std::cout << "[DuckDB Scheduler (Dummy)] No task to execute." << std::endl;
		}
	}
};

// --- Dummy DatabaseInstance ---
class DatabaseInstance {
	public:
	// Provide access to the static scheduler instance
	TaskScheduler &GetScheduler() {
		return TaskScheduler::GetScheduler(*this);
	}
};

// --- Dummy ClientContext ---
class ClientContext {
	public:
	// Store reference to DatabaseInstance
	DatabaseInstance &db_instance_;
	// Constructor takes DatabaseInstance
	ClientContext(DatabaseInstance &db) : db_instance_(db) {
	}
	DatabaseInstance &GetDatabase() {
		return db_instance_;
	} // Method to access DB
};

} // namespace duckdb
#endif // DUCKDB_TYPES_DEFINED_STANDALONE
// --- End Dummy DuckDB Classes ---

// Global pointer to the main task pool manager
std::unique_ptr<diskann::DiskannTaskPool> g_diskann_task_pool;

// --- Dummy Extension Init/Shutdown ---

// Wrap these in extern "C" if they are meant to be loaded dynamically by DuckDB
// extern "C" /*DUCKDB_EXTENSION_API*/ void diskann_extension_init(duckdb::DatabaseInstance &db) {
void diskann_extension_init(duckdb::DatabaseInstance &db) {
	std::cout << "========== Initializing DiskANN Extension (Unified Pool) ==========" << std::endl;
	try {
		size_t num_custom_threads = 2; // Example: Use 2 custom threads
		g_diskann_task_pool = std::make_unique<diskann::DiskannTaskPool>(db, num_custom_threads);
		g_diskann_task_pool->Start();

		// Example task submissions during init
		g_diskann_task_pool->SubmitToCustomPool({[]() {
			                                        std::cout << "[Custom Pool] Initial task from TaskPool executed.\n";
		                                        }},
		                                        diskann::scheduler::TaskPriority::MEDIUM);
		g_diskann_task_pool->SubmitToDuckDbScheduler(
		    {diskann::scheduler::TaskPriority::MEDIUM, [](duckdb::ClientContext &ctx) {
			     std::cout << "[DuckDB Task] Initial task from TaskPool executed via dispatcher.\n";
		     }});
		std::cout << "DiskANN Task Pool initialized and started." << std::endl;
	} catch (const std::exception &e) {
		std::cerr << "FATAL ERROR during DiskANNTaskPool initialization: " << e.what() << std::endl;
		g_diskann_task_pool.reset(); // Clean up partial initialization
		                             // Depending on context, might need to signal failure differently
	} catch (...) {
		std::cerr << "FATAL UNKNOWN ERROR during DiskANNTaskPool initialization." << std::endl;
		g_diskann_task_pool.reset();
	}
	std::cout << "========== DiskANN Extension Initialized ==========" << std::endl;
}

// extern "C" /*DUCKDB_EXTENSION_API*/ void diskann_extension_shutdown() {
void diskann_extension_shutdown() {
	std::cout << "========== Shutting down DiskANN Extension ==========" << std::endl;
	if (g_diskann_task_pool) {
		try {
			g_diskann_task_pool->Stop();
		} catch (const std::exception &e) {
			std::cerr << "Error during DiskANNTaskPool Stop(): " << e.what() << std::endl;
		} catch (...) {
			std::cerr << "Unknown error during DiskANNTaskPool Stop()." << std::endl;
		}
		g_diskann_task_pool.reset(); // Release the unique_ptr
	}
	std::cout << "DiskANN Task Pool shut down." << std::endl;
	std::cout << "========== DiskANN Extension Shutdown Complete ==========" << std::endl;
}

// --- Main Function (for standalone testing) ---
int main() {
	std::cout << "--- Standalone Test (Unified DiskannTaskPool with Blocking Custom Pool) ---" << std::endl;
	duckdb::DatabaseInstance dummy_db; // Create dummy DB instance

	try {
		diskann_extension_init(dummy_db);

		if (!g_diskann_task_pool) {
			std::cerr << "Test failed: g_diskann_task_pool not initialized after init call." << std::endl;
			return 1;
		}

		std::cout << "\n--- Submitting test tasks via DiskannTaskPool --- " << std::endl;
		// Submit a mix of tasks
		for (int i = 0; i < 5; ++i) {
			g_diskann_task_pool->SubmitToCustomPool({[i]() {
				                                        std::cout << "[Custom Pool] Task HIGH " << i << " done.\n";
				                                        std::this_thread::sleep_for(std::chrono::milliseconds(25));
			                                        }},
			                                        diskann::scheduler::TaskPriority::HIGH);
			if (i % 2 == 0) {
				g_diskann_task_pool->SubmitToCustomPool({[i]() {
					                                        std::cout << "[Custom Pool] Task MEDIUM " << i / 2 << " done.\n";
					                                        std::this_thread::sleep_for(std::chrono::milliseconds(55));
				                                        }},
				                                        diskann::scheduler::TaskPriority::MEDIUM);
			}
			// Submit to DuckDB Scheduler
			g_diskann_task_pool->SubmitToDuckDbScheduler(
			    {diskann::scheduler::TaskPriority::HIGH, [i](duckdb::ClientContext &ctx) {
				     std::cout << "[DuckDB Task] Task HIGH " << i << " done.\n";
				     std::this_thread::sleep_for(std::chrono::milliseconds(35));
			     }});
			if (i % 2 == 0) {
				g_diskann_task_pool->SubmitToDuckDbScheduler(
				    {diskann::scheduler::TaskPriority::MEDIUM, [i](duckdb::ClientContext &ctx) {
					     std::cout << "[DuckDB Task] Task MEDIUM " << i / 2 << " done.\n";
					     std::this_thread::sleep_for(std::chrono::milliseconds(65));
				     }});
			}
		}

		std::cout << "\nTasks submitted. Waiting for processing (adjust time if needed)..." << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(8)); // Wait long enough for tasks to likely finish

		std::cout << "\n--- Initiating shutdown --- " << std::endl;
		diskann_extension_shutdown();

	} catch (const std::exception &e) {
		std::cerr << "Error in main test execution: " << e.what() << std::endl;
		// Ensure shutdown is called even if errors occur mid-test
		if (g_diskann_task_pool) {
			try {
				diskann_extension_shutdown();
			} catch (...) {
			}
		}
		return 1;
	} catch (...) {
		std::cerr << "Unknown error in main test execution." << std::endl;
		if (g_diskann_task_pool) {
			try {
				diskann_extension_shutdown();
			} catch (...) {
			}
		}
		return 1;
	}

	std::cout << "--- Test Complete --- " << std::endl;
	return 0;
}