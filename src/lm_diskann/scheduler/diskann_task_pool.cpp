// ==========================================================================
// File: diskann/diskann_task_pool.cpp
// Description: Implements the main task pool manager.
// ==========================================================================
#include "diskann_task_pool.hpp" // Correct path

// Include the implementation details from the scheduler namespace
#include "custom_pool_worker.hpp" // Was scheduler/
#include "task_dispatcher.hpp"    // Was scheduler/

#include <iostream>
#include <memory> // Included for make_unique

namespace diskann {

DiskannTaskPool::DiskannTaskPool(duckdb::DatabaseInstance &db_instance, size_t num_custom_threads) {
	std::cout << "[DiskannTaskPool] Initializing..." << std::endl;
	try {
		custom_pool_manager_ = std::make_unique<scheduler::BackgroundWorkerPool>(num_custom_threads);
		duckdb_dispatcher_ = std::make_unique<scheduler::DiskannTaskDispatcher>(db_instance);
	} catch (const std::exception &e) {
		std::cerr << "[DiskannTaskPool] FATAL: Error during construction: " << e.what() << std::endl;
		custom_pool_manager_.reset();
		duckdb_dispatcher_.reset();
		throw;
	} catch (...) {
		std::cerr << "[DiskannTaskPool] FATAL: Unknown error during construction." << std::endl;
		custom_pool_manager_.reset();
		duckdb_dispatcher_.reset();
		throw;
	}
	std::cout << "[DiskannTaskPool] Initialized." << std::endl;
}

DiskannTaskPool::~DiskannTaskPool() {
	std::cout << "[DiskannTaskPool] Destructor called, stopping services..." << std::endl;
	Stop();
}

void DiskannTaskPool::Start() {
	std::cout << "[DiskannTaskPool] Starting services..." << std::endl;
	if (duckdb_dispatcher_) {
		try {
			duckdb_dispatcher_->Start();
		} catch (const std::exception &e) {
			std::cerr << "[DiskannTaskPool] Error starting DuckDB dispatcher: " << e.what() << std::endl;
			Stop();
			throw;
		} catch (...) {
			std::cerr << "[DiskannTaskPool] Unknown error starting DuckDB dispatcher." << std::endl;
			Stop();
			throw;
		}
	} else {
		std::cerr << "[DiskannTaskPool] Cannot Start: DuckDB dispatcher not initialized." << std::endl;
	}
	std::cout << "[DiskannTaskPool] Services started." << std::endl;
}

void DiskannTaskPool::Stop() {
	std::cout << "[DiskannTaskPool] Stopping services..." << std::endl;
	if (duckdb_dispatcher_) {
		try {
			duckdb_dispatcher_->Stop();
		} catch (const std::exception &e) {
			std::cerr << "[DiskannTaskPool] Error stopping DuckDB dispatcher: " << e.what() << std::endl;
		} catch (...) {
			std::cerr << "[DiskannTaskPool] Unknown error stopping DuckDB dispatcher." << std::endl;
		}
	}
	if (custom_pool_manager_) {
		try {
			custom_pool_manager_->stop_workers();
		} catch (const std::exception &e) {
			std::cerr << "[DiskannTaskPool] Error stopping custom pool workers: " << e.what() << std::endl;
		} catch (...) {
			std::cerr << "[DiskannTaskPool] Unknown error stopping custom pool workers." << std::endl;
		}
	}
	std::cout << "[DiskannTaskPool] Services stopped." << std::endl;
}

void DiskannTaskPool::SubmitToCustomPool(scheduler::CustomPoolTask task, scheduler::TaskPriority priority) {
	if (custom_pool_manager_) {
		try {
			custom_pool_manager_->submit_task(std::move(task), priority);
		} catch (const std::exception &e) {
			std::cerr << "[DiskannTaskPool] Error submitting to custom pool: " << e.what() << std::endl;
		}
	} else {
		std::cerr << "[DiskannTaskPool] Error: Custom pool not initialized. Cannot submit task." << std::endl;
	}
}

void DiskannTaskPool::SubmitToDuckDbScheduler(scheduler::DuckDbSchedulerTaskDefinition task_def) {
	if (duckdb_dispatcher_) {
		try {
			duckdb_dispatcher_->SubmitTask(std::move(task_def));
		} catch (const std::exception &e) {
			std::cerr << "[DiskannTaskPool] Error submitting to DuckDB dispatcher: " << e.what() << std::endl;
		}
	} else {
		std::cerr << "[DiskannTaskPool] Error: DuckDB dispatcher not initialized. Cannot submit task." << std::endl;
	}
}

} // namespace diskann