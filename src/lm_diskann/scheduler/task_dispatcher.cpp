// ==========================================================================
// File: diskann/common/task_dispatcher.cpp
// ==========================================================================
#include "task_dispatcher.hpp" // Correct path

#include "executor_task.hpp" // Correct path

#include <exception>
#include <iostream>
#include <memory> // For std::make_unique, std::make_shared

// --- DuckDB Headers (Actual includes needed for implementation) ---
// Replace with actual headers
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/database_instance.hpp>
#include <duckdb/parallel/executor_task.hpp> // Include base class definition
#include <duckdb/parallel/producer_token.hpp>
#include <duckdb/parallel/task_scheduler.hpp>
// --- End DuckDB Headers ---

namespace diskann {
namespace scheduler { // Changed from common

DiskannTaskDispatcher::DiskannTaskDispatcher(duckdb::DatabaseInstance &db_instance)
    : db_(db_instance), stop_requested_(false), high_priority_tasks_processed_since_medium_(0) {
	try {
		auto &scheduler = duckdb::TaskScheduler::GetScheduler(db_);
		producer_token_ = scheduler.CreateProducer();
		std::cout << "[Dispatcher] Producer token created." << std::endl;
	} catch (const std::exception &e) {
		std::cerr << "[Dispatcher] FATAL: Failed to initialize TaskScheduler or ProducerToken: " << e.what() << std::endl;
		throw;
	} catch (...) {
		std::cerr << "[Dispatcher] FATAL: Unknown error during TaskScheduler/ProducerToken initialization." << std::endl;
		throw;
	}
}

DiskannTaskDispatcher::~DiskannTaskDispatcher() {
	Stop();
}

void DiskannTaskDispatcher::Start() {
	if (dispatcher_thread_.joinable()) {
		std::cout << "[Dispatcher] Thread already started." << std::endl;
		return;
	}
	std::cout << "[Dispatcher] Starting dispatcher thread..." << std::endl;
	stop_requested_.store(false, std::memory_order_release);
	dispatcher_thread_ = std::jthread([this]() { dispatcher_loop(); });
}

void DiskannTaskDispatcher::Stop() {
	bool already_stopped = stop_requested_.exchange(true, std::memory_order_acq_rel);
	if (!already_stopped && dispatcher_thread_.joinable()) {
		std::cout << "[Dispatcher] Stopping dispatcher thread..." << std::endl;
		cv_.notify_one();
	}
}

void DiskannTaskDispatcher::SubmitTask(scheduler::DuckDbSchedulerTaskDefinition task_def) { // common:: -> scheduler::
	if (stop_requested_.load(std::memory_order_acquire)) {
		std::cout << "[Dispatcher] Warning: Task submitted after stop requested." << std::endl;
		return;
	}
	task_def.submission_time = std::chrono::steady_clock::now();
	{
		std::lock_guard<std::mutex> lock(queue_mutex_);
		if (task_def.priority == scheduler::TaskPriority::HIGH) { // common:: -> scheduler::
			high_priority_queue_.push_back(std::move(task_def));
		} else {
			medium_priority_queue_.push_back(std::move(task_def));
		}
	}
	cv_.notify_one();
}

void DiskannTaskDispatcher::dispatcher_loop() {
	std::cout << "[Dispatcher] Dispatcher thread running." << std::endl;
	duckdb::TaskScheduler *scheduler_ptr = nullptr;
	try {
		scheduler_ptr = &duckdb::TaskScheduler::GetScheduler(db_);
	} catch (const std::exception &e) {
		std::cerr << "[Dispatcher Loop] FATAL: Cannot get TaskScheduler: " << e.what() << std::endl;
		stop_requested_.store(true, std::memory_order_release);
		return;
	}
	if (!scheduler_ptr) {
		std::cerr << "[Dispatcher Loop] FATAL: TaskScheduler is null ptr." << std::endl;
		stop_requested_.store(true, std::memory_order_release);
		return;
	}
	auto &scheduler = *scheduler_ptr;

	if (!producer_token_) {
		std::cerr << "[Dispatcher Loop] FATAL: ProducerToken is null. Dispatcher cannot operate." << std::endl;
		stop_requested_.store(true, std::memory_order_release);
		return;
	}

	while (!stop_requested_.load(std::memory_order_acquire)) {
		scheduler::DuckDbSchedulerTaskDefinition selected_task_def; // common:: -> scheduler::
		bool task_selected = false;

		{
			std::unique_lock<std::mutex> lock(queue_mutex_);
			cv_.wait(lock, [this] {
				return stop_requested_.load(std::memory_order_relaxed) || !high_priority_queue_.empty() ||
				       !medium_priority_queue_.empty();
			});

			if (stop_requested_.load(std::memory_order_relaxed))
				break;

			if (!high_priority_queue_.empty() &&
			    (high_priority_tasks_processed_since_medium_ < HIGH_PRIORITY_BURST_LIMIT || medium_priority_queue_.empty())) {
				selected_task_def = std::move(high_priority_queue_.front());
				high_priority_queue_.pop_front();
				high_priority_tasks_processed_since_medium_++;
				task_selected = true;
			} else if (!medium_priority_queue_.empty()) {
				selected_task_def = std::move(medium_priority_queue_.front());
				medium_priority_queue_.pop_front();
				high_priority_tasks_processed_since_medium_ = 0;
				task_selected = true;
			} else if (!high_priority_queue_.empty()) {
				selected_task_def = std::move(high_priority_queue_.front());
				high_priority_queue_.pop_front();
				task_selected = true;
			}
		}

		if (task_selected) {
			try {
				auto task_context = std::make_shared<duckdb::ClientContext>(db_);
				auto executor_task = std::make_shared<scheduler::GenericDiskannTask>(
				    task_context, std::move(selected_task_def)); // common:: -> scheduler::
				scheduler.ScheduleTask(*producer_token_, std::move(executor_task));
			} catch (const std::exception &e) {
				std::cerr << "[Dispatcher Loop] Error creating/scheduling task: " << e.what() << std::endl;
			} catch (...) {
				std::cerr << "[Dispatcher Loop] Unknown error creating/scheduling task." << std::endl;
			}
		}
	}
	std::cout << "[Dispatcher] Dispatcher thread exiting." << std::endl;
}

} // namespace scheduler
} // namespace diskann