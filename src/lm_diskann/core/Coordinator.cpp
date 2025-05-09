#include "Coordinator.hpp"

#include "../common/ann.hpp"
#include "../common/duckdb_types.hpp"
#include "../store/IShadowStorageService.hpp"
#include "IGraphManager.hpp"
#include "ISearcher.hpp"
#include "IStorageManager.hpp"
#include "index_config.hpp"

#include <vector>

// For now, include iostream for placeholder messages if needed.
// Replace with proper logging mechanism later.
#include <iostream>

// You will need to include actual headers for these dependencies
// once they are defined/refactored.
// #include "diskann/core/IStorageManager.hpp"
// #include "diskann/core/IGraphManager.hpp"
// #include "diskann/core/ISearcher.hpp"
// #include "diskann/core/IndexConfig.hpp"
// #include "diskann/store/IShadowStorageService.hpp"
// #include "duckdb.hpp" // For ClientContext if used

namespace diskann {
namespace core {

Coordinator::Coordinator(std::unique_ptr<IStorageManager> storage_manager, std::unique_ptr<IGraphManager> graph_manager,
                         std::unique_ptr<ISearcher> searcher,
                         std::unique_ptr<store::IShadowStorageService> shadow_storage_service,
                         const LmDiskannConfig &config
                         // duckdb::ClientContext& context
                         )
    : storage_manager_(std::move(storage_manager)), graph_manager_(std::move(graph_manager)),
      searcher_(std::move(searcher)), shadow_storage_service_(std::move(shadow_storage_service)), config_(config)
// context_(context)
{
	graph_entry_point_ptr_ = common::IndexPointer(); // Initialize to invalid state
	delete_queue_head_ptr_ = common::IndexPointer(); // Initialize to invalid state

	std::cout << "Coordinator: Initialized." << std::endl;
}

Coordinator::~Coordinator() {
	std::cout << "Coordinator: Destroyed." << std::endl;
}

void Coordinator::BuildIndex(const std::string &data_path /*, other params */) {
	std::cout << "Coordinator: BuildIndex called with data_path: " << data_path << std::endl;
	// 1. Initialize IndexConfig if not already done.
	// 2. Use StorageManager to prepare storage for the new index.
	// 3. Use GraphManager to build the graph from data (potentially read via
	// StorageManager or directly).
	//    - This might involve calls to distance functions, quantization, etc.
	// 4. Save the initial graph and metadata using StorageManager.
	// 5. Manage shadow store interactions via IShadowStorageService for
	// transactional integrity if needed during build.
	index_path_ = ""; // Potentially set based on config or derived from data_path
	index_loaded_ = true;
}

void Coordinator::Search(const float *query_vector, common::idx_t k_neighbors,
                         std::vector<common::row_t> &result_row_ids, common::idx_t search_list_size) {
	// std::cout << "Coordinator: Search called." << std::endl;
	if (!index_loaded_) {
		throw std::runtime_error("Coordinator: Error - Index not loaded for search.");
	}
	if (!searcher_ || !graph_manager_) {
		throw std::runtime_error("Coordinator: Searcher or GraphManager not initialized.");
	}
	if (k_neighbors == 0) {
		throw std::runtime_error("Coordinator: k_neighbors cannot be 0 for search.");
	}

	// Clear previous results
	result_row_ids.clear();

	// Determine the actual search list size (L_search)
	common::idx_t L_search = (search_list_size > 0) ? search_list_size : config_.l_search;
	if (L_search < k_neighbors) {
		// Warn or adjust? L_search (" << L_search << ") < k (" <<
		// k_neighbors << ")" << std::endl;
		L_search = k_neighbors;
	}

	try {
		// Delegate the actual search algorithm to the ISearcher implementation
		// Pass the graph_manager for graph traversal access.
		// The ISearcher interface might need refinement based on exact needs.
		searcher_->Search(query_vector, config_, graph_manager_.get(), k_neighbors, result_row_ids, L_search);

		// ISearcher implementation should populate result_row_ids directly with the
		// final top-k sorted RowIDs. No further sorting needed here if ISearcher
		// handles it.

	} catch (const std::exception &e) {
		std::cerr << "Coordinator: Error during Search: " << e.what() << std::endl;
		result_row_ids.clear(); // Ensure empty results on error
		throw;                  // Re-throw
	}
}

void Coordinator::Insert(const float *data_vector, size_t data_dim, common::row_t label) {
	std::cout << "Coordinator: Insert called for label: " << label << std::endl;
	if (!index_loaded_) {
		throw std::runtime_error("Coordinator: Error - Index not loaded for insert.");
	}
	if (data_dim != config_.dimensions) {
		// This check might be better done in LmDiskannIndex before calling
		// coordinator
		throw std::runtime_error("Coordinator: Insert data dimension mismatch.");
	}
	if (!graph_manager_ || !searcher_ || !storage_manager_) { // storage_manager needed potentially for shadow
		                                                        // service coordination
		throw std::runtime_error("Coordinator: Managers not initialized for Insert.");
	}

	common::IndexPointer new_node_ptr;
	bool node_added = false;
	try {
		// 1. Allocate node and store vector data (via GraphManager)
		node_added = graph_manager_->AddNode(label, data_vector, config_.dimensions, new_node_ptr);
		if (!node_added || new_node_ptr.Get() == 0) {
			throw std::runtime_error("Failed to add node via GraphManager.");
		}

		// 2. Find candidate neighbors for the new node (using Searcher)
		//    Need a temporary vector to hold search results (RowIDs)
		//    The number of neighbors to search for depends on L_insert from config
		std::vector<common::row_t> candidate_neighbors;
		// Note: Search interface might need refinement. Does it return distances?
		// How many neighbors to search for? L_insert? A fixed number?
		// Assuming searcher can find appropriate candidates for RobustPrune.
		// Placeholder call - ISearcher::Search might not be the right method for
		// this. Maybe IGraphManager should handle finding neighbors for insertion?
		// For now, assume we get candidates somehow. Let's use L_insert neighbors
		// for pruning.
		searcher_->Search(data_vector, config_, graph_manager_.get(), config_.l_insert, candidate_neighbors,
		                  config_.l_insert);

		// Remove self from candidates if present
		// candidate_neighbors.erase(std::remove(candidate_neighbors.begin(),
		// candidate_neighbors.end(), label), candidate_neighbors.end());

		// 3. Prune neighbors and establish connections (via GraphManager)
		// RobustPrune expects candidates to be modified in-place.
		graph_manager_->RobustPrune(new_node_ptr, data_vector, candidate_neighbors, config_);

		// 4. Update entry point if necessary (via GraphManager)
		// GraphManager might handle this internally during RobustPrune or AddNode
		// graph_manager_->UpdateEntryPointIfNeeded(new_node_ptr, label);

		// 5. Log insertion with Shadow Service (if available)
		if (shadow_storage_service_) {
			shadow_storage_service_->LogInsert(label, new_node_ptr);
		}

		// 6. Mark index as dirty
		SetDirty(true);

	} catch (const std::exception &e) {
		std::cerr << "Coordinator: Error during Insert for label " << label << ": " << e.what() << std::endl;
		// Rollback? If AddNode succeeded but subsequent steps failed, need cleanup.
		if (node_added) {
			try {
				graph_manager_->FreeNode(label); // Attempt to free the allocated node
			} catch (...) {                    /* Best effort cleanup */
			}
		}
		// TODO: Rollback shadow log entry if applicable
		throw; // Re-throw
	}
}

void Coordinator::Delete(common::row_t label) {
	std::cout << "Coordinator: Delete called for label: " << label << std::endl;
	if (!index_loaded_) {
		std::cerr << "Coordinator: Error - Index not loaded for delete." << std::endl;
		return;
	}

	try {
		// 1. Log to shadow store (if available)
		if (shadow_storage_service_) {
			shadow_storage_service_->LogDelete(label);
		}

		// 2. Add to persistent delete queue (via StorageManager)
		if (storage_manager_) {
			storage_manager_->EnqueueDeletion(label, delete_queue_head_ptr_); // delete_queue_head_ptr_ is member of
			                                                                  // Coordinator
		} else {
			throw std::runtime_error("StorageManager not available in Coordinator::Delete");
		}

		// 3. Handle logical deletion in graph (e.g., remove from entry point
		// candidates)
		if (graph_manager_) {
			graph_manager_->HandleNodeDeletion(label);
		} else {
			throw std::runtime_error("GraphManager not available in Coordinator::Delete");
		}

		// 4. Free the node block (this might be better handled by StorageManager
		// during vacuum?) For now, let GraphManager handle freeing the block
		// associated with the row_id. This assumes GraphManager handles the mapping
		// and allocator interaction.
		if (graph_manager_) {
			graph_manager_->FreeNode(label);
		}

		// 5. Mark index as dirty
		SetDirty(true);

	} catch (const std::exception &e) {
		// TODO: Proper error handling/logging. Maybe rollback shadow log?
		std::cerr << "Coordinator: Error during Delete for label " << label << ": " << e.what() << std::endl;
		// Re-throw or handle error appropriately
		throw; // Re-throwing for now
	}
}

void Coordinator::Update(common::row_t label, const float *new_data_vector, size_t data_dim) {
	(void)new_data_vector; // Marked as unused
	(void)data_dim;        // Marked as unused
	std::cout << "Coordinator: Update called for label: " << label << std::endl;
	if (!index_loaded_) {
		std::cerr << "Coordinator: Error - Index not loaded for update." << std::endl;
		return;
	}
	// This is often implemented as a delete then insert.
	// Coordinator::Delete(label);
	// Coordinator::Insert(new_data_vector, data_dim, label);
	// Ensure transactional integrity across these operations using
	// IShadowStorageService.
}

void Coordinator::LoadIndex(const std::string &index_path) {
	std::cout << "Coordinator: LoadIndex called for path: " << index_path << std::endl;
	// 1. Use StorageManager to load the graph data (`graph.lmd`) and metadata
	// from disk.
	// 2. Populate GraphManager with the loaded graph structure.
	// 3. Use IShadowStorageService to load any relevant shadow store state or
	// apply pending operations.
	// 4. Set internal state (e.g., `index_loaded_ = true`, `index_path_`).

	// Conceptual: IStorageManager needs a method to load data and populate these
	// fields. This might involve passing references to config_, graph_manager_
	// and other state members, or returning a struct with all loaded data. For
	// now, let's assume it populates the coordinator's members directly or via a
	// returned struct.
	if (!storage_manager_) {
		throw std::runtime_error("StorageManager is not initialized in Coordinator.");
	}

	// Placeholder for actual call to IStorageManager.
	// This signature for storage_manager_->LoadIndexContents is illustrative.
	// It would need to populate graph_entry_point_ptr_, graph_entry_point_rowid_,
	// delete_queue_head_ptr_, and potentially update config_ and graph_manager_.
	storage_manager_->LoadIndexContents(index_path, config_,
	                                    graph_manager_.get(), // or however graph data is passed
	                                    graph_entry_point_ptr_, graph_entry_point_rowid_, delete_queue_head_ptr_);

	this->index_path_ = index_path;
	this->index_loaded_ = true;
	this->is_dirty_ = false;                         // Crucial: A freshly loaded index is not dirty
	graph_entry_point_ptr_ = common::IndexPointer(); // Assuming LoadIndexContents might not set if empty
	delete_queue_head_ptr_ = common::IndexPointer(); // Assuming LoadIndexContents might not set if empty
	std::cout << "Coordinator: Index loaded successfully from " << index_path << std::endl;
}

void Coordinator::SaveIndex(const std::string &index_path) {
	std::cout << "Coordinator: SaveIndex called for path: " << index_path << std::endl;
	if (!index_loaded_) {
		std::cerr << "Coordinator: Error - No index loaded to save." << std::endl;
		return;
	}
	// 1. Use StorageManager to save the current graph and metadata to disk.
	// 2. Ensure any pending shadow operations are flushed/committed via
	// IShadowStorageService.
}

void Coordinator::InitializeIndex(::diskann::common::idx_t estimated_cardinality) {
	std::cout << "Coordinator: InitializeIndex called with estimated_cardinality: " << estimated_cardinality << std::endl;
	if (!storage_manager_ || !graph_manager_) {
		throw std::runtime_error("Coordinator: Managers not initialized for InitializeIndex.");
	}
	// 1. Initialize IndexConfig (if not already done or if it needs defaults for
	// a new index)
	//    - config_ should be usable here.

	// 2. Tell StorageManager to prepare for a new index at index_path_ (if path
	// is known)
	//    or get a new storage location. For now, assume index_path_ might be set
	//    from config or externally. This might involve creating metadata files.
	//    storage_manager_->InitializeNewStorage(index_path_, config_);

	// 3. Tell GraphManager to set up an empty graph structure based on config.
	//    graph_manager_->InitializeEmptyGraph(config_, graph_entry_point_ptr_,
	//    graph_entry_point_rowid_); The entry point for an empty graph might be
	//    null/invalid.

	// 4. Set delete queue head to an empty state.
	//    delete_queue_head_ptr_ =  ... // some null/empty representation
	delete_queue_head_ptr_ = common::IndexPointer();

	// 5. Set internal state
	// this->index_path_ = ... // determined by storage_manager_ or config
	this->index_loaded_ = true; // Index is now 'loaded' in an empty state
	this->is_dirty_ = true;     // New index, needs to be saved if anything is added
	graph_entry_point_ptr_ = common::IndexPointer();
	graph_entry_point_rowid_ = common::NumericLimits<common::row_t>::Maximum(); // Use
	                                                                            // common::NumericLimits
	delete_queue_head_ptr_ = common::IndexPointer();
	std::cout << "Coordinator: New index initialized." << std::endl;
}

void Coordinator::HandleCommitDrop() {
	std::cout << "Coordinator: HandleCommitDrop called." << std::endl;
	// Instruct managers to clean up their state related to this index instance
	if (graph_manager_) {
		graph_manager_->Reset();
	}
	if (storage_manager_) {
		// StorageManager might need to clean up memory, close files,
		// or potentially delete storage artifacts if appropriate here.
		// Adding a Reset or similar method to IStorageManager might be needed.
		// For now, assume it doesn't hold state needing reset, or Reset is part of
		// its destructor. storage_manager_->Reset();
	}
	if (searcher_) {
		// Searcher might also have state to reset.
		// searcher_->Reset();
	}
	if (shadow_storage_service_) {
		// Shadow service might need to discard pending changes or clean up.
		// shadow_storage_service_->RollbackChanges(); // Or similar
	}

	// Reset Coordinator's own state
	index_loaded_ = false;
	is_dirty_ = false;
	index_path_.clear();
	graph_entry_point_ptr_ = common::IndexPointer();
	graph_entry_point_rowid_ = -1;
	delete_queue_head_ptr_ = common::IndexPointer();

	// The unique_ptrs to managers will be destroyed when Coordinator is
	// destroyed.
}

void Coordinator::PerformVacuum() {
	std::cout << "Coordinator: PerformVacuum called." << std::endl;
	if (!storage_manager_) {
		throw std::runtime_error("StorageManager not initialized in Coordinator::PerformVacuum");
	}
	try {
		// Delegate processing the delete queue to the storage manager
		storage_manager_->ProcessDeletionQueue(delete_queue_head_ptr_);
		// After processing, the index state might have changed significantly
		SetDirty(true); // Mark as dirty if vacuum modifies state that needs saving
	} catch (const std::exception &e) {
		std::cerr << "Coordinator: Error during PerformVacuum: " << e.what() << std::endl;
		// Handle or re-throw
		throw;
	}
}

common::idx_t Coordinator::GetInMemorySize() const {
	common::idx_t total_size = sizeof(*this);
	// Add size of config_ itself. If LmDiskannConfig contains dynamic members
	// (e.g. std::string path), a more accurate GetInMemorySize() would be needed
	// for LmDiskannConfig.
	total_size += sizeof(config_); // Placeholder for actual config size contribution

	if (graph_manager_) {
		total_size += graph_manager_->GetInMemorySize();
	}
	if (storage_manager_) {
		total_size += storage_manager_->GetInMemorySize();
	}
	if (searcher_) {
		total_size += searcher_->GetInMemorySize();
	}
	// IShadowStorageService typically wouldn't hold large data in memory itself,
	// but if it did, it would need GetInMemorySize too.
	return total_size;
}

} // namespace core
} // namespace diskann
