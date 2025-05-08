#pragma once

#include "../common/ann.hpp"
#include "../common/duckdb_types.hpp"
#include "index_config.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Include full definitions for interface types used in std::unique_ptr members
#include "../store/IShadowStorageService.hpp"
#include "IGraphManager.hpp"
#include "ISearcher.hpp"
#include "IStorageManager.hpp"
#include "duckdb/storage/index_storage_info.hpp"

namespace duckdb {
class ClientContext;
// class IndexStorageInfo; // No longer needed as full include is added
} // namespace duckdb

// Forward declare types that might be used in method signatures
// For example, if you have specific structs/classes for queries, results, data
// points
// class QueryType;
// class ResultType;
// class DataType;

// Forward declarations are no longer needed as full headers are included above
// class IGraphManager;
// class IStorageManager;
// class ISearcher;
//
// namespace store {
// class IShadowStorageService;
// }

namespace diskann {
namespace core {

class Coordinator {
	public:
	// Constructor expecting injected dependencies
	Coordinator(std::unique_ptr<IStorageManager> storage_manager, std::unique_ptr<IGraphManager> graph_manager,
	            std::unique_ptr<ISearcher> searcher, std::unique_ptr<store::IShadowStorageService> shadow_storage_service,
	            const LmDiskannConfig &config
	            // duckdb::ClientContext& context // If direct context is needed
	);

	~Coordinator();

	// Core high-level operations
	// Parameters are illustrative and will need to be refined
	void BuildIndex(const std::string &data_path /*, other params */);

	// std::vector<ResultType> Search(const QueryType& query, int top_k);
	void Search(const float *query_vector, common::idx_t k_neighbors, std::vector<common::row_t> &result_row_ids,
	            common::idx_t search_list_size = 0);

	void Insert(const float *data_vector, size_t data_dim, common::row_t label);

	void Delete(common::row_t label);

	void Update(common::row_t label, const float *new_data_vector, size_t data_dim);

	void LoadIndex(const std::string &index_path);

	void SaveIndex(const std::string &index_path);

	void InitializeIndex(common::idx_t estimated_cardinality);

	common::idx_t GetInMemorySize() const;

	::duckdb::IndexStorageInfo GetIndexStorageInfo();

	// --- New lifecycle/maintenance methods ---
	void HandleCommitDrop();
	void PerformVacuum();

	// Other potential methods
	// void Consolidate();
	// IndexStats GetStats();

	bool IsDirty() const {
		return is_dirty_;
	}
	void SetDirty(bool dirty) {
		is_dirty_ = dirty;
	}

	// Accessors for state needed by LmDiskannIndex (after Coordinator manages
	// them)
	common::IndexPointer GetGraphEntryPointPtr() const {
		return graph_entry_point_ptr_;
	}
	common::row_t GetGraphEntryPointRowId() const {
		return graph_entry_point_rowid_;
	}

	// Accessors for internal managers and config (needed for delegation)
	const LmDiskannConfig &GetConfig() const {
		return config_;
	}
	IGraphManager *GetGraphManager() const {
		return graph_manager_.get();
	} // Return raw ptr, non-owning
	IStorageManager *GetStorageManager() const {
		return storage_manager_.get();
	} // Return raw ptr, non-owning
	ISearcher *GetSearcher() const {
		return searcher_.get();
	} // Return raw ptr, non-owning

	private:
	std::unique_ptr<IStorageManager> storage_manager_;
	std::unique_ptr<IGraphManager> graph_manager_;
	std::unique_ptr<ISearcher> searcher_;
	std::unique_ptr<store::IShadowStorageService> shadow_storage_service_;
	LmDiskannConfig config_;
	// duckdb::ClientContext& context_; // If direct context is needed

	// --- Helper Methods (Private to LmDiskannIndex) ---
	// These are now part of public coordinator methods or handled by managers
	// /** @brief Initializes structures for a new, empty index. */
	// void InitializeNewIndex(idx_t estimated_cardinality);
	// /** @brief Loads index metadata and state from existing storage info. */
	// void LoadFromStorage(const ::duckdb::IndexStorageInfo &storage_info);

	// Internal state
	std::string index_path_;
	bool index_loaded_ = false;
	bool is_dirty_ = false;
	common::IndexPointer graph_entry_point_ptr_;
	common::row_t graph_entry_point_rowid_ = common::NumericLimits<common::row_t>::Maximum();
	common::IndexPointer delete_queue_head_ptr_;

	// Potentially other state variables like graph entry point, metadata cache,
	// etc.
};

} // namespace core
} // namespace diskann
