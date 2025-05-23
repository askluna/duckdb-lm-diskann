/**
 * @file LmDiskannIndex.hpp
 * @brief Contains the main LmDiskannIndex class definition.
 */
#pragma once

// Main DuckDB includes needed by the index class itself
#include "duckdb.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/execution/index/bound_index.hpp"
#include "duckdb/execution/index/fixed_size_allocator.hpp" // Include FixedSizeAllocator header
#include "duckdb/execution/index/index_pointer.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/parser/parsed_data/create_index_info.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/storage/index_storage_info.hpp"
#include "duckdb/storage/table_io_manager.hpp"

// Include headers for the refactored components
#include "../common/ann.hpp"
#include "../common/duckdb_types.hpp"
#include "../core/Coordinator.hpp"
#include "../core/index_config.hpp"
#include "LmDiskannScanState.hpp"

// Add these for the ParseOptions method
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/types/value.hpp"

#include <map>    // Using std::map for in-memory RowID mapping for now
#include <memory> // For unique_ptr
#include <string>
#include <vector>

namespace diskann {
namespace db {

// Forward declarations for components used internally
// class FixedSizeAllocator; // Now included
class ClientContext;
class DataChunk;
class Vector;
class BufferHandle;
// class ART; // If using ART for mapping

// --- Helper Struct for DuckDB Integration State --- //
/**
 * @brief Holds references and objects related to DuckDB integration.
 * @details Bundles database references, IO manager, allocator, etc.
 */
struct LmDiskannDBState {
	/**
	 * @brief Constructor for LmDiskannDBState.
	 * @param db_ref Reference to the AttachedDatabase.
	 * @param io_manager Reference to the TableIOManager.
	 * @param type_ref The logical type of the indexed column.
	 */
	LmDiskannDBState(::duckdb::AttachedDatabase &db_ref, ::duckdb::TableIOManager &io_manager,
	                 const ::duckdb::LogicalType &type_ref)
	    : db(db_ref), table_io_manager(io_manager), indexed_column_type(type_ref) {
	}

	::duckdb::AttachedDatabase &db;             // Reference to the database instance.
	::duckdb::TableIOManager &table_io_manager; // Reference to DuckDB's IO manager.
	::duckdb::LogicalType indexed_column_type;  // The full logical type of the indexed
	                                            // column (e.g., ARRAY(FLOAT, 128)).
	::duckdb::IndexPointer metadata_ptr;        // Pointer to the block holding persistent index metadata.
};

// --- Core LmDiskannIndex Class --- //
/**
 * @brief Orchestrates LM-DiskANN operations.
 * @details Interfaces with DuckDB's index system and delegates tasks to
 *          specialized modules (config, node, storage, search, distance).
 */
class LmDiskannIndex : public ::duckdb::BoundIndex {
	public:
	/**
	 * @brief Type name used for registration and in CREATE INDEX ... USING
	 * LM_DISKANN.
	 */
	static constexpr const char *TYPE_NAME = "LM_DISKANN";

	/**
	 * @brief Constructor: Initializes parameters, storage, and loads/creates the
	 * index.
	 * @param name Index name.
	 * @param index_constraint_type Type of index constraint (UNIQUE, PRIMARY_KEY,
	 * FOREIGN_KEY, NONE).
	 * @param column_ids List of physical column identifiers bound to the index
	 * expressions.
	 * @param table_io_manager The table's IO manager.
	 * @param unbound_expressions The unbound expressions used to create the
	 * index.
	 * @param db The attached database instance.
	 * @param options Parsed WITH clause options.
	 * @param storage_info Information about existing index storage (if loading).
	 * @param estimated_cardinality Estimated number of rows in the table.
	 */
	LmDiskannIndex(const ::duckdb::string &name, ::duckdb::IndexConstraintType index_constraint_type,
	               const ::duckdb::vector<::duckdb::column_t> &column_ids, ::duckdb::TableIOManager &table_io_manager,
	               const ::duckdb::vector<::duckdb::unique_ptr<::duckdb::Expression>> &unbound_expressions,
	               ::duckdb::AttachedDatabase &db, const ::duckdb::case_insensitive_map_t<::duckdb::Value> &options,
	               const ::duckdb::IndexStorageInfo &storage_info, idx_t estimated_cardinality);

	/**
	 * @brief Destructor
	 */
	~LmDiskannIndex() override;

	// --- Overridden BoundIndex Virtual Methods --- //
	/** @brief Appends data to the index. Called during bulk loading. */
	::duckdb::ErrorData Append(::duckdb::IndexLock &lock, ::duckdb::DataChunk &entries,
	                           ::duckdb::Vector &row_identifiers) override;
	/** @brief Commits a drop operation, freeing allocated resources. */
	void CommitDrop(::duckdb::IndexLock &index_lock) override;
	/** @brief Deletes entries from the index. */
	void Delete(::duckdb::IndexLock &lock, ::duckdb::DataChunk &entries, ::duckdb::Vector &row_identifiers) override;
	/** @brief Inserts data into the index. */
	::duckdb::ErrorData Insert(::duckdb::IndexLock &lock, ::duckdb::DataChunk &data, ::duckdb::Vector &row_ids) override;
	/** @brief Retrieves storage information (metadata pointer, allocator stats).
	 */
	::duckdb::IndexStorageInfo GetStorageInfo(const bool get_buffers);
	/** @brief Gets the estimated in-memory size of the index structures. */
	idx_t GetInMemorySize();
	/** @brief Merges another index into this one (not implemented). */
	bool MergeIndexes(::duckdb::IndexLock &state, ::duckdb::BoundIndex &other_index) override;
	/** @brief Performs vacuuming operations (e.g., processing delete queue). */
	void Vacuum(::duckdb::IndexLock &state) override;
	/** @brief Verifies index integrity and returns a string representation
	 * (verification TBD). */
	::duckdb::string VerifyAndToString(::duckdb::IndexLock &state, const bool only_verify) override;
	/** @brief Verifies allocator integrity (delegated). */
	void VerifyAllocations(::duckdb::IndexLock &state) override;
	/** @brief Generates a constraint violation message (not applicable). */
	::duckdb::string GetConstraintViolationMessage(::duckdb::VerifyExistenceType verify_type, idx_t failed_index,
	                                               ::duckdb::DataChunk &input) override;

	// --- LM-DiskANN Specific Methods for Scanning --- //
	/**
	 * @brief Initializes the state for an index scan (k-NN search).
	 * @param context The client context.
	 * @param query_vector The query vector.
	 * @param k The number of nearest neighbors to find.
	 * @return A unique pointer to the initialized scan state.
	 */
	::duckdb::unique_ptr<::duckdb::IndexScanState> InitializeScan(::duckdb::ClientContext &context,
	                                                              const ::duckdb::Vector &query_vector, idx_t k);
	/**
	 * @brief Performs one step of the index scan, filling the result vector.
	 * @param state The current scan state.
	 * @param result The output vector to store resulting RowIDs.
	 * @return The number of results produced in this step.
	 */
	idx_t Scan(::duckdb::IndexScanState &state, ::duckdb::Vector &result);

	// --- Public Accessors (Potentially needed by other modules/friends) --- //
	/** @brief Get the attached database reference. */
	::duckdb::AttachedDatabase &GetAttachedDatabase() const {
		return db_state_.db;
	}
	/** @brief Get the fixed size allocator reference (via Coordinator ->
	 * GraphManager). */
	::duckdb::FixedSizeAllocator &GetAllocator() {
		if (!coordinator_ || !coordinator_->GetGraphManager())
			throw ::duckdb::InternalException("Coordinator or GraphManager not initialized in GetAllocator");
		// NOTE: IGraphManager might not expose GetAllocator directly.
		// This depends on the final IGraphManager design. Assuming GraphManager
		// concrete class has it for now. This might require casting or a different
		// way to access the allocator if strictly using interface. For now, comment
		// out or assume a way exists. return
		// coordinator_->GetGraphManager()->GetAllocator();
		throw ::duckdb::NotImplementedException("GetAllocator via Coordinator TBD");
	}
	/** @brief Get the calculated node layout offsets (via Coordinator ->
	 * Config). */
	core::NodeLayoutOffsets GetNodeLayout() const {
		if (!coordinator_)
			throw ::duckdb::InternalException("Coordinator not initialized in GetNodeLayout");
		// NodeLayoutOffsets might be part of config or calculated/held by
		// GraphManager or Coordinator itself. Assuming Coordinator::GetConfig()
		// provides access or it needs another accessor. return
		// coordinator_->GetConfig().node_layout; // If config holds it For now,
		// let's assume it needs calculation based on config.
		return CalculateLayoutInternal(coordinator_->GetConfig()); // Re-calculate based on coordinator's config
	}
	/** @brief Get the vector dimensions (via Coordinator -> Config). */
	idx_t GetDimensions() const {
		if (!coordinator_)
			throw ::duckdb::InternalException("Coordinator not initialized in GetDimensions");
		return coordinator_->GetConfig().dimensions;
	}
	/** @brief Get the size of the compressed ternary edge representation (via
	 * Coordinator -> Config). */
	idx_t GetEdgeVectorSizeBytes() const {
		if (!coordinator_)
			throw ::duckdb::InternalException("Coordinator not initialized in GetEdgeVectorSizeBytes");
		// Recalculate based on coordinator's config dimensions
		return core::GetTernaryEdgeSizeBytes(coordinator_->GetConfig().dimensions);
	}
	/** @brief Get the distance metric type (via Coordinator -> Config). */
	common::LmDiskannMetricType GetMetricType() const {
		if (!coordinator_)
			throw ::duckdb::InternalException("Coordinator not initialized in GetMetricType");
		return coordinator_->GetConfig().metric_type;
	}
	/** @brief Get the max neighbor degree (R) (via Coordinator -> Config). */
	uint32_t GetR() const {
		if (!coordinator_)
			throw ::duckdb::InternalException("Coordinator not initialized in GetR");
		return coordinator_->GetConfig().r;
	}
	/** @brief Get the alpha parameter for pruning (via Coordinator -> Config).
	 */
	float GetAlpha() const {
		if (!coordinator_)
			throw ::duckdb::InternalException("Coordinator not initialized in GetAlpha");
		return coordinator_->GetConfig().alpha;
	}
	/** @brief Get the search list size (via Coordinator -> Config). */
	uint32_t GetLSearch() const {
		if (!coordinator_)
			throw ::duckdb::InternalException("Coordinator not initialized in GetLSearch");
		return coordinator_->GetConfig().l_search;
	}
	/** @brief Get the node vector type (via Coordinator -> Config). */
	common::LmDiskannVectorType GetNodeVectorType() const {
		if (!coordinator_)
			throw ::duckdb::InternalException("Coordinator not initialized in GetNodeVectorType");
		return coordinator_->GetConfig().node_vector_type;
	}

	// --- New Public Methods for GraphOperations --- //
	/** @brief Public wrapper for calculating approximate distance. */
	float PublicCalculateApproxDistance(const float *query_ptr, ::duckdb::const_data_ptr_t compressed_neighbor_ptr);
	/** @brief Public wrapper for compressing a vector for edge storage. */
	void PublicCompressVectorForEdge(const float *input_vector, ::duckdb::data_ptr_t output_compressed_vector);
	/** @brief Public wrapper for converting a raw node vector to float. */
	void PublicConvertNodeVectorToFloat(::duckdb::const_data_ptr_t raw_node_vector, float *float_vector_out);
	/** @brief Public method to mark the index as dirty. */
	void PublicMarkDirty(bool dirty_state = true);

	private:
	friend class LmDiskannStorageHelper; // Allow storage helpers access
	/**
	 * @brief Allows search access, passing config and db_state
	 * @param scan_state The search state.
	 * @param index The index instance.
	 * @param config The index configuration.
	 * @param find_exact_distances Flag for final re-ranking.
	 */
	friend void PerformSearch(LmDiskannScanState &scan_state, LmDiskannIndex &index, const core::LmDiskannConfig &config,
	                          bool find_exact_distances);

	// --- Core Parameters (Held in Config Struct) --- //
	// /** @brief Parsed and validated configuration parameters. */
	// core::LmDiskannConfig config_; // Coordinator manages config
	// /** @brief Calculated layout based on config. */
	// core::NodeLayoutOffsets node_layout_; // Coordinator or its managers
	// handle this
	// /** @brief Final aligned size of each node's block on disk. */
	// idx_t block_size_bytes_; // Coordinator or its managers handle this

	// --- DuckDB Integration State (subset of original) ---
	LmDiskannDBState db_state_;
	/** @brief Path to the index-specific data directory (e.g.,
	 * [db_name].lmd_idx/[index_name]/) */
	::duckdb::string index_data_path_;

	/** @brief Internal format version (for metadata check). */
	uint8_t format_version_;

	// --- Core LM-DiskANN Components (Refactored) --- //
	// ::duckdb::unique_ptr<core::GraphManager> node_manager_; // Coordinator has
	// graph_manager_
	// ::duckdb::unique_ptr<core::GraphOperations> graph_operations_; // Logic
	// moved into GraphManager/Searcher/Coordinator

	// --- Graph State (Now managed by Coordinator or its components) --- //
	// ::duckdb::IndexPointer graph_entry_point_ptr_; // Coordinator manages this
	// state
	// ::duckdb::row_t graph_entry_point_rowid_;      // Coordinator manages this
	// state
	// ::duckdb::IndexPointer delete_queue_head_ptr_; // Coordinator manages this
	// state bool is_dirty_ = false; // Coordinator manages this flag

	std::unique_ptr<core::Coordinator> coordinator_; // Coordinator instance

	// --- Helper Methods (Private to LmDiskannIndex) ---
	void InitializeNewIndex(idx_t estimated_cardinality);
	void LoadFromStorage(const ::duckdb::IndexStorageInfo &storage_info); // Keep declaration if called by
	                                                                      // constructor path indirectly
	core::LmDiskannConfig ParseOptions(const ::duckdb::case_insensitive_map_t<::duckdb::Value> &options);

	// --- Original Private Distance/Conversion Helpers (to be wrapped or moved)
	// --- //
	/** @brief Calculates approximate distance using ternary compressed vector.
	 * Uses config_. */
	float CalculateApproxDistance(const float *query_ptr, ::duckdb::const_data_ptr_t compressed_neighbor_ptr);
	/** @brief Compresses a float vector for edge storage (Ternary). Uses config_.
	 */
	void CompressVectorForEdge(const float *input_vector, ::duckdb::data_ptr_t output_compressed_vector);
	/** @brief Calculates exact distance between two raw vectors. Uses config_. */
	template <typename T_QUERY, typename T_NODE> // T_QUERY is likely float
	float CalculateExactDistance(const T_QUERY *query_ptr, ::duckdb::const_data_ptr_t node_vector_ptr);
	/** @brief Converts raw node vector to float. Uses config_. */
	void ConvertNodeVectorToFloat(::duckdb::const_data_ptr_t raw_node_vector, float *float_vector_out);
};

} // namespace db
} // namespace diskann
