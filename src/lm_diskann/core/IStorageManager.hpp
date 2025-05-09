#pragma once

#include "../common/ann.hpp"
#include "../common/duckdb_types.hpp"
#include "index_config.hpp"

#include <memory>
#include <string>

// Forward declare DuckDB types used in the interface
namespace duckdb {
struct IndexStorageInfo;
}

// Forward declarations to avoid circular dependencies
// These types will need to be fully defined in their respective headers.
namespace diskann {
namespace core {
class IGraphManager; // For passing graph data or interacting with graph manager
// Define or include common types like IndexPointer and row_t
// For now, using placeholder typedefs or assuming they will be globally
// available/included. Consider creating a common_types.hpp or similar.
} // namespace core
} // namespace diskann

namespace diskann {
namespace core {

/**
 * @brief Interface for managing the persistence of the DiskANN index.
 *
 * Defines the contract for loading, saving, and initializing index-related data
 * from/to storage. This includes the main graph, metadata, and potentially
 * other auxiliary structures.
 */
class IStorageManager {
	public:
	virtual ~IStorageManager() = default;

	/**
	 * @brief Loads the index data from the specified path.
	 *
	 * This method is responsible for reading all necessary components of the
	 * index from storage, such as metadata, graph structure, and any
	 * auxiliary data structures. It should populate the provided output
	 * parameters with the loaded data.
	 *
	 * @param index_path Path to the directory or file where the index is stored.
	 * @param config_out Output parameter to be populated with the loaded index
	 * configuration.
	 * @param graph_manager_out Pointer to the graph manager instance to be
	 * populated with graph data.
	 * @param entry_point_ptr_out Output parameter for the loaded graph entry
	 * point pointer.
	 * @param entry_point_rowid_out Output parameter for the loaded graph entry
	 * point row ID.
	 * @param delete_queue_head_out Output parameter for the head of the delete
	 * queue.
	 * @// TODO: Add other necessary parameters, e.g., for loading RowID maps or
	 * other state.
	 */
	virtual void LoadIndexContents(const std::string &index_path, LmDiskannConfig &config_out,
	                               IGraphManager *graph_manager_out, // Or another way to pass graph data
	                               common::IndexPointer &entry_point_ptr_out, common::row_t &entry_point_rowid_out,
	                               common::IndexPointer &delete_queue_head_out
	                               // Consider returning a struct with all loaded data instead
	                               // of many out-params
	                               ) = 0;

	/**
	 * @brief Initializes storage for a new, empty index.
	 *
	 * This could involve creating necessary files (e.g., metadata file) or
	 * allocating initial blocks.
	 *
	 * @param index_path Path where the new index will be stored.
	 * @param config The configuration for the new index.
	 */
	virtual void InitializeNewStorage(const std::string &index_path, const LmDiskannConfig &config) = 0;

	/**
	 * @brief Saves the current state of the index to the specified path.
	 *
	 * This method persists all components of the index, including metadata,
	 * the graph structure, and any other relevant state.
	 *
	 * @param index_path Path to the directory or file where the index should be
	 * saved.
	 * @param config The current index configuration to be saved.
	 * @param graph_manager Pointer to the graph manager containing the graph data
	 * to save.
	 * @param entry_point_ptr The current graph entry point pointer.
	 * @param entry_point_rowid The current graph entry point row ID.
	 * @param delete_queue_head The head of the delete queue.
	 * @// TODO: Add other necessary parameters to save the complete index state.
	 */
	virtual void SaveIndexContents(const std::string &index_path, const LmDiskannConfig &config,
	                               const IGraphManager *graph_manager, // Or another way to access graph data
	                               common::IndexPointer entry_point_ptr, common::row_t entry_point_rowid,
	                               common::IndexPointer delete_queue_head) = 0;

	/**
	 * @brief Returns the estimated in-memory size used by the storage manager.
	 * @return Size in bytes.
	 */
	virtual common::idx_t GetInMemorySize() const = 0;

	/**
	 * @brief Adds a node marked for deletion to the persistent delete queue.
	 *
	 * @param row_id The RowID of the node to enqueue for deletion.
	 * @param delete_queue_head_ptr Reference to the head pointer of the delete
	 * queue (will be updated).
	 */
	virtual void EnqueueDeletion(common::row_t row_id, common::IndexPointer &delete_queue_head_ptr) = 0;

	/**
	 * @brief Processes the delete queue, potentially reclaiming space (Vacuum).
	 * @param delete_queue_head_ptr Reference to the head pointer of the delete
	 * queue.
	 */
	virtual void ProcessDeletionQueue(common::IndexPointer &delete_queue_head_ptr) = 0;

	// --- Block-level operations needed by GraphManager ---

	/**
	 * @brief Allocates a new block in storage for a node.
	 *
	 * The storage manager is responsible for finding or creating space for this
	 * node. The returned node_data_out is a pointer to the raw memory of the
	 * block. The caller (GraphManager) is responsible for initializing this
	 * memory using NodeAccessors.
	 *
	 * @param row_id The row_id associated with this node (for tracking/metadata
	 * if needed by storage).
	 * @param node_ptr_out Output parameter: The IndexPointer (block_id, offset)
	 * for the allocated block.
	 * @param node_data_out Output parameter: A mutable pointer to the allocated
	 * block's data.
	 * @return True if allocation was successful, false otherwise.
	 */
	virtual bool AllocateNodeBlock(common::row_t row_id, common::IndexPointer &node_ptr_out,
	                               common::data_ptr_t &node_data_out) = 0;

	/**
	 * @brief Retrieves a constant pointer to a node's block data.
	 *
	 * Used for read-only access to the node's vector or neighbors.
	 * The lifetime of the returned pointer is managed by the StorageManager
	 * (e.g., pinned buffer).
	 *
	 * @param node_ptr The IndexPointer of the node to retrieve.
	 * @return A constant pointer to the block's data, or nullptr if not
	 * found/error.
	 */
	virtual common::const_data_ptr_t GetNodeBlockData(common::IndexPointer node_ptr) const = 0;

	/**
	 * @brief Retrieves a mutable pointer to a node's block data.
	 *
	 * Used for modifying a node's vector or neighbors.
	 * The lifetime of the returned pointer is managed by the StorageManager.
	 * After modification, MarkBlockDirty should be called.
	 *
	 * @param node_ptr The IndexPointer of the node to retrieve.
	 * @return A mutable pointer to the block's data, or nullptr if not
	 * found/error.
	 */
	virtual common::data_ptr_t GetMutableNodeBlockData(common::IndexPointer node_ptr) = 0;

	/**
	 * @brief Marks a block as dirty, indicating its contents have changed and
	 * need to be persisted.
	 *
	 * @param node_ptr The IndexPointer of the block that was modified.
	 */
	virtual void MarkBlockDirty(common::IndexPointer node_ptr) = 0;

	// Add other necessary virtual methods, e.g.:
	// virtual void GetMetadata(...) = 0;
	// virtual void SaveMetadata(...) = 0;
	// virtual bool IndexExists(const std::string& index_path) = 0;
	// virtual void DeleteIndexStorage(const std::string& index_path) = 0;
};

} // namespace core
} // namespace diskann
