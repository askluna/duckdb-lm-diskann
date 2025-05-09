/**
 * @file StorageManager.hpp
 * @brief Declares functions related to LM-DiskANN index persistence and storage
 * management.
 * TODO: MAKE INTO CLASS
 */
#pragma once

#include "IStorageManager.hpp" // Include the interface
// #include "duckdb.hpp" // No longer directly needed for members
// #include "duckdb/common/common.hpp"                        // For row_t, handled by common::types
// #include "duckdb/execution/index/fixed_size_allocator.hpp" // Allocator is removed
// #include "duckdb/execution/index/index_pointer.hpp"        // Handled by common::types
// #include "duckdb/storage/buffer/buffer_handle.hpp"         // Buffer manager removed
// #include "duckdb/storage/buffer_manager.hpp" // Buffer manager removed
// #include "duckdb/storage/index_storage_info.hpp" // This is a problematic DuckDB specific type

#include "../common/duckdb_types.hpp"          // For common types like common::idx_t, common::row_t
#include "../store/IFileSystemService.hpp"     // New dependency
#include "../store/IPrimaryStorageService.hpp" // New dependency for fetching base table vectors
#include "../store/IShadowStorageService.hpp"  // New dependency
#include "index_config.hpp"                    // Include for LmDiskannConfig, LmDiskannMetadata, NodeLayoutOffsets

#include <memory> // For std::unique_ptr (though allocator_ is removed, maybe for future caches)
#include <string> // For std::string in method signatures

namespace diskann {
namespace core {
// Forward declarations
// class AttachedDatabase; // Removed
struct NodeLayoutOffsets; // Already in index_config.hpp

class StorageManager : public virtual core::IStorageManager {
	public:
	/**
	 * @brief Constructor for StorageManager.
	 * @param config The index configuration.
	 * @param node_layout Pre-calculated node layout offsets.
	 * @param file_system_service Service for primary graph file I/O (graph.lmd).
	 * @param shadow_storage_service Service for shadow store and metadata I/O (diskann_store.duckdb).
	 * @param primary_storage_service Service for fetching original vectors from the base table.
	 */
	StorageManager(const LmDiskannConfig &config, const NodeLayoutOffsets &node_layout,
	               store::IFileSystemService *file_system_service, store::IShadowStorageService *shadow_storage_service,
	               store::IPrimaryStorageService *primary_storage_service);

	~StorageManager() override;

	// --- IStorageManager Interface Implementation ---
	// (Signatures remain the same as in IStorageManager.hpp for now)
	void LoadIndexContents(const std::string &index_path, LmDiskannConfig &config_out, IGraphManager *graph_manager_out,
	                       common::IndexPointer &entry_point_ptr_out, common::row_t &entry_point_rowid_out,
	                       common::IndexPointer &delete_queue_head_out) override;

	void InitializeNewStorage(const std::string &index_path, const LmDiskannConfig &config) override;

	void SaveIndexContents(const std::string &index_path, const LmDiskannConfig &config,
	                       const IGraphManager *graph_manager, common::IndexPointer entry_point_ptr,
	                       common::row_t entry_point_rowid, common::IndexPointer delete_queue_head) override;

	common::idx_t GetInMemorySize() const override;

	// This method is problematic for full decoupling as it returns a DuckDB specific type.
	// For now, keeping signature as per IStorageManager. Will need to be addressed.
	::duckdb::IndexStorageInfo GetIndexStorageInfo() override;

	void EnqueueDeletion(common::row_t row_id, common::IndexPointer &delete_queue_head_ptr) override;

	void ProcessDeletionQueue(common::IndexPointer &delete_queue_head_ptr) override;

	// AllocateNodeBlock now needs to use IFileSystemService for graph.lmd blocks.
	// The concept of row_id here is for context, actual block doesn't store it directly in this model.
	bool AllocateNodeBlock(common::row_t row_id, common::IndexPointer &node_ptr_out,
	                       common::data_ptr_t &node_data_out) override;

	common::const_data_ptr_t GetNodeBlockData(common::IndexPointer node_ptr) const override;

	common::data_ptr_t GetMutableNodeBlockData(common::IndexPointer node_ptr) override;

	void MarkBlockDirty(common::IndexPointer node_ptr) override;

	private:
	// Removed DuckDB specific members
	// ::duckdb::BufferManager &buffer_manager_;
	// std::unique_ptr<::duckdb::FixedSizeAllocator> allocator_;
	// ::duckdb::AttachedDatabase &db_;

	LmDiskannConfig config_;
	NodeLayoutOffsets node_layout_;
	store::IFileSystemService *file_system_service_;       // Raw pointer, lifetime managed externally
	store::IShadowStorageService *shadow_storage_service_; // Raw pointer, lifetime managed externally
	store::IPrimaryStorageService *primary_storage_service_;

	// In-memory cache for frequently accessed/dirty primary graph blocks (graph.lmd)
	// This would be a new component if StorageManager handles caching. For now, assume direct passthrough.
	// Example: std::unique_ptr<NodeBlockCache> primary_block_cache_;

	// --- Private Helper Methods (Derived from old global functions) ---
	// These methods are intended to encapsulate logic that will use the service members.
	// Their signatures are changed to remove direct DuckDB dependencies.

	/**
	 * @brief Tries to find the IndexPointer for a given row_id using the shadow store.
	 * @param row_id The row_id to look up.
	 * @param node_ptr Output parameter for the found IndexPointer.
	 * @return True if found, false otherwise.
	 * V2 NOTE: This responsibility moves to IShadowStorageService.
	 */
	bool TryGetNodePointerFromShadow(common::row_t row_id, common::IndexPointer &node_ptr);

	/**
	 * @brief Allocates a new node in the shadow store (e.g., updates ART map).
	 * @param row_id The row_id for which to allocate and map an IndexPointer.
	 * @return The allocated IndexPointer.
	 * V2 NOTE: RowID map update by IShadowStorageService. Block allocation in graph.lmd is separate.
	 */
	common::IndexPointer AllocateAndMapNodeInShadowStore(common::row_t row_id);

	/**
	 * @brief Deletes a node from the shadow store map and handles associated block freeing logic via services.
	 * @param row_id The row_id to delete.
	 * V2 NOTE: RowID map removal by IShadowStorageService. Block freeing in graph.lmd is separate.
	 */
	void DeleteNodeFromShadowStoreAndFreeBlock(common::row_t row_id);

	// GetNodeBuffer is largely superseded by GetNodeBlockData / GetMutableNodeBlockData using IFileSystemService.
	// For now, we won't add a direct replacement but expect its logic to be in those interface methods.

	/**
	 * @brief Persists LM-DiskANN metadata using the shadow storage service.
	 * @param metadata The metadata to persist.
	 * V2 NOTE: Metadata is managed by IShadowStorageService in diskann_store.duckdb.
	 */
	void PersistLmDiskannMetadata(const LmDiskannMetadata &metadata);

	/**
	 * @brief Loads LM-DiskANN metadata using the shadow storage service.
	 * @param metadata_out Output parameter for the loaded metadata.
	 * V2 NOTE: Metadata is loaded by IShadowStorageService from diskann_store.duckdb.
	 */
	void LoadLmDiskannMetadata(LmDiskannMetadata &metadata_out);

	/**
	 * @brief Gets a valid entry point row_id.
	 *        Uses persisted pointer or falls back to a random node from the shadow store.
	 * @param graph_entry_point_ptr Persisted entry point pointer (can be invalid).
	 * @return A valid row_id for an entry point.
	 */
	common::row_t GetConsistentEntryPointRowId(common::IndexPointer graph_entry_point_ptr);

	/**
	 * @brief Gets a random node ID from the shadow store mapping table.
	 * @return A random row_id.
	 * V2 NOTE: This would query IShadowStorageService for a random RowID.
	 */
	common::row_t GetRandomNodeIDFromShadowStore();
};

// Placeholder global functions from StorageManager.cpp are removed from here as they were not class members.
// Their logic, if still needed, will be part of the class methods or new services.

} // namespace core
} // namespace diskann
