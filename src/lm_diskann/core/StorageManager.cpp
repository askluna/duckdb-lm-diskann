#include "StorageManager.hpp"

#include "../common/duckdb_types.hpp" // For common type aliases and exceptions
#include "../store/IFileSystemService.hpp"
#include "../store/IPrimaryStorageService.hpp"
#include "../store/IShadowStorageService.hpp"
#include "index_config.hpp" // For LmDiskannConfig, LmDiskannMetadata, NodeLayoutOffsets

// #include "duckdb/storage/index_storage_info.hpp" // Intentionally removed: GetIndexStorageInfo will throw

#include <iostream> // For std::cout, std::cerr (temporary logging)

namespace diskann {
namespace core {

// Constructor Definition
StorageManager::StorageManager(const LmDiskannConfig &config, const NodeLayoutOffsets &node_layout,
                               store::IFileSystemService *file_system_service,
                               store::IShadowStorageService *shadow_storage_service,
                               store::IPrimaryStorageService *primary_storage_service)
    : config_(config), node_layout_(node_layout), file_system_service_(file_system_service),
      shadow_storage_service_(shadow_storage_service), primary_storage_service_(primary_storage_service) {
	if (!file_system_service_) {
		throw common::InternalException("StorageManager: IFileSystemService pointer cannot be null.");
	}
	if (!shadow_storage_service_) {
		throw common::InternalException("StorageManager: IShadowStorageService pointer cannot be null.");
	}
	if (!primary_storage_service_) {
		throw common::InternalException("StorageManager: IPrimaryStorageService pointer cannot be null.");
	}
	std::cout << "StorageManager: Constructed with services." << std::endl;
}

StorageManager::~StorageManager() {
	std::cout << "StorageManager: Destructed." << std::endl;
}

// --- Private Helper Methods Implementation (Derived from old global functions) ---

bool StorageManager::TryGetNodePointerFromShadow(common::row_t row_id, common::IndexPointer &node_ptr) {
	(void)row_id;
	(void)node_ptr;
	// Original global function was: bool TryGetNodePointer(::duckdb::row_t row_id, ::duckdb::IndexPointer &node_ptr,
	// ::duckdb::AttachedDatabase &db)
	// FIXME: Implement RowID map lookup (e.g., using ART) via shadow_storage_service_
	// This function would encapsulate the ART lookup logic.
	// V2 NOTE: This responsibility moves to IShadowStorageService.
	/* --- Original Logic (for reference, needs adaptation to services) ---
	throw common::NotImplementedException(
	    "Persistent TryGetNodePointer is not implemented and belongs to IShadowStorageService under V2.");
	return false;
	--- End Original Logic --- */
	throw common::NotImplementedException(
	    "TryGetNodePointerFromShadow to be refactored using IShadowStorageService. See V2 notes.");
}

common::IndexPointer StorageManager::AllocateAndMapNodeInShadowStore(common::row_t row_id) {
	(void)row_id;
	// Original global function was: ::duckdb::IndexPointer AllocateNode(::duckdb::row_t row_id,
	// ::duckdb::AttachedDatabase &db, ::duckdb::FixedSizeAllocator &allocator)
	// FIXME: Implement RowID map insertion via shadow_storage_service_
	// This function would encapsulate the allocator->New() call AND the ART
	// insert logic.
	// V2 NOTE: StorageManager::AllocateNodeBlock provides block from graph.lmd. RowID map update by
	// IShadowStorageService.
	/* --- Original Logic (for reference, needs adaptation to services) ---
	throw common::NotImplementedException("Persistent AllocateNode is not implemented. V2 separates block allocation "
	                                      "(StorageManager) and map update (IShadowStorageService).");
	return common::IndexPointer(); // Return default (invalid) common::IndexPointer
	--- End Original Logic --- */
	throw common::NotImplementedException("AllocateAndMapNodeInShadowStore to be refactored for V2 using "
	                                      "IShadowStorageService. Block allocation is separate.");
}

void StorageManager::DeleteNodeFromShadowStoreAndFreeBlock(common::row_t row_id) {
	(void)row_id;
	// Original global function was: void DeleteNodeFromMapAndFreeBlock(::duckdb::row_t row_id, ::duckdb::AttachedDatabase
	// &db, ::duckdb::FixedSizeAllocator &allocator)
	// FIXME: Implement RowID map deletion via shadow_storage_service_
	// This function would encapsulate the ART delete logic AND the
	// allocator->Free() call (which is now handled by IFileSystemService or shadow store for its own blocks).
	// V2 NOTE: StorageManager::FreeBlock adds to graph.lmd free list (via IFileSystemService). RowID map removal by
	// IShadowStorageService.
	/* --- Original Logic (for reference, needs adaptation to services) ---
	throw common::NotImplementedException(
	    "Persistent DeleteNodeFromMapAndFreeBlock is not implemented. V2 separates responsibilities.");
	--- End Original Logic --- */
	throw common::NotImplementedException(
	    "DeleteNodeFromShadowStoreAndFreeBlock to be refactored for V2. V2 separates responsibilities.");
}

void StorageManager::PersistLmDiskannMetadata(const LmDiskannMetadata &metadata) {
	(void)metadata;
	// Original global function was: void PersistMetadata(common::IndexPointer metadata_ptr, ::duckdb::AttachedDatabase
	// &db, ::duckdb::FixedSizeAllocator &allocator, const LmDiskannMetadata &metadata) V2 NOTE: This function will
	// change. Metadata is managed by IShadowStorageService in diskann_store.duckdb. StorageManager::SaveIndexContents
	// will trigger IShadowStorageService to save its metadata.
	/* --- Original Logic Clues (for reference, needs adaptation to services) ---
	// if (metadata_ptr.Get() == 0) { // Corrected IsValid() check // This check becomes less relevant if
	shadow_storage_service_ handles allocation of metadata block
	// 	throw common::InternalException("Cannot persist LM_DISKANN metadata: metadata pointer is invalid.");
	// }
	// Direct writing of LmDiskannMetadata using DuckDB allocators for graph.lmd metadata block is not V2.
	// auto &buffer_manager = ::duckdb::BufferManager::GetBufferManager(db); // db not available
	// auto handle = buffer_manager.Pin(allocator.GetMetaBlock(metadata_ptr.GetBlockId())); // allocator, metadata_ptr not
	available as params
	// ::duckdb::MetadataWriter writer(handle.GetFileBuffer(), metadata_ptr.GetOffset());

	// // --- Serialize Parameters --- (Example based on old signature, LmDiskannMetadata struct is now passed)
	// writer.Write<uint8_t>(metadata.format_version);
	// writer.Write<common::LmDiskannMetricType>(metadata.metric_type);
	// writer.Write<common::LmDiskannVectorType>(metadata.node_vector_type);
	// writer.Write<common::idx_t>(metadata.dimensions);
	// writer.Write<uint32_t>(metadata.r);
	// writer.Write<uint32_t>(metadata.l_insert);
	// writer.Write<float>(metadata.alpha);
	// writer.Write<uint32_t>(metadata.l_search);
	// writer.Write<common::idx_t>(metadata.block_size_bytes);
	// writer.Write<common::IndexPointer>(metadata.graph_entry_point_ptr);
	// writer.Write<common::IndexPointer>(metadata.delete_queue_head_ptr);
	// handle.SetModified();
	--- End Original Logic Clues --- */
	throw common::NotImplementedException(
	    "PersistLmDiskannMetadata to be refactored for V2 using IShadowStorageService.");
}

void StorageManager::LoadLmDiskannMetadata(LmDiskannMetadata &metadata_out) {
	(void)metadata_out;
	// Original global function was: void LoadMetadata(common::IndexPointer metadata_ptr, ::duckdb::AttachedDatabase &db,
	// ::duckdb::FixedSizeAllocator &allocator, LmDiskannMetadata &metadata_out) V2 NOTE: This function will change.
	// Metadata is loaded by IShadowStorageService from diskann_store.duckdb.
	/* --- Original Logic Clues (for reference, needs adaptation to services) ---
	// if (!metadata_ptr.IsValid()) { // metadata_ptr not available as param
	//   throw common::IOException( // Was ::duckdb::IOException
	//       "Cannot load LM_DISKANN index: metadata pointer is invalid.");
	// }
	// auto &buffer_manager = ::duckdb::BufferManager::GetBufferManager(db); // db not available
	// auto handle =
	//     buffer_manager.Pin(allocator.GetMetaBlock(metadata_ptr.GetBlockId())); // allocator not available
	// ::duckdb::MetadataReader reader(handle.GetFileBuffer(),
	//                                 metadata_ptr.GetOffset());

	// // --- Deserialize Parameters --- (Example based on old signature)
	// reader.Read<uint8_t>(metadata_out.format_version);
	// reader.Read<common::LmDiskannMetricType>(metadata_out.metric_type);
	// reader.Read<common::LmDiskannVectorType>(metadata_out.node_vector_type);
	// reader.Read<common::idx_t>(metadata_out.dimensions);
	// reader.Read<uint32_t>(metadata_out.r);
	// reader.Read<uint32_t>(metadata_out.l_insert);
	// reader.Read<float>(metadata_out.alpha);
	// reader.Read<uint32_t>(metadata_out.l_search);
	// reader.Read<common::idx_t>(metadata_out.block_size_bytes);
	// reader.Read<common::IndexPointer>(metadata_out.graph_entry_point_ptr);
	// reader.Read<common::IndexPointer>(metadata_out.delete_queue_head_ptr);
	--- End Original Logic Clues --- */
	throw common::NotImplementedException("LoadLmDiskannMetadata to be refactored for V2 using IShadowStorageService.");
}

common::row_t StorageManager::GetConsistentEntryPointRowId(common::IndexPointer graph_entry_point_ptr) {
	// Original global function was: common::row_t GetEntryPointRowId(common::IndexPointer graph_entry_point_ptr,
	// ::duckdb::AttachedDatabase &db, ::duckdb::FixedSizeAllocator &allocator) V2 NOTE: This logic needs to be updated.
	// If graph_entry_point_ptr is valid (and refers to a block in graph.lmd),
	// we might need to read the block header via IFileSystemService to get the row_id.
	// Otherwise, fall back to GetRandomNodeIDFromShadowStore().
	if (graph_entry_point_ptr.Get() != 0) {
		std::cerr << "Warning: GetConsistentEntryPointRowId: Cannot get row_id from pointer yet (V2 requires reading block "
		             "header via IFileSystemService or using IShadowStorageService)."
		          << std::endl;
		// Original logic from global GetEntryPointRowId might have had more here, or assumed an early return.
		// For V2, resolving graph_entry_point_ptr to a row_id will likely involve shadow_storage_service_
		// or reading a block header via file_system_service_.
		// Current fallback is to GetRandomNodeIDFromShadowStore if not resolved.
	}
	// Fallback or primary mechanism depending on V2 design:
	return GetRandomNodeIDFromShadowStore();
}

common::row_t StorageManager::GetRandomNodeIDFromShadowStore() {
	// Original global function was: common::row_t GetRandomNodeID(::duckdb::AttachedDatabase &db,
	// ::duckdb::FixedSizeAllocator &allocator) V2 NOTE: This would query IShadowStorageService for a random RowID from
	// its mapping table.
	/* --- Original Logic Clues (for reference, needs adaptation to services) ---
	// common::Printer::Warning("GetRandomNodeID not implemented, returning invalid rowid."); // Using common::Printer
	from duckdb_types.hpp
	// return common::NumericLimits<common::row_t>::Maximum();
	--- End Original Logic Clues --- */
	throw common::NotImplementedException("GetRandomNodeIDFromShadowStore to use IShadowStorageService.");
}

// --- IStorageManager Interface Implementation (Adapted) ---

void StorageManager::LoadIndexContents(const std::string &index_path, LmDiskannConfig &config_out,
                                       IGraphManager *graph_manager_out, common::IndexPointer &entry_point_ptr_out,
                                       common::row_t &entry_point_rowid_out,
                                       common::IndexPointer &delete_queue_head_out) {
	(void)index_path;
	(void)config_out;
	(void)graph_manager_out;
	(void)entry_point_ptr_out;
	(void)entry_point_rowid_out;
	(void)delete_queue_head_out;
	// V2 Notes from old LoadMetadata & GetEntryPointRowId have been moved to helper methods.
	// This method will orchestrate calls to helper methods like LoadLmDiskannMetadata,
	// GetConsistentEntryPointRowId, and interact with IFileSystemService and IGraphManager.
	// Conceptual calls:
	// LoadLmDiskannMetadata(config_out); // Fills config_out using shadow_storage_service_
	// shadow_storage_service_->LoadIndexStatePointers(entry_point_ptr_out, delete_queue_head_out);
	// entry_point_rowid_out = GetConsistentEntryPointRowId(entry_point_ptr_out);

	// std::string graph_file_path = index_path + "/graph.lmd"; // Example derivation
	// file_system_service_->Open(graph_file_path, true /* read_only */);
	// graph_manager_out->LoadGraphStructure(file_system_service_, entry_point_rowid_out /* or ptr */);
	throw common::NotImplementedException(
	    "StorageManager::LoadIndexContents partially refactored, needs full service integration.");
}

void StorageManager::InitializeNewStorage(const std::string &index_path, const LmDiskannConfig &config) {
	(void)index_path;
	(void)config;
	// This method will orchestrate calls to PersistLmDiskannMetadata (for initial config)
	// and interact with IFileSystemService to set up graph.lmd and IShadowStorageService for its setup.
	// Conceptual calls:
	// shadow_storage_service_->InitializeShadowStore(index_path, config);
	// PersistLmDiskannMetadata(config); // This might be part of InitializeShadowStore or separate

	// std::string graph_file_path = index_path + "/graph.lmd"; // Example derivation
	// file_system_service_->Open(graph_file_path, false); // Open for writing
	// file_system_service_->Truncate(0); // Ensure it's empty
	// Potentially write an initial empty graph header via file_system_service_
	// file_system_service_->Close(); // Or keep open if subsequent operations expect it
	throw common::NotImplementedException(
	    "StorageManager::InitializeNewStorage partially refactored, needs full service integration.");
}

void StorageManager::SaveIndexContents(const std::string &index_path, const LmDiskannConfig &config,
                                       const IGraphManager *graph_manager, common::IndexPointer entry_point_ptr,
                                       common::row_t entry_point_rowid, common::IndexPointer delete_queue_head) {
	// This method will orchestrate calls to PersistLmDiskannMetadata, graph_manager saving (using IFileSystemService),
	// and IShadowStorageService to save its state (e.g., rowid maps, delete queue pointers).
	// Conceptual calls:
	// std::string graph_file_path = index_path + "/graph.lmd";
	// file_system_service_->Open(graph_file_path, false);
	// graph_manager->SaveGraphStructure(file_system_service_);
	// file_system_service_->Sync();
	// file_system_service_->Close();

	// shadow_storage_service_->SaveLmDiskannMetadata(config); // includes config_
	// shadow_storage_service_->SaveIndexStatePointers(entry_point_ptr, delete_queue_head);
	// shadow_storage_service_->SaveRowIdMappingTable(); // Or similar if it manages the ART-like map
	throw common::NotImplementedException(
	    "StorageManager::SaveIndexContents needs rethink for delta-based vs full save w/ services.");
}

common::idx_t StorageManager::GetInMemorySize() const {
	common::idx_t size = sizeof(*this);
	// If StorageManager maintains any caches itself (e.g., for graph.lmd blocks),
	// add their size. For now, assuming services manage their own caches or are pass-through.
	return size;
}

void StorageManager::EnqueueDeletion(common::row_t row_id, common::IndexPointer &delete_queue_head_ptr) {
	// Original global function was: void EnqueueDeletion(common::row_t deleted_row_id, common::IndexPointer
	// &delete_queue_head_ptr, ::duckdb::AttachedDatabase &db, ::duckdb::FixedSizeAllocator &allocator) V2 NOTE: This
	// responsibility moves to IShadowStorageService. Comments from original global function regarding its logic
	// (allocating new queue entry, linking) are relevant for IShadowStorageService impl.
	/* --- Original Logic Clues (for reference, needs adaptation to services) ---
	// IndexPointer new_queue_entry_ptr = allocator.New(); // allocator not available, IShadowStorageService handles its
	own storage
	// if (!new_queue_entry_ptr.IsValid()) {
	// 	throw common::IOException("Failed to allocate block for delete queue entry."); // Was ::duckdb::IOException
	// }

	// auto handle = GetNodeBuffer(new_queue_entry_ptr, db, allocator, true); // GetNodeBuffer logic superseded, services
	manage buffers
	// auto data_ptr = handle.Ptr();

	// Store<common::row_t>(deleted_row_id, data_ptr); // deleted_row_id is row_id
	// common::IndexPointer current_head = delete_queue_head_ptr;
	// Store<common::block_id_t>(current_head.GetBlockId(), data_ptr + sizeof(common::row_t));
	// Store<uint32_t>(current_head.GetOffset(), data_ptr + sizeof(common::row_t) + sizeof(common::block_id_t));

	// handle.SetModified();
	// delete_queue_head_ptr = new_queue_entry_ptr; // Service would update this if it's an out param from service call
	--- End Original Logic Clues --- */
	throw common::NotImplementedException(
	    "EnqueueDeletion to be implemented via IShadowStorageService. See V2 notes from original global function.");
}

void StorageManager::ProcessDeletionQueue(common::IndexPointer &delete_queue_head_ptr) {
	// Original global function was: void ProcessDeletionQueue(common::IndexPointer &delete_queue_head_ptr,
	// ::duckdb::AttachedDatabase &db, ::duckdb::FixedSizeAllocator &allocator, const NodeLayoutOffsets &layout,
	// common::idx_t edge_vector_size_bytes) V2 NOTE: This responsibility moves to IShadowStorageService. Comments from
	// original global function are relevant for IShadowStorageService impl. if (delete_queue_head_ptr.Get() != 0) { ... }
	// // Original check
	/* --- Original Logic Clues (for reference, needs adaptation to services) ---
	// if (delete_queue_head_ptr.Get() != 0) { // Corrected IsValid() check
	// 	throw common::NotImplementedException("ProcessDeletionQueue: Processing deferred deletions is not "
	// 	                                      "implemented fully and needs V2 design.");
	// }
	// The actual processing logic (iterating queue, marking nodes deleted, reclaiming space)
	// would be complex and is now the responsibility of IShadowStorageService and coordinated actions.
	--- End Original Logic Clues --- */
	throw common::NotImplementedException(
	    "ProcessDeletionQueue to be implemented via IShadowStorageService. See V2 notes from original global function.");
}

bool StorageManager::AllocateNodeBlock(common::row_t row_id, common::IndexPointer &node_ptr_out,
                                       common::data_ptr_t &node_data_out) {
	// Original global function AllocateNode also handled map update, which is now separate.
	// This method is about allocating a block in graph.lmd using IFileSystemService.
	// V2 NOTE: StorageManager::AllocateNodeBlock provides block from graph.lmd.
	// Comments from original AllocateNode regarding allocator->New() are relevant for IFileSystemService block
	// management.
	/* --- Original Logic Clues from AllocateNode related to block allocation (for reference) ---
	// The part of AllocateNode that did `allocator.New()` is relevant to how IFileSystemService might manage blocks.
	// For instance, IFileSystemService might have its own free list or append-only strategy.
	--- End Original Logic Clues --- */
	// V2 NOTE from old GetNodeBuffer: This function will change significantly. It will use StorageManager's internal
	// cache and custom file I/O for graph.lmd, not DuckDB BufferManager for graph.lmd.
	throw common::NotImplementedException("AllocateNodeBlock requires rethink with IFileSystemService regarding buffer "
	                                      "management and block allocation strategy (free list vs append).");
}

common::const_data_ptr_t StorageManager::GetNodeBlockData(common::IndexPointer node_ptr) const {
	// Original global function was: ::duckdb::BufferHandle GetNodeBuffer(common::IndexPointer node_ptr,
	// ::duckdb::AttachedDatabase &db, ::duckdb::FixedSizeAllocator &allocator, bool write_lock) V2 NOTE: This function
	// will change significantly. It will use StorageManager's internal cache and custom file I/O for graph.lmd, not
	// DuckDB BufferManager for graph.lmd. Comments from GetNodeBuffer are relevant here for using IFileSystemService to
	// get data.
	/* --- Original GetNodeBuffer Logic Clues (for reference, needs adaptation to services) ---
	// if (node_ptr.Get() == 0) { // Corrected IsValid() check
	// 	throw common::IOException("Invalid node pointer provided to GetNodeBuffer.");
	// }
	// auto &buffer_manager = ::duckdb::BufferManager::GetBufferManager(db); // db not available
	// This function correctly uses the passed allocator reference.
	// return buffer_manager.Pin(allocator.GetBlock(node_ptr)); // allocator not available, services manage buffers
	--- End Original GetNodeBuffer Logic Clues --- */
	throw common::NotImplementedException("GetNodeBlockData requires robust buffer management via IFileSystemService or "
	                                      "internal cache. See V2 notes from GetNodeBuffer.");
}

common::data_ptr_t StorageManager::GetMutableNodeBlockData(common::IndexPointer node_ptr) {
	// Original global function was: ::duckdb::BufferHandle GetNodeBuffer(...)
	// V2 NOTE: Similar to GetNodeBlockData, but for mutable access.
	// Comments from GetNodeBuffer are relevant here for using IFileSystemService to get data.
	/* --- Original GetNodeBuffer Logic Clues (for reference, needs adaptation to services) ---
	// (Similar to GetNodeBlockData, but the GetNodeBuffer call would have write_lock = true)
	// The handle returned would allow modification, and then handle.SetModified() would be called.
	// This dirtying mechanism needs to be translated to the new service model (e.g., MarkBlockDirty).
	--- End Original GetNodeBuffer Logic Clues --- */
	throw common::NotImplementedException(
	    "GetMutableNodeBlockData requires robust buffer management and dirty tracking. See V2 notes from GetNodeBuffer.");
}

void StorageManager::MarkBlockDirty(common::IndexPointer node_ptr) {
	// This implies a caching layer within StorageManager or richer IFileSystemService that needs notification.
	// If IFileSystemService provides buffers directly, it might have its own dirty tracking,
	// or this method signals to flush a block managed by StorageManager's own cache.
	throw common::NotImplementedException(
	    "MarkBlockDirty implies an internal cache or interaction with file system service cache.");
}

} // namespace core
} // namespace diskann
