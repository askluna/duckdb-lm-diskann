#include "StorageManager.hpp"

#include "../common/ann.hpp"           // For LmDiskann enums
#include "../common/duckdb_types.hpp"  // For common type aliases
#include "duckdb/common/exception.hpp" // For duckdb::IOException, duckdb::NotImplementedException
#include "duckdb/common/limits.hpp"    // For NumericLimits
#include "duckdb/common/printer.hpp"   // For duckdb::Printer
#include "duckdb/execution/index/fixed_size_allocator.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/data_pointer.hpp" // For Load/Store
#include "duckdb/storage/metadata/metadata_reader.hpp"
#include "duckdb/storage/metadata/metadata_writer.hpp"

// Include for ART index (if/when implemented)
// #include "duckdb/storage/art/art.hpp"
// #include "duckdb/storage/art/art_key.hpp"

#include <iostream> // Added for std::cerr
#include <random>   // For random node selection placeholder

namespace diskann {
namespace core {

// Forward declaration for GetRandomNodeID
common::row_t GetRandomNodeID(::duckdb::AttachedDatabase &db, ::duckdb::FixedSizeAllocator &allocator);

// --- Storage Management Implementation ---
// NOTE: These functions currently act as placeholders or interfaces.
// The actual implementation using the in-memory map resides within
// LmDiskannIndex class methods for now. When persistent mapping (ART)
// is added, the logic will move primarily into these functions,
// taking the ART map reference as a parameter.
// V2 NOTE: StorageManager will manage graph.lmd with custom I/O, not DuckDB allocators/BM for it.
// IShadowStorageService will handle DuckDB interactions for metadata/lookup tables.

// Tries to find the IndexPointer for a given row_id using the map
bool TryGetNodePointer(::duckdb::row_t row_id, ::duckdb::IndexPointer &node_ptr,
                       ::duckdb::AttachedDatabase &db /*, ART* rowid_map */) {
	// FIXME: Implement RowID map lookup (e.g., using ART)
	// This function would encapsulate the ART lookup logic.
	// V2 NOTE: This responsibility moves to IShadowStorageService.
	throw common::NotImplementedException(
	    "Persistent TryGetNodePointer is not implemented and belongs to IShadowStorageService under V2.");
	return false;
}

// Allocates a new block for a node and updates the RowID map
::duckdb::IndexPointer AllocateNode(::duckdb::row_t row_id, ::duckdb::AttachedDatabase &db,
                                    ::duckdb::FixedSizeAllocator &allocator /*, ART* rowid_map */) {
	// FIXME: Implement RowID map insertion
	// This function would encapsulate the allocator->New() call AND the ART
	// insert logic.
	// V2 NOTE: StorageManager::AllocateNodeBlock provides block from graph.lmd. RowID map update by
	// IShadowStorageService.
	throw common::NotImplementedException("Persistent AllocateNode is not implemented. V2 separates block allocation "
	                                      "(StorageManager) and map update (IShadowStorageService).");
	return common::IndexPointer(); // Return default (invalid) common::IndexPointer
}

// Deletes a node from the RowID map and potentially frees the block
void DeleteNodeFromMapAndFreeBlock(::duckdb::row_t row_id, ::duckdb::AttachedDatabase &db,
                                   ::duckdb::FixedSizeAllocator &allocator /*, ART* rowid_map */) {
	// FIXME: Implement RowID map deletion
	// This function would encapsulate the ART delete logic AND the
	// allocator->Free() call.
	// V2 NOTE: StorageManager::FreeBlock adds to graph.lmd free list. RowID map removal by IShadowStorageService.
	throw ::duckdb::NotImplementedException(
	    "Persistent DeleteNodeFromMapAndFreeBlock is not implemented. V2 separates responsibilities.");
}

// Pins block using IndexPointer.
::duckdb::BufferHandle GetNodeBuffer(common::IndexPointer node_ptr, ::duckdb::AttachedDatabase &db,
                                     ::duckdb::FixedSizeAllocator &allocator, bool write_lock) {
	if (node_ptr.Get() == 0) { // Corrected IsValid() check
		throw ::duckdb::IOException("Invalid node pointer provided to GetNodeBuffer.");
	}
	// V2 NOTE: This function will change significantly. It will use StorageManager's internal cache
	// and custom file I/O for graph.lmd, not DuckDB BufferManager for graph.lmd.
	// auto &buffer_manager = ::duckdb::BufferManager::GetBufferManager(db);
	// This function correctly uses the passed allocator reference.
	// return buffer_manager.Pin(allocator.GetBlock(node_ptr));
	throw common::NotImplementedException("GetNodeBuffer needs V2 refactoring for custom file I/O.");
	// To satisfy return type for now, though it's unreachable:
	// return ::duckdb::BufferHandle();
}

// Writes index parameters and state pointers to the metadata block.
void PersistMetadata(common::IndexPointer metadata_ptr, ::duckdb::AttachedDatabase &db,
                     ::duckdb::FixedSizeAllocator &allocator, const LmDiskannMetadata &metadata) {

	if (metadata_ptr.Get() == 0) { // Corrected IsValid() check
		throw ::duckdb::InternalException("Cannot persist LM_DISKANN metadata: metadata pointer is invalid.");
	}
	// V2 NOTE: This function will change. Metadata is managed by IShadowStorageService in diskann_store.duckdb.
	// StorageManager::SaveIndexContents will trigger IShadowStorageService to save its metadata.
	// Direct writing of LmDiskannMetadata using DuckDB allocators for graph.lmd metadata block is not V2.
	// auto &buffer_manager = ::duckdb::BufferManager::GetBufferManager(db);
	// auto handle = buffer_manager.Pin(allocator.GetMetaBlock(metadata_ptr.GetBlockId()));
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
	throw common::NotImplementedException("PersistMetadata needs V2 refactoring for IShadowStorageService.");
}

// Reads index parameters and state pointers from the metadata block.
void LoadMetadata(common::IndexPointer metadata_ptr, ::duckdb::AttachedDatabase &db,
                  ::duckdb::FixedSizeAllocator &allocator, LmDiskannMetadata &metadata_out) {

	// V2 NOTE: This function will change. Metadata is loaded by IShadowStorageService from diskann_store.duckdb.
	// if (!metadata_ptr.IsValid()) {
	//   throw ::duckdb::IOException(
	//       "Cannot load LM_DISKANN index: metadata pointer is invalid.");
	// }
	// auto &buffer_manager = ::duckdb::BufferManager::GetBufferManager(db);
	// auto handle =
	//     buffer_manager.Pin(allocator.GetMetaBlock(metadata_ptr.GetBlockId()));
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
	throw common::NotImplementedException("LoadMetadata needs V2 refactoring for IShadowStorageService.");
}

// Adds row_id to the persistent delete queue (placeholder).
void EnqueueDeletion(common::row_t deleted_row_id, common::IndexPointer &delete_queue_head_ptr,
                     ::duckdb::AttachedDatabase &db, ::duckdb::FixedSizeAllocator &allocator) {
	// V2 NOTE: This responsibility moves to IShadowStorageService.
	// IndexPointer new_queue_entry_ptr = allocator.New();
	// if (!new_queue_entry_ptr.IsValid()) {
	// 	throw ::duckdb::IOException("Failed to allocate block for delete queue entry.");
	// }

	// auto handle = GetNodeBuffer(new_queue_entry_ptr, db, allocator, true);
	// auto data_ptr = handle.Ptr();

	// Store<common::row_t>(deleted_row_id, data_ptr);
	// common::IndexPointer current_head = delete_queue_head_ptr;
	// Store<common::block_id_t>(current_head.GetBlockId(), data_ptr + sizeof(common::row_t));
	// Store<uint32_t>(current_head.GetOffset(), data_ptr + sizeof(common::row_t) + sizeof(common::block_id_t));

	// handle.SetModified();
	// delete_queue_head_ptr = new_queue_entry_ptr;
	throw common::NotImplementedException("EnqueueDeletion belongs to IShadowStorageService under V2.");
}

// Processes the queue during Vacuum (placeholder).
void ProcessDeletionQueue(common::IndexPointer &delete_queue_head_ptr, ::duckdb::AttachedDatabase &db,
                          ::duckdb::FixedSizeAllocator &allocator, const NodeLayoutOffsets &layout,
                          common::idx_t edge_vector_size_bytes) {
	if (delete_queue_head_ptr.Get() != 0) { // Corrected IsValid() check
		throw common::NotImplementedException("ProcessDeletionQueue: Processing deferred deletions is not "
		                                      "implemented fully and needs V2 design.");
	}
}

// Gets a valid entry point row_id (persisted or random - placeholder).
common::row_t GetEntryPointRowId(common::IndexPointer graph_entry_point_ptr, ::duckdb::AttachedDatabase &db,
                                 ::duckdb::FixedSizeAllocator &allocator) {
	if (graph_entry_point_ptr.Get() != 0) { // Corrected IsValid() check
		std::cerr << "Warning: GetEntryPointRowId: Cannot get row_id from pointer yet (V2 requires reading block header)."
		          << std::endl;
		return common::NumericLimits<common::row_t>::Maximum();
	}
	return GetRandomNodeID(db, allocator);
}

// Gets a random node ID from the index (placeholder).
common::row_t GetRandomNodeID(::duckdb::AttachedDatabase &db, ::duckdb::FixedSizeAllocator &allocator) {
	// V2 NOTE: This would query IShadowStorageService for a random RowID from its mapping table.
	// Printer::Warning("GetRandomNodeID not implemented, returning invalid rowid.");
	return ::duckdb::NumericLimits<::duckdb::row_t>::Maximum();
}

} // namespace core
} // namespace diskann
