#pragma once

#include "duckdb.hpp"
#include "duckdb/execution/index/index_pointer.hpp"
#include "duckdb/storage/buffer/buffer_handle.hpp"
#include "duckdb/storage/index_storage_info.hpp"
#include "duckdb/common/common.hpp" // Include common for row_t
#include "config.hpp" // Include for enums AND LMDiskannMetadata struct

namespace duckdb {

// Forward declarations needed
class FixedSizeAllocator;
class AttachedDatabase;
struct NodeLayoutOffsets;

// --- Storage Management Interface (Placeholders/Signatures) --- //

// Looks up row_id in the map (placeholder - needs implementation using ART)
bool TryGetNodePointer(row_t row_id, IndexPointer &node_ptr, AttachedDatabase &db /* , ART* rowid_map */);

// Allocates block and updates map (placeholder - needs implementation using ART)
IndexPointer AllocateNode(row_t row_id, AttachedDatabase &db, FixedSizeAllocator &allocator /* , ART* rowid_map */);

// Deletes from map and potentially frees block (placeholder - needs implementation using ART)
void DeleteNodeFromMapAndFreeBlock(row_t row_id, AttachedDatabase &db, FixedSizeAllocator &allocator /* , ART* rowid_map */);

// Pins block using IndexPointer.
BufferHandle GetNodeBuffer(IndexPointer node_ptr, AttachedDatabase &db, FixedSizeAllocator &allocator, bool write_lock = false);

// Writes index metadata struct to the metadata block.
void PersistMetadata(IndexPointer metadata_ptr, AttachedDatabase &db, FixedSizeAllocator &allocator,
                     const LMDiskannMetadata &metadata);

// Reads index metadata struct from the metadata block.
void LoadMetadata(IndexPointer metadata_ptr, AttachedDatabase &db, FixedSizeAllocator &allocator,
                  LMDiskannMetadata &metadata_out);

// Adds row_id to the persistent delete queue (placeholder).
void EnqueueDeletion(row_t deleted_row_id, IndexPointer &delete_queue_head_ptr, AttachedDatabase &db, FixedSizeAllocator &allocator);

// Processes the queue during Vacuum (placeholder).
void ProcessDeletionQueue(IndexPointer &delete_queue_head_ptr, AttachedDatabase &db, FixedSizeAllocator &allocator, const NodeLayoutOffsets& layout, idx_t edge_vector_size_bytes /*, ART* rowid_map */);

// Gets a valid entry point row_id (persisted or random - placeholder).
row_t GetEntryPointRowId(IndexPointer graph_entry_point_ptr, AttachedDatabase &db, FixedSizeAllocator &allocator /*, ART* rowid_map */);

// Gets a random node ID from the index (placeholder - needs ART implementation).
row_t GetRandomNodeID(AttachedDatabase &db, FixedSizeAllocator &allocator /*, ART* rowid_map */);

} // namespace duckdb
