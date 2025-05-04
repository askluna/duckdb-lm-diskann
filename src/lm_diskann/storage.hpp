#pragma once

#include "duckdb.hpp"
#include "duckdb/execution/index/index_pointer.hpp"
#include "duckdb/storage/buffer/buffer_handle.hpp"
#include "duckdb/storage/index_storage_info.hpp"
#include "config.hpp" // For parameters

namespace duckdb {

class FixedSizeAllocator;
class AttachedDatabase;
class MetadataWriter;
class MetadataReader;

// --- Storage Management Interface ---
// Defines functions for interacting with disk storage, managing blocks,
// persisting metadata, and handling the conceptual RowID map and delete queue.
// These could be part of a class or standalone functions.

// Looks up row_id in the map (placeholder). Returns true if found.
bool TryGetNodePointer(row_t row_id, IndexPointer &node_ptr, AttachedDatabase &db /* Need DB/Allocator context */);

// Allocates block and updates map (placeholder).
IndexPointer AllocateNode(row_t row_id, AttachedDatabase &db, FixedSizeAllocator &allocator /* Need context */);

// Deletes from map and potentially frees block (placeholder).
void DeleteNodeFromMapAndFreeBlock(row_t row_id, AttachedDatabase &db, FixedSizeAllocator &allocator /* Need context */);

// Pins block using IndexPointer.
BufferHandle GetNodeBuffer(IndexPointer node_ptr, AttachedDatabase &db, FixedSizeAllocator &allocator, bool write_lock = false);

// Writes index parameters and state pointers to the metadata block.
void PersistMetadata(IndexPointer metadata_ptr, AttachedDatabase &db, FixedSizeAllocator &allocator,
                     uint8_t format_version, LMDiskannMetricType metric_type,
                     LMDiskannVectorType node_vector_type, LMDiskannEdgeType edge_vector_type_param,
                     idx_t dimensions, uint32_t r, uint32_t l_insert, float alpha, uint32_t l_search,
                     idx_t block_size_bytes, IndexPointer graph_entry_point_ptr,
                     IndexPointer delete_queue_head_ptr /*, IndexPointer rowid_map_root_ptr */);

// Reads index parameters and state pointers from the metadata block.
void LoadMetadata(IndexPointer metadata_ptr, AttachedDatabase &db, FixedSizeAllocator &allocator,
                  uint8_t &format_version, LMDiskannMetricType &metric_type,
                  LMDiskannVectorType &node_vector_type, LMDiskannEdgeType &edge_vector_type_param,
                  idx_t &dimensions, uint32_t &r, uint32_t &l_insert, float &alpha, uint32_t &l_search,
                  idx_t &block_size_bytes, IndexPointer &graph_entry_point_ptr,
                  IndexPointer &delete_queue_head_ptr /*, IndexPointer &rowid_map_root_ptr */);

// Adds row_id to the persistent delete queue (placeholder).
void EnqueueDeletion(row_t deleted_row_id, IndexPointer &delete_queue_head_ptr, AttachedDatabase &db, FixedSizeAllocator &allocator);

// Processes the queue during Vacuum (placeholder).
void ProcessDeletionQueue(IndexPointer &delete_queue_head_ptr, AttachedDatabase &db, FixedSizeAllocator &allocator, const NodeLayoutOffsets& layout, idx_t edge_vector_size_bytes);

// Gets a valid entry point row_id (persisted or random - placeholder).
row_t GetEntryPointRowId(IndexPointer graph_entry_point_ptr, AttachedDatabase &db, FixedSizeAllocator &allocator /*, ART* rowid_map */);

// Gets a random node ID from the index (placeholder).
row_t GetRandomNodeID(AttachedDatabase &db, FixedSizeAllocator &allocator /*, ART* rowid_map */);


} // namespace duckdb
