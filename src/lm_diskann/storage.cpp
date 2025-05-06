
#include "storage.hpp"
#include "node.hpp" // For accessors if needed by queue processing

#include "duckdb/common/limits.hpp" // For NumericLimits
#include "duckdb/common/printer.hpp"
#include "duckdb/execution/index/fixed_size_allocator.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/data_pointer.hpp" // For Load/Store
#include "duckdb/storage/metadata/metadata_reader.hpp"
#include "duckdb/storage/metadata/metadata_writer.hpp"

// Include for ART index (if/when implemented)
// #include "duckdb/storage/art/art.hpp"
// #include "duckdb/storage/art/art_key.hpp"

#include <random> // For random node selection placeholder

namespace duckdb {

// --- Storage Management Implementation ---
// NOTE: These functions currently act as placeholders or interfaces.
// The actual implementation using the in-memory map resides within
// LmDiskannIndex class methods for now. When persistent mapping (ART)
// is added, the logic will move primarily into these functions,
// taking the ART map reference as a parameter.

// Tries to find the IndexPointer for a given row_id using the map
bool TryGetNodePointer(row_t row_id, IndexPointer &node_ptr,
                       AttachedDatabase &db /*, ART* rowid_map */) {
  // FIXME: Implement RowID map lookup (e.g., using ART)
  // This function would encapsulate the ART lookup logic.
  throw NotImplementedException(
      "Persistent TryGetNodePointer is not implemented.");
  return false;
}

// Allocates a new block for a node and updates the RowID map
IndexPointer AllocateNode(row_t row_id, AttachedDatabase &db,
                          FixedSizeAllocator &allocator /*, ART* rowid_map */) {
  // FIXME: Implement RowID map insertion
  // This function would encapsulate the allocator->New() call AND the ART
  // insert logic.
  throw NotImplementedException("Persistent AllocateNode is not implemented.");
  return IndexPointer();
}

// Deletes a node from the RowID map and potentially frees the block
void DeleteNodeFromMapAndFreeBlock(
    row_t row_id, AttachedDatabase &db,
    FixedSizeAllocator &allocator /*, ART* rowid_map */) {
  // FIXME: Implement RowID map deletion
  // This function would encapsulate the ART delete logic AND the
  // allocator->Free() call.
  throw NotImplementedException(
      "Persistent DeleteNodeFromMapAndFreeBlock is not implemented.");
}

// Pins block using IndexPointer.
BufferHandle GetNodeBuffer(IndexPointer node_ptr, AttachedDatabase &db,
                           FixedSizeAllocator &allocator, bool write_lock) {
  if (!node_ptr.IsValid()) {
    throw IOException("Invalid node pointer provided to GetNodeBuffer.");
  }
  auto &buffer_manager = BufferManager::GetBufferManager(db);
  // This function correctly uses the passed allocator reference.
  return buffer_manager.Pin(allocator.GetBlock(node_ptr));
}

// Writes index parameters and state pointers to the metadata block.
void PersistMetadata(
    IndexPointer metadata_ptr, AttachedDatabase &db,
    FixedSizeAllocator &allocator, uint8_t format_version,
    LmDiskannMetricType metric_type, LmDiskannVectorType node_vector_type,
    LmDiskannEdgeType edge_vector_type_param, idx_t dimensions, uint32_t r,
    uint32_t l_insert, float alpha, uint32_t l_search, idx_t block_size_bytes,
    IndexPointer graph_entry_point_ptr,
    IndexPointer delete_queue_head_ptr /*, IndexPointer rowid_map_root_ptr */) {

  if (!metadata_ptr.IsValid()) {
    throw InternalException(
        "Cannot persist LM_DISKANN metadata: metadata pointer is invalid.");
  }
  auto &buffer_manager = BufferManager::GetBufferManager(db);
  auto handle =
      buffer_manager.Pin(allocator.GetMetaBlock(metadata_ptr.GetBlockId()));
  MetadataWriter writer(handle.GetFileBuffer(), metadata_ptr.GetOffset());

  // --- Serialize Parameters ---
  writer.Write<uint8_t>(format_version);
  writer.Write<LmDiskannMetricType>(metric_type);
  writer.Write<LmDiskannVectorType>(node_vector_type);
  writer.Write<idx_t>(dimensions);
  writer.Write<uint32_t>(r);
  writer.Write<uint32_t>(l_insert);
  writer.Write<float>(alpha);
  writer.Write<uint32_t>(l_search);
  writer.Write<idx_t>(block_size_bytes);
  // Serialize graph_entry_point_ptr_ and delete_queue_head_ptr_
  writer.Write<IndexPointer>(graph_entry_point_ptr);
  writer.Write<IndexPointer>(delete_queue_head_ptr);
  // Serialize rowid_map_root_ptr_ (Get the root pointer from the ART index)
  // writer.Write<IndexPointer>(rowid_map_root_ptr);

  handle.SetModified();
}

// Reads index parameters and state pointers from the metadata block.
void LoadMetadata(
    IndexPointer metadata_ptr, AttachedDatabase &db,
    FixedSizeAllocator &allocator, uint8_t &format_version,
    LmDiskannMetricType &metric_type, LmDiskannVectorType &node_vector_type,
    LmDiskannEdgeType &edge_vector_type_param, idx_t &dimensions, uint32_t &r,
    uint32_t &l_insert, float &alpha, uint32_t &l_search,
    idx_t &block_size_bytes, IndexPointer &graph_entry_point_ptr,
    IndexPointer
        &delete_queue_head_ptr /*, IndexPointer &rowid_map_root_ptr */) {

  if (!metadata_ptr.IsValid()) {
    throw IOException(
        "Cannot load LM_DISKANN index: metadata pointer is invalid.");
  }
  auto &buffer_manager = BufferManager::GetBufferManager(db);
  auto handle =
      buffer_manager.Pin(allocator.GetMetaBlock(metadata_ptr.GetBlockId()));
  MetadataReader reader(handle.GetFileBuffer(), metadata_ptr.GetOffset());

  // --- Deserialize Parameters ---
  reader.Read<uint8_t>(format_version);
  // Version check should happen in the caller (LmDiskannIndex constructor)
  reader.Read<LmDiskannMetricType>(metric_type);
  reader.Read<LmDiskannVectorType>(node_vector_type);
  reader.Read<idx_t>(dimensions);
  reader.Read<uint32_t>(r);
  reader.Read<uint32_t>(l_insert);
  reader.Read<float>(alpha);
  reader.Read<uint32_t>(l_search);
  reader.Read<idx_t>(block_size_bytes);
  reader.Read<IndexPointer>(graph_entry_point_ptr);
  reader.Read<IndexPointer>(delete_queue_head_ptr);
  // reader.Read<IndexPointer>(rowid_map_root_ptr);
}

// Adds row_id to the persistent delete queue (placeholder).
void EnqueueDeletion(row_t deleted_row_id, IndexPointer &delete_queue_head_ptr,
                     AttachedDatabase &db, FixedSizeAllocator &allocator) {
  // FIXME: Implement persistent delete queue using allocator_
  // This implementation uses the main node allocator, which might waste space.
  // A separate allocator for small queue entries might be better.
  IndexPointer new_queue_entry_ptr = allocator.New(); // Use passed allocator
  if (!new_queue_entry_ptr.IsValid()) {
    throw IOException("Failed to allocate block for delete queue entry.");
  }

  auto handle =
      GetNodeBuffer(new_queue_entry_ptr, db, allocator, true); // Writable
  auto data_ptr = handle.Ptr();

  // Write deleted_row_id and current delete_queue_head_ptr_
  Store<row_t>(deleted_row_id, data_ptr);
  IndexPointer current_head = delete_queue_head_ptr; // Read current head
  // Write IndexPointer (block_id + offset)
  Store<block_id_t>(current_head.GetBlockId(), data_ptr + sizeof(row_t));
  Store<uint32_t>(current_head.GetOffset(),
                  data_ptr + sizeof(row_t) + sizeof(block_id_t));

  handle.SetModified();

  // Update delete_queue_head_ptr_ to point to the new block
  delete_queue_head_ptr = new_queue_entry_ptr;

  // Need to mark metadata as dirty because delete_queue_head_ptr changed
  // This should be handled by the caller (LmDiskannIndex::Delete) setting its
  // dirty flag. Printer::Warning("EnqueueDeletion using main allocator;
  // metadata dirty flag not set here.");
}

// Processes the queue during Vacuum (placeholder).
void ProcessDeletionQueue(IndexPointer &delete_queue_head_ptr,
                          AttachedDatabase &db, FixedSizeAllocator &allocator,
                          const NodeLayoutOffsets &layout,
                          idx_t edge_vector_size_bytes) {
  // FIXME: Implement processing logic during Vacuum
  // Requires iterating through *all* nodes or a reverse index.
  if (delete_queue_head_ptr.IsValid()) {
    Printer::Warning("ProcessDeletionQueue: Processing deferred deletions is "
                     "not implemented.");
    // Conceptual clear after processing:
    // IndexPointer current_ptr = delete_queue_head_ptr;
    // std::vector<IndexPointer> blocks_to_free;
    // while(current_ptr.IsValid()) {
    //     // Read deleted_id and next_ptr from block at current_ptr
    //     // Add current_ptr to blocks_to_free
    //     // current_ptr = next_ptr;
    // }
    // // Free blocks
    // for(auto ptr : blocks_to_free) { allocator.Free(ptr); }
    // delete_queue_head_ptr.Clear();
    // Need to mark metadata as dirty
  }
}

// Gets a valid entry point row_id (persisted or random - placeholder).
row_t GetEntryPointRowId(IndexPointer graph_entry_point_ptr,
                         AttachedDatabase &db,
                         FixedSizeAllocator &allocator /*, ART* rowid_map */) {
  // FIXME: Implement reliable entry point fetching
  if (graph_entry_point_ptr.IsValid()) {
    // Need inverse mapping from IndexPointer to row_id
    // This might involve reading the block pointed to by graph_entry_point_ptr
    // if the row_id isn't stored elsewhere.
    Printer::Warning("GetEntryPointRowId: Cannot get row_id from pointer yet.");
    return -2; // Placeholder indicating valid pointer but unknown rowid
  }
  // Fallback: Get a random node ID
  return GetRandomNodeID(db, allocator /*, rowid_map */);
}

// Gets a random node ID from the index (placeholder).
row_t GetRandomNodeID(AttachedDatabase &db,
                      FixedSizeAllocator &allocator /*, ART* rowid_map */) {
  // FIXME: Implement random node selection using RowID map iteration/sampling
  // Requires iterating the ART index or sampling keys.
  // Placeholder: Return invalid rowid
  // Printer::Warning("GetRandomNodeID not implemented, returning invalid
  // rowid.");
  return NumericLimits<row_t>::Maximum();
}

} // namespace duckdb
