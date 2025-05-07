/**
 * @file storage_manager.hpp
 * @brief Declares functions related to LM-DiskANN index persistence and storage
 * management.
 */
#pragma once

#include "duckdb.hpp"
#include "duckdb/common/common.hpp" // Include common for row_t
#include "duckdb/execution/index/index_pointer.hpp"
#include "duckdb/storage/buffer/buffer_handle.hpp"
#include "duckdb/storage/index_storage_info.hpp"
#include "index_config.hpp" // Include for enums AND LmDiskannMetadata struct

namespace diskann {
namespace core {
// Forward declarations needed
class FixedSizeAllocator;
class AttachedDatabase;
struct NodeLayoutOffsets;

// --- Storage Management Interface (Placeholders/Signatures) --- //
// These functions currently assume an external map/ART is used.
// They might be moved into LmDiskannIndex or a dedicated storage class later.

/**
 * @brief Tries to find the node pointer for a given RowID.
 * @details Placeholder - needs implementation using ART.
 * @param row_id The RowID to lookup.
 * @param[out] node_ptr Output parameter for the found IndexPointer.
 * @param db Attached database reference (needed for BufferManager access).
 * @param rowid_map Pointer to the ART instance (or similar map).
 * @return True if found, false otherwise.
 */
// bool TryGetNodePointer(row_t row_id, IndexPointer &node_ptr, AttachedDatabase
// &db /* , ART* rowid_map */);

/**
 * @brief Allocates a new node block and updates the RowID map.
 * @details Placeholder - needs implementation using ART.
 * @param row_id The RowID for the new node.
 * @param db Attached database reference.
 * @param allocator The FixedSizeAllocator for node blocks.
 * @param rowid_map Pointer to the ART instance (or similar map).
 * @return IndexPointer to the newly allocated block.
 * @throws ConstraintException if the row_id already exists.
 * @throws InternalException if allocation fails.
 */
// IndexPointer AllocateNode(row_t row_id, AttachedDatabase &db,
// FixedSizeAllocator &allocator /* , ART* rowid_map */);

/**
 * @brief Deletes a node from the RowID map and frees its associated block.
 * @details Placeholder - needs implementation using ART.
 * @param row_id The RowID to delete.
 * @param db Attached database reference.
 * @param allocator The FixedSizeAllocator for node blocks.
 * @param rowid_map Pointer to the ART instance (or similar map).
 */
// void DeleteNodeFromMapAndFreeBlock(row_t row_id, AttachedDatabase &db,
// FixedSizeAllocator &allocator /* , ART* rowid_map */);

/**
 * @brief Pins a block buffer using its IndexPointer.
 * @param node_ptr The IndexPointer to the block.
 * @param db Attached database reference.
 * @param allocator The FixedSizeAllocator used by the index.
 * @param write_lock Whether to acquire a write lock (defaults to false).
 * @return BufferHandle for the pinned block.
 * @throws IOException if the pointer is invalid or pinning fails.
 */
::duckdb::BufferHandle GetNodeBuffer(::duckdb::IndexPointer node_ptr,
                                     ::duckdb::AttachedDatabase &db,
                                     FixedSizeAllocator &allocator,
                                     bool write_lock = false);

// --- Metadata Persistence --- //

/**
 * @brief Persists the index metadata to a specified block.
 * @param metadata_ptr The IndexPointer to the metadata block.
 * @param db The attached database reference.
 * @param allocator The FixedSizeAllocator used by the index.
 * @param metadata The metadata struct to persist.
 */
void PersistMetadata(::duckdb::IndexPointer metadata_ptr,
                     ::duckdb::AttachedDatabase &db,
                     FixedSizeAllocator &allocator,
                     const LmDiskannMetadata &metadata);

/**
 * @brief Loads the index metadata from a specified block.
 * @param metadata_ptr The IndexPointer to the metadata block.
 * @param db The attached database reference.
 * @param allocator The FixedSizeAllocator used by the index.
 * @param[out] metadata The metadata struct to load into.
 */
void LoadMetadata(::duckdb::IndexPointer metadata_ptr,
                  ::duckdb::AttachedDatabase &db, FixedSizeAllocator &allocator,
                  LmDiskannMetadata &metadata);

// --- Delete Queue Management --- //

/**
 * @brief Enqueues a RowID for deletion (placeholder implementation).
 * @details This likely involves allocating a small block for the queue entry
 *          and linking it to the current queue head.
 * @param deleted_row_id The RowID of the node being deleted.
 * @param[in,out] delete_queue_head_ptr Reference to the head pointer of the
 * delete queue (will be updated).
 * @param db The attached database reference.
 * @param allocator The FixedSizeAllocator (or potentially a dedicated one for
 * the queue).
 */
void EnqueueDeletion(::duckdb::row_t deleted_row_id,
                     ::duckdb::IndexPointer &delete_queue_head_ptr,
                     ::duckdb::AttachedDatabase &db,
                     FixedSizeAllocator &allocator);

/**
 * @brief Processes the deletion queue (placeholder implementation).
 * @details Called during VACUUM. Iterates the queue, finds referring nodes,
 *          updates their neighbor lists, and frees queue blocks.
 * @param delete_queue_head_ptr Reference to the head pointer of the delete
 * queue.
 * @param db The attached database reference.
 * @param allocator The FixedSizeAllocator.
 * @param index The index instance (needed for finding referring nodes,
 * potentially via ART).
 */
// Forward declare LmDiskannIndex if needed
// class LmDiskannIndex;
// void ProcessDeletionQueue(IndexPointer &delete_queue_head_ptr,
// AttachedDatabase &db, FixedSizeAllocator &allocator, LmDiskannIndex &index);

// --- Entry Point / Node ID Retrieval --- //
// These also depend on the RowID mapping implementation (ART)

/**
 * @brief Retrieves the RowID stored within a specific node block (if stored
 * there).
 * @details Placeholder - Requires node layout definition and ART integration
 * for reliable inverse lookup.
 * @param node_ptr Pointer to the node block.
 * @param db The attached database reference.
 * @param allocator The FixedSizeAllocator used by the index.
 * @return The RowID found in the node block (or potentially MAX_ROW_ID).
 * @throws IOException if the block cannot be read.
 */
::duckdb::row_t GetEntryPointRowId(::duckdb::IndexPointer node_ptr,
                                   ::duckdb::AttachedDatabase &db,
                                   FixedSizeAllocator &allocator);

/**
 * @brief Gets a random node ID from the index.
 * @details Placeholder - needs ART implementation for efficient random
 * sampling.
 * @param db The attached database reference.
 * @param allocator The FixedSizeAllocator used by the index.
 * @param rowid_map Pointer to the ART instance.
 * @return A random RowID from the index, or MAX_ROW_ID if empty.
 */
// row_t GetRandomNodeID(AttachedDatabase &db, FixedSizeAllocator &allocator /*,
// ART* rowid_map */);

} // namespace core
} // namespace diskann
