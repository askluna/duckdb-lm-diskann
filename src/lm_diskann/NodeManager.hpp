/**
 * @file NodeManager.hpp
 * @brief Defines the NodeManager class for managing node allocations,
 *        RowID mappings, and raw node data access.
 */
#pragma once

#include "duckdb.hpp"
#include "duckdb/common/limits.hpp" // For NumericLimits in GetRandomNodeID
#include "duckdb/common/random_engine.hpp" // For RandomEngine in GetRandomNodeID
#include "duckdb/execution/index/fixed_size_allocator.hpp"
#include "duckdb/execution/index/index_pointer.hpp"
#include "duckdb/storage/buffer_manager.hpp" // Required for allocator initialization

#include <map>
#include <memory> // For unique_ptr
#include <random> // For GetRandomNodeID's distribution

namespace duckdb {

class NodeManager {
public:
  /**
   * @brief Constructor for NodeManager.
   * @details Initializes the internal FixedSizeAllocator for managing node
   * blocks.
   * @param buffer_manager Reference to DuckDB's buffer manager.
   * @param block_size_bytes The size of each block to be managed by the
   * allocator.
   */
  NodeManager(BufferManager &buffer_manager, idx_t block_size_bytes);

  /**
   * @brief Allocates a new block for a node and maps the row ID.
   * @param row_id The row ID to associate with the new node.
   * @return IndexPointer to the newly allocated block.
   * @throws InternalException if the allocator fails to create a new block or
   * if not initialized.
   */
  IndexPointer AllocateNode(row_t row_id);

  /**
   * @brief Removes a node's mapping and frees its block in the allocator.
   * @param row_id The row ID of the node to delete.
   */
  void FreeNode(row_t row_id);

  /**
   * @brief Tries to retrieve the IndexPointer for a given row ID from the
   * in-memory map.
   * @param row_id The row ID to look up.
   * @param node_ptr Output parameter for the found IndexPointer. If found, it's
   * populated; otherwise, it's cleared.
   * @return True if the row ID was found and its pointer is valid, false
   * otherwise.
   */
  bool TryGetNodePointer(row_t row_id, IndexPointer &node_ptr) const;

  /**
   * @brief Gets a mutable pointer to the data within a node's block.
   * @param node_ptr The IndexPointer of the node.
   * @return A writable data_ptr_t to the start of the node's block data.
   * @throws IOException if node_ptr is invalid.
   * @throws InternalException if the allocator is not initialized.
   */
  data_ptr_t GetNodeDataMutable(IndexPointer node_ptr);

  /**
   * @brief Gets a read-only pointer to the data within a node's block.
   * @param node_ptr The IndexPointer of the node.
   * @return A read-only const_data_ptr_t to the start of the node's block data.
   * @throws IOException if node_ptr is invalid.
   * @throws InternalException if the allocator is not initialized.
   */
  const_data_ptr_t GetNodeData(IndexPointer node_ptr) const;

  /**
   * @brief Resets the allocator and clears the RowID map.
   * @details This is typically used during the CommitDrop operation of the
   * index.
   */
  void Reset();

  /**
   * @brief Gets the estimated in-memory size of structures managed by this
   * class.
   * @details This includes the allocator's in-memory footprint and the RowID
   * map overhead.
   * @return Estimated size in bytes.
   */
  idx_t GetInMemorySize() const;

  /**
   * @brief Gets the underlying FixedSizeAllocator.
   * @return Reference to the FixedSizeAllocator.
   * @throws InternalException if the allocator is not initialized.
   */
  FixedSizeAllocator &GetAllocator();

  /**
   * @brief Gets the number of distinct nodes currently mapped.
   * @return The number of nodes in the rowid_to_node_ptr_map_.
   */
  idx_t GetNodeCount() const;

  /**
   * @brief Gets a random RowID from the current set of nodes.
   * @details This is a placeholder and uses inefficient map iteration. Should
   * be replaced with ART sampling if available.
   * @param engine A random engine to use for selecting a random index.
   * @return A random row ID from the map, or NumericLimits<row_t>::Maximum() if
   * the map is empty.
   */
  row_t GetRandomNodeID(RandomEngine &engine);

  // TODO: Add methods for persisting/loading the rowid_to_node_ptr_map_ if it
  // becomes an ART index and needs separate serialization from the main index
  // metadata. Example:
  // void SerializeMap(duckdb::MetadataWriter &writer);
  // void DeserializeMap(duckdb::MetadataReader &reader);

private:
  //! Manages disk blocks for index nodes.
  unique_ptr<FixedSizeAllocator> allocator_;
  //! Maps DuckDB row_t to the IndexPointer of the node block.
  std::map<row_t, IndexPointer> rowid_to_node_ptr_map_;
};

} // namespace duckdb