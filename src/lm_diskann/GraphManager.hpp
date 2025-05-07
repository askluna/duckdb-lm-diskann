/**
 * @file GraphManager.hpp
 * @brief Defines the GraphManager class for managing node allocations,
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

#include "duckdb.hpp"
#include "duckdb/common/types/row/row_layout.hpp" // For row_t ? Check if correct include
#include "duckdb/storage/data_pointer.hpp" // For data_ptr_t, const_data_ptr_t
#include "index_config.hpp"

namespace diskann {
namespace core {

class GraphManager {
public:
  /**
   * @brief Constructor for GraphManager.
   * @details Initializes the internal FixedSizeAllocator for managing node
   * blocks.
   * @param buffer_manager Reference to DuckDB's buffer manager.
   * @param block_size_bytes The size of each block to be managed by the
   * allocator.
   */
  GraphManager(::duckdb::BufferManager &buffer_manager, idx_t block_size_bytes);

  /**
   * @brief Allocates a new block for a node and maps the row ID.
   * @param row_id The row ID to associate with the new node.
   * @return IndexPointer to the newly allocated block.
   * @throws InternalException if the allocator fails to create a new block or
   * if not initialized.
   */
  ::duckdb::IndexPointer AllocateNode(::duckdb::row_t row_id);

  /**
   * @brief Removes a node's mapping and frees its block in the allocator.
   * @param row_id The row ID of the node to delete.
   */
  void FreeNode(::duckdb::row_t row_id);

  /**
   * @brief Tries to retrieve the IndexPointer for a given row ID from the
   * in-memory map.
   * @param row_id The row ID to look up.
   * @param node_ptr Output parameter for the found IndexPointer. If found, it's
   * populated; otherwise, it's cleared.
   * @return True if the row ID was found and its pointer is valid, false
   * otherwise.
   */
  bool TryGetNodePointer(::duckdb::row_t row_id,
                         ::duckdb::IndexPointer &node_ptr) const;

  /**
   * @brief Gets a mutable pointer to the data within a node's block.
   * @param node_ptr The IndexPointer of the node.
   * @return A writable data_ptr_t to the start of the node's block data.
   * @throws IOException if node_ptr is invalid.
   * @throws InternalException if the allocator is not initialized.
   */
  ::duckdb::data_ptr_t GetNodeDataMutable(::duckdb::IndexPointer node_ptr);

  /**
   * @brief Gets a read-only pointer to the data within a node's block.
   * @param node_ptr The IndexPointer of the node.
   * @return A read-only const_data_ptr_t to the start of the node's block data.
   * @throws IOException if node_ptr is invalid.
   * @throws InternalException if the allocator is not initialized.
   */
  ::duckdb::const_data_ptr_t GetNodeData(::duckdb::IndexPointer node_ptr) const;

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
  ::duckdb::FixedSizeAllocator &GetAllocator();

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
  ::duckdb::row_t GetRandomNodeID(::duckdb::RandomEngine &engine);

  // TODO: Add methods for persisting/loading the rowid_to_node_ptr_map_ if it
  // becomes an ART index and needs separate serialization from the main index
  // metadata. Example:
  // void SerializeMap(duckdb::MetadataWriter &writer);
  // void DeserializeMap(duckdb::MetadataReader &reader);

private:
  //! Manages disk blocks for index nodes.
  std::unique_ptr<::duckdb::FixedSizeAllocator> allocator_;
  //! Maps DuckDB row_t to the IndexPointer of the node block.
  std::map<::duckdb::row_t, ::duckdb::IndexPointer> rowid_to_node_ptr_map_;
};

// Forward declare structs if not included via index_config.hpp
// struct NodeLayoutOffsets;
// struct TernaryPlanesView;
// struct MutableTernaryPlanesView;

/**
 * @brief Provides static methods for accessing data within a raw node block
 * buffer.
 * @details Uses a NodeLayoutOffsets struct to determine field locations.
 *          Assumes TERNARY compressed neighbor representation.
 */
class NodeAccessors {
public:
  /**
   * @brief Initializes a raw node block with default values (e.g., 0 neighbor
   * count).
   * @param node_block_ptr Pointer to the start of the node block buffer.
   * @param block_size_bytes Total size of the block (used for potential
   * clearing).
   */
  static void InitializeNodeBlock(::duckdb::data_ptr_t node_block_ptr,
                                  idx_t block_size_bytes);

  /**
   * @brief Gets the number of neighbors currently stored for the node.
   * @param node_block_ptr Pointer to the start of the node block buffer.
   * @return The neighbor count (uint16_t).
   */
  static uint16_t GetNeighborCount(::duckdb::const_data_ptr_t node_block_ptr);

  /**
   * @brief Sets the number of neighbors for the node.
   * @param node_block_ptr Pointer to the start of the node block buffer.
   * @param count The new neighbor count.
   */
  static void SetNeighborCount(::duckdb::data_ptr_t node_block_ptr,
                               uint16_t count);

  /**
   * @brief Gets a constant pointer to the node's full vector data.
   * @param node_block_ptr Pointer to the start of the node block buffer.
   * @param layout The calculated layout offsets for the node block.
   * @return Constant pointer to the vector data.
   */
  static ::duckdb::const_data_ptr_t
  GetNodeVector(::duckdb::const_data_ptr_t node_block_ptr,
                const NodeLayoutOffsets &layout);

  /**
   * @brief Gets a mutable pointer to the node's full vector data.
   * @param node_block_ptr Pointer to the start of the node block buffer.
   * @param layout The calculated layout offsets for the node block.
   * @return Mutable pointer to the vector data.
   */
  static ::duckdb::data_ptr_t
  GetNodeVectorMutable(::duckdb::data_ptr_t node_block_ptr,
                       const NodeLayoutOffsets &layout);

  /**
   * @brief Gets a constant pointer to the array of neighbor RowIDs.
   * @param node_block_ptr Pointer to the start of the node block buffer.
   * @param layout The calculated layout offsets for the node block.
   * @return Constant pointer to the start of the RowID array.
   */
  static ::duckdb::const_data_ptr_t
  GetNeighborIDsPtr(::duckdb::const_data_ptr_t node_block_ptr,
                    const NodeLayoutOffsets &layout);

  /**
   * @brief Gets a mutable pointer to the array of neighbor RowIDs.
   * @param node_block_ptr Pointer to the start of the node block buffer.
   * @param layout The calculated layout offsets for the node block.
   * @return Mutable pointer to the start of the RowID array.
   */
  static ::duckdb::row_t *
  GetNeighborIDsPtrMutable(::duckdb::data_ptr_t node_block_ptr,
                           const NodeLayoutOffsets &layout);

  /**
   * @brief Gets a view of the compressed ternary planes for a specific
   * neighbor.
   * @param node_block_ptr Pointer to the start of the node block buffer.
   * @param layout The calculated layout offsets for the node block.
   * @param neighbor_idx The index (0 to R-1) of the neighbor.
   * @param dimensions The dimensionality of the vectors.
   * @return A TernaryPlanesView pointing to the neighbor's planes.
   */
  static TernaryPlanesView
  GetNeighborTernaryPlanes(::duckdb::const_data_ptr_t node_block_ptr,
                           const NodeLayoutOffsets &layout,
                           uint16_t neighbor_idx, idx_t dimensions);

  /**
   * @brief Gets a mutable view of the compressed ternary planes for a specific
   * neighbor.
   * @param node_block_ptr Pointer to the start of the node block buffer.
   * @param layout The calculated layout offsets for the node block.
   * @param neighbor_idx The index (0 to R-1) of the neighbor.
   * @param dimensions The dimensionality of the vectors.
   * @return A MutableTernaryPlanesView pointing to the neighbor's planes.
   */
  static MutableTernaryPlanesView
  GetNeighborTernaryPlanesMutable(::duckdb::data_ptr_t node_block_ptr,
                                  const NodeLayoutOffsets &layout,
                                  uint16_t neighbor_idx, idx_t dimensions);
};

} // namespace core
} // namespace diskann
