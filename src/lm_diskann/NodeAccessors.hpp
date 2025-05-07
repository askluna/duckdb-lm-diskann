/**
 * @file NodeAccessors.hpp
 * @brief Provides low-level accessors for reading and writing data within a
 * node's disk block.
 */
#pragma once

#include "duckdb.hpp"
#include "duckdb/common/types/row/row_layout.hpp" // For row_t ? Check if correct include
#include "duckdb/storage/data_pointer.hpp" // For data_ptr_t, const_data_ptr_t
#include "index_config.hpp"                // For NodeLayoutOffsets

#include <cstdint>

namespace duckdb {

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
  static void InitializeNodeBlock(data_ptr_t node_block_ptr,
                                  idx_t block_size_bytes);

  /**
   * @brief Gets the number of neighbors currently stored for the node.
   * @param node_block_ptr Pointer to the start of the node block buffer.
   * @return The neighbor count (uint16_t).
   */
  static uint16_t GetNeighborCount(const_data_ptr_t node_block_ptr);

  /**
   * @brief Sets the number of neighbors for the node.
   * @param node_block_ptr Pointer to the start of the node block buffer.
   * @param count The new neighbor count.
   */
  static void SetNeighborCount(data_ptr_t node_block_ptr, uint16_t count);

  /**
   * @brief Gets a constant pointer to the node's full vector data.
   * @param node_block_ptr Pointer to the start of the node block buffer.
   * @param layout The calculated layout offsets for the node block.
   * @return Constant pointer to the vector data.
   */
  static const_data_ptr_t GetNodeVector(const_data_ptr_t node_block_ptr,
                                        const NodeLayoutOffsets &layout);

  /**
   * @brief Gets a mutable pointer to the node's full vector data.
   * @param node_block_ptr Pointer to the start of the node block buffer.
   * @param layout The calculated layout offsets for the node block.
   * @return Mutable pointer to the vector data.
   */
  static data_ptr_t GetNodeVectorMutable(data_ptr_t node_block_ptr,
                                         const NodeLayoutOffsets &layout);

  /**
   * @brief Gets a constant pointer to the array of neighbor RowIDs.
   * @param node_block_ptr Pointer to the start of the node block buffer.
   * @param layout The calculated layout offsets for the node block.
   * @return Constant pointer to the start of the RowID array.
   */
  static const row_t *GetNeighborIDsPtr(const_data_ptr_t node_block_ptr,
                                        const NodeLayoutOffsets &layout);

  /**
   * @brief Gets a mutable pointer to the array of neighbor RowIDs.
   * @param node_block_ptr Pointer to the start of the node block buffer.
   * @param layout The calculated layout offsets for the node block.
   * @return Mutable pointer to the start of the RowID array.
   */
  static row_t *GetNeighborIDsPtrMutable(data_ptr_t node_block_ptr,
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
  GetNeighborTernaryPlanes(const_data_ptr_t node_block_ptr,
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
  GetNeighborTernaryPlanesMutable(data_ptr_t node_block_ptr,
                                  const NodeLayoutOffsets &layout,
                                  uint16_t neighbor_idx, idx_t dimensions);
};

} // namespace duckdb