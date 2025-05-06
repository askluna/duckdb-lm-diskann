/**
 * @file node.cpp
 * @brief Implements low-level node block accessor functions.
 */
#include "LmDiskannNodeAccessors.hpp"
#include "config.hpp" // Include config again here for layout struct definition
#include "duckdb/common/helper.hpp"        // For AlignValue
#include "duckdb/common/limits.hpp"        // For NumericLimits
#include "duckdb/storage/data_pointer.hpp" // For Load/Store
#include "ternary_quantization.hpp"        // For WordsPerPlane

#include <cstring> // For memset

namespace duckdb {

// --- LmDiskannNodeAccessors Method Implementations --- //

void LmDiskannNodeAccessors::InitializeNodeBlock(data_ptr_t node_block_ptr,
                                                 idx_t block_size_bytes) {
  // Zero out the entire block initially
  memset(node_block_ptr, 0, block_size_bytes);
  // Set neighbor count to 0 (although memset already does this)
  SetNeighborCount(node_block_ptr, 0);
}

uint16_t
LmDiskannNodeAccessors::GetNeighborCount(const_data_ptr_t node_block_ptr) {
  // Neighbor count is always at offset 0
  return Load<uint16_t>(node_block_ptr);
}

void LmDiskannNodeAccessors::SetNeighborCount(data_ptr_t node_block_ptr,
                                              uint16_t count) {
  // Neighbor count is always at offset 0
  Store<uint16_t>(count, node_block_ptr);
}

const_data_ptr_t
LmDiskannNodeAccessors::GetNodeVector(const_data_ptr_t node_block_ptr,
                                      const NodeLayoutOffsets &layout) {
  return node_block_ptr + layout.node_vector_offset;
}

data_ptr_t
LmDiskannNodeAccessors::GetNodeVectorMutable(data_ptr_t node_block_ptr,
                                             const NodeLayoutOffsets &layout) {
  return node_block_ptr + layout.node_vector_offset;
}

const row_t *
LmDiskannNodeAccessors::GetNeighborIDsPtr(const_data_ptr_t node_block_ptr,
                                          const NodeLayoutOffsets &layout) {
  return reinterpret_cast<const row_t *>(node_block_ptr +
                                         layout.neighbor_ids_offset);
}

row_t *LmDiskannNodeAccessors::GetNeighborIDsPtrMutable(
    data_ptr_t node_block_ptr, const NodeLayoutOffsets &layout) {
  return reinterpret_cast<row_t *>(node_block_ptr + layout.neighbor_ids_offset);
}

TernaryPlanesView LmDiskannNodeAccessors::GetNeighborTernaryPlanes(
    const_data_ptr_t node_block_ptr, const NodeLayoutOffsets &layout,
    uint16_t neighbor_idx, idx_t dimensions) {
  TernaryPlanesView view;
  view.dimensions = dimensions;
  view.words_per_plane = WordsPerPlane(dimensions);
  idx_t plane_size_bytes = view.words_per_plane * sizeof(uint64_t);

  // Calculate base pointers for the plane arrays
  const_data_ptr_t pos_planes_base =
      node_block_ptr + layout.neighbor_pos_planes_offset;
  const_data_ptr_t neg_planes_base =
      node_block_ptr + layout.neighbor_neg_planes_offset;

  // Calculate pointer for the specific neighbor's plane
  view.positive_plane = pos_planes_base + (neighbor_idx * plane_size_bytes);
  view.negative_plane = neg_planes_base + (neighbor_idx * plane_size_bytes);
  return view;
}

MutableTernaryPlanesView
LmDiskannNodeAccessors::GetNeighborTernaryPlanesMutable(
    data_ptr_t node_block_ptr, const NodeLayoutOffsets &layout,
    uint16_t neighbor_idx, idx_t dimensions) {
  MutableTernaryPlanesView view;
  view.dimensions = dimensions;
  view.words_per_plane = WordsPerPlane(dimensions);
  idx_t plane_size_bytes = view.words_per_plane * sizeof(uint64_t);

  // Calculate base pointers for the plane arrays
  data_ptr_t pos_planes_base =
      node_block_ptr + layout.neighbor_pos_planes_offset;
  data_ptr_t neg_planes_base =
      node_block_ptr + layout.neighbor_neg_planes_offset;

  // Calculate pointer for the specific neighbor's plane
  view.positive_plane = pos_planes_base + (neighbor_idx * plane_size_bytes);
  view.negative_plane = neg_planes_base + (neighbor_idx * plane_size_bytes);
  return view;
}

} // namespace duckdb
