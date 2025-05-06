/**
 * @file NodeAccessors.cpp
 * @brief Implements low-level accessors for node data blocks.
 */
#include "NodeAccessors.hpp"
#include "duckdb/common/types.hpp"  // For data_ptr_t, const_data_ptr_t
#include "ternary_quantization.hpp" // For WordsPerPlane
#include <cstring>                  // For memset, memcpy

namespace duckdb {

void NodeAccessors::InitializeNodeBlock(data_ptr_t node_block_ptr,
                                        idx_t block_size_bytes) {
  // For now, just ensure the neighbor count is 0. The rest can be
  // uninitialized. If a full clear is needed later: memset(node_block_ptr, 0,
  // block_size_bytes);
  if (block_size_bytes < sizeof(uint16_t)) {
    // This should not happen if block_size_bytes is correctly calculated
    return;
  }
  SetNeighborCount(node_block_ptr, 0);
}

uint16_t NodeAccessors::GetNeighborCount(const_data_ptr_t node_block_ptr) {
  return Load<uint16_t>(node_block_ptr +
                        NodeLayoutOffsets().neighbor_count_offset);
}

void NodeAccessors::SetNeighborCount(data_ptr_t node_block_ptr,
                                     uint16_t count) {
  Store<uint16_t>(count,
                  node_block_ptr + NodeLayoutOffsets().neighbor_count_offset);
}

const_data_ptr_t NodeAccessors::GetNodeVector(const_data_ptr_t node_block_ptr,
                                              const NodeLayoutOffsets &layout) {
  return node_block_ptr + layout.node_vector_offset;
}

data_ptr_t
NodeAccessors::GetNodeVectorMutable(data_ptr_t node_block_ptr,
                                    const NodeLayoutOffsets &layout) {
  return node_block_ptr + layout.node_vector_offset;
}

const row_t *NodeAccessors::GetNeighborIDsPtr(const_data_ptr_t node_block_ptr,
                                              const NodeLayoutOffsets &layout) {
  return reinterpret_cast<const row_t *>(node_block_ptr +
                                         layout.neighbor_ids_offset);
}

row_t *
NodeAccessors::GetNeighborIDsPtrMutable(data_ptr_t node_block_ptr,
                                        const NodeLayoutOffsets &layout) {
  return reinterpret_cast<row_t *>(node_block_ptr + layout.neighbor_ids_offset);
}

TernaryPlanesView NodeAccessors::GetNeighborTernaryPlanes(
    const_data_ptr_t node_block_ptr, const NodeLayoutOffsets &layout,
    uint16_t neighbor_idx, idx_t dimensions) {
  TernaryPlanesView planes_view;
  planes_view.dimensions = dimensions;
  planes_view.words_per_plane =
      WordsPerPlane(dimensions); // From ternary_quantization.hpp

  idx_t plane_size_bytes = planes_view.words_per_plane * sizeof(uint64_t);

  const_data_ptr_t pos_planes_start =
      node_block_ptr + layout.neighbor_pos_planes_offset;
  planes_view.positive_plane =
      pos_planes_start + (neighbor_idx * plane_size_bytes);

  const_data_ptr_t neg_planes_start =
      node_block_ptr + layout.neighbor_neg_planes_offset;
  planes_view.negative_plane =
      neg_planes_start + (neighbor_idx * plane_size_bytes);

  return planes_view;
}

MutableTernaryPlanesView NodeAccessors::GetNeighborTernaryPlanesMutable(
    data_ptr_t node_block_ptr, const NodeLayoutOffsets &layout,
    uint16_t neighbor_idx, idx_t dimensions) {
  MutableTernaryPlanesView planes_view;
  planes_view.dimensions = dimensions;
  planes_view.words_per_plane =
      WordsPerPlane(dimensions); // From ternary_quantization.hpp

  idx_t plane_size_bytes = planes_view.words_per_plane * sizeof(uint64_t);

  data_ptr_t pos_planes_start =
      node_block_ptr + layout.neighbor_pos_planes_offset;
  planes_view.positive_plane =
      pos_planes_start + (neighbor_idx * plane_size_bytes);

  data_ptr_t neg_planes_start =
      node_block_ptr + layout.neighbor_neg_planes_offset;
  planes_view.negative_plane =
      neg_planes_start + (neighbor_idx * plane_size_bytes);

  return planes_view;
}

} // namespace duckdb