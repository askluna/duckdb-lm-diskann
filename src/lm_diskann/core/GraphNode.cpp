#include "GraphNode.hpp"

#include "../common/ann.hpp" // For common::data_ptr_t, common::idx_t, common::row_t
#include "index_config.hpp"  // For NodeLayoutOffsets, TernaryPlanesView, MutableTernaryPlanesView

#include <cstring>   // For std::memset
#include <stdexcept> // For std::runtime_error (example error handling)

namespace diskann {
namespace core {

void GraphNode::InitializeNodeBlock(common::data_ptr_t node_block_ptr, common::idx_t block_size_bytes,
                                    const NodeLayoutOffsets &layout) {
	if (!node_block_ptr) {
		// Or throw an exception, depending on desired error handling
		return;
	}
	// Example: Zero out the block. Actual initialization might be more complex.
	std::memset(node_block_ptr, 0, block_size_bytes);

	// Initialize neighbor count to 0
	if (layout.neighbor_count_offset + sizeof(uint16_t) <= block_size_bytes) {
		*reinterpret_cast<uint16_t *>(node_block_ptr + layout.neighbor_count_offset) = 0;
	} else {
		// Handle error: layout indicates offset is out of bounds
		// This could be an assertion or an exception
	}
}

uint16_t GraphNode::GetNeighborCount(common::const_data_ptr_t node_block_ptr, const NodeLayoutOffsets &layout) {
	if (!node_block_ptr) {
		// Consider error handling, e.g., throw or return a sentinel value
		return 0; // Stub return
	}
	// Assuming layout.node_block_size_bytes is available and checked if necessary
	return *reinterpret_cast<const uint16_t *>(node_block_ptr + layout.neighbor_count_offset);
}

void GraphNode::SetNeighborCount(common::data_ptr_t node_block_ptr, const NodeLayoutOffsets &layout, uint16_t count) {
	if (!node_block_ptr) {
		// Consider error handling
		return; // Stub action
	}
	*reinterpret_cast<uint16_t *>(node_block_ptr + layout.neighbor_count_offset) = count;
}

const unsigned char *GraphNode::GetRawNodeVector(common::const_data_ptr_t node_block_ptr,
                                                 const NodeLayoutOffsets &layout) {
	if (!node_block_ptr) {
		return nullptr; // Stub return
	}
	return node_block_ptr + layout.node_vector_offset;
}

unsigned char *GraphNode::GetRawNodeVectorMutable(common::data_ptr_t node_block_ptr, const NodeLayoutOffsets &layout) {
	if (!node_block_ptr) {
		return nullptr; // Stub return
	}
	return node_block_ptr + layout.node_vector_offset;
}

const common::row_t *GraphNode::GetNeighborIDsPtr(common::const_data_ptr_t node_block_ptr,
                                                  const NodeLayoutOffsets &layout) {
	if (!node_block_ptr) {
		return nullptr; // Stub return
	}
	return reinterpret_cast<const common::row_t *>(node_block_ptr + layout.neighbor_ids_offset);
}

common::row_t *GraphNode::GetNeighborIDsPtrMutable(common::data_ptr_t node_block_ptr, const NodeLayoutOffsets &layout) {
	if (!node_block_ptr) {
		return nullptr; // Stub return
	}
	return reinterpret_cast<common::row_t *>(node_block_ptr + layout.neighbor_ids_offset);
}

TernaryPlanesView GraphNode::GetNeighborTernaryPlanes(common::const_data_ptr_t node_block_ptr,
                                                      const NodeLayoutOffsets &layout, uint16_t neighbor_idx,
                                                      common::idx_t dimensions) {
	if (!node_block_ptr || layout.ternary_edge_size_bytes == 0 || dimensions == 0) {
		return TernaryPlanesView {nullptr, nullptr, 0, 0}; // Stub return for an empty/invalid view
	}

	common::idx_t bytes_per_single_plane = (dimensions + 7) / 8;
	common::idx_t words_per_plane = (bytes_per_single_plane + sizeof(uint64_t) - 1) / sizeof(uint64_t);

	// Assuming neighbor_idx is valid and within bounds (0 to R-1).
	// Proper bound checking would require knowing max degree (R) from config.
	const unsigned char *positive_plane_for_neighbor =
	    (node_block_ptr + layout.neighbor_pos_planes_offset) + (neighbor_idx * bytes_per_single_plane);
	const unsigned char *negative_plane_for_neighbor =
	    (node_block_ptr + layout.neighbor_neg_planes_offset) + (neighbor_idx * bytes_per_single_plane);

	// TODO: Add boundary checks to ensure these pointers are within the allocated block for the node.
	// e.g., positive_plane_for_neighbor + bytes_per_single_plane <= node_block_ptr + layout.total_node_size

	return TernaryPlanesView {positive_plane_for_neighbor, negative_plane_for_neighbor, dimensions, words_per_plane};
}

MutableTernaryPlanesView GraphNode::GetNeighborTernaryPlanesMutable(common::data_ptr_t node_block_ptr,
                                                                    const NodeLayoutOffsets &layout,
                                                                    uint16_t neighbor_idx, common::idx_t dimensions) {
	if (!node_block_ptr || layout.ternary_edge_size_bytes == 0 || dimensions == 0) {
		return MutableTernaryPlanesView {nullptr, nullptr, 0, 0}; // Stub return
	}

	common::idx_t bytes_per_single_plane = (dimensions + 7) / 8;
	common::idx_t words_per_plane = (bytes_per_single_plane + sizeof(uint64_t) - 1) / sizeof(uint64_t);

	unsigned char *positive_plane_for_neighbor =
	    (node_block_ptr + layout.neighbor_pos_planes_offset) + (neighbor_idx * bytes_per_single_plane);
	unsigned char *negative_plane_for_neighbor =
	    (node_block_ptr + layout.neighbor_neg_planes_offset) + (neighbor_idx * bytes_per_single_plane);

	// TODO: Add boundary checks

	return MutableTernaryPlanesView {positive_plane_for_neighbor, negative_plane_for_neighbor, dimensions,
	                                 words_per_plane};
}

} // namespace core
} // namespace diskann
