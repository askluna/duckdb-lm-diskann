/**
 * @file GraphManager.hpp
 * @brief Defines the GraphManager class for managing node allocations,
 *        RowID mappings, and raw node data access.
 */
#pragma once

#include "../common/ann.hpp"
#include "../common/duckdb_types.hpp"
#include "IGraphManager.hpp" // Inherit from the interface
#include "index_config.hpp"  // For LmDiskannConfig, NodeLayoutOffsets, TernaryPlanesView, MutableTernaryPlanesView

// Forward declare interfaces for dependencies
namespace diskann {
namespace core {
class IStorageManager; // Forward declaration
class ISearcher;       // Forward declaration
} // namespace core
} // namespace diskann

// DuckDB includes previously needed for BufferManager/FixedSizeAllocator are
// removed as these are no longer direct responsibilities of GraphManager.

#include <map>
#include <memory> // For std::unique_ptr if needed by future implementations
#include <vector>

namespace diskann {
namespace core {

// Forward declare for LmDiskannConfig
struct LmDiskannConfig;
struct NodeLayoutOffsets;
struct TernaryPlanesView;        // Assuming this is defined elsewhere or will be.
struct MutableTernaryPlanesView; // Defined in index_config.hpp

class GraphNode {
	public:
	/// @brief Initializes a new node block (e.g., zeroing memory, setting default
	/// neighbor count).
	static void InitializeNodeBlock(common::data_ptr_t node_block_ptr, common::idx_t block_size_bytes,
	                                const NodeLayoutOffsets &layout);

	/// @brief Gets the current count of neighbors for the node.
	static uint16_t GetNeighborCount(common::const_data_ptr_t node_block_ptr, const NodeLayoutOffsets &layout);

	/// @brief Sets the count of neighbors for the node.
	static void SetNeighborCount(common::data_ptr_t node_block_ptr, const NodeLayoutOffsets &layout, uint16_t count);

	/// @brief Gets a const pointer to the start of the node's raw vector data
	/// (e.g., float or int8).
	static const unsigned char *GetRawNodeVector(common::const_data_ptr_t node_block_ptr,
	                                             const NodeLayoutOffsets &layout);

	/// @brief Gets a mutable pointer to the start of the node's raw vector data.
	static unsigned char *GetRawNodeVectorMutable(common::data_ptr_t node_block_ptr, const NodeLayoutOffsets &layout);

	/// @brief Gets a const pointer to the start of the node's neighbor ID list
	/// (array of common::row_t).
	static const common::row_t *GetNeighborIDsPtr(common::const_data_ptr_t node_block_ptr,
	                                              const NodeLayoutOffsets &layout);

	/// @brief Gets a mutable pointer to the start of the node's neighbor ID list.
	static common::row_t *GetNeighborIDsPtrMutable(common::data_ptr_t node_block_ptr, const NodeLayoutOffsets &layout);

	// --- Ternary Quantization Specific Accessors ---
	// These are only relevant if layout indicates ternary data is present (e.g.,
	// layout.ternary_edge_size_bytes > 0)

	/// @brief Gets a view of the ternary planes for a specific neighbor edge.
	static TernaryPlanesView GetNeighborTernaryPlanes(common::const_data_ptr_t node_block_ptr,
	                                                  const NodeLayoutOffsets &layout,
	                                                  uint16_t neighbor_idx,     // Index of the neighbor edge (0 to R-1)
	                                                  common::idx_t dimensions); // Vector dimensions from config

	/// @brief Gets a mutable view of the ternary planes for a specific neighbor
	/// edge.
	static MutableTernaryPlanesView
	GetNeighborTernaryPlanesMutable(common::data_ptr_t node_block_ptr, const NodeLayoutOffsets &layout,
	                                uint16_t neighbor_idx,     // Index of the neighbor edge
	                                common::idx_t dimensions); // Vector dimensions from config
};

} // namespace core
} // namespace diskann
