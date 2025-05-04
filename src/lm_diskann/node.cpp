#include "node.hpp"
#include "duckdb/storage/data_pointer.hpp" // For Load/Store
#include "config.hpp" // Include config again here for layout struct definition

#include <cstring> // For memset

namespace duckdb {

// --- Node Block Accessor Implementation ---
namespace LMDiskannNodeAccessors {

    // --- Getters (const version) ---
    uint16_t GetNeighborCount(const_data_ptr_t block_ptr) {
        // Neighbor count is always at the beginning of the block.
        return Load<uint16_t>(block_ptr + 0 /* layout.neighbor_count_offset assumed 0 */);
    }

    const_data_ptr_t GetNodeVector(const_data_ptr_t block_ptr, const NodeLayoutOffsets& layout) {
        return block_ptr + layout.node_vector_offset;
    }

    const row_t* GetNeighborIDs(const_data_ptr_t block_ptr, const NodeLayoutOffsets& layout) {
        return reinterpret_cast<const row_t*>(block_ptr + layout.neighbor_ids_offset);
    }

    const_data_ptr_t GetNeighborPositivePlane(const_data_ptr_t block_ptr, const NodeLayoutOffsets& layout, uint32_t neighbor_idx, idx_t plane_size_bytes) {
        // Note: Assumes neighbor_idx < R (max neighbors allowed by layout)
        return block_ptr + layout.neighbor_pos_planes_offset + (neighbor_idx * plane_size_bytes);
    }

    const_data_ptr_t GetNeighborNegativePlane(const_data_ptr_t block_ptr, const NodeLayoutOffsets& layout, uint32_t neighbor_idx, idx_t plane_size_bytes) {
        // Note: Assumes neighbor_idx < R
        return block_ptr + layout.neighbor_neg_planes_offset + (neighbor_idx * plane_size_bytes);
    }

    // --- Setters (non-const version) ---
    void SetNeighborCount(data_ptr_t block_ptr, uint16_t count) {
        Store<uint16_t>(count, block_ptr + 0 /* layout.neighbor_count_offset assumed 0 */);
    }

    data_ptr_t GetNodeVectorMutable(data_ptr_t block_ptr, const NodeLayoutOffsets& layout) {
        return block_ptr + layout.node_vector_offset;
    }

    row_t* GetNeighborIDsMutable(data_ptr_t block_ptr, const NodeLayoutOffsets& layout) {
        return reinterpret_cast<row_t*>(block_ptr + layout.neighbor_ids_offset);
    }

    data_ptr_t GetNeighborPositivePlaneMutable(data_ptr_t block_ptr, const NodeLayoutOffsets& layout, uint32_t neighbor_idx, idx_t plane_size_bytes) {
        // Note: Assumes neighbor_idx < R
        return block_ptr + layout.neighbor_pos_planes_offset + (neighbor_idx * plane_size_bytes);
    }

    data_ptr_t GetNeighborNegativePlaneMutable(data_ptr_t block_ptr, const NodeLayoutOffsets& layout, uint32_t neighbor_idx, idx_t plane_size_bytes) {
        // Note: Assumes neighbor_idx < R
        return block_ptr + layout.neighbor_neg_planes_offset + (neighbor_idx * plane_size_bytes);
    }

    // --- Initialization Helper ---
    void InitializeNodeBlock(data_ptr_t block_ptr, idx_t block_size) {
        memset(block_ptr, 0, block_size);
        SetNeighborCount(block_ptr, 0);
        // Initialize any other flags here if added to layout
    }

} // namespace LMDiskannNodeAccessors

} // namespace duckdb
