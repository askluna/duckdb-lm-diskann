#include "node.hpp"
#include "duckdb/common/types/data_ptr.hpp" // For Load/Store

#include <cstring> // For memset

namespace duckdb {

// --- Node Block Accessor Implementation ---
namespace LMDiskannNodeAccessors {

    // --- Getters (const version) ---
    uint16_t GetNeighborCount(const_data_ptr_t block_ptr) {
        // OFFSET_NEIGHBOR_COUNT is defined in config.cpp, maybe move to config.hpp?
        // For now, assume it's implicitly available or replace with literal 0.
        return Load<uint16_t>(block_ptr + 0 /* OFFSET_NEIGHBOR_COUNT */);
    }

    const_data_ptr_t GetNodeVectorPtr(const_data_ptr_t block_ptr, const NodeLayoutOffsets& layout) {
        return block_ptr + layout.node_vector;
    }

    const row_t* GetNeighborIDsPtr(const_data_ptr_t block_ptr, const NodeLayoutOffsets& layout) {
        return reinterpret_cast<const row_t*>(block_ptr + layout.neighbor_ids);
    }

    const_data_ptr_t GetPosPlanePtr(const_data_ptr_t block_ptr, const NodeLayoutOffsets& layout, uint32_t neighbor_idx, idx_t plane_size_bytes) {
        // Note: Assumes neighbor_idx < R (max neighbors allowed by layout)
        return block_ptr + layout.pos_planes + (neighbor_idx * plane_size_bytes);
    }

    const_data_ptr_t GetNegPlanePtr(const_data_ptr_t block_ptr, const NodeLayoutOffsets& layout, uint32_t neighbor_idx, idx_t plane_size_bytes) {
        // Note: Assumes neighbor_idx < R
        return block_ptr + layout.neg_planes + (neighbor_idx * plane_size_bytes);
    }

    // --- Setters (non-const version) ---
    void SetNeighborCount(data_ptr_t block_ptr, uint16_t count) {
        Store<uint16_t>(count, block_ptr + 0 /* OFFSET_NEIGHBOR_COUNT */);
    }

    data_ptr_t GetNodeVectorPtrMutable(data_ptr_t block_ptr, const NodeLayoutOffsets& layout) {
        return block_ptr + layout.node_vector;
    }

    row_t* GetNeighborIDsPtrMutable(data_ptr_t block_ptr, const NodeLayoutOffsets& layout) {
        return reinterpret_cast<row_t*>(block_ptr + layout.neighbor_ids);
    }

    data_ptr_t GetPosPlanePtrMutable(data_ptr_t block_ptr, const NodeLayoutOffsets& layout, uint32_t neighbor_idx, idx_t plane_size_bytes) {
        // Note: Assumes neighbor_idx < R
        return block_ptr + layout.pos_planes + (neighbor_idx * plane_size_bytes);
    }

    data_ptr_t GetNegPlanePtrMutable(data_ptr_t block_ptr, const NodeLayoutOffsets& layout, uint32_t neighbor_idx, idx_t plane_size_bytes) {
        // Note: Assumes neighbor_idx < R
        return block_ptr + layout.neg_planes + (neighbor_idx * plane_size_bytes);
    }

    // --- Initialization Helper ---
    void InitializeNodeBlock(data_ptr_t block_ptr, idx_t block_size) {
        memset(block_ptr, 0, block_size);
        SetNeighborCount(block_ptr, 0);
        // Initialize any other flags here if added to layout
    }

} // namespace LMDiskannNodeAccessors

} // namespace duckdb
