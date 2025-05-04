
#include "node.hpp"
#include "duckdb/common/types/data_ptr.hpp" // For Load/Store

#include <cstring> // For memset

namespace duckdb {

// --- Node Block Accessor Implementation ---
namespace LMDiskannNodeAccessors {

    // --- Getters (const version) ---
    uint16_t GetNeighborCount(const_data_ptr_t block_ptr) {
        return Load<uint16_t>(block_ptr + OFFSET_NEIGHBOR_COUNT);
    }

    const_data_ptr_t GetNodeVectorPtr(const_data_ptr_t block_ptr, const NodeLayoutOffsets& layout) {
        return block_ptr + layout.node_vector;
    }

    const row_t* GetNeighborIDsPtr(const_data_ptr_t block_ptr, const NodeLayoutOffsets& layout) {
        return reinterpret_cast<const row_t*>(block_ptr + layout.neighbor_ids);
    }

    const_data_ptr_t GetCompressedNeighborPtr(const_data_ptr_t block_ptr, const NodeLayoutOffsets& layout, uint32_t neighbor_idx, idx_t edge_vector_size_bytes) {
        // Note: Assumes neighbor_idx < GetNeighborCount(block_ptr)
        return block_ptr + layout.compressed_neighbors + (neighbor_idx * edge_vector_size_bytes);
    }

    // --- Setters (non-const version) ---
    void SetNeighborCount(data_ptr_t block_ptr, uint16_t count) {
        Store<uint16_t>(count, block_ptr + OFFSET_NEIGHBOR_COUNT);
    }

    data_ptr_t GetNodeVectorPtrMutable(data_ptr_t block_ptr, const NodeLayoutOffsets& layout) {
        return block_ptr + layout.node_vector;
    }

    row_t* GetNeighborIDsPtrMutable(data_ptr_t block_ptr, const NodeLayoutOffsets& layout) {
        return reinterpret_cast<row_t*>(block_ptr + layout.neighbor_ids);
    }

    data_ptr_t GetCompressedNeighborPtrMutable(data_ptr_t block_ptr, const NodeLayoutOffsets& layout, uint32_t neighbor_idx, idx_t edge_vector_size_bytes) {
        // Note: Assumes neighbor_idx < R (max neighbors)
        return block_ptr + layout.compressed_neighbors + (neighbor_idx * edge_vector_size_bytes);
    }

    // --- Initialization Helper ---
    void InitializeNodeBlock(data_ptr_t block_ptr, idx_t block_size) {
        memset(block_ptr, 0, block_size);
        SetNeighborCount(block_ptr, 0);
        // Initialize any other flags here if added to layout
    }

} // namespace LMDiskannNodeAccessors

} // namespace duckdb
