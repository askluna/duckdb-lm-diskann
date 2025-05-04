#pragma once

#include "duckdb.hpp"
#include "config.hpp" // For NodeLayoutOffsets
#include "duckdb/storage/data_pointer.hpp" // For data_ptr_t

#include <cstdint>

namespace duckdb {

// --- Node Block Accessors ---
// Provides low-level, type-safe functions to read/write data within a raw node block buffer.
// These functions rely on the pre-calculated NodeLayoutOffsets.
namespace LMDiskannNodeAccessors {

    // --- Getters (const version) ---

    // Reads the neighbor count (uint16_t) from the start of the block.
    uint16_t GetNeighborCount(const_data_ptr_t block_ptr);

    // Returns a pointer to the start of the node's full vector data.
    const_data_ptr_t GetNodeVectorPtr(const_data_ptr_t block_ptr, const NodeLayoutOffsets& layout);

    // Returns a pointer to the start of the neighbor row_t ID array.
    const row_t* GetNeighborIDsPtr(const_data_ptr_t block_ptr, const NodeLayoutOffsets& layout);

    // Returns a pointer to the start of the specified neighbor's positive ternary plane data.
    // Note: No bounds check here for performance; caller must ensure neighbor_idx is valid.
    const_data_ptr_t GetPosPlanePtr(const_data_ptr_t block_ptr, const NodeLayoutOffsets& layout, uint32_t neighbor_idx, idx_t plane_size_bytes);

    // Returns a pointer to the start of the specified neighbor's negative ternary plane data.
    // Note: No bounds check here for performance; caller must ensure neighbor_idx is valid.
    const_data_ptr_t GetNegPlanePtr(const_data_ptr_t block_ptr, const NodeLayoutOffsets& layout, uint32_t neighbor_idx, idx_t plane_size_bytes);

    // --- Setters (non-const version) ---

    // Writes the neighbor count (uint16_t) to the start of the block.
    void SetNeighborCount(data_ptr_t block_ptr, uint16_t count);

    // Returns a writable pointer to the start of the node's full vector data.
    data_ptr_t GetNodeVectorPtrMutable(data_ptr_t block_ptr, const NodeLayoutOffsets& layout);

    // Returns a writable pointer to the start of the neighbor row_t ID array.
    row_t* GetNeighborIDsPtrMutable(data_ptr_t block_ptr, const NodeLayoutOffsets& layout);

    // Returns a writable pointer to the start of the specified neighbor's positive ternary plane data.
    // Note: No bounds check here for performance; caller must ensure neighbor_idx is valid.
    data_ptr_t GetPosPlanePtrMutable(data_ptr_t block_ptr, const NodeLayoutOffsets& layout, uint32_t neighbor_idx, idx_t plane_size_bytes);

    // Returns a writable pointer to the start of the specified neighbor's negative ternary plane data.
    // Note: No bounds check here for performance; caller must ensure neighbor_idx is valid.
    data_ptr_t GetNegPlanePtrMutable(data_ptr_t block_ptr, const NodeLayoutOffsets& layout, uint32_t neighbor_idx, idx_t plane_size_bytes);

    // --- Initialization Helper ---

    // Zeroes out a block and sets initial neighbor count to 0.
    void InitializeNodeBlock(data_ptr_t block_ptr, idx_t block_size);

} // namespace LMDiskannNodeAccessors

} // namespace duckdb
