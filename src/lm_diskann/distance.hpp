#pragma once

#include "config.hpp" // Include config for TernaryPlanesView and LMDiskannMetricType
#include "duckdb/common/types.hpp" // For idx_t
#include "duckdb/storage/data_pointer.hpp" // For const_data_ptr_t

#include <cstdint>

// Forward declarations no longer needed as config.hpp is included
// namespace duckdb {
//     enum class LMDiskannMetricType : uint8_t;
// }

namespace duckdb {

// --- Exact Distance Calculation --- //

//! Computes the EXACT distance between two full-precision FLOAT vectors.
//! Used for final re-ranking in LM-DiskANN search.
float ComputeExactDistanceFloat(const float *a_ptr,
                                const float *b_ptr,
                                idx_t dimensions,
                                LMDiskannMetricType metric_type);

// --- Distance / Similarity Calculation --- //

//! Computes the approximate SIMILARITY between a full query vector (float)
//! and a compressed TERNARY neighbor vector using its separate planes.
//! Returns a raw similarity score (higher is better). Used during graph traversal.
//! Note: Implementation requires ternary_quantization.hpp
float ComputeApproxSimilarityTernary(const float *query_float_ptr,
                                     const TernaryPlanesView& neighbor_planes, // Use struct view
                                     idx_t dimensions);

} // namespace duckdb
