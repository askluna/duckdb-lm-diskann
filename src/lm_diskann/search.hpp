#pragma once

#include "duckdb/common/common.hpp" // For idx_t?
#include "config.hpp" // Include for TernaryPlaneBatchView
#include <vector> // For std::vector
#include <utility> // For std::pair
#include <cstdint> // For uint64_t

namespace duckdb {

class LMDiskannIndex; // Forward declare to access members/types
struct LMDiskannScanState; // Forward declare scan state struct

// --- Search Algorithm ---
// Implements the core graph traversal (beam search) logic.

// Performs the beam search.
// - scan_state: Holds query info, candidates, visited set, and results.
// - index: Provides access to index parameters, storage, distance functions.
// - find_exact_distances: If true, calculates exact distances for top candidates found.
void PerformSearch(LMDiskannScanState &scan_state, LMDiskannIndex &index, bool find_exact_distances);

//! Performs a Top-K nearest neighbor search using a batch of ternary encoded vectors.
//! Uses the provided batch view to access contiguous plane data.
//! Note: Moved from ternary_quantization.hpp
void TopKTernarySearch(const float* query,
                       size_t dims,                  // Query vector dimension (for encoding query)
                       const TernaryPlaneBatchView& database_batch, // Batch of DB vectors
                       size_t K,                     // Number of neighbors to find
                       const uint64_t* neighIDs,     // IDs corresponding to vectors in database_batch
                       std::vector<std::pair<float, uint64_t>>& out); // Output: pairs of <similarity_score, ID>

} // namespace duckdb

