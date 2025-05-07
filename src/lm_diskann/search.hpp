/**
 * @file search.hpp
 * @brief Declares the core beam search function for LM-DiskANN.
 */
#pragma once

#include "LmDiskannScanState.hpp"   // For LmDiskannScanState
#include "duckdb/common/common.hpp" // For idx_t?
#include "index_config.hpp" // Include for TernaryPlaneBatchView and LmDiskannConfig
#include <cstdint>          // For uint64_t
#include <utility>          // For std::pair
#include <vector>           // For std::vector

namespace duckdb {

class LmDiskannIndex;      // Forward declare to access members/types
struct LmDiskannScanState; // Forward declare scan state struct
struct LmDiskannConfig;

// --- Search Algorithm ---
// Implements the core beam search function for LM-DiskANN.

/**
 * @brief Performs the beam search algorithm to find approximate nearest
 * neighbors.
 *
 * @details This function implements the core graph traversal logic for DiskANN.
 * It starts from one or more entry points, explores the graph using a priority
 * queue (beam) of candidates, and maintains a set of visited nodes to avoid
 * redundant work. The search depth and width are controlled by L_search (from
 * config).
 *
 * @param scan_state The current scan state, containing the query vector,
 * candidate queue, visited set, etc.
 * @param index The LmDiskannIndex instance, providing access to node data and
 * configuration.
 * @param config The index configuration (L_search, metric, dimensions, etc.).
 * @param find_exact_distances If true, calculate exact distances for the final
 * top-k results (expensive).
 */
void PerformSearch(LmDiskannScanState &scan_state, LmDiskannIndex &index,
                   const LmDiskannConfig &config, bool find_exact_distances);

/**
 * @brief Performs a Top-K nearest neighbor search using a batch of ternary
 * encoded vectors.
 * @details Uses the provided batch view to access contiguous plane data.
 * @note Moved from ternary_quantization.hpp
 * @param query Pointer to the query vector (float).
 * @param dims Query vector dimension (for encoding query).
 * @param database_batch Batch view of the pre-encoded database vectors.
 * @param K Number of nearest neighbors to find.
 * @param neighIDs Array of RowIDs corresponding to vectors in database_batch.
 * @param out Output vector to store pairs of <similarity_score, ID>.
 */
void TopKTernarySearch(
    const float *query,
    size_t dims, // Query vector dimension (for encoding query)
    const TernaryPlaneBatchView &database_batch, // Batch of DB vectors
    size_t K,                                    // Number of neighbors to find
    const uint64_t *neighIDs, // IDs corresponding to vectors in database_batch
    std::vector<std::pair<float, uint64_t>>
        &out); // Output: pairs of <similarity_score, ID>

} // namespace duckdb
