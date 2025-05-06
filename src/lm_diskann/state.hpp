/**
 * @file state.hpp
 * @brief Defines the scan state structure used during LM-DiskANN index
 * searches.
 */
#pragma once

#include "duckdb.hpp"
#include "duckdb/common/types/row/row_layout.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/storage/data_pointer.hpp"
#include "duckdb/storage/table/scan_state.hpp"

#include <queue>
#include <set>     // Using std::set for visited
#include <utility> // For std::pair
#include <vector>

namespace duckdb {

/**
 * @brief Holds the state required during an LM-DiskANN index scan (k-NN
 * search).
 *
 * @details This includes the query vector, parameters like k and L_search,
 *          the candidate priority queue for the beam search, the set of visited
 * nodes, and the final top-k results found so far.
 */
struct LmDiskannScanState : public IndexScanState {
  /**
   * @brief Constructor for LmDiskannScanState.
   * @param query_vec The query vector.
   * @param k_param The number of nearest neighbors requested (top-k).
   * @param l_search_param The search list size parameter (L_search).
   */
  LmDiskannScanState(const Vector &query_vec, idx_t k_param,
                     uint32_t l_search_param);

  Vector query_vector; // The query vector itself (keeps data alive).
  const_data_ptr_t query_vector_ptr; // Pointer to the query vector data
                                     // (usually float). Cast before use.
  idx_t k;                           // Number of nearest neighbors requested.
  uint32_t l_search;                 // Search list size parameter (L_search).

  // Type alias for distance/node pairs (using float for distance).
  using dist_node_pair_t = std::pair<float, row_t>;

  // Min-priority queue for candidates (stores {-distance, node_id} to get max
  // distance at top). We use negative distance because priority_queue is a
  // max-heap.
  std::priority_queue<dist_node_pair_t> candidates;

  // Max-priority queue to store the final top-k results found so far {distance,
  // node_id}. This is populated *after* the main search, potentially with exact
  // distances.
  std::priority_queue<dist_node_pair_t> top_candidates;

  // Set of nodes (RowIDs) already visited during the search to avoid
  // cycles/redundancy.
  std::set<row_t> visited;

  // --- Fields below might be used by the IndexScanExecutor --- //
  // std::vector<row_t> result_rowids; // Row IDs of potential candidates -
  // handled by base class? std::vector<float> result_scores; // Corresponding
  // negated similarity scores - handled by base class?
};

} // namespace duckdb
