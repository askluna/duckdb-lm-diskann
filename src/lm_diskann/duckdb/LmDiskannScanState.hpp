/**
 * @file LmDiskannScanState.hpp
 * @brief Defines the scan state structure used during LM-DiskANN index
 * searches.
 */
#pragma once

#include "../common/types.hpp" // Include common types
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

namespace diskann {
namespace duckdb {

/**
 * @brief Holds the state required during an LM-DiskANN index scan (k-NN
 * search).
 *
 * @details This includes the query vector, parameters like k and L_search,
 *          the candidate priority queue for the beam search, the set of visited
 * nodes, and the final top-k results found so far.
 */
struct LmDiskannScanState : public ::duckdb::IndexScanState {
  /**
   * @brief Constructor for LmDiskannScanState.
   * @param query_vec The query vector.
   * @param k_param The number of nearest neighbors requested (top-k).
   * @param l_search_param The search list size parameter (L_search).
   */
  LmDiskannScanState(const ::duckdb::Vector &query_vec, common::idx_t k_param,
                     common::idx_t l_search_param); // Use common::idx_t
  ~LmDiskannScanState() override = default; // Add virtual destructor override

  ::duckdb::Vector query_vector_storage; // Stores the query vector data.
  const float
      *query_vector_ptr;  // Pointer to the raw float data of the query vector.
  common::idx_t k;        // Number of nearest neighbors requested.
  common::idx_t l_search; // Search list size parameter (L_search).

  // --- State for search process (managed internally by Searcher
  // implementation) --- std::priority_queue<std::pair<float, common::row_t>>
  // candidates; // No longer needed here, Searcher manages internally
  // std::set<common::row_t> visited; // No longer needed here, Searcher manages
  // internally

  // --- Results ---
  // Orchestrator::Search directly populates this vector with the final top-k
  // RowIDs.
  std::vector<common::row_t> result_row_ids;
};

} // namespace duckdb
} // namespace diskann