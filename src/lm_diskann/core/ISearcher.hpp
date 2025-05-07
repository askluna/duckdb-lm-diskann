#pragma once

#include "../common/types.hpp"
#include "index_config.hpp" // For LmDiskannConfig
#include <vector>

namespace diskann {
namespace core {

class IGraphManager; // Forward declaration for graph access

class ISearcher {
public:
  virtual ~ISearcher() = default;

  /**
   * @brief Performs a k-nearest neighbor search on the graph.
   *
   * @param query_vector The vector to search for.
   * @param config The current index configuration (contains L_search, etc.).
   * @param graph_manager Provides access to graph data (nodes, neighbors, entry
   * point).
   * @param k_neighbors The number of nearest neighbors to retrieve.
   * @param result_row_ids Output vector to store the RowIDs of the k nearest
   * neighbors.
   * @param search_list_size Override for L_search from config, if > 0.
   */
  virtual void
  Search(const float *query_vector, const LmDiskannConfig &config,
         IGraphManager *graph_manager, // Interface to access graph structure
         common::idx_t k_neighbors, std::vector<common::row_t> &result_row_ids,
         common::idx_t search_list_size = 0 // Optional override for L_search
         ) = 0;

  // Potentially other search-related methods, e.g., for batched search
  // or search with filtering if that becomes a core responsibility.

  /**
   * @brief Returns the estimated in-memory size used by the searcher (if any).
   * @return Size in bytes.
   */
  virtual common::idx_t GetInMemorySize() const = 0;
};

} // namespace core
} // namespace diskann