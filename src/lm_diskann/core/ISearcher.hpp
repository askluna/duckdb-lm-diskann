#pragma once

#include "../common/ann.hpp"
#include "../common/duckdb_types.hpp"
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
	virtual void Search(const float *query_vector, const LmDiskannConfig &config,
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

	/**
	 * @brief Searches for a diverse set of candidate neighbors for a new node
	 * being inserted.
	 *
	 * This is typically used during index construction (e.g., by
	 * GraphManager::AddNode) to find initial connection points before robust
	 * pruning. The search starts from the current_entry_point and explores the
	 * graph.
	 *
	 * @param new_node_vector The vector data of the new node to find neighbors
	 * for.
	 * @param dimensions The dimensionality of the vector.
	 * @param config The current index configuration (e.g., for search list size
	 * parameters).
	 * @param graph_manager Provides access to the graph structure for traversal.
	 * @param num_candidates_to_find The desired number of candidate row_ids to
	 * return.
	 * @param current_entry_point The graph's current entry point to start the
	 * search from. If invalid, the searcher might use a different strategy or
	 * fewer results.
	 * @param candidate_row_ids_out Output vector to store the RowIDs of the found
	 * candidate neighbors.
	 */
	virtual void SearchForInitialCandidates(const float *new_node_vector, common::idx_t dimensions,
	                                        const LmDiskannConfig &config,
	                                        IGraphManager *graph_manager, // Interface to access graph structure
	                                        common::idx_t num_candidates_to_find,
	                                        common::IndexPointer current_entry_point,
	                                        std::vector<common::row_t> &candidate_row_ids_out) = 0;
};

} // namespace core
} // namespace diskann