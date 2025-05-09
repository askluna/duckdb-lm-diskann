/**
 * @file Searcher.hpp
 * @brief Defines the interface and implementation for searching the LM-DiskANN graph.
 */
#pragma once

#include "../common/ann.hpp"
#include "../common/duckdb_types.hpp"
#include "ISearcher.hpp" // Ensuring ISearcher is included
#include "duckdb/common/atomic.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/execution/index/fixed_size_allocator.hpp" // For FixedSizeAllocator
#include "index_config.hpp"                                // For LmDiskannConfig, NodeLayoutOffsets

#include <queue> // For std::priority_queue
#include <vector>

// Forward declarations for interfaces used
namespace diskann {
namespace core {
class IGraphManager;
class IStorageManager;
} // namespace core
} // namespace diskann

namespace diskann {
namespace core {

/**
 * @brief Performs the actual search algorithm on the graph.
 * @details This is a free function implementing the beam search logic.
 * @param query_vector_ptr Pointer to the query vector.
 * @param k_neighbors Number of nearest neighbors to find (k).
 * @param l_search Actual search list size to use (L_search).
 * @param result_row_ids Output vector to store resulting RowIDs.
 * @param config The index configuration.
 * @param node_layout The calculated node layout offsets.
 * @param graph_manager Interface to access graph structure (neighbors) and node vectors.
 * @param storage_manager Interface to access raw node block data.
 * @param final_pass If true, calculate exact distances for top candidates; otherwise, use approximate distances.
 */
void PerformSearch(const float *query_vector_ptr, common::idx_t k_neighbors, common::idx_t l_search,
                   std::vector<common::row_t> &result_row_ids, const LmDiskannConfig &config,
                   const NodeLayoutOffsets &node_layout, IGraphManager &graph_manager, IStorageManager &storage_manager,
                   bool final_pass);

// Potentially move NodeCandidate and SearchResult structs here if they are
// core concepts

class Searcher : public ISearcher {
	public:
	// Constructor now takes IStorageManager
	Searcher(IStorageManager *storage_manager);

	/**
	 * @brief Performs ANN search.
	 * @param query_vector The vector to search for.
	 * @param config The index configuration.
	 * @param graph_manager The graph manager.
	 * @param k_neighbors The number of neighbors to find (k).
	 * @param result_row_ids Output vector of found RowIDs.
	 * @param search_list_size The search list size (L). If 0, uses config default.
	 */
	void Search(const float *query_vector, const LmDiskannConfig &config, IGraphManager *graph_manager,
	            common::idx_t k_neighbors, std::vector<common::row_t> &result_row_ids,
	            common::idx_t search_list_size) override;

	/**
	 * @brief Searches for initial candidates, typically used during index build/insert.
	 * @param new_node_vector The vector data of the new node to find neighbors for.
	 * @param dimensions The dimensionality of the vector.
	 * @param config The current index configuration (e.g., for search list size parameters).
	 * @param graph_manager Provides access to the graph structure for traversal.
	 * @param num_candidates_to_find The desired number of candidate row_ids to return.
	 * @param current_entry_point The graph's current entry point to start the search from.
	 * @param candidate_row_ids_out Output vector to store the RowIDs of the found candidate neighbors.
	 */
	void SearchForInitialCandidates(const float *new_node_vector, common::idx_t dimensions, const LmDiskannConfig &config,
	                                IGraphManager *graph_manager, common::idx_t num_candidates_to_find,
	                                common::IndexPointer current_entry_point,
	                                std::vector<common::row_t> &candidate_row_ids_out) override;

	// Potentially GetInMemorySize if Searcher holds significant state
	common::idx_t GetInMemorySize() const override;

	private:
	IStorageManager *storage_manager_; // Added member
};

// DUPLICATED PerformSearch and Searcher class REMOVED from here

} // namespace core
} // namespace diskann
