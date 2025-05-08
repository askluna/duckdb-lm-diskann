/**
 * @file Searcher.cpp
 * @brief Implements the beam search algorithm for LM-DiskANN.
 */

#include "Searcher.hpp"

// Core interfaces and types
#include "../common/ann.hpp"
#include "../common/duckdb_types.hpp"
// #include "../db/LmDiskannScanState.hpp" // REMOVED: LmDiskannScanState is not used directly by core searcher
#include "IGraphManager.hpp"
#include "IStorageManager.hpp"
#include "distance.hpp" // For ComputeExactDistanceFloat and ConvertRawVectorToFloat (assuming moved here)
#include "index_config.hpp"

// Standard library
#include <algorithm> // For std::max
#include <cstring>   // For memcpy
#include <iostream>
#include <queue>
#include <set>
#include <vector>

namespace diskann {
namespace core {

// Assuming INVALID_INDEX_POINTER definition is moved to common/types.hpp or similar accessible header.
// If not, define it here:
// const common::IndexPointer INVALID_INDEX_POINTER = common::IndexPointer();

struct NodeCandidate {
	float distance;
	common::row_t row_id;

	bool operator>(const NodeCandidate &other) const {
		return distance > other.distance;
	}
};

// TODO: Move ConvertRawVectorToFloat declaration/definition to distance.hpp/cpp or a new core utility file.
// Placeholder signature assuming it's moved:
// bool ConvertRawVectorToFloat(common::const_data_ptr_t raw_vector_data, float *float_vector_out,
//                              const LmDiskannConfig &config, const NodeLayoutOffsets &node_layout);

void PerformSearch(const float *query_vector_ptr, common::idx_t k_neighbors, common::idx_t l_search_param,
                   std::vector<common::row_t> &result_row_ids, const LmDiskannConfig &config,
                   const NodeLayoutOffsets &node_layout, IGraphManager &graph_manager, IStorageManager &storage_manager,
                   bool final_pass) {

	// if (scan_state.l_search == 0 || scan_state.l_search < scan_state.k) {
	// 	scan_state.l_search = std::max(static_cast<common::idx_t>(config.l_search), scan_state.k);
	// }
	// Use l_search_param directly as it's already been determined by the caller (Searcher::Search)
	common::idx_t current_l_search = l_search_param;
	if (current_l_search == 0) { // Fallback if not set properly or too small
		current_l_search = config.l_search;
	}
	if (current_l_search < k_neighbors) { // Ensure L_search is at least K
		current_l_search = k_neighbors;
	}

	// Searcher manages these internally
	std::set<common::row_t> visited_nodes;
	std::priority_queue<NodeCandidate, std::vector<NodeCandidate>, std::greater<NodeCandidate>>
	    top_candidates; // Min-heap for L candidates

	std::priority_queue<NodeCandidate, std::vector<NodeCandidate>, std::greater<NodeCandidate>> candidate_beam;

	// --- Initialization ---
	common::row_t entry_point_rowid = graph_manager.GetEntryPointRowId();
	common::IndexPointer entry_point_ptr = graph_manager.GetEntryPointPointer();

	// TODO: Check visibility/tombstone for entry point via IShadowStorageService?
	if (entry_point_rowid == common::NumericLimits<common::row_t>::Maximum() ||
	    entry_point_ptr.Get() == 0) { // Corrected IsValid() check
		return;
	}

	// TODO: Block fetching should check shadow store first (needs IShadowStorageService param)
	common::const_data_ptr_t entry_node_block = storage_manager.GetNodeBlockData(entry_point_ptr);
	if (!entry_node_block) {
		return;
	}
	// Replace NodeAccessors::GetRawNodeVector
	const unsigned char *entry_raw_vector = entry_node_block + node_layout.node_vector_offset;

	// Check if pointer is within reasonable bounds of the block (basic sanity check)
	// This assumes block size is known or calculable. For now, skipping detailed bounds check.

	std::vector<float> entry_float_vector_storage(config.dimensions);
	// Call assuming ConvertRawVectorToFloat is a free function in diskann::core
	if (!diskann::common::ConvertRawVectorToFloat(entry_raw_vector, entry_float_vector_storage.data(), config.dimensions,
	                                              config.node_vector_type)) {
		// Log error or handle conversion failure
		return;
	}

	float dist_to_entry = diskann::core::ComputeExactDistanceFloat(query_vector_ptr, entry_float_vector_storage.data(),
	                                                               config.dimensions, config.metric_type);

	candidate_beam.push({dist_to_entry, entry_point_rowid});
	visited_nodes.insert(entry_point_rowid);
	top_candidates.push({dist_to_entry, entry_point_rowid});

	// --- Beam Search Loop ---
	float lowest_dist_in_results = dist_to_entry;

	while (!candidate_beam.empty()) {
		NodeCandidate current_candidate = candidate_beam.top();
		candidate_beam.pop();

		if (current_candidate.distance > lowest_dist_in_results && top_candidates.size() >= current_l_search) {
			// Beam cutoff heuristic (can be refined)
		}

		// TODO: Architectural issue: How to get IndexPointer from row_id?
		//       V2 Plan says IShadowStorageService handles map. PerformSearch needs access.
		//       Using graph_manager.TryGetNodePointer for now, but flagged by linter & V2 plan.
		common::IndexPointer current_node_ptr;
		if (!graph_manager.TryGetNodePointer(current_candidate.row_id, current_node_ptr)) {
			continue;
		}

		// TODO: Check visibility/tombstone for current_node_ptr via IShadowStorageService?

		std::vector<common::row_t> neighbor_row_ids;
		if (!graph_manager.GetNeighbors(current_node_ptr, neighbor_row_ids)) {
			continue;
		}

		for (common::row_t neighbor_rowid : neighbor_row_ids) {
			if (neighbor_rowid == common::NumericLimits<common::row_t>::Maximum())
				continue;

			if (visited_nodes.count(neighbor_rowid)) {
				continue;
			}

			visited_nodes.insert(neighbor_rowid);

			float dist_to_neighbor;
			// TODO: Architectural issue: How to get IndexPointer from row_id?
			//       V2 Plan says IShadowStorageService handles map. PerformSearch needs access.
			//       Using graph_manager.TryGetNodePointer for now.
			common::IndexPointer neighbor_ptr;
			if (!graph_manager.TryGetNodePointer(neighbor_rowid, neighbor_ptr)) {
				continue;
			}

			// TODO: Check visibility/tombstone for neighbor_ptr via IShadowStorageService?

			// TODO: Block fetching should check shadow store first (needs IShadowStorageService param)
			common::const_data_ptr_t neighbor_node_block = storage_manager.GetNodeBlockData(neighbor_ptr);
			if (!neighbor_node_block) {
				continue;
			}

			// Replace NodeAccessors::GetRawNodeVector
			const unsigned char *neighbor_raw_vector = neighbor_node_block + node_layout.node_vector_offset;

			std::vector<float> neighbor_float_vector_storage(config.dimensions);
			// Call assuming ConvertRawVectorToFloat is a free function in diskann::core
			if (!diskann::common::ConvertRawVectorToFloat(neighbor_raw_vector, neighbor_float_vector_storage.data(),
			                                              config.dimensions, config.node_vector_type)) {
				continue;
			}

			// TODO: Re-introduce approximate distance calculation using ternary planes if config.use_ternary_quantization is
			// true. This requires accessing compressed edge data, potentially from current_node_block, not
			// neighbor_node_block. Needs NodeLayoutOffsets and likely a replacement for
			// NodeAccessors::GetNeighborTernaryPlanes. Using exact distance for now regardless of config flag.
			dist_to_neighbor = diskann::core::ComputeExactDistanceFloat(
			    query_vector_ptr, neighbor_float_vector_storage.data(), config.dimensions, config.metric_type);

			if (top_candidates.size() < current_l_search || dist_to_neighbor < lowest_dist_in_results) {
				candidate_beam.push({dist_to_neighbor, neighbor_rowid});
				top_candidates.push({dist_to_neighbor, neighbor_rowid});

				if (top_candidates.size() > current_l_search) {
					top_candidates.pop();
				}
				if (!top_candidates.empty()) {
					lowest_dist_in_results = top_candidates.top().distance;
				}
			}
		}
	}

	if (final_pass) {
		// Re-ranking logic: This part needs access to node data and ConvertRawVectorToFloat
		std::priority_queue<NodeCandidate, std::vector<NodeCandidate>, std::greater<NodeCandidate>> final_results_min_heap;
		std::vector<NodeCandidate> temp_top_candidates_vec;
		while (!top_candidates.empty()) {
			temp_top_candidates_vec.push_back(top_candidates.top());
			top_candidates.pop();
		}

		std::vector<float> node_float_vector_storage(config.dimensions);
		for (const auto &candidate : temp_top_candidates_vec) {
			// TODO: Architectural issue: How to get IndexPointer from row_id?
			common::IndexPointer node_ptr;
			if (!graph_manager.TryGetNodePointer(candidate.row_id, node_ptr))
				continue;

			// TODO: Check visibility/tombstone for node_ptr via IShadowStorageService?

			// TODO: Block fetching should check shadow store first.
			common::const_data_ptr_t node_block = storage_manager.GetNodeBlockData(node_ptr);
			if (!node_block)
				continue;

			// Replace NodeAccessors::GetRawNodeVector
			const unsigned char *raw_vector = node_block + node_layout.node_vector_offset;

			// Call assuming ConvertRawVectorToFloat is a free function in diskann::core
			if (!raw_vector || !diskann::common::ConvertRawVectorToFloat(raw_vector, node_float_vector_storage.data(),
			                                                             config.dimensions, config.node_vector_type)) {
				continue;
			}

			float exact_dist = diskann::core::ComputeExactDistanceFloat(query_vector_ptr, node_float_vector_storage.data(),
			                                                            config.dimensions, config.metric_type);

			final_results_min_heap.push({exact_dist, candidate.row_id});
			if (final_results_min_heap.size() > k_neighbors) {
				final_results_min_heap.pop();
			}
		}

		// Populate result_row_ids from final_results_min_heap
		result_row_ids.clear();
		std::vector<NodeCandidate> final_k_candidates_vec;
		while (!final_results_min_heap.empty()) {
			final_k_candidates_vec.push_back(final_results_min_heap.top());
			final_results_min_heap.pop();
		}
		// final_results_min_heap is a min-heap (smallest distance on top).
		// To get results typically ordered by similarity (closest first), reverse if needed.
		// For now, assume current order (smallest distance first from min-heap pop) is fine for result_row_ids.
		// Or if specific order is needed: std::reverse(final_k_candidates_vec.begin(), final_k_candidates_vec.end());
		result_row_ids.reserve(final_k_candidates_vec.size());
		for (const auto &cand : final_k_candidates_vec) {
			result_row_ids.push_back(cand.row_id);
		}
	}
}

// Implementation for Searcher class methods
Searcher::Searcher(IStorageManager *storage_manager) : storage_manager_(storage_manager) {
	if (!storage_manager_) {
		// Or throw an exception, depending on desired strictness
		std::cerr << "Warning: Searcher initialized with null IStorageManager." << std::endl;
	}
}

void Searcher::Search(const float *query_vector, const LmDiskannConfig &config, IGraphManager *graph_manager,
                      common::idx_t k_neighbors, std::vector<common::row_t> &result_row_ids,
                      common::idx_t search_list_size_param) {
	if (!graph_manager) {
		throw std::runtime_error("Searcher::Search: GraphManager is null.");
	}
	if (!storage_manager_) { // Use the member variable
		throw std::runtime_error("Searcher::Search: StorageManager member is null. Was Searcher properly initialized?");
	}

	common::idx_t l_search = (search_list_size_param > 0) ? search_list_size_param : config.l_search;
	if (l_search < k_neighbors) {
		l_search = k_neighbors; // L_search must be at least K
	}

	const NodeLayoutOffsets node_layout = diskann::core::CalculateLayoutInternal(config); // Correct function call

	result_row_ids.clear(); // Clear previous results

	// Call the refactored PerformSearch free function
	PerformSearch(query_vector, k_neighbors, l_search, result_row_ids, config, node_layout, *graph_manager,
	              *storage_manager_, true /* final_pass */);
}

void Searcher::SearchForInitialCandidates(const float *query_vector, const LmDiskannConfig &config,
                                          IGraphManager *graph_manager, common::idx_t num_candidates_to_find,
                                          std::vector<common::row_t> &candidate_row_ids,
                                          common::idx_t search_list_size_param) {
	if (!graph_manager) {
		throw std::runtime_error("Searcher::SearchForInitialCandidates: GraphManager is null.");
	}
	if (!storage_manager_) { // Use the member variable
		throw std::runtime_error(
		    "Searcher::SearchForInitialCandidates: StorageManager member is null. Was Searcher properly initialized?");
	}

	common::idx_t l_search =
	    (search_list_size_param > 0) ? search_list_size_param : config.l_insert; // Use l_insert for initial candidates
	if (l_search < num_candidates_to_find) {
		l_search = num_candidates_to_find;
	}

	const NodeLayoutOffsets node_layout = diskann::core::CalculateLayoutInternal(config); // Correct function call

	candidate_row_ids.clear();

	// Call PerformSearch, k_neighbors is num_candidates_to_find, final_pass is typically false for initial candidates
	PerformSearch(query_vector, num_candidates_to_find, l_search, candidate_row_ids, config, node_layout, *graph_manager,
	              *storage_manager_, false /* not final_pass */);
}

} // namespace core
} // namespace diskann
