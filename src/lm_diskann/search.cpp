#include "search.hpp"
#include "lm_diskann_index.hpp" // Include main index header
#include "node.hpp" // For node accessors
#include "distance.hpp" // For distance functions (placeholders)
#include "state.hpp" // For LMDiskannScanState

#include "duckdb/storage/buffer/buffer_handle.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/common/vector.hpp"

#include <algorithm> // For std::sort
#include <queue> // For priority_queue
#include <vector>
#include <unordered_set>
#include <utility> // For std::pair

namespace duckdb {

// Define CandidateNode using the negated similarity score as distance
using CandidateNode = std::pair<float, idx_t>; // Pair: (negated_similarity_score, node_id)

// Define the priority queue as a min-heap
using CandidateQueue = std::priority_queue<CandidateNode, std::vector<CandidateNode>, std::greater<CandidateNode>>;

// --- Search Helper Implementation ---
void PerformSearch(LMDiskannScanState &scan_state, LMDiskannIndex &index, bool find_exact_distances) {
    // find_exact_distances parameter might be repurposed for re-ranking later

    CandidateQueue candidates_pqueue;
    std::unordered_set<idx_t> visited_nodes;

    // 1. Initialize with Entry Point(s)
    idx_t entry_point_rowid = index.GetEntryPoint(); // Assumes GetEntryPoint returns row_id
    if (entry_point_rowid == idx_t(-1)) {
        // Handle empty index case
        scan_state.result_rowids.clear();
        scan_state.result_scores.clear(); // Or similarity scores
        return;
    }

    IndexPointer entry_point_ptr;
    if (!index.TryGetNodePointer(entry_point_rowid, entry_point_ptr)) {
        throw InternalException("Entry point rowid %llu not found in map during search.", entry_point_rowid);
    }

    // Calculate initial distance/similarity to entry point
    auto entry_node_handle = index.GetNodeBuffer(entry_point_ptr);
    auto entry_node_block = entry_node_handle.Ptr();
    const float* entry_node_vector = reinterpret_cast<const float*>(LMDiskannNodeAccessors::GetNodeVectorPtr(entry_node_block, index.node_layout_));
    float exact_entry_dist = ComputeExactDistanceFloat(scan_state.query_vector_ptr, entry_node_vector,
                                                      index.dimensions_, index.metric_type_);

    candidates_pqueue.push({exact_entry_dist, entry_point_rowid});

    idx_t iterations = 0;
    idx_t max_iterations = 2000; // Add a safety break

    // Main search loop
    while (!candidates_pqueue.empty() && iterations < max_iterations) {
        iterations++;

        // Get the best candidate (lowest negated similarity score == highest similarity)
        CandidateNode best_candidate = candidates_pqueue.top();
        candidates_pqueue.pop();

        idx_t current_rowid = best_candidate.second;

        // Check if already visited
        if (visited_nodes.count(current_rowid)) {
            continue;
        }
        visited_nodes.insert(current_rowid);

        // Use result members from scan_state
        scan_state.result_scores.push_back(best_candidate.first);
        scan_state.result_rowids.push_back(current_rowid);

        // Prune results if exceeding L_search? Or handle results at the end?
        // Let's keep all visited for now, prune/re-rank at the end.

        // Termination condition: If the best candidate popped is worse than
        // the farthest node in our potential result set (size L_search), we might stop.
        // (Needs refinement based on how results are stored/pruned)

        // Get current node's data
        IndexPointer current_node_ptr;
        if (!index.TryGetNodePointer(current_rowid, current_node_ptr)) {
            // Node might have been deleted concurrently? Log or ignore.
            continue;
        }
        auto current_node_handle = index.GetNodeBuffer(current_node_ptr);
        auto current_node_block = current_node_handle.Ptr();

        // Iterate through neighbors
        uint32_t neighbor_count = LMDiskannNodeAccessors::GetNeighborCount(current_node_block);
        const row_t* neighbor_ids_ptr = LMDiskannNodeAccessors::GetNeighborIDsPtr(current_node_block, index.node_layout_);
        if (!neighbor_ids_ptr) { throw InternalException("Failed to get neighbor IDs pointer"); }

        for (uint32_t i = 0; i < neighbor_count; ++i) {
            row_t neighbor_rowid = neighbor_ids_ptr[i];

            if (visited_nodes.count(neighbor_rowid)) {
                continue;
            }

            // Get neighbor's compressed ternary planes
            const_data_ptr_t pos_plane_ptr = LMDiskannNodeAccessors::GetPosPlanePtr(current_node_block, index.node_layout_, i, index.edge_vector_size_bytes_);
            const_data_ptr_t neg_plane_ptr = LMDiskannNodeAccessors::GetNegPlanePtr(current_node_block, index.node_layout_, i, index.dimensions_);
            if (!pos_plane_ptr || !neg_plane_ptr) { throw InternalException("Failed to get neighbor plane pointers"); }

            // Calculate APPROXIMATE score using ternary planes
            float approx_similarity_score = ComputeApproxDistance(
                scan_state.query_vector_ptr,
                pos_plane_ptr,
                neg_plane_ptr,
                index.dimensions_,
                index.metric_type_
            );

            float approx_distance = -approx_similarity_score;

            // Use l_search from scan_state
            if (candidates_pqueue.size() < scan_state.l_search || approx_distance < candidates_pqueue.top().first) {
                 candidates_pqueue.push({approx_distance, neighbor_rowid});
                 if (candidates_pqueue.size() > scan_state.l_search) {
                      candidates_pqueue.pop();
                 }
            }
        }
    }

    // --- Post-Processing --- //
    // 1. Refine Results: Keep only the top L_search results from visited nodes based on scores
    //    (Requires storing scores associated with visited_nodes or re-evaluating)
    //    For now, we have potential candidates in scan_state.result_rowids/scores
    //    Sort these and take top L_search?

    // Sort the collected results (pairs of <negated_similarity, row_id>)
    std::vector<std::pair<float, row_t>> final_candidates;
    for(size_t i=0; i<scan_state.result_rowids.size(); ++i) {
        final_candidates.push_back({scan_state.result_scores[i], scan_state.result_rowids[i]});
    }
    // Sort by negated similarity (ascending, so highest similarity first)
    std::sort(final_candidates.begin(), final_candidates.end());

    // Use l_search from scan_state
    idx_t num_candidates = std::min((idx_t)final_candidates.size(), scan_state.l_search);
    final_candidates.resize(num_candidates);

    // 2. TODO: Re-ranking Step (Crucial for Accuracy)
    //    - Fetch full vectors for these `final_candidates`.
    //    - Calculate EXACT distances using `ComputeExactDistanceFloat`.
    //    - Sort again based on EXACT distance.
    //    - Truncate to final top-k results.

    // Use result members from scan_state
    scan_state.result_rowids.clear();
    scan_state.result_scores.clear();
    for (idx_t i = 0; i < num_candidates; ++i) {
        scan_state.result_rowids.push_back(final_candidates[i].second);
        scan_state.result_scores.push_back(final_candidates[i].first);
    }

    // Final truncation to k (if needed, maybe done by caller?)
    if (scan_state.result_rowids.size() > scan_state.k) {
        scan_state.result_rowids.resize(scan_state.k);
        scan_state.result_scores.resize(scan_state.k);
    }
}

} // namespace duckdb
