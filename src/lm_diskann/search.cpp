
#include "search.hpp"
#include "lm_diskann_index.hpp" // Include main index header
#include "lm_diskann_node.hpp" // For node accessors
#include "lm_diskann_storage.hpp" // For storage functions (placeholders)
#include "lm_diskann_distance.hpp" // For distance functions (placeholders)
#include "lm_diskann_state.hpp" // For LMDiskannScanState

#include "duckdb/storage/buffer_handle.hpp"
#include "duckdb/common/printer.hpp"

#include <algorithm> // For std::sort

namespace duckdb {

// --- Search Helper Implementation ---
void PerformSearch(LMDiskannScanState &scan_state, LMDiskannIndex &index, bool find_exact_distances) {
    // Main beam search loop based on diskAnnSearchInternal

    while (!scan_state.candidates.empty()) {
        // Check termination condition: If the best candidate in the queue is worse than the k-th node found so far.
        if (find_exact_distances && scan_state.top_candidates.size() >= scan_state.k) {
            float worst_found_dist = scan_state.top_candidates.back().first; // Assumes top_candidates is sorted
            if (scan_state.candidates.top().first >= worst_found_dist) {
                // Optimization: Check if the node is already in top_candidates before breaking
                bool already_in_top_k = false;
                row_t best_candidate_rowid = scan_state.candidates.top().second;
                for(const auto& top_cand : scan_state.top_candidates) {
                    if (top_cand.second == best_candidate_rowid) {
                        already_in_top_k = true;
                        break;
                    }
                }
                if (!already_in_top_k) {
                    break; // Early termination
                }
            }
        }

        // 1. Select best candidate row_id from priority queue
        // float candidate_dist_approx = scan_state.candidates.top().first;
        row_t candidate_row_id = scan_state.candidates.top().second;
        scan_state.candidates.pop();

        if (scan_state.visited.count(candidate_row_id)) {
            continue; // Already visited
        }
        // 2. Mark visited
        scan_state.visited.insert(candidate_row_id);

        // 3. Get node pointer using the (placeholder) storage function
        IndexPointer node_ptr;
        // Accessing storage requires passing context (db, allocator) held by the index instance
        // This highlights the need for either a StorageManager class or passing index context.
        // For now, directly call the placeholder function (assuming it can access necessary context).
        // FIXME: Replace with proper storage access via index instance or manager class
        if (!TryGetNodePointer(candidate_row_id, node_ptr, index.db_, *index.allocator_ /*, index.rowid_map_ */ )) {
             Printer::Warning("Node %lld not found in map during search (likely deleted).", candidate_row_id);
             continue;
        }

        // 4. Read node block
        BufferHandle handle;
        try {
             // FIXME: Replace with proper storage access
             handle = GetNodeBuffer(node_ptr, index.db_, *index.allocator_);
        } catch (std::exception &e) {
             Printer::Warning("Failed to read block for node %lld during search: %s", candidate_row_id, e.what());
             continue; // Cannot process this node
        }
        auto block_data = handle.Ptr();

        // 5. Calculate exact distance if needed and add to top candidates
        if (find_exact_distances) {
             const_data_ptr_t node_vec_raw_ptr = LMDiskannNodeAccessors::GetNodeVectorPtr(block_data, index.node_layout_);
             float exact_dist = index.CalculateDistance<float, float>(scan_state.query_vector_ptr, reinterpret_cast<const float*>(node_vec_raw_ptr)); // Assuming float node/query

             // Add to top_candidates and keep it sorted and capped at size k
             if (scan_state.top_candidates.size() < scan_state.k || exact_dist < scan_state.top_candidates.back().first) {
                  scan_state.top_candidates.push_back({exact_dist, candidate_row_id});
                  std::sort(scan_state.top_candidates.begin(), scan_state.top_candidates.end()); // Keep sorted
                  if (scan_state.top_candidates.size() > scan_state.k) {
                       scan_state.top_candidates.pop_back(); // Keep only top k
                  }
             }
        }

        // 6. Get neighbor info
        uint16_t ncount = LMDiskannNodeAccessors::GetNeighborCount(block_data);
        const row_t* neighbor_ids = LMDiskannNodeAccessors::GetNeighborIDsPtr(block_data, index.node_layout_);

        // 7. Iterate through neighbors
        for (uint16_t i = 0; i < ncount; ++i) {
            row_t neighbor_id = neighbor_ids[i];
            if (scan_state.visited.count(neighbor_id)) {
                continue; // Skip visited neighbors
            }

            // 8. Get compressed neighbor ptr
            const_data_ptr_t compressed_ptr = LMDiskannNodeAccessors::GetCompressedNeighborPtr(block_data, index.node_layout_, i, index.edge_vector_size_bytes_);

            // 9. Calculate approximate distance using index's method
            float approx_dist = index.CalculateApproxDistance(scan_state.query_vector_ptr, compressed_ptr);

            // 10. Add to candidates queue if promising
            if (scan_state.candidates.size() < scan_state.l_search || approx_dist < scan_state.candidates.top().first) {
                 // FIXME: Need efficient check for duplicates in priority queue.
                 scan_state.candidates.push({approx_dist, neighbor_id});
                 if (scan_state.candidates.size() > scan_state.l_search) {
                      scan_state.candidates.pop();
                 }
            }
        }
    } // End while loop
}


} // namespace duckdb
