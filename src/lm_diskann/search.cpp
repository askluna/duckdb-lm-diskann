#include "search.hpp"
#include "lm_diskann_index.hpp" // Include main index header
#include "node.hpp" // For node accessors
#include "distance.hpp" // For ComputeExactDistanceFloat, ComputeApproxSimilarityTernary
#include "state.hpp" // For LMDiskannScanState
#include "ternary_quantization.hpp" // For EncodeTernary, GetKernel, WordsPerPlane (needed by TopKTernarySearch)
#include "config.hpp" // For config structs (needed potentially by index members)

#include "duckdb/storage/buffer/buffer_handle.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/common/vector.hpp" // For FlatVector - needed?
#include "duckdb/common/assert.hpp" // For assert

#include <algorithm> // For std::sort, std::min
#include <queue> // For priority_queue
#include <vector>
#include <unordered_set>
#include <utility> // For std::pair

namespace duckdb {

// Define CandidateNode using the negated similarity score as distance
using CandidateNode = std::pair<float, idx_t>; // Pair: (distance_score, node_id)

// Define the priority queue as a min-heap (stores nodes with smallest distance at top)
using CandidateQueue = std::priority_queue<CandidateNode, std::vector<CandidateNode>, std::greater<CandidateNode>>;

// --- Search Helper Implementation --- //

// Helper to calculate plane size bytes
inline idx_t GetPlaneSizeBytes(const LMDiskannIndex &index) {
     // Use public getter instead of accessing private member
     idx_t dimensions = index.GetDimensions(); // Assuming this getter exists
     if (dimensions == 0) {
         throw InternalException("Index dimensions are 0, cannot calculate plane size");
     }
     return GetTernaryPlaneSizeBytes(dimensions);
}

void PerformSearch(LMDiskannScanState &scan_state, LMDiskannIndex &index, bool find_exact_distances) {
    // find_exact_distances parameter might be repurposed for re-ranking later

    CandidateQueue candidates_pqueue; // Min-heap: stores <distance, rowid>
    std::unordered_set<idx_t> visited_nodes;
    std::vector<CandidateNode> results; // Store potential results <distance, rowid>

    // 1. Initialize with Entry Point(s)
    idx_t entry_point_rowid = index.GetEntryPoint();
    if (entry_point_rowid == idx_t(-1)) {
        scan_state.result_rowids.clear();
        scan_state.result_scores.clear();
        return;
    }

    IndexPointer entry_point_ptr;
    if (!index.TryGetNodePointer(entry_point_rowid, entry_point_ptr)) {
        throw InternalException("Entry point rowid %llu not found in map during search.", entry_point_rowid);
    }

    // Calculate exact distance to entry point
    auto entry_node_handle = index.GetNodeBuffer(entry_point_ptr);
    auto entry_node_block = entry_node_handle.Ptr();
    // Fix: Use correct accessor name
    const float* entry_node_vector = reinterpret_cast<const float*>(LMDiskannNodeAccessors::GetNodeVector(entry_node_block, index.node_layout_));
    float exact_entry_dist = ComputeExactDistanceFloat(scan_state.query_vector_ptr, entry_node_vector,
                                                      index.dimensions_, index.metric_type_);

    candidates_pqueue.push({exact_entry_dist, entry_point_rowid});
    results.push_back({exact_entry_dist, entry_point_rowid});
    visited_nodes.insert(entry_point_rowid);

    idx_t iterations = 0;
    idx_t max_iterations = 2000; // Safety break

    // Main search loop
    while (!candidates_pqueue.empty() && iterations < max_iterations) {
        iterations++;

        // Get the best candidate (closest node found so far)
        CandidateNode best_candidate = candidates_pqueue.top();
        candidates_pqueue.pop();

        // If the best node in the queue is already further than the L_search'th result,
        // we can potentially prune (beam search optimization)
        // Requires keeping results sorted. Let's implement simpler best-first for now.

        idx_t current_rowid = best_candidate.second;

        // Get current node's data
        IndexPointer current_node_ptr;
        if (!index.TryGetNodePointer(current_rowid, current_node_ptr)) continue; // Node deleted?
        auto current_node_handle = index.GetNodeBuffer(current_node_ptr);
        auto current_node_block = current_node_handle.Ptr();

        // Iterate through neighbors
        uint16_t neighbor_count = LMDiskannNodeAccessors::GetNeighborCount(current_node_block);
        // Fix: Use correct accessor name
        const row_t* neighbor_ids_ptr = LMDiskannNodeAccessors::GetNeighborIDs(current_node_block, index.node_layout_);
        if (!neighbor_ids_ptr) continue; // Should not happen if count > 0

        idx_t plane_size_bytes = GetPlaneSizeBytes(index); // Use helper

        for (uint16_t i = 0; i < neighbor_count; ++i) {
            row_t neighbor_rowid = neighbor_ids_ptr[i];

            if (visited_nodes.count(neighbor_rowid)) {
                continue;
            }
            visited_nodes.insert(neighbor_rowid); // Mark visited immediately

            // Get neighbor's compressed ternary planes
            // Fix: Use correct accessor names
            const_data_ptr_t pos_plane_ptr = LMDiskannNodeAccessors::GetNeighborPositivePlane(current_node_block, index.node_layout_, i, plane_size_bytes);
            const_data_ptr_t neg_plane_ptr = LMDiskannNodeAccessors::GetNeighborNegativePlane(current_node_block, index.node_layout_, i, plane_size_bytes);

            if (!pos_plane_ptr || !neg_plane_ptr) {
                 // Log error or throw? Continue for now.
                 continue;
            }

            TernaryPlanesView neighbor_planes {pos_plane_ptr, neg_plane_ptr, WordsPerPlane(index.dimensions_)};

            // Fix: Calculate APPROXIMATE score using ternary planes
            float approx_similarity_score = ComputeApproxSimilarityTernary(
                scan_state.query_vector_ptr,
                neighbor_planes,
                index.dimensions_
            );

            // Convert similarity score to a distance for the min-heap
            // For Cosine/IP, distance = -similarity (or 1-similarity for cosine distance)
            // Since the ternary kernel score correlates with Cosine/IP, use negated score.
            float approx_distance = -approx_similarity_score;

            // Check if this neighbor is potentially within the top L_search
            // Keep track of the current L_search worst distance in the results list
            float worst_dist_in_results = std::numeric_limits<float>::max();
            if (results.size() >= scan_state.l_search) {
                // Results list needs to be kept sorted or use a max-heap to find worst easily.
                // Simple linear scan for now (inefficient but correct).
                 std::sort(results.begin(), results.end()); // Keep sorted by distance
                 if (results.size() > scan_state.l_search) results.resize(scan_state.l_search);
                 worst_dist_in_results = results.back().first;
            } else {
                // Sort only when full? Simpler: just check size
            }

            if (results.size() < scan_state.l_search || approx_distance < worst_dist_in_results) {
                // Add to candidate queue and results list
                 candidates_pqueue.push({approx_distance, neighbor_rowid});
                 results.push_back({approx_distance, neighbor_rowid});
                 // Maintain results list size (inefficiently here)
                 std::sort(results.begin(), results.end());
                 if (results.size() > scan_state.l_search) results.resize(scan_state.l_search);
            }
        }
    }

    // --- Post-Processing --- //
    // Results list already contains the top candidates based on approx distance
    // Sort one last time and truncate to L_search
    std::sort(results.begin(), results.end());
    if (results.size() > scan_state.l_search) {
        results.resize(scan_state.l_search);
    }

    // 2. TODO: Re-ranking Step (Crucial for Accuracy)
    //    - Fetch full vectors for these `results`.
    //    - Calculate EXACT distances using `ComputeExactDistanceFloat`.
    //    - Sort again based on EXACT distance.
    //    - Truncate to final top-k results.

    // Copy results to scan_state (currently using approx distances)
    scan_state.result_rowids.clear();
    scan_state.result_scores.clear();
    for (const auto& res : results) {
        scan_state.result_rowids.push_back(res.second);
        scan_state.result_scores.push_back(res.first); // Store approx distance for now
    }

    // Final truncation to k
    if (scan_state.result_rowids.size() > scan_state.k) {
        scan_state.result_rowids.resize(scan_state.k);
        scan_state.result_scores.resize(scan_state.k);
    }
}


//--------------------------------------------------------------------
// Top-K Ternary Search Function
//--------------------------------------------------------------------

/**
 * @brief Performs a Top-K nearest neighbor search using ternary encoded vectors.
 *
 * @param query Pointer to the original floating-point query vector.
 * @param posPlaneData Pointer to the start of the concatenated positive bit-planes
 * for all N database vectors. Data is assumed to be laid out
 * contiguously: [vec0_pos_plane, vec1_pos_plane, ...].
 * @param negPlaneData Pointer to the start of the concatenated negative bit-planes
 * for all N database vectors, laid out similarly.
 * @param N The total number of database vectors represented in the plane data.
 * @param dims The original dimensionality of the vectors.
 * @param K The number of nearest neighbors to retrieve.
 * @param neighIDs Pointer to an array of uint64_t IDs corresponding to the vectors
 * in posPlaneData/negPlaneData (must have size N). The ID at index `i`
 * corresponds to the vector whose planes start at `posPlaneData + i * words`.
 * @param[out] out A std::vector of std::pair<float, uint64_t> which will be filled
 * with the top K results, sorted by score (highest score first).
 * The pair contains (similarity_score, neighbor_ID).
 *
 * @details
 * 1. Encodes the floating-point query vector into its ternary bit-planes.
 * 2. Selects the fastest available ternary dot product kernel (using GetKernel).
 * 3. Iterates through the N database vectors:
 * a. Calculates the pointers to the current database vector's bit-planes.
 * b. Computes the raw ternary dot product score using the selected kernel.
 * c. Normalizes the raw score (divides by dims) as a proxy for similarity.
 * d. Maintains a min-priority queue (min-heap) of size K to keep track of the
 * current top K candidates based on the normalized score.
 * 4. Extracts the results from the priority queue into the output vector, sorted
 * from highest score to lowest score.
 */
inline void TopKTernarySearch(const float* query,
                              const uint64_t* posPlaneData,
                              const uint64_t* negPlaneData,
                              size_t N,                 // Number of database vectors
                              size_t dims,              // Original vector dimension
                              size_t K,                 // Number of neighbors to find
                              const uint64_t* neighIDs, // IDs for database vectors
                              std::vector<std::pair<float, uint64_t>>& out) // Output vector
{
    // --- Input Validation ---
    assert(query != nullptr && "Query vector pointer cannot be null");
    assert(posPlaneData != nullptr && "Positive plane data pointer cannot be null");
    assert(negPlaneData != nullptr && "Negative plane data pointer cannot be null");
    assert(neighIDs != nullptr && "Neighbor IDs pointer cannot be null");
    assert(dims > 0 && "Dimensions must be positive");
    assert(K > 0 && "K must be greater than 0");

    // Handle edge case: No database vectors to search.
    if (N == 0) {
        out.clear(); // Ensure output is empty
        return;
    }
    // Handle edge case: K is larger than the number of database vectors.
    K = std::min(K, N);
    if (K == 0) { // If K became 0 after std::min (e.g., K was initially 0 or N was 0)
        out.clear();
        return;
    }


    // --- Preparation ---
    // Calculate the number of 64-bit words per vector plane.
    const size_t words_per_vector = WordsPerPlane(dims);

    // Allocate temporary buffers for the encoded query vector's planes.
    // Using std::vector ensures proper memory management.
    std::vector<uint64_t> query_pos_plane_vec(words_per_vector);
    std::vector<uint64_t> query_neg_plane_vec(words_per_vector);

    // Encode the float query vector into the temporary ternary planes.
    EncodeTernary(query, query_pos_plane_vec.data(), query_neg_plane_vec.data(), dims);
    // Get pointers to the encoded query data for passing to the kernel.
    const uint64_t* qp = query_pos_plane_vec.data();
    const uint64_t* qn = query_neg_plane_vec.data();

    // Get the best available dot product kernel function pointer.
    dot_fun_t dot_kernel = GetKernel();

    // Define the type for score-ID pairs.
    using ScoreIdPair = std::pair<float, uint64_t>;

    // Define a custom comparator for the priority queue to make it a *min-heap*
    // based on the score (the float element of the pair).
    // `a > b` means `a` has lower priority (larger score goes deeper in min-heap).
    auto min_heap_comparator = [](const ScoreIdPair& a, const ScoreIdPair& b) {
        return a.first > b.first; // Smallest score has highest priority (at the top)
    };

    // Create the min-priority queue. It will store at most K elements.
    std::priority_queue<ScoreIdPair, std::vector<ScoreIdPair>, decltype(min_heap_comparator)>
        min_heap(min_heap_comparator);

    // --- Search Loop ---
    // Iterate through each database vector.
    for(size_t idx = 0; idx < N; ++idx){
        // Calculate pointers to the start of the current database vector's planes.
        const uint64_t* current_vpos = posPlaneData + idx * words_per_vector;
        const uint64_t* current_vneg = negPlaneData + idx * words_per_vector;

        // Compute the raw ternary dot product score using the selected kernel.
        int64_t raw_score = dot_kernel(qp, qn, current_vpos, current_vneg, words_per_vector);

        // Normalize the raw score. Dividing by dims provides a rough measure of
        // similarity per dimension, ranging approximately from -1.0 to +1.0.
        // This is NOT cosine similarity, but serves as a ranking score.
        // Avoid division by zero if dims is somehow 0 (though asserted earlier).
        float normalized_score = (dims > 0)
                               ? static_cast<float>(raw_score) / static_cast<float>(dims)
                               : 0.0f;

        // --- Update Top-K Heap ---
        // If the heap isn't full yet (has less than K elements), add the current result.
        if (min_heap.size() < K){
            min_heap.emplace(normalized_score, neighIDs[idx]);
        }
        // If the heap is full, check if the current score is better (higher) than
        // the worst score currently in the heap (which is at the top of the min-heap).
        else if (normalized_score > min_heap.top().first){
            min_heap.pop(); // Remove the element with the lowest score.
            min_heap.emplace(normalized_score, neighIDs[idx]); // Add the new, better element.
        }
        // Otherwise (current score is not better than the worst in the heap), do nothing.
    }

    // --- Result Extraction ---
    // The min-heap now contains the top K results, but ordered with the lowest score
    // at the top. We need to extract them into the output vector sorted highest-to-lowest.
    size_t result_count = min_heap.size(); // Should be equal to K unless N < K
    out.resize(result_count);

    // Pop elements from the min-heap (smallest score first) and place them
    // into the output vector from back to front to achieve descending score order.
    for (size_t i = 0; i < result_count; ++i) {
        out[result_count - 1 - i] = min_heap.top();
        min_heap.pop();
    }
}

} // namespace duckdb
