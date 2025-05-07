#include "search.hpp"
#include "LmDiskannIndex.hpp"     // Include main index header
#include "LmDiskannScanState.hpp" // For LmDiskannScanState
#include "NodeAccessors.hpp"      // For node accessors
#include "distance.hpp" // For ComputeExactDistanceFloat, ComputeApproxSimilarityTernary
#include "index_config.hpp" // For config structs (needed potentially by index members)
#include "ternary_quantization.hpp" // For EncodeTernary, GetKernel, WordsPerPlane (needed by TopKTernarySearch)

#include "duckdb/common/assert.hpp" // For assert
#include "duckdb/common/printer.hpp"
#include "duckdb/common/vector.hpp" // For FlatVector - needed?
#include "duckdb/storage/buffer/buffer_handle.hpp"

#include <algorithm> // For std::sort, std::min
#include <queue>     // For priority_queue
#include <unordered_set>
#include <utility> // For std::pair
#include <vector>

namespace duckdb {

// Define CandidateNode using the negated similarity score as distance
using CandidateNode =
    std::pair<float, idx_t>; // Pair: (distance_score, node_id)

// Define the priority queue as a min-heap (stores nodes with smallest distance
// at top)
using CandidateQueue =
    std::priority_queue<CandidateNode, std::vector<CandidateNode>,
                        std::greater<CandidateNode>>;

// --- Search Helper Implementation --- //

// Helper to calculate plane size bytes
inline idx_t GetPlaneSizeBytes(const LmDiskannIndex &index) {
  // Use public getter instead of accessing private member
  idx_t dimensions = index.GetDimensions(); // Assuming this getter exists
  if (dimensions == 0) {
    throw InternalException(
        "Index dimensions are 0, cannot calculate plane size");
  }
  return GetTernaryPlaneSizeBytes(dimensions);
}

void PerformSearch(LmDiskannScanState &scan_state, LmDiskannIndex &index,
                   const LmDiskannConfig &config, bool find_exact_distances) {
  // Max-heap for candidates (stores {distance, node_id}) - we want smallest
  // distance = highest priority So we store {-distance, node_id} or use
  // std::greater
  using candidate_pair_t = std::pair<float, row_t>;
  std::priority_queue<candidate_pair_t> candidate_pqueue; // Max-heap

  // 1. Initialize with Entry Point(s)
  // Access private helpers via friend status or make them public/internal
  // helpers? Assuming friend access is okay for now.
  row_t entry_point_rowid = index.GetEntryPoint();
  if (entry_point_rowid == NumericLimits<row_t>::Maximum()) {
    // No entry point, index is empty or deleted entry not replaced
    // Ensure scan_state.candidates is empty (it should be initially)
    // std::priority_queue<candidate_pair_t>().swap(scan_state.candidates); //
    // Clear if needed
    return; // No search possible
  }

  IndexPointer entry_point_ptr;
  if (!index.TryGetNodePointer(entry_point_rowid, entry_point_ptr)) {
    // Entry point exists but node pointer not found (map inconsistency?)
    Printer::Print(StringUtil::Format(
        "Warning: Entry point rowid %lld not found in map during search.",
        entry_point_rowid));
    // Ensure scan_state.candidates is empty
    // std::priority_queue<candidate_pair_t>().swap(scan_state.candidates);
    return;
  }

  // Calculate exact distance to entry point
  vector<float> entry_node_float_vec(config.dimensions);
  vector<float> query_float_vec(config.dimensions);
  try {
    auto entry_node_handle = index.GetNodeBuffer(entry_point_ptr);
    auto entry_node_block = entry_node_handle.Ptr();
    const_data_ptr_t entry_node_raw_vec =
        NodeAccessors::GetNodeVector(entry_node_block, index.GetNodeLayout());

    // Assuming query_vector_ptr points to float data
    memcpy(query_float_vec.data(), scan_state.query_vector_ptr,
           config.dimensions * sizeof(float));

    index.ConvertNodeVectorToFloat(entry_node_raw_vec,
                                   entry_node_float_vec.data());

    float exact_entry_dist = ComputeExactDistanceFloat(
        query_float_vec.data(), entry_node_float_vec.data(), config.dimensions,
        config.metric_type);

    // Use negative distance for max-heap behavior (closest is highest priority)
    candidate_pqueue.push({-exact_entry_dist, entry_point_rowid});
    scan_state.visited.insert(entry_point_rowid);

  } catch (const std::exception &e) {
    Printer::Print(
        StringUtil::Format("Warning: Failed to process entry point %lld: %s",
                           entry_point_rowid, e.what()));
    // Ensure scan_state.candidates is empty
    // std::priority_queue<candidate_pair_t>().swap(scan_state.candidates);
    return;
  }

  // --- Main Search Loop --- //
  idx_t iterations = 0;

  // Use the top_candidates max-heap from scan_state to track best results
  // Note: scan_state.candidates should be the main exploration queue, not used
  // here? Let's rename the local one to exploration_queue and use
  // scan_state.top_candidates for results.
  std::priority_queue<candidate_pair_t>().swap(
      scan_state.top_candidates); // Clear results heap

  while (!candidate_pqueue.empty()) {
    iterations++;

    candidate_pair_t best_candidate = candidate_pqueue.top();
    candidate_pqueue.pop();
    float current_best_neg_dist =
        best_candidate.first; // This is negative distance

    // Pruning condition: If the candidate from exploration queue is further
    // away (more negative distance) than the k-th element currently in our
    // results heap (worst distance in results), and results heap is full, we
    // can stop exploring this path.
    if (scan_state.top_candidates.size() >= config.l_search &&
        current_best_neg_dist < scan_state.top_candidates.top().first) {
      // Note: top_candidates stores {pos_distance, rowid}, so top() gives
      // largest distance candidate_pqueue stores {-distance, rowid}. If
      // -cand_dist < -results_worst_dist, then cand_dist > results_worst_dist.
      // This logic needs refinement. Let's stick to exploring based on L_search
      // limit first.
    }

    row_t current_rowid = best_candidate.second;

    IndexPointer current_node_ptr;
    try {
      if (!index.TryGetNodePointer(current_rowid, current_node_ptr))
        continue; // Node deleted?
      auto current_node_handle = index.GetNodeBuffer(current_node_ptr);
      auto current_node_block = current_node_handle.Ptr();
      const NodeLayoutOffsets &layout = index.GetNodeLayout();

      uint16_t neighbor_count =
          NodeAccessors::GetNeighborCount(current_node_block);
      const row_t *neighbor_ids_ptr =
          NodeAccessors::GetNeighborIDsPtr(current_node_block, layout);
      if (!neighbor_ids_ptr)
        continue;

      for (uint16_t i = 0; i < neighbor_count; ++i) {
        row_t neighbor_rowid = neighbor_ids_ptr[i];

        if (scan_state.visited.count(neighbor_rowid)) {
          continue;
        }
        scan_state.visited.insert(neighbor_rowid);

        TernaryPlanesView neighbor_planes =
            NodeAccessors::GetNeighborTernaryPlanes(current_node_block, layout,
                                                    i, config.dimensions);

        if (!neighbor_planes.IsValid()) {
          Printer::Print(StringUtil::Format(
              "Warning: Invalid ternary planes for neighbor %lld of node %lld",
              neighbor_rowid, current_rowid));
          continue;
        }

        float approx_distance = index.CalculateApproxDistance(
            query_float_vec.data(), neighbor_planes.positive_plane);

        // Add to exploration queue if potentially better than the L_search
        // worst result
        if (scan_state.top_candidates.size() < config.l_search ||
            approx_distance < scan_state.top_candidates.top().first) {
          candidate_pqueue.push(
              {-approx_distance, neighbor_rowid}); // Push negative distance

          // Update results heap (top_candidates stores positive distance)
          scan_state.top_candidates.push({approx_distance, neighbor_rowid});
          if (scan_state.top_candidates.size() > config.l_search) {
            scan_state.top_candidates.pop(); // Keep only L_search best
          }
        }
      }
    } catch (const std::exception &e) {
      Printer::Print(StringUtil::Format(
          "Warning: Error processing node %lld during search: %s",
          current_rowid, e.what()));
      continue;
    }
  }

  // --- Post-Processing --- //
  // scan_state.top_candidates now holds the L_search best candidates based on
  // approximate distance.

  if (find_exact_distances && scan_state.k > 0) {
    // Re-rank the top_candidates using exact distances
    std::vector<candidate_pair_t> final_candidates; // Use {exact_dist, rowid}
    final_candidates.reserve(scan_state.top_candidates.size());
    vector<float> node_float_vec(config.dimensions);

    while (!scan_state.top_candidates.empty()) {
      row_t cand_rowid = scan_state.top_candidates.top().second;
      scan_state.top_candidates.pop();

      IndexPointer cand_ptr;
      try {
        if (!index.TryGetNodePointer(cand_rowid, cand_ptr))
          continue;
        auto cand_handle = index.GetNodeBuffer(cand_ptr);
        auto cand_block = cand_handle.Ptr();
        const_data_ptr_t cand_raw_vec =
            NodeAccessors::GetNodeVector(cand_block, index.GetNodeLayout());

        index.ConvertNodeVectorToFloat(cand_raw_vec, node_float_vec.data());

        float exact_dist = ComputeExactDistanceFloat(
            query_float_vec.data(), node_float_vec.data(), config.dimensions,
            config.metric_type);
        final_candidates.push_back({exact_dist, cand_rowid});

      } catch (const std::exception &e) {
        Printer::Print(
            StringUtil::Format("Warning: Failed to re-rank candidate %lld: %s",
                               cand_rowid, e.what()));
      }
    }

    // Sort final candidates by exact distance (ascending)
    std::sort(final_candidates.begin(), final_candidates.end());

    // Re-populate the top_candidates heap (which is a max-heap) up to k
    // results. This is slightly awkward as top_candidates is used for L_search
    // AND final results. Let's clear it and repopulate.
    std::priority_queue<candidate_pair_t>().swap(scan_state.top_candidates);
    for (const auto &final_cand : final_candidates) {
      if (scan_state.top_candidates.size() < scan_state.k) {
        scan_state.top_candidates.push(final_cand); // Push {exact_dist, rowid}
      } else if (final_cand.first < scan_state.top_candidates.top().first) {
        // If the new exact distance is better than the worst in the heap,
        // replace
        scan_state.top_candidates.pop();
        scan_state.top_candidates.push(final_cand);
      } else {
        // Since final_candidates is sorted, no need to check further if heap is
        // full break; // Optimization: break early if remaining candidates are
        // worse
      }
    }
  } // End if(find_exact_distances)

  // PerformSearch is now complete. The results are in
  // scan_state.top_candidates. LmDiskannIndex::Scan will extract the RowIDs
  // from this heap.
}

//--------------------------------------------------------------------
// Top-K Ternary Search Function
//--------------------------------------------------------------------

/**
 * @brief Performs a Top-K nearest neighbor search using a batch of ternary
 * encoded vectors.
 *
 * @param query Pointer to the original floating-point query vector.
 * @param dims The original dimensionality of the query vector (used for
 * encoding).
 * @param database_batch A view describing the batch of pre-encoded database
 * vectors.
 * @param K The number of nearest neighbors to retrieve.
 * @param neighIDs Pointer to an array of uint64_t IDs corresponding to the
 * vectors in the database_batch.
 * @param[out] out A std::vector of std::pair<float, uint64_t> which will be
 * filled with the top K results, sorted by score (highest score first). The
 * pair contains (similarity_score, neighbor_ID).
 *
 * @details (Remains largely the same)
 * 1. Encodes the floating-point query vector into its ternary bit-planes using
 * 'dims'.
 * 2. Selects the fastest available ternary dot product kernel.
 * 3. Iterates through the N database vectors described by 'database_batch':
 *    a. Calculates pointers to the current database vector's bit-planes using
 * offsets. b. Computes the raw ternary dot product score. c. Normalizes the
 * score using 'dims'. d. Maintains a min-priority queue (min-heap) of size K.
 * 4. Extracts results.
 */
inline void TopKTernarySearch(
    const float *query,
    size_t dims,                                  // Query vector dimension
    const TernaryPlaneBatchView &database_batch,  // Batch of DB vectors
    size_t K,                                     // Number of neighbors to find
    const uint64_t *neighIDs,                     // IDs for database vectors
    std::vector<std::pair<float, uint64_t>> &out) // Output vector
{
  // --- Input Validation ---
  // Extract N from the batch view for checks
  size_t N = database_batch.num_vectors;

  assert(query != nullptr && "Query vector pointer cannot be null");
  assert(database_batch.IsValid() && "Database batch view is invalid");
  assert(neighIDs != nullptr && "Neighbor IDs pointer cannot be null");
  assert(dims > 0 && "Query dimensions must be positive");
  assert(K > 0 && "K must be greater than 0");
  // Optional: Check consistency between dims and words_per_plane?
  // assert(database_batch.words_per_plane == WordsPerPlane(dims) && "Dimension
  // mismatch");

  out.clear(); // Clear output vector initially
  if (N == 0)
    return;           // Handle empty database
  K = std::min(K, N); // Adjust K if larger than database size
  if (K == 0)
    return;

  // --- Preparation ---
  // Extract batch properties from the view
  const uint64_t *posPlaneData = database_batch.positive_planes_start;
  const uint64_t *negPlaneData = database_batch.negative_planes_start;
  const size_t words_per_vector = database_batch.words_per_plane;

  // Allocate temporary buffers for the encoded query vector's planes (using
  // query dims)
  std::vector<uint64_t> query_pos_plane_vec(words_per_vector);
  std::vector<uint64_t> query_neg_plane_vec(words_per_vector);
  EncodeTernary(query, query_pos_plane_vec.data(), query_neg_plane_vec.data(),
                dims);
  const uint64_t *qp = query_pos_plane_vec.data();
  const uint64_t *qn = query_neg_plane_vec.data();

  dot_fun_t dot_kernel = GetDotKernel();

  using ScoreIdPair = std::pair<float, uint64_t>;
  auto min_heap_comparator = [](const ScoreIdPair &a, const ScoreIdPair &b) {
    return a.first >
           b.first; // Smallest score has highest priority (at the top)
  };
  std::priority_queue<ScoreIdPair, std::vector<ScoreIdPair>,
                      decltype(min_heap_comparator)>
      min_heap(min_heap_comparator);

  // --- Search Loop ---
  for (size_t idx = 0; idx < N; ++idx) { // Use N from batch view
    const uint64_t *current_vpos = posPlaneData + idx * words_per_vector;
    const uint64_t *current_vneg = negPlaneData + idx * words_per_vector;
    int64_t raw_score =
        dot_kernel(qp, qn, current_vpos, current_vneg, words_per_vector);

    // Normalize score using query dimensions
    float normalized_score =
        (dims > 0) ? static_cast<float>(raw_score) / static_cast<float>(dims)
                   : 0.0f;

    // --- Update Top-K Heap ---
    if (min_heap.size() < K) {
      min_heap.emplace(normalized_score, neighIDs[idx]);
    } else if (normalized_score > min_heap.top().first) {
      min_heap.pop();
      min_heap.emplace(normalized_score, neighIDs[idx]);
    }
  }

  // --- Result Extraction ---
  size_t result_count = min_heap.size();
  out.resize(result_count);
  for (size_t i = 0; i < result_count; ++i) {
    out[result_count - 1 - i] = min_heap.top();
    min_heap.pop();
  }
}

} // namespace duckdb
