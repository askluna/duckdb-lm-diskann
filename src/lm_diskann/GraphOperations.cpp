/**
 * @file GraphOperations.cpp
 * @brief Implements the GraphOperations class for managing LM-DiskANN graph
 * structure.
 */
#include "GraphOperations.hpp"
#include "LmDiskannIndex.hpp"     // For LmDiskannIndex context (PerformSearch)
#include "LmDiskannScanState.hpp" // For creating scan states for internal searches
#include "NodeAccessors.hpp"      // For direct block manipulation
#include "NodeManager.hpp"        // For node access
#include "distance.hpp" // For ComputeExactDistanceFloat and other distance utils

#include "duckdb/common/limits.hpp"
#include "duckdb/common/printer.hpp" // For optional warnings/debug
#include "duckdb/common/random_engine.hpp"
#include "duckdb/common/types/vector.hpp" // For creating temporary vectors for search

#include <algorithm> // For std::sort, std::unique
#include <queue>     // For PerformSearch's priority_queue if replicated here
#include <vector>

namespace duckdb {

GraphOperations::GraphOperations(const LmDiskannConfig &config,
                                 const NodeLayoutOffsets &node_layout,
                                 NodeManager &node_manager,
                                 LmDiskannIndex &index_context)
    : config_(config), node_layout_(node_layout), node_manager_(node_manager),
      index_context_(index_context) {}

// Placeholder for SearchForCandidates, which would replicate PerformSearch or
// call it via index_context_ This is a complex function that might be better
// called via index_context if PerformSearch remains public/friend For now, a
// simplified sketch that shows how it might be used internally.
std::vector<std::pair<float, row_t>>
GraphOperations::SearchForCandidates(const float *target_vector_float,
                                     uint32_t K, uint32_t scan_list_size) {

  // Create a temporary DuckDB vector for the query
  Vector query_vec_handle(
      LogicalType::ARRAY(LogicalType::FLOAT, config_.dimensions));
  memcpy(FlatVector::GetData<float>(query_vec_handle), target_vector_float,
         config_.dimensions * sizeof(float));
  query_vec_handle.Flatten(1);

  // Initialize a scan state for this internal search
  LmDiskannScanState search_state(query_vec_handle, K, scan_list_size);

  // Perform the search using the LmDiskannIndex context
  // This assumes PerformSearch is accessible from LmDiskannIndex, possibly via
  // a friend declaration or public method.
  // index_context_.PerformSearch(search_state, config_, false); // false = do
  // not find exact distances yet for candidates For now, direct call to global
  // PerformSearch (if it was made so, or called via index_context_)
  PerformSearch(search_state, index_context_, this->config_, false);

  std::vector<std::pair<float, row_t>> candidates;
  candidates.reserve(search_state.top_candidates.size());
  while (!search_state.top_candidates.empty()) {
    candidates.push_back(search_state.top_candidates.top());
    search_state.top_candidates.pop();
  }
  // PerformSearch populates a max-heap, so results are furthest-first.
  // For candidate lists, often distance ascending is preferred. Re-sort if
  // needed or ensure PerformSearch provides sorted. For RobustPrune, the exact
  // order before pruning might not matter as much as having diverse candidates.
  std::reverse(candidates.begin(), candidates.end()); // Make it closest-first
  return candidates;
}

void GraphOperations::RobustPrune(
    row_t node_rowid, IndexPointer node_ptr, const float *node_vector_float,
    std::vector<std::pair<float, row_t>> &candidates, bool is_new_node_prune) {

  uint32_t max_neighbors = config_.r;
  data_ptr_t node_data = node_manager_.GetNodeDataMutable(node_ptr);

  // If this is not a new node, add its existing neighbors to the candidate set
  if (!is_new_node_prune) {
    uint16_t current_neighbor_count =
        NodeAccessors::GetNeighborCount(node_data);
    const row_t *current_neighbor_ids =
        NodeAccessors::GetNeighborIDsPtr(node_data, node_layout_);

    for (uint16_t i = 0; i < current_neighbor_count; ++i) {
      row_t existing_id = current_neighbor_ids[i];
      if (existing_id == NumericLimits<row_t>::Maximum())
        continue;

      bool already_candidate = false;
      for (const auto &cand : candidates) {
        if (cand.second == existing_id) {
          already_candidate = true;
          break;
        }
      }
      if (already_candidate)
        continue;

      // We need the vector of the existing neighbor to calculate distance to it
      // This might involve fetching another node's data.
      // For simplicity here, we use the compressed representation if available
      // to get approx distance. A full RobustPrune might require exact
      // distances.
      TernaryPlanesView existing_neighbor_planes =
          NodeAccessors::GetNeighborTernaryPlanes(node_data, node_layout_, i,
                                                  config_.dimensions);
      if (existing_neighbor_planes.IsValid()) {
        float dist = index_context_.PublicCalculateApproxDistance(
            node_vector_float, existing_neighbor_planes.positive_plane);
        candidates.push_back({dist, existing_id});
      }
    }
  }

  // Sort by row_id first for unique()
  std::sort(candidates.begin(), candidates.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; });
  auto unique_end = std::unique(
      candidates.begin(), candidates.end(),
      [](const auto &a, const auto &b) { return a.second == b.second; });
  candidates.erase(unique_end, candidates.end());

  // Sort again by distance for alpha pruning
  std::sort(candidates.begin(), candidates.end());

  std::vector<row_t> final_neighbor_ids;
  std::vector<vector<uint8_t>>
      final_compressed_neighbors_data; // Store full compressed edge
  final_neighbor_ids.reserve(max_neighbors);
  final_compressed_neighbors_data.reserve(max_neighbors);

  vector<float> candidate_vector_float_storage(config_.dimensions);

  for (const auto &candidate_pair : candidates) {
    if (final_neighbor_ids.size() >= max_neighbors)
      break;

    row_t candidate_id = candidate_pair.second;
    if (candidate_id == node_rowid)
      continue; // Don't connect to self

    IndexPointer candidate_node_ptr;
    if (!node_manager_.TryGetNodePointer(candidate_id, candidate_node_ptr))
      continue;

    const_data_ptr_t candidate_raw_data =
        node_manager_.GetNodeData(candidate_node_ptr);
    const_data_ptr_t candidate_raw_vector =
        NodeAccessors::GetNodeVector(candidate_raw_data, node_layout_);
    index_context_.PublicConvertNodeVectorToFloat(
        candidate_raw_vector, candidate_vector_float_storage.data());

    bool pruned = false;
    float dist_node_to_candidate = ComputeExactDistanceFloat(
        node_vector_float, candidate_vector_float_storage.data(),
        config_.dimensions, config_.metric_type);

    for (size_t i = 0; i < final_neighbor_ids.size(); ++i) {
      row_t existing_final_id = final_neighbor_ids[i];
      IndexPointer existing_final_node_ptr;
      if (!node_manager_.TryGetNodePointer(existing_final_id,
                                           existing_final_node_ptr))
        continue;

      vector<float> existing_final_vector_float_storage(config_.dimensions);
      const_data_ptr_t existing_final_raw_data =
          node_manager_.GetNodeData(existing_final_node_ptr);
      const_data_ptr_t existing_final_raw_vector =
          NodeAccessors::GetNodeVector(existing_final_raw_data, node_layout_);
      index_context_.PublicConvertNodeVectorToFloat(
          existing_final_raw_vector,
          existing_final_vector_float_storage.data());

      float dist_existing_final_to_candidate =
          ComputeExactDistanceFloat(existing_final_vector_float_storage.data(),
                                    candidate_vector_float_storage.data(),
                                    config_.dimensions, config_.metric_type);

      if (dist_node_to_candidate >
          config_.alpha * dist_existing_final_to_candidate) {
        pruned = true;
        break;
      }
    }

    if (!pruned) {
      final_neighbor_ids.push_back(candidate_id);
      final_compressed_neighbors_data.emplace_back(
          node_layout_.ternary_edge_size_bytes);
      index_context_.PublicCompressVectorForEdge(
          candidate_vector_float_storage.data(),
          final_compressed_neighbors_data.back().data());
    }
  }

  uint16_t final_count = static_cast<uint16_t>(final_neighbor_ids.size());
  NodeAccessors::SetNeighborCount(node_data, final_count);
  row_t *dest_ids_ptr =
      NodeAccessors::GetNeighborIDsPtrMutable(node_data, node_layout_);

  idx_t plane_size_bytes = GetTernaryPlaneSizeBytes(config_.dimensions);
  data_ptr_t dest_pos_planes_base =
      node_data + node_layout_.neighbor_pos_planes_offset;
  data_ptr_t dest_neg_planes_base =
      node_data + node_layout_.neighbor_neg_planes_offset;

  for (uint16_t i = 0; i < final_count; ++i) {
    dest_ids_ptr[i] = final_neighbor_ids[i];
    memcpy(dest_pos_planes_base + i * plane_size_bytes,
           final_compressed_neighbors_data[i].data(), plane_size_bytes);
    memcpy(dest_neg_planes_base + i * plane_size_bytes,
           final_compressed_neighbors_data[i].data() + plane_size_bytes,
           plane_size_bytes);
  }
  // Clear out remaining neighbor slots
  for (uint16_t i = final_count; i < config_.r; ++i) {
    dest_ids_ptr[i] = NumericLimits<row_t>::Maximum();
    memset(dest_pos_planes_base + i * plane_size_bytes, 0, plane_size_bytes);
    memset(dest_neg_planes_base + i * plane_size_bytes, 0, plane_size_bytes);
  }
}

void GraphOperations::InsertNode(row_t new_node_rowid,
                                 IndexPointer new_node_ptr,
                                 const float *new_node_vector_float) {
  if (graph_entry_point_rowid_ == NumericLimits<row_t>::Maximum()) {
    // This is the first node, set it as entry point with 0 neighbors
    data_ptr_t new_node_data = node_manager_.GetNodeDataMutable(new_node_ptr);
    NodeAccessors::SetNeighborCount(new_node_data, 0);
    graph_entry_point_rowid_ = new_node_rowid;
    graph_entry_point_ptr_ = new_node_ptr;
    index_context_.PublicMarkDirty(); // Mark index as dirty
    return;
  }

  // Find candidate neighbors using search
  std::vector<std::pair<float, row_t>> candidates = SearchForCandidates(
      new_node_vector_float, config_.l_insert, config_.l_insert);

  // Add self to candidates for pruning with existing neighbors (if any were
  // added by RobustPrune's first pass) candidates.push_back({0.0f,
  // new_node_rowid}); // Original RobustPrune took care of this logic.

  // Prune neighbors for the new node
  RobustPrune(new_node_rowid, new_node_ptr, new_node_vector_float, candidates,
              true);

  // Update neighbors of the new node to point back (reciprocal edges)
  const_data_ptr_t new_node_data_ro = node_manager_.GetNodeData(
      new_node_ptr); // Read-only for its new neighbors
  uint16_t final_new_neighbor_count =
      NodeAccessors::GetNeighborCount(new_node_data_ro);
  const row_t *final_new_neighbor_ids =
      NodeAccessors::GetNeighborIDsPtr(new_node_data_ro, node_layout_);

  vector<float> neighbor_node_vector_float_storage(config_.dimensions);

  for (uint16_t i = 0; i < final_new_neighbor_count; ++i) {
    row_t neighbor_rowid = final_new_neighbor_ids[i];
    if (neighbor_rowid == NumericLimits<row_t>::Maximum() ||
        neighbor_rowid == new_node_rowid)
      continue;

    IndexPointer neighbor_ptr;
    if (!node_manager_.TryGetNodePointer(neighbor_rowid, neighbor_ptr))
      continue;

    const_data_ptr_t neighbor_raw_data =
        node_manager_.GetNodeData(neighbor_ptr);
    const_data_ptr_t neighbor_raw_vector =
        NodeAccessors::GetNodeVector(neighbor_raw_data, node_layout_);
    index_context_.PublicConvertNodeVectorToFloat(
        neighbor_raw_vector, neighbor_node_vector_float_storage.data());

    // Create candidate list for the neighbor, including the new node
    std::vector<std::pair<float, row_t>> neighbor_candidates;
    float dist_neighbor_to_new = ComputeExactDistanceFloat(
        neighbor_node_vector_float_storage.data(), new_node_vector_float,
        config_.dimensions, config_.metric_type);
    neighbor_candidates.push_back({dist_neighbor_to_new, new_node_rowid});

    // Prune for the neighbor (is_new_node_prune = false, as we are adding to
    // existing node)
    RobustPrune(neighbor_rowid, neighbor_ptr,
                neighbor_node_vector_float_storage.data(), neighbor_candidates,
                false);
  }
  index_context_.PublicMarkDirty();
}

void GraphOperations::HandleNodeDeletion(row_t deleted_node_rowid) {
  if (deleted_node_rowid == graph_entry_point_rowid_) {
    graph_entry_point_ptr_.Clear();
    graph_entry_point_rowid_ = NumericLimits<row_t>::Maximum();
    // Attempt to find a new entry point immediately if possible, or let
    // SelectEntryPointForSearch handle it lazily. For now, just clear.
    // SelectEntryPointForSearch will fix it on next search.
    index_context_.PublicMarkDirty();
  }
  // TODO: Graph repair logic if a deleted node was a critical connector for
  // some parts of the graph. This is complex and typically handled by periodic
  // rebuilds or more advanced maintenance in DiskANN variants.
}

row_t GraphOperations::SelectEntryPointForSearch(RandomEngine &engine) {
  if (graph_entry_point_rowid_ != NumericLimits<row_t>::Maximum()) {
    IndexPointer ptr_check;
    // Verify current entry point is still valid in the node manager
    if (node_manager_.TryGetNodePointer(graph_entry_point_rowid_, ptr_check)) {
      return graph_entry_point_rowid_;
    }
    // Entry point was deleted or became invalid
    graph_entry_point_ptr_.Clear();
    graph_entry_point_rowid_ = NumericLimits<row_t>::Maximum();
    index_context_.PublicMarkDirty();
  }

  if (node_manager_.GetNodeCount() == 0) {
    return NumericLimits<row_t>::Maximum(); // No nodes, no entry point
  }

  row_t random_id = node_manager_.GetRandomNodeID(engine);
  if (random_id != NumericLimits<row_t>::Maximum()) {
    IndexPointer random_ptr;
    if (node_manager_.TryGetNodePointer(random_id, random_ptr)) {
      graph_entry_point_rowid_ = random_id;
      graph_entry_point_ptr_ = random_ptr;
      index_context_.PublicMarkDirty();
      return random_id;
    }
  }
  // If GetRandomNodeID fails or its pointer isn't found (shouldn't happen if
  // map is consistent) Fallback: Iterate map for any valid node (less ideal)
  // This part is a safety net, ideally GetRandomNodeID + TryGetNodePointer is
  // robust. For now, if random selection fails, we indicate no entry point
  // found.
  return NumericLimits<row_t>::Maximum();
}

/**
 * @brief Gets the current graph entry point pointer.
 * @return IndexPointer to the entry point node.
 */
IndexPointer GraphOperations::GetGraphEntryPointPointer() const {
  return graph_entry_point_ptr_;
}

/**
 * @brief Gets the row_id of the current graph entry point.
 * @return row_t of the entry point node.
 */
row_t GraphOperations::GetGraphEntryPointRowId() const {
  return graph_entry_point_rowid_;
}

} // namespace duckdb
