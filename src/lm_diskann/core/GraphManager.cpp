/**
 * @file GraphManager.cpp
 * @brief Implements the GraphManager class for managing node allocations,
 *        RowID mappings, and raw node data access for the LM-DiskANN index.
 */
#include "GraphManager.hpp"
#include "../common/types.hpp" // Ensure this is first for common::IndexPointer definition
#include "ISearcher.hpp"       // Added: Interface for search operations
#include "IStorageManager.hpp" // Added: Interface for storage operations
#include "distance.hpp"        // For calculate_distance, ConvertToFloat
#include "index_config.hpp" // For LmDiskannConfig, NodeLayoutOffsets, WordsPerPlane, LmDiskannMetricType
#include "ternary_quantization.hpp" // For WordsPerPlane (if not fully in index_config.hpp)

#include <algorithm> // For std::remove_if, std::find, std::fill, std::min, std::max
#include <cstring>   // For std::memcpy, std::memset
#include <stdexcept> // For std::runtime_error
#include <vector>

namespace diskann {
namespace core {

// --- Constructor & Destructor ---

GraphManager::GraphManager(const LmDiskannConfig &config,
                           const NodeLayoutOffsets &node_layout,
                           common::idx_t block_size_bytes,
                           IStorageManager *storage_manager,
                           ISearcher *searcher)
    : config_(config), node_layout_(node_layout),
      block_size_bytes_(block_size_bytes), storage_manager_(storage_manager),
      searcher_(searcher),      // Initialize new members
      graph_entry_point_ptr_(), // Default construct
      graph_entry_point_rowid_(
          common::NumericLimits<common::row_t>::Maximum()) {
  graph_entry_point_ptr_.block_id =
      common::INVALID_BLOCK_ID; // Explicitly set members
  graph_entry_point_ptr_.offset = 0;

  if (!storage_manager_) {
    throw std::runtime_error(
        "GraphManager: StorageManager pointer cannot be null.");
  }
  if (!searcher_) {
    throw std::runtime_error("GraphManager: Searcher pointer cannot be null.");
  }
}

// Destructor is defaulted in the header, no definition needed here.

// --- IGraphManager Interface Implementation ---

void GraphManager::InitializeEmptyGraph(
    const LmDiskannConfig &config_param,
    common::IndexPointer &entry_point_ptr_out,
    common::row_t &entry_point_rowid_out) {
  [[maybe_unused]] const auto &unused_config = config_param; // Mark as unused
  rowid_to_node_ptr_map_.clear();
  graph_entry_point_ptr_.block_id = common::INVALID_BLOCK_ID;
  graph_entry_point_rowid_ = common::NumericLimits<common::row_t>::Maximum();

  entry_point_ptr_out = graph_entry_point_ptr_;
  entry_point_rowid_out = graph_entry_point_rowid_;
  // Comment: Orchestrator is responsible for ensuring StorageManager also
  // initializes its state.
}

bool GraphManager::AddNode(common::row_t row_id, const float *vector_data,
                           common::idx_t dimensions,
                           common::IndexPointer &node_ptr_out) {
  if (dimensions != config_.dimensions) {
    throw common::NotImplementedException(
        "AddNode: Provided dimensions mismatch with index configuration.");
  }

  common::data_ptr_t new_node_block_data = nullptr;

  // Placeholder: Actual implementation requires StorageManager integration.
  // Steps involved:
  // 1. Call StorageManager to allocate a new node block, get IndexPointer
  // (node_ptr_out).
  //    - If allocation fails (e.g., disk full), return false.
  // 2. Call StorageManager to get a mutable data_ptr_t to the allocated block.
  // 3. Use NodeAccessors::InitializeNodeBlock to prepare the block (zero, set
  // neighbor count to 0).
  // 4. Get mutable raw vector pointer: auto* raw_vec_ptr =
  // NodeAccessors::GetRawNodeVectorMutable(...);
  // 5. Convert and copy vector_data to raw_vec_ptr:
  //    - If config_.node_vector_type == LmDiskannVectorType::FLOAT32,
  //    std::memcpy.
  //    - Else (e.g., INT8), perform quantization and copy.
  // 6. Update internal map: rowid_to_node_ptr_map_[row_id] = node_ptr_out;
  // 7. If GetNodeCount() == 1 (this is the first node), call
  // SetEntryPoint(node_ptr_out, row_id).
  // 8. Perform graph connection (the complex part):
  //    - Find candidate neighbors (e.g., using Searcher::SearchForNeighbors or
  //    similar initial search).
  //    - Call this->RobustPrune(node_ptr_out, vector_data, candidates,
  //    config_);
  //    - For each final neighbor from RobustPrune, update their neighbor lists
  //    to include this new node,
  //      and update this new node's neighbor list. This involves:
  //      - Getting neighbor block data via StorageManager.
  //      - Using NodeAccessors to modify neighbor lists.
  //      - Telling StorageManager the neighbor block is dirty.
  // 9. Tell StorageManager the new node block (node_ptr_out) is dirty.

  bool allocation_success = storage_manager_->AllocateNodeBlock(
      row_id, node_ptr_out, new_node_block_data);

  if (!allocation_success || !new_node_block_data) {
    // Failed to allocate block or get a valid data pointer
    return false;
  }

  NodeAccessors::InitializeNodeBlock(new_node_block_data, block_size_bytes_,
                                     node_layout_);

  // Copy vector data
  unsigned char *raw_vec_ptr =
      NodeAccessors::GetRawNodeVectorMutable(new_node_block_data, node_layout_);
  if (!raw_vec_ptr) {
    // This should not happen if InitializeNodeBlock and layout are correct
    // Consider how to handle this error; perhaps deallocate the block?
    return false;
  }
  if (config_.node_vector_type == LmDiskannVectorType::FLOAT32) {
    std::memcpy(raw_vec_ptr, vector_data, dimensions * sizeof(float));
  } else if (config_.node_vector_type == LmDiskannVectorType::INT8) {
    // TODO: Implement quantization from float to int8
    // For now, throw or return false, as this is critical
    // diskann::core::quantize_vector(vector_data,
    // reinterpret_cast<int8_t*>(raw_vec_ptr), dimensions, ...);
    throw common::NotImplementedException(
        "AddNode: INT8 quantization not implemented.");
    // return false;
  } else {
    throw common::NotImplementedException(
        "AddNode: Unsupported vector type for storage.");
    // return false;
  }

  rowid_to_node_ptr_map_[row_id] = node_ptr_out;

  // If this is the first node, set it as the entry point
  if (rowid_to_node_ptr_map_.size() == 1) {
    SetEntryPoint(node_ptr_out, row_id);
  }

  // Search for initial candidates for RobustPrune
  // The number of candidates to fetch (e.g., L_search parameter, often related
  // to R) Using config_.r * 2 as a placeholder for L_search or similar search
  // list size parameter.
  common::idx_t num_initial_candidates =
      config_.r * 2; // Example search parameter
  std::vector<common::row_t> initial_candidate_row_ids;

  // The searcher might need the current entry point to start its search
  // Or it might have its own strategy (e.g., random nodes if graph is too
  // sparse)
  if (graph_entry_point_ptr_.block_id != common::INVALID_BLOCK_ID) {
    searcher_->SearchForInitialCandidates(
        vector_data, dimensions, config_, this, num_initial_candidates,
        graph_entry_point_ptr_, initial_candidate_row_ids);
  } else {
    // Handle case with no entry point (e.g. very first nodes being added, or
    // searcher handles it)
    // For now, if no entry point, we can't find neighbors this way. RobustPrune
    // will handle empty list.
    // Alternatively, the searcher could be called with an invalid entry point
    // if it can handle that:
    searcher_->SearchForInitialCandidates(
        vector_data, dimensions, config_, this, num_initial_candidates,
        graph_entry_point_ptr_, // Pass invalid ptr
        initial_candidate_row_ids);
  }

  // Prune and connect the new node
  RobustPrune(node_ptr_out, vector_data, initial_candidate_row_ids, config_);

  // After RobustPrune, initial_candidate_row_ids contains the final R
  // neighbors. Update the new node's neighbor list.
  NodeAccessors::SetNeighborCount(
      new_node_block_data, node_layout_,
      static_cast<uint16_t>(initial_candidate_row_ids.size()));
  common::row_t *new_node_neighbors_ptr =
      NodeAccessors::GetNeighborIDsPtrMutable(new_node_block_data,
                                              node_layout_);
  if (!new_node_neighbors_ptr)
    return false; // Should not happen

  for (size_t i = 0; i < initial_candidate_row_ids.size(); ++i) {
    new_node_neighbors_ptr[i] = initial_candidate_row_ids[i];
  }
  // Initialize remaining neighbor slots if any (RobustPrune should return
  // exactly R or fewer)
  for (size_t i = initial_candidate_row_ids.size(); i < config_.r; ++i) {
    new_node_neighbors_ptr[i] =
        common::NumericLimits<common::row_t>::Maximum(); // Mark as
                                                         // invalid/empty
  }

  // TODO: Bidirectional linking: For each final neighbor in
  // initial_candidate_row_ids, update *their* neighbor lists to include
  // 'row_id' (the new node). This also involves RobustPrune on those neighbors
  // with 'row_id' as a candidate. This is a complex part of the Vamana
  // algorithm (Interconnect step). For now, this stub focuses on one-way
  // connection from new_node to its neighbors. Example for one neighbor: if
  // (!initial_candidate_row_ids.empty()) {
  //   common::row_t first_neighbor_row_id = initial_candidate_row_ids[0];
  //   common::IndexPointer neighbor_ptr;
  //   if (TryGetNodePointer(first_neighbor_row_id, neighbor_ptr)) {
  //     common::data_ptr_t neighbor_block_data =
  //     storage_manager_->GetMutableNodeBlockData(neighbor_ptr); if
  //     (neighbor_block_data) {
  //        // ... logic to add row_id to this neighbor's list, potentially
  //        calling RobustPrune for it ...
  //        // NodeAccessors::... to update neighbor_block_data
  //        storage_manager_->MarkBlockDirty(neighbor_ptr);
  //     }
  //   }
  // }

  storage_manager_->MarkBlockDirty(node_ptr_out);
  return true;
}

// Static helper function to get node data - REMOVED, use StorageManager instead
// static common::const_data_ptr_t GetNodeData_Placeholder(...)

bool GraphManager::GetNodeVector(common::IndexPointer node_ptr,
                                 float *vector_out,
                                 common::idx_t dimensions) const {
  if (dimensions != config_.dimensions) {
    return false;
  }
  if (node_ptr.block_id == common::INVALID_BLOCK_ID) {
    return false;
  }

  common::const_data_ptr_t node_block =
      storage_manager_->GetNodeBlockData(node_ptr);
  if (!node_block) {
    // StorageManager could not provide the block data (e.g., IO error, invalid
    // pointer)
    return false;
  }

  const unsigned char *raw_vector =
      NodeAccessors::GetRawNodeVector(node_block, node_layout_);
  if (!raw_vector) {
    // This implies an issue with layout or the block itself.
    return false;
  }

  if (config_.node_vector_type == LmDiskannVectorType::FLOAT32) {
    std::memcpy(vector_out, raw_vector, dimensions * sizeof(float));
  } else if (config_.node_vector_type == LmDiskannVectorType::INT8) {
    // TODO: Implement dequantization from int8 to float.
    // diskann::core::dequantize_vector(reinterpret_cast<const
    // int8_t*>(raw_vector), vector_out, dimensions, ...);
    throw common::NotImplementedException(
        "GetNodeVector: Dequantization from INT8 not implemented.");
    // return false;
  } else {
    throw common::NotImplementedException(
        "GetNodeVector: Unsupported vector type for retrieval.");
    // return false;
  }
  // No need to delete node_block, StorageManager handles its lifecycle.
  return true;
}

bool GraphManager::GetNeighbors(
    common::IndexPointer node_ptr,
    std::vector<common::row_t> &neighbor_row_ids_out) const {
  neighbor_row_ids_out.clear();
  if (node_ptr.block_id == common::INVALID_BLOCK_ID) {
    return false;
  }

  common::const_data_ptr_t node_block =
      storage_manager_->GetNodeBlockData(node_ptr);
  if (!node_block) {
    return false;
  }

  uint16_t count = NodeAccessors::GetNeighborCount(node_block, node_layout_);
  const common::row_t *ids_ptr =
      NodeAccessors::GetNeighborIDsPtr(node_block, node_layout_);

  if (!ids_ptr) {
    // Should not happen if layout is correct and block is valid
    return false;
  }

  neighbor_row_ids_out.reserve(count);
  for (uint16_t i = 0; i < count; ++i) {
    if (ids_ptr[i] !=
        common::NumericLimits<
            common::row_t>::Maximum()) { // Filter invalid/empty slots
      neighbor_row_ids_out.push_back(ids_ptr[i]);
    }
  }
  // No need to delete node_block, StorageManager handles its lifecycle.
  return true;
}

void GraphManager::RobustPrune(
    common::IndexPointer node_to_connect_ptr,
    const float *node_to_connect_vector_data,
    std::vector<common::row_t>
        &candidate_row_ids, // Input: initial candidates, Output: final selected
                            // neighbors
    const LmDiskannConfig &config_param) { // Use config_param for R, alpha

  if (candidate_row_ids.empty()) {
    return; // No candidates to prune or connect to.
  }

  // This will store pairs of (distance, candidate_row_id)
  std::vector<std::pair<float, common::row_t>> dist_candidates;
  dist_candidates.reserve(candidate_row_ids.size());

  // 1. Calculate distances from node_to_connect to all initial candidates
  for (common::row_t cand_row_id : candidate_row_ids) {
    common::IndexPointer cand_ptr;
    if (TryGetNodePointer(cand_row_id, cand_ptr)) {
      // Calculate distance using the provided vector for node_to_connect and
      // fetching cand_ptr's vector
      float dist = CalculateDistanceInternal(
          cand_ptr, node_to_connect_vector_data, config_param.dimensions);
      dist_candidates.push_back({dist, cand_row_id});
    } else {
      // Candidate row_id not found in current graph map, might be an error or
      // stale ID. Optionally log this.
    }
  }

  // Sort candidates by distance (ascending)
  std::sort(dist_candidates.begin(), dist_candidates.end());

  // Clear the input/output vector and prepare to fill it with final neighbors
  candidate_row_ids.clear();

  // 2. Implement Vamana pruning logic (Section 4 of DiskANN paper)
  // TODO: Implement the full alpha-based pruning. For now, a simplified
  // version:
  //       Select the top R distinct neighbors by distance.

  for (const auto &pair : dist_candidates) {
    if (candidate_row_ids.size() >= config_param.r) {
      break; // Already found R neighbors
    }
    // Ensure we are not trying to connect a node to itself if its row_id was in
    // candidates This check is tricky if node_to_connect_ptr is not directly
    // comparable to candidate_row_ids. Assume node_to_connect_ptr's original
    // row_id is not in candidate_row_ids unless it's a different node. Or, the
    // caller of RobustPrune (AddNode) should filter self out from
    // initial_candidate_row_ids. For now, assuming candidate_row_ids does not
    // contain the node being connected.

    // Check for duplicates already added (should not happen if initial
    // candidates are distinct row_ids)
    bool already_added = false;
    for (common::row_t added_id : candidate_row_ids) {
      if (added_id == pair.second) {
        already_added = true;
        break;
      }
    }
    if (!already_added) {
      candidate_row_ids.push_back(pair.second);
    }
  }

  // The actual RobustPrune algorithm is more complex:
  // while result.size < R:
  //   p = nearest candidate not in result
  //   add p to result
  //   for each r in result:
  //     if alpha * dist(p,r) < dist(node_to_connect, r):  <-- This is one
  //     variant, or comparing to dist(node_to_connect, p)
  //       prune r from result (this is complex, usually means future candidates
  //       are occluded by p)
  // This simplified version just takes the R closest. A full implementation is
  // a major TODO.
  if (candidate_row_ids.size() >
      config_param
          .r) { // Should not happen with current logic, but as safeguard
    candidate_row_ids.resize(config_param.r);
  }
}

void GraphManager::SetEntryPoint(common::IndexPointer node_ptr,
                                 common::row_t row_id) {
  // Basic validation: if one is valid, the other should be too (or both invalid
  // for reset)
  if ((node_ptr.block_id == common::INVALID_BLOCK_ID) !=
      (row_id == common::NumericLimits<common::row_t>::Maximum())) {
    // Log a warning about inconsistent entry point state if desired.
  }
  graph_entry_point_ptr_ = node_ptr;
  graph_entry_point_rowid_ = row_id;
}

common::IndexPointer GraphManager::GetEntryPointPointer() const {
  return graph_entry_point_ptr_;
}

common::row_t GraphManager::GetEntryPointRowId() const {
  return graph_entry_point_rowid_;
}

void GraphManager::HandleNodeDeletion(common::row_t row_id) {
  if (row_id == graph_entry_point_rowid_) {
    graph_entry_point_ptr_.block_id = common::INVALID_BLOCK_ID;
    graph_entry_point_rowid_ = common::NumericLimits<common::row_t>::Maximum();
  }
  // Comment: Actual removal from neighbor lists of other nodes is complex and
  // usually part of a separate process (e.g., vacuuming, or search-time
  // filtering of deleted nodes). This function primarily ensures a deleted node
  // isn't the active entry point.
}

void GraphManager::FreeNode(common::row_t row_id) {
  // This is a logical removal from GraphManager's tracking.
  // Orchestrator must inform StorageManager to actually reclaim/mark the block
  // as free.
  if (rowid_to_node_ptr_map_.erase(row_id) > 0) {
    // If the freed node was the entry point, reset entry point to invalid.
    if (row_id == graph_entry_point_rowid_) {
      common::IndexPointer invalid_ptr;
      invalid_ptr.block_id = common::INVALID_BLOCK_ID;
      invalid_ptr.offset = 0;
      SetEntryPoint(invalid_ptr,
                    common::NumericLimits<common::row_t>::Maximum());
    }
  } // else: node was not found in map, nothing to erase.
}

common::idx_t GraphManager::GetNodeCount() const {
  return rowid_to_node_ptr_map_.size();
}

common::idx_t GraphManager::GetInMemorySize() const {
  common::idx_t size = sizeof(*this); // Size of GraphManager object itself.
  // Approximate size of the std::map.
  // This is a rough estimate. For more accuracy, platform-specific details or a
  // custom allocator tracking would be needed.
  size += rowid_to_node_ptr_map_.size() *
          (sizeof(common::row_t) + sizeof(common::IndexPointer) + // Key + Value
           sizeof(void *) * 3 + sizeof(bool) +
           sizeof(int) // Approx overhead per std::map node (varies by STL impl)
          );
  return size;
}

void GraphManager::Reset() {
  rowid_to_node_ptr_map_.clear();
  graph_entry_point_ptr_.block_id = common::INVALID_BLOCK_ID;
  graph_entry_point_rowid_ = common::NumericLimits<common::row_t>::Maximum();
  // Comment: Orchestrator is responsible for ensuring StorageManager also
  // resets its state (e.g., clear all blocks).
}

// --- Non-interface public methods ---

bool GraphManager::TryGetNodePointer(common::row_t row_id,
                                     common::IndexPointer &node_ptr_out) const {
  auto iter = rowid_to_node_ptr_map_.find(row_id);
  if (iter != rowid_to_node_ptr_map_.end()) {
    node_ptr_out = iter->second;
    return true;
  }
  node_ptr_out.block_id =
      common::INVALID_BLOCK_ID; // Ensure it's invalid if not found
  node_ptr_out.offset = 0;      // Also zero out offset for an invalid pointer
  return false;
}

common::row_t GraphManager::GetRandomNodeID(common::RandomEngine &engine) {
  if (rowid_to_node_ptr_map_.empty()) {
    return common::NumericLimits<common::row_t>::Maximum();
  }
  auto map_size_val = rowid_to_node_ptr_map_.size();
  if (map_size_val ==
      0) { // Should be caught by empty() check, but good for safety
    return common::NumericLimits<common::row_t>::Maximum();
  }
  common::idx_t random_idx =
      engine.NextRandomInteger<common::idx_t>() % map_size_val;
  auto iter = rowid_to_node_ptr_map_.begin();
  std::advance(iter, random_idx);
  return iter->first;
}

// --- Private Helper Methods ---

// Internal distance calculation helpers, using config_ members consistently.
float GraphManager::CalculateDistanceInternal(
    common::IndexPointer node1_ptr, common::IndexPointer node2_ptr) const {
  std::vector<float> vec1(config_.dimensions);
  std::vector<float> vec2(config_.dimensions);

  // GetNodeVector is const, so no const_cast needed anymore if it's correctly
  // implemented.
  if (!GetNodeVector(node1_ptr, vec1.data(), config_.dimensions) ||
      !GetNodeVector(node2_ptr, vec2.data(), config_.dimensions)) {
    // Error handling: could not retrieve one or both vectors.
    // Depending on policy, could return infinity, throw, or a special error
    // value.
    throw std::runtime_error(
        "CalculateDistanceInternal: Failed to get node vectors.");
  }
  return diskann::core::CalculateDistance( // Uppercase C
      vec1.data(), vec2.data(), config_.dimensions, config_.metric_type);
}

float GraphManager::CalculateDistanceInternal(const float *vec1,
                                              const float *vec2,
                                              common::idx_t dimensions) const {
  if (dimensions != config_.dimensions) {
    throw std::runtime_error("CalculateDistanceInternal: Dimension mismatch "
                             "for raw vector comparison.");
  }
  // This one doesn't depend on GetNodeVector, it's direct.
  return diskann::core::CalculateDistance( // Uppercase C
      vec1, vec2, config_.dimensions, config_.metric_type);
}

float GraphManager::CalculateDistanceInternal(common::IndexPointer node_ptr,
                                              const float *query_vec,
                                              common::idx_t dimensions) const {
  if (dimensions != config_.dimensions) {
    throw std::runtime_error("CalculateDistanceInternal: Dimension mismatch "
                             "for node vs query vector comparison.");
  }
  std::vector<float> node_vec(config_.dimensions);
  if (!GetNodeVector(node_ptr, node_vec.data(), config_.dimensions)) {
    throw std::runtime_error("CalculateDistanceInternal: Failed to get node "
                             "vector for comparison with query.");
  }
  return diskann::core::CalculateDistance( // Uppercase C
      node_vec.data(), query_vec, config_.dimensions, config_.metric_type);
}

// --- NodeAccessors Implementations ---
// These are static methods, typically defined here for encapsulation if not in
// a separate NodeAccessors.cpp

void NodeAccessors::InitializeNodeBlock(common::data_ptr_t node_block_ptr,
                                        common::idx_t block_size_bytes,
                                        const NodeLayoutOffsets &layout) {
  std::memset(node_block_ptr, 0, block_size_bytes);
  // Explicitly set neighbor count to 0. This assumes neighbor_count_offset is
  // valid.
  if (layout.neighbor_count_offset + sizeof(uint16_t) <= block_size_bytes) {
    *reinterpret_cast<uint16_t *>(node_block_ptr +
                                  layout.neighbor_count_offset) = 0;
  } else {
    // This indicates a critical error in layout calculation or
    // block_size_bytes.
    throw std::runtime_error(
        "InitializeNodeBlock: layout.neighbor_count_offset is out of bounds "
        "relative to block_size_bytes.");
  }
  // Other initializations (e.g., marking all neighbor slots as invalid) could
  // go here.
}

uint16_t
NodeAccessors::GetNeighborCount(common::const_data_ptr_t node_block_ptr,
                                const NodeLayoutOffsets &layout) {
  // Assuming node_block_ptr is valid and layout.neighbor_count_offset is within
  // bounds.
  return *reinterpret_cast<const uint16_t *>(node_block_ptr +
                                             layout.neighbor_count_offset);
}

void NodeAccessors::SetNeighborCount(common::data_ptr_t node_block_ptr,
                                     const NodeLayoutOffsets &layout,
                                     uint16_t count) {
  // Add check: count <= config.r (max degree)
  // Assuming node_block_ptr is valid and layout.neighbor_count_offset is within
  // bounds.
  *reinterpret_cast<uint16_t *>(node_block_ptr + layout.neighbor_count_offset) =
      count;
}

const unsigned char *
NodeAccessors::GetRawNodeVector(common::const_data_ptr_t node_block_ptr,
                                const NodeLayoutOffsets &layout) {
  // Assuming node_block_ptr is valid and layout.node_vector_offset is within
  // bounds.
  return node_block_ptr + layout.node_vector_offset;
}

unsigned char *
NodeAccessors::GetRawNodeVectorMutable(common::data_ptr_t node_block_ptr,
                                       const NodeLayoutOffsets &layout) {
  // Assuming node_block_ptr is valid and layout.node_vector_offset is within
  // bounds.
  return node_block_ptr + layout.node_vector_offset;
}

const common::row_t *
NodeAccessors::GetNeighborIDsPtr(common::const_data_ptr_t node_block_ptr,
                                 const NodeLayoutOffsets &layout) {
  // Assuming node_block_ptr is valid and layout.neighbor_ids_offset is within
  // bounds.
  return reinterpret_cast<const common::row_t *>(node_block_ptr +
                                                 layout.neighbor_ids_offset);
}

common::row_t *
NodeAccessors::GetNeighborIDsPtrMutable(common::data_ptr_t node_block_ptr,
                                        const NodeLayoutOffsets &layout) {
  // Assuming node_block_ptr is valid and layout.neighbor_ids_offset is within
  // bounds.
  return reinterpret_cast<common::row_t *>(node_block_ptr +
                                           layout.neighbor_ids_offset);
}

// --- Ternary Quantization Accessors Implementation ---
TernaryPlanesView NodeAccessors::GetNeighborTernaryPlanes(
    common::const_data_ptr_t node_block_ptr, const NodeLayoutOffsets &layout,
    uint16_t neighbor_idx, common::idx_t dimensions) {
  // Check if ternary data is expected by this layout.
  if (layout.ternary_edge_size_bytes == 0) {
    return {nullptr, nullptr, 0, 0}; // Return an invalid/empty view.
  }
  common::idx_t words_per_plane = diskann::core::WordsPerPlane(dimensions);
  common::idx_t single_neighbor_ternary_data_size =
      layout.ternary_edge_size_bytes; // This should be 2 * words_per_plane *
                                      // sizeof(uint64_t)

  // neighbor_pos_planes_offset should point to the start of all positive planes
  // for *all* neighbors. Then, for a specific neighbor_idx, we offset into this
  // array.
  common::const_data_ptr_t all_pos_planes_start =
      node_block_ptr + layout.neighbor_pos_planes_offset;
  common::const_data_ptr_t neighbor_pos_plane =
      all_pos_planes_start +
      (neighbor_idx * words_per_plane * sizeof(uint64_t));

  common::const_data_ptr_t all_neg_planes_start =
      node_block_ptr + layout.neighbor_neg_planes_offset;
  common::const_data_ptr_t neighbor_neg_plane =
      all_neg_planes_start +
      (neighbor_idx * words_per_plane * sizeof(uint64_t));

  return {neighbor_pos_plane, neighbor_neg_plane, dimensions, words_per_plane};
}

MutableTernaryPlanesView NodeAccessors::GetNeighborTernaryPlanesMutable(
    common::data_ptr_t node_block_ptr, const NodeLayoutOffsets &layout,
    uint16_t neighbor_idx, common::idx_t dimensions) {
  if (layout.ternary_edge_size_bytes == 0) {
    return {nullptr, nullptr, 0, 0};
  }
  common::idx_t words_per_plane = diskann::core::WordsPerPlane(dimensions);
  common::idx_t single_neighbor_ternary_data_size =
      layout.ternary_edge_size_bytes;

  common::data_ptr_t all_pos_planes_start =
      node_block_ptr + layout.neighbor_pos_planes_offset;
  common::data_ptr_t neighbor_pos_plane =
      all_pos_planes_start +
      (neighbor_idx * words_per_plane * sizeof(uint64_t));

  common::data_ptr_t all_neg_planes_start =
      node_block_ptr + layout.neighbor_neg_planes_offset;
  common::data_ptr_t neighbor_neg_plane =
      all_neg_planes_start +
      (neighbor_idx * words_per_plane * sizeof(uint64_t));

  return {neighbor_pos_plane, neighbor_neg_plane, dimensions, words_per_plane};
}

} // namespace core
} // namespace diskann
