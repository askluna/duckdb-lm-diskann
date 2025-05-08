/**
 * @file GraphManager.cpp
 * @brief Implements the GraphManager class for managing node allocations,
 *        RowID mappings, and raw node data access for the LM-DiskANN index.
 */
#include "GraphManager.hpp"
#include "../common/types.hpp" // For common types, INVALID_BLOCK_ID, RandomEngine
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
                           common::idx_t block_size_bytes)
    : config_(config), node_layout_(node_layout),
      block_size_bytes_(block_size_bytes),
      graph_entry_point_rowid_(
          common::NumericLimits<common::row_t>::Maximum()) {
  graph_entry_point_ptr_.block_id =
      common::INVALID_BLOCK_ID; // Initialize to invalid state
  // Comment: rowid_to_node_ptr_map_ is default constructed (empty map).
}

// Destructor is defaulted in the header, no definition needed here.

// --- IGraphManager Interface Implementation ---

void GraphManager::InitializeEmptyGraph(
    const LmDiskannConfig & /*config_param*/, // config_ is already a member
    common::IndexPointer &entry_point_ptr_out,
    common::row_t &entry_point_rowid_out) {
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

  throw common::NotImplementedException(
      "GraphManager::AddNode: Full implementation pending StorageManager and "
      "Searcher integration.");
  return false; // Placeholder
}

// Static helper function to get node data - TEMPORARY STAND-IN for
// StorageManager
static common::const_data_ptr_t
GetNodeData_Placeholder(common::IndexPointer node_ptr,
                        common::idx_t expected_block_size,
                        const char *error_location_tag) {
  if (node_ptr.block_id == common::INVALID_BLOCK_ID) {
    // DUCKDB_ASSERT_MSG(false, std::string("GetNodeData_Placeholder: Attempted
    // to access invalid node pointer in ") + error_location_tag);
    return nullptr; // Invalid pointer passed in
  }
  // THIS IS A MEMORY LEAK AND INCORRECT. Placeholder for StorageManager call.
  unsigned char *some_memory = new unsigned char[expected_block_size];
  std::memset(some_memory, 0,
              expected_block_size); // Initialize to prevent uninitialized reads
  // In a real scenario, StorageManager would return a pointer to a managed
  // buffer.
  return some_memory;
}

bool GraphManager::GetNodeVector(common::IndexPointer node_ptr,
                                 float *vector_out,
                                 common::idx_t dimensions) const {
  if (dimensions != config_.dimensions) {
    // Consider throwing or logging an error for dimension mismatch.
    return false;
  }

  common::const_data_ptr_t node_block =
      GetNodeData_Placeholder(node_ptr, block_size_bytes_, "GetNodeVector");
  if (!node_block) {
    return false; // Invalid node_ptr or GetNodeData_Placeholder failed
  }

  const unsigned char *raw_vector =
      NodeAccessors::GetRawNodeVector(node_block, node_layout_);
  if (!raw_vector) {
    // This implies an issue with layout or the block itself. Placeholder
    // GetNodeData_Placeholder might not reflect real scenarios. delete[]
    // node_block; // If GetNodeData_Placeholder allocated, it needs to be
    // deleted.
    return false;
  }

  // Handle different vector types. For now, only FLOAT32 is directly converted.
  if (config_.node_vector_type == LmDiskannVectorType::FLOAT32) {
    // core::ConvertToFloat is assumed to handle this correctly.
    // Ensure ConvertToFloat is available and included via distance.hpp or
    // similar.
    std::memcpy(vector_out, raw_vector,
                dimensions * sizeof(float)); // Simpler if it's already float
    // core::ConvertToFloat(raw_vector, vector_out, dimensions); // Use if
    // raw_vector is not float*
  } else if (config_.node_vector_type == LmDiskannVectorType::INT8) {
    // Example: Dequantize from int8 to float. Requires quantization params not
    // yet in LmDiskannConfig. delete[] node_block;
    throw common::NotImplementedException(
        "GetNodeVector: Dequantization from INT8 not implemented.");
  } else {
    // delete[] node_block;
    throw common::NotImplementedException(
        "GetNodeVector: Unsupported node_vector_type in config.");
  }

  // delete[] node_block; // IMPORTANT: Placeholder GetNodeData_Placeholder
  // allocates, so this *must* be called. In a real system with StorageManager,
  // buffer management is different.
  return true;
}

bool GraphManager::GetNeighbors(
    common::IndexPointer node_ptr,
    std::vector<common::row_t> &neighbor_row_ids_out) const {
  neighbor_row_ids_out.clear();
  common::const_data_ptr_t node_block =
      GetNodeData_Placeholder(node_ptr, block_size_bytes_, "GetNeighbors");
  if (!node_block) {
    return false;
  }

  uint16_t count = NodeAccessors::GetNeighborCount(node_block, node_layout_);
  const common::row_t *ids_ptr =
      NodeAccessors::GetNeighborIDsPtr(node_block, node_layout_);

  if (!ids_ptr) {
    // delete[] node_block;
    return false;
  }

  neighbor_row_ids_out.reserve(count);
  for (uint16_t i = 0; i < count; ++i) {
    // Filter out uninitialized/invalid neighbor IDs, which might be marked with
    // MAX row_t.
    if (ids_ptr[i] != common::NumericLimits<common::row_t>::Maximum()) {
      neighbor_row_ids_out.push_back(ids_ptr[i]);
    }
  }
  // delete[] node_block; // Placeholder cleanup
  return true;
}

void GraphManager::RobustPrune(
    common::IndexPointer node_to_connect, const float *node_vector_data,
    std::vector<common::row_t> &candidate_row_ids,
    const LmDiskannConfig &config_param) { // Use config_param for R, alpha
  // Placeholder: Full RobustPrune is complex.
  // Requires:
  // - Access to vectors of all candidates (via GetNodeVector for each candidate
  // row_id).
  // - Distance calculations (using CalculateDistanceInternal or
  // diskann::core::calculate_distance).
  // - Pruning logic based on config_param.r (max degree) and
  // config_param.alpha.
  // - Modifying neighbor list of node_to_connect:
  //   - Get its data block (mutable) via StorageManager.
  //   - Use NodeAccessors::SetNeighborCount and
  //   NodeAccessors::GetNeighborIDsPtrMutable.
  //   - Mark block dirty with StorageManager.

  // Example of what might be needed (highly simplified):
  // std::vector<std::pair<float, common::row_t>> dist_candidates;
  // for (common::row_t cand_row_id : candidate_row_ids) {
  //   common::IndexPointer cand_ptr;
  //   if (TryGetNodePointer(cand_row_id, cand_ptr)) {
  //     dist_candidates.push_back({CalculateDistanceInternal(node_to_connect,
  //     cand_ptr), cand_row_id});
  //   }
  // }
  // Sort dist_candidates, apply alpha pruning, select top R.
  // candidate_row_ids should be updated with the final pruned list.

  throw common::NotImplementedException(
      "GraphManager::RobustPrune: Full implementation pending StorageManager "
      "and detailed pruning logic.");
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
      SetEntryPoint(common::IndexPointer{common::INVALID_BLOCK_ID, 0},
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
  auto map_size = rowid_to_node_ptr_map_.size();
  // common::RandomEngine from DuckDB uses NextRandomRange(min, max) inclusive
  // of max.
  common::idx_t random_idx =
      engine.NextRandomRange<common::idx_t>(0, map_size - 1);
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
  // GetNodeVector must be const
  if (!const_cast<GraphManager *>(this)->GetNodeVector(node1_ptr, vec1.data(),
                                                       config_.dimensions) ||
      !const_cast<GraphManager *>(this)->GetNodeVector(node2_ptr, vec2.data(),
                                                       config_.dimensions)) {
    // Or, make a non-const GetNodeVector if it truly doesn't modify, or pass
    // const_cast around GetNodeData_Placeholder. For now, assuming
    // GetNodeVector can be called like this for internal helpers. A cleaner way
    // would be to have GetNodeVector fill a pre-allocated vector.
    throw std::runtime_error(
        "CalculateDistanceInternal: Failed to get node vectors.");
  }
  return diskann::core::calculate_distance(
      vec1.data(), vec2.data(), config_.dimensions, config_.metric_type);
}

float GraphManager::CalculateDistanceInternal(const float *vec1,
                                              const float *vec2,
                                              common::idx_t dimensions) const {
  if (dimensions != config_.dimensions) {
    throw std::runtime_error("CalculateDistanceInternal: Dimension mismatch "
                             "for raw vector comparison.");
  }
  return diskann::core::CalculateDistance(vec1, vec2, config_.dimensions,
                                          config_.metric_type);
}

float GraphManager::CalculateDistanceInternal(common::IndexPointer node_ptr,
                                              const float *query_vec,
                                              common::idx_t dimensions) const {
  if (dimensions != config_.dimensions) {
    throw std::runtime_error("CalculateDistanceInternal: Dimension mismatch "
                             "for node vs query vector comparison.");
  }
  std::vector<float> node_vec(config_.dimensions);
  if (!const_cast<GraphManager *>(this)->GetNodeVector(
          node_ptr, node_vec.data(), config_.dimensions)) {
    throw std::runtime_error("CalculateDistanceInternal: Failed to get node "
                             "vector for comparison with query.");
  }
  return diskann::core::calculate_distance(
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
