/**
 * @file GraphManager.cpp
 * @brief Implements the GraphManager class for managing node allocations,
 *        RowID mappings, and raw node data access for the LM-DiskANN index.
 */
#include "GraphManager.hpp"
#include "../common/types.hpp" // Common types
#include "distance.hpp"        // For ConvertToFloat and distance calcs
#include "index_config.hpp"    // For LmDiskannConfig, NodeLayoutOffsets

#include "ternary_quantization.hpp" // For WordsPerPlane
#include <cstring>

#include <map>
#include <memory> // For make_unique
#include <random> // For std::uniform_int_distribution
#include <vector>

namespace diskann {
namespace core {

GraphManager::GraphManager(const LmDiskannConfig &config,
                           const NodeLayoutOffsets &node_layout,
                           common::idx_t block_size_bytes)
    : config_(config), node_layout_(node_layout),
      block_size_bytes_(block_size_bytes),
      graph_entry_point_rowid_(
          common::NumericLimits<common::row_t>::Maximum()) // Initialize
                                                           // invalid
{
  // Intent: FixedSizeAllocator was initialized here using BufferManager and
  // block_size_bytes_. GraphManager no longer directly manages the allocator.
  // This responsibility will shift to StorageManager, using IFileSystem or
  // IShadowStorageService.
  graph_entry_point_ptr_ = common::IndexPointer(); // Initialize invalid
}

void GraphManager::InitializeEmptyGraph(
    const LmDiskannConfig
        &config, // Passed config might be used to re-init allocator if needed
    common::IndexPointer &entry_point_ptr_out,
    common::row_t &entry_point_rowid_out) {
  // Intent: The FixedSizeAllocator was initialized here for a new index.
  // This will be handled by StorageManager in the future.
  rowid_to_node_ptr_map_.clear();
  graph_entry_point_ptr_ = common::IndexPointer();
  graph_entry_point_rowid_ = common::NumericLimits<common::row_t>::Maximum();

  entry_point_ptr_out = graph_entry_point_ptr_;
  entry_point_rowid_out = graph_entry_point_rowid_;
}

bool GraphManager::AddNode(common::row_t row_id, const float *vector_data,
                           common::idx_t dimensions,
                           common::IndexPointer &node_ptr_out) {
  // TODO: Implement AddNode logic. This will involve:
  // 1. Requesting node allocation from StorageManager (which returns an
  // IndexPointer).
  // 2. Storing vector_data at the location indicated by IndexPointer (via
  // StorageManager).
  // 3. Updating rowid_to_node_ptr_map_.
  // 4. Finding candidate neighbors (perhaps using ISearcher).
  // 5. Performing robust pruning.
  // 6. Connecting the new node by updating neighbor lists (via StorageManager
  // for persistence).
  throw common::NotImplementedException(
      "GraphManager::AddNode not fully implemented after allocator removal.");
  return false;
}

// Helper function to get node data - THIS IS A TEMPORARY STAND-IN
// In the future, GraphManager will request node data from StorageManager.
static common::const_data_ptr_t
GetNodeData_TEMP(common::IndexPointer node_ptr) {
  // THIS IS NOT A REAL IMPLEMENTATION AND WILL SEGFAULT OR RETURN GARBAGE
  // IT EXISTS ONLY TO ALLOW NodeAccessors TO BE CALLED IN
  // GetNodeVector/GetNeighbors WITHOUT THE ALLOCATOR. THIS MUST BE REPLACED BY
  // STORAGE MANAGER INTEGRATION.
  if (node_ptr.block_id == -1)
    return nullptr; // crude check for invalid
  unsigned char *some_memory = new unsigned char[1024]; // LEAK! Placeholder!
  return some_memory;
}

bool GraphManager::GetNodeVector(common::IndexPointer node_ptr,
                                 float *vector_out,
                                 common::idx_t dimensions) const {
  // Intent: Previously, this method retrieved node data using
  // allocator_->Get(node_ptr). Now, it needs to request data from
  // StorageManager. NodeAccessors still expects a data_ptr_t. throw
  // common::NotImplementedException(
  //     "GraphManager::GetNodeVector needs StorageManager integration.");
  // return false;

  // TEMPORARY: Using a placeholder to allow NodeAccessors to be called.
  // THIS WILL BE REPLACED.
  common::const_data_ptr_t node_block = GetNodeData_TEMP(node_ptr);
  if (!node_block)
    return false;

  try {
    const unsigned char *raw_vector =
        NodeAccessors::GetNodeVector(node_block, node_layout_);
    core::ConvertToFloat(raw_vector, vector_out, dimensions);
    // delete[] node_block; // Clean up temporary allocation if GetNodeData_TEMP
    // was used
    return true;
  } catch (...) {
    // delete[] node_block; // Clean up temporary allocation
    return false;
  }
}

bool GraphManager::GetNeighbors(
    common::IndexPointer node_ptr,
    std::vector<common::row_t> &neighbor_row_ids_out) const {
  neighbor_row_ids_out.clear();
  // Intent: Previously, this method retrieved node data using
  // allocator_->Get(node_ptr). Now, it needs to request data from
  // StorageManager. throw common::NotImplementedException(
  //     "GraphManager::GetNeighbors needs StorageManager integration.");
  // return false;

  // TEMPORARY: Using a placeholder to allow NodeAccessors to be called.
  // THIS WILL BE REPLACED.
  common::const_data_ptr_t node_block = GetNodeData_TEMP(node_ptr);
  if (!node_block)
    return false;

  try {
    uint16_t count = NodeAccessors::GetNeighborCount(node_block);
    const common::row_t *ids_ptr =
        NodeAccessors::GetNeighborIDsPtr(node_block, node_layout_);
    neighbor_row_ids_out.reserve(count);
    for (uint16_t i = 0; i < count; ++i) {
      if (ids_ptr[i] != common::NumericLimits<common::row_t>::Maximum()) {
        neighbor_row_ids_out.push_back(ids_ptr[i]);
      }
    }
    // delete[] node_block; // Clean up temporary allocation
    return true;
  } catch (...) {
    neighbor_row_ids_out.clear();
    // delete[] node_block; // Clean up temporary allocation
    return false;
  }
}

void GraphManager::RobustPrune(common::IndexPointer node_to_connect,
                               const float *node_vector_data,
                               std::vector<common::row_t> &candidate_row_ids,
                               const LmDiskannConfig &config) {
  // TODO: Copy and adapt logic from GraphOperations::RobustPrune
  // This will require access to neighbor vectors, likely via GetNodeVector.
  throw common::NotImplementedException(
      "GraphManager::RobustPrune not implemented yet after allocator removal.");
}

void GraphManager::SetEntryPoint(common::IndexPointer node_ptr,
                                 common::row_t row_id) {
  graph_entry_point_ptr_ = node_ptr;
  graph_entry_point_rowid_ = row_id;
}

common::IndexPointer GraphManager::GetEntryPointPointer() const {
  return graph_entry_point_ptr_;
}

common::row_t GraphManager::GetEntryPointRowId() const {
  if (graph_entry_point_rowid_ ==
      common::NumericLimits<common::row_t>::Maximum()) {
  }
  return graph_entry_point_rowid_;
}

void GraphManager::HandleNodeDeletion(common::row_t row_id) {
  if (row_id == graph_entry_point_rowid_) {
    graph_entry_point_ptr_ = common::IndexPointer();
    graph_entry_point_rowid_ = common::NumericLimits<common::row_t>::Maximum();
  }
}

void GraphManager::FreeNode(common::row_t row_id) {
  // Intent: Previously, this found the node_ptr for row_id and called
  // allocator_->Free(). Node deallocation is now the responsibility of
  // StorageManager. GraphManager might still need to update its
  // rowid_to_node_ptr_map_.
  auto iter = rowid_to_node_ptr_map_.find(row_id);
  if (iter != rowid_to_node_ptr_map_.end()) {
    // Comment: Original: allocator_->Free(iter->second);
    rowid_to_node_ptr_map_.erase(iter);
  } // else: node not found, nothing to do for this map.
  // No exception is thrown if row_id not found, consistent with original
  // FixedSizeAllocator behavior.
}

common::idx_t GraphManager::GetNodeCount() const {
  return rowid_to_node_ptr_map_.size();
}

common::idx_t GraphManager::GetInMemorySize() const {
  common::idx_t size = sizeof(*this);
  // Intent: Previously added allocator_->GetInMemorySize().
  // This is now tracked by StorageManager.
  size += rowid_to_node_ptr_map_.size() *
          (sizeof(common::row_t) + sizeof(common::IndexPointer) +
           sizeof(void *) * 2); // Approx map overhead
  size += sizeof(config_);      // Add size of stored config
  return size;
}

void GraphManager::Reset() {
  // Intent: Previously called allocator_->Reset().
  // Resetting storage is now StorageManager's responsibility.
  rowid_to_node_ptr_map_.clear();
  graph_entry_point_ptr_ = common::IndexPointer();
  graph_entry_point_rowid_ = common::NumericLimits<common::row_t>::Maximum();
}

bool GraphManager::TryGetNodePointer(common::row_t row_id,
                                     common::IndexPointer &node_ptr) const {
  auto iter = rowid_to_node_ptr_map_.find(row_id);
  if (iter != rowid_to_node_ptr_map_.end()) {
    node_ptr = iter->second;
    return true;
  }
  node_ptr = common::IndexPointer();
  return false;
}

/* Intent: GetNodeDataMutable and GetNodeData previously used
 * allocator_->Get(node_ptr) to return a raw data pointer to the node's block.
 * This responsibility now shifts to StorageManager, which will provide methods
 * to access node data, possibly returning a buffer handle or a copy, depending
 * on its implementation (IFileSystem vs IShadowStorageService). GraphManager
 * will call these StorageManager methods.
 */
// common::data_ptr_t
// GraphManager::GetNodeDataMutable(common::IndexPointer node_ptr) {
//   throw common::NotImplementedException(
//       "GraphManager::GetNodeDataMutable needs StorageManager integration.");
// }

// common::const_data_ptr_t
// GraphManager::GetNodeData(common::IndexPointer node_ptr) const {
//   throw common::NotImplementedException(
//       "GraphManager::GetNodeData needs StorageManager integration.");
// }

common::row_t GraphManager::GetRandomNodeID(common::RandomEngine &engine) {
  if (rowid_to_node_ptr_map_.empty()) {
    return common::NumericLimits<common::row_t>::Maximum();
  }
  std::uniform_int_distribution<common::idx_t> dist(
      0, rowid_to_node_ptr_map_.size() - 1);
  common::idx_t random_idx = dist(engine);
  auto iter = rowid_to_node_ptr_map_.begin();
  std::advance(iter, random_idx);
  return iter->first;
}

// Intent: AllocateBlockForNode previously used allocator_->New() to get a new
// block. Node allocation is now StorageManager's responsibility.
// common::IndexPointer GraphManager::AllocateBlockForNode() {
//   throw common::NotImplementedException(
//       "GraphManager::AllocateBlockForNode needs StorageManager
//       integration.");
// }

float GraphManager::CalculateDistance(common::row_t neighbor_row_id,
                                      const float *query_vector) {
  // TODO: Implement distance calculation. This will require:
  // 1. Getting the IndexPointer for neighbor_row_id (using TryGetNodePointer).
  // 2. Getting the vector for that IndexPointer (using GetNodeVector, which
  // needs StorageManager).
  // 3. Calling appropriate distance function from distance.hpp.
  throw common::NotImplementedException(
      "GraphManager::CalculateDistance helper not fully implemented after "
      "allocator removal.");
  return 0.0f;
}

} // namespace core
} // namespace diskann
