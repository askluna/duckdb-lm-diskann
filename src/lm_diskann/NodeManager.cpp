/**
 * @file NodeManager.cpp
 * @brief Implements the NodeManager class for managing node allocations,
 *        RowID mappings, and raw node data access for the LM-DiskANN index.
 */
#include "NodeManager.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/common/printer.hpp"       // For optional warnings/debug
#include "duckdb/common/random_engine.hpp" // For RandomEngine
#include "duckdb/storage/storage_info.hpp" // For Storage::SECTOR_SIZE (not directly used here but often related)

#include <random> // For std::uniform_int_distribution

namespace duckdb {

NodeManager::NodeManager(BufferManager &buffer_manager, idx_t block_size_bytes)
    : allocator_(
          make_uniq<FixedSizeAllocator>(buffer_manager, block_size_bytes)) {
  if (block_size_bytes == 0 || block_size_bytes % Storage::SECTOR_SIZE != 0) {
    // This check might be more appropriate in LmDiskannIndex constructor or
    // config validation For now, adding a basic sanity check for allocator
    // construction. throw InternalException("NodeManager: block_size_bytes must
    // be non-zero and a multiple of SECTOR_SIZE.");
  }
}

IndexPointer NodeManager::AllocateNode(row_t row_id) {
  if (!allocator_) {
    throw InternalException(
        "NodeManager::AllocateNode: Allocator not initialized.");
  }

  auto it = rowid_to_node_ptr_map_.find(row_id);
  if (it != rowid_to_node_ptr_map_.end()) {
    // Node with this row_id already exists in the map.
    // This could be a reused row_id after deletions, or an attempt to
    // double-insert. For now, let's assume if it's in the map, we return its
    // pointer. LmDiskannIndex::AllocateNode had a more nuanced check for this.
    // Printer::Print(StringUtil::Format("Warning: NodeManager::AllocateNode
    // called for existing row_id %lld. Reusing block.", row_id));
    return it->second;
  }

  IndexPointer new_node_ptr = allocator_->New();
  if (new_node_ptr.Get() == 0) {
    throw InternalException("NodeManager::AllocateNode: FixedSizeAllocator "
                            "failed to allocate a new block.");
  }
  rowid_to_node_ptr_map_[row_id] = new_node_ptr;
  return new_node_ptr;
}

void NodeManager::FreeNode(row_t row_id) {
  if (!allocator_) {
    // Optionally, allow this if map might exist without allocator (e.g. during
    // shutdown sequence) For now, assume allocator must exist if we are
    // freeing. throw InternalException("NodeManager::FreeNode: Allocator not
    // initialized.");
    return;
  }
  auto it = rowid_to_node_ptr_map_.find(row_id);
  if (it != rowid_to_node_ptr_map_.end()) {
    allocator_->Free(it->second);
    rowid_to_node_ptr_map_.erase(it);
  } else {
    // Optional: Log warning if trying to free a non-existent node mapping
    // Printer::Print(StringUtil::Format("Warning: NodeManager::FreeNode called
    // for non-existent row_id %lld.", row_id));
  }
}

bool NodeManager::TryGetNodePointer(row_t row_id,
                                    IndexPointer &node_ptr) const {
  auto it = rowid_to_node_ptr_map_.find(row_id);
  if (it != rowid_to_node_ptr_map_.end()) {
    node_ptr = it->second;
    // Optionally, could add a check here: if (!allocator_ ||
    // !allocator_->Has(node_ptr)) return false;
    return true;
  }
  node_ptr.Clear();
  return false;
}

data_ptr_t NodeManager::GetNodeDataMutable(IndexPointer node_ptr) {
  if (node_ptr.Get() == 0) {
    throw IOException("NodeManager::GetNodeDataMutable: Invalid (null) node "
                      "pointer provided.");
  }
  if (!allocator_) {
    throw InternalException(
        "NodeManager::GetNodeDataMutable: Allocator not initialized.");
  }
  return allocator_->Get(node_ptr, true); // Get mutable pointer, mark dirty
}

const_data_ptr_t NodeManager::GetNodeData(IndexPointer node_ptr) const {
  if (node_ptr.Get() == 0) {
    throw IOException(
        "NodeManager::GetNodeData: Invalid (null) node pointer provided.");
  }
  if (!allocator_) {
    throw InternalException(
        "NodeManager::GetNodeData: Allocator not initialized.");
  }
  return allocator_->Get(node_ptr,
                         false); // Get const pointer, don't mark dirty
}

void NodeManager::Reset() {
  if (allocator_) {
    allocator_->Reset();
  }
  rowid_to_node_ptr_map_.clear();
}

idx_t NodeManager::GetInMemorySize() const {
  idx_t size = 0;
  if (allocator_) {
    size += allocator_->GetInMemorySize();
  }
  // Estimate map overhead: roughly (sizeof(row_t) + sizeof(IndexPointer) +
  // map_node_overhead) * count Using a common estimate for map node overhead
  // (e.g., 3 * sizeof(void*))
  size += rowid_to_node_ptr_map_.size() *
          (sizeof(row_t) + sizeof(IndexPointer) + (3 * sizeof(void *)));
  return size;
}

FixedSizeAllocator &NodeManager::GetAllocator() {
  if (!allocator_) {
    throw InternalException(
        "NodeManager::GetAllocator: Allocator not initialized.");
  }
  return *allocator_;
}

idx_t NodeManager::GetNodeCount() const {
  return rowid_to_node_ptr_map_.size();
}

row_t NodeManager::GetRandomNodeID(RandomEngine &engine) {
  if (rowid_to_node_ptr_map_.empty()) {
    return NumericLimits<row_t>::Maximum();
  }
  // Use NextRandomInteger() and modulo for a random index
  idx_t random_idx = engine.NextRandomInteger() % rowid_to_node_ptr_map_.size();
  auto it = rowid_to_node_ptr_map_.begin();
  std::advance(it, random_idx);
  return it->first;
}

} // namespace duckdb