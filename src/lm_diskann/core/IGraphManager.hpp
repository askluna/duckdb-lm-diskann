#pragma once

#include "../common/types.hpp"
#include "index_config.hpp" // For LmDiskannConfig
#include <memory>           // For unique_ptr if implementations return it
#include <vector>

namespace diskann {
namespace core {

class IGraphManager {
public:
  virtual ~IGraphManager() = default;

  // Initializes an empty graph structure.
  virtual void InitializeEmptyGraph(
      const LmDiskannConfig &config,
      common::IndexPointer
          &entry_point_ptr_out, // Output: initial (invalid) entry point ptr
      common::row_t
          &entry_point_rowid_out // Output: initial (invalid) entry point rowid
      ) = 0;

  // Allocates space for a new node and stores its vector data.
  virtual bool
  AddNode(common::row_t row_id,
          const float *vector_data,          // Assuming float for now, could be
                                             // templated or use unsigned char*
          common::idx_t dimensions,          // From config
          common::IndexPointer &node_ptr_out // Output: pointer to the new node
          ) = 0;

  // Retrieves the vector for a given node.
  virtual bool GetNodeVector(common::IndexPointer node_ptr,
                             float *vector_out,       // Output buffer
                             common::idx_t dimensions // From config
  ) const = 0;

  // Retrieves the neighbors for a given node.
  // Returns RowIDs of neighbors.
  virtual bool
  GetNeighbors(common::IndexPointer node_ptr,
               std::vector<common::row_t> &neighbor_row_ids_out) const = 0;

  // Prunes a list of candidate neighbors for a node and updates its
  // connections.
  virtual void RobustPrune(
      common::IndexPointer
          node_to_connect, // The node whose neighbors are being pruned
      const float *node_vector_data, // Vector data for node_to_connect
      std::vector<common::row_t>
          &candidate_row_ids, // In/Out: candidates, pruned to final neighbors
      const LmDiskannConfig &config
      // Potentially needs access to other node vectors via this interface or
      // another service
      ) = 0;

  // Sets the graph's entry point.
  virtual void SetEntryPoint(common::IndexPointer node_ptr,
                             common::row_t row_id) = 0;
  virtual common::IndexPointer GetEntryPointPointer() const = 0;
  virtual common::row_t GetEntryPointRowId() const = 0;

  // Handles the logical deletion of a node (e.g., removing from consideration
  // as entry point).
  virtual void HandleNodeDeletion(common::row_t row_id) = 0;

  // Frees the node associated with a row_id (used during actual deletion by
  // LmDiskannIndex)
  virtual void FreeNode(common::row_t row_id) = 0;

  // Returns the number of active nodes in the graph.
  virtual common::idx_t GetNodeCount() const = 0;

  // Returns the estimated in-memory size used by the graph manager.
  virtual common::idx_t GetInMemorySize() const = 0;

  // Resets the graph manager to an empty state (e.g., for CommitDrop).
  virtual void Reset() = 0;

  // Potentially methods to get mutable/const data pointers if
  // Orchestrator/Searcher need direct access virtual common::data_ptr_t
  // GetNodeDataMutable(common::IndexPointer node_ptr) = 0; virtual
  // common::const_data_ptr_t GetNodeData(common::IndexPointer node_ptr) const =
  // 0;
};

} // namespace core
} // namespace diskann