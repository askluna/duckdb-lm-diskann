/**
 * @file GraphOperations.hpp
 * @brief Defines the GraphOperations class for managing graph structure,
 *        including insertions, deletions, and entry point management.
 */
#pragma once

#include "config.hpp" // For LmDiskannConfig and NodeLayoutOffsets
#include "duckdb.hpp"
#include "duckdb/execution/index/index_pointer.hpp"

#include <utility> // For std::pair
#include <vector>

namespace duckdb {

// Forward declarations
class NodeManager;    // Updated from LmDiskannNodeManager
class LmDiskannIndex; // For calling PerformSearch or other index context
class LmDiskannScanState;
class RandomEngine;

class GraphOperations {
public:
  /**
   * @brief Constructor for GraphOperations.
   * @param config The index configuration.
   * @param node_layout The calculated layout of nodes on disk.
   * @param node_manager Reference to the node manager for data access.
   * @param index_context Reference to the main index for context (e.g.,
   * search).
   * @param graph_entry_point_rowid Reference to the index's current entry point
   * rowid.
   */
  GraphOperations(
      const LmDiskannConfig &config, const NodeLayoutOffsets &node_layout,
      NodeManager &node_manager, LmDiskannIndex &index_context,
      row_t &graph_entry_point_rowid,     // Managed by LmDiskannIndex, opaquely
                                          // updated here
      IndexPointer &graph_entry_point_ptr // Managed by LmDiskannIndex, opaquely
                                          // updated here
  );

  /**
   * @brief Processes the insertion of a new node into the graph.
   * @details Finds neighbors, performs pruning, and updates graph structure.
   * @param new_node_rowid RowID of the new node.
   * @param new_node_ptr IndexPointer to the new node's block.
   * @param new_node_vector_float Pointer to the new node's vector data (as
   * float).
   */
  void InsertNode(row_t new_node_rowid, IndexPointer new_node_ptr,
                  const float *new_node_vector_float);

  /**
   * @brief Handles graph updates resulting from a node deletion.
   * @details For now, this primarily ensures the entry point is still valid.
   *          Future enhancements could include graph repair.
   * @param deleted_node_rowid The row_id of the node that was deleted.
   */
  void HandleNodeDeletion(row_t deleted_node_rowid);

  /**
   * @brief Selects a valid entry point for starting a search.
   * @details If the current entry point is invalid, selects a new one.
   * @param engine A random engine to use for selecting a random node if needed.
   * @return A valid row_t to be used as a search entry point.
   */
  row_t SelectEntryPointForSearch(RandomEngine &engine);

private:
  /**
   * @brief Applies the Robust Prune algorithm to select neighbors for a node.
   * @param node_rowid RowID of the node being pruned.
   * @param node_ptr IndexPointer to the node's block.
   * @param node_vector_float Pointer to the node's vector data (as float).
   * @param candidates Initial list of potential neighbors (distance, row_id
   * pairs).
   * @param is_new_node_prune True if this is pruning for a brand new node
   * (slightly different logic for existing neighbors).
   */
  void RobustPrune(row_t node_rowid, IndexPointer node_ptr,
                   const float *node_vector_float,
                   std::vector<std::pair<float, row_t>> &candidates,
                   bool is_new_node_prune);

  /**
   * @brief Performs a search to find candidate neighbors for a given vector.
   * @param target_vector_float The vector to find neighbors for.
   * @param K The number of candidates to find (typically L_insert or L_search).
   * @param scan_list_size The beam width for the search.
   * @return A vector of (distance, row_id) pairs for candidate neighbors.
   */
  std::vector<std::pair<float, row_t>>
  SearchForCandidates(const float *target_vector_float, uint32_t K,
                      uint32_t scan_list_size);

  const LmDiskannConfig &config_;
  const NodeLayoutOffsets &node_layout_;
  NodeManager &node_manager_; // Updated from LmDiskannNodeManager
  LmDiskannIndex
      &index_context_; // Provides context, e.g., PerformSearch capability
  row_t &graph_entry_point_rowid_;      // Reference to LmDiskannIndex's member
  IndexPointer &graph_entry_point_ptr_; // Reference to LmDiskannIndex's member
};

} // namespace duckdb
