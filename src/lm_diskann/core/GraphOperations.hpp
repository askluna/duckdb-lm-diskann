/**
 * @file GraphOperations.hpp
 * @brief Defines the GraphOperations class for managing graph structure,
 *        including insertions, deletions, and entry point management.
 */
#pragma once

#include "duckdb.hpp"
#include "duckdb/common/limits.hpp" // For NumericLimits
#include "duckdb/common/random_engine.hpp"
#include "duckdb/execution/index/index_pointer.hpp"
#include "index_config.hpp" // For LmDiskannConfig and NodeLayoutOffsets

// ADD these includes for the full definitions
#include "LmDiskannIndex.hpp"     // Defines ::diskann::duckdb::LmDiskannIndex
#include "LmDiskannScanState.hpp" // Defines ::duckdb::LmDiskannScanState

#include <utility> // For std::pair
#include <vector>

namespace diskann {
namespace core {

// Forward declarations
class GraphManager;

class GraphOperations {
public:
  /**
   * @brief Constructor for GraphOperations.
   * @param config The index configuration.
   * @param node_layout The calculated layout of nodes on disk.
   * @param node_manager Reference to the node manager for data access.
   * @param index_context Reference to the main index for context (e.g.,
   * search capabilities).
   */
  GraphOperations(const LmDiskannConfig &config,
                  const NodeLayoutOffsets &node_layout,
                  GraphManager &node_manager,
                  ::diskann::duckdb::LmDiskannIndex &index_context);

  /**
   * @brief Processes the insertion of a new node into the graph.
   * @details Finds neighbors, performs pruning, and updates graph structure.
   * @param new_node_rowid RowID of the new node.
   * @param new_node_ptr IndexPointer to the new node's block.
   * @param new_node_vector_float Pointer to the new node's vector data (as
   * float).
   */
  void InsertNode(::duckdb::row_t new_node_rowid,
                  ::duckdb::IndexPointer new_node_ptr,
                  const float *new_node_vector_float);

  /**
   * @brief Handles graph updates resulting from a node deletion.
   * @details For now, this primarily ensures the entry point is still valid.
   *          Future enhancements could include graph repair.
   * @param deleted_node_rowid The row_id of the node that was deleted.
   */
  void HandleNodeDeletion(::duckdb::row_t deleted_node_rowid);

  /**
   * @brief Selects a valid entry point for starting a search.
   * @details If the current entry point is invalid, selects a new one.
   * @param engine A random engine to use for selecting a random node if needed.
   * @return A valid row_t to be used as a search entry point.
   */
  ::duckdb::row_t SelectEntryPointForSearch(::duckdb::RandomEngine &engine);

  /**
   * @brief Gets the current graph entry point pointer.
   * @return IndexPointer to the entry point node. Can be invalid if no entry
   * point.
   */
  ::duckdb::IndexPointer GetGraphEntryPointPointer() const;

  /**
   * @brief Gets the row_id of the current graph entry point.
   * @return row_t of the entry point node. Can be
   * NumericLimits<row_t>::Maximum() if no entry point.
   */
  ::duckdb::row_t GetGraphEntryPointRowId() const;

  /**
   * @brief Sets the entry point state loaded from persisted metadata.
   * @param ptr The IndexPointer to the entry point node.
   * @param row_id The row_t of the entry point node.
   */
  void SetLoadedEntryPoint(::duckdb::IndexPointer ptr, ::duckdb::row_t row_id);

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
  void RobustPrune(::duckdb::row_t node_rowid, ::duckdb::IndexPointer node_ptr,
                   const float *node_vector_float,
                   std::vector<std::pair<float, ::duckdb::row_t>> &candidates,
                   bool is_new_node_prune);

  /**
   * @brief Performs a search to find candidate neighbors for a given vector.
   * @param target_vector_float The vector to find neighbors for.
   * @param K The number of candidates to find (typically L_insert or L_search).
   * @param scan_list_size The beam width for the search.
   * @return A vector of (distance, row_id) pairs for candidate neighbors.
   */
  std::vector<std::pair<float, ::duckdb::row_t>>
  SearchForCandidates(const float *target_vector_float, uint32_t K,
                      uint32_t scan_list_size);

  const LmDiskannConfig &config_;
  const NodeLayoutOffsets &node_layout_;
  GraphManager &node_manager_;
  ::diskann::duckdb::LmDiskannIndex
      &index_context_; // QUALIFIED HERE: Provides context, e.g., PerformSearch
                       // capability

  // --- Entry Point State (Owned by GraphOperations) ---
  /** @brief Pointer to the current graph entry point node. Initialized to
   * invalid. */
  ::duckdb::IndexPointer graph_entry_point_ptr_;
  /** @brief Cached row_t of the entry point node. Initialized to invalid. */
  ::duckdb::row_t graph_entry_point_rowid_ =
      ::duckdb::NumericLimits<::duckdb::row_t>::Maximum();
};

} // namespace core
} // namespace diskann
