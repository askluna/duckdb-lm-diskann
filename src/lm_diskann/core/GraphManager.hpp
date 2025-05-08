/**
 * @file GraphManager.hpp
 * @brief Defines the GraphManager class for managing node allocations,
 *        RowID mappings, and raw node data access.
 */
#pragma once

#include "../common/types.hpp" // Common types
#include "IGraphManager.hpp"   // Inherit from the interface
#include "index_config.hpp"    // For LmDiskannConfig, NodeLayoutOffsets

// DuckDB includes previously needed for BufferManager/FixedSizeAllocator are
// removed as these are no longer direct responsibilities of GraphManager.

#include <map>
#include <memory>
#include <vector>

namespace duckdb { // Forward declare instead of include where possible
class RandomEngine;
} // namespace duckdb

namespace diskann {
namespace core {

class GraphManager : public IGraphManager { // Inherit from interface
public:
  /**
   * @brief Constructor for GraphManager.
   * @param config The index configuration.
   * @param node_layout Pre-calculated node layout offsets.
   * @param block_size_bytes The aligned size of each node block (may be used
   * for layout interpretation).
   */
  GraphManager(const LmDiskannConfig &config,
               const NodeLayoutOffsets &node_layout,
               common::idx_t block_size_bytes);

  ~GraphManager() override = default;

  // --- IGraphManager Interface Implementation ---
  void InitializeEmptyGraph(
      const LmDiskannConfig &
          config, // Config might be needed here too, or stored from constructor
      common::IndexPointer &entry_point_ptr_out,
      common::row_t &entry_point_rowid_out) override;

  bool AddNode(
      common::row_t row_id, const float *vector_data,
      common::idx_t dimensions, // Pass explicitly or get from stored config_
      common::IndexPointer &node_ptr_out) override;

  bool GetNodeVector(
      common::IndexPointer node_ptr, float *vector_out,
      common::idx_t dimensions // Pass explicitly or get from stored config_
  ) const override;

  bool
  GetNeighbors(common::IndexPointer node_ptr,
               std::vector<common::row_t> &neighbor_row_ids_out) const override;

  void RobustPrune(common::IndexPointer node_to_connect,
                   const float *node_vector_data,
                   std::vector<common::row_t> &candidate_row_ids,
                   const LmDiskannConfig &config // Config needed for alpha, R
                   ) override;

  void SetEntryPoint(common::IndexPointer node_ptr,
                     common::row_t row_id) override;
  common::IndexPointer GetEntryPointPointer() const override;
  common::row_t GetEntryPointRowId() const override;
  void HandleNodeDeletion(common::row_t row_id) override;
  void FreeNode(common::row_t row_id) override;
  common::idx_t GetNodeCount() const override;
  common::idx_t GetInMemorySize() const override;
  void Reset() override;

  // --- Methods that previously relied on direct allocator access ---
  // These will need to be re-implemented or their callers adapted to use
  // StorageManager or other services.

  bool TryGetNodePointer(common::row_t row_id,
                         common::IndexPointer &node_ptr) const;

  // ::duckdb::data_ptr_t GetNodeDataMutable(common::IndexPointer node_ptr); //
  // To be re-evaluated
  // ::duckdb::const_data_ptr_t GetNodeData(common::IndexPointer node_ptr)
  // const; // To be re-evaluated

  common::row_t GetRandomNodeID(common::RandomEngine &engine);

private:
  /// Configuration for the index.
  LmDiskannConfig config_;
  /// Pre-calculated layout offsets.
  NodeLayoutOffsets node_layout_;
  /// Aligned size of node blocks (potentially used for layout interpretation).
  common::idx_t block_size_bytes_;
  /// Maps DuckDB row_t to the IndexPointer of the node block.
  /// Future: This mapping might live in StorageManager or be part of what
  /// IShadowStorageService provides.
  std::map<common::row_t, common::IndexPointer> rowid_to_node_ptr_map_;
  /// Pointer to the graph's entry point node.
  common::IndexPointer graph_entry_point_ptr_;
  /// RowID of the graph's entry point node.
  common::row_t graph_entry_point_rowid_ =
      common::NumericLimits<common::row_t>::Maximum(); // Initialize to
                                                       // invalid

  // Helper to allocate node without adding to map (used by AddNode)
  // common::IndexPointer AllocateBlockForNode(); // This was
  // allocator-dependent

  // Helper for RobustPrune - needs access to distances/vectors
  float
  CalculateDistance(common::row_t neighbor_row_id,
                    const float *query_vector); // This likely needs vector
                                                // access via an interface
};

// NodeAccessors class can remain here or be moved to NodeAccessors.hpp/.cpp
// Update it to use common::idx_t, common::row_t, unsigned char* where
// appropriate
class NodeAccessors {
public:
  static void InitializeNodeBlock(::duckdb::data_ptr_t node_block_ptr,
                                  common::idx_t block_size_bytes);
  static uint16_t GetNeighborCount(::duckdb::const_data_ptr_t node_block_ptr);
  static void SetNeighborCount(::duckdb::data_ptr_t node_block_ptr,
                               uint16_t count);
  static const unsigned char *
  GetNodeVector(::duckdb::const_data_ptr_t node_block_ptr,
                const NodeLayoutOffsets &layout); // Return unsigned char*
  static unsigned char *GetNodeVectorMutable(
      ::duckdb::data_ptr_t node_block_ptr,
      const NodeLayoutOffsets &layout); // Return unsigned char*
  static const common::row_t *
  GetNeighborIDsPtr(::duckdb::const_data_ptr_t node_block_ptr,
                    const NodeLayoutOffsets &layout); // Return common::row_t*
  static common::row_t *GetNeighborIDsPtrMutable(
      ::duckdb::data_ptr_t node_block_ptr,
      const NodeLayoutOffsets &layout); // Return common::row_t*
  static TernaryPlanesView
  GetNeighborTernaryPlanes(::duckdb::const_data_ptr_t node_block_ptr,
                           const NodeLayoutOffsets &layout,
                           uint16_t neighbor_idx, common::idx_t dimensions);
  static MutableTernaryPlanesView GetNeighborTernaryPlanesMutable(
      ::duckdb::data_ptr_t node_block_ptr, const NodeLayoutOffsets &layout,
      uint16_t neighbor_idx, common::idx_t dimensions);
};

} // namespace core
} // namespace diskann
