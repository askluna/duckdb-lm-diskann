/**
 * @file GraphManager.hpp
 * @brief Defines the GraphManager class for managing node allocations,
 *        RowID mappings, and raw node data access.
 */
#pragma once

#include "../common/ann.hpp"
#include "../common/duckdb_types.hpp"
#include "IGraphManager.hpp" // Inherit from the interface
#include "index_config.hpp"  // For LmDiskannConfig, NodeLayoutOffsets, TernaryPlanesView, MutableTernaryPlanesView

// Forward declare interfaces for dependencies
namespace diskann {
namespace core {
class IStorageManager; // Forward declaration
class ISearcher;       // Forward declaration
} // namespace core
} // namespace diskann

// DuckDB includes previously needed for BufferManager/FixedSizeAllocator are
// removed as these are no longer direct responsibilities of GraphManager.

#include <map>
#include <memory> // For std::unique_ptr if needed by future implementations
#include <vector>

namespace diskann {
namespace core {

// Forward declare for LmDiskannConfig
struct LmDiskannConfig;
struct NodeLayoutOffsets;
struct TernaryPlanesView;        // Assuming this is defined elsewhere or will be.
struct MutableTernaryPlanesView; // Defined in index_config.hpp

class GraphManager : public IGraphManager {
	public:
	/**
	 * @brief Constructor for GraphManager.
	 * @param config The index configuration. Copied internally.
	 * @param node_layout Pre-calculated node layout offsets. Copied internally.
	 * @param block_size_bytes The aligned size of each node block, as determined
	 * by StorageManager.
	 * @param storage_manager Raw pointer to the storage manager instance. Must
	 * outlive GraphManager.
	 * @param searcher Raw pointer to the searcher instance. Must outlive
	 * GraphManager.
	 */
	GraphManager(const LmDiskannConfig &config, const NodeLayoutOffsets &node_layout, common::idx_t block_size_bytes,
	             IStorageManager *storage_manager, // Added
	             ISearcher *searcher);             // Added

	~GraphManager() override = default;

	// --- IGraphManager Interface Implementation ---
	void InitializeEmptyGraph(const LmDiskannConfig &config_param, // Passed for potential re-init scenarios
	                          common::IndexPointer &entry_point_ptr_out, common::row_t &entry_point_rowid_out) override;

	bool AddNode(common::row_t row_id, const float *vector_data,
	             common::idx_t dimensions,          // Expected dimensions, checked against config_
	             common::IndexPointer &node_ptr_out // Output: pointer to the new node block
	             ) override;

	bool GetNodeVector(common::IndexPointer node_ptr, float *vector_out, // Output buffer
	                   common::idx_t dimensions                          // Expected dimensions, checked against config_
	) const override;

	bool GetNeighbors(common::IndexPointer node_ptr,
	                  std::vector<common::row_t> &neighbor_row_ids_out // Output vector
	) const override;

	void RobustPrune(common::IndexPointer node_to_connect, const float *node_vector_data,
	                 std::vector<common::row_t> &candidate_row_ids, // In/Out candidates
	                 const LmDiskannConfig &config_param            // Passed for pruning params (R, alpha)
	                 ) override;

	void SetEntryPoint(common::IndexPointer node_ptr, common::row_t row_id) override;
	common::IndexPointer GetEntryPointPointer() const override;
	common::row_t GetEntryPointRowId() const override;
	common::row_t SelectEntryPointForSearch(common::RandomEngine &engine) override;
	void HandleNodeDeletion(common::row_t row_id) override;
	void FreeNode(common::row_t row_id) override;
	common::idx_t GetNodeCount() const override;
	common::idx_t GetInMemorySize() const override;
	void Reset() override;

	// --- Non-interface public methods ---

	/**
	 * @brief Tries to get the IndexPointer for a given row_id from the internal
	 * map.
	 * @param row_id The row_id to look up.
	 * @param node_ptr_out Output: The IndexPointer if found, else an invalid
	 * pointer.
	 * @return True if found, false otherwise.
	 */
	bool TryGetNodePointer(common::row_t row_id, common::IndexPointer &node_ptr_out) const override;

	/**
	 * @brief Gets a random node ID from the currently managed nodes.
	 * @param engine Random number generator.
	 * @return A valid row_t if nodes exist, else NumericLimits<row_t>::Maximum().
	 */
	common::row_t GetRandomNodeID(common::RandomEngine &engine); // Not virtual, no override

	private:
	/// Configuration for the index (const to ensure it's set at construction and
	/// immutable).
	const LmDiskannConfig config_;
	/// Pre-calculated layout offsets (const and immutable post-construction).
	const NodeLayoutOffsets node_layout_;
	/// Aligned size of node blocks (const, set by StorageManager via
	/// Coordinator).
	const common::idx_t block_size_bytes_;
	/// Raw pointer to the storage manager. Not owned by GraphManager.
	IStorageManager *storage_manager_; // Added
	/// Raw pointer to the searcher. Not owned by GraphManager.
	ISearcher *searcher_; // Added

	/// Maps DuckDB row_t to the IndexPointer of the node block. This is the
	/// primary in-memory graph structure.
	std::map<common::row_t, common::IndexPointer> rowid_to_node_ptr_map_;

	/// Pointer to the graph's entry point node block.
	common::IndexPointer graph_entry_point_ptr_;
	/// RowID of the graph's entry point node.
	common::row_t graph_entry_point_rowid_;

	// Internal helper for distance calculations, ensuring consistent use of
	// config_ members. Overloads allow flexibility in how vectors are provided.
	float CalculateDistanceInternal(common::IndexPointer node1_ptr, common::IndexPointer node2_ptr) const;
	float CalculateDistanceInternal(const float *vec1, const float *vec2, common::idx_t dimensions) const;
	float CalculateDistanceInternal(common::IndexPointer node_ptr, const float *query_vec,
	                                common::idx_t dimensions) const;

	// Helper to compress a float32 vector for edge storage (e.g. ternary planes)
	// Returns true on success, false on failure (e.g. unsupported, buffer too
	// small)
	bool CompressFloatVectorForEdge(const float *float_vector_in, common::data_ptr_t compressed_edge_data_out,
	                                common::idx_t dimensions) const;
};

} // namespace core
} // namespace diskann
