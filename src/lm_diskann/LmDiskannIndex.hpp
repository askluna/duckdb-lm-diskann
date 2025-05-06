/**
 * @file LmDiskannIndex.hpp
 * @brief Contains the main LmDiskannIndex class definition.
 */
#pragma once

// Main DuckDB includes needed by the index class itself
#include "duckdb.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/execution/index/bound_index.hpp"
#include "duckdb/execution/index/fixed_size_allocator.hpp" // Include FixedSizeAllocator header
#include "duckdb/execution/index/index_pointer.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/parser/parsed_data/create_index_info.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/storage/index_storage_info.hpp"
#include "duckdb/storage/table_io_manager.hpp"

// Include headers for the refactored components
#include "config.hpp" // Include the new config header
#include "state.hpp"  // For scan state definition

#include <map>    // Using std::map for in-memory RowID mapping for now
#include <memory> // For unique_ptr
#include <string>
#include <vector>

namespace duckdb {

// Forward declarations for components used internally
// class FixedSizeAllocator; // Now included
class ClientContext;
class DataChunk;
class Vector;
class BufferHandle;
// class ART; // If using ART for mapping

// --- Helper Struct for DuckDB Integration State --- //
/**
 * @brief Holds references and objects related to DuckDB integration.
 * @details Bundles database references, IO manager, allocator, etc.
 */
struct LmDiskannDBState {
  /**
   * @brief Constructor for LmDiskannDBState.
   * @param db_ref Reference to the AttachedDatabase.
   * @param io_manager Reference to the TableIOManager.
   * @param type_ref The logical type of the indexed column.
   */
  LmDiskannDBState(AttachedDatabase &db_ref, TableIOManager &io_manager,
                   const LogicalType &type_ref)
      : db(db_ref), table_io_manager(io_manager),
        indexed_column_type(type_ref) {}

  AttachedDatabase &db;             // Reference to the database instance.
  TableIOManager &table_io_manager; // Reference to DuckDB's IO manager.
  LogicalType indexed_column_type;  // The full logical type of the indexed
                                    // column (e.g., ARRAY(FLOAT, 128)).
  unique_ptr<FixedSizeAllocator>
      allocator; // Manages disk blocks for index nodes.
  IndexPointer
      metadata_ptr; // Pointer to the block holding persistent index metadata.
};

// --- Core LmDiskannIndex Class --- //
/**
 * @brief Orchestrates LM-DiskANN operations.
 * @details Interfaces with DuckDB's index system and delegates tasks to
 *          specialized modules (config, node, storage, search, distance).
 */
class LmDiskannIndex : public BoundIndex {
public:
  /**
   * @brief Type name used for registration and in CREATE INDEX ... USING
   * LM_DISKANN.
   */
  static constexpr const char *TYPE_NAME = "LM_DISKANN";

  /**
   * @brief Constructor: Initializes parameters, storage, and loads/creates the
   * index.
   * @param name Index name.
   * @param index_constraint_type Type of index constraint (UNIQUE, PRIMARY_KEY,
   * FOREIGN_KEY, NONE).
   * @param column_ids List of physical column identifiers bound to the index
   * expressions.
   * @param table_io_manager The table's IO manager.
   * @param unbound_expressions The unbound expressions used to create the
   * index.
   * @param db The attached database instance.
   * @param options Parsed WITH clause options.
   * @param storage_info Information about existing index storage (if loading).
   * @param estimated_cardinality Estimated number of rows in the table.
   */
  LmDiskannIndex(const string &name, IndexConstraintType index_constraint_type,
                 const vector<column_t> &column_ids,
                 TableIOManager &table_io_manager,
                 const vector<unique_ptr<Expression>> &unbound_expressions,
                 AttachedDatabase &db,
                 const case_insensitive_map_t<Value> &options,
                 const IndexStorageInfo &storage_info,
                 idx_t estimated_cardinality);

  /**
   * @brief Destructor
   */
  ~LmDiskannIndex() override;

  // --- Overridden BoundIndex Virtual Methods --- //
  /** @brief Appends data to the index. Called during bulk loading. */
  ErrorData Append(IndexLock &lock, DataChunk &entries,
                   Vector &row_identifiers) override;
  /** @brief Commits a drop operation, freeing allocated resources. */
  void CommitDrop(IndexLock &index_lock) override;
  /** @brief Deletes entries from the index. */
  void Delete(IndexLock &lock, DataChunk &entries,
              Vector &row_identifiers) override;
  /** @brief Inserts data into the index. */
  ErrorData Insert(IndexLock &lock, DataChunk &data, Vector &row_ids) override;
  /** @brief Retrieves storage information (metadata pointer, allocator stats).
   */
  IndexStorageInfo GetStorageInfo(const bool get_buffers);
  /** @brief Gets the estimated in-memory size of the index structures. */
  idx_t GetInMemorySize();
  /** @brief Merges another index into this one (not implemented). */
  bool MergeIndexes(IndexLock &state, BoundIndex &other_index) override;
  /** @brief Performs vacuuming operations (e.g., processing delete queue). */
  void Vacuum(IndexLock &state) override;
  /** @brief Verifies index integrity and returns a string representation
   * (verification TBD). */
  string VerifyAndToString(IndexLock &state, const bool only_verify) override;
  /** @brief Verifies allocator integrity (delegated). */
  void VerifyAllocations(IndexLock &state) override;
  /** @brief Generates a constraint violation message (not applicable). */
  string GetConstraintViolationMessage(VerifyExistenceType verify_type,
                                       idx_t failed_index,
                                       DataChunk &input) override;

  // --- LM-DiskANN Specific Methods for Scanning --- //
  /**
   * @brief Initializes the state for an index scan (k-NN search).
   * @param context The client context.
   * @param query_vector The query vector.
   * @param k The number of nearest neighbors to find.
   * @return A unique pointer to the initialized scan state.
   */
  unique_ptr<IndexScanState>
  InitializeScan(ClientContext &context, const Vector &query_vector, idx_t k);
  /**
   * @brief Performs one step of the index scan, filling the result vector.
   * @param state The current scan state.
   * @param result The output vector to store resulting RowIDs.
   * @return The number of results produced in this step.
   */
  idx_t Scan(IndexScanState &state, Vector &result);

  // --- Public Accessors (Potentially needed by other modules/friends) --- //
  /** @brief Get the attached database reference. */
  AttachedDatabase &GetAttachedDatabase() const { return db_state_.db; }
  /** @brief Get the fixed size allocator reference. */
  FixedSizeAllocator &GetAllocator() const { return *db_state_.allocator; }
  /** @brief Get the calculated node layout offsets. */
  const NodeLayoutOffsets &GetNodeLayout() const { return node_layout_; }
  /** @brief Get the vector dimensions. */
  idx_t GetDimensions() const { return config_.dimensions; }
  /** @brief Get the size of the compressed ternary edge representation. */
  idx_t GetEdgeVectorSizeBytes() const {
    return node_layout_.ternary_edge_size_bytes;
  }
  /** @brief Get the distance metric type. */
  LmDiskannMetricType GetMetricType() const { return config_.metric_type; }
  /** @brief Get the max neighbor degree (R). */
  uint32_t GetR() const { return config_.r; }
  /** @brief Get the alpha parameter for pruning. */
  float GetAlpha() const { return config_.alpha; }
  /** @brief Get the search list size. */
  uint32_t GetLSearch() const { return config_.l_search; }
  /** @brief Get the node vector type. */
  LmDiskannVectorType GetNodeVectorType() const {
    return config_.node_vector_type;
  }

private:
  friend class LmDiskannStorageHelper; // Allow storage helpers access
  /**
   * @brief Allows search access, passing config and db_state
   * @param scan_state The search state.
   * @param index The index instance.
   * @param config The index configuration.
   * @param find_exact_distances Flag for final re-ranking.
   */
  friend void PerformSearch(LmDiskannScanState &scan_state,
                            LmDiskannIndex &index,
                            const LmDiskannConfig &config,
                            bool find_exact_distances);

  // --- Core Parameters (Held in Config Struct) --- //
  //! @brief Parsed and validated configuration parameters.
  LmDiskannConfig config_;
  //! @brief Calculated layout based on config.
  NodeLayoutOffsets node_layout_;
  //! @brief Final aligned size of each node's block on disk.
  idx_t block_size_bytes_;

  // --- DuckDB Integration State --- //
  //! @brief Holds DuckDB-related references and state.
  LmDiskannDBState db_state_;

  //! @brief Internal format version (for metadata check).
  uint8_t format_version_;

  // --- RowID Mapping (In-Memory Placeholder) --- //
  //! @brief Maps row_t -> IndexPointer (temporary).
  std::map<row_t, IndexPointer> in_memory_rowid_map_;
  // IndexPointer rowid_map_root_ptr_; // Persisted root of the ART index (when
  // implemented)

  // --- Entry Point --- //
  //! @brief Pointer to the current graph entry point node.
  IndexPointer graph_entry_point_ptr_;
  //! @brief Cached row_id of the entry point node.
  row_t graph_entry_point_rowid_;

  // --- Delete Queue --- //
  //! @brief Pointer to the head of the delete queue linked list.
  IndexPointer delete_queue_head_ptr_;

  //! @brief Tracks if changes need persisting.
  bool is_dirty_ = false;

  // --- Helper Methods (Private to LmDiskannIndex) --- //
  /** @brief Initializes structures for a new, empty index. */
  void InitializeNewIndex(idx_t estimated_cardinality);
  /** @brief Loads index metadata and state from existing storage info. */
  void LoadFromStorage(const IndexStorageInfo &storage_info);

  // --- Storage interaction helpers (using in-memory map for now) --- //
  /** @brief Tries to find the node pointer for a given RowID using the current
   * map. */
  bool TryGetNodePointer(row_t row_id,
                         IndexPointer &node_ptr); // Looks up row_id in the map
  /** @brief Allocates a new node block and adds it to the map. */
  IndexPointer AllocateNode(row_t row_id); // Allocates block and updates map
  /** @brief Removes a node from the map and frees its block. */
  void DeleteNodeFromMapAndFreeBlock(
      row_t row_id); // Deletes from map and potentially frees block
  /** @brief Gets a mutable data pointer to the node data using its
   * IndexPointer. */
  data_ptr_t GetNodeDataMutable(IndexPointer node_ptr);
  /** @brief Gets a read-only data pointer to the node data using its
   * IndexPointer. */
  const_data_ptr_t GetNodeData(IndexPointer node_ptr);

  // --- Insertion Helper --- //
  /** @brief Finds neighbors for a new node and connects them (updates new node
   * and neighbors). */
  void FindAndConnectNeighbors(row_t new_node_rowid, IndexPointer new_node_ptr,
                               const float *new_node_vector);
  /** @brief Applies robust pruning to a node's potential neighbors. Updates the
   * node's block. */
  void RobustPrune(row_t node_rowid, IndexPointer node_ptr,
                   std::vector<std::pair<float, row_t>> &candidates);

  // --- Entry Point Helpers --- //
  /** @brief Gets the current valid entry point RowID (might involve lookup or
   * random selection). */
  row_t GetEntryPoint();
  /** @brief Sets the current entry point pointer and cached RowID. */
  void SetEntryPoint(row_t row_id, IndexPointer node_ptr);
  /** @brief Gets a random RowID from the current set of nodes (placeholder
   * using map). */
  row_t GetRandomNodeID(); // Placeholder using map

  // --- Deletion Helper --- //
  /** @brief Adds a deleted RowID to the persistent delete queue. */
  void EnqueueDeletion(row_t deleted_row_id);
  /** @brief Processes the delete queue (placeholder - called by Vacuum). */
  void ProcessDeletionQueue(); // Placeholder

  // --- Distance Helpers --- //
  /** @brief Calculates approximate distance using ternary compressed vector.
   * Uses config_. */
  float CalculateApproxDistance(const float *query_ptr,
                                const_data_ptr_t compressed_neighbor_ptr);
  /** @brief Compresses a float vector for edge storage (Ternary). Uses config_.
   */
  void CompressVectorForEdge(const float *input_vector,
                             data_ptr_t output_compressed_vector);
  /** @brief Calculates exact distance between two raw vectors. Uses config_. */
  template <typename T_QUERY, typename T_NODE> // T_QUERY is likely float
  float CalculateExactDistance(const T_QUERY *query_ptr,
                               const_data_ptr_t node_vector_ptr);
  /** @brief Converts raw node vector to float. Uses config_. */
  void ConvertNodeVectorToFloat(const_data_ptr_t raw_node_vector,
                                float *float_vector_out);
};

} // namespace duckdb
