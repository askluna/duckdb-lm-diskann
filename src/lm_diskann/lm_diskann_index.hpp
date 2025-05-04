#pragma once

// Main DuckDB includes needed by the index class itself
#include "duckdb.hpp"
#include "duckdb/execution/index/bound_index.hpp"
#include "duckdb/storage/index_storage_info.hpp"
#include "duckdb/storage/table_io_manager.hpp"
#include "duckdb/execution/index/index_pointer.hpp"
#include "duckdb/parser/parsed_data/create_index_info.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/common/case_insensitive_map.hpp"

// Include headers for the refactored components
#include "config.hpp"
#include "state.hpp" // For scan state definition

#include <memory> // For unique_ptr
#include <vector>
#include <string>
#include <map> // Using std::map for in-memory RowID mapping for now

namespace duckdb {

// Forward declarations for components used internally
class FixedSizeAllocator;
class ClientContext;
class DataChunk;
class Vector;
class BufferHandle;
// class ART; // If using ART for mapping

// --- Core LMDiskannIndex Class ---
// Orchestrates LM-DiskANN operations, interfacing with DuckDB's index system
// and delegating tasks to specialized modules (config, node, storage, search, distance).
class LMDiskannIndex : public BoundIndex {
public:
    // Name used for registration and CREATE INDEX ... USING LM_DISKANN
    static constexpr const char *TYPE_NAME = "LM_DISKANN";

    // Constructor: Initializes parameters, storage, and loads/creates the index.
    LMDiskannIndex(const string &name, IndexConstraintType index_constraint_type,
                   const vector<column_t> &column_ids, TableIOManager &table_io_manager,
                   const vector<unique_ptr<Expression>> &unbound_expressions,
                   AttachedDatabase &db, const case_insensitive_map_t<Value> &options,
                   const IndexStorageInfo &storage_info, idx_t estimated_cardinality);

    ~LMDiskannIndex() override; // Destructor

    // --- Overridden BoundIndex Virtual Methods ---
    // These methods are called by DuckDB during DML operations, checkpointing, etc.
    ErrorData Append(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) override;
    void CommitDrop(IndexLock &index_lock) override;
    void Delete(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) override;
    ErrorData Insert(IndexLock &lock, DataChunk &data, Vector &row_ids) override;
    IndexStorageInfo GetStorageInfo(const bool get_buffers);
    idx_t GetInMemorySize();
    bool MergeIndexes(IndexLock &state, BoundIndex &other_index) override;
    void Vacuum(IndexLock &state) override;
    string VerifyAndToString(IndexLock &state, const bool only_verify) override;
    void VerifyAllocations(IndexLock &state) override;
    string GetConstraintViolationMessage(VerifyExistenceType verify_type, idx_t failed_index,
                                         DataChunk &input) override;

    // --- LM-DiskANN Specific Methods for Scanning ---
    // Called by the custom scan operator/function implementation.
    unique_ptr<IndexScanState> InitializeScan(ClientContext &context, const Vector &query_vector, idx_t k);
    idx_t Scan(IndexScanState &state, Vector &result);

    // --- Public Accessors (Potentially needed by other modules/friends) ---
    // Allows other components (like search) to access necessary index state.
    AttachedDatabase &GetAttachedDatabase() const { return db_; }
    FixedSizeAllocator &GetAllocator() const { return *allocator_; }
    const NodeLayoutOffsets& GetNodeLayout() const { return node_layout_; }
    idx_t GetDimensions() const { return dimensions_; }
    idx_t GetEdgeVectorSizeBytes() const { return edge_vector_size_bytes_; }
    LMDiskannVectorType GetResolvedEdgeType() const { return resolved_edge_vector_type_; }
    LMDiskannMetricType GetMetricType() const { return metric_type_; }
    uint32_t GetR() const { return r_; }
    float GetAlpha() const { return alpha_; }
    // Add accessors for L_insert, L_search etc. if needed


private:
    friend class LMDiskannStorageHelper; // Allow storage helpers access
    friend void PerformSearch(LMDiskannScanState &scan_state, LMDiskannIndex &index, bool find_exact_distances); // Allow search access

    // --- Core Parameters (Resolved) ---
    LMDiskannMetricType metric_type_;
    LMDiskannVectorType node_vector_type_;
    LMDiskannEdgeType edge_vector_type_param_; // As specified by user
    LMDiskannVectorType resolved_edge_vector_type_; // Actual type used for edges
    idx_t dimensions_;
    uint32_t r_; // Max neighbors (degree)
    uint32_t l_insert_;
    float alpha_;
    uint32_t l_search_;
    uint8_t format_version_; // Internal format version

    // --- Calculated Parameters ---
    idx_t node_vector_size_bytes_;
    idx_t edge_vector_size_bytes_;
    idx_t block_size_bytes_; // Final aligned size of each node's block on disk
    NodeLayoutOffsets node_layout_; // Stores calculated offsets within a block

    // --- DuckDB Integration ---
    AttachedDatabase &db_; // Reference to the database instance
    TableIOManager &table_io_manager_; // Reference to DuckDB's IO manager
    LogicalType indexed_column_type_;  // Store the full type (e.g., ARRAY(FLOAT, 128))
    unique_ptr<FixedSizeAllocator> allocator_; // Manages disk blocks for nodes
    IndexPointer metadata_ptr_; // Pointer to the block holding persistent metadata

    // --- RowID Mapping (In-Memory Placeholder) ---
    std::map<row_t, IndexPointer> in_memory_rowid_map_; // Maps row_t -> IndexPointer
    // IndexPointer rowid_map_root_ptr_; // Persisted root of the ART index (when implemented)

    // --- Entry Point ---
    IndexPointer graph_entry_point_ptr_; // Store the IndexPointer of the entry node
    row_t graph_entry_point_rowid_; // Store the row_id (needs update if entry deleted)

    // --- Delete Queue ---
    IndexPointer delete_queue_head_ptr_; // Pointer to the head of the delete queue linked list

    bool is_dirty_ = false; // Tracks if changes need persisting

    // --- Helper Methods (Private to LMDiskannIndex) ---
    // These coordinate calls to the specialized modules.
    void InitializeNewIndex(idx_t estimated_cardinality);
    void LoadFromStorage(const IndexStorageInfo &storage_info);

    // --- Storage interaction helpers (using in-memory map for now) ---
    bool TryGetNodePointer(row_t row_id, IndexPointer &node_ptr); // Looks up row_id in the map
    IndexPointer AllocateNode(row_t row_id); // Allocates block and updates map
    void DeleteNodeFromMapAndFreeBlock(row_t row_id); // Deletes from map and potentially frees block
    BufferHandle GetNodeBuffer(IndexPointer node_ptr, bool write_lock = false); // Pins block using IndexPointer

    // --- Distance Function Wrappers ---
    // Provide convenient access to distance calculations using index state.
    template <typename T_A, typename T_B>
    float CalculateDistance(const T_A *vec_a, const T_B *vec_b);
    float CalculateApproxDistance(const float *query_ptr, const_data_ptr_t compressed_neighbor_ptr);

    // --- Insertion Helper ---
    void FindAndConnectNeighbors(row_t new_node_rowid, IndexPointer new_node_ptr, const float *new_node_vector);
    void RobustPrune(row_t node_rowid, IndexPointer node_ptr, std::vector<std::pair<float, row_t>>& candidates, uint32_t max_neighbors); // Pruning helper

    // --- Entry Point Helpers ---
    row_t GetEntryPoint();
    void SetEntryPoint(row_t row_id, IndexPointer node_ptr);
    row_t GetRandomNodeID(); // Placeholder

    // --- Deletion Helper ---
    void EnqueueDeletion(row_t deleted_row_id);
    void ProcessDeletionQueue(); // Placeholder

};


} // namespace duckdb
