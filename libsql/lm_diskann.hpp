#pragma once

#include "duckdb.hpp" // Main DuckDB header
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/enums/index_constraint_type.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/execution/index/bound_index.hpp"
#include "duckdb/execution/index/index_pointer.hpp" // For IndexPointer
#include "duckdb/main/attached_database.hpp"        // For AttachedDatabase
#include "duckdb/parser/parsed_data/create_index_info.hpp" // For CreateIndexInput
#include "duckdb/planner/expression.hpp"           // For unbound_expressions
#include "duckdb/storage/buffer/buffer_handle.hpp" // For BufferHandle
#include "duckdb/storage/data_pointer.hpp"         // For data_ptr_t
#include "duckdb/storage/index_storage_info.hpp"
#include "duckdb/storage/table_io_manager.hpp"

#include "duckdb/common/unordered_set.hpp" // For unordered_set
#include <cstdint>                         // For standard integer types
#include <memory>                          // For unique_ptr
#include <queue>                           // For priority_queue
#include <string>
#include <utility> // For pair
#include <vector>

namespace duckdb {

// Forward declarations
class FixedSizeAllocator;
class IndexScanState;
class ClientContext;
class DataChunk;
class Vector;
class ART; // Forward declare ART for the mapping index

// --- Enums for LM-DiskANN Parameters ---

// Corresponds to nDistanceFunc / VECTOR_METRIC_TYPE_PARAM_ID
enum class LmDiskannMetricType : uint8_t {
  UNKNOWN = 0,
  L2 = 1,     // VECTOR_METRIC_TYPE_L2
  COSINE = 2, // VECTOR_METRIC_TYPE_COS
  IP = 3      // Inner Product
              // Add HAMMING later if needed for FLOAT1BIT
};

// Corresponds to nNodeVectorType / VECTOR_TYPE_PARAM_ID
enum class LmDiskannVectorType : uint8_t {
  UNKNOWN = 0,
  FLOAT32 = 1, // VECTOR_TYPE_FLOAT
  INT8 = 2,    // VECTOR_TYPE_INT8
  FLOAT16 = 3  // VECTOR_TYPE_FLOAT16 (Requires conversion/handling)
               // Add other types if needed (e.g., BFLOAT16)
};

// Corresponds to nEdgeVectorType / VECTOR_COMPRESS_NEIGHBORS_PARAM_ID
enum class LmDiskannEdgeType : uint8_t {
  SAME_AS_NODE = 0, // Default: Use node's type
  FLOAT32 = 1,      // VECTOR_TYPE_FLOAT
  FLOAT16 = 2,      // VECTOR_TYPE_FLOAT16
  INT8 = 3,         // VECTOR_TYPE_INT8
  FLOAT1BIT = 4     // VECTOR_TYPE_FLOAT1BIT (Cosine/Hamming only)
};

// --- Struct to hold calculated layout offsets ---
struct NodeLayoutOffsets {
  idx_t neighbor_count = 0; // Offset of the neighbor count (uint16_t)
  idx_t node_vector = 0;  // Offset of the start of the node's full vector data
  idx_t neighbor_ids = 0; // Offset of the start of the neighbor row_t array
  idx_t compressed_neighbors =
      0; // Offset of the start of the compressed neighbor vectors array
  idx_t total_size = 0; // Total size *before* final block alignment
};

// --- Core LmDiskannIndex Class ---

class LmDiskannIndex : public BoundIndex {
public:
  // Name used for registration and CREATE INDEX ... USING LM_DISKANN
  static constexpr const char *TYPE_NAME = "LM_DISKANN";

  // Constructor mirroring HNSWIndex
  LmDiskannIndex(const string &name, IndexConstraintType index_constraint_type,
                 const vector<column_t> &column_ids,
                 TableIOManager &table_io_manager,
                 const vector<unique_ptr<Expression>> &unbound_expressions,
                 AttachedDatabase &db,
                 const case_insensitive_map_t<Value> &options,
                 const IndexStorageInfo &storage_info,
                 idx_t estimated_cardinality);

  ~LmDiskannIndex() override; // Destructor

  // --- Overridden BoundIndex Virtual Methods ---
  ErrorData Append(IndexLock &lock, DataChunk &entries,
                   Vector &row_identifiers) override;
  void CommitDrop(IndexLock &index_lock) override;
  void Delete(IndexLock &lock, DataChunk &entries,
              Vector &row_identifiers) override;
  ErrorData Insert(IndexLock &lock, DataChunk &data, Vector &row_ids) override;
  IndexStorageInfo GetStorageInfo(const bool get_buffers) override;
  idx_t GetInMemorySize();
  bool MergeIndexes(IndexLock &state, BoundIndex &other_index) override;
  void Vacuum(IndexLock &state) override;
  string VerifyAndToString(IndexLock &state, const bool only_verify) override;
  void VerifyAllocations(IndexLock &state) override;
  string GetConstraintViolationMessage(VerifyExistenceType verify_type,
                                       idx_t failed_index,
                                       DataChunk &input) override;

  // --- LM-DiskANN Specific Methods for Scanning ---
  // These will be called by the custom scan operator/function implementation
  unique_ptr<IndexScanState>
  InitializeScan(ClientContext &context, const Vector &query_vector, idx_t k);
  idx_t Scan(IndexScanState &state, Vector &result);

  // --- Static Helper for Parameter Parsing ---
  // Can be called during index creation binding phase if needed
  static void ParseOptions(const case_insensitive_map_t<Value> &options,
                           LmDiskannMetricType &metric_type,
                           LmDiskannEdgeType &edge_type, uint32_t &r,
                           uint32_t &l_insert, float &alpha,
                           uint32_t &l_search);

private:
  // --- Core Parameters (Resolved) ---
  LmDiskannMetricType metric_type_;
  LmDiskannVectorType node_vector_type_;
  LmDiskannEdgeType edge_vector_type_param_;      // As specified by user
  LmDiskannVectorType resolved_edge_vector_type_; // Actual type used for edges
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
  AttachedDatabase &db_;             // Reference to the database instance
  TableIOManager &table_io_manager_; // Reference to DuckDB's IO manager
  LogicalType
      indexed_column_type_; // Store the full type (e.g., ARRAY(FLOAT, 128))
  unique_ptr<FixedSizeAllocator> allocator_; // Manages disk blocks for nodes
  IndexPointer
      metadata_ptr_; // Pointer to the block holding persistent metadata

  // --- RowID Mapping (Placeholder - Needs ART Index) ---
  // unique_ptr<ART> rowid_map_; // Maps row_t -> IndexPointer
  // IndexPointer rowid_map_root_ptr_; // Persisted root of the ART index

  // --- Entry Point ---
  IndexPointer
      graph_entry_point_ptr_;     // Store the IndexPointer of the entry node
  row_t graph_entry_point_rowid_; // Store the row_id (needs update if entry
                                  // deleted)

  // --- Delete Queue ---
  IndexPointer delete_queue_head_ptr_; // Pointer to the head of the delete
                                       // queue linked list

  bool is_dirty_ = false; // Tracks if changes need persisting

  // --- Helper Methods ---
  void InitializeNewIndex(idx_t estimated_cardinality);
  void LoadFromStorage(const IndexStorageInfo &storage_info);
  void PersistMetadata(); // Writes parameters to the metadata block

  // Parameter validation and size calculation
  void ValidateParameters();
  void CalculateSizesAndLayout(); // Renamed to include layout calculation
  static idx_t GetVectorTypeSizeBytes(LmDiskannVectorType type);
  static idx_t GetEdgeVectorTypeSizeBytes(LmDiskannEdgeType type,
                                          LmDiskannVectorType node_type);

  // Storage interaction helpers
  bool TryGetNodePointer(row_t row_id,
                         IndexPointer &node_ptr); // Looks up row_id in the map
  IndexPointer AllocateNode(row_t row_id); // Allocates block and updates map
  void DeleteNodeFromMapAndFreeBlock(
      row_t row_id); // Deletes from map and potentially frees block
  BufferHandle
  GetNodeBuffer(IndexPointer node_ptr,
                bool write_lock = false); // Pins block using IndexPointer

  // --- Distance Function ---
  // Calculates distance between two potentially different vector types
  template <typename T_A, typename T_B>
  float CalculateDistance(const T_A *vec_a, const T_B *vec_b);

  // --- Approximate Distance Function ---
  // Calculates distance between full query (assumed float) and compressed
  // neighbor
  float CalculateApproxDistance(const float *query_ptr,
                                const_data_ptr_t compressed_neighbor_ptr);

  // --- Placeholder for PQ Component ---
  // unique_ptr<ProductQuantizer> pq_; // Load/manage PQ codebooks here

  // --- Search Helper ---
  // Performs the core beam search, used by Insert and Scan
  void PerformSearch(LmDiskannScanState &scan_state, bool find_exact_distances);

  // --- Insertion Helper ---
  // Finds neighbors and updates graph during insertion
  void FindAndConnectNeighbors(row_t new_node_rowid, IndexPointer new_node_ptr,
                               const float *new_node_vector);

  // --- Deletion Helpers ---
  void EnqueueDeletion(
      row_t deleted_row_id);   // Adds row_id to the persistent delete queue
  void ProcessDeletionQueue(); // Processes the queue during Vacuum

  // --- Entry Point Helpers ---
  row_t
  GetEntryPoint(); // Gets a valid entry point row_id (persisted or random)
  void
  SetEntryPoint(row_t row_id,
                IndexPointer node_ptr); // Updates the persisted entry point
  row_t GetRandomNodeID(); // Placeholder for getting a random node (needs map
                           // iteration)
};

// --- Helper Structs (Mirroring C version conceptually) ---

// Represents the state during a search operation
// Used by InitializeScan/Scan and internally by Insert
struct LmDiskannScanState : public IndexScanState {
  Vector
      query_vector_handle; // Handle to the query vector for lifetime management
  const float *query_vector_ptr; // Raw pointer to query data (assumed float)
  idx_t k;                       // Number of neighbors requested

  // Candidate priority queue (min-heap based on distance)
  // Stores pairs of (distance, row_t)
  std::priority_queue<std::pair<float, row_t>,
                      std::vector<std::pair<float, row_t>>,
                      std::greater<std::pair<float, row_t>>>
      candidates;

  // Visited set (using row_t as key)
  duckdb::unordered_set<row_t> visited;

  // Top-k results found so far (for re-ranking if needed)
  // Store pairs of (exact_distance, row_t)
  std::vector<std::pair<float, row_t>> top_candidates;

  // Search parameters used for this scan
  uint32_t l_search;

  // Constructor
  LmDiskannScanState(const Vector &query, idx_t k_value,
                     uint32_t l_search_value)
      : query_vector_handle(query), k(k_value), l_search(l_search_value) {
    // Assuming query is always FLOAT for now
    if (query.GetType().id() != LogicalTypeId::ARRAY ||
        ArrayType::GetChildType(query.GetType()).id() != LogicalTypeId::FLOAT) {
      throw BinderException(
          "LmDiskannScanState: Query vector must be ARRAY<FLOAT>.");
    }
    // Ensure the query vector is flattened for direct access
    query_vector_handle.Flatten(1); // Assuming query is a single vector
    query_vector_ptr = FlatVector::GetData<float>(query_vector_handle);
  }
};

} // namespace duckdb
