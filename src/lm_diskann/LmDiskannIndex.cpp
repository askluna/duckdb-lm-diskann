/*******************************************************************************
 * @file LmDiskannIndex.cpp
 * @brief Implementation of the LmDiskannIndex class for DuckDB.
 * @details This file contains the implementation details for managing,
 *searching, inserting into, and deleting from an LM-DiskANN index within
 *DuckDB. It interacts with DuckDB's storage and execution systems.
 ******************************************************************************/
#include "LmDiskannIndex.hpp"

// Include refactored component headers
#include "GraphOperations.hpp" // New
#include "LmDiskannScanState.hpp"
#include "NodeAccessors.hpp" // Now directly used by GraphOperations primarily
#include "NodeManager.hpp"   // New
#include "config.hpp"
#include "distance.hpp" // For distance/conversion functions
#include "search.hpp"   // For PerformSearch
#include "storage.hpp"  // For Load/PersistMetadata, GetEntryPointRowId etc.

// Include necessary DuckDB headers used in this file
#include "duckdb/common/constants.hpp" // For NumericLimits
#include "duckdb/common/helper.hpp"    // For AlignValue
#include "duckdb/common/limits.hpp"    // For NumericLimits
#include "duckdb/common/printer.hpp"
#include "duckdb/common/random_engine.hpp" // For GetSystemRandom
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp" // For Flatten, Slice
#include "duckdb/execution/index/fixed_size_allocator.hpp"
#include "duckdb/main/client_context.hpp"                  // Needed for Vacuum?
#include "duckdb/parser/parsed_data/create_index_info.hpp" // For ArrayType info
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/index_storage_info.hpp"
#include "duckdb/storage/metadata/metadata_reader.hpp"
#include "duckdb/storage/metadata/metadata_writer.hpp"

#include <algorithm> // For std::sort, std::min, std::max
#include <cstring>   // For memcpy, memset
#include <map>       // For in-memory map placeholder
#include <random> // For default_random_engine, uniform_int_distribution (used in GetRandomNodeID placeholder)
#include <set>    // For intermediate pruning steps (if RobustPrune uses it)
#include <vector>

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp" // Required for table
#include "duckdb/common/file_system.hpp" // Required for FileSystem
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/database_manager.hpp" // Required for DatabaseManager
#include "duckdb/parser/parsed_data/create_index_info.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/storage/index_storage_info.hpp"
#include "duckdb/storage/storage_manager.hpp" // Required for StorageManager
#include "duckdb/storage/table_io_manager.hpp"

namespace duckdb {

// --- LmDiskannIndex Constructor --- //

LmDiskannIndex::LmDiskannIndex(
    const string &name, IndexConstraintType index_constraint_type,
    /**
     * @brief Constructor for LmDiskannIndex.
     * @param name Index name.
     * @param index_constraint_type Type of constraint (e.g., UNIQUE).
     * @param column_ids Physical column IDs covered by the index.
     */
    const vector<column_t> &column_ids, TableIOManager &table_io_manager,
    const vector<unique_ptr<Expression>> &unbound_expressions,
    AttachedDatabase &db, const case_insensitive_map_t<Value> &options,
    const IndexStorageInfo &storage_info, idx_t estimated_cardinality)
    : BoundIndex(name, LmDiskannIndex::TYPE_NAME, index_constraint_type,
                 column_ids, table_io_manager, unbound_expressions, db),
      // Initialize db_state_ first, as other parts might depend on it (like
      // allocator)
      db_state_(db, table_io_manager,
                unbound_expressions[0]
                    ->return_type), // Initialize db_state with references and
                                    // derived logical type
      format_version_(LMDISKANN_CURRENT_FORMAT_VERSION),
      graph_entry_point_rowid_(
          NumericLimits<row_t>::Maximum()), // Initialize entry point rowid as
                                            // invalid
      is_dirty_(false)                      // Initialize is_dirty_
{
  // 1. Parse WITH clause options into the config struct
  config_ = ParseOptions(options);

  // 2. Derive dimensions and node_vector_type from the indexed column type
  if (db_state_.indexed_column_type.id() != LogicalTypeId::ARRAY ||
      ArrayType::GetChildType(db_state_.indexed_column_type).id() ==
          LogicalTypeId::INVALID) {
    throw BinderException("LM_DISKANN index can only be created on ARRAY types "
                          "(e.g., FLOAT[N]).");
  }
  config_.dimensions = ArrayType::GetSize(db_state_.indexed_column_type);
  if (config_.dimensions == 0) {
    throw BinderException("LM_DISKANN index array dimensions cannot be zero.");
  }
  auto array_child_type =
      ArrayType::GetChildType(db_state_.indexed_column_type);
  if (array_child_type.id() == LogicalTypeId::FLOAT) {
    config_.node_vector_type = LmDiskannVectorType::FLOAT32;
  } else if (array_child_type.id() == LogicalTypeId::TINYINT) {
    config_.node_vector_type = LmDiskannVectorType::INT8;
  } else {
    throw BinderException(
        "LM_DISKANN index ARRAY child type must be FLOAT or TINYINT, found: " +
        array_child_type.ToString());
  }

  // 3. Validate all configuration parameters (including derived ones)
  ValidateParameters(config_);

  // 4. Calculate node layout based on the fully populated config
  node_layout_ = CalculateLayoutInternal(config_);

  // 5. Calculate final block size (aligned to storage sector size)
  block_size_bytes_ =
      AlignValue<idx_t, Storage::SECTOR_SIZE>(node_layout_.total_node_size);

  // 6. Initialize the FixedSizeAllocator (assuming this happens around here)
  // Make sure db_state_.allocator is initialized after block_size_bytes_ is
  // set. This part of the code might already exist correctly. Example:
  auto &buffer_manager = BufferManager::GetBufferManager(db_state_.db);
  db_state_.allocator =
      make_uniq<FixedSizeAllocator>(buffer_manager, block_size_bytes_);

  // Instantiate NodeManager (manages allocator and rowid_map)
  // The original db_state_.allocator is now owned by node_manager_
  auto &buffer_manager = BufferManager::GetBufferManager(db_state_.db);
  node_manager_ = make_uniq<NodeManager>(buffer_manager, block_size_bytes_);

  // Instantiate GraphOperations (needs config, layout, node_manager, and this
  // index context)
  graph_operations_ = make_uniq<GraphOperations>(
      config_, node_layout_, *node_manager_, *this, graph_entry_point_rowid_,
      graph_entry_point_ptr_);

  // Determine and create the index-specific directory path
  // This path will be used by InitializeNewIndex or LoadFromStorage
  auto &fs = FileSystem::Get(db);
  string db_lmd_root_path_str = db.GetName() + ".lmd_idx";
  string specific_index_dir_str = fs.JoinPath(db_lmd_root_path_str, this->name);

  // Store the determined path in the configuration
  this->index_data_path_ = specific_index_dir_str;

  if (!storage_info.IsValid()) {
    // This is a new index, create directories if they don't exist
    try {
      // Create the root directory for all LM-DiskANN indexes for this database
      // if it doesn't exist
      if (!fs.DirectoryExists(db_lmd_root_path_str)) {
        fs.CreateDirectory(db_lmd_root_path_str);
      }

      // Now create the specific directory for this index
      if (!fs.DirectoryExists(this->index_data_path_)) {
        fs.CreateDirectory(this->index_data_path_);
      } else {
        // Directory for this specific new index already exists, which is an
        // error.
        throw CatalogException(
            StringUtil::Format("Cannot create LM-DiskANN index: directory '%s' "
                               "already exists for new index '%s'. "
                               "Please ensure the path is clear or drop "
                               "potentially orphaned index artifacts.",
                               this->index_data_path_, this->name));
      }
    } catch (const PermissionException &e) {
      throw PermissionException(StringUtil::Format(
          "Failed to create directory structure for LM-DiskANN index '%s' "
          "(path: '%s') due to insufficient permissions: %s",
          this->name, this->index_data_path_, e.what()));
    } catch (const IOException &e) {
      throw IOException(
          StringUtil::Format("Failed to create directory structure for "
                             "LM-DiskANN index '%s' (path: '%s'): %s",
                             this->name, this->index_data_path_, e.what()));
    } catch (const std::exception
                 &e) { // Catch-all for other potential issues during FS ops
      throw IOException(StringUtil::Format(
          "An unexpected error occurred while creating directory structure for "
          "LM-DiskANN index '%s' (path: '%s'): %s",
          this->name, this->index_data_path_, e.what()));
    }

    // Initialize a brand new index
    InitializeNewIndex(estimated_cardinality);
  } else {
    // Load an existing index from storage
    // The config_.path should ideally be loaded from metadata within
    // LoadFromStorage For now, we assume it's correctly derived or will be set
    // by LoadFromStorage. A basic check here ensures the directory we expect
    // (or loaded) exists.
    if (this->index_data_path_.empty()) {
      // If LoadFromStorage was supposed to set this and didn't, or if it's
      // unexpectedly empty. This path could also be reconstructed here if not
      // persisted but derived. For now, if it's empty on load, it's an issue or
      // needs to be derived like above. Let's assume for loading,
      // LoadFromStorage would set it or it's derived same way. If it's derived
      // same way, it would be set before this 'else' branch.
    }
    if (!this->index_data_path_.empty() &&
        !fs.DirectoryExists(this->index_data_path_)) {
      throw IOException(StringUtil::Format(
          "LM-DiskANN index directory '%s' not found for existing index '%s'. "
          "The index files may be missing or corrupted.",
          this->index_data_path_, this->name));
    }
    LoadFromStorage(storage_info);
  }

  // --- Logging --- //
  Printer::Print(StringUtil::Format(
      "LM_DISKANN Index '%s': Metric=%s, Node Type=%s, Dim=%lld, R=%d, "
      "L_insert=%d, Alpha=%.2f, L_search=%d, BlockSize=%lld, EdgeType=TERNARY",
      name, LmDiskannMetricTypeToString(config_.metric_type),
      LmDiskannVectorTypeToString(config_.node_vector_type), config_.dimensions,
      config_.r, config_.l_insert, config_.alpha, config_.l_search,
      block_size_bytes_));
}

LmDiskannIndex::~LmDiskannIndex() = default;

// --- New Public Wrapper Methods ---
void LmDiskannIndex::PublicMarkDirty(bool dirty_state) {
  this->is_dirty_ = dirty_state;
}

float LmDiskannIndex::PublicCalculateApproxDistance(
    const float *query_ptr, const_data_ptr_t compressed_neighbor_ptr) {
  return this->CalculateApproxDistance(query_ptr, compressed_neighbor_ptr);
}

void LmDiskannIndex::PublicCompressVectorForEdge(
    const float *input_vector, data_ptr_t output_compressed_vector) {
  this->CompressVectorForEdge(input_vector, output_compressed_vector);
}

void LmDiskannIndex::PublicConvertNodeVectorToFloat(
    const_data_ptr_t raw_node_vector, float *float_vector_out) {
  this->ConvertNodeVectorToFloat(raw_node_vector, float_vector_out);
}

// --- BoundIndex Method Implementations (Updated) --- //

/**
 * @brief Appends a chunk of data to the index.
 * @details Called during initial table loading or bulk insertions.
 *          Iterates through the input chunk and calls Insert for each row.
 * @param lock Lock protecting the index during modification.
 * @param input DataChunk containing the vectors to append.
 * @param row_ids Corresponding row identifiers for the vectors in the input
 * chunk.
 * @return ErrorData indicating success or failure.
 */
ErrorData LmDiskannIndex::Append(IndexLock &lock, DataChunk &input,
                                 Vector &row_ids) {
  if (input.size() == 0) {
    return ErrorData();
  }
  row_ids.Flatten(input.size());

  DataChunk input_chunk;
  input_chunk.InitializeEmpty({db_state_.indexed_column_type});
  Vector row_id_vector(LogicalType::ROW_TYPE);

  for (idx_t i = 0; i < input.size(); ++i) {
    input_chunk.Reset();
    input_chunk.data[0].Slice(input.data[0], i, i + 1);
    input_chunk.SetCardinality(1);

    row_id_vector.Slice(row_ids, i, i + 1);
    row_id_vector.Flatten(1);

    auto err = Insert(lock, input_chunk, row_id_vector);
    if (err.HasError()) {
      return err;
    }
  }
  is_dirty_ = true;
  return ErrorData();
}

/**
 * @brief Finalizes the dropping of the index.
 * @details Resets the allocator and clears internal pointers.
 * @param index_lock Lock protecting the index state.
 */
void LmDiskannIndex::CommitDrop(IndexLock &index_lock) {
  if (db_state_.allocator) {
    db_state_.allocator->Reset();
  }
  db_state_.metadata_ptr.Clear();
  in_memory_rowid_map_.clear();
  // TODO: Drop ART resources when implemented
  delete_queue_head_ptr_.Clear();
}

/**
 * @brief Deletes entries from the index based on row identifiers.
 * @details Removes the node from the in-memory map, frees its block, adds it to
 * the delete queue, and potentially clears the entry point if it was deleted.
 * @param lock Lock protecting the index during modification.
 * @param entries DataChunk containing the vectors (ignored, deletion is by
 * row_id).
 * @param row_identifiers Vector of row identifiers to delete.
 */
void LmDiskannIndex::Delete(IndexLock &lock, DataChunk &entries,
                            Vector &row_identifiers) {
  row_identifiers.Flatten(entries.size());
  auto row_ids_data = FlatVector::GetData<row_t>(row_identifiers);
  bool changes_made = false;

  for (idx_t i = 0; i < entries.size(); ++i) {
    row_t row_id = row_ids_data[i];
    try {
      DeleteNodeFromMapAndFreeBlock(row_id);
      EnqueueDeletion(row_id);

      if (row_id == graph_entry_point_rowid_) {
        graph_entry_point_ptr_.Clear();
        graph_entry_point_rowid_ = NumericLimits<row_t>::Maximum();
      }
      changes_made = true;

    } catch (NotImplementedException &e) {
      throw;
    } catch (std::exception &e) {
      Printer::Print(StringUtil::Format(
          "Warning: Failed to delete node for row_id %lld: %s", row_id,
          e.what()));
    }
  }
  if (changes_made) {
    is_dirty_ = true;
  }
}

/**
 * @brief Inserts a single vector into the index.
 * @details Allocates a new node, converts the input vector, finds neighbors
 * using search, connects the new node using RobustPrune, and updates neighbors
 * reciprocally.
 * @param lock Lock protecting the index during modification.
 * @param data DataChunk containing the single vector to insert.
 * @param row_ids Vector containing the single row identifier for the vector.
 * @return ErrorData indicating success or failure.
 */
ErrorData LmDiskannIndex::Insert(IndexLock &lock, DataChunk &data,
                                 Vector &row_ids) {
  if (data.size() == 0) {
    return ErrorData();
  }
  D_ASSERT(data.size() == 1);
  D_ASSERT(data.ColumnCount() == 1);
  D_ASSERT(row_ids.GetVectorType() == VectorType::FLAT_VECTOR);

  auto &input_vector_handle = data.data[0];
  input_vector_handle.Flatten(1);
  auto row_id = FlatVector::GetData<row_t>(row_ids)[0];
  const_data_ptr_t input_vector_raw_ptr =
      FlatVector::GetData(input_vector_handle);

  vector<float> input_vector_float_storage(config_.dimensions);
  const float *input_vector_float_ptr = nullptr;
  try {
    if (config_.node_vector_type == LmDiskannVectorType::FLOAT32) {
      input_vector_float_ptr =
          reinterpret_cast<const float *>(input_vector_raw_ptr);
    } else if (config_.node_vector_type == LmDiskannVectorType::INT8) {
      ConvertNodeVectorToFloat(input_vector_raw_ptr,
                               input_vector_float_storage.data());
      input_vector_float_ptr = input_vector_float_storage.data();
    } else {
      return ErrorData("Unsupported node vector type for insertion.");
    }
  } catch (const std::exception &e) {
    return ErrorData(
        StringUtil::Format("Error converting input vector: %s", e.what()));
  }

  if (!input_vector_float_ptr) {
    return ErrorData(
        "Internal error: Failed to obtain float pointer for input vector.");
  }

  IndexPointer new_node_ptr;
  try {
    new_node_ptr = AllocateNode(row_id);
  } catch (const std::exception &e) {
    return ErrorData(StringUtil::Format("Error allocating node: %s", e.what()));
  }

  data_ptr_t new_node_data = nullptr;
  try {
    new_node_data = GetNodeDataMutable(new_node_ptr);

    NodeAccessors::InitializeNodeBlock(new_node_data, block_size_bytes_);

    memcpy(NodeAccessors::GetNodeVectorMutable(new_node_data, node_layout_),
           input_vector_raw_ptr,
           GetVectorTypeSizeBytes(config_.node_vector_type) *
               config_.dimensions);

    row_t entry_point_row_id = GetEntryPoint();

    if (entry_point_row_id == NumericLimits<row_t>::Maximum()) {
      NodeAccessors::SetNeighborCount(new_node_data, 0);
      SetEntryPoint(row_id, new_node_ptr);
    } else {
      FindAndConnectNeighbors(row_id, new_node_ptr, input_vector_float_ptr);
    }

    is_dirty_ = true;
    return ErrorData();

  } catch (const std::exception &e) {
    try {
      DeleteNodeFromMapAndFreeBlock(row_id);
    } catch (...) {
    } // Best effort cleanup
    return ErrorData(StringUtil::Format(
        "Failed during Insert for node %lld: %s", row_id, e.what()));
  }
}

/**
 * @brief Retrieves storage information about the index.
 * @details Used by DuckDB for checkpointing. Fills an IndexStorageInfo struct
 *          with allocator information. Metadata pointer persistence is handled
 * internally.
 * @param get_buffers Flag indicating whether buffer handles are needed (ignored
 * here).
 * @return IndexStorageInfo containing allocator details.
 */
IndexStorageInfo LmDiskannIndex::GetStorageInfo(bool get_buffers) {
  IndexStorageInfo info;
  info.name = name;
  // REMOVED: IndexStorageInfo has no metadata_pointer field.
  // This needs to be stored/retrieved via the allocator/metadata block.
  if (db_state_.allocator) {
    // GUESS: Allocator info might be retrieved this way
    info.allocator_infos.push_back(db_state_.allocator->GetInfo());
    // TODO: Verify correct field names and methods in IndexStorageInfo and
    // FixedSizeAllocator
  }
  return info;
}

/**
 * @brief Estimates the in-memory size of the index.
 * @return Estimated size in bytes (allocator + map overhead).
 */
idx_t LmDiskannIndex::GetInMemorySize() {
  idx_t base_size = 0;
  if (db_state_.allocator) {
    base_size += db_state_.allocator->GetInMemorySize();
  }
  base_size +=
      in_memory_rowid_map_.size() *
      (sizeof(row_t) + sizeof(IndexPointer) + 16); // Estimate map overhead
  // TODO: Add ART in-memory size when implemented
  return base_size;
}

/**
 * @brief Merges another index into this one.
 * @warning Not implemented for LM-DiskANN.
 * @param state Index lock.
 * @param other_index The index to merge into this one.
 * @return Always returns false (not implemented).
 */
bool LmDiskannIndex::MergeIndexes(IndexLock &state, BoundIndex &other_index) {
  throw NotImplementedException("LmDiskannIndex::MergeIndexes not implemented");
  return false;
}

/**
 * @brief Performs vacuuming operations on the index.
 * @details Currently a placeholder; intended to process the deletion queue.
 * @param state Index lock.
 */
void LmDiskannIndex::Vacuum(IndexLock &state) {
  // FIXME: ProcessDeletionQueue needs implementation (currently placeholder in
  // storage.hpp) ProcessDeletionQueue(delete_queue_head_ptr_, db_state_.db,
  // *db_state_.allocator, *this);
  Printer::Print(
      "LmDiskannIndex::Vacuum called, ProcessDeletionQueue not implemented.");

  // TODO: Check if allocator has vacuum functionality
  // if (db_state_.allocator) { db_state_.allocator->Vacuum(); }
}

/**
 * @brief Verifies index integrity (placeholder) and returns a string
 * representation.
 * @param state Index lock.
 * @param only_verify If true, only perform verification without generating
 * string.
 * @return A string describing the index state.
 */
string LmDiskannIndex::VerifyAndToString(IndexLock &state,
                                         const bool only_verify) {
  // TODO: Implement actual verification logic
  string result = "LmDiskannIndex [Not Verified]";
  result += StringUtil::Format(
      "\n - Config: Metric=%s, Type=%s, Dim=%lld, R=%d, L_insert=%d, "
      "Alpha=%.2f, L_search=%d",
      LmDiskannMetricTypeToString(config_.metric_type),
      LmDiskannVectorTypeToString(config_.node_vector_type), config_.dimensions,
      config_.r, config_.l_insert, config_.alpha, config_.l_search);
  result += StringUtil::Format(
      "\n - Allocator Blocks Used: %lld",
      db_state_.allocator ? db_state_.allocator->GetSegmentCount() : 0);
  result += StringUtil::Format("\n - In-Memory Map Size: %lld",
                               in_memory_rowid_map_.size());
  result += StringUtil::Format(
      "\n - Entry Point RowID: %lld",
      static_cast<long long>(
          graph_entry_point_rowid_)); // Use static_cast for clarity
  // Replace ToString() with manual formatting
  result += StringUtil::Format(
      "\n - Metadata Ptr: [BufferID=%lld, Offset=%lld, Meta=%d]",
      db_state_.metadata_ptr.GetBufferId(), db_state_.metadata_ptr.GetOffset(),
      db_state_.metadata_ptr.GetMetadata());
  result += StringUtil::Format(
      "\n - Delete Queue Head: [BufferID=%lld, Offset=%lld, Meta=%d]",
      delete_queue_head_ptr_.GetBufferId(), delete_queue_head_ptr_.GetOffset(),
      delete_queue_head_ptr_.GetMetadata());
  return result;
}

/**
 * @brief Verifies allocator allocations (placeholder).
 * @param state Index lock.
 */
void LmDiskannIndex::VerifyAllocations(IndexLock &state) {
  // TODO: Check if allocator has verification method
  // if (db_state_.allocator) { db_state_.allocator->Verify(); }
}

/**
 * @brief Generates a constraint violation message.
 * @warning Not supported/applicable for LM-DiskANN similarity index.
 * @param verify_type Type of verification.
 * @param failed_index Index of the failed row.
 * @param input Input chunk causing violation.
 * @return Static error message.
 */
string LmDiskannIndex::GetConstraintViolationMessage(
    VerifyExistenceType verify_type, idx_t failed_index, DataChunk &input) {
  return "Constraint violation in LM_DISKANN index (Not supported)";
}

// --- Scan Method Implementations --- //

/**
 * @brief Initializes the state for an index scan (k-NN search).
 * @param context The client context.
 * @param query_vector The query vector (must be ARRAY<FLOAT>).
 * @param k The number of nearest neighbors to find.
 * @return A unique pointer to the initialized scan state.
 * @throws BinderException if query vector type/dimension is incorrect or k is
 * 0.
 */
unique_ptr<IndexScanState>
LmDiskannIndex::InitializeScan(ClientContext &context,
                               const Vector &query_vector, idx_t k) {
  if (query_vector.GetType().id() != LogicalTypeId::ARRAY ||
      ArrayType::GetChildType(query_vector.GetType()).id() !=
          LogicalTypeId::FLOAT) {
    throw BinderException("LM_DISKANN query vector must be ARRAY<FLOAT>.");
  }
  idx_t query_dims = ArrayType::GetSize(query_vector.GetType());
  if (query_dims != config_.dimensions) {
    throw BinderException(
        "Query vector dimension (%d) does not match index dimension (%d).",
        query_dims, config_.dimensions);
  }
  if (k == 0) {
    throw BinderException("Cannot perform index scan with k=0");
  }

  auto scan_state =
      make_uniq<LmDiskannScanState>(query_vector, k, config_.l_search);

  // PerformSearch will handle finding the entry point and initializing
  // candidates
  return std::move(scan_state);
}

/**
 * @brief Performs one step of the index scan (beam search).
 * @details Calls PerformSearch, extracts results from the scan state's priority
 * queue, sorts them, and fills the result vector with RowIDs.
 * @param state The current LmDiskannScanState.
 * @param result The output vector to store resulting RowIDs.
 * @return The number of results produced in this step (up to vector size or k).
 */
idx_t LmDiskannIndex::Scan(IndexScanState &state, Vector &result) {
  auto &scan_state = state.Cast<LmDiskannScanState>();
  idx_t output_count = 0;
  auto result_data = FlatVector::GetData<row_t>(result);

  // Perform the beam search, populating scan_state.top_candidates
  PerformSearch(scan_state, *this, config_,
                true); // Find exact distances for final ranking

  // Extract top-k results from the state's max-heap ({distance, rowid})
  std::vector<std::pair<float, row_t>> final_results;
  final_results.reserve(scan_state.top_candidates.size());
  while (!scan_state.top_candidates.empty()) {
    final_results.push_back(scan_state.top_candidates.top());
    scan_state.top_candidates.pop();
  }

  // Sort by distance ascending (first element of pair)
  std::sort(final_results.begin(), final_results.end());

  // Fill the result vector up to k or vector size
  for (const auto &candidate : final_results) {
    if (output_count < STANDARD_VECTOR_SIZE && output_count < scan_state.k) {
      result_data[output_count++] = candidate.second; // Get the row_id
    } else {
      break;
    }
  }

  return output_count;
}

// --- Helper Method Implementations (Private to LmDiskannIndex) --- //

/**
 * @brief Initializes metadata and state for a brand new index.
 * @param estimated_cardinality Estimated number of rows (unused currently).
 */
void LmDiskannIndex::InitializeNewIndex(idx_t estimated_cardinality) {
  if (!db_state_.allocator) {
    throw InternalException("Allocator not initialized in InitializeNewIndex");
  }
  db_state_.metadata_ptr = db_state_.allocator->New();
  delete_queue_head_ptr_.Clear();
  graph_entry_point_ptr_.Clear();
  graph_entry_point_rowid_ = NumericLimits<row_t>::Maximum();
  // TODO: Initialize ART root pointer when implemented:
  // rowid_map_root_ptr_.Clear();

  LmDiskannMetadata initial_metadata;
  initial_metadata.format_version = format_version_;
  initial_metadata.metric_type = config_.metric_type;
  initial_metadata.node_vector_type = config_.node_vector_type;
  initial_metadata.dimensions = config_.dimensions;
  initial_metadata.r = config_.r;
  initial_metadata.l_insert = config_.l_insert;
  initial_metadata.alpha = config_.alpha;
  initial_metadata.l_search = config_.l_search;
  initial_metadata.block_size_bytes = block_size_bytes_;
  initial_metadata.graph_entry_point_ptr = graph_entry_point_ptr_;
  initial_metadata.delete_queue_head_ptr = delete_queue_head_ptr_;
  // initial_metadata.rowid_map_root_ptr = rowid_map_root_ptr_;

  PersistMetadata(db_state_.metadata_ptr, db_state_.db, *db_state_.allocator,
                  initial_metadata);
  is_dirty_ = true;
}

/**
 * @brief Loads index state and configuration from existing storage.
 * @param storage_info Storage information provided by DuckDB during load.
 */
void LmDiskannIndex::LoadFromStorage(const IndexStorageInfo &storage_info) {
  if (!db_state_.allocator || db_state_.metadata_ptr.Get() == 0) {
    throw InternalException(
        "Allocator or metadata pointer invalid in LoadFromStorage");
  }

  LmDiskannMetadata loaded_metadata;
  LoadMetadata(db_state_.metadata_ptr, db_state_.db, *db_state_.allocator,
               loaded_metadata);

  if (loaded_metadata.format_version != format_version_) {
    throw IOException(
        StringUtil::Format("LM_DISKANN index format version mismatch: Found "
                           "%d, expected %d. Index may be incompatible.",
                           loaded_metadata.format_version, format_version_));
  }

  config_.metric_type = loaded_metadata.metric_type;
  config_.node_vector_type = loaded_metadata.node_vector_type;
  config_.dimensions = loaded_metadata.dimensions;
  config_.r = loaded_metadata.r;
  config_.l_insert = loaded_metadata.l_insert;
  config_.alpha = loaded_metadata.alpha;
  config_.l_search = loaded_metadata.l_search;
  block_size_bytes_ = loaded_metadata.block_size_bytes;
  graph_entry_point_ptr_ = loaded_metadata.graph_entry_point_ptr;
  delete_queue_head_ptr_ = loaded_metadata.delete_queue_head_ptr;
  // rowid_map_root_ptr_ = loaded_metadata.rowid_map_root_ptr; // TODO: Load ART
  // root

  node_layout_ = CalculateLayoutInternal(config_);
  idx_t expected_block_size =
      AlignValue<idx_t, Storage::SECTOR_SIZE>(node_layout_.total_node_size);
  if (block_size_bytes_ != expected_block_size) {
    throw IOException(StringUtil::Format(
        "LM_DISKANN loaded block size (%lld) inconsistent with recalculated "
        "size (%lld) based on loaded parameters.",
        block_size_bytes_, expected_block_size));
  }
  // TODO: Implement block size validation logic
  // This section checks if the block size of the allocator matches the expected
  // loaded block size. If they do not match, an IOException is thrown to
  // indicate a potential inconsistency in the index state. if
  // (db_state_.allocator->GetBlockSize() != block_size_bytes_) {
  //   throw IOException(StringUtil::Format(
  //       "LM_DISKANN allocator block size (%lld) does not match loaded block "
  //       "size (%lld).",
  //       db_state_.allocator->GetBlockSize(), block_size_bytes_));
  // }

  // TODO: Load ART and populate in-memory map placeholder
  Printer::Print("Warning: LmDiskannIndex loaded, but in-memory RowID map NOT "
                 "populated from storage (ART integration needed).");

  if (graph_entry_point_ptr_.Get() != 0) {
    // FIXME: GetEntryPointRowId needs implementation (or RowID stored in node)
    graph_entry_point_rowid_ = GetEntryPointRowId(
        graph_entry_point_ptr_, db_state_.db, *db_state_.allocator);
  } else {
    graph_entry_point_rowid_ = NumericLimits<row_t>::Maximum();
  }

  is_dirty_ = false;
}

// --- Distance Helper Wrappers --- //

/**
 * @brief Calculates approximate distance between a query vector and a
 * compressed neighbor vector.
 * @param query_ptr Pointer to the float query vector.
 * @param compressed_neighbor_ptr Pointer to the compressed (Ternary) neighbor
 * vector.
 * @return Approximate distance.
 */
float LmDiskannIndex::CalculateApproxDistance(
    const float *query_ptr, const_data_ptr_t compressed_neighbor_ptr) {
  return duckdb::CalculateApproxDistance(query_ptr, compressed_neighbor_ptr,
                                         config_);
}

/**
 * @brief Compresses a float vector into the Ternary format for edge storage.
 * @param input_vector Pointer to the input float vector.
 * @param output_compressed_vector Pointer to the output buffer for the
 * compressed vector.
 */
void LmDiskannIndex::CompressVectorForEdge(
    const float *input_vector, data_ptr_t output_compressed_vector) {
  if (!duckdb::CompressVectorForEdge(input_vector, output_compressed_vector,
                                     config_)) {
    throw InternalException("Failed to compress vector into Ternary format.");
  }
}

/**
 * @brief Calculates exact distance between a query vector and a raw node
 * vector.
 * @tparam T_QUERY Type of the query vector elements (usually float).
 * @tparam T_NODE Type of the node vector elements (float or int8_t).
 * @param query_ptr Pointer to the query vector.
 * @param node_vector_ptr Pointer to the raw node vector data.
 * @return Exact distance.
 */
template <typename T_QUERY, typename T_NODE>
float LmDiskannIndex::CalculateExactDistance(const T_QUERY *query_ptr,
                                             const_data_ptr_t node_vector_ptr) {
  return duckdb::CalculateDistance<T_QUERY, T_NODE>(
      query_ptr, reinterpret_cast<const T_NODE *>(node_vector_ptr), config_);
}

/**
 * @brief Converts a raw node vector (potentially int8_t) to a float vector.
 * @param raw_node_vector Pointer to the raw node vector data.
 * @param float_vector_out Pointer to the output buffer for the float vector.
 */
void LmDiskannIndex::ConvertNodeVectorToFloat(const_data_ptr_t raw_node_vector,
                                              float *float_vector_out) {
  if (config_.node_vector_type == LmDiskannVectorType::FLOAT32) {
    memcpy(float_vector_out, raw_node_vector,
           config_.dimensions * sizeof(float));
  } else if (config_.node_vector_type == LmDiskannVectorType::INT8) {
    duckdb::ConvertToFloat<int8_t>(
        reinterpret_cast<const int8_t *>(raw_node_vector), float_vector_out,
        config_.dimensions);
  } else {
    throw InternalException(
        "Unsupported node vector type in ConvertNodeVectorToFloat.");
  }
}

// Explicitly instantiate templates used within this file
template float
LmDiskannIndex::CalculateExactDistance<float, float>(const float *,
                                                     const_data_ptr_t);
template float
LmDiskannIndex::CalculateExactDistance<float, int8_t>(const float *,
                                                      const_data_ptr_t);

// --- Insertion Helper --- //
/**
 * @brief Finds potential neighbors for a new node and connects them.
 * @details Performs a search starting from the entry point to find candidate
 * neighbors. Calls RobustPrune to select the final neighbors for the new node.
 *          Updates the selected neighbors to potentially add a reciprocal edge
 * back to the new node.
 * @param new_node_rowid RowID of the node being inserted.
 * @param new_node_ptr Pointer to the new node's block.
 * @param new_node_vector_float Pointer to the new node's vector data (as
 * float).
 */
void LmDiskannIndex::FindAndConnectNeighbors(
    row_t new_node_rowid, IndexPointer new_node_ptr,
    const float *new_node_vector_float) {
  row_t entry_point = GetEntryPoint();
  if (entry_point == NumericLimits<row_t>::Maximum()) {
    throw InternalException(
        "FindAndConnectNeighbors called with no entry point.");
  }

  Vector query_vec_handle(
      LogicalType::ARRAY(LogicalType::FLOAT, config_.dimensions));
  memcpy(FlatVector::GetData<float>(query_vec_handle), new_node_vector_float,
         config_.dimensions * sizeof(float));
  query_vec_handle.Flatten(1);

  LmDiskannScanState search_state(query_vec_handle, config_.l_insert,
                                  config_.l_insert);
  PerformSearch(search_state, *this, config_, false);

  std::vector<std::pair<float, row_t>> potential_neighbors;
  potential_neighbors.reserve(search_state.top_candidates.size() + 1);
  while (!search_state.top_candidates.empty()) {
    potential_neighbors.push_back(search_state.top_candidates.top());
    search_state.top_candidates.pop();
  }
  potential_neighbors.push_back({0.0f, new_node_rowid}); // Add self

  RobustPrune(new_node_rowid, new_node_ptr, potential_neighbors);

  // --- Update Neighbors (Reciprocal Edges) --- //
  auto new_node_data_ro = GetNodeData(new_node_ptr);
  uint16_t final_new_neighbor_count =
      NodeAccessors::GetNeighborCount(new_node_data_ro);
  const row_t *final_new_neighbor_ids =
      NodeAccessors::GetNeighborIDsPtr(new_node_data_ro, node_layout_);

  vector<uint8_t> new_node_compressed_storage(
      node_layout_.ternary_edge_size_bytes);
  CompressVectorForEdge(new_node_vector_float,
                        new_node_compressed_storage.data());

  vector<float> neighbor_float_vec(config_.dimensions);
  for (uint16_t i = 0; i < final_new_neighbor_count; ++i) {
    row_t neighbor_rowid = final_new_neighbor_ids[i];
    if (neighbor_rowid == NumericLimits<row_t>::Maximum())
      continue;

    IndexPointer neighbor_ptr;
    if (!TryGetNodePointer(neighbor_rowid, neighbor_ptr))
      continue;

    std::vector<std::pair<float, row_t>> neighbor_candidates;
    try {
      const_data_ptr_t neighbor_data_ro;
      const_data_ptr_t neighbor_vec_ptr_raw;

      neighbor_data_ro = GetNodeData(neighbor_ptr);
      neighbor_vec_ptr_raw =
          NodeAccessors::GetNodeVector(neighbor_data_ro, node_layout_);
      ConvertNodeVectorToFloat(neighbor_vec_ptr_raw, neighbor_float_vec.data());

      float dist_neighbor_to_new = ComputeExactDistanceFloat(
          neighbor_float_vec.data(), new_node_vector_float, config_.dimensions,
          config_.metric_type);

      neighbor_candidates.push_back({dist_neighbor_to_new, new_node_rowid});

      RobustPrune(neighbor_rowid, neighbor_ptr, neighbor_candidates);

    } catch (const std::exception &e) {
      Printer::Print(StringUtil::Format(
          "Warning: Failed to update neighbor %lld for new node %lld: %s",
          neighbor_rowid, new_node_rowid, e.what()));
    }
  }
}

// --- Deletion Helper --- //
/**
 * @brief Adds a row ID to the persistent deletion queue.
 * @param deleted_row_id The row ID of the node marked for deletion.
 */
void LmDiskannIndex::EnqueueDeletion(row_t deleted_row_id) {
  duckdb::EnqueueDeletion(deleted_row_id, delete_queue_head_ptr_, db_state_.db,
                          *db_state_.allocator);
  is_dirty_ = true;
}

/**
 * @brief Processes the deletion queue (Placeholder).
 * @warning Not implemented. Intended to be called during VACUUM.
 */
void LmDiskannIndex::ProcessDeletionQueue() {
  Printer::Print("LmDiskannIndex::ProcessDeletionQueue is not implemented.");
}

} // namespace duckdb
