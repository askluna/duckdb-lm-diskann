/*******************************************************************************
 * @file LmDiskannIndex.cpp
 * @brief Implementation of the LmDiskannIndex class for DuckDB.
 * @details This file contains the implementation details for managing,
 *searching, inserting into, and deleting from an LM-DiskANN index within
 *DuckDB. It interacts with DuckDB's storage and execution systems.
 ******************************************************************************/
#include "LmDiskannIndex.hpp"

// Include refactored component headers
#include "GraphManager.hpp"    // New
#include "GraphOperations.hpp" // New
#include "LmDiskannScanState.hpp"
#include "distance.hpp" // For distance/conversion functions
#include "index_config.hpp"
#include "search.hpp"          // For PerformSearch
#include "storage_manager.hpp" // For Load/PersistMetadata, GetEntryPointRowId etc.

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
#include "duckdb/storage/storage_manager.hpp" // Required for GraphManager
#include "duckdb/storage/table_io_manager.hpp"

namespace diskann {
namespace duckdb {

// --- LmDiskannIndex Constructor --- //

LmDiskannIndex::LmDiskannIndex(
    const ::duckdb::string &name,
    ::duckdb::IndexConstraintType index_constraint_type,
    /**
     * @brief Constructor for LmDiskannIndex.
     * @param name Index name.
     * @param index_constraint_type Type of constraint (e.g., UNIQUE).
     * @param column_ids Physical column IDs covered by the index.
     */
    const ::duckdb::vector<::duckdb::column_t> &column_ids,
    ::duckdb::TableIOManager &table_io_manager,
    const ::duckdb::vector<::duckdb::unique_ptr<::duckdb::Expression>>
        &unbound_expressions,
    ::duckdb::AttachedDatabase &db,
    const ::duckdb::case_insensitive_map_t<::duckdb::Value> &options,
    const ::duckdb::IndexStorageInfo &storage_info, idx_t estimated_cardinality)
    : BoundIndex(name, LmDiskannIndex::TYPE_NAME, index_constraint_type,
                 column_ids, table_io_manager, unbound_expressions, db),
      db_state_(db, table_io_manager, unbound_expressions[0]->return_type),
      format_version_(core::LMDISKANN_CURRENT_FORMAT_VERSION),
      is_dirty_(false) {
  // 1. Parse WITH clause options into the config struct
  config_ = core::ParseOptions(options);

  // 2. Derive dimensions and node_vector_type from the indexed column type
  if (db_state_.indexed_column_type.id() != ::duckdb::LogicalTypeId::ARRAY ||
      ::duckdb::ArrayType::GetChildType(db_state_.indexed_column_type).id() ==
          ::duckdb::LogicalTypeId::INVALID) {
    throw ::duckdb::BinderException(
        "LM_DISKANN index can only be created on ARRAY types "
        "(e.g., FLOAT[N]).");
  }
  config_.dimensions =
      ::duckdb::ArrayType::GetSize(db_state_.indexed_column_type);
  if (config_.dimensions == 0) {
    throw ::duckdb::BinderException(
        "LM_DISKANN index array dimensions cannot be zero.");
  }
  auto array_child_type =
      ::duckdb::ArrayType::GetChildType(db_state_.indexed_column_type);
  if (array_child_type.id() == ::duckdb::LogicalTypeId::FLOAT) {
    config_.node_vector_type = core::LmDiskannVectorType::FLOAT32;
  } else if (array_child_type.id() == ::duckdb::LogicalTypeId::TINYINT) {
    config_.node_vector_type = core::LmDiskannVectorType::INT8;
  } else {
    throw ::duckdb::BinderException(
        "LM_DISKANN index ARRAY child type must be FLOAT or TINYINT, found: " +
        array_child_type.ToString());
  }

  // 3. Validate all configuration parameters (including derived ones)
  ValidateParameters(config_);

  // 4. Calculate node layout based on the fully populated config
  node_layout_ = CalculateLayoutInternal(config_);

  // 5. Calculate final block size (aligned to storage sector size)
  block_size_bytes_ =
      ::duckdb::AlignValue<idx_t, ::duckdb::Storage::SECTOR_SIZE>(
          node_layout_.total_node_size);

  // 6. Initialize GraphManager which creates its own FixedSizeAllocator
  auto &buffer_manager =
      ::duckdb::BufferManager::GetBufferManager(db_state_.db);
  node_manager_ =
      make_uniq<core::GraphManager>(buffer_manager, block_size_bytes_);

  // Instantiate GraphOperations (needs config, layout, node_manager, and this
  // index context)
  graph_operations_ = make_uniq<core::GraphOperations>(config_, node_layout_,
                                                       *node_manager_, *this);

  // Determine and create the index-specific directory path
  // This path will be used by InitializeNewIndex or LoadFromStorage
  auto &fs = ::duckdb::FileSystem::Get(db);
  ::duckdb::string db_lmd_root_path_str = db.GetName() + ".lmd_idx";
  ::duckdb::string specific_index_dir_str =
      fs.JoinPath(db_lmd_root_path_str, this->name);

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
        throw ::duckdb::CatalogException(::duckdb::StringUtil::Format(
            "Cannot create LM-DiskANN index: directory '%s' "
            "already exists for new index '%s'. "
            "Please ensure the path is clear or drop "
            "potentially orphaned index artifacts.",
            this->index_data_path_, this->name));
      }
    } catch (const ::duckdb::PermissionException &e) {
      throw ::duckdb::PermissionException(::duckdb::StringUtil::Format(
          "Failed to create directory structure for LM-DiskANN index '%s' "
          "(path: '%s') due to insufficient permissions: %s",
          this->name, this->index_data_path_, e.what()));
    } catch (const ::duckdb::IOException &e) {
      throw ::duckdb::IOException(::duckdb::StringUtil::Format(
          "Failed to create directory structure for "
          "LM-DiskANN index '%s' (path: '%s'): %s",
          this->name, this->index_data_path_, e.what()));
    } catch (const std::exception
                 &e) { // Catch-all for other potential issues during FS ops
      throw ::duckdb::IOException(::duckdb::StringUtil::Format(
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
      throw ::duckdb::IOException(::duckdb::StringUtil::Format(
          "LM-DiskANN index directory '%s' not found for existing index '%s'. "
          "The index files may be missing or corrupted.",
          this->index_data_path_, this->name));
    }
    LoadFromStorage(storage_info);
  }

  // --- Logging --- //
  ::duckdb::Printer::Print(::duckdb::StringUtil::Format(
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
    const float *query_ptr,
    ::duckdb::const_data_ptr_t compressed_neighbor_ptr) {
  return this->CalculateApproxDistance(query_ptr, compressed_neighbor_ptr);
}

void LmDiskannIndex::PublicCompressVectorForEdge(
    const float *input_vector, ::duckdb::data_ptr_t output_compressed_vector) {
  this->CompressVectorForEdge(input_vector, output_compressed_vector);
}

void LmDiskannIndex::PublicConvertNodeVectorToFloat(
    ::duckdb::const_data_ptr_t raw_node_vector, float *float_vector_out) {
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
::duckdb::ErrorData LmDiskannIndex::Append(::duckdb::IndexLock &lock,
                                           ::duckdb::DataChunk &input,
                                           ::duckdb::Vector &row_ids) {
  if (input.size() == 0) {
    return ::duckdb::ErrorData();
  }
  row_ids.Flatten(input.size());

  ::duckdb::DataChunk input_chunk;
  input_chunk.InitializeEmpty({db_state_.indexed_column_type});
  ::duckdb::Vector row_id_vector(::duckdb::LogicalType::ROW_TYPE);

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
  return ::duckdb::ErrorData();
}

/**
 * @brief Finalizes the dropping of the index.
 * @details Resets the allocator and clears internal pointers.
 * @param index_lock Lock protecting the index state.
 */
void LmDiskannIndex::CommitDrop(::duckdb::IndexLock &index_lock) {
  if (node_manager_) {
    node_manager_->Reset();
  }
  db_state_.metadata_ptr.Clear();
  // in_memory_rowid_map_ is managed by GraphManager and cleared in its
  // Reset()
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
void LmDiskannIndex::Delete(::duckdb::IndexLock &lock,
                            ::duckdb::DataChunk &entries,
                            ::duckdb::Vector &row_identifiers) {
  row_identifiers.Flatten(entries.size());
  auto row_ids_data =
      ::duckdb::FlatVector::GetData<::duckdb::row_t>(row_identifiers);
  bool changes_made = false;

  for (idx_t i = 0; i < entries.size(); ++i) {
    ::duckdb::row_t row_id = row_ids_data[i];
    try {
      if (node_manager_) {
        node_manager_->FreeNode(row_id);
      }
      if (node_manager_) { // Need allocator for the global EnqueueDeletion
        core::EnqueueDeletion(row_id, delete_queue_head_ptr_, db_state_.db,
                              node_manager_->GetAllocator());
      }

      if (graph_operations_) {
        graph_operations_->HandleNodeDeletion(row_id);
      }
      changes_made = true;

    } catch (::duckdb::NotImplementedException &e) {
      throw;
    } catch (std::exception &e) {
      ::duckdb::Printer::Print(::duckdb::StringUtil::Format(
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
::duckdb::ErrorData LmDiskannIndex::Insert(::duckdb::IndexLock &lock,
                                           ::duckdb::DataChunk &data,
                                           ::duckdb::Vector &row_ids) {
  if (data.size() == 0) {
    return ::duckdb::ErrorData();
  }
  D_ASSERT(data.size() == 1);
  D_ASSERT(data.ColumnCount() == 1);
  D_ASSERT(row_ids.GetVectorType() == VectorType::FLAT_VECTOR);

  auto &input_vector_handle = data.data[0];
  input_vector_handle.Flatten(1);
  auto row_id = ::duckdb::FlatVector::GetData<::duckdb::row_t>(row_ids)[0];
  ::duckdb::const_data_ptr_t input_vector_raw_ptr =
      ::duckdb::FlatVector::GetData(input_vector_handle);

  ::duckdb::vector<float> input_vector_float_storage(config_.dimensions);
  const float *input_vector_float_ptr = nullptr;
  try {
    if (config_.node_vector_type == core::LmDiskannVectorType::FLOAT32) {
      input_vector_float_ptr =
          reinterpret_cast<const float *>(input_vector_raw_ptr);
    } else if (config_.node_vector_type == core::LmDiskannVectorType::INT8) {
      ConvertNodeVectorToFloat(input_vector_raw_ptr,
                               input_vector_float_storage.data());
      input_vector_float_ptr = input_vector_float_storage.data();
    } else {
      return ::duckdb::ErrorData("Unsupported node vector type for insertion.");
    }
  } catch (const std::exception &e) {
    return ::duckdb::ErrorData(::duckdb::StringUtil::Format(
        "Error converting input vector: %s", e.what()));
  }

  if (!input_vector_float_ptr) {
    return ::duckdb::ErrorData(
        "Internal error: Failed to obtain float pointer for input vector.");
  }

  ::duckdb::IndexPointer new_node_ptr;
  try {
    if (!node_manager_) {
      return ::duckdb::ErrorData("GraphManager not initialized during Insert.");
    }
    new_node_ptr = node_manager_->AllocateNode(row_id);
  } catch (const std::exception &e) {
    return ::duckdb::ErrorData(
        ::duckdb::StringUtil::Format("Error allocating node: %s", e.what()));
  }

  ::duckdb::data_ptr_t new_node_data = nullptr;
  try {
    if (!node_manager_) {
      return ::duckdb::ErrorData(
          "GraphManager not initialized before GetNodeDataMutable.");
    }
    new_node_data = node_manager_->GetNodeDataMutable(new_node_ptr);

    core::NodeAccessors::InitializeNodeBlock(new_node_data, block_size_bytes_);

    memcpy(
        core::NodeAccessors::GetNodeVectorMutable(new_node_data, node_layout_),
        input_vector_raw_ptr,
        GetVectorTypeSizeBytes(config_.node_vector_type) * config_.dimensions);

    if (!graph_operations_) {
      return ::duckdb::ErrorData(
          "GraphOperations not initialized during Insert.");
    }
    graph_operations_->InsertNode(row_id, new_node_ptr, input_vector_float_ptr);

    is_dirty_ = true;
    return ::duckdb::ErrorData();

  } catch (const std::exception &e) {
    try {
      if (node_manager_) {
        node_manager_->FreeNode(row_id);
      }
    } catch (...) {
    } // Best effort cleanup
    return ::duckdb::ErrorData(::duckdb::StringUtil::Format(
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
::duckdb::IndexStorageInfo LmDiskannIndex::GetStorageInfo(bool get_buffers) {
  ::duckdb::IndexStorageInfo info;
  info.name = name;
  if (node_manager_) {
    info.allocator_infos.push_back(node_manager_->GetAllocator().GetInfo());
  }
  return info;
}

/**
 * @brief Estimates the in-memory size of the index.
 * @return Estimated size in bytes (allocator + map overhead).
 */
idx_t LmDiskannIndex::GetInMemorySize() {
  idx_t base_size = 0;
  if (node_manager_) {
    base_size +=
        node_manager_->GetInMemorySize(); // GraphManager's size includes its
                                          // allocator and map
  }
  return base_size;
}

/**
 * @brief Merges another index into this one.
 * @warning Not implemented for LM-DiskANN.
 * @param state Index lock.
 * @param other_index The index to merge into this one.
 * @return Always returns false (not implemented).
 */
bool LmDiskannIndex::MergeIndexes(::duckdb::IndexLock &state,
                                  BoundIndex &other_index) {
  throw ::duckdb::NotImplementedException(
      "LmDiskannIndex::MergeIndexes not implemented");
  return false;
}

/**
 * @brief Performs vacuuming operations on the index.
 * @details Currently a placeholder; intended to process the deletion queue.
 * @param state Index lock.
 */
void LmDiskannIndex::Vacuum(::duckdb::IndexLock &state) {
  // FIXME: ProcessDeletionQueue needs implementation (currently placeholder in
  // storage_manager.hpp) ProcessDeletionQueue(delete_queue_head_ptr_,
  // db_state_.db, *db_state_.allocator, *this);
  ::duckdb::Printer::Print(
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
::duckdb::string LmDiskannIndex::VerifyAndToString(::duckdb::IndexLock &state,
                                                   const bool only_verify) {
  ::duckdb::string result = "LmDiskannIndex [Not Verified]";
  result += ::duckdb::StringUtil::Format(
      "\n - Config: Metric=%s, Type=%s, Dim=%lld, R=%d, L_insert=%d, "
      "Alpha=%.2f, L_search=%d",
      LmDiskannMetricTypeToString(config_.metric_type),
      LmDiskannVectorTypeToString(config_.node_vector_type), config_.dimensions,
      config_.r, config_.l_insert, config_.alpha, config_.l_search);
  result += ::duckdb::StringUtil::Format(
      "\n - Allocator Blocks Used: %lld",
      node_manager_ ? node_manager_->GetAllocator().GetSegmentCount() : 0);
  result += ::duckdb::StringUtil::Format(
      "\n - Node Count (from GraphManager): %lld",
      node_manager_ ? node_manager_->GetNodeCount() : 0);
  result += ::duckdb::StringUtil::Format(
      "\n - Entry Point RowID (from GraphOperations): %lld",
      static_cast<long long>(
          graph_operations_
              ? graph_operations_->GetGraphEntryPointRowId()
              : ::duckdb::NumericLimits<::duckdb::row_t>::Maximum()));
  result += ::duckdb::StringUtil::Format(
      "\n - Metadata Ptr: [BufferID=%lld, Offset=%lld, Meta=%d]",
      db_state_.metadata_ptr.GetBufferId(), db_state_.metadata_ptr.GetOffset(),
      db_state_.metadata_ptr.GetMetadata());
  result += ::duckdb::StringUtil::Format(
      "\n - Delete Queue Head: [BufferID=%lld, Offset=%lld, Meta=%d]",
      delete_queue_head_ptr_.GetBufferId(), delete_queue_head_ptr_.GetOffset(),
      delete_queue_head_ptr_.GetMetadata());
  return result;
}

/**
 * @brief Verifies allocator allocations (placeholder).
 * @param state Index lock.
 */
void LmDiskannIndex::VerifyAllocations(::duckdb::IndexLock &state) {
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
::duckdb::string LmDiskannIndex::GetConstraintViolationMessage(
    ::duckdb::VerifyExistenceType verify_type, idx_t failed_index,
    ::duckdb::DataChunk &input) {
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
::duckdb::unique_ptr<::duckdb::IndexScanState>
LmDiskannIndex::InitializeScan(::duckdb::ClientContext &context,
                               const ::duckdb::Vector &query_vector, idx_t k) {
  if (query_vector.GetType().id() != ::duckdb::LogicalTypeId::ARRAY ||
      ::duckdb::ArrayType::GetChildType(query_vector.GetType()).id() !=
          ::duckdb::LogicalTypeId::FLOAT) {
    throw ::duckdb::BinderException(
        "LM_DISKANN query vector must be ARRAY<FLOAT>.");
  }
  idx_t query_dims = ::duckdb::ArrayType::GetSize(query_vector.GetType());
  if (query_dims != config_.dimensions) {
    throw ::duckdb::BinderException(
        "Query vector dimension (%d) does not match index dimension (%d).",
        query_dims, config_.dimensions);
  }
  if (k == 0) {
    throw ::duckdb::BinderException("Cannot perform index scan with k=0");
  }

  auto scan_state = ::duckdb::make_unique<core::LmDiskannScanState>(
      query_vector, k, config_.l_search);

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
idx_t LmDiskannIndex::Scan(::duckdb::IndexScanState &state,
                           ::duckdb::Vector &result) {
  auto &scan_state = state.Cast<core::LmDiskannScanState>();
  idx_t output_count = 0;
  auto result_data = ::duckdb::FlatVector::GetData<::duckdb::row_t>(result);

  // Perform the beam search, populating scan_state.top_candidates
  PerformSearch(scan_state, *this, config_,
                true); // Find exact distances for final ranking

  // Extract top-k results from the state's max-heap ({distance, rowid})
  std::vector<std::pair<float, ::duckdb::row_t>> final_results;
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
  if (!node_manager_) {
    throw ::duckdb::InternalException(
        "GraphManager not initialized in InitializeNewIndex");
  }
  db_state_.metadata_ptr = node_manager_->GetAllocator().New();
  delete_queue_head_ptr_.Clear();
  // GraphOperations state is managed internally. Entry point is initially
  // invalid.

  core::LmDiskannMetadata initial_metadata;
  initial_metadata.format_version = format_version_;
  initial_metadata.metric_type = config_.metric_type;
  initial_metadata.node_vector_type = config_.node_vector_type;
  initial_metadata.dimensions = config_.dimensions;
  initial_metadata.r = config_.r;
  initial_metadata.l_insert = config_.l_insert;
  initial_metadata.alpha = config_.alpha;
  initial_metadata.l_search = config_.l_search;
  initial_metadata.block_size_bytes = block_size_bytes_;
  if (graph_operations_) { // Get current entry point (likely invalid/null
                           // initially)
    initial_metadata.graph_entry_point_ptr =
        graph_operations_->GetGraphEntryPointPointer();
  } else {
    initial_metadata.graph_entry_point_ptr
        .Clear(); // Should not happen if constructor ran
  }
  initial_metadata.delete_queue_head_ptr = delete_queue_head_ptr_;

  PersistMetadata(db_state_.metadata_ptr, db_state_.db,
                  node_manager_->GetAllocator(), initial_metadata);
  is_dirty_ = true;
}

/**
 * @brief Loads index state and configuration from existing storage.
 * @param storage_info Storage information provided by DuckDB during load.
 */
void LmDiskannIndex::LoadFromStorage(const IndexStorageInfo &storage_info) {
  if (!node_manager_ ||
      db_state_.metadata_ptr.Get() == 0) { // Check Get() != 0 for metadata_ptr
    throw ::duckdb::InternalException(
        "GraphManager or metadata pointer invalid in LoadFromStorage");
  }

  core::LmDiskannMetadata loaded_metadata;
  ::duckdb::LoadMetadata(db_state_.metadata_ptr, db_state_.db,
                         node_manager_->GetAllocator(), loaded_metadata);

  if (loaded_metadata.format_version != format_version_) {
    throw ::duckdb::IOException(::duckdb::StringUtil::Format(
        "LM_DISKANN index format version mismatch: Found "
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
  delete_queue_head_ptr_ = loaded_metadata.delete_queue_head_ptr;

  node_layout_ = CalculateLayoutInternal(config_);
  idx_t expected_block_size =
      ::duckdb::AlignValue<idx_t, ::duckdb::Storage::SECTOR_SIZE>(
          node_layout_.total_node_size);
  if (block_size_bytes_ != expected_block_size) {
    throw ::duckdb::IOException(::duckdb::StringUtil::Format(
        "LM_DISKANN loaded block size (%lld) inconsistent with recalculated "
        "size (%lld) based on loaded parameters.",
        block_size_bytes_, expected_block_size));
  }

  // TODO: GraphManager should handle populating its RowID map from storage
  // (e.g., by scanning allocator blocks if no ART yet)
  ::duckdb::Printer::Print(
      "Warning: LmDiskannIndex loaded, but GraphManager's RowID map NOT "
      "populated from storage (full scan or ART integration in GraphManager "
      "needed). This may lead to issues.");

  ::duckdb::row_t loaded_entry_point_rowid =
      ::duckdb::NumericLimits<::duckdb::row_t>::Maximum();
  if (loaded_metadata.graph_entry_point_ptr.Get() != 0) {
    loaded_entry_point_rowid = ::duckdb::GetEntryPointRowId(
        loaded_metadata.graph_entry_point_ptr, db_state_.db,
        node_manager_->GetAllocator());
  }

  if (graph_operations_) {
    graph_operations_->SetLoadedEntryPoint(
        loaded_metadata.graph_entry_point_ptr, loaded_entry_point_rowid);
  } else {
    // This case should ideally not be reached if constructor logic is sound
    ::duckdb::Printer::Print("Warning: GraphOperations not initialized during "
                             "LoadFromStorage when trying to set entry point.");
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
    const float *query_ptr,
    ::duckdb::const_data_ptr_t compressed_neighbor_ptr) {
  return core::CalculateApproxDistance(query_ptr, compressed_neighbor_ptr,
                                       config_);
}

/**
 * @brief Compresses a float vector into the Ternary format for edge storage.
 * @param input_vector Pointer to the input float vector.
 * @param output_compressed_vector Pointer to the output buffer for the
 * compressed vector.
 */
void LmDiskannIndex::CompressVectorForEdge(
    const float *input_vector, ::duckdb::data_ptr_t output_compressed_vector) {
  if (!core::CompressVectorForEdge(input_vector, output_compressed_vector,
                                   config_)) {
    throw ::duckdb::InternalException(
        "Failed to compress vector into Ternary format.");
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
float LmDiskannIndex::CalculateExactDistance(
    const T_QUERY *query_ptr, ::duckdb::const_data_ptr_t node_vector_ptr) {
  return CalculateDistance<T_QUERY, T_NODE>(
      query_ptr, reinterpret_cast<const T_NODE *>(node_vector_ptr), config_);
}

/**
 * @brief Converts a raw node vector (potentially int8_t) to a float vector.
 * @param raw_node_vector Pointer to the raw node vector data.
 * @param float_vector_out Pointer to the output buffer for the float vector.
 */
void LmDiskannIndex::ConvertNodeVectorToFloat(
    ::duckdb::const_data_ptr_t raw_node_vector, float *float_vector_out) {
  if (config_.node_vector_type == core::LmDiskannVectorType::FLOAT32) {
    memcpy(float_vector_out, raw_node_vector,
           config_.dimensions * sizeof(float));
  } else if (config_.node_vector_type == core::LmDiskannVectorType::INT8) {
    core::ConvertToFloat<int8_t>(
        reinterpret_cast<const int8_t *>(raw_node_vector), float_vector_out,
        config_.dimensions);
  } else {
    throw ::duckdb::InternalException(
        "Unsupported node vector type in ConvertNodeVectorToFloat.");
  }
}

// Explicitly instantiate templates used within this file
template float LmDiskannIndex::CalculateExactDistance<float, float>(
    const float *, ::duckdb::const_data_ptr_t);
template float LmDiskannIndex::CalculateExactDistance<float, int8_t>(
    const float *, ::duckdb::const_data_ptr_t);

// --- Insertion Helper --- //
// FindAndConnectNeighbors and its helpers like RobustPrune (member version) are
// removed.

// --- Deletion Helper --- //
// EnqueueDeletion (member version) and ProcessDeletionQueue are removed.

} // namespace duckdb
} // namespace diskann
