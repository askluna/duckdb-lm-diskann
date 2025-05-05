/*******************************************************************************
 * @file lm_diskann_index.cpp
 * @brief Implementation of the LMDiskannIndex class for DuckDB.
 ******************************************************************************/
#include "lm_diskann_index.hpp"

// Include refactored component headers
#include "config.hpp"
#include "node.hpp"
#include "storage.hpp" // For Load/PersistMetadata, GetEntryPointRowId etc.
#include "search.hpp" // For PerformSearch
#include "distance.hpp" // For distance/conversion functions
#include "state.hpp" // For LMDiskannScanState

// Include necessary DuckDB headers used in this file
#include "duckdb/execution/index/fixed_size_allocator.hpp"
#include "duckdb/storage/index_storage_info.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp" // For Flatten, Slice
#include "duckdb/common/limits.hpp" // For NumericLimits
#include "duckdb/common/printer.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/metadata/metadata_reader.hpp"
#include "duckdb/storage/metadata/metadata_writer.hpp"
#include "duckdb/parser/parsed_data/create_index_info.hpp" // For ArrayType info
#include "duckdb/main/client_context.hpp" // Needed for Vacuum?
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/random_engine.hpp" // For GetSystemRandom
#include "duckdb/common/helper.hpp" // For AlignValue
#include "duckdb/common/constants.hpp" // For NumericLimits

#include <vector>
#include <cstring> // For memcpy, memset
#include <algorithm> // For std::sort, std::min, std::max
#include <random> // For default_random_engine, uniform_int_distribution (used in GetRandomNodeID placeholder)
#include <map> // For in-memory map placeholder
#include <set> // For intermediate pruning steps (if RobustPrune uses it)

namespace duckdb {

// --- LMDiskannIndex Constructor --- //

LMDiskannIndex::LMDiskannIndex(const string &name, IndexConstraintType index_constraint_type,
                               const vector<column_t> &column_ids, TableIOManager &table_io_manager,
                               const vector<unique_ptr<Expression>> &unbound_expressions,
                               AttachedDatabase &db, const case_insensitive_map_t<Value> &options,
                               const IndexStorageInfo &storage_info, idx_t estimated_cardinality)
    : BoundIndex(name, LMDiskannIndex::TYPE_NAME, index_constraint_type, column_ids, table_io_manager,
                 unbound_expressions, db),
      // Initialize db_state_ first, as other parts might depend on it (like allocator)
      db_state_(db, table_io_manager, logical_types[0]), // Initialize db_state with references and derived logical type
      format_version_(LMDISKANN_CURRENT_FORMAT_VERSION),
      graph_entry_point_rowid_(NumericLimits<row_t>::Maximum()) // Initialize entry point rowid as invalid
{
    // 1. Parse Options into config struct
    config_ = ParseOptions(options);

    // 2. Derive Dimensions and Node Vector Type from column info
    if (logical_types.size() != 1) {
        throw BinderException("LM_DISKANN index can only be created on a single column.");
    }
    // db_state_.indexed_column_type should be set by db_state_ constructor
    if (db_state_.indexed_column_type.id() != LogicalTypeId::ARRAY || ArrayType::GetChildType(db_state_.indexed_column_type).id() == LogicalTypeId::INVALID) {
        throw BinderException("LM_DISKANN index can only be created on ARRAY types (e.g., FLOAT[N]).");
    }
    auto array_child_type = ArrayType::GetChildType(db_state_.indexed_column_type);
    auto array_size = ArrayType::GetSize(db_state_.indexed_column_type);

    // Set derived parameters in the config struct
    config_.dimensions = array_size;
    LogicalTypeId column_child_id = array_child_type.id();
    if (column_child_id == LogicalTypeId::FLOAT) {
        config_.node_vector_type = LMDiskannVectorType::FLOAT32;
    } else if (column_child_id == LogicalTypeId::TINYINT) {
        config_.node_vector_type = LMDiskannVectorType::INT8;
    } else {
        throw BinderException("LM_DISKANN index created on ARRAY type, but child type must be FLOAT or TINYINT, found: %s", array_child_type.ToString());
    }

    // 3. Validate the final config (now including derived parameters)
    ValidateParameters(config_);

    // 4. Calculate Layout based on final config
    node_layout_ = CalculateLayoutInternal(config_);

    // 5. Calculate final block size (aligned)
    // Align to DuckDB's storage sector size
    block_size_bytes_ = AlignValue<idx_t, Storage::SECTOR_SIZE>(node_layout_.total_node_size);
    // TODO: Add check against max block size?

    // 6. Initialize Storage (Allocator)
    auto &buffer_manager = BufferManager::GetBufferManager(db_state_.db);
    db_state_.allocator = make_uniq<FixedSizeAllocator>(buffer_manager, block_size_bytes_);

    // 7. Load or Initialize Index State
    if (storage_info.IsValid()) {
        // Use the metadata pointer from the storage info
        // GUESS: Field name is `metadata_pointer`
        LoadFromStorage(storage_info); // Uses config_ and db_state_
    } else {
        InitializeNewIndex(estimated_cardinality);
    }

     // --- Logging --- //
      Printer::Print(StringUtil::Format("LM_DISKANN Index '%s': Metric=%s, Node Type=%s, Dim=%lld, R=%d, L_insert=%d, Alpha=%.2f, L_search=%d, BlockSize=%lld, EdgeType=TERNARY",
                                       name,
                                       LMDiskannMetricTypeToString(config_.metric_type),
                                       LMDiskannVectorTypeToString(config_.node_vector_type),
                                       config_.dimensions,
                                       config_.r,
                                       config_.l_insert,
                                       config_.alpha,
                                       config_.l_search,
                                       block_size_bytes_));
}

LMDiskannIndex::~LMDiskannIndex() = default;

// --- BoundIndex Method Implementations --- //

ErrorData LMDiskannIndex::Append(IndexLock &lock, DataChunk &input, Vector &row_ids) {
    if (input.size() == 0) {
        return ErrorData();
    }
    row_ids.Flatten(input.size());

    DataChunk input_chunk;
    input_chunk.InitializeEmpty({db_state_.indexed_column_type});
    Vector row_id_vector(LogicalType::ROW_TYPE);

    for(idx_t i = 0; i < input.size(); ++i) {
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

void LMDiskannIndex::CommitDrop(IndexLock &index_lock) {
    if (db_state_.allocator) {
        db_state_.allocator->Reset();
    }
    db_state_.metadata_ptr.Clear();
    in_memory_rowid_map_.clear();
    // TODO: Drop ART resources when implemented
    delete_queue_head_ptr_.Clear();
}

void LMDiskannIndex::Delete(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) {
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
             Printer::Print(StringUtil::Format("Warning: Failed to delete node for row_id %lld: %s", row_id, e.what()));
        }
    }
    if (changes_made) {
        is_dirty_ = true;
    }
}

ErrorData LMDiskannIndex::Insert(IndexLock &lock, DataChunk &data, Vector &row_ids) {
     if (data.size() == 0) { return ErrorData(); }
     D_ASSERT(data.size() == 1);
     D_ASSERT(data.ColumnCount() == 1);
     D_ASSERT(row_ids.GetVectorType() == VectorType::FLAT_VECTOR);

     auto &input_vector_handle = data.data[0];
     input_vector_handle.Flatten(1);
     auto row_id = FlatVector::GetData<row_t>(row_ids)[0];
     const_data_ptr_t input_vector_raw_ptr = FlatVector::GetData(input_vector_handle);

     vector<float> input_vector_float_storage(config_.dimensions);
     const float* input_vector_float_ptr = nullptr;
     try {
         if (config_.node_vector_type == LMDiskannVectorType::FLOAT32) {
             input_vector_float_ptr = reinterpret_cast<const float*>(input_vector_raw_ptr);
         } else if (config_.node_vector_type == LMDiskannVectorType::INT8) {
             ConvertNodeVectorToFloat(input_vector_raw_ptr, input_vector_float_storage.data());
             input_vector_float_ptr = input_vector_float_storage.data();
         } else {
             return ErrorData("Unsupported node vector type for insertion.");
         }
     } catch (const std::exception &e) { return ErrorData(StringUtil::Format("Error converting input vector: %s", e.what())); }

     if (!input_vector_float_ptr) {
         return ErrorData("Internal error: Failed to obtain float pointer for input vector.");
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
 
         LMDiskannNodeAccessors::InitializeNodeBlock(new_node_data, block_size_bytes_);

         memcpy(LMDiskannNodeAccessors::GetNodeVectorMutable(new_node_data, node_layout_),
                input_vector_raw_ptr, GetVectorTypeSizeBytes(config_.node_vector_type) * config_.dimensions);

         row_t entry_point_row_id = GetEntryPoint();

         if (entry_point_row_id == NumericLimits<row_t>::Maximum()) {
             LMDiskannNodeAccessors::SetNeighborCount(new_node_data, 0);
             SetEntryPoint(row_id, new_node_ptr);
         } else {
             FindAndConnectNeighbors(row_id, new_node_ptr, input_vector_float_ptr);
         }

         is_dirty_ = true;
         return ErrorData();

     } catch (const std::exception &e) {
         try { DeleteNodeFromMapAndFreeBlock(row_id); } catch (...) {} // Best effort cleanup
         return ErrorData(StringUtil::Format("Failed during Insert for node %lld: %s", row_id, e.what()));
     }
}

IndexStorageInfo LMDiskannIndex::GetStorageInfo(bool get_buffers) {
    IndexStorageInfo info;
    info.name = name;
    // REMOVED: IndexStorageInfo has no metadata_pointer field.
    // This needs to be stored/retrieved via the allocator/metadata block.
    if (db_state_.allocator) {
        // GUESS: Allocator info might be retrieved this way
        info.allocator_infos.push_back(db_state_.allocator->GetInfo());
        // TODO: Verify correct field names and methods in IndexStorageInfo and FixedSizeAllocator
    }
    return info;
}

idx_t LMDiskannIndex::GetInMemorySize() {
    idx_t base_size = 0;
    if (db_state_.allocator) {
         base_size += db_state_.allocator->GetInMemorySize();
    }
    base_size += in_memory_rowid_map_.size() * (sizeof(row_t) + sizeof(IndexPointer) + 16); // Estimate map overhead
    // TODO: Add ART in-memory size when implemented
    return base_size;
}

bool LMDiskannIndex::MergeIndexes(IndexLock &state, BoundIndex &other_index) {
    throw NotImplementedException("LMDiskannIndex::MergeIndexes not implemented");
    return false;
}

void LMDiskannIndex::Vacuum(IndexLock &state) {
    // FIXME: ProcessDeletionQueue needs implementation (currently placeholder in storage.hpp)
    // ProcessDeletionQueue(delete_queue_head_ptr_, db_state_.db, *db_state_.allocator, *this);
    Printer::Print("LMDiskannIndex::Vacuum called, ProcessDeletionQueue not implemented.");

    // TODO: Check if allocator has vacuum functionality
    // if (db_state_.allocator) { db_state_.allocator->Vacuum(); }
}

string LMDiskannIndex::VerifyAndToString(IndexLock &state, const bool only_verify) {
    // TODO: Implement actual verification logic
    string result = "LMDiskannIndex [Not Verified]";
    result += StringUtil::Format("\n - Config: Metric=%s, Type=%s, Dim=%lld, R=%d, L_insert=%d, Alpha=%.2f, L_search=%d",
                                  LMDiskannMetricTypeToString(config_.metric_type),
                                  LMDiskannVectorTypeToString(config_.node_vector_type),
                                  config_.dimensions, config_.r, config_.l_insert, config_.alpha, config_.l_search);
    result += StringUtil::Format("\n - Allocator Blocks Used: %lld", db_state_.allocator ? db_state_.allocator->GetSegmentCount() : 0);
    result += StringUtil::Format("\n - In-Memory Map Size: %lld", in_memory_rowid_map_.size());
    result += StringUtil::Format("\n - Entry Point RowID: %lld", static_cast<long long>(graph_entry_point_rowid_)); // Use static_cast for clarity
    // Replace ToString() with manual formatting
    result += StringUtil::Format("\n - Metadata Ptr: [BufferID=%lld, Offset=%lld, Meta=%d]",
                                 db_state_.metadata_ptr.GetBufferId(), db_state_.metadata_ptr.GetOffset(), db_state_.metadata_ptr.GetMetadata());
    result += StringUtil::Format("\n - Delete Queue Head: [BufferID=%lld, Offset=%lld, Meta=%d]",
                                 delete_queue_head_ptr_.GetBufferId(), delete_queue_head_ptr_.GetOffset(), delete_queue_head_ptr_.GetMetadata());
    return result;
}

void LMDiskannIndex::VerifyAllocations(IndexLock &state) {
    // TODO: Check if allocator has verification method
    // if (db_state_.allocator) { db_state_.allocator->Verify(); }
}

string LMDiskannIndex::GetConstraintViolationMessage(VerifyExistenceType verify_type, idx_t failed_index, DataChunk &input) {
    return "Constraint violation in LM_DISKANN index (Not supported)";
}

// --- Scan Method Implementations --- //

unique_ptr<IndexScanState> LMDiskannIndex::InitializeScan(ClientContext &context, const Vector &query_vector, idx_t k) {
    if (query_vector.GetType().id() != LogicalTypeId::ARRAY || ArrayType::GetChildType(query_vector.GetType()).id() != LogicalTypeId::FLOAT) {
        throw BinderException("LM_DISKANN query vector must be ARRAY<FLOAT>.");
    }
    idx_t query_dims = ArrayType::GetSize(query_vector.GetType());
    if (query_dims != config_.dimensions) {
        throw BinderException("Query vector dimension (%d) does not match index dimension (%d).", query_dims, config_.dimensions);
    }
    if (k == 0) {
         throw BinderException("Cannot perform index scan with k=0");
    }

    auto scan_state = make_uniq<LMDiskannScanState>(query_vector, k, config_.l_search);

    // PerformSearch will handle finding the entry point and initializing candidates
    return std::move(scan_state);
}

idx_t LMDiskannIndex::Scan(IndexScanState &state, Vector &result) {
     auto &scan_state = state.Cast<LMDiskannScanState>();
     idx_t output_count = 0;
     auto result_data = FlatVector::GetData<row_t>(result);

     // Perform the beam search, populating scan_state.top_candidates
     PerformSearch(scan_state, *this, config_, true); // Find exact distances for final ranking

     // Extract top-k results from the state's max-heap ({distance, rowid})
     std::vector<std::pair<float, row_t>> final_results;
     final_results.reserve(scan_state.top_candidates.size());
     while(!scan_state.top_candidates.empty()) {
          final_results.push_back(scan_state.top_candidates.top());
          scan_state.top_candidates.pop();
     }

     // Sort by distance ascending (first element of pair)
     std::sort(final_results.begin(), final_results.end());

     // Fill the result vector up to k or vector size
     for (const auto& candidate : final_results) {
          if (output_count < STANDARD_VECTOR_SIZE && output_count < scan_state.k) {
               result_data[output_count++] = candidate.second; // Get the row_id
          } else {
               break;
          }
     }

     return output_count;
}


// --- Helper Method Implementations (Private to LMDiskannIndex) --- //

void LMDiskannIndex::InitializeNewIndex(idx_t estimated_cardinality) {
    if (!db_state_.allocator) {
        throw InternalException("Allocator not initialized in InitializeNewIndex");
    }
    db_state_.metadata_ptr = db_state_.allocator->New();
    delete_queue_head_ptr_.Clear();
    graph_entry_point_ptr_.Clear();
    graph_entry_point_rowid_ = NumericLimits<row_t>::Maximum();
    // TODO: Initialize ART root pointer when implemented: rowid_map_root_ptr_.Clear();

    LMDiskannMetadata initial_metadata;
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

    PersistMetadata(db_state_.metadata_ptr, db_state_.db, *db_state_.allocator, initial_metadata);
    is_dirty_ = true;
}

void LMDiskannIndex::LoadFromStorage(const IndexStorageInfo &storage_info) {
    if (!db_state_.allocator || db_state_.metadata_ptr.Get() == 0) {
        throw InternalException("Allocator or metadata pointer invalid in LoadFromStorage");
    }

    LMDiskannMetadata loaded_metadata;
    LoadMetadata(db_state_.metadata_ptr, db_state_.db, *db_state_.allocator, loaded_metadata);

    if (loaded_metadata.format_version != format_version_) {
        throw IOException(StringUtil::Format("LM_DISKANN index format version mismatch: Found %d, expected %d. Index may be incompatible.",
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
    // rowid_map_root_ptr_ = loaded_metadata.rowid_map_root_ptr; // TODO: Load ART root

    node_layout_ = CalculateLayoutInternal(config_);
    idx_t expected_block_size = AlignValue<idx_t, Storage::SECTOR_SIZE>(node_layout_.total_node_size);
    if (block_size_bytes_ != expected_block_size) {
         throw IOException(StringUtil::Format("LM_DISKANN loaded block size (%lld) inconsistent with recalculated size (%lld) based on loaded parameters.",
                                              block_size_bytes_, expected_block_size));
    }
    if (db_state_.allocator->GetBlockSize() != block_size_bytes_) {
         throw IOException(StringUtil::Format("LM_DISKANN allocator block size (%lld) does not match loaded block size (%lld).",
                                              db_state_.allocator->GetBlockSize(), block_size_bytes_));
    }

    // TODO: Load ART and populate in-memory map placeholder
    Printer::Print("Warning: LMDiskannIndex loaded, but in-memory RowID map NOT populated from storage (ART integration needed).");

    if (graph_entry_point_ptr_.Get() != 0) {
        // FIXME: GetEntryPointRowId needs implementation (or RowID stored in node)
        graph_entry_point_rowid_ = GetEntryPointRowId(graph_entry_point_ptr_, db_state_.db, *db_state_.allocator);
    } else {
        graph_entry_point_rowid_ = NumericLimits<row_t>::Maximum();
    }

    is_dirty_ = false;
}

// --- Distance Helper Wrappers --- //

float LMDiskannIndex::CalculateApproxDistance(const float *query_ptr, const_data_ptr_t compressed_neighbor_ptr) {
    return duckdb::CalculateApproxDistance(query_ptr, compressed_neighbor_ptr, config_);
}

void LMDiskannIndex::CompressVectorForEdge(const float* input_vector, data_ptr_t output_compressed_vector) {
    if (!duckdb::CompressVectorForEdge(input_vector, output_compressed_vector, config_)) {
         throw InternalException("Failed to compress vector into Ternary format.");
    }
}

template<typename T_QUERY, typename T_NODE>
float LMDiskannIndex::CalculateExactDistance(const T_QUERY *query_ptr, const_data_ptr_t node_vector_ptr) {
    return duckdb::CalculateDistance<T_QUERY, T_NODE>(query_ptr, reinterpret_cast<const T_NODE*>(node_vector_ptr), config_);
}

void LMDiskannIndex::ConvertNodeVectorToFloat(const_data_ptr_t raw_node_vector, float* float_vector_out) {
    if (config_.node_vector_type == LMDiskannVectorType::FLOAT32) {
        memcpy(float_vector_out, raw_node_vector, config_.dimensions * sizeof(float));
    } else if (config_.node_vector_type == LMDiskannVectorType::INT8) {
        duckdb::ConvertToFloat<int8_t>(reinterpret_cast<const int8_t*>(raw_node_vector), float_vector_out, config_.dimensions);
    } else {
        throw InternalException("Unsupported node vector type in ConvertNodeVectorToFloat.");
    }
}

// Explicitly instantiate templates used within this file
template float LMDiskannIndex::CalculateExactDistance<float, float>(const float*, const_data_ptr_t);
template float LMDiskannIndex::CalculateExactDistance<float, int8_t>(const float*, const_data_ptr_t);

// --- Robust Pruning Helper --- //

void LMDiskannIndex::RobustPrune(row_t node_rowid, IndexPointer node_ptr,
                                 std::vector<std::pair<float, row_t>>& candidates) {

    uint32_t max_neighbors = config_.r;
    data_ptr_t node_data = nullptr;

    try {
        // Get writable data pointer
        node_data = GetNodeDataMutable(node_ptr);
        
        uint16_t current_neighbor_count = LMDiskannNodeAccessors::GetNeighborCount(node_data);
        row_t* current_neighbor_ids = LMDiskannNodeAccessors::GetNeighborIDsPtrMutable(node_data, node_layout_);

        vector<float> node_vector_float(config_.dimensions);
        const_data_ptr_t node_vector_raw_ptr = LMDiskannNodeAccessors::GetNodeVector(node_data, node_layout_);
        ConvertNodeVectorToFloat(node_vector_raw_ptr, node_vector_float.data());

        for (uint16_t i = 0; i < current_neighbor_count; ++i) {
            row_t existing_id = current_neighbor_ids[i];
            if (existing_id == NumericLimits<row_t>::Maximum()) continue;

            bool already_candidate = false;
            for(const auto& cand : candidates) {
                if (cand.second == existing_id) {
                    already_candidate = true;
                    break;
                }
            }
            if (already_candidate) continue;

            TernaryPlanesView existing_neighbor_planes = LMDiskannNodeAccessors::GetNeighborTernaryPlanes(node_data, node_layout_, i, config_.dimensions);
            if (existing_neighbor_planes.IsValid()) {
                 float dist = CalculateApproxDistance(node_vector_float.data(), existing_neighbor_planes.positive_plane);
                 candidates.push_back({dist, existing_id});
            } else {
                 Printer::Print(StringUtil::Format("Warning: Invalid planes for existing neighbor %lld in RobustPrune.", existing_id));
            }
        }

        // Sort by row_id first for unique()
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
             return a.second < b.second;
        });
        auto unique_end = std::unique(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
             return a.second == b.second;
        });
        candidates.erase(unique_end, candidates.end());

        // Sort again by distance for alpha pruning
        std::sort(candidates.begin(), candidates.end());

        std::vector<row_t> final_neighbor_ids;
        std::vector<vector<uint8_t>> final_compressed_neighbors;
        final_neighbor_ids.reserve(max_neighbors);
        final_compressed_neighbors.reserve(max_neighbors);
        vector<float> candidate_vector_float(config_.dimensions);

        for (const auto& candidate : candidates) {
            if (final_neighbor_ids.size() >= max_neighbors) break;

            row_t candidate_id = candidate.second;
            if (candidate_id == node_rowid) continue;

            IndexPointer candidate_ptr;
            if (!TryGetNodePointer(candidate_id, candidate_ptr)) continue;

            const_data_ptr_t cand_data_ptr;
            const_data_ptr_t cand_vec_ptr_raw;
            try {
                cand_data_ptr = GetNodeData(candidate_ptr); // Read-only access
                cand_vec_ptr_raw = LMDiskannNodeAccessors::GetNodeVector(cand_data_ptr, node_layout_);
                ConvertNodeVectorToFloat(cand_vec_ptr_raw, candidate_vector_float.data());
            } catch (...) { continue; }

            bool pruned = false;
            float exact_dist_node_to_candidate = ComputeExactDistanceFloat(
                 node_vector_float.data(), candidate_vector_float.data(), config_.dimensions, config_.metric_type
            );

            for (size_t i = 0; i < final_neighbor_ids.size(); ++i) {
                row_t existing_final_id = final_neighbor_ids[i];
                IndexPointer existing_final_ptr;
                if (!TryGetNodePointer(existing_final_id, existing_final_ptr)) continue;

                vector<float> existing_final_vector_float(config_.dimensions);
                try {
                    auto existing_final_data_ptr = GetNodeData(existing_final_ptr); // Read-only access
                    auto existing_final_vec_ptr_raw = LMDiskannNodeAccessors::GetNodeVector(existing_final_data_ptr, node_layout_);
                    ConvertNodeVectorToFloat(existing_final_vec_ptr_raw, existing_final_vector_float.data());
                } catch (...) { continue; }

                float dist_existing_final_to_candidate = ComputeExactDistanceFloat(
                    existing_final_vector_float.data(), candidate_vector_float.data(), config_.dimensions, config_.metric_type
                );

                if (exact_dist_node_to_candidate > config_.alpha * dist_existing_final_to_candidate) {
                    pruned = true;
                    break;
                }
            }

            if (!pruned) {
                final_neighbor_ids.push_back(candidate_id);
                final_compressed_neighbors.emplace_back(node_layout_.ternary_edge_size_bytes);
                CompressVectorForEdge(candidate_vector_float.data(), final_compressed_neighbors.back().data());
            }
        }

        uint16_t final_count = static_cast<uint16_t>(final_neighbor_ids.size());
        D_ASSERT(final_count <= max_neighbors);

        LMDiskannNodeAccessors::SetNeighborCount(node_data, final_count);
        row_t* dest_ids = LMDiskannNodeAccessors::GetNeighborIDsPtrMutable(node_data, node_layout_);

        idx_t plane_size_bytes = GetTernaryPlaneSizeBytes(config_.dimensions);
        data_ptr_t dest_pos_planes_base = node_data + node_layout_.neighbor_pos_planes_offset;
        data_ptr_t dest_neg_planes_base = node_data + node_layout_.neighbor_neg_planes_offset;

        for (uint16_t i = 0; i < final_count; ++i) {
            dest_ids[i] = final_neighbor_ids[i];
            data_ptr_t dest_pos_plane_i = dest_pos_planes_base + i * plane_size_bytes;
            data_ptr_t dest_neg_plane_i = dest_neg_planes_base + i * plane_size_bytes;
            memcpy(dest_pos_plane_i, final_compressed_neighbors[i].data(), plane_size_bytes);
            memcpy(dest_neg_plane_i, final_compressed_neighbors[i].data() + plane_size_bytes, plane_size_bytes);
        }
        for (uint16_t i = final_count; i < max_neighbors; ++i) {
             dest_ids[i] = NumericLimits<row_t>::Maximum();
             data_ptr_t dest_pos_plane_i = dest_pos_planes_base + i * plane_size_bytes;
             data_ptr_t dest_neg_plane_i = dest_neg_planes_base + i * plane_size_bytes;
             memset(dest_pos_plane_i, 0, plane_size_bytes);
             memset(dest_neg_plane_i, 0, plane_size_bytes);
        }

        // Mark buffer modified (assuming handle destructor manages this)
        // FIXME: Verify block modification handling
        // auto block = BufferManager::GetBufferManager(db_state_.db).GetBlock(handle.GetBlockId());
        // if (block) block->SetModified();

    } catch (const std::exception& e) {
         Printer::Print(StringUtil::Format("Error during RobustPrune for node %lld: %s", node_rowid, e.what()));
         // Re-throw? Or just log and potentially leave node in inconsistent state?
         throw;
    }
}


// --- Insertion Helper --- //
void LMDiskannIndex::FindAndConnectNeighbors(row_t new_node_rowid, IndexPointer new_node_ptr, const float *new_node_vector_float) {
    row_t entry_point = GetEntryPoint();
    if (entry_point == NumericLimits<row_t>::Maximum()) {
        throw InternalException("FindAndConnectNeighbors called with no entry point.");
    }

    Vector query_vec_handle(LogicalType::ARRAY(LogicalType::FLOAT, config_.dimensions));
    memcpy(FlatVector::GetData<float>(query_vec_handle), new_node_vector_float, config_.dimensions * sizeof(float));
    query_vec_handle.Flatten(1);

    LMDiskannScanState search_state(query_vec_handle, config_.l_insert, config_.l_insert);
    PerformSearch(search_state, *this, config_, false);

    std::vector<std::pair<float, row_t>> potential_neighbors;
    potential_neighbors.reserve(search_state.top_candidates.size() + 1);
    while(!search_state.top_candidates.empty()) {
         potential_neighbors.push_back(search_state.top_candidates.top());
         search_state.top_candidates.pop();
    }
    potential_neighbors.push_back({0.0f, new_node_rowid}); // Add self

    RobustPrune(new_node_rowid, new_node_ptr, potential_neighbors);

    // --- Update Neighbors (Reciprocal Edges) --- //
    auto new_node_data_ro = GetNodeData(new_node_ptr);
    uint16_t final_new_neighbor_count = LMDiskannNodeAccessors::GetNeighborCount(new_node_data_ro);
    const row_t* final_new_neighbor_ids = LMDiskannNodeAccessors::GetNeighborIDsPtr(new_node_data_ro, node_layout_);

    vector<uint8_t> new_node_compressed_storage(node_layout_.ternary_edge_size_bytes);
    CompressVectorForEdge(new_node_vector_float, new_node_compressed_storage.data());

    vector<float> neighbor_float_vec(config_.dimensions);
    for(uint16_t i = 0; i < final_new_neighbor_count; ++i) {
         row_t neighbor_rowid = final_new_neighbor_ids[i];
         if (neighbor_rowid == NumericLimits<row_t>::Maximum()) continue;

         IndexPointer neighbor_ptr;
         if (!TryGetNodePointer(neighbor_rowid, neighbor_ptr)) continue;

         std::vector<std::pair<float, row_t>> neighbor_candidates;
         try {
             const_data_ptr_t neighbor_data_ro;
             const_data_ptr_t neighbor_vec_ptr_raw;

             neighbor_data_ro = GetNodeData(neighbor_ptr);
             neighbor_vec_ptr_raw = LMDiskannNodeAccessors::GetNodeVector(neighbor_data_ro, node_layout_);
             ConvertNodeVectorToFloat(neighbor_vec_ptr_raw, neighbor_float_vec.data());

             float dist_neighbor_to_new = ComputeExactDistanceFloat(neighbor_float_vec.data(), new_node_vector_float,
                                                                     config_.dimensions, config_.metric_type);

             neighbor_candidates.push_back({dist_neighbor_to_new, new_node_rowid});

             RobustPrune(neighbor_rowid, neighbor_ptr, neighbor_candidates);

         } catch (const std::exception &e) {
              Printer::Print(StringUtil::Format("Warning: Failed to update neighbor %lld for new node %lld: %s", neighbor_rowid, new_node_rowid, e.what()));
         }
    }
}


// --- Deletion Helper --- //
void LMDiskannIndex::EnqueueDeletion(row_t deleted_row_id) {
    duckdb::EnqueueDeletion(deleted_row_id, delete_queue_head_ptr_, db_state_.db, *db_state_.allocator);
    is_dirty_ = true;
}

void LMDiskannIndex::ProcessDeletionQueue() {
     Printer::Print("LMDiskannIndex::ProcessDeletionQueue is not implemented.");
}


// --- Entry Point Helpers --- //
row_t LMDiskannIndex::GetEntryPoint() {
     if (graph_entry_point_rowid_ != NumericLimits<row_t>::Maximum()) {
         IndexPointer ptr_check;
         if (TryGetNodePointer(graph_entry_point_rowid_, ptr_check)) {
            return graph_entry_point_rowid_;
         } else {
            Printer::Print(StringUtil::Format("Warning: Cached entry point %lld deleted.", graph_entry_point_rowid_));
            graph_entry_point_ptr_.Clear();
            graph_entry_point_rowid_ = NumericLimits<row_t>::Maximum();
            is_dirty_ = true;
         }
     }

     row_t random_id = GetRandomNodeID();
     if (random_id != NumericLimits<row_t>::Maximum()) {
          IndexPointer random_ptr;
          if(TryGetNodePointer(random_id, random_ptr)) {
             SetEntryPoint(random_id, random_ptr);
             return random_id;
          } else {
               Printer::Print(StringUtil::Format("Warning: Random node %lld chosen as entry point could not be found in map.", random_id));
          }
     }
     return NumericLimits<row_t>::Maximum();
}

void LMDiskannIndex::SetEntryPoint(row_t row_id, IndexPointer node_ptr) {
    graph_entry_point_rowid_ = row_id;
    graph_entry_point_ptr_ = node_ptr;
    is_dirty_ = true;
    // PersistMetadata(); // Persist immediately? Let Checkpoint handle it.
}

row_t LMDiskannIndex::GetRandomNodeID() {
    // Placeholder: Inefficient map iteration. Replace with ART sampling.
    if (in_memory_rowid_map_.empty()) {
        return NumericLimits<row_t>::Maximum();
    }
    // Instantiate a local RandomEngine. Consider using RandomEngine::Get(context) if available.
    RandomEngine generator;
    std::uniform_int_distribution<idx_t> distribution(0, in_memory_rowid_map_.size() - 1);
    idx_t random_idx = distribution(generator);
    auto it = std::next(in_memory_rowid_map_.begin(), random_idx);
    return (it != in_memory_rowid_map_.end()) ? it->first : NumericLimits<row_t>::Maximum();
}

// --- Storage interaction helpers (using in-memory map for now) --- //

bool LMDiskannIndex::TryGetNodePointer(row_t row_id, IndexPointer &node_ptr) {
    auto it = in_memory_rowid_map_.find(row_id);
    if (it != in_memory_rowid_map_.end()) {
        node_ptr = it->second;
        // TODO: Verify pointer validity in allocator?
        return true;
    }
    node_ptr.Clear();
    return false;
}

IndexPointer LMDiskannIndex::AllocateNode(row_t row_id) {
    if (in_memory_rowid_map_.count(row_id)) {
        IndexPointer existing_ptr;
        if (TryGetNodePointer(row_id, existing_ptr)) {
             Printer::Print(StringUtil::Format("Warning: AllocateNode called for existing row_id %lld. Reusing block.", row_id));
             return existing_ptr;
        }
        in_memory_rowid_map_.erase(row_id);
    }
    if (!db_state_.allocator) {
         throw InternalException("Allocator not initialized in AllocateNode.");
    }
    IndexPointer new_node_ptr = db_state_.allocator->New();
    if (new_node_ptr.Get() == 0) {
         throw InternalException("Failed to allocate new block in AllocateNode.");
    }
    in_memory_rowid_map_[row_id] = new_node_ptr;
    return new_node_ptr;
}

void LMDiskannIndex::DeleteNodeFromMapAndFreeBlock(row_t row_id) {
    auto it = in_memory_rowid_map_.find(row_id);
    if (it != in_memory_rowid_map_.end()) {
        IndexPointer node_ptr = it->second;
        if (db_state_.allocator) {
             db_state_.allocator->Free(node_ptr);
        }
        in_memory_rowid_map_.erase(it);
    } else {
        // Optional Warning
    }
}

data_ptr_t LMDiskannIndex::GetNodeDataMutable(IndexPointer node_ptr) {
     // Replace IsValid with check against 0
     if (node_ptr.Get() == 0) { throw IOException("Invalid node pointer provided to GetNodeDataMutable."); }
     if (!db_state_.allocator) { throw InternalException("Allocator not initialized in GetNodeBuffer."); }
     return db_state_.allocator->Get(node_ptr, true); // Get mutable pointer, mark dirty
}

const_data_ptr_t LMDiskannIndex::GetNodeData(IndexPointer node_ptr) {
     // Replace IsValid with check against 0
     if (node_ptr.Get() == 0) { throw IOException("Invalid node pointer provided to GetNodeData."); }
     if (!db_state_.allocator) { throw InternalException("Allocator not initialized in GetNodeBuffer."); }
     // Use the allocator's Get method directly, requesting non-dirty access
     return db_state_.allocator->Get(node_ptr, false); // Get const pointer, don't mark dirty
}

} // namespace duckdb

