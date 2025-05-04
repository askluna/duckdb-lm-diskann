#include "lm_diskann_index.hpp"

// Include refactored component headers
#include "config.hpp"
#include "node.hpp"
#include "storage.hpp" // Include for prototypes (even if placeholders)
#include "search.hpp"
#include "distance.hpp"
#include "state.hpp"

// Include necessary DuckDB headers used in this file
#include "duckdb/execution/index/fixed_size_allocator.hpp"
#include "duckdb/storage/index_storage_info.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp" // For Flatten
#include "duckdb/common/limits.hpp" // For NumericLimits
#include "duckdb/common/printer.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/metadata/metadata_reader.hpp"
#include "duckdb/storage/metadata/metadata_writer.hpp"

#include <vector>
#include <cstring> // For memcpy
#include <algorithm> // For std::sort, std::min, std::max
#include <random> // For default_random_engine
#include <map> // For in-memory map
#include <set> // For intermediate pruning steps

namespace duckdb {

// --- LMDiskannIndex Constructor ---

LMDiskannIndex::LMDiskannIndex(const string &name, IndexConstraintType index_constraint_type,
                               const vector<column_t> &column_ids, TableIOManager &table_io_manager,
                               const vector<unique_ptr<Expression>> &unbound_expressions,
                               AttachedDatabase &db, const case_insensitive_map_t<Value> &options,
                               const IndexStorageInfo &storage_info, idx_t estimated_cardinality)
    : BoundIndex(name, LMDiskannIndex::TYPE_NAME, index_constraint_type, column_ids, table_io_manager,
                 unbound_expressions, db),
      db_(db), // Store reference to database
      table_io_manager_(table_io_manager),
      format_version_(LMDISKANN_CURRENT_FORMAT_VERSION), // Set current format version
      graph_entry_point_rowid_(NumericLimits<row_t>::Maximum()) // Initialize entry point rowid as invalid
{
    // Basic validation (moved from original constructor)
    if (index_constraint_type != IndexConstraintType::NONE) {
        throw NotImplementedException("LM_DISKANN indexes do not support UNIQUE or PRIMARY KEY constraints");
    }
    if (logical_types.size() != 1) {
         throw BinderException("LM_DISKANN index can only be created over a single column.");
    }
    if (logical_types[0].id() != LogicalTypeId::ARRAY) {
        throw BinderException("LM_DISKANN index can only be created over ARRAY types.");
    }

    // --- 1. Determine Node Vector Type and Dimensions ---
    indexed_column_type_ = logical_types[0];
    auto &vector_child_type = ArrayType::GetChildType(indexed_column_type_);
    dimensions_ = ArrayType::GetSize(indexed_column_type_);
    if (dimensions_ == 0) {
         throw BinderException("Cannot create LM_DISKANN index on ARRAY with unknown size.");
    }
    constexpr idx_t MAX_VECTOR_SZ = 65536; // Define appropriately
    if (dimensions_ > MAX_VECTOR_SZ) {
         throw BinderException("Cannot create LM_DISKANN index on ARRAY with dimensions > %d", MAX_VECTOR_SZ);
    }

    switch (vector_child_type.id()) {
    case LogicalTypeId::FLOAT: node_vector_type_ = LMDiskannVectorType::FLOAT32; break;
    case LogicalTypeId::TINYINT: node_vector_type_ = LMDiskannVectorType::INT8; break;
    case LogicalTypeId::FLOAT16: node_vector_type_ = LMDiskannVectorType::FLOAT16; break;
    default: throw BinderException("Unsupported vector type for LM_DISKANN index: %s.", vector_child_type.ToString());
    }

    // --- 2. Parse User-Provided Options (using config module) ---
    ParseOptions(options, metric_type_, edge_vector_type_param_, r_, l_insert_, alpha_, l_search_);

    // --- 3. Resolve Edge Vector Type ---
    if (edge_vector_type_param_ == LMDiskannEdgeType::SAME_AS_NODE) {
        resolved_edge_vector_type_ = node_vector_type_;
    } else {
        // Map EdgeType enum back to VectorType enum if applicable
        switch(edge_vector_type_param_) {
            case LMDiskannEdgeType::FLOAT32: resolved_edge_vector_type_ = LMDiskannVectorType::FLOAT32; break;
            case LMDiskannEdgeType::FLOAT16: resolved_edge_vector_type_ = LMDiskannVectorType::FLOAT16; break;
            case LMDiskannEdgeType::INT8:    resolved_edge_vector_type_ = LMDiskannVectorType::INT8; break;
            case LMDiskannEdgeType::FLOAT1BIT: resolved_edge_vector_type_ = LMDiskannVectorType::UNKNOWN; break;
            case LMDiskannEdgeType::TERNARY: resolved_edge_vector_type_ = LMDiskannVectorType::UNKNOWN; break; // Special type
            default: throw InternalException("Unexpected LMDiskannEdgeType parameter");
        }
    }

    // --- 4. Validate Parameters (using config module) ---
    ValidateParameters(metric_type_, edge_vector_type_param_, r_, l_insert_, alpha_, l_search_);

    // --- 5. Calculate Sizes and Layout (using config module) ---
    CalculateSizesAndLayout(); // Sets member variables

    // --- 6. Initialize Storage ---
    allocator_ = make_uniq<FixedSizeAllocator>(block_size_bytes_, table_io_manager_.GetIndexBlockManager());

    // --- 7. Load Existing Index or Initialize New One ---
    if (storage_info.IsValid()) {
        LoadFromStorage(storage_info);
    } else {
        InitializeNewIndex(estimated_cardinality);
    }

    // --- 8. Initialize RowID Mapping (In-Memory - No Load Needed Yet) ---
    // If persistent map were used: LoadRowIDMap();

    // --- 9. Load PQ Codebooks (Placeholder) ---
    // LoadPQ();

    // --- Logging ---
    Printer::Print(StringUtil::Format("LM_DISKANN Index '%s': Metric=%d, Dim=%lld, R=%d, L_insert=%d, Alpha=%.2f, L_search=%d, BlockSize=%lld, EdgeType=%d",
                                      name, (int)metric_type_, dimensions_, r_, l_insert_, alpha_, l_search_, block_size_bytes_, (int)edge_vector_type_param_));
}

LMDiskannIndex::~LMDiskannIndex() = default;

// --- BoundIndex Method Implementations (Delegating) ---

ErrorData LMDiskannIndex::Append(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) {
    // Use Insert logic for now, can be optimized later
    if (entries.size() == 0) {
        return ErrorData();
    }
    row_identifiers.Flatten(entries.size()); // Ensure row identifiers are flat

    DataChunk input_chunk;
    input_chunk.InitializeEmpty({entries.data[0].GetType()});
    Vector row_id_vector(LogicalType::ROW_TYPE);

    for(idx_t i = 0; i < entries.size(); ++i) {
        input_chunk.Reset();
        input_chunk.data[0].Slice(entries.data[0], i, i + 1);
        input_chunk.SetCardinality(1);

        row_id_vector.Slice(row_identifiers, i, i + 1);
        row_id_vector.Flatten(1);

        auto err = Insert(lock, input_chunk, row_id_vector);
        if (err.HasError()) {
            return err;
        }
    }
    return ErrorData(); // Success
}

void LMDiskannIndex::CommitDrop(IndexLock &index_lock) {
    // Call storage module helper if created, or handle directly
    if (allocator_) {
        allocator_->Reset();
    }
    metadata_ptr_.Clear();
    // Drop RowID map resources (clear in-memory map)
    in_memory_rowid_map_.clear();
    // Drop Delete queue resources
    // FIXME: Need to free blocks used by delete queue if not using main allocator
    delete_queue_head_ptr_.Clear();
}

void LMDiskannIndex::Delete(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) {
    auto row_ids_data = FlatVector::GetData<row_t>(row_identifiers);
    bool changes_made = false;
    for (idx_t i = 0; i < entries.size(); ++i) {
        row_t row_id = row_ids_data[i];
        try {
             // 1. Remove from RowID map and free block (immediate)
             DeleteNodeFromMapAndFreeBlock(row_id); // Uses in-memory map

             // 2. Add to persistent delete queue for deferred neighbor updates
             EnqueueDeletion(row_id); // Uses allocator, updates head pointer

             // 3. Handle entry point deletion
             if (row_id == graph_entry_point_rowid_) {
                  graph_entry_point_ptr_.Clear();
                  graph_entry_point_rowid_ = NumericLimits<row_t>::Maximum();
                  is_dirty_ = true; // Mark dirty as entry point changed
             }
             changes_made = true; // Set flag if deletion happened

        } catch (NotImplementedException &e) {
             throw; // Re-throw if map deletion isn't implemented
        } catch (std::exception &e) {
             Printer::Warning("Failed to delete node for row_id %lld: %s", row_id, e.what());
        }
    }
    // is_dirty_ is set by DeleteNodeFromMapAndFreeBlock and EnqueueDeletion
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

     // --- Convert input vector to float for calculations ---
     vector<float> input_vector_float_storage; // Allocate only if needed
     const float* input_vector_float_ptr;
     try {
         if (node_vector_type_ == LMDiskannVectorType::FLOAT32) {
             input_vector_float_ptr = reinterpret_cast<const float*>(input_vector_raw_ptr);
         } else {
             input_vector_float_storage.resize(dimensions_);
             if (node_vector_type_ == LMDiskannVectorType::FLOAT16) {
                 ConvertToFloat(reinterpret_cast<const float16_t*>(input_vector_raw_ptr), input_vector_float_storage.data(), dimensions_);
             } else if (node_vector_type_ == LMDiskannVectorType::INT8) {
                 ConvertToFloat(reinterpret_cast<const int8_t*>(input_vector_raw_ptr), input_vector_float_storage.data(), dimensions_);
             } else { return ErrorData("Unsupported node vector type for insertion."); }
             input_vector_float_ptr = input_vector_float_storage.data();
         }
     } catch (std::exception &e) { return ErrorData(e.what()); }
     // --- End Input Vector Handling ---

     // 1. Allocate block (using in-memory map helper)
     IndexPointer new_node_ptr;
     try {
         new_node_ptr = AllocateNode(row_id);
     } catch (std::exception &e) { return ErrorData(e); }

     // 2. Pin buffer (using storage helper)
     auto new_node_handle = GetNodeBuffer(new_node_ptr, true); // Writable
     auto new_node_data = new_node_handle.Ptr();

     // 3. Initialize block (using node helper)
     LMDiskannNodeAccessors::InitializeNodeBlock(new_node_data, block_size_bytes_);

     // 4. Write node's full vector (using node helper)
     memcpy(LMDiskannNodeAccessors::GetNodeVectorPtrMutable(new_node_data, node_layout_),
            input_vector_raw_ptr, node_vector_size_bytes_);

     // 5. Find neighbors and connect
     row_t entry_point_row_id = GetEntryPoint(); // Uses internal state + storage helpers

     if (entry_point_row_id == NumericLimits<row_t>::Maximum()) {
         LMDiskannNodeAccessors::SetNeighborCount(new_node_data, 0);
         SetEntryPoint(row_id, new_node_ptr); // Uses internal state + storage helpers
     } else {
         try {
             // Calls search, distance, node, storage modules internally
             FindAndConnectNeighbors(row_id, new_node_ptr, input_vector_float_ptr);
         } catch (std::exception &e) {
             // Clean up allocated node if connection fails?
             try { DeleteNodeFromMapAndFreeBlock(row_id); } catch (...) {} // Best effort cleanup
             return ErrorData(StringUtil::Format("Failed to connect neighbors for node %lld: %s", row_id, e.what()));
         }
     }

     // 6. Mark buffer modified
     new_node_handle.SetModified();

     return ErrorData(); // Success
}

IndexStorageInfo LMDiskannIndex::GetStorageInfo(const bool get_buffers) {
     // Persist metadata if dirty
     if (is_dirty_) {
          PersistMetadata(); // Persists internal state via storage module
     }
     IndexStorageInfo info;
     info.name = name;
     info.root_block = metadata_ptr_.GetBlockId();
     info.root_offset = metadata_ptr_.GetOffset();
     info.allocator_infos.push_back(allocator_->GetInfo());
     // Add rowid_map_root persistence info if implemented
     if (get_buffers) { /* Handle WAL buffer logic if ever needed */ }
     return info;
}

idx_t LMDiskannIndex::GetInMemorySize() {
    // Delegate to storage module if created, or calculate here
    idx_t base_size = allocator_ ? allocator_->GetInMemorySize() : 0;
    // Add RowID map size (in-memory version)
    base_size += in_memory_rowid_map_.size() * (sizeof(row_t) + sizeof(IndexPointer) + 16); // Estimate map overhead
    return base_size;
}

bool LMDiskannIndex::MergeIndexes(IndexLock &state, BoundIndex &other_index) {
    throw NotImplementedException("LMDiskannIndex::MergeIndexes not implemented");
    return false;
}

void LMDiskannIndex::Vacuum(IndexLock &state) {
    // Call storage module helper
    ProcessDeletionQueue(); // Placeholder call
    // is_dirty_ should be set within ProcessDeletionQueue if changes occur
}

string LMDiskannIndex::VerifyAndToString(IndexLock &state, const bool only_verify) {
    // Delegate verification logic to storage/search modules if needed
    if (only_verify) { return "VerifyAndToString(verify_only) not implemented for LM_DISKANN"; }
    else { return "VerifyAndToString not implemented for LM_DISKANN"; }
}

void LMDiskannIndex::VerifyAllocations(IndexLock &state) {
    // Delegate to storage module if needed
    // allocator_->Verify();
}

string LMDiskannIndex::GetConstraintViolationMessage(VerifyExistenceType verify_type, idx_t failed_index, DataChunk &input) {
    return "Constraint violation in LM_DISKANN index (Not supported)";
}

// --- Scan Method Implementations ---

unique_ptr<IndexScanState> LMDiskannIndex::InitializeScan(ClientContext &context, const Vector &query_vector, idx_t k) {
    // 1. Validate query vector (using config module helpers if needed)
    if (query_vector.GetType().id() != LogicalTypeId::ARRAY || ArrayType::GetChildType(query_vector.GetType()).id() != LogicalTypeId::FLOAT) {
        throw BinderException("LM_DISKANN query vector must be ARRAY<FLOAT>.");
    }
    idx_t query_dims = ArrayType::GetSize(query_vector.GetType());
    if (query_dims != dimensions_) { throw BinderException("Query vector dimension mismatch."); }

    // 2. Create scan state (using state module struct)
    auto scan_state = make_uniq<LMDiskannScanState>(query_vector, k, l_search_);

    // 3. Find initial entry point(s) (using storage module helpers)
    row_t start_node_id = GetEntryPoint(); // Uses internal state + storage helpers

    if (start_node_id != NumericLimits<row_t>::Maximum()) {
        IndexPointer start_ptr;
        if (TryGetNodePointer(start_node_id, start_ptr)) { // Uses in-memory map
            try {
                auto handle = GetNodeBuffer(start_ptr); // Uses internal state + storage helpers
                auto block_data = handle.Ptr();
                auto node_vec_ptr = LMDiskannNodeAccessors::GetNodeVectorPtr(block_data, node_layout_);
                // Calculate initial distance using the appropriate type
                float approx_dist = CalculateApproxDistance(scan_state->query_vector_ptr, node_vec_ptr); // Uses internal state + distance helpers
                scan_state->candidates.push({approx_dist, start_node_id});
            } catch (std::exception &e) { Printer::Warning("Failed to initialize scan with start node %lld: %s", start_node_id, e.what()); }
        } else {
             Printer::Warning("Persisted entry point node %lld not found, trying random.", start_node_id);
             start_node_id = GetRandomNodeID(); // Uses in-memory map iteration
             if (start_node_id != NumericLimits<row_t>::Maximum() && TryGetNodePointer(start_node_id, start_ptr)) {
                  try {
                     auto handle = GetNodeBuffer(start_ptr);
                     auto block_data = handle.Ptr();
                     auto node_vec_ptr = LMDiskannNodeAccessors::GetNodeVectorPtr(block_data, node_layout_);
                     float approx_dist = CalculateApproxDistance(scan_state->query_vector_ptr, node_vec_ptr);
                     scan_state->candidates.push({approx_dist, start_node_id});
                  } catch (std::exception &e) { Printer::Warning("Failed to initialize scan with random node %lld: %s", start_node_id, e.what()); }
             }
        }
    } else { Printer::Warning("No valid entry point found for scan initialization."); }

    return std::move(scan_state);
}

idx_t LMDiskannIndex::Scan(IndexScanState &state, Vector &result) {
     auto &scan_state = state.Cast<LMDiskannScanState>();
     idx_t output_count = 0;
     auto result_data = FlatVector::GetData<row_t>(result);

     // Perform the beam search (call search module function)
     PerformSearch(scan_state, *this, true); // Pass self as context, find exact distances

     // Extract top-k results
     std::sort(scan_state.top_candidates.begin(), scan_state.top_candidates.end());
     for (const auto& candidate : scan_state.top_candidates) {
          if (output_count < STANDARD_VECTOR_SIZE && output_count < scan_state.k) {
               result_data[output_count++] = candidate.second;
          } else { break; }
     }
     return output_count;
}


// --- Helper Method Implementations (Private to LMDiskannIndex) ---

void LMDiskannIndex::InitializeNewIndex(idx_t estimated_cardinality) {
     // Call storage module helper
     metadata_ptr_ = allocator_->New();
     graph_entry_point_ptr_.Clear();
     graph_entry_point_rowid_ = NumericLimits<row_t>::Maximum();
     delete_queue_head_ptr_.Clear();
     in_memory_rowid_map_.clear(); // Clear map for new index
     is_dirty_ = true;
     PersistMetadata(); // Persists internal state via storage module
}

void LMDiskannIndex::LoadFromStorage(const IndexStorageInfo &storage_info) {
     // Call storage module helper
     LoadMetadata(storage_info.GetMetadataPointer(), db_, *allocator_,
                  format_version_, metric_type_, node_vector_type_, edge_vector_type_param_,
                  dimensions_, r_, l_insert_, alpha_, l_search_, block_size_bytes_,
                  graph_entry_point_ptr_, delete_queue_head_ptr_ /*, rowid_map_root_ptr_ */);

     // Post-load processing (moved from original constructor)
     if (edge_vector_type_param_ == LMDiskannEdgeType::SAME_AS_NODE) { resolved_edge_vector_type_ = node_vector_type_; }
     else {
         switch(edge_vector_type_param_) {
             case LMDiskannEdgeType::FLOAT32: resolved_edge_vector_type_ = LMDiskannVectorType::FLOAT32; break;
             case LMDiskannEdgeType::FLOAT16: resolved_edge_vector_type_ = LMDiskannVectorType::FLOAT16; break;
             case LMDiskannEdgeType::INT8:    resolved_edge_vector_type_ = LMDiskannVectorType::INT8; break;
             case LMDiskannEdgeType::FLOAT1BIT: resolved_edge_vector_type_ = LMDiskannVectorType::UNKNOWN; break;
             case LMDiskannEdgeType::TERNARY: resolved_edge_vector_type_ = LMDiskannVectorType::UNKNOWN; break; // Special type
             default: throw InternalException("Unexpected loaded LMDiskannEdgeType parameter");
         }
     }
     CalculateSizesAndLayout(); // Recalculate layout
     // Check block size consistency
     idx_t expected_block_size = block_size_bytes_;
     // ... (block size check logic as before) ...
     ValidateParameters(metric_type_, edge_vector_type_param_, r_, l_insert_, alpha_, l_search_); // Re-validate

     // FIXME: Load RowID map from persistent storage if implemented
     // For in-memory, we need to rebuild it by scanning allocator, which is slow.
     // Or accept that the in-memory map is lost on reload.
     in_memory_rowid_map_.clear();
     Printer::Warning("In-memory RowID map is not persisted/loaded.");


     // Get entry point rowid if pointer is valid
     if (graph_entry_point_ptr_.IsValid()) {
         // FIXME: Need inverse mapping from IndexPointer to row_id
         graph_entry_point_rowid_ = -2; // Placeholder
     } else { graph_entry_point_rowid_ = NumericLimits<row_t>::Maximum(); }

     is_dirty_ = false;
}

void LMDiskannIndex::PersistMetadata() {
     // Call storage module helper
     // Update map root pointer if needed: rowid_map_root_ptr = rowid_map_->GetRoot();
     duckdb::PersistMetadata(metadata_ptr_, db_, *allocator_,
                             format_version_, metric_type_, node_vector_type_, edge_vector_type_param_,
                             dimensions_, r_, l_insert_, alpha_, l_search_, block_size_bytes_,
                             graph_entry_point_ptr_, delete_queue_head_ptr_ /*, rowid_map_root_ptr_ */);
     is_dirty_ = false;
}

// Wrapper for distance calculation using index state
template <typename T_A, typename T_B>
float LMDiskannIndex::CalculateDistance(const T_A *vec_a, const T_B *vec_b) {
    // Delegate to distance module function
    // Determine types based on template arguments
    LMDiskannVectorType type_a = LMDiskannVectorType::UNKNOWN;
    LMDiskannVectorType type_b = LMDiskannVectorType::UNKNOWN;
    if constexpr (std::is_same_v<T_A, float>) type_a = LMDiskannVectorType::FLOAT32;
    else if constexpr (std::is_same_v<T_A, float16_t>) type_a = LMDiskannVectorType::FLOAT16;
    else if constexpr (std::is_same_v<T_A, int8_t>) type_a = LMDiskannVectorType::INT8;

    if constexpr (std::is_same_v<T_B, float>) type_b = LMDiskannVectorType::FLOAT32;
    else if constexpr (std::is_same_v<T_B, float16_t>) type_b = LMDiskannVectorType::FLOAT16;
    else if constexpr (std::is_same_v<T_B, int8_t>) type_b = LMDiskannVectorType::INT8;

    if (type_a == LMDiskannVectorType::UNKNOWN || type_b == LMDiskannVectorType::UNKNOWN) {
        throw InternalException("Unknown vector type in CalculateDistance template.");
    }

    return ComputeDistance(const_data_ptr_cast(vec_a), type_a,
                           const_data_ptr_cast(vec_b), type_b,
                           dimensions_, metric_type_);
}
// Instantiate template for float, float
template float LMDiskannIndex::CalculateDistance<float, float>(const float*, const float*);
// Instantiate templates needed for RobustPrune/FindAndConnectNeighbors
template float LMDiskannIndex::CalculateDistance<float16_t, float>(const float16_t*, const float*);
template float LMDiskannIndex::CalculateDistance<int8_t, float>(const int8_t*, const float*);


// Wrapper for approximate distance calculation using index state
float LMDiskannIndex::CalculateApproxDistance(const float *query_ptr, const_data_ptr_t compressed_neighbor_ptr) {
    // Delegate to distance module function
    return ComputeApproxDistance(query_ptr, compressed_neighbor_ptr, dimensions_, metric_type_, resolved_edge_vector_type_);
}

// Helper to apply robust pruning to a node's neighbor list, considering new candidates.
// Modifies the node's block directly.
// `candidates` contains pairs of (distance_from_node_to_candidate, candidate_rowid)
void LMDiskannIndex::RobustPrune(row_t node_rowid, IndexPointer node_ptr,
                                 std::vector<std::pair<float, row_t>>& candidates, // Input: New candidates (dist, id)
                                 uint32_t max_neighbors) { // R

    auto handle = GetNodeBuffer(node_ptr, true); // Get writable buffer
    auto node_data = handle.Ptr();

    uint16_t current_neighbor_count = LMDiskannNodeAccessors::GetNeighborCount(node_data);
    row_t* current_neighbor_ids = LMDiskannNodeAccessors::GetNeighborIDsPtrMutable(node_data, node_layout_);
    data_ptr_t current_compressed_base = LMDiskannNodeAccessors::GetCompressedNeighborPtrMutable(node_data, node_layout_, 0, edge_vector_size_bytes_);

    // Add existing neighbors to the candidate pool
    vector<float> node_vector_float(dimensions_); // For distance calculations
    auto node_vector_raw_ptr = LMDiskannNodeAccessors::GetNodeVectorPtr(node_data, node_layout_);
    // Convert node vector to float for consistent distance calculation during pruning
    try {
        if (node_vector_type_ == LMDiskannVectorType::FLOAT32) {
            ConvertToFloat(reinterpret_cast<const float*>(node_vector_raw_ptr), node_vector_float.data(), dimensions_);
        } else if (node_vector_type_ == LMDiskannVectorType::FLOAT16) {
            ConvertToFloat(reinterpret_cast<const float16_t*>(node_vector_raw_ptr), node_vector_float.data(), dimensions_);
        } else if (node_vector_type_ == LMDiskannVectorType::INT8) {
            ConvertToFloat(reinterpret_cast<const int8_t*>(node_vector_raw_ptr), node_vector_float.data(), dimensions_);
        } else { throw InternalException("Unsupported node type in RobustPrune"); }
    } catch (...) { /* Handle conversion error */ return; }


    for (uint16_t i = 0; i < current_neighbor_count; ++i) {
        row_t existing_id = current_neighbor_ids[i];
        // Check if this existing neighbor is already among the new candidates
        bool already_candidate = false;
        for(const auto& cand : candidates) {
            if (cand.second == existing_id) {
                already_candidate = true;
                break;
            }
        }
        if (already_candidate) continue;

        const_data_ptr_t existing_compressed = current_compressed_base + i * edge_vector_size_bytes_;
        // Calculate distance from node to existing neighbor (approximate)
        float dist = CalculateApproxDistance(node_vector_float.data(), existing_compressed);
        candidates.push_back({dist, existing_id});
    }

    // Sort all candidates (existing + new) by distance
    std::sort(candidates.begin(), candidates.end());

    // Remove duplicates (keeping the one with the smallest distance)
    std::vector<std::pair<float, row_t>> unique_candidates;
    if (!candidates.empty()) {
        unique_candidates.push_back(candidates[0]);
        for (size_t i = 1; i < candidates.size(); ++i) {
            if (candidates[i].second != candidates[i-1].second) {
                unique_candidates.push_back(candidates[i]);
            }
            // If IDs are same, the sort already put the one with smaller distance first
        }
    }
    // candidates = std::move(unique_candidates); // Use the unique list - NO! Keep original `candidates` for alpha pruning


    // Apply alpha pruning and select final neighbors
    std::vector<row_t> final_neighbor_ids;
    std::vector<vector<uint8_t>> final_compressed_neighbors; // Store compressed vecs temporarily
    final_neighbor_ids.reserve(max_neighbors);
    final_compressed_neighbors.reserve(max_neighbors);

    vector<float> candidate_vector_float(dimensions_); // Temp storage for candidate float vector

    for (const auto& candidate : unique_candidates) { // Iterate through unique candidates sorted by distance
        if (final_neighbor_ids.size() >= max_neighbors) break; // Reached limit

        row_t candidate_id = candidate.second;
        if (candidate_id == node_rowid) continue; // Skip self

        IndexPointer candidate_ptr;
        if (!TryGetNodePointer(candidate_id, candidate_ptr)) continue; // Skip deleted/invalid

        // Get candidate vector (float version) - needed for pruning checks
        BufferHandle cand_handle;
        const_data_ptr_t cand_vec_ptr_raw;
        try {
            cand_handle = GetNodeBuffer(candidate_ptr); // Read-only is fine
            cand_vec_ptr_raw = LMDiskannNodeAccessors::GetNodeVectorPtr(cand_handle.Ptr(), node_layout_);
            // Convert candidate vector to float
            if (node_vector_type_ == LMDiskannVectorType::FLOAT32) { ConvertToFloat(reinterpret_cast<const float*>(cand_vec_ptr_raw), candidate_vector_float.data(), dimensions_); }
            else if (node_vector_type_ == LMDiskannVectorType::FLOAT16) { ConvertToFloat(reinterpret_cast<const float16_t*>(cand_vec_ptr_raw), candidate_vector_float.data(), dimensions_); }
            else if (node_vector_type_ == LMDiskannVectorType::INT8) { ConvertToFloat(reinterpret_cast<const int8_t*>(cand_vec_ptr_raw), candidate_vector_float.data(), dimensions_); }
            else { continue; } // Skip unsupported type
        } catch (...) { continue; /* Skip read/conversion errors */ }


        // Check alpha pruning against already selected final neighbors
        bool pruned = false;
        float dist_node_to_candidate = candidate.first; // Use precalculated distance from node to candidate

        for (size_t i = 0; i < final_neighbor_ids.size(); ++i) {
            row_t existing_final_id = final_neighbor_ids[i];
            // Need float vector of the existing final neighbor 'i' to calculate distance
            IndexPointer existing_final_ptr;
            if (!TryGetNodePointer(existing_final_id, existing_final_ptr)) { continue; } // Should not happen

            vector<float> existing_final_vector_float(dimensions_); // Temp storage
            try {
                auto existing_final_handle = GetNodeBuffer(existing_final_ptr);
                auto existing_final_block_data = existing_final_handle.Ptr();
                auto existing_final_vec_ptr_raw = LMDiskannNodeAccessors::GetNodeVectorPtr(existing_final_block_data, node_layout_);
                // Convert existing final neighbor vector to float
                 if (node_vector_type_ == LMDiskannVectorType::FLOAT32) { ConvertToFloat(reinterpret_cast<const float*>(existing_final_vec_ptr_raw), existing_final_vector_float.data(), dimensions_); }
                 else if (node_vector_type_ == LMDiskannVectorType::FLOAT16) { ConvertToFloat(reinterpret_cast<const float16_t*>(existing_final_vec_ptr_raw), existing_final_vector_float.data(), dimensions_); }
                 else if (node_vector_type_ == LMDiskannVectorType::INT8) { ConvertToFloat(reinterpret_cast<const int8_t*>(existing_final_vec_ptr_raw), existing_final_vector_float.data(), dimensions_); }
                 else { continue; }
            } catch (...) { continue; /* Skip errors */ }


            // Calculate distance between the existing final neighbor and the current candidate
            float dist_existing_final_to_candidate = CalculateDistance<float, float>(
                existing_final_vector_float.data(), candidate_vector_float.data()
            );

            // Pruning rule: if node->candidate is further than alpha * existing_final->candidate
            if (dist_node_to_candidate > alpha_ * dist_existing_final_to_candidate) {
                pruned = true;
                break;
            }
        }

        if (!pruned) {
            // Add to final list
            final_neighbor_ids.push_back(candidate_id);
            // Compress and store candidate's float vector
            final_compressed_neighbors.emplace_back(edge_vector_size_bytes_);
            if (!CompressVectorForEdge(candidate_vector_float.data(), final_compressed_neighbors.back().data(), dimensions_, resolved_edge_vector_type_)) { // Use distance module helper
                 throw InternalException("Failed to compress vector during pruning.");
            }
        }
    }

    // Write final neighbors back to the node block
    uint16_t final_count = static_cast<uint16_t>(final_neighbor_ids.size());
    // Ensure we don't exceed R
    if (final_count > max_neighbors) {
        // This shouldn't happen if the loop condition `final_neighbor_ids.size() >= max_neighbors` is correct
        final_count = max_neighbors;
        final_neighbor_ids.resize(final_count);
        final_compressed_neighbors.resize(final_count);
    }

    LMDiskannNodeAccessors::SetNeighborCount(node_data, final_count);
    row_t* dest_ids = LMDiskannNodeAccessors::GetNeighborIDsPtrMutable(node_data, node_layout_);
    data_ptr_t dest_compressed_base = LMDiskannNodeAccessors::GetCompressedNeighborPtrMutable(node_data, node_layout_, 0, edge_vector_size_bytes_);

    for (uint16_t i = 0; i < final_count; ++i) {
        dest_ids[i] = final_neighbor_ids[i];
        memcpy(dest_compressed_base + i * edge_vector_size_bytes_,
               final_compressed_neighbors[i].data(),
               edge_vector_size_bytes_);
    }
    // Zero out remaining slots if necessary (important for consistency)
    uint16_t max_possible_neighbors = r_; // Use the index's R value
    if (final_count < max_possible_neighbors) {
         // Zero out slots from final_count up to R
         uint16_t zero_start_idx = final_count;
         uint16_t zero_count = max_possible_neighbors - final_count;

         if (zero_count > 0) {
             memset(dest_ids + zero_start_idx, 0, zero_count * sizeof(row_t));
             memset(dest_compressed_base + zero_start_idx * edge_vector_size_bytes_, 0,
                    zero_count * edge_vector_size_bytes_);
         }
    }


    handle.SetModified(); // Mark the node block as modified
}


// Insertion Helper - coordinates search, pruning, and updates
void LMDiskannIndex::FindAndConnectNeighbors(row_t new_node_rowid, IndexPointer new_node_ptr, const float *new_node_vector_float) {
    // 1. Perform Search to find candidate neighbors
    row_t entry_point = GetEntryPoint();
    if (entry_point == NumericLimits<row_t>::Maximum()) { return; } // Should have been handled by caller

    Vector query_vec_handle(LogicalType::ARRAY(LogicalType::FLOAT, dimensions_));
    memcpy(FlatVector::GetData<float>(query_vec_handle), new_node_vector_float, dimensions_ * sizeof(float));
    query_vec_handle.Flatten(1);

    LMDiskannScanState search_state(query_vec_handle, l_insert_, l_insert_); // Use l_insert

    // Initialize search state with entry point
    IndexPointer entry_ptr;
    if (TryGetNodePointer(entry_point, entry_ptr)) { // Uses in-memory map
        try {
            auto handle = GetNodeBuffer(entry_ptr);
            auto block_data = handle.Ptr();
            auto node_vec_ptr = LMDiskannNodeAccessors::GetNodeVectorPtr(block_data, node_layout_);
            float approx_dist = CalculateApproxDistance(search_state.query_vector_ptr, node_vec_ptr);
            search_state.candidates.push({approx_dist, entry_point});
        } catch(...) { /* Handle error */ }
    } else { /* Handle missing entry point */ }

    PerformSearch(search_state, *this, false); // Find approximate neighbors

    // 2. Collect potential neighbors (visited nodes) with distances
    std::vector<std::pair<float, row_t>> potential_neighbors;
    potential_neighbors.reserve(search_state.visited.size());
    for(row_t visited_id : search_state.visited) {
         if (visited_id == new_node_rowid) continue; // Skip self
         IndexPointer visited_ptr;
         if (!TryGetNodePointer(visited_id, visited_ptr)) continue; // Skip deleted
         try {
             auto handle = GetNodeBuffer(visited_ptr);
             auto block_data = handle.Ptr();
             auto node_vec_ptr_raw = LMDiskannNodeAccessors::GetNodeVectorPtr(block_data, node_layout_);
             // Calculate exact distance from new node to visited node
             float dist;
             // Handle node type for distance calculation
             if (node_vector_type_ == LMDiskannVectorType::FLOAT32) {
                 dist = CalculateDistance<float, float>(new_node_vector_float, reinterpret_cast<const float*>(node_vec_ptr_raw));
             } else if (node_vector_type_ == LMDiskannVectorType::FLOAT16) {
                 dist = CalculateDistance<float, float16_t>(new_node_vector_float, reinterpret_cast<const float16_t*>(node_vec_ptr_raw));
             } else if (node_vector_type_ == LMDiskannVectorType::INT8) {
                 dist = CalculateDistance<float, int8_t>(new_node_vector_float, reinterpret_cast<const int8_t*>(node_vec_ptr_raw));
             } else { continue; } // Skip unsupported
             potential_neighbors.push_back({dist, visited_id});
         } catch (...) { /* Skip read/conversion errors */ }
    }
    // Add self with distance 0 to ensure it's considered during pruning (as per original paper)
    potential_neighbors.push_back({0.0f, new_node_rowid});


    // --- Update New Node's Neighbors using Robust Pruning ---
    RobustPrune(new_node_rowid, new_node_ptr, potential_neighbors, r_); // Prune and update new node's list


    // --- 4. Update Neighbors (Reciprocal Edges) ---
    // Get the final neighbors added to the new node
    auto new_node_handle_ro = GetNodeBuffer(new_node_ptr); // Read-only is fine now
    auto new_node_data_ro = new_node_handle_ro.Ptr();
    uint16_t final_new_neighbor_count = LMDiskannNodeAccessors::GetNeighborCount(new_node_data_ro);
    const row_t* final_new_neighbor_ids = LMDiskannNodeAccessors::GetNeighborIDsPtr(new_node_data_ro, node_layout_);

    // Prepare compressed version of the new node's vector
    vector<uint8_t> new_node_compressed_storage(edge_vector_size_bytes_);
    if (!CompressVectorForEdge(new_node_vector_float, new_node_compressed_storage.data(), dimensions_, resolved_edge_vector_type_)) { // Use distance module helper
        throw InternalException("Failed to compress new node vector for reciprocal edges.");
    }

    // Iterate through the final neighbors of the new node
    for(uint16_t i = 0; i < final_new_neighbor_count; ++i) {
         row_t neighbor_rowid = final_new_neighbor_ids[i];
         IndexPointer neighbor_ptr;
         if (!TryGetNodePointer(neighbor_rowid, neighbor_ptr)) continue; // Skip if neighbor deleted

         // Prepare candidate list for the neighbor (just the new node for now)
         std::vector<std::pair<float, row_t>> neighbor_candidates;
         float dist_neighbor_to_new; // Calculate this distance
         // Read neighbor's vector
         auto neighbor_handle_ro = GetNodeBuffer(neighbor_ptr);
         auto neighbor_data_ro = neighbor_handle_ro.Ptr();
         auto neighbor_vec_ptr_raw = LMDiskannNodeAccessors::GetNodeVectorPtr(neighbor_data_ro, node_layout_);

         // Calculate distance based on neighbor's node type
         if (node_vector_type_ == LMDiskannVectorType::FLOAT32) {
             dist_neighbor_to_new = CalculateDistance<float,float>(reinterpret_cast<const float*>(neighbor_vec_ptr_raw), new_node_vector_float);
         } else if (node_vector_type_ == LMDiskannVectorType::FLOAT16) {
             // Need templated CalculateDistance or ConvertToFloat first
             vector<float> neighbor_float(dimensions_);
             ConvertToFloat(reinterpret_cast<const float16_t*>(neighbor_vec_ptr_raw), neighbor_float.data(), dimensions_);
             dist_neighbor_to_new = CalculateDistance<float,float>(neighbor_float.data(), new_node_vector_float);
         } else if (node_vector_type_ == LMDiskannVectorType::INT8) {
             vector<float> neighbor_float(dimensions_);
             ConvertToFloat(reinterpret_cast<const int8_t*>(neighbor_vec_ptr_raw), neighbor_float.data(), dimensions_);
             dist_neighbor_to_new = CalculateDistance<float,float>(neighbor_float.data(), new_node_vector_float);
         } else { continue; } // Skip unsupported

         neighbor_candidates.push_back({dist_neighbor_to_new, new_node_rowid});

         // Apply robust pruning to the neighbor, considering the new node
         RobustPrune(neighbor_rowid, neighbor_ptr, neighbor_candidates, r_);
         // RobustPrune handles getting write lock, updating, and setting modified flag
    }
}


// --- Deletion Helper ---
void LMDiskannIndex::EnqueueDeletion(row_t deleted_row_id) {
    // Use storage module helper (placeholder implementation)
    duckdb::EnqueueDeletion(deleted_row_id, delete_queue_head_ptr_, db_, *allocator_);
    is_dirty_ = true; // Mark metadata dirty as queue head changed
}

// --- Entry Point Helpers ---
row_t LMDiskannIndex::GetEntryPoint() {
     // This implementation uses the in-memory map and cached rowid
     if (graph_entry_point_rowid_ != NumericLimits<row_t>::Maximum()) {
         IndexPointer ptr_check;
         if (TryGetNodePointer(graph_entry_point_rowid_, ptr_check)) { // Use in-memory map version
            return graph_entry_point_rowid_;
         } else {
            // Entry point was deleted, clear cache and find new one
            Printer::Warning("Cached entry point %lld deleted.", graph_entry_point_rowid_);
            graph_entry_point_ptr_.Clear();
            graph_entry_point_rowid_ = NumericLimits<row_t>::Maximum();
            is_dirty_ = true; // Need to persist cleared entry point
         }
     }
     // Fallback to random
     row_t random_id = GetRandomNodeID(); // Uses in-memory map version
     if (random_id != NumericLimits<row_t>::Maximum()) {
          IndexPointer random_ptr;
          if(TryGetNodePointer(random_id, random_ptr)) { // Use in-memory map version
             SetEntryPoint(random_id, random_ptr); // Cache it
             return random_id;
          }
     }
     return NumericLimits<row_t>::Maximum(); // No entry point found
}

void LMDiskannIndex::SetEntryPoint(row_t row_id, IndexPointer node_ptr) {
    graph_entry_point_rowid_ = row_id;
    graph_entry_point_ptr_ = node_ptr;
    is_dirty_ = true; // Need to persist the new entry point pointer
}

row_t LMDiskannIndex::GetRandomNodeID() {
    // Placeholder using in-memory map
    if (in_memory_rowid_map_.empty()) {
        return NumericLimits<row_t>::Maximum();
    }
    // Generate a random index into the map - requires converting map to vector or similar
    // This is slow, but okay for a placeholder.
    std::vector<row_t> keys;
    keys.reserve(in_memory_rowid_map_.size());
    for(auto const& [key, val] : in_memory_rowid_map_) {
        keys.push_back(key);
    }
    if (keys.empty()) { // Should match check above, but defensive
         return NumericLimits<row_t>::Maximum();
    }
    // Use DuckDB's random engine for potentially better seeding/quality
    auto& generator = RandomEngine::GetSystemRandom();
    std::uniform_int_distribution<idx_t> distribution(0, keys.size() - 1);
    idx_t random_idx = distribution(generator);
    return keys[random_idx];
}

// --- Storage interaction helpers (using in-memory map for now) ---
bool LMDiskannIndex::TryGetNodePointer(row_t row_id, IndexPointer &node_ptr) {
    auto it = in_memory_rowid_map_.find(row_id);
    if (it != in_memory_rowid_map_.end()) {
        node_ptr = it->second;
        return true;
    }
    node_ptr.Clear();
    return false;
}

IndexPointer LMDiskannIndex::AllocateNode(row_t row_id) {
    if (in_memory_rowid_map_.count(row_id)) {
        IndexPointer existing_ptr;
        if (TryGetNodePointer(row_id, existing_ptr)) {
             Printer::Warning("AllocateNode called for existing row_id %lld with valid pointer. Reusing block.", row_id);
             return existing_ptr; // Allow reusing block if entry exists? Or throw?
        }
        throw ConstraintException("Cannot allocate node for row_id %lld: already exists in mapping index (or map inconsistent).", row_id);
    }
    IndexPointer new_node_ptr = allocator_->New();
    if (!new_node_ptr.IsValid()) {
         throw InternalException("Failed to allocate new block in AllocateNode.");
    }
    in_memory_rowid_map_[row_id] = new_node_ptr;
    // is_dirty_ = true; // In-memory map doesn't directly make metadata dirty
    return new_node_ptr;
}

void LMDiskannIndex::DeleteNodeFromMapAndFreeBlock(row_t row_id) {
    auto it = in_memory_rowid_map_.find(row_id);
    if (it != in_memory_rowid_map_.end()) {
        IndexPointer node_ptr = it->second;
        allocator_->Free(node_ptr); // Free the associated block
        in_memory_rowid_map_.erase(it);
        // is_dirty_ = true; // In-memory map doesn't directly make metadata dirty
    } else {
        Printer::Warning("DeleteNodeFromMap: Node with row_id %lld not found in map.", row_id);
    }
}

BufferHandle LMDiskannIndex::GetNodeBuffer(IndexPointer node_ptr, bool write_lock) {
     if (!node_ptr.IsValid()) { throw IOException("Invalid node pointer provided to GetNodeBuffer."); }
    auto &buffer_manager = BufferManager::GetBufferManager(db_);
    return buffer_manager.Pin(allocator_->GetBlock(node_ptr));
}


} // namespace duckdb
