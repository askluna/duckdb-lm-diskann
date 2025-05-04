#include "lm_diskann.hpp" // Include the header we just defined

#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/execution/index/fixed_size_allocator.hpp"
#include "duckdb/storage/buffer_manager.hpp" // For BufferHandle
#include "duckdb/storage/metadata/metadata_reader.hpp"
#include "duckdb/storage/metadata/metadata_writer.hpp"
#include "duckdb/common/types/vector.hpp" // For Vector operations
#include "duckdb/common/types/data_chunk.hpp" // For DataChunk
#include "duckdb/main/client_context.hpp" // For ClientContext in scan
#include "duckdb/parser/parsed_data/create_index_info.hpp" // For CreateIndexInput
#include "duckdb/planner/expression/bound_columnref_expression.hpp" // For column type info
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp" // Potentially needed for table info
#include "duckdb/storage/single_file_block_manager.hpp" // For GetBlockId
#include "duckdb/storage/data_pointer.hpp" // For data_ptr_t // For data_ptr_t, Load/Store
#include "duckdb/execution/index/index_pointer.hpp" // For IndexPointer
#include "duckdb/common/limits.hpp" // For NumericLimits
#include "duckdb/common/vector_operations/vector_operations.hpp" // For vector copy/cast if needed

// Include for ART index (if/when implemented)
// #include "duckdb/storage/art/art.hpp"
// #include "duckdb/storage/art/art_key.hpp"

#include <cmath> // For sqrt, etc.
#include <limits> // For numeric_limits
#include <stdexcept> // For exceptions
#include <cstring> // For memcpy, memset
#include <algorithm> // For std::min, std::max, std::sort
#include <random> // For random entry point selection
#include <vector> // For std::vector

namespace duckdb {

// --- Constants for Parameter Parsing ---
const char *LMDISKANN_METRIC_OPTION = "METRIC";
const char *LMDISKANN_EDGE_TYPE_OPTION = "EDGE_TYPE";
const char *LMDISKANN_R_OPTION = "R";
const char *LMDISKANN_L_INSERT_OPTION = "L_INSERT";
const char *LMDISKANN_ALPHA_OPTION = "ALPHA";
const char *LMDISKANN_L_SEARCH_OPTION = "L_SEARCH";

// --- Default Parameter Values ---
// Match these with vectorIndexInt.h defaults where applicable
const LMDiskannMetricType LMDISKANN_DEFAULT_METRIC = LMDiskannMetricType::L2;
const LMDiskannEdgeType LMDISKANN_DEFAULT_EDGE_TYPE = LMDiskannEdgeType::SAME_AS_NODE;
const uint32_t LMDISKANN_DEFAULT_R = 64;
const uint32_t LMDISKANN_DEFAULT_L_INSERT = 128; // VECTOR_INSERT_L_DEFAULT
const float LMDISKANN_DEFAULT_ALPHA = 1.2f;      // VECTOR_PRUNING_ALPHA_DEFAULT
const uint32_t LMDISKANN_DEFAULT_L_SEARCH = 100; // VECTOR_SEARCH_L_DEFAULT
const uint8_t LMDISKANN_CURRENT_FORMAT_VERSION = 3; // VECTOR_FORMAT_DEFAULT

// --- Node Block Layout Constants (Offsets within a block) ---
constexpr idx_t OFFSET_NEIGHBOR_COUNT = 0; // uint16_t
constexpr idx_t NODE_VECTOR_ALIGNMENT = 8; // Align vectors to 8 bytes (adjust if needed)

// --- Delete Queue Block Layout ---
// Simple layout: [row_t deleted_id][IndexPointer next_block_ptr]
// IndexPointer itself is block_id (int64) + offset (uint32)
constexpr idx_t DELETE_QUEUE_ENTRY_SIZE = sizeof(row_t) + sizeof(block_id_t) + sizeof(uint32_t);


// --- Helper Function Implementations ---

idx_t LMDiskannIndex::GetVectorTypeSizeBytes(LMDiskannVectorType type) {
    switch (type) {
    case LMDiskannVectorType::FLOAT32:
        return sizeof(float);
    case LMDiskannVectorType::INT8:
        return sizeof(int8_t);
    case LMDiskannVectorType::FLOAT16:
        return sizeof(float16_t); // DuckDB's half-float type
    default:
        throw InternalException("Unsupported LMDiskannVectorType for size calculation");
    }
}

idx_t LMDiskannIndex::GetEdgeVectorTypeSizeBytes(LMDiskannEdgeType type, LMDiskannVectorType node_type) {
     LMDiskannVectorType resolved_type;
     if (type == LMDiskannEdgeType::SAME_AS_NODE) {
         resolved_type = node_type;
     } else {
        // Map EdgeType enum back to VectorType enum for size calculation
        switch(type) {
            case LMDiskannEdgeType::FLOAT32: resolved_type = LMDiskannVectorType::FLOAT32; break;
            case LMDiskannEdgeType::FLOAT16: resolved_type = LMDiskannVectorType::FLOAT16; break;
            case LMDiskannEdgeType::INT8:    resolved_type = LMDiskannVectorType::INT8; break;
            case LMDiskannEdgeType::FLOAT1BIT: return 1; // Special case: 1 bit per dimension (needs careful handling)
            default: throw InternalException("Unsupported LMDiskannEdgeType for size calculation");
        }
     }
     // Handle UNKNOWN case resulting from FLOAT1BIT mapping
     if (resolved_type == LMDiskannVectorType::UNKNOWN) {
        throw InternalException("Cannot calculate size for UNKNOWN vector type (likely FLOAT1BIT issue)");
     }
     return GetVectorTypeSizeBytes(resolved_type);
}

// Static parser function (can be called from constructor or elsewhere)
void LMDiskannIndex::ParseOptions(const case_insensitive_map_t<Value> &options,
                                   LMDiskannMetricType &metric_type,
                                   LMDiskannEdgeType &edge_type,
                                   uint32_t &r, uint32_t &l_insert,
                                   float &alpha, uint32_t &l_search) {
    // Apply defaults first
    metric_type = LMDISKANN_DEFAULT_METRIC;
    edge_type = LMDISKANN_DEFAULT_EDGE_TYPE;
    r = LMDISKANN_DEFAULT_R;
    l_insert = LMDISKANN_DEFAULT_L_INSERT;
    alpha = LMDISKANN_DEFAULT_ALPHA;
    l_search = LMDISKANN_DEFAULT_L_SEARCH;

    // Override with user-provided options
    auto it = options.find(LMDISKANN_METRIC_OPTION);
    if (it != options.end()) {
        string metric_str = StringUtil::Upper(it->second.ToString());
        if (metric_str == "L2") {
            metric_type = LMDiskannMetricType::L2;
        } else if (metric_str == "COSINE") {
            metric_type = LMDiskannMetricType::COSINE;
        } else if (metric_str == "IP") {
            metric_type = LMDiskannMetricType::IP;
        } else {
            throw BinderException("Unsupported METRIC type '%s' for LM_DISKANN index. Supported types: L2, COSINE, IP", metric_str);
        }
    }

    it = options.find(LMDISKANN_EDGE_TYPE_OPTION);
    if (it != options.end()) {
        string edge_type_str = StringUtil::Upper(it->second.ToString());
        if (edge_type_str == "FLOAT32") {
             edge_type = LMDiskannEdgeType::FLOAT32;
        } else if (edge_type_str == "FLOAT16") {
             edge_type = LMDiskannEdgeType::FLOAT16;
        } else if (edge_type_str == "INT8") {
             edge_type = LMDiskannEdgeType::INT8;
        } else if (edge_type_str == "FLOAT1BIT") {
             edge_type = LMDiskannEdgeType::FLOAT1BIT;
        } else {
             throw BinderException("Unsupported EDGE_TYPE '%s' for LM_DISKANN index. Supported types: FLOAT32, FLOAT16, INT8, FLOAT1BIT", edge_type_str);
        }
    }
    // Note: SAME_AS_NODE is the default, not explicitly parsed here.

    it = options.find(LMDISKANN_R_OPTION);
    if (it != options.end()) {
        r = it->second.GetValue<uint32_t>();
    }

    it = options.find(LMDISKANN_L_INSERT_OPTION);
    if (it != options.end()) {
        l_insert = it->second.GetValue<uint32_t>();
    }

    it = options.find(LMDISKANN_ALPHA_OPTION);
    if (it != options.end()) {
        alpha = it->second.GetValue<float>();
    }

    it = options.find(LMDISKANN_L_SEARCH_OPTION);
    if (it != options.end()) {
        l_search = it->second.GetValue<uint32_t>();
    }
}


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
    // Define MAX_VECTOR_SZ appropriately, e.g., #define MAX_VECTOR_SZ 65536
    constexpr idx_t MAX_VECTOR_SZ = 65536;
    if (dimensions_ > MAX_VECTOR_SZ) {
         throw BinderException("Cannot create LM_DISKANN index on ARRAY with dimensions > %d", MAX_VECTOR_SZ);
    }

    switch (vector_child_type.id()) {
    case LogicalTypeId::FLOAT:
        node_vector_type_ = LMDiskannVectorType::FLOAT32;
        break;
    case LogicalTypeId::TINYINT: // Assuming TINYINT maps to INT8
        node_vector_type_ = LMDiskannVectorType::INT8;
        break;
    case LogicalTypeId::FLOAT16: // Assuming DuckDB uses FLOAT16 type ID
        node_vector_type_ = LMDiskannVectorType::FLOAT16;
        break;
        // Add cases for SMALLINT -> INT16 etc. if needed
    default:
        throw BinderException("Unsupported vector type for LM_DISKANN index: %s. Supported types: FLOAT[], TINYINT[], FLOAT16[]", vector_child_type.ToString());
    }

    // --- 2. Parse User-Provided Options ---
    ParseOptions(options, metric_type_, edge_vector_type_param_, r_, l_insert_, alpha_, l_search_);

    // --- 3. Resolve Edge Vector Type ---
    if (edge_vector_type_param_ == LMDiskannEdgeType::SAME_AS_NODE) {
        resolved_edge_vector_type_ = node_vector_type_;
    } else {
        // Map EdgeType enum back to VectorType enum
        switch(edge_vector_type_param_) {
            case LMDiskannEdgeType::FLOAT32: resolved_edge_vector_type_ = LMDiskannVectorType::FLOAT32; break;
            case LMDiskannEdgeType::FLOAT16: resolved_edge_vector_type_ = LMDiskannVectorType::FLOAT16; break;
            case LMDiskannEdgeType::INT8:    resolved_edge_vector_type_ = LMDiskannVectorType::INT8; break;
            case LMDiskannEdgeType::FLOAT1BIT:
                 // Size calculation handles this, but validation needed
                 resolved_edge_vector_type_ = LMDiskannVectorType::UNKNOWN; // Special case handled by size calc
                 break;
            default: throw InternalException("Unexpected LMDiskannEdgeType parameter");
        }
    }

    // --- 4. Validate Parameters ---
    ValidateParameters();

    // --- 5. Calculate Sizes and Layout ---
    CalculateSizesAndLayout(); // Computes sizes and node_layout_

    // --- 6. Initialize Storage ---
    // The block size MUST be determined before initializing the allocator
    allocator_ = make_uniq<FixedSizeAllocator>(block_size_bytes_, table_io_manager_.GetIndexBlockManager());

    // --- 7. Load Existing Index or Initialize New One ---
    if (storage_info.IsValid()) {
        // Load metadata and allocator state from storage_info
        LoadFromStorage(storage_info);
    } else {
        // Initialize metadata for a new index
        InitializeNewIndex(estimated_cardinality);
    }

    // --- 8. Initialize RowID Mapping (Placeholder) ---
    // if (storage_info.IsValid() && rowid_map_root_ptr_.IsValid()) {
    //     LoadRowIDMap();
    // } else {
    //     InitializeRowIDMap();
    // }

    // --- 9. Load PQ Codebooks (Placeholder) ---
    // pq_ = make_uniq<ProductQuantizer>(... parameters ...);
    // pq_->Load(...); // Load from file or storage

    // --- Logging ---
    Printer::Print(StringUtil::Format("LM_DISKANN Index '%s': Metric=%d, Dim=%lld, R=%d, L_insert=%d, Alpha=%.2f, L_search=%d, BlockSize=%lld",
                                      name, (int)metric_type_, dimensions_, r_, l_insert_, alpha_, l_search_, block_size_bytes_));

}

LMDiskannIndex::~LMDiskannIndex() = default; // Default destructor is likely sufficient for now

void LMDiskannIndex::ValidateParameters() {
    if (r_ == 0) throw BinderException("LM_DISKANN parameter R must be > 0");
    if (l_insert_ == 0) throw BinderException("LM_DISKANN parameter L_INSERT must be > 0");
    if (alpha_ < 1.0f) throw BinderException("LM_DISKANN parameter ALPHA must be >= 1.0");
    if (l_search_ == 0) throw BinderException("LM_DISKANN parameter L_SEARCH must be > 0");

    if (edge_vector_type_param_ == LMDiskannEdgeType::FLOAT1BIT && metric_type_ != LMDiskannMetricType::COSINE) {
         throw BinderException("LM_DISKANN EDGE_TYPE 'FLOAT1BIT' is only supported with METRIC 'COSINE'");
    }
    // Add more validation as needed (e.g., max dimensions, max R based on block size)
}

// Calculates the layout offsets within a node block
NodeLayoutOffsets CalculateLayoutInternal(idx_t dimensions, idx_t r,
                                          idx_t node_vector_size_bytes,
                                          idx_t edge_vector_size_bytes) {
    NodeLayoutOffsets layout;

    // Offset 0: Neighbor count (uint16_t)
    layout.neighbor_count = OFFSET_NEIGHBOR_COUNT; // = 0
    idx_t current_offset = sizeof(uint16_t);
    // Add other fixed-size metadata here if needed (e.g., flags like 'deleted')
    // Example: Add a deleted flag (uint8_t)
    // layout.deleted_flag = current_offset;
    // current_offset += sizeof(uint8_t);

    // Align for node vector
    current_offset = AlignValue(current_offset, NODE_VECTOR_ALIGNMENT);
    layout.node_vector = current_offset;
    current_offset += node_vector_size_bytes;

    // Align for neighbor IDs (row_t is usually 64-bit, likely aligned)
    current_offset = AlignValue(current_offset, sizeof(row_t));
    layout.neighbor_ids = current_offset;
    current_offset += r * sizeof(row_t);

    // Align for compressed neighbors (depends on edge_vector_size_bytes alignment)
    // Assume minimal alignment needed for edge vectors unless complex types used
    idx_t edge_alignment = (edge_vector_size_bytes > 1) ? std::min((idx_t)8, NextPowerOfTwo(edge_vector_size_bytes)) : 1;
    current_offset = AlignValue(current_offset, edge_alignment);
    layout.compressed_neighbors = current_offset;
    current_offset += r * edge_vector_size_bytes;

    layout.total_size = current_offset; // Size *before* final block alignment
    return layout;
}


void LMDiskannIndex::CalculateSizesAndLayout() {
     node_vector_size_bytes_ = GetVectorTypeSizeBytes(node_vector_type_) * dimensions_;

     if (edge_vector_type_param_ == LMDiskannEdgeType::FLOAT1BIT) {
         // Special case: 1 bit per dimension, rounded up to the nearest byte
         edge_vector_size_bytes_ = (dimensions_ + 7) / 8;
     } else {
         edge_vector_size_bytes_ = GetEdgeVectorTypeSizeBytes(edge_vector_type_param_, node_vector_type_) * dimensions_;
     }

     // Calculate the required size and internal offsets based on the precise layout
     node_layout_ = CalculateLayoutInternal(dimensions_, r_, node_vector_size_bytes_, edge_vector_size_bytes_);
     block_size_bytes_ = node_layout_.total_size;


     // Ensure block size is reasonable and aligned (e.g., to 512 or 4096 bytes)
     // Align to at least sector size for better I/O
     block_size_bytes_ = AlignValue(block_size_bytes_, (idx_t)Storage::SECTOR_SIZE);

     // Define DISKANN_MAX_BLOCK_SZ appropriately, e.g., #define DISKANN_MAX_BLOCK_SZ (128 * 1024 * 1024)
     constexpr idx_t DISKANN_MAX_BLOCK_SZ = 128 * 1024 * 1024;
     if (block_size_bytes_ > DISKANN_MAX_BLOCK_SZ) {
          throw BinderException("Calculated LM_DISKANN block size (%lld bytes) exceeds maximum allowed (%lld bytes). Reduce R or vector dimensions.",
                                block_size_bytes_, DISKANN_MAX_BLOCK_SZ);
     }
     // Ensure minimum block size if necessary (e.g., sector size)
     block_size_bytes_ = std::max(block_size_bytes_, (idx_t)Storage::SECTOR_SIZE);
}


void LMDiskannIndex::InitializeNewIndex(idx_t estimated_cardinality) {
     // Allocate the persistent metadata block
     metadata_ptr_ = allocator_->New();
     is_dirty_ = true; // Mark dirty to ensure metadata gets persisted

     // Initialize RowID map (e.g., create a new ART index)
     // rowid_map_ = make_uniq<ART>(db_, ART::ARTType::ROW_ID_ART, allocator_); // Example
     // rowid_map_root_ptr_ = rowid_map_->GetRoot(); // Get initial root pointer

     // Initialize graph entry point and delete queue head
     graph_entry_point_ptr_.Clear();
     graph_entry_point_rowid_ = NumericLimits<row_t>::Maximum(); // Invalid rowid
     delete_queue_head_ptr_.Clear();

     // Persist initial parameters and pointers to the metadata block
     PersistMetadata();
}

void LMDiskannIndex::LoadFromStorage(const IndexStorageInfo &storage_info) {
     metadata_ptr_.Set(storage_info.root_block, storage_info.root_offset); // Use combined root info
     if (!metadata_ptr_.IsValid()) {
          throw IOException("Cannot load LM_DISKANN index: metadata pointer is invalid.");
     }

     // Initialize the allocator with persisted info
     if (storage_info.allocator_infos.empty()) {
          throw IOException("Cannot load LM_DISKANN index: missing allocator info.");
     }
     allocator_->Init(storage_info.allocator_infos[0]);

     // Read parameters from the metadata block
     auto &buffer_manager = BufferManager::GetBufferManager(db_);
     auto handle = buffer_manager.Pin(allocator_->GetMetaBlock(metadata_ptr_.GetBlockId()));
     MetadataReader reader(handle.GetFileBuffer(), metadata_ptr_.GetOffset());


     // --- Deserialize Parameters ---
     reader.Read<uint8_t>(format_version_);
     if (format_version_ != LMDISKANN_CURRENT_FORMAT_VERSION) {
          // Handle version mismatch - potentially upgrade or throw error
          throw IOException("LM_DISKANN index format version mismatch. Found %d, expected %d.",
                            format_version_, LMDISKANN_CURRENT_FORMAT_VERSION);
     }
     reader.Read<LMDiskannMetricType>(metric_type_);
     reader.Read<LMDiskannVectorType>(node_vector_type_);
     reader.Read<LMDiskannEdgeType>(edge_vector_type_param_); // Read the user's choice
     reader.Read<idx_t>(dimensions_);
     reader.Read<uint32_t>(r_);
     reader.Read<uint32_t>(l_insert_);
     reader.Read<float>(alpha_);
     reader.Read<uint32_t>(l_search_);
     reader.Read<idx_t>(block_size_bytes_);
     // Deserialize graph_entry_point_ptr_ and delete_queue_head_ptr_
     reader.Read<IndexPointer>(graph_entry_point_ptr_);
     reader.Read<IndexPointer>(delete_queue_head_ptr_);
     // Deserialize rowid_map_root_ptr_ if stored here
     // reader.Read<IndexPointer>(rowid_map_root_ptr_);

     // --- Post-Load Processing ---

     // Re-calculate derived/validated parameters based on loaded values
     if (edge_vector_type_param_ == LMDiskannEdgeType::SAME_AS_NODE) {
         resolved_edge_vector_type_ = node_vector_type_;
     } else {
         // Map EdgeType enum back to VectorType enum
         switch(edge_vector_type_param_) {
             case LMDiskannEdgeType::FLOAT32: resolved_edge_vector_type_ = LMDiskannVectorType::FLOAT32; break;
             case LMDiskannEdgeType::FLOAT16: resolved_edge_vector_type_ = LMDiskannVectorType::FLOAT16; break;
             case LMDiskannEdgeType::INT8:    resolved_edge_vector_type_ = LMDiskannVectorType::INT8; break;
             case LMDiskannEdgeType::FLOAT1BIT: resolved_edge_vector_type_ = LMDiskannVectorType::UNKNOWN; break;
             default: throw InternalException("Unexpected loaded LMDiskannEdgeType parameter");
         }
     }
     // Recalculate sizes and layout based on loaded parameters to ensure consistency
     CalculateSizesAndLayout(); // This recalculates node_layout_ and checks block_size_bytes_

     // Check persisted block size against recalculated one
     idx_t expected_block_size = block_size_bytes_; // Already calculated in CalculateSizesAndLayout
     idx_t persisted_block_size;
     // Re-read block size from metadata to compare
     {
        // Need to create a *new* reader as the previous one's position is advanced
        MetadataReader reader_check(handle.GetFileBuffer(), metadata_ptr_.GetOffset());
        // Seek or re-read up to block_size_bytes_ field
        reader_check.Skip(sizeof(uint8_t) + sizeof(LMDiskannMetricType) + sizeof(LMDiskannVectorType) + sizeof(LMDiskannEdgeType) + sizeof(idx_t) + sizeof(uint32_t) * 2 + sizeof(float) + sizeof(uint32_t));
        reader_check.Read<idx_t>(persisted_block_size);
     }


     if (persisted_block_size != expected_block_size) {
           throw IOException("LM_DISKANN index load error: persisted block size (%lld) does not match calculated block size (%lld) for loaded parameters. Index may be corrupt or from an incompatible version.",
                            persisted_block_size, expected_block_size);
     }

     ValidateParameters(); // Re-validate loaded parameters

     // Load the RowID map using the deserialized root pointer
     // if (rowid_map_root_ptr_.IsValid()) {
     //    rowid_map_ = make_uniq<ART>(db_, ART::ARTType::ROW_ID_ART, allocator_, rowid_map_root_ptr_);
     // } else {
     //    // Map was likely empty or not persisted correctly
     //    InitializeRowIDMap(); // Reinitialize empty map
     // }

     // Get entry point rowid if pointer is valid
     if (graph_entry_point_ptr_.IsValid()) {
         // FIXME: Need inverse mapping from IndexPointer to row_id
         // This might involve reading the block pointed to by graph_entry_point_ptr_
         // if the row_id isn't stored elsewhere.
         // graph_entry_point_rowid_ = GetRowIdFromPointer(graph_entry_point_ptr_);
         graph_entry_point_rowid_ = -2; // Placeholder indicating valid pointer but unknown rowid
         Printer::Warning("Loading entry point pointer, but rowid lookup not implemented.");
     } else {
         graph_entry_point_rowid_ = NumericLimits<row_t>::Maximum(); // Invalid
     }


     is_dirty_ = false; // Loaded state is not dirty initially
}

void LMDiskannIndex::PersistMetadata() {
     if (!metadata_ptr_.IsValid()) {
          throw InternalException("Cannot persist LM_DISKANN metadata: metadata pointer is invalid.");
     }
     // Pin the metadata block (make sure it's writable)
     auto &buffer_manager = BufferManager::GetBufferManager(db_);
     auto handle = buffer_manager.Pin(allocator_->GetMetaBlock(metadata_ptr_.GetBlockId()));
     MetadataWriter writer(handle.GetFileBuffer(), metadata_ptr_.GetOffset());

     // --- Serialize Parameters ---
     writer.Write<uint8_t>(format_version_);
     writer.Write<LMDiskannMetricType>(metric_type_);
     writer.Write<LMDiskannVectorType>(node_vector_type_);
     writer.Write<LMDiskannEdgeType>(edge_vector_type_param_); // Persist the user's choice
     writer.Write<idx_t>(dimensions_);
     writer.Write<uint32_t>(r_);
     writer.Write<uint32_t>(l_insert_);
     writer.Write<float>(alpha_);
     writer.Write<uint32_t>(l_search_);
     writer.Write<idx_t>(block_size_bytes_);
     // Serialize graph_entry_point_ptr_ and delete_queue_head_ptr_
     writer.Write<IndexPointer>(graph_entry_point_ptr_);
     writer.Write<IndexPointer>(delete_queue_head_ptr_);
     // Serialize rowid_map_root_ptr_ (Get the root pointer from the ART index)
     // writer.Write<IndexPointer>(rowid_map_ ? rowid_map_->GetRoot() : IndexPointer());


     // Mark the block as modified
     handle.SetModified();
     is_dirty_ = false; // Metadata is now persisted
}


IndexStorageInfo LMDiskannIndex::GetStorageInfo(const bool get_buffers) {
     // Persist any dirty metadata (including potentially updated RowID map root, entry point, delete queue)
     // if (rowid_map_) {
     //    rowid_map_root_ptr_ = rowid_map_->GetRoot();
     // }
     if (is_dirty_) { // is_dirty_ might be set by map, entry point, or queue updates
          PersistMetadata();
     }

     IndexStorageInfo info;
     info.name = name;
     // Store metadata block ID and offset
     info.root_block = metadata_ptr_.GetBlockId();
     info.root_offset = metadata_ptr_.GetOffset();


     // Get allocator state (includes buffer IDs for persistence)
     info.allocator_infos.push_back(allocator_->GetInfo());

     // Handle buffer serialization for WAL (currently not supported for custom indexes)
     if (get_buffers) {
          // info.buffers.push_back(allocator_->InitSerializationToWAL()); // Likely empty/no-op
     }

     return info;
}


// --- Storage Helper Methods ---

// Tries to find the IndexPointer for a given row_id using the map
bool LMDiskannIndex::TryGetNodePointer(row_t row_id, IndexPointer &node_ptr) {
    // FIXME: Implement RowID map lookup (e.g., using ART)
    // This is a critical piece. Without it, we cannot find node blocks.
    // Conceptual implementation:
    // if (!rowid_map_) {
    //     // Map not loaded or initialized, maybe log error or return false
    //     return false;
    // }
    // ARTKey key(&row_id, sizeof(row_t)); // Create key from row_id
    // IndexPointer pointer_val;
    // if (rowid_map_->Lookup(key, pointer_val)) { // Perform lookup
    //      node_ptr = pointer_val;
    //      return true; // Found
    // }
    // return false; // Not found
    // Printer::Warning("LMDiskannIndex::TryGetNodePointer is not implemented (using placeholder).");
    // Placeholder: Pretend node not found until implemented
    node_ptr.Clear(); // Ensure node_ptr is invalid if lookup fails
    return false;
}

// Allocates a new block for a node and updates the RowID map
IndexPointer LMDiskannIndex::AllocateNode(row_t row_id) {
    // FIXME: Implement RowID map insertion
    if (!allocator_) {
        throw InternalException("Allocator not initialized in AllocateNode");
    }
    // if (!rowid_map_) {
    //     throw InternalException("RowID map is not initialized in AllocateNode");
    // }

    // // Check if row_id already exists in map
    // IndexPointer existing_ptr;
    // if (TryGetNodePointer(row_id, existing_ptr)) {
    //     // Handle collision: Maybe overwrite? Delete old node first? Throw error?
    //     // For now, let's throw an error.
    //     throw ConstraintException("Cannot allocate node for row_id %lld: already exists in mapping index.", row_id);
    // }

    IndexPointer new_node_ptr = allocator_->New(); // Allocate block

    // // Insert into the map
    // ARTKey key(&row_id, sizeof(row_t));
    // bool success = rowid_map_->Insert(key, new_node_ptr);
    // if (!success) {
    //      // This shouldn't happen if the check above passed, but handle defensively
    //      allocator_->Free(new_node_ptr); // Free the newly allocated block on failure
    //      throw InternalException("Failed to insert row_id %lld into mapping index after check.", row_id);
    // }
    // is_dirty_ = true; // Mark index dirty as map has changed (needs metadata persistence)
    // Printer::Warning("LMDiskannIndex::AllocateNode is not implemented (using placeholder).");
    // Placeholder: Return the allocated pointer, but map isn't updated
    if (!new_node_ptr.IsValid()) {
         throw InternalException("Failed to allocate new block in AllocateNode.");
    }
    is_dirty_ = true; // Mark dirty because map *should* have changed
    return new_node_ptr;
}

// Deletes a node from the RowID map and potentially frees the block
// This is the immediate part of deletion. Neighbor updates are queued.
void LMDiskannIndex::DeleteNodeFromMapAndFreeBlock(row_t row_id) {
    // FIXME: Implement RowID map deletion
    // IndexPointer node_ptr;
    // if (TryGetNodePointer(row_id, node_ptr)) { // Find the pointer first
    //     ARTKey key(&row_id, sizeof(row_t));
    //     bool deleted = rowid_map_->Delete(key); // Remove from map
    //     if (deleted) {
    //         allocator_->Free(node_ptr); // Free the associated block
    //         is_dirty_ = true; // Mark index dirty
    //     } else {
    //         // Should not happen if TryGetNodePointer succeeded
    //         Printer::Warning("LMDiskannIndex::DeleteNodeFromMap: Failed to delete key %lld from map after finding it.", row_id);
    //     }
    // } else {
    //     // Node not found, maybe log a warning or ignore?
    //     Printer::Warning("LMDiskannIndex::DeleteNodeFromMap: Node with row_id %lld not found in map.", row_id);
    // }
    Printer::Warning("LMDiskannIndex::DeleteNodeFromMapAndFreeBlock is not implemented (using placeholder).");
    is_dirty_ = true; // Mark dirty because map *should* have changed
}


// Gets a pinned buffer handle for a specific node's block using its IndexPointer
BufferHandle LMDiskannIndex::GetNodeBuffer(IndexPointer node_ptr, bool write_lock) {
    if (!allocator_) {
        throw InternalException("LMDiskannIndex allocator is not initialized.");
    }
     if (!node_ptr.IsValid()) {
         throw IOException("Invalid node pointer provided to GetNodeBuffer.");
     }

    auto &buffer_manager = BufferManager::GetBufferManager(db_);
    // Use GetBlockId() and GetOffset() from the IndexPointer
    // Need to use GetBlock which handles the mapping from IndexPointer to block_id_t
    return buffer_manager.Pin(allocator_->GetBlock(node_ptr));
}

// --- Node Data Accessor Implementation ---
namespace LMDiskannNodeAccessors {

    // --- Getters (const version) ---
    uint16_t GetNeighborCount(const_data_ptr_t block_ptr) {
        // Reads the neighbor count (uint16_t) from the start of the block.
        return Load<uint16_t>(block_ptr + OFFSET_NEIGHBOR_COUNT);
    }

    const_data_ptr_t GetNodeVectorPtr(const_data_ptr_t block_ptr, const NodeLayoutOffsets& layout) {
        // Returns a pointer to the start of the node's full vector data.
        return block_ptr + layout.node_vector;
    }

    // Get pointer to the array of neighbor row_t IDs
    const row_t* GetNeighborIDsPtr(const_data_ptr_t block_ptr, const NodeLayoutOffsets& layout) {
        // Returns a pointer to the start of the neighbor row_t ID array.
        return reinterpret_cast<const row_t*>(block_ptr + layout.neighbor_ids);
    }

    // Get pointer to the i-th compressed neighbor vector data
    const_data_ptr_t GetCompressedNeighborPtr(const_data_ptr_t block_ptr, const NodeLayoutOffsets& layout, uint32_t neighbor_idx, idx_t edge_vector_size_bytes) {
        // Returns a pointer to the start of the specified neighbor's compressed vector data.
        // Note: No bounds check here for performance; caller must ensure neighbor_idx is valid.
        return block_ptr + layout.compressed_neighbors + (neighbor_idx * edge_vector_size_bytes);
    }

    // --- Setters (non-const version) ---
    void SetNeighborCount(data_ptr_t block_ptr, uint16_t count) {
        // Writes the neighbor count (uint16_t) to the start of the block.
        Store<uint16_t>(count, block_ptr + OFFSET_NEIGHBOR_COUNT);
    }

    // Get mutable pointer to the node vector data area
    data_ptr_t GetNodeVectorPtrMutable(data_ptr_t block_ptr, const NodeLayoutOffsets& layout) {
        // Returns a writable pointer to the start of the node's full vector data.
        return block_ptr + layout.node_vector;
    }

    // Get mutable pointer to the array of neighbor row_t IDs
    row_t* GetNeighborIDsPtrMutable(data_ptr_t block_ptr, const NodeLayoutOffsets& layout) {
        // Returns a writable pointer to the start of the neighbor row_t ID array.
        return reinterpret_cast<row_t*>(block_ptr + layout.neighbor_ids);
    }

    // Get mutable pointer to the i-th compressed neighbor vector data area
    data_ptr_t GetCompressedNeighborPtrMutable(data_ptr_t block_ptr, const NodeLayoutOffsets& layout, uint32_t neighbor_idx, idx_t edge_vector_size_bytes) {
        // Returns a writable pointer to the start of the specified neighbor's compressed vector data.
        // Note: No bounds check here for performance; caller must ensure neighbor_idx is valid.
        return block_ptr + layout.compressed_neighbors + (neighbor_idx * edge_vector_size_bytes);
    }

    // --- Initialization Helper ---
    void InitializeNodeBlock(data_ptr_t block_ptr, idx_t block_size) {
        // Zero out the block initially.
        memset(block_ptr, 0, block_size);
        // Set neighbor count to 0 explicitly.
        SetNeighborCount(block_ptr, 0);
        // Any other default initialization (e.g., flags) goes here.
    }

} // namespace LMDiskannNodeAccessors


// --- Distance Function Implementations ---

// Helper to potentially convert vector data to float for calculation
template <typename T>
static void ConvertToFloat(const T *src, float *dst, idx_t count) {
    if constexpr (std::is_same_v<T, float>) {
        memcpy(dst, src, count * sizeof(float));
    } else if constexpr (std::is_same_v<T, float16_t>) {
        for (idx_t i = 0; i < count; ++i) {
            dst[i] = float16_t::ToFloat(src[i]);
        }
    } else if constexpr (std::is_same_v<T, int8_t>) {
        // FIXME: This simple cast is likely inaccurate.
        // Proper INT8 distance needs scale/offset or specialized kernels.
        // Using this for now as a placeholder.
        for (idx_t i = 0; i < count; ++i) {
            // Example: Scale to [-1, 1] range approximately
            dst[i] = static_cast<float>(src[i]) / 128.0f;
        }
    } else {
        throw NotImplementedException("Unsupported type for ConvertToFloat");
    }
}

// Generic distance calculation wrapper
template <typename T_A, typename T_B>
float LMDiskannIndex::CalculateDistance(const T_A *vec_a, const T_B *vec_b) {
    // Convert both vectors to float for using DuckDB's VectorDistance
    // This is inefficient for non-float types but provides a starting point.
    // Optimization: Implement specialized kernels later.
    vector<float> vec_a_float(dimensions_);
    vector<float> vec_b_float(dimensions_);

    ConvertToFloat(vec_a, vec_a_float.data(), dimensions_);
    ConvertToFloat(vec_b, vec_b_float.data(), dimensions_);

    const float *a_ptr = vec_a_float.data();
    const float *b_ptr = vec_b_float.data();

    switch (metric_type_) {
        case LMDiskannMetricType::L2:
             // Returns squared L2 distance
             return VectorDistance::Exec<float, float, float>(a_ptr, b_ptr, dimensions_, VectorDistanceType::L2);
        case LMDiskannMetricType::COSINE:
             // Returns 1 - cosine_similarity. Lower is better.
             return VectorDistance::Exec<float, float, float>(a_ptr, b_ptr, dimensions_, VectorDistanceType::COSINE);
        case LMDiskannMetricType::IP:
             // Returns -inner_product. Lower is better.
             return -VectorDistance::Exec<float, float, float>(a_ptr, b_ptr, dimensions_, VectorDistanceType::IP);
        default:
             throw InternalException("Unknown metric type in CalculateDistance");
    }
}

// Explicit template instantiations for types used
template float LMDiskannIndex::CalculateDistance<float, float>(const float*, const float*);
template float LMDiskannIndex::CalculateDistance<float, float16_t>(const float*, const float16_t*);
template float LMDiskannIndex::CalculateDistance<float, int8_t>(const float*, const int8_t*);
// Add more if needed, e.g., <float16_t, float16_t>


// Approximate distance between full query (float) and compressed neighbor
float LMDiskannIndex::CalculateApproxDistance(const float *query_ptr,
                                              const_data_ptr_t compressed_neighbor_ptr) {

    // Dispatch based on the *resolved* edge type stored in the index
    switch(resolved_edge_vector_type_) {
        case LMDiskannVectorType::FLOAT32:
            return CalculateDistance<float, float>(query_ptr, reinterpret_cast<const float*>(compressed_neighbor_ptr));
        case LMDiskannVectorType::FLOAT16:
            return CalculateDistance<float, float16_t>(query_ptr, reinterpret_cast<const float16_t*>(compressed_neighbor_ptr));
        case LMDiskannVectorType::INT8:
            // Warning: Accuracy depends heavily on how INT8 was created and how distance is calculated.
            return CalculateDistance<float, int8_t>(query_ptr, reinterpret_cast<const int8_t*>(compressed_neighbor_ptr));
        case LMDiskannVectorType::UNKNOWN: // Represents FLOAT1BIT
            if (metric_type_ == LMDiskannMetricType::COSINE) {
                 // FIXME: Implement Hamming distance / popcount for FLOAT1BIT Cosine approximation
                 // Calculate Hamming distance between query (needs binarization) and compressed_neighbor_ptr
                 // Convert Hamming distance to approximate Cosine distance.
                 throw NotImplementedException("Approximate distance for FLOAT1BIT not implemented.");
            } else {
                 throw InternalException("FLOAT1BIT edge type used with non-COSINE metric.");
            }
            break;
        default:
             throw InternalException("Unsupported resolved edge vector type in CalculateApproxDistance.");
    }
}


// --- Core Algorithm Structure Placeholders ---

ErrorData LMDiskannIndex::Append(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) {
    // Use Insert logic for now, can be optimized later
    if (entries.size() == 0) {
        return ErrorData();
    }
    // Ensure row identifiers are flat
    row_identifiers.Flatten(entries.size());

    DataChunk input_chunk;
    input_chunk.InitializeEmpty({entries.data[0].GetType()});
    Vector row_id_vector(LogicalType::ROW_TYPE);

    for(idx_t i = 0; i < entries.size(); ++i) {
        // Slice the input vector
        input_chunk.Reset();
        input_chunk.data[0].Slice(entries.data[0], i, i + 1);
        input_chunk.SetCardinality(1);

        // Slice the row identifier
        row_id_vector.Slice(row_identifiers, i, i + 1);
        row_id_vector.Flatten(1); // Ensure flat vector

        // Call Insert for the single row
        auto err = Insert(lock, input_chunk, row_id_vector);
        if (err.HasError()) {
            // Attempt to rollback or handle partial failure? Difficult without transactions.
            return err; // Propagate the first error encountered
        }
    }
    return ErrorData(); // Success
}

void LMDiskannIndex::CommitDrop(IndexLock &index_lock) {
    // Reset the allocator to free all blocks
    if (allocator_) {
        allocator_->Reset();
    }
    metadata_ptr_.Clear();
    // Also need to drop the RowID map if it exists and is managed separately
    // if (rowid_map_) {
    //    rowid_map_->CommitDrop(); // Assuming ART has a similar method
    // }
    // Clear delete queue? Reset doesn't explicitly clear blocks, just metadata.
    // Need to ensure blocks are actually marked free in BlockManager.
    delete_queue_head_ptr_.Clear(); // Reset queue head
}

void LMDiskannIndex::Delete(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) {
    auto row_ids_data = FlatVector::GetData<row_t>(row_identifiers);
    bool changes_made = false;
    for (idx_t i = 0; i < entries.size(); ++i) {
        row_t row_id = row_ids_data[i];
        try {
             // 1. Remove from RowID map and free block (immediate)
             DeleteNodeFromMapAndFreeBlock(row_id); // Placeholder

             // 2. Add to persistent delete queue for deferred neighbor updates
             EnqueueDeletion(row_id); // Placeholder

             // 3. Handle entry point deletion
             if (row_id == graph_entry_point_rowid_) {
                  graph_entry_point_ptr_.Clear();
                  graph_entry_point_rowid_ = NumericLimits<row_t>::Maximum();
                  // Need to find a new entry point later if needed, or on next insert/scan
                  is_dirty_ = true; // Mark dirty as entry point changed
             }
             changes_made = true;

        } catch (NotImplementedException &e) {
             throw; // Re-throw if map deletion isn't implemented
        } catch (std::exception &e) {
             // Log error or handle cases where node doesn't exist?
             Printer::Warning("Failed to delete node for row_id %lld: %s", row_id, e.what());
        }
    }
    // is_dirty_ is set by DeleteNodeFromMapAndFreeBlock and EnqueueDeletion
}

ErrorData LMDiskannIndex::Insert(IndexLock &lock, DataChunk &data, Vector &row_ids) {
    if (data.size() == 0) {
        return ErrorData();
    }
    D_ASSERT(data.size() == 1); // Process one vector at a time for now
    D_ASSERT(data.ColumnCount() == 1);
    D_ASSERT(row_ids.GetVectorType() == VectorType::FLAT_VECTOR);

    auto &input_vector_handle = data.data[0];
    input_vector_handle.Flatten(1); // Ensure flat for direct access
    auto row_id = FlatVector::GetData<row_t>(row_ids)[0];

    // --- Get Input Vector Pointer (Handle Different Types) ---
    const_data_ptr_t input_vector_raw_ptr = FlatVector::GetData(input_vector_handle);
    vector<float> input_vector_float_storage; // Storage if conversion needed
    const float* input_vector_float_ptr; // Pointer to use for calculations

    if (node_vector_type_ == LMDiskannVectorType::FLOAT32) {
        input_vector_float_ptr = reinterpret_cast<const float*>(input_vector_raw_ptr);
    } else {
        // Convert input vector to float for neighbor finding logic
        input_vector_float_storage.resize(dimensions_);
        if (node_vector_type_ == LMDiskannVectorType::FLOAT16) {
            ConvertToFloat(reinterpret_cast<const float16_t*>(input_vector_raw_ptr), input_vector_float_storage.data(), dimensions_);
        } else if (node_vector_type_ == LMDiskannVectorType::INT8) {
            ConvertToFloat(reinterpret_cast<const int8_t*>(input_vector_raw_ptr), input_vector_float_storage.data(), dimensions_);
        } else {
            return ErrorData("Unsupported node vector type for insertion.");
        }
        input_vector_float_ptr = input_vector_float_storage.data();
    }
    // --- End Input Vector Handling ---


    // 1. Allocate block for the new node
    IndexPointer new_node_ptr;
    try {
        new_node_ptr = AllocateNode(row_id); // Allocates block and updates map (placeholder)
    } catch (std::exception &e) {
        return ErrorData(e);
    }

    // 2. Pin the new node's buffer (writable)
    auto new_node_handle = GetNodeBuffer(new_node_ptr, true);
    auto new_node_data = new_node_handle.Ptr();

    // 3. Initialize the block
    LMDiskannNodeAccessors::InitializeNodeBlock(new_node_data, block_size_bytes_);

    // 4. Write the node's full vector data (use original raw pointer)
    memcpy(LMDiskannNodeAccessors::GetNodeVectorPtrMutable(new_node_data, node_layout_),
           input_vector_raw_ptr, node_vector_size_bytes_);

    // 5. Find neighbors and connect
    row_t entry_point_row_id = GetEntryPoint(); // Get a valid entry point

    if (entry_point_row_id == NumericLimits<row_t>::Maximum()) {
        // This is the first node being inserted
        LMDiskannNodeAccessors::SetNeighborCount(new_node_data, 0);
        // Set this node as the entry point
        SetEntryPoint(row_id, new_node_ptr);
    } else {
        // Find neighbors and connect using the float version of the input vector
        try {
            FindAndConnectNeighbors(row_id, new_node_ptr, input_vector_float_ptr);
        } catch (std::exception &e) {
             // Clean up allocated node if connection fails?
             // DeleteNodeFromMapAndFreeBlock(row_id); // Risky if map isn't transactional
             return ErrorData(StringUtil::Format("Failed to connect neighbors for node %lld: %s", row_id, e.what()));
        }
    }

    // 6. Mark the new node's buffer as modified
    new_node_handle.SetModified();

    // is_dirty_ should be set by AllocateNode and SetEntryPoint
    return ErrorData(); // Success
}

// Helper to compress a vector based on the resolved edge type
// Assumes input is float, output is written to dest_ptr
// Returns true on success, false on failure (e.g., unsupported type)
static bool CompressVectorForEdge(const float* input_float_ptr, data_ptr_t dest_ptr,
                                  idx_t dimensions, LMDiskannVectorType resolved_edge_type) {
     switch(resolved_edge_type) {
         case LMDiskannVectorType::FLOAT32:
             memcpy(dest_ptr, input_float_ptr, dimensions * sizeof(float));
             return true;
         case LMDiskannVectorType::FLOAT16:
             {
                 float16_t* dest_f16 = reinterpret_cast<float16_t*>(dest_ptr);
                 for(idx_t i=0; i<dimensions; ++i) {
                     dest_f16[i] = float16_t::FromFloat(input_float_ptr[i]);
                 }
                 return true;
             }
         case LMDiskannVectorType::INT8:
             {
                 // FIXME: Simple cast/scaling. Needs proper quantization scheme.
                 int8_t* dest_i8 = reinterpret_cast<int8_t*>(dest_ptr);
                 for(idx_t i=0; i<dimensions; ++i) {
                     // Example: Scale to [-127, 127] assuming input is roughly [-1, 1]
                     float clamped = std::max(-1.0f, std::min(1.0f, input_float_ptr[i]));
                     dest_i8[i] = static_cast<int8_t>(clamped * 127.0f);
                 }
                 return true;
             }
         case LMDiskannVectorType::UNKNOWN: // FLOAT1BIT
             // FIXME: Implement binarization
             throw NotImplementedException("Compression to FLOAT1BIT not implemented.");
             return false; // Unreachable
         default:
             return false; // Unsupported type
     }
}


void LMDiskannIndex::FindAndConnectNeighbors(row_t new_node_rowid, IndexPointer new_node_ptr, const float *new_node_vector_float) {
    // 1. Perform a search to find candidate neighbors
    // Create a temporary query vector handle (needed for ScanState)
    Vector query_vec_handle(LogicalType::ARRAY(LogicalType::FLOAT, dimensions_));
    auto query_data_ptr = FlatVector::GetData<float>(query_vec_handle);
    memcpy(query_data_ptr, new_node_vector_float, dimensions_ * sizeof(float));
    query_vec_handle.Flatten(1);

    LMDiskannScanState search_state(query_vec_handle, l_insert_, l_insert_); // Use l_insert for k and L
    PerformSearch(search_state, false); // Find approximate neighbors

    // Candidates are in search_state.candidates (priority queue) and search_state.visited (set)
    // We need the combined set of visited nodes as potential neighbors.

    // 2. Robust Pruning & Update New Node
    // Collect potential neighbors (visited nodes from search)
    std::vector<std::pair<float, row_t>> potential_neighbors;
    duckdb::unordered_set<row_t> added_neighbor_ids; // Track added to avoid duplicates

    // Iterate through visited list (more reliable than candidates queue)
    // Need to actually store visited nodes with distances during search for this...
    // Alternative: Use top_candidates from search if PerformSearch stored them?
    // Let's use the priority queue content for now as an approximation.
    // FIXME: This needs refinement. The original C code uses the visited list.
    while (!search_state.candidates.empty()) {
         potential_neighbors.push_back(search_state.candidates.top());
         search_state.candidates.pop();
    }
    // Sort potential neighbors by distance
    std::sort(potential_neighbors.begin(), potential_neighbors.end());


    // --- Update New Node's Neighbors ---
    auto new_node_handle = GetNodeBuffer(new_node_ptr, true);
    auto new_node_data = new_node_handle.Ptr();
    uint16_t current_neighbor_count = 0;
    row_t* new_neighbor_ids = LMDiskannNodeAccessors::GetNeighborIDsPtrMutable(new_node_data, node_layout_);
    data_ptr_t new_compressed_neighbors_base = LMDiskannNodeAccessors::GetCompressedNeighborPtrMutable(new_node_data, node_layout_, 0, edge_vector_size_bytes_);

    // Temporary storage for neighbor vectors
    vector<float> neighbor_vector_float(dimensions_);

    for (const auto& candidate : potential_neighbors) {
        if (current_neighbor_count >= r_) break; // Stop if we have R neighbors

        row_t neighbor_rowid = candidate.second;
        if (neighbor_rowid == new_node_rowid) continue; // Don't add self

        IndexPointer neighbor_ptr;
        if (!TryGetNodePointer(neighbor_rowid, neighbor_ptr)) continue; // Skip if neighbor deleted

        // Read neighbor's full vector
        auto neighbor_handle = GetNodeBuffer(neighbor_ptr);
        auto neighbor_block_data = neighbor_handle.Ptr();
        auto neighbor_node_vec_ptr_raw = LMDiskannNodeAccessors::GetNodeVectorPtr(neighbor_block_data, node_layout_);

        // Convert neighbor vector to float for pruning calculations
        // FIXME: Handle different node_vector_type_
         if (node_vector_type_ != LMDiskannVectorType::FLOAT32) {
              throw NotImplementedException("FindAndConnectNeighbors for non-FLOAT32 node vectors not implemented.");
         }
        const float* neighbor_node_vec_ptr_float = reinterpret_cast<const float*>(neighbor_node_vec_ptr_raw);


        // --- Robust Pruning Check (Simplified from C code's diskAnnReplaceEdgeIdx) ---
        // Check if any *existing* neighbor in the new node's list prunes this candidate
        bool pruned = false;
        float dist_new_to_candidate = CalculateDistance<float, float>(new_node_vector_float, neighbor_node_vec_ptr_float);

        for (uint16_t i = 0; i < current_neighbor_count; ++i) {
            row_t existing_neighbor_id = new_neighbor_ids[i];
            IndexPointer existing_neighbor_ptr;
            if (!TryGetNodePointer(existing_neighbor_id, existing_neighbor_ptr)) continue;

            auto existing_neighbor_handle = GetNodeBuffer(existing_neighbor_ptr);
            auto existing_neighbor_block_data = existing_neighbor_handle.Ptr();
            auto existing_neighbor_vec_ptr_raw = LMDiskannNodeAccessors::GetNodeVectorPtr(existing_neighbor_block_data, node_layout_);
            // FIXME: Handle non-FLOAT32 node_vector_type_
            const float* existing_neighbor_vec_ptr_float = reinterpret_cast<const float*>(existing_neighbor_vec_ptr_raw);


            float dist_existing_to_candidate = CalculateDistance<float, float>(existing_neighbor_vec_ptr_float, neighbor_node_vec_ptr_float);

            if (dist_new_to_candidate > alpha_ * dist_existing_to_candidate) {
                pruned = true;
                break;
            }
        }
        if (pruned) continue; // This candidate is pruned by an existing neighbor

        // Add the neighbor
        new_neighbor_ids[current_neighbor_count] = neighbor_rowid;
        // Compress neighbor vector and store it
        data_ptr_t dest_compress_ptr = new_compressed_neighbors_base + current_neighbor_count * edge_vector_size_bytes_;
        if (!CompressVectorForEdge(neighbor_node_vec_ptr_float, dest_compress_ptr, dimensions_, resolved_edge_vector_type_)) {
             throw InternalException("Failed to compress neighbor vector for new node.");
        }
        added_neighbor_ids.insert(neighbor_rowid); // Track added neighbors
        current_neighbor_count++;
    }
    LMDiskannNodeAccessors::SetNeighborCount(new_node_data, current_neighbor_count);
    new_node_handle.SetModified(); // Mark new node block as modified


    // --- 4. Update Neighbors (Reciprocal Edges) ---
    // Iterate through the neighbors *actually added* to the new node
    vector<float> new_node_compressed_storage(edge_vector_size_bytes_); // Temp storage for compressed new vec
    data_ptr_t new_node_compressed_ptr = nullptr;
    if (resolved_edge_vector_type_ != LMDiskannVectorType::FLOAT32) {
         if (!CompressVectorForEdge(new_node_vector_float, new_node_compressed_storage.data(), dimensions_, resolved_edge_vector_type_)) {
             throw InternalException("Failed to compress new node vector for reciprocal edges.");
         }
         new_node_compressed_ptr = new_node_compressed_storage.data();
    } else {
         new_node_compressed_ptr = (data_ptr_t)new_node_vector_float; // Use original if float
    }


    for(row_t neighbor_rowid : added_neighbor_ids) {
         IndexPointer neighbor_ptr;
         if (!TryGetNodePointer(neighbor_rowid, neighbor_ptr)) continue;

         auto neighbor_handle = GetNodeBuffer(neighbor_ptr, true); // Get writable buffer
         auto neighbor_data = neighbor_handle.Ptr();

         // --- Robust Pruning on Neighbor ---
         // This logic needs to be carefully translated from diskAnnReplaceEdgeIdx/diskAnnPruneEdges
         // It involves calculating distances between the neighbor and its existing neighbors,
         // comparing them with distances involving the new node, and potentially replacing
         // an existing edge with the new edge (neighbor -> new_node).

         // FIXME: Implement robust pruning logic for the neighbor node here.
         // Placeholder: Naively add if space allows, without proper pruning.
         uint16_t neighbor_current_count = LMDiskannNodeAccessors::GetNeighborCount(neighbor_data);
         if (neighbor_current_count < r_) {
             row_t* neighbor_neighbor_ids = LMDiskannNodeAccessors::GetNeighborIDsPtrMutable(neighbor_data, node_layout_);
             data_ptr_t neighbor_compressed_base = LMDiskannNodeAccessors::GetCompressedNeighborPtrMutable(neighbor_data, node_layout_, 0, edge_vector_size_bytes_);

             neighbor_neighbor_ids[neighbor_current_count] = new_node_rowid;
             memcpy(neighbor_compressed_base + neighbor_current_count * edge_vector_size_bytes_,
                    new_node_compressed_ptr, edge_vector_size_bytes_);
             LMDiskannNodeAccessors::SetNeighborCount(neighbor_data, neighbor_current_count + 1);
             neighbor_handle.SetModified();
         } else {
             // FIXME: Need pruning logic to decide if new node replaces an existing edge
             Printer::Warning("Neighbor %lld is full (R=%d), skipping reciprocal edge addition due to missing pruning logic.", neighbor_rowid, r_);
         }
    }

    // throw NotImplementedException("LMDiskannIndex::FindAndConnectNeighbors not fully implemented (Robust Pruning)");
}


idx_t LMDiskannIndex::GetInMemorySize() {
    // LM-DiskANN aims for minimal memory footprint.
    // Return size of allocator metadata + RowID map + any small caches.
    idx_t base_size = allocator_ ? allocator_->GetInMemorySize() : 0;
    // idx_t map_size = rowid_map_ ? rowid_map_->GetInMemorySize() : 0;
    // Add size of PQ codebooks if loaded
    // Add size of any node cache
    return base_size; // + map_size;
}

bool LMDiskannIndex::MergeIndexes(IndexLock &state, BoundIndex &other_index) {
    throw NotImplementedException("LMDiskannIndex::MergeIndexes not implemented");
    return false;
}

void LMDiskannIndex::Vacuum(IndexLock &state) {
    // Process the deletion queue to perform deferred neighbor updates
    ProcessDeletionQueue(); // Placeholder

    // Optional: Add other vacuum tasks like graph compaction or defragmentation
    // allocator_->Vacuum(); // If allocator supports vacuuming
    // is_dirty_ should be set by ProcessDeletionQueue if changes were made
}

string LMDiskannIndex::VerifyAndToString(IndexLock &state, const bool only_verify) {
    // TODO: Implement verification
    // - Check metadata consistency.
    // - Iterate through nodes (requires RowID map or full scan).
    // - Verify neighbor counts and links.
    // - Check RowID map consistency against allocated blocks.
    if (only_verify) {
        return "VerifyAndToString(verify_only) not implemented for LM_DISKANN";
    } else {
        // Could print summary statistics: node count, avg neighbors, etc.
        return "VerifyAndToString not implemented for LM_DISKANN";
    }
}

void LMDiskannIndex::VerifyAllocations(IndexLock &state) {
    // TODO: Implement allocation verification if possible with FixedSizeAllocator
    // allocator_->Verify(); // If allocator supports verification
}

string LMDiskannIndex::GetConstraintViolationMessage(VerifyExistenceType verify_type, idx_t failed_index,
                                                     DataChunk &input) {
    // LM_DISKANN does not support constraints
    return "Constraint violation in LM_DISKANN index (Not supported)";
}


// --- Scan Method Implementations ---

unique_ptr<IndexScanState> LMDiskannIndex::InitializeScan(ClientContext &context, const Vector &query_vector, idx_t k) {
    // 1. Validate query vector
    if (query_vector.GetType().id() != LogicalTypeId::ARRAY ||
        ArrayType::GetChildType(query_vector.GetType()).id() != LogicalTypeId::FLOAT) {
        throw BinderException("LM_DISKANN query vector must be ARRAY<FLOAT>.");
    }
    idx_t query_dims = ArrayType::GetSize(query_vector.GetType());
    if (query_dims != dimensions_) {
        throw BinderException("Query vector dimension (%llu) does not match index dimension (%llu).",
                              query_dims, dimensions_);
    }

    // 2. Create scan state
    auto scan_state = make_uniq<LMDiskannScanState>(query_vector, k, l_search_); // Use index's L_search

    // 3. Find initial entry point(s)
    row_t start_node_id = GetEntryPoint(); // Get a valid entry point row_id

    if (start_node_id != NumericLimits<row_t>::Maximum()) {
        IndexPointer start_ptr;
        if (TryGetNodePointer(start_node_id, start_ptr)) {
            try {
                auto handle = GetNodeBuffer(start_ptr);
                auto block_data = handle.Ptr();
                // Calculate initial distance (approximate is fine for queue)
                // Use node vector itself for initial approx distance
                auto node_vec_ptr = LMDiskannNodeAccessors::GetNodeVectorPtr(block_data, node_layout_);
                // Need to handle node vector type here for approx distance!
                 float approx_dist;
                 if (node_vector_type_ == LMDiskannVectorType::FLOAT32) {
                     approx_dist = CalculateApproxDistance(scan_state->query_vector_ptr, node_vec_ptr);
                 } else {
                     // FIXME: Need conversion or specialized approx distance for non-float node vectors
                     throw NotImplementedException("Initial scan distance for non-FLOAT32 node vectors not implemented.");
                 }

                scan_state->candidates.push({approx_dist, start_node_id});
            } catch (std::exception &e) {
                 // Failed to read/process start node, proceed with empty candidates
                 Printer::Warning("Failed to initialize scan with start node %lld: %s", start_node_id, e.what());
            }
        } else {
             // Start node not found (maybe deleted?), try random?
             Printer::Warning("Persisted entry point node %lld not found, trying random.", start_node_id);
             start_node_id = GetRandomNodeID(); // Placeholder
             if (start_node_id != NumericLimits<row_t>::Maximum() && TryGetNodePointer(start_node_id, start_ptr)) {
                 // Try again with random node
                  try {
                     auto handle = GetNodeBuffer(start_ptr);
                     auto block_data = handle.Ptr();
                     auto node_vec_ptr = LMDiskannNodeAccessors::GetNodeVectorPtr(block_data, node_layout_);
                     // FIXME: Handle node vector type
                      float approx_dist;
                     if (node_vector_type_ == LMDiskannVectorType::FLOAT32) {
                         approx_dist = CalculateApproxDistance(scan_state->query_vector_ptr, node_vec_ptr);
                     } else {
                         throw NotImplementedException("Initial scan distance for non-FLOAT32 node vectors not implemented.");
                     }
                     scan_state->candidates.push({approx_dist, start_node_id});
                  } catch (std::exception &e) {
                     Printer::Warning("Failed to initialize scan with random node %lld: %s", start_node_id, e.what());
                  }
             }
        }
    } else {
         // Index might be empty or entry point logic failed completely
         Printer::Warning("No valid entry point found for scan initialization.");
    }

    // If candidates is still empty, Scan method will return 0 results.

    return std::move(scan_state);
}

idx_t LMDiskannIndex::Scan(IndexScanState &state, Vector &result) {
     auto &scan_state = state.Cast<LMDiskannScanState>();
     idx_t output_count = 0;
     auto result_data = FlatVector::GetData<row_t>(result);

     // Perform the beam search
     PerformSearch(scan_state, true); // Find exact distances for top candidates

     // Extract top-k results from scan_state.top_candidates
     // Sort top_candidates by distance (first element of pair)
     std::sort(scan_state.top_candidates.begin(), scan_state.top_candidates.end());

     for (const auto& candidate : scan_state.top_candidates) {
          if (output_count < STANDARD_VECTOR_SIZE && output_count < scan_state.k) {
               result_data[output_count++] = candidate.second; // candidate.second is row_t
          } else {
               break;
          }
     }

     return output_count;
}

// --- Search Helper Implementation ---
void LMDiskannIndex::PerformSearch(LMDiskannScanState &scan_state, bool find_exact_distances) {
    // Main beam search loop based on diskAnnSearchInternal

    while (!scan_state.candidates.empty()) {
        // Check if the best candidate in the queue is worse than the worst
        // candidate we've already found and fully evaluated (if we have k results).
        if (find_exact_distances && scan_state.top_candidates.size() >= scan_state.k) {
             // Find max distance in top_candidates (it's sorted, so last element)
             float worst_found_dist = scan_state.top_candidates.back().first;
             if (scan_state.candidates.top().first >= worst_found_dist) {
                  // Best candidate in queue is worse than the K we already found.
                  // Check if the candidate *itself* is already in top_candidates
                  // (can happen if added with approx distance then exact distance later).
                  bool already_in_top_k = false;
                  for(const auto& top_cand : scan_state.top_candidates) {
                      if (top_cand.second == scan_state.candidates.top().second) {
                          already_in_top_k = true;
                          break;
                      }
                  }
                  if (!already_in_top_k) {
                     // If the best candidate isn't already evaluated and is worse than
                     // the current k-th best, we can likely stop.
                     break;
                  }
             }
        }

        // 1. Select best candidate row_id from scan_state.candidates
        float candidate_dist_approx = scan_state.candidates.top().first; // Approx distance used for queue ordering
        row_t candidate_row_id = scan_state.candidates.top().second;
        scan_state.candidates.pop();

        if (scan_state.visited.count(candidate_row_id)) {
            continue; // Already visited
        }
        // 2. Mark visited
        scan_state.visited.insert(candidate_row_id);

        // 3. Get node pointer
        IndexPointer node_ptr;
        if (!TryGetNodePointer(candidate_row_id, node_ptr)) {
             // This node might have been deleted since being added to the queue
             Printer::Warning("Node %lld not found in map during search (likely deleted).", candidate_row_id);
             continue;
        }

        // 4. Read node block
        BufferHandle handle;
        try {
             handle = GetNodeBuffer(node_ptr);
        } catch (std::exception &e) {
             Printer::Warning("Failed to read block for node %lld during search: %s", candidate_row_id, e.what());
             continue; // Cannot process this node
        }
        auto block_data = handle.Ptr();

        // 5. Calculate exact distance if needed and add to top candidates
        if (find_exact_distances) {
             const_data_ptr_t node_vec_raw_ptr = LMDiskannNodeAccessors::GetNodeVectorPtr(block_data, node_layout_);
             float exact_dist;
             // Calculate exact distance based on node type
             switch(node_vector_type_) {
                 case LMDiskannVectorType::FLOAT32:
                     exact_dist = CalculateDistance<float, float>(scan_state.query_vector_ptr, reinterpret_cast<const float*>(node_vec_raw_ptr));
                     break;
                 case LMDiskannVectorType::FLOAT16:
                     exact_dist = CalculateDistance<float, float16_t>(scan_state.query_vector_ptr, reinterpret_cast<const float16_t*>(node_vec_raw_ptr));
                     break;
                 case LMDiskannVectorType::INT8:
                     exact_dist = CalculateDistance<float, int8_t>(scan_state.query_vector_ptr, reinterpret_cast<const int8_t*>(node_vec_raw_ptr));
                     break;
                 default:
                     throw InternalException("Unsupported node vector type for exact distance calculation.");
             }


             // Add to top_candidates and keep it sorted and capped at size k
             if (scan_state.top_candidates.size() < scan_state.k || exact_dist < scan_state.top_candidates.back().first) {
                  scan_state.top_candidates.push_back({exact_dist, candidate_row_id});
                  std::sort(scan_state.top_candidates.begin(), scan_state.top_candidates.end()); // Keep sorted
                  if (scan_state.top_candidates.size() > scan_state.k) {
                       scan_state.top_candidates.pop_back(); // Keep only top k
                  }
             }
        }

        // 6. Get neighbor info
        uint16_t ncount = LMDiskannNodeAccessors::GetNeighborCount(block_data);
        const row_t* neighbor_ids = LMDiskannNodeAccessors::GetNeighborIDsPtr(block_data, node_layout_);

        // 7. Iterate through neighbors
        for (uint16_t i = 0; i < ncount; ++i) {
            row_t neighbor_id = neighbor_ids[i];
            if (scan_state.visited.count(neighbor_id)) {
                continue; // Skip visited neighbors
            }

            // 8. Get compressed neighbor ptr
            const_data_ptr_t compressed_ptr = LMDiskannNodeAccessors::GetCompressedNeighborPtr(block_data, node_layout_, i, edge_vector_size_bytes_);

            // 9. Calculate approximate distance
            float approx_dist = CalculateApproxDistance(scan_state.query_vector_ptr, compressed_ptr);

            // 10. Add to candidates queue if promising
            // Only add if it's better than the worst in the queue OR the queue isn't full
            if (scan_state.candidates.size() < scan_state.l_search || approx_dist < scan_state.candidates.top().first) {
                 // FIXME: Need efficient check for duplicates in priority queue.
                 // For now, allow duplicates.
                 scan_state.candidates.push({approx_dist, neighbor_id});
                 // Prune queue if it exceeds L_search size
                 if (scan_state.candidates.size() > scan_state.l_search) {
                      scan_state.candidates.pop();
                 }
            }
        }
    } // End while loop
}

// --- Deletion Queue Helpers ---
void LMDiskannIndex::EnqueueDeletion(row_t deleted_row_id) {
    // FIXME: Implement persistent delete queue using allocator_
    // 1. Allocate a small block (size DELETE_QUEUE_ENTRY_SIZE)
    //    IndexPointer new_queue_entry_ptr = allocator_->New(DELETE_QUEUE_ENTRY_SIZE); // Need allocator variant?
    //    For now, assume we use the main allocator and block_size_bytes_
    IndexPointer new_queue_entry_ptr = allocator_->New();
    if (!new_queue_entry_ptr.IsValid()) {
         throw IOException("Failed to allocate block for delete queue entry.");
    }

    // 2. Pin the new block (writable)
    auto handle = GetNodeBuffer(new_queue_entry_ptr, true);
    auto data_ptr = handle.Ptr();

    // 3. Write deleted_row_id and current delete_queue_head_ptr_
    Store<row_t>(deleted_row_id, data_ptr);
    IndexPointer current_head = delete_queue_head_ptr_; // Read current head
    // Write IndexPointer (block_id + offset)
    Store<block_id_t>(current_head.GetBlockId(), data_ptr + sizeof(row_t));
    Store<uint32_t>(current_head.GetOffset(), data_ptr + sizeof(row_t) + sizeof(block_id_t));

    // 4. Mark buffer modified
    handle.SetModified();

    // 5. Update delete_queue_head_ptr_ to point to the new block
    delete_queue_head_ptr_ = new_queue_entry_ptr;

    // 6. Mark index as dirty (metadata needs persistence)
    is_dirty_ = true;
    // Printer::Warning("LMDiskannIndex::EnqueueDeletion not fully implemented (using main allocator).");
}

void LMDiskannIndex::ProcessDeletionQueue() {
    // FIXME: Implement processing logic during Vacuum
    // This requires iterating through *all* nodes in the index, which is
    // very expensive without extra structures. This is a major challenge
    // for efficient eager deletion in graph indexes.
    // A full implementation is complex and likely beyond this scope.
    // Conceptual Steps:
    // 1. Read all deleted IDs from the queue into memory.
    // 2. Iterate through *all* nodes in the index (using RowID map iteration).
    // 3. For each node:
    //    a. Pin its block (writable).
    //    b. Check its neighbor list for any of the deleted IDs.
    //    c. If found, remove the edge(s) (shift remaining neighbors).
    //    d. Update neighbor count.
    //    e. Mark buffer modified if changed.
    // 4. Free the delete queue blocks.
    // 5. Reset delete_queue_head_ptr_.
    // 6. Mark index dirty if changes were made.

    if (delete_queue_head_ptr_.IsValid()) {
        Printer::Warning("LMDiskannIndex::ProcessDeletionQueue: Processing deferred deletions is not implemented.");
        // Conceptually clear the queue after processing (or attempting)
        // delete_queue_head_ptr_.Clear();
        // is_dirty_ = true;
    }
}

// --- Entry Point Helpers ---
row_t LMDiskannIndex::GetEntryPoint() {
    // If we have a cached valid rowid, return it
    if (graph_entry_point_rowid_ != NumericLimits<row_t>::Maximum()) {
        IndexPointer ptr_check;
        if (TryGetNodePointer(graph_entry_point_rowid_, ptr_check)) {
            // Optional: Could check if ptr_check matches graph_entry_point_ptr_
            return graph_entry_point_rowid_;
        } else {
            // Entry point was deleted, clear cache and find new one
            Printer::Warning("Cached entry point %lld deleted.", graph_entry_point_rowid_);
            graph_entry_point_ptr_.Clear();
            graph_entry_point_rowid_ = NumericLimits<row_t>::Maximum();
            is_dirty_ = true; // Need to persist cleared entry point
        }
    }

    // If no valid cached entry point, try using the persisted pointer
    if (graph_entry_point_ptr_.IsValid()) {
        // FIXME: Need inverse mapping from IndexPointer to row_id
        // This might involve reading the block pointed to by graph_entry_point_ptr_
        // if the row_id isn't stored elsewhere.
        // For now, return placeholder indicating pointer is valid but rowid unknown
        Printer::Warning("Entry point pointer is valid, but rowid lookup not implemented. Cannot use as entry point yet.");
        // return -2; // Placeholder
    }

    // Fallback: Get a random node ID
    row_t random_id = GetRandomNodeID(); // Placeholder
    if (random_id != NumericLimits<row_t>::Maximum()) {
         IndexPointer random_ptr;
         if(TryGetNodePointer(random_id, random_ptr)) {
            // Cache this random node as the new entry point
            SetEntryPoint(random_id, random_ptr);
            return random_id;
         }
    }

    // No valid entry point found (index might be empty)
    return NumericLimits<row_t>::Maximum();
}

void LMDiskannIndex::SetEntryPoint(row_t row_id, IndexPointer node_ptr) {
    graph_entry_point_rowid_ = row_id;
    graph_entry_point_ptr_ = node_ptr;
    is_dirty_ = true; // Need to persist the new entry point
}

row_t LMDiskannIndex::GetRandomNodeID() {
    // FIXME: Implement random node selection using RowID map iteration/sampling
    // Requires iterating the ART index or sampling keys.
    // This needs the ART map to be implemented.
    // Placeholder: Return invalid rowid
    // Printer::Warning("LMDiskannIndex::GetRandomNodeID not implemented, returning invalid rowid.");
    return NumericLimits<row_t>::Maximum();
}


} // namespace duckdb
