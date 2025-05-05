#pragma once

#include "duckdb.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/execution/index/index_pointer.hpp"

#include <cstdint>

namespace duckdb {

// --- Enums for LM-DiskANN Parameters ---
// Defines the types for configuration options like distance metric,
// vector storage types (for nodes).

// Corresponds to nDistanceFunc / VECTOR_METRIC_TYPE_PARAM_ID
enum class LMDiskannMetricType : uint8_t {
    UNKNOWN = 0,
    L2 = 1,      // VECTOR_METRIC_TYPE_L2
    COSINE = 2,  // VECTOR_METRIC_TYPE_COS
    IP = 3       // Inner Product
    // Add HAMMING later if needed for FLOAT1BIT
};

// Corresponds to nNodeVectorType / VECTOR_TYPE_PARAM_ID
enum class LMDiskannVectorType : uint8_t {
    UNKNOWN = 0,
    FLOAT32 = 1, // VECTOR_TYPE_FLOAT
    INT8 = 2,    // VECTOR_TYPE_INT8
    FLOAT16 = 3  // VECTOR_TYPE_FLOAT16 (Requires conversion/handling)
    // Add other types if needed (e.g., BFLOAT16)
};

// --- Configuration Constants ---
// Defines string keys for CREATE INDEX options and default values.

// Option strings used in WITH clause, grouped in a struct
struct LmDiskannOptionKeys {
    static constexpr const char* METRIC = "METRIC";
    static constexpr const char* R = "R";
    static constexpr const char* L_INSERT = "L_INSERT";
    static constexpr const char* ALPHA = "ALPHA";
    static constexpr const char* L_SEARCH = "L_SEARCH";
};

// Default parameter values grouped in a struct
struct LmDiskannConfigDefaults {
    static constexpr LMDiskannMetricType METRIC = LMDiskannMetricType::L2;
    static constexpr uint32_t R = 64;
    static constexpr uint32_t L_INSERT = 128;
    static constexpr float ALPHA = 1.2f;
    static constexpr uint32_t L_SEARCH = 100;
};

// Format version (separate from parameter defaults)
inline constexpr uint8_t LMDISKANN_CURRENT_FORMAT_VERSION = 3;

// --- Configuration Struct ---
struct LMDiskannConfig {
    // Parameters parsed from options
    LMDiskannMetricType metric_type = LmDiskannConfigDefaults::METRIC;
    uint32_t r = LmDiskannConfigDefaults::R;
    uint32_t l_insert = LmDiskannConfigDefaults::L_INSERT;
    float alpha = LmDiskannConfigDefaults::ALPHA;
    uint32_t l_search = LmDiskannConfigDefaults::L_SEARCH;

    // Parameters derived from table/column info (passed in separately or added later)
    idx_t dimensions = 0;
    LMDiskannVectorType node_vector_type = LMDiskannVectorType::UNKNOWN;

    // Could add calculated values here too if frequently needed together
    // idx_t node_vector_size_bytes = 0;
    // idx_t ternary_plane_size_bytes = 0;
};

// --- Struct to hold calculated layout offsets ---
// Stores the byte offsets of different data sections within a node's disk block.
// Crucial for low-level node accessors. Assumes TERNARY compressed neighbors.
struct NodeLayoutOffsets {
    idx_t neighbor_count_offset = 0; // Offset of the neighbor count (uint16_t) - Typically 0
    idx_t node_vector_offset = 0;    // Offset of the start of the node's full vector data
    idx_t neighbor_ids_offset = 0;   // Offset of the start of the neighbor row_t array
    idx_t neighbor_pos_planes_offset = 0; // Offset of the start of the positive ternary planes array
    idx_t neighbor_neg_planes_offset = 0; // Offset of the start of the negative ternary planes array
    idx_t total_node_size = 0;     // Total size *before* final block alignment (used for allocation/memcpy)
};


// --- Configuration Functions ---
// Provides functions to parse options, validate parameters, calculate sizes,
// and determine the node block layout.

// Parses options from the CREATE INDEX statement's WITH clause into a config struct.
LMDiskannConfig ParseOptions(const case_insensitive_map_t<Value> &options);

// Validates the combination of parameters within the config struct.
void ValidateParameters(const LMDiskannConfig &config);

// Calculates the byte size of different node vector types.
idx_t GetVectorTypeSizeBytes(LMDiskannVectorType type);

// Calculates the byte size of the compressed ternary format per neighbor (one plane).
idx_t GetTernaryPlaneSizeBytes(idx_t dimensions);

// Calculates the internal layout offsets within a node block based on config.
// Requires dimensions and node_vector_type to be set in the config.
NodeLayoutOffsets CalculateLayoutInternal(const LMDiskannConfig &config);

// --- Metadata Struct --- //
// Holds all parameters persisted in the index metadata block.
struct LMDiskannMetadata {
    uint8_t format_version = 0;                  // Internal format version for compatibility
    LMDiskannMetricType metric_type = LMDiskannMetricType::UNKNOWN; // Distance metric used
    LMDiskannVectorType node_vector_type = LMDiskannVectorType::UNKNOWN; // Type of vectors stored in nodes
    // Edge type is implicitly Ternary, no need to store explicitly
    idx_t dimensions = 0;                        // Vector dimensionality
    uint32_t r = 0;                              // Max neighbors per node (graph degree)
    uint32_t l_insert = 0;                       // Search list size during insertion
    float alpha = 0.0f;                          // Pruning factor during insertion
    uint32_t l_search = 0;                       // Search list size during query
    idx_t block_size_bytes = 0;                  // Size of each node block on disk
    IndexPointer graph_entry_point_ptr;          // Pointer to the entry node block
    IndexPointer delete_queue_head_ptr;          // Pointer to the head of the delete queue block
    // IndexPointer rowid_map_root_ptr; // TODO: Add when ART is integrated
};

} // namespace duckdb
