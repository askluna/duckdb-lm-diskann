#pragma once

#include "duckdb.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/types/value.hpp"

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

// --- Struct to hold calculated layout offsets ---
// Stores the byte offsets of different data sections within a node's disk block.
// Crucial for low-level node accessors.
struct NodeLayoutOffsets {
    idx_t neighbor_count = 0; // Offset of the neighbor count (uint16_t)
    idx_t node_vector = 0;    // Offset of the start of the node's full vector data
    idx_t neighbor_ids = 0;   // Offset of the start of the neighbor row_t array
    idx_t pos_planes = 0;     // Offset of the start of the positive ternary planes array
    idx_t neg_planes = 0;     // Offset of the start of the negative ternary planes array
    idx_t total_size = 0;     // Total size *before* final block alignment
};


// --- Configuration Constants ---
// Defines string keys for CREATE INDEX options and default values.

extern const char *LMDISKANN_METRIC_OPTION;
extern const char *LMDISKANN_R_OPTION;
extern const char *LMDISKANN_L_INSERT_OPTION;
extern const char *LMDISKANN_ALPHA_OPTION;
extern const char *LMDISKANN_L_SEARCH_OPTION;

extern const LMDiskannMetricType LMDISKANN_DEFAULT_METRIC;
extern const uint32_t LMDISKANN_DEFAULT_R;
extern const uint32_t LMDISKANN_DEFAULT_L_INSERT;
extern const float LMDISKANN_DEFAULT_ALPHA;
extern const uint32_t LMDISKANN_DEFAULT_L_SEARCH;
extern const uint8_t LMDISKANN_CURRENT_FORMAT_VERSION;

// --- Configuration Functions ---
// Provides functions to parse options, validate parameters, calculate sizes,
// and determine the node block layout.

// Parses options from the CREATE INDEX statement's WITH clause.
void ParseOptions(const case_insensitive_map_t<Value> &options,
                  LMDiskannMetricType &metric_type,
                  uint32_t &r, uint32_t &l_insert,
                  float &alpha, uint32_t &l_search);

// Validates the combination of parameters. Implicitly assumes TERNARY neighbors.
void ValidateParameters(LMDiskannMetricType metric_type,
                        uint32_t r, uint32_t l_insert, float alpha, uint32_t l_search);

// Calculates the byte size of different vector types.
idx_t GetVectorTypeSizeBytes(LMDiskannVectorType type);
// Calculates the byte size of the compressed ternary format per neighbor (both planes).
idx_t GetTernaryVectorSizeBytes(idx_t dimensions);

// Calculates the internal layout offsets within a node block.
// Implicitly assumes TERNARY neighbors.
NodeLayoutOffsets CalculateLayoutInternal(idx_t dimensions, idx_t r,
                                          idx_t node_vector_size_bytes);

} // namespace duckdb
