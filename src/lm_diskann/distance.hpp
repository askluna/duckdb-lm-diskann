#pragma once

#include "duckdb.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/types/value.hpp"

#include <cstdint>

namespace duckdb {

// --- Enums for LM-DiskANN Parameters ---
// Defines the types for configuration options like distance metric,
// vector storage types (for nodes and compressed edges).

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

// Corresponds to nEdgeVectorType / VECTOR_COMPRESS_NEIGHBORS_PARAM_ID
enum class LMDiskannEdgeType : uint8_t {
    SAME_AS_NODE = 0, // Default: Use node's type
    FLOAT32 = 1,      // VECTOR_TYPE_FLOAT
    FLOAT16 = 2,      // VECTOR_TYPE_FLOAT16
    INT8 = 3,         // VECTOR_TYPE_INT8
    FLOAT1BIT = 4,    // VECTOR_TYPE_FLOAT1BIT (Cosine/Hamming only)
    TERNARY = 5       // Ternary (+1, 0, -1) quantization (Cosine/IP proxy)
};

// --- Struct to hold calculated layout offsets ---
// Stores the byte offsets of different data sections within a node's disk block.
// Crucial for low-level node accessors.
struct NodeLayoutOffsets {
    idx_t neighbor_count = 0; // Offset of the neighbor count (uint16_t)
    idx_t node_vector = 0;    // Offset of the start of the node's full vector data
    idx_t neighbor_ids = 0;   // Offset of the start of the neighbor row_t array
    idx_t compressed_neighbors = 0; // Offset of the start of the compressed neighbor vectors array
    idx_t total_size = 0;     // Total size *before* final block alignment
};


// --- Configuration Constants ---
// Defines string keys for CREATE INDEX options and default values.

extern const char *LMDISKANN_METRIC_OPTION;
extern const char *LMDISKANN_EDGE_TYPE_OPTION;
extern const char *LMDISKANN_R_OPTION;
extern const char *LMDISKANN_L_INSERT_OPTION;
extern const char *LMDISKANN_ALPHA_OPTION;
extern const char *LMDISKANN_L_SEARCH_OPTION;

extern const LMDiskannMetricType LMDISKANN_DEFAULT_METRIC;
extern const LMDiskannEdgeType LMDISKANN_DEFAULT_EDGE_TYPE;
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
                  LMDiskannEdgeType &edge_type,
                  uint32_t &r, uint32_t &l_insert,
                  float &alpha, uint32_t &l_search);

// Validates the combination of parameters (e.g., FLOAT1BIT only with COSINE).
void ValidateParameters(LMDiskannMetricType metric_type, LMDiskannEdgeType edge_type_param,
                        uint32_t r, uint32_t l_insert, float alpha, uint32_t l_search);

// Calculates the byte size of different vector types.
idx_t GetVectorTypeSizeBytes(LMDiskannVectorType type);
// Calculates the byte size of the chosen edge compression format.
idx_t GetEdgeVectorTypeSizeBytes(LMDiskannEdgeType type, LMDiskannVectorType node_type, idx_t dimensions);

// Calculates the internal layout offsets within a node block.
NodeLayoutOffsets CalculateLayoutInternal(idx_t dimensions, idx_t r,
                                          idx_t node_vector_size_bytes,
                                          idx_t edge_vector_size_bytes);

} // namespace duckdb
