#pragma once

// #include "config.hpp" // Avoid direct include to prevent redefinitions
#include "duckdb/common/constants.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/storage/data_pointer.hpp"
#include "duckdb/common/types/value.hpp" // Needed for Value type
#include "duckdb/common/case_insensitive_map.hpp" // Needed for case_insensitive_map_t
// #include "duckdb/common/types/uhugeint.hpp" // Likely not needed here directly

#include <cstdint>

// Forward declare types defined in config.hpp
namespace duckdb {
    enum class LMDiskannMetricType : uint8_t;
    // struct NodeLayoutOffsets; // Not needed for function signatures here
    // class Value; // Included above
    // template <typename T> class case_insensitive_map_t; // Included above
}

namespace duckdb {

// Forward declarations are necessary if config.hpp isn't included
enum class LMDiskannVectorType : uint8_t;

// --- Removed forward declarations for LMDiskannVectorType and LMDiskannMetricType ---

// --- Enums for LM-DiskANN Parameters ---
// Defines the types for configuration options like distance metric,
// vector storage types (for nodes and compressed edges).

// Corresponds to nDistanceFunc / VECTOR_METRIC_TYPE_PARAM_ID
enum class LMDiskannMetricType : uint8_t {
	UNKNOWN = 0,
	L2 = 1,     // VECTOR_METRIC_TYPE_L2
	COSINE = 2, // VECTOR_METRIC_TYPE_COS
	IP = 3,      // Inner Product
	HAMMING = 4, // Hamming distance (for FLOAT1BIT)
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

// Validates the combination of parameters.
void ValidateParameters(LMDiskannMetricType metric_type,
                        uint32_t r, uint32_t l_insert, float alpha, uint32_t l_search);

// Calculates the byte size of different vector types.
idx_t GetVectorTypeSizeBytes(LMDiskannVectorType type);

// Calculates the internal layout offsets within a node block.
NodeLayoutOffsets CalculateLayoutInternal(idx_t dimensions, idx_t r,
                                          idx_t node_vector_size_bytes,
                                          idx_t edge_vector_size_bytes);

// Calculates distance between two full-precision vectors (potentially different types).
// Assumes conversion to float internally for calculation.
float ComputeDistance(const_data_ptr_t vec_a_ptr, LMDiskannVectorType type_a,
                      const_data_ptr_t vec_b_ptr, LMDiskannVectorType type_b,
                      idx_t dimensions, LMDiskannMetricType metric_type);


// --- Core Distance/Similarity/Compression Functions for LM-DiskANN (Ternary Neighbors) ---

//! Computes the EXACT distance between two full-precision FLOAT vectors.
//! Used for final re-ranking.
float ComputeExactDistanceFloat(const float *a_ptr,
                                const float *b_ptr,
                                idx_t dimensions,
                                LMDiskannMetricType metric_type);


//! Computes the approximate SIMILARITY between a full query vector (float)
//! and a compressed TERNARY neighbor vector using its separate planes.
//! Returns a raw similarity score (higher is better).
//! Used during graph traversal.
float ComputeApproxSimilarityTernary(const float *query_float_ptr,
                                     const_data_ptr_t neighbor_pos_plane_ptr, // Neighbor's pos plane
                                     const_data_ptr_t neighbor_neg_plane_ptr, // Neighbor's neg plane
                                     idx_t dimensions,
                                     LMDiskannMetricType metric_type);

//! Compresses a float vector into the TERNARY format using separate pos/neg planes.
//! Used during index building to store compressed neighbors.
void CompressVectorToTernary(const float* input_float_ptr,
                             data_ptr_t dest_pos_plane_ptr, // Output pos plane
                             data_ptr_t dest_neg_plane_ptr, // Output neg plane
                             idx_t dimensions);

//! Computes the EXACT distance between two full-precision FLOAT vectors.
float ComputeExactDistanceFloat(const float *a_ptr, const float *b_ptr,
                               idx_t dimensions, LMDiskannMetricType metric_type);


//! Computes the approximate SIMILARITY between a full query vector (float)
//! and a compressed TERNARY neighbor vector using its separate planes.
//! Returns a raw similarity score (higher is better).
//! The caller needs to handle converting this to a distance if needed (e.g., negate).
float ComputeApproxDistance(const float *query_ptr,
                            const_data_ptr_t pos_plane_ptr, // Neighbor's pos plane
                            const_data_ptr_t neg_plane_ptr, // Neighbor's neg plane
                            idx_t dimensions, LMDiskannMetricType metric_type);

//! Compresses a float vector into the TERNARY format using separate pos/neg planes.
void CompressVectorForEdge(const float* input_float_ptr,
                           data_ptr_t dest_pos_plane_ptr, // Output pos plane
                           data_ptr_t dest_neg_plane_ptr, // Output neg plane
                           idx_t dimensions);

} // namespace duckdb
