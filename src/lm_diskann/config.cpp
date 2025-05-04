
#include "config.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/limits.hpp" // For NumericLimits
#include "duckdb/storage/storage_info.hpp" // For Storage::SECTOR_SIZE

#include <cmath>
#include <algorithm> // For std::min, std::max

namespace duckdb {

// --- Configuration Constants ---
const char *LMDISKANN_METRIC_OPTION = "METRIC";
const char *LMDISKANN_EDGE_TYPE_OPTION = "EDGE_TYPE";
const char *LMDISKANN_R_OPTION = "R";
const char *LMDISKANN_L_INSERT_OPTION = "L_INSERT";
const char *LMDISKANN_ALPHA_OPTION = "ALPHA";
const char *LMDISKANN_L_SEARCH_OPTION = "L_SEARCH";

const LMDiskannMetricType LMDISKANN_DEFAULT_METRIC = LMDiskannMetricType::L2;
const LMDiskannEdgeType LMDISKANN_DEFAULT_EDGE_TYPE = LMDiskannEdgeType::SAME_AS_NODE;
const uint32_t LMDISKANN_DEFAULT_R = 64;
const uint32_t LMDISKANN_DEFAULT_L_INSERT = 128; // VECTOR_INSERT_L_DEFAULT
const float LMDISKANN_DEFAULT_ALPHA = 1.2f;      // VECTOR_PRUNING_ALPHA_DEFAULT
const uint32_t LMDISKANN_DEFAULT_L_SEARCH = 100; // VECTOR_SEARCH_L_DEFAULT
const uint8_t LMDISKANN_CURRENT_FORMAT_VERSION = 3; // VECTOR_FORMAT_DEFAULT

// --- Node Block Layout Constants ---
constexpr idx_t OFFSET_NEIGHBOR_COUNT = 0; // uint16_t
constexpr idx_t NODE_VECTOR_ALIGNMENT = 8; // Align vectors to 8 bytes (adjust if needed)
constexpr idx_t DISKANN_MAX_BLOCK_SZ = 128 * 1024 * 1024; // Max allowed block size


// --- Configuration Functions ---

idx_t GetVectorTypeSizeBytes(LMDiskannVectorType type) {
    switch (type) {
    case LMDiskannVectorType::FLOAT32:
        return sizeof(float);
    case LMDiskannVectorType::INT8:
        return sizeof(int8_t);
    // case LMDiskannVectorType::FLOAT16:
    //     return sizeof(float16_t); // DuckDB's half-float type
    default:
        throw InternalException("Unsupported LMDiskannVectorType for size calculation");
    }
}

// Calculates the byte size per vector for the chosen edge compression format.
idx_t GetEdgeVectorTypeSizeBytes(LMDiskannEdgeType type, LMDiskannVectorType node_type, idx_t dimensions) {
     LMDiskannVectorType resolved_type;
     if (type == LMDiskannEdgeType::SAME_AS_NODE) {
         resolved_type = node_type;
     } else {
        // Map EdgeType enum back to VectorType enum for size calculation (if applicable)
        switch(type) {
            case LMDiskannEdgeType::FLOAT32: resolved_type = LMDiskannVectorType::FLOAT32; break;
            case LMDiskannEdgeType::FLOAT16: resolved_type = LMDiskannVectorType::FLOAT16; break;
            case LMDiskannEdgeType::INT8:    resolved_type = LMDiskannVectorType::INT8; break;
            case LMDiskannEdgeType::FLOAT1BIT:
                 // 1 bit per dimension, round up to nearest byte
                 return (dimensions + 7) / 8;
            case LMDiskannEdgeType::TERNARY:
                 // 2 bits per dimension (pos/neg planes), round up to nearest byte
                 return (dimensions * 2 + 7) / 8;
            default: throw InternalException("Unsupported LMDiskannEdgeType for size calculation");
        }
     }
     // Handle UNKNOWN case resulting from FLOAT1BIT/TERNARY mapping (should not happen here)
     if (resolved_type == LMDiskannVectorType::UNKNOWN) {
        throw InternalException("Cannot calculate size for UNKNOWN vector type");
     }
     return GetVectorTypeSizeBytes(resolved_type) * dimensions; // Size for standard types
}

void ParseOptions(const case_insensitive_map_t<Value> &options,
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
        } else if (edge_type_str == "TERNARY") { // Added Ternary
             edge_type = LMDiskannEdgeType::TERNARY;
        } else {
             throw BinderException("Unsupported EDGE_TYPE '%s' for LM_DISKANN index. Supported types: FLOAT32, FLOAT16, INT8, FLOAT1BIT, TERNARY", edge_type_str);
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

void ValidateParameters(LMDiskannMetricType metric_type, LMDiskannEdgeType edge_type_param,
                        uint32_t r, uint32_t l_insert, float alpha, uint32_t l_search) {
    if (r == 0) throw BinderException("LM_DISKANN parameter R must be > 0");
    if (l_insert == 0) throw BinderException("LM_DISKANN parameter L_INSERT must be > 0");
    if (alpha < 1.0f) throw BinderException("LM_DISKANN parameter ALPHA must be >= 1.0");
    if (l_search == 0) throw BinderException("LM_DISKANN parameter L_SEARCH must be > 0");

    if (edge_type_param == LMDiskannEdgeType::FLOAT1BIT && metric_type != LMDiskannMetricType::COSINE) {
         throw BinderException("LM_DISKANN EDGE_TYPE 'FLOAT1BIT' is only supported with METRIC 'COSINE'");
    }
    if (edge_type_param == LMDiskannEdgeType::TERNARY && metric_type == LMDiskannMetricType::L2) {
         // Ternary dot product is not a good proxy for L2 distance
         throw BinderException("LM_DISKANN EDGE_TYPE 'TERNARY' is not recommended with METRIC 'L2'. Use COSINE or IP.");
    }
    // Add more validation as needed
}


NodeLayoutOffsets CalculateLayoutInternal(idx_t dimensions, idx_t r,
                                          idx_t node_vector_size_bytes,
                                          idx_t edge_vector_size_bytes) {
    NodeLayoutOffsets layout;

    // Offset 0: Neighbor count (uint16_t)
    layout.neighbor_count = OFFSET_NEIGHBOR_COUNT; // = 0
    idx_t current_offset = sizeof(uint16_t);
    // Add other fixed-size metadata here if needed

    // Align for node vector
    current_offset = AlignValue(current_offset, NODE_VECTOR_ALIGNMENT);
    layout.node_vector = current_offset;
    current_offset += node_vector_size_bytes;

    // Align for neighbor IDs (row_t is usually 64-bit, likely aligned)
    current_offset = AlignValue(current_offset, sizeof(row_t));
    layout.neighbor_ids = current_offset;
    current_offset += r * sizeof(row_t);

    // Align for compressed neighbors
    // For bit-based formats (TERNARY, FLOAT1BIT), alignment might be less critical,
    // but aligning to byte boundary (1) is minimal. Aligning to larger boundary (e.g., 8)
    // might be beneficial if SIMD kernels expect aligned loads, even if data itself isn't full width.
    idx_t edge_alignment = (edge_vector_size_bytes > 1) ? std::min((idx_t)8, NextPowerOfTwo(edge_vector_size_bytes)) : 1;
    // Let's force 8-byte alignment for simplicity if edge size > 1 byte.
    edge_alignment = (edge_vector_size_bytes > 1) ? 8 : 1;
    current_offset = AlignValue(current_offset, edge_alignment);
    layout.compressed_neighbors = current_offset;
    current_offset += r * edge_vector_size_bytes;

    layout.total_size = current_offset; // Size *before* final block alignment
    return layout;
}


} // namespace duckdb
