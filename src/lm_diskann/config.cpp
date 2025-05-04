#include "config.hpp"
#include "ternary_quantization.hpp" // Needed for WordsPerPlane

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/limits.hpp" // For NumericLimits
#include "duckdb/storage/storage_info.hpp" // For Storage::SECTOR_SIZE

#include <cmath>
#include <algorithm> // For std::min, std::max

namespace duckdb {

// --- Configuration Constants ---
const char *LMDISKANN_METRIC_OPTION = "METRIC";
// const char *LMDISKANN_EDGE_TYPE_OPTION = "EDGE_TYPE"; // Removed
const char *LMDISKANN_R_OPTION = "R";
const char *LMDISKANN_L_INSERT_OPTION = "L_INSERT";
const char *LMDISKANN_ALPHA_OPTION = "ALPHA";
const char *LMDISKANN_L_SEARCH_OPTION = "L_SEARCH";

const LMDiskannMetricType LMDISKANN_DEFAULT_METRIC = LMDiskannMetricType::L2;
const uint32_t LMDISKANN_DEFAULT_R = 64;
const uint32_t LMDISKANN_DEFAULT_L_INSERT = 128; // VECTOR_INSERT_L_DEFAULT
const float LMDISKANN_DEFAULT_ALPHA = 1.2f;      // VECTOR_PRUNING_ALPHA_DEFAULT
const uint32_t LMDISKANN_DEFAULT_L_SEARCH = 100; // VECTOR_SEARCH_L_DEFAULT
const uint8_t LMDISKANN_CURRENT_FORMAT_VERSION = 3; // VECTOR_FORMAT_DEFAULT

// --- Node Block Layout Constants ---
constexpr idx_t OFFSET_NEIGHBOR_COUNT = 0; // uint16_t
constexpr idx_t NODE_VECTOR_ALIGNMENT = 8; // Align vectors to 8 bytes (adjust if needed)
constexpr idx_t PLANE_ALIGNMENT = 8;       // Align ternary planes to 8 bytes
constexpr idx_t DISKANN_MAX_BLOCK_SZ = 128 * 1024 * 1024; // Max allowed block size


// --- Configuration Functions ---

idx_t GetVectorTypeSizeBytes(LMDiskannVectorType type) {
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

// Calculates the byte size per neighbor for the ternary edge compression format (both planes).
// Equivalent to ceil(dims * 2 / 8.0)
idx_t GetTernaryVectorSizeBytes(idx_t dimensions) {
    return (dimensions * 2 + 7) / 8;
}

// Calculates the byte size for *one* ternary plane (pos or neg) for one neighbor.
// Equivalent to ceil(dims / 64.0) * sizeof(uint64_t)
// Note: This uses the helper from ternary_quantization.hpp
idx_t GetTernaryPlaneSizeBytes(idx_t dimensions) {
    return WordsPerPlane(dimensions) * sizeof(uint64_t);
}

void ParseOptions(const case_insensitive_map_t<Value> &options,
                  LMDiskannMetricType &metric_type,
                  uint32_t &r, uint32_t &l_insert,
                  float &alpha, uint32_t &l_search) {
    // Apply defaults first
    metric_type = LMDISKANN_DEFAULT_METRIC;
    // edge_type = LMDISKANN_DEFAULT_EDGE_TYPE; // Removed
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

    // Removed EDGE_TYPE parsing block

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

void ValidateParameters(LMDiskannMetricType metric_type,
                        uint32_t r, uint32_t l_insert, float alpha, uint32_t l_search) {
    if (r == 0) throw BinderException("LM_DISKANN parameter R must be > 0");
    if (l_insert == 0) throw BinderException("LM_DISKANN parameter L_INSERT must be > 0");
    if (alpha < 1.0f) throw BinderException("LM_DISKANN parameter ALPHA must be >= 1.0");
    if (l_search == 0) throw BinderException("LM_DISKANN parameter L_SEARCH must be > 0");

    // Implicitly assume neighbors are TERNARY
    if (metric_type == LMDiskannMetricType::L2) {
         // Ternary dot product is not a good proxy for L2 distance
         throw BinderException("LM_DISKANN with TERNARY neighbors is not compatible with METRIC 'L2'. Use COSINE or IP.");
    }
    // Add more validation as needed
}


NodeLayoutOffsets CalculateLayoutInternal(idx_t dimensions, idx_t r,
                                          idx_t node_vector_size_bytes) {
                                          // idx_t edge_vector_size_bytes) { // Removed
    NodeLayoutOffsets layout;
    idx_t current_offset = 0;

    // Offset 0: Neighbor count (uint16_t)
    layout.neighbor_count = OFFSET_NEIGHBOR_COUNT; // = 0
    current_offset = sizeof(uint16_t);
    // Add other fixed-size metadata here if needed

    // Align for node vector
    current_offset = AlignValue<idx_t, NODE_VECTOR_ALIGNMENT>(current_offset);
    layout.node_vector = current_offset;
    current_offset += node_vector_size_bytes;

    // Align for neighbor IDs (row_t is usually 64-bit, likely aligned)
    current_offset = AlignValue<idx_t, sizeof(row_t)>(current_offset);
    layout.neighbor_ids = current_offset;
    current_offset += r * sizeof(row_t);

    // Calculate size of one plane (pos or neg) for ONE neighbor
    idx_t plane_size_bytes_per_neighbor = GetTernaryPlaneSizeBytes(dimensions);

    // Align for positive planes array
    current_offset = AlignValue<idx_t, PLANE_ALIGNMENT>(current_offset);
    layout.pos_planes = current_offset;
    current_offset += r * plane_size_bytes_per_neighbor; // Total size for all positive planes

    // Align for negative planes array
    current_offset = AlignValue<idx_t, PLANE_ALIGNMENT>(current_offset);
    layout.neg_planes = current_offset;
    current_offset += r * plane_size_bytes_per_neighbor; // Total size for all negative planes

    layout.total_size = current_offset; // Size *before* final block alignment

    // Optional: Check against max block size (though this should ideally happen later)
    // if (AlignValue<idx_t, Storage::SECTOR_SIZE>(layout.total_size) > DISKANN_MAX_BLOCK_SZ) {
    //     throw BinderException("Calculated node block size exceeds maximum allowed (%llu bytes)", DISKANN_MAX_BLOCK_SZ);
    // }

    return layout;
}


} // namespace duckdb
