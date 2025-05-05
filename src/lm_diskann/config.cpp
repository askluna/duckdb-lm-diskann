#include "config.hpp"
#include "ternary_quantization.hpp" // Needed for WordsPerPlane

// #include "duckdb/parser/binder/binder_exception.hpp" // Removed: File not found
#include "duckdb/common/exception.hpp" // Using base exception
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/limits.hpp" // For NumericLimits
#include "duckdb/storage/storage_info.hpp" // For Storage::SECTOR_SIZE
#include "duckdb/common/helper.hpp" // For AlignValue
#include "duckdb/common/types/value.hpp" // Needed for ParseOptions
#include "duckdb/common/case_insensitive_map.hpp" // Needed for ParseOptions

#include <cmath>
#include <algorithm> // For std::min, std::max

namespace duckdb {

// Constant definitions are now in config.hpp using inline constexpr

// --- Node Block Layout Constants (Keep definitions here if only used internally in cpp) ---
constexpr idx_t OFFSET_NEIGHBOR_COUNT = 0; // uint16_t
constexpr idx_t NODE_VECTOR_ALIGNMENT = 8; // Align vectors to 8 bytes (adjust if needed)
constexpr idx_t PLANE_ALIGNMENT = 8;       // Align ternary planes to 8 bytes
constexpr idx_t DISKANN_MAX_BLOCK_SZ = 128 * 1024 * 1024; // Max allowed block size


// --- Configuration Functions --- //

idx_t GetVectorTypeSizeBytes(LMDiskannVectorType type) {
    switch (type) {
    case LMDiskannVectorType::FLOAT32:
        return sizeof(float);
    case LMDiskannVectorType::INT8:
        return sizeof(int8_t);
    case LMDiskannVectorType::FLOAT16:
        // Assuming float16_t is defined appropriately, e.g., via duckdb/common/types/value.hpp or similar
        #ifdef DUCKDB_HAS_FLOAT16
          return sizeof(float16_t);
        #else
          throw InternalException("FLOAT16 support not compiled in");
        #endif
    default:
        throw InternalException("Unsupported LMDiskannVectorType for size calculation");
    }
}


// Calculates the byte size for *one* ternary plane (pos or neg) for one neighbor.
// Equivalent to ceil(dims / 64.0) * sizeof(uint64_t)
// Note: This uses the helper from ternary_quantization.hpp
idx_t GetTernaryPlaneSizeBytes(idx_t dimensions) {
    if (dimensions == 0) {
        throw InternalException("Cannot calculate plane size for 0 dimensions");
    }
    return WordsPerPlane(dimensions) * sizeof(uint64_t);
}


LMDiskannConfig ParseOptions(const case_insensitive_map_t<Value> &options) {
    LMDiskannConfig config; // Starts with default values

    // Override with user-provided options
    auto it = options.find(LmDiskannOptionKeys::METRIC);
    if (it != options.end()) {
        string metric_str = StringUtil::Upper(it->second.ToString());
        if (metric_str == "L2") {
            config.metric_type = LMDiskannMetricType::L2;
        } else if (metric_str == "COSINE") {
            config.metric_type = LMDiskannMetricType::COSINE;
        } else if (metric_str == "IP") {
            config.metric_type = LMDiskannMetricType::IP;
        } else {
            // Use base Exception
             throw Exception(ExceptionType::INVALID_INPUT, StringUtil::Format("Unsupported METRIC type '%s' for LM_DISKANN index. Supported types: L2, COSINE, IP", metric_str));
        }
    }

    it = options.find(LmDiskannOptionKeys::R);
    if (it != options.end()) {
        config.r = it->second.GetValue<uint32_t>();
    }

    it = options.find(LmDiskannOptionKeys::L_INSERT);
    if (it != options.end()) {
        config.l_insert = it->second.GetValue<uint32_t>();
    }

    it = options.find(LmDiskannOptionKeys::ALPHA);
    if (it != options.end()) {
        config.alpha = it->second.GetValue<float>();
    }

    it = options.find(LmDiskannOptionKeys::L_SEARCH);
    if (it != options.end()) {
        config.l_search = it->second.GetValue<uint32_t>();
    }

    // Dimensions and node_vector_type are typically set later based on column info
    // Validation is called separately after those are set.

    return config;
}


void ValidateParameters(const LMDiskannConfig &config) {
    // Use base Exception
    if (config.r == 0) throw Exception(ExceptionType::INVALID_INPUT, "LM_DISKANN parameter R must be > 0");
    if (config.l_insert == 0) throw Exception(ExceptionType::INVALID_INPUT, "LM_DISKANN parameter L_INSERT must be > 0");
    if (config.alpha < 1.0f) throw Exception(ExceptionType::INVALID_INPUT, "LM_DISKANN parameter ALPHA must be >= 1.0");
    if (config.l_search == 0) throw Exception(ExceptionType::INVALID_INPUT, "LM_DISKANN parameter L_SEARCH must be > 0");

    // Implicitly assume neighbors are TERNARY
    if (config.metric_type == LMDiskannMetricType::L2) {
         // Ternary dot product is not a good proxy for L2 distance
         throw Exception(ExceptionType::INVALID_INPUT, "LM_DISKANN with TERNARY neighbors is not compatible with METRIC 'L2'. Use COSINE or IP.");
    }
    if (config.dimensions == 0) {
         throw Exception(ExceptionType::INVALID_INPUT, "LM_DISKANN dimensions must be set and > 0 before validation");
    }
     if (config.node_vector_type == LMDiskannVectorType::UNKNOWN) {
         throw Exception(ExceptionType::INVALID_INPUT, "LM_DISKANN node_vector_type must be set before validation");
    }
    // Add more validation as needed
}


NodeLayoutOffsets CalculateLayoutInternal(const LMDiskannConfig &config) {
    NodeLayoutOffsets layout;
    idx_t current_offset = 0;

    // Make sure required config members are set
    if (config.dimensions == 0 || config.node_vector_type == LMDiskannVectorType::UNKNOWN) {
        throw InternalException("CalculateLayoutInternal requires dimensions and node_vector_type to be set in config");
    }

    idx_t node_vector_size_bytes = GetVectorTypeSizeBytes(config.node_vector_type) * config.dimensions;

    // Offset 0: Neighbor count (uint16_t)
    layout.neighbor_count_offset = OFFSET_NEIGHBOR_COUNT; // = 0
    current_offset = sizeof(uint16_t);
    // Add other fixed-size metadata here if needed

    // Align for node vector
    current_offset = AlignValue<idx_t, NODE_VECTOR_ALIGNMENT>(current_offset);
    layout.node_vector_offset = current_offset;
    current_offset += node_vector_size_bytes;

    // Align for neighbor IDs (row_t is usually 64-bit, likely aligned)
    current_offset = AlignValue<idx_t, sizeof(row_t)>(current_offset);
    layout.neighbor_ids_offset = current_offset;
    current_offset += config.r * sizeof(row_t);

    // Calculate size of one plane (pos or neg) for ONE neighbor
    idx_t plane_size_bytes_per_neighbor = GetTernaryPlaneSizeBytes(config.dimensions);

    // Align for positive planes array
    current_offset = AlignValue<idx_t, PLANE_ALIGNMENT>(current_offset);
    layout.neighbor_pos_planes_offset = current_offset;
    current_offset += config.r * plane_size_bytes_per_neighbor; // Total size for all positive planes

    // Align for negative planes array
    current_offset = AlignValue<idx_t, PLANE_ALIGNMENT>(current_offset);
    layout.neighbor_neg_planes_offset = current_offset;
    current_offset += config.r * plane_size_bytes_per_neighbor; // Total size for all negative planes

    layout.total_node_size = current_offset; // Size *before* final block alignment

    // Optional: Check against max block size (though this should ideally happen later)
    // Needs Storage::BLOCK_SIZE if uncommented
    // if (AlignValue<idx_t, Storage::BLOCK_SIZE>(layout.total_node_size) > DISKANN_MAX_BLOCK_SZ) {
    //     throw Exception(ExceptionType::INVALID_INPUT, StringUtil::Format("Calculated node block size exceeds maximum allowed (%llu bytes)", DISKANN_MAX_BLOCK_SZ));
    // }

    return layout;
}


} // namespace duckdb
