/**
 * @file config.cpp
 * @brief Implements functions for parsing, validating, and calculating LM-DiskANN configuration.
 */
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
// Offset of the neighbor count field within the node block.
constexpr idx_t OFFSET_NEIGHBOR_COUNT = 0; // uint16_t
// Alignment requirement for node vectors within the block.
constexpr idx_t NODE_VECTOR_ALIGNMENT = 8;
// Alignment requirement for ternary plane arrays within the block.
constexpr idx_t PLANE_ALIGNMENT = 8;
constexpr idx_t DISKANN_MAX_BLOCK_SZ = 128 * 1024 * 1024; // Max allowed block size


// --- Configuration Functions --- //

idx_t GetVectorTypeSizeBytes(LMDiskannVectorType type) {
    switch (type) {
    case LMDiskannVectorType::FLOAT32:
        return sizeof(float);
    case LMDiskannVectorType::INT8:
        return sizeof(int8_t);
    case LMDiskannVectorType::UNKNOWN:
    default:
        throw InternalException("Unsupported or UNKNOWN LMDiskannVectorType for size calculation");
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

idx_t GetTernaryEdgeSizeBytes(idx_t dimensions) {
    return 2 * GetTernaryPlaneSizeBytes(dimensions);
}

LMDiskannConfig ParseOptions(const case_insensitive_map_t<Value> &options) {
    LMDiskannConfig config; // Starts with default values

    for (const auto &entry : options) {
        const string &key_upper = StringUtil::Upper(entry.first);
        const Value &val = entry.second;

        if (key_upper == LmDiskannOptionKeys::METRIC) {
            string metric_str = StringUtil::Upper(val.ToString());
            if (metric_str == "L2") {
                config.metric_type = LMDiskannMetricType::L2;
            } else if (metric_str == "COSINE") {
                config.metric_type = LMDiskannMetricType::COSINE;
            } else if (metric_str == "IP") {
                config.metric_type = LMDiskannMetricType::IP;
            } else {
                throw Exception(ExceptionType::INVALID_INPUT, StringUtil::Format("Unsupported METRIC type '%s' for LM_DISKANN index. Supported types: L2, COSINE, IP", metric_str));
            }
        } else if (key_upper == LmDiskannOptionKeys::R) {
            config.r = val.GetValue<uint32_t>();
        } else if (key_upper == LmDiskannOptionKeys::L_INSERT) {
            config.l_insert = val.GetValue<uint32_t>();
        } else if (key_upper == LmDiskannOptionKeys::ALPHA) {
            config.alpha = val.GetValue<float>();
        } else if (key_upper == LmDiskannOptionKeys::L_SEARCH) {
            config.l_search = val.GetValue<uint32_t>();
        } else {
             // Error on unknown option - helps catch typos
             throw Exception(ExceptionType::INVALID_INPUT, StringUtil::Format("Unknown option '%s' for LM_DISKANN index. Allowed options: METRIC, R, L_INSERT, ALPHA, L_SEARCH", entry.first));
        }
        // NODE_TYPE and DIMENSIONS are derived from the column type later
        // EDGE_TYPE is implicitly TERNARY
    }

    return config;
}


void ValidateParameters(const LMDiskannConfig &config) {
    if (config.r == 0) throw Exception(ExceptionType::INVALID_INPUT, "LM_DISKANN parameter R must be > 0");
    if (config.l_insert == 0) throw Exception(ExceptionType::INVALID_INPUT, "LM_DISKANN parameter L_INSERT must be > 0");
    if (config.alpha < 1.0f) throw Exception(ExceptionType::INVALID_INPUT, "LM_DISKANN parameter ALPHA must be >= 1.0");
    if (config.l_search == 0) throw Exception(ExceptionType::INVALID_INPUT, "LM_DISKANN parameter L_SEARCH must be > 0");
    if (config.l_insert < config.r) throw Exception(ExceptionType::INVALID_INPUT, StringUtil::Format("LM_DISKANN L_INSERT (%u) must be >= R (%u)", config.l_insert, config.r));

    // Validate required parameters that are set later
    if (config.dimensions == 0) {
         throw Exception(ExceptionType::INVALID_INPUT, "LM_DISKANN dimensions must be set (derived from column type) and > 0 before validation");
    }
    if (config.node_vector_type == LMDiskannVectorType::UNKNOWN) {
         throw Exception(ExceptionType::INVALID_INPUT, "LM_DISKANN node_vector_type must be set (derived from column type) before validation");
    }

    // Implicitly assume neighbors are TERNARY
    // Ternary dot product is not a reliable proxy for L2 distance.
    if (config.metric_type == LMDiskannMetricType::L2) {
         throw Exception(ExceptionType::INVALID_INPUT, "LM_DISKANN with implicit TERNARY neighbors is not compatible with METRIC 'L2'. Use COSINE or IP.");
    }

    // Add more validation as needed (e.g., max dimensions, max R?)
}


NodeLayoutOffsets CalculateLayoutInternal(const LMDiskannConfig &config) {
    NodeLayoutOffsets layout;
    idx_t current_offset = 0;

    // Ensure required config members needed for layout are set
    if (config.dimensions == 0 || config.node_vector_type == LMDiskannVectorType::UNKNOWN) {
        throw InternalException("CalculateLayoutInternal requires dimensions and node_vector_type to be set in config");
    }

    idx_t node_vector_size_bytes = GetVectorTypeSizeBytes(config.node_vector_type) * config.dimensions;

    // Offset 0: Neighbor count (uint16_t)
    layout.neighbor_count_offset = OFFSET_NEIGHBOR_COUNT; // = 0
    current_offset = sizeof(uint16_t);

    // Align for node vector
    current_offset = AlignValue<idx_t, NODE_VECTOR_ALIGNMENT>(current_offset);
    layout.node_vector_offset = current_offset;
    current_offset += node_vector_size_bytes;

    // Align for neighbor IDs (row_t is usually 64-bit, likely already aligned but enforce)
    current_offset = AlignValue<idx_t, sizeof(row_t)>(current_offset);
    layout.neighbor_ids_offset = current_offset;
    current_offset += config.r * sizeof(row_t);

    // Calculate size of compressed TERNARY edge representation for ONE neighbor
    layout.ternary_edge_size_bytes = GetTernaryEdgeSizeBytes(config.dimensions);
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

    return layout;
}

// --- Utility Functions --- //

const char* LMDiskannMetricTypeToString(LMDiskannMetricType type) {
	switch (type) {
	case LMDiskannMetricType::L2: return "L2";
	case LMDiskannMetricType::COSINE: return "COSINE";
	case LMDiskannMetricType::IP: return "IP";
	default: return "UNKNOWN_METRIC";
	}
}

const char* LMDiskannVectorTypeToString(LMDiskannVectorType type) {
	switch (type) {
	case LMDiskannVectorType::FLOAT32: return "FLOAT32";
	case LMDiskannVectorType::INT8: return "INT8";
	case LMDiskannVectorType::UNKNOWN: return "UNKNOWN_TYPE";
	default: return "INVALID_TYPE";
	}
}

} // namespace duckdb
