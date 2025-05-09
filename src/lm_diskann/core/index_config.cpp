/**
 * @file index_config.cpp
 * @brief Implements functions for parsing, validating, and calculating
 * LM-DiskANN configuration.
 */
#include "index_config.hpp"

#include "../common/ann.hpp"
#include "../common/duckdb_types.hpp"
#include "ternary_quantization.hpp" // Needed for WordsPerPlane. Path reverted based on linter hint.

// #include "duckdb/parser/binder/binder_exception.hpp" // Removed: File not
// found
#include "duckdb/common/exception.hpp" // Using base exception
#include "duckdb/common/helper.hpp"    // For AlignValue
#include "duckdb/common/limits.hpp"    // For NumericLimits
#include "duckdb/common/string_util.hpp"
#include "duckdb/storage/storage_info.hpp" // For Storage::SECTOR_SIZE

#include <algorithm> // For std::min, std::max
#include <cmath>

namespace diskann {
namespace core {

// Constant definitions are now in index_config.hpp using inline constexpr

// --- Node Block Layout Constants (Keep definitions here if only used
// internally in cpp) --- Offset of the neighbor count field within the node
// block.
constexpr common::idx_t OFFSET_NEIGHBOR_COUNT = 0; // uint16_t
// Alignment requirement for node vectors within the block.
constexpr common::idx_t NODE_VECTOR_ALIGNMENT = 8;
// Alignment requirement for ternary plane arrays within the block.
constexpr common::idx_t PLANE_ALIGNMENT = 8;
constexpr common::idx_t DISKANN_MAX_BLOCK_SZ = 128 * 1024 * 1024; // Max allowed block size

// --- Configuration Functions --- //

common::idx_t GetVectorTypeSizeBytes(common::LmDiskannVectorType type) {
	switch (type) {
	case common::LmDiskannVectorType::FLOAT32:
		return sizeof(float);
	case common::LmDiskannVectorType::INT8:
		return sizeof(int8_t);
	case common::LmDiskannVectorType::UNKNOWN:
	default:
		throw ::duckdb::InternalException("Unsupported or UNKNOWN LmDiskannVectorType for size calculation");
	}
}

// Calculates the byte size for *one* ternary plane (pos or neg) for one
// neighbor. Equivalent to ceil(dims / 64.0) * sizeof(uint64_t) Note: This uses
// the helper from ternary_quantization.hpp
common::idx_t GetTernaryPlaneSizeBytes(common::idx_t dimensions) {
	if (dimensions == 0) {
		throw ::duckdb::InternalException("Cannot calculate plane size for 0 dimensions");
	}
	return WordsPerPlane(dimensions) * sizeof(uint64_t);
}

common::idx_t GetTernaryEdgeSizeBytes(common::idx_t dimensions) {
	return 2 * GetTernaryPlaneSizeBytes(dimensions);
}

void ValidateParameters(const LmDiskannConfig &config) {
	if (config.r == 0)
		throw ::duckdb::Exception(::duckdb::ExceptionType::INVALID_INPUT, "LM_DISKANN parameter R must be > 0");
	if (config.l_insert == 0)
		throw ::duckdb::Exception(::duckdb::ExceptionType::INVALID_INPUT, "LM_DISKANN parameter L_INSERT must be > 0");
	if (config.alpha < 1.0f)
		throw ::duckdb::Exception(::duckdb::ExceptionType::INVALID_INPUT, "LM_DISKANN parameter ALPHA must be >= 1.0");
	if (config.l_search == 0)
		throw ::duckdb::Exception(::duckdb::ExceptionType::INVALID_INPUT, "LM_DISKANN parameter L_SEARCH must be > 0");
	if (config.l_insert < config.r)
		throw ::duckdb::Exception(::duckdb::ExceptionType::INVALID_INPUT,
		                          ::duckdb::StringUtil::Format("LM_DISKANN L_INSERT (%u) must be >= R "
		                                                       "(%u)",
		                                                       config.l_insert, config.r));

	// Validate required parameters that are set later
	if (config.dimensions == 0) {
		throw ::duckdb::Exception(::duckdb::ExceptionType::INVALID_INPUT,
		                          "LM_DISKANN dimensions must be set (derived from column "
		                          "type) and > 0 before validation");
	}
	if (config.node_vector_type == common::LmDiskannVectorType::UNKNOWN) {
		throw ::duckdb::Exception(::duckdb::ExceptionType::INVALID_INPUT,
		                          "LM_DISKANN node_vector_type must be set (derived from "
		                          "column type) before validation");
	}

	// Implicitly assume neighbors are TERNARY
	// Ternary dot product is not a reliable proxy for L2 distance.
	if (config.metric_type == common::LmDiskannMetricType::L2) {
		throw ::duckdb::Exception(::duckdb::ExceptionType::INVALID_INPUT,
		                          "LM_DISKANN with implicit TERNARY neighbors is not "
		                          "compatible with METRIC 'L2'. Use COSINE or IP.");
	}

	// Add more validation as needed (e.g., max dimensions, max R?)
}

NodeLayoutOffsets CalculateLayoutInternal(const LmDiskannConfig &config) {
	NodeLayoutOffsets layout;
	common::idx_t current_offset = 0;

	// Ensure required config members needed for layout are set
	if (config.dimensions == 0 || config.node_vector_type == common::LmDiskannVectorType::UNKNOWN) {
		throw ::duckdb::InternalException("CalculateLayoutInternal requires dimensions and "
		                                  "node_vector_type to be set in config");
	}

	common::idx_t node_vector_size_bytes = GetVectorTypeSizeBytes(config.node_vector_type) * config.dimensions;

	// Offset 0: Neighbor count (uint16_t)
	layout.neighbor_count_offset = OFFSET_NEIGHBOR_COUNT; // = 0
	current_offset = sizeof(uint16_t);

	// Align for node vector
	current_offset = ::duckdb::AlignValue<common::idx_t, NODE_VECTOR_ALIGNMENT>(current_offset);
	layout.node_vector_offset = current_offset;
	current_offset += node_vector_size_bytes;

	// Align for neighbor IDs (row_t is usually 64-bit, likely already aligned but
	// enforce)
	current_offset = ::duckdb::AlignValue<common::idx_t, sizeof(::duckdb::row_t)>(current_offset);
	layout.neighbor_ids_offset = current_offset;
	current_offset += config.r * sizeof(::duckdb::row_t);

	// Calculate size of compressed TERNARY edge representation for ONE neighbor
	layout.ternary_edge_size_bytes = GetTernaryEdgeSizeBytes(config.dimensions);
	common::idx_t plane_size_bytes_per_neighbor = GetTernaryPlaneSizeBytes(config.dimensions);

	// Align for positive planes array
	current_offset = ::duckdb::AlignValue<common::idx_t, PLANE_ALIGNMENT>(current_offset);
	layout.neighbor_pos_planes_offset = current_offset;
	current_offset += config.r * plane_size_bytes_per_neighbor; // Total size for all positive planes

	// Align for negative planes array
	current_offset = ::duckdb::AlignValue<common::idx_t, PLANE_ALIGNMENT>(current_offset);
	layout.neighbor_neg_planes_offset = current_offset;
	current_offset += config.r * plane_size_bytes_per_neighbor; // Total size for all negative planes

	layout.total_node_size = current_offset; // Size *before* final block alignment

	return layout;
}

// --- Utility Functions --- //

} // namespace core
} // namespace diskann