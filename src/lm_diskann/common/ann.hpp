#pragma once

#include "duckdb_types.hpp" // For common DuckDB types (after rename)

#include <string> // For potential future string usage

namespace diskann {
namespace common {

// --- Enums for LM-DiskANN Parameters --- //

/**
 * @brief Metric types supported by the index.
 * Corresponds to nDistanceFunc / VECTOR_METRIC_TYPE_PARAM_ID
 */
enum class LmDiskannMetricType : uint8_t {
	UNKNOWN = 0, // Default / Unset state
	L2 = 1,      // Euclidean distance
	COSINE = 2,  // Cosine similarity (often converted to distance: 1 - cosine)
	IP = 3,      // Inner product
	HAMMING = 4  // Hamming distance (Potentially for binary vectors)
};

/**
 * @brief Data types supported for the main node vectors.
 * Corresponds to nNodeVectorType / VECTOR_TYPE_PARAM_ID
 */
enum class LmDiskannVectorType : uint8_t {
	UNKNOWN = 0, // Default / Unset state
	FLOAT32 = 1, // 32-bit floating point
	INT8 = 2,    // 8-bit integer (requires quantization parameters potentially)
};

/**
 * @brief Converts a raw vector (e.g., int8) to a float32 vector.
 * @param raw_vector_data Pointer to the raw vector data.
 * @param float_vector_out Output buffer for the float32 vector. Must be pre-allocated.
 * @param dimensions Dimensionality of the vector.
 * @param raw_vector_type The data type of the raw vector (must be one of common::LmDiskannVectorType).
 * @return True if conversion was successful, false otherwise (e.g., unsupported type).
 */
inline bool ConvertRawVectorToFloat(const_data_ptr_t raw_vector_data, float *float_vector_out, idx_t dimensions,
                                    common::LmDiskannVectorType raw_vector_type) {
	if (!raw_vector_data || !float_vector_out) {
		return false;
	}

	switch (raw_vector_type) {
	case common::LmDiskannVectorType::FLOAT32:
		std::memcpy(float_vector_out, raw_vector_data, dimensions * sizeof(float));
		return true;
	case common::LmDiskannVectorType::INT8: {
		const int8_t *int8_data = reinterpret_cast<const int8_t *>(raw_vector_data);
		for (idx_t i = 0; i < dimensions; ++i) {
			float_vector_out[i] = static_cast<float>(int8_data[i]);
		}
		return true;
	}
	default: // Includes common::LmDiskannVectorType::UNKNOWN
		return false;
	}
}

/**
 * @brief Helper function to convert enum LmDiskannMetricType to string.
 * @param type The metric type enum.
 * @return String representation.
 */
inline const char *LmDiskannMetricTypeToString(common::LmDiskannMetricType type) {
	switch (type) {
	case common::LmDiskannMetricType::L2:
		return "L2";
	case common::LmDiskannMetricType::COSINE:
		return "COSINE";
	case common::LmDiskannMetricType::IP:
		return "IP";
	case common::LmDiskannMetricType::HAMMING:
		return "HAMMING";
	case common::LmDiskannMetricType::UNKNOWN:
	default:
		return "UNKNOWN_METRIC";
	}
}

/**
 * @brief Helper function to convert enum LmDiskannVectorType to string.
 * @param type The vector type enum.
 * @return String representation.
 */
inline const char *LmDiskannVectorTypeToString(common::LmDiskannVectorType type) {
	switch (type) {
	case common::LmDiskannVectorType::FLOAT32:
		return "FLOAT32";
	case common::LmDiskannVectorType::INT8:
		return "INT8";
	case common::LmDiskannVectorType::UNKNOWN:
		return "UNKNOWN_TYPE";
	default:
		return "INVALID_VECTOR_TYPE";
	}
}

} // namespace common
} // namespace diskann