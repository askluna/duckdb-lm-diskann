/**
 * @file distance.hpp
 * @brief Defines functions for calculating vector distances and similarities.
 */
#pragma once

#include "../common/ann.hpp"
#include "../common/duckdb_types.hpp"
#include "index_config.hpp"         // Include config for TernaryPlanesView and LmDiskannMetricType
#include "ternary_quantization.hpp" // Needed for Approx Similarity, CompressVectorForEdge

#include <cmath> // For std::sqrt, std::fabs, std::max, std::min in ComputeExactDistanceFloat
#include <cstdint>
#include <stdexcept>   // For runtime_error in ComputeApproxSimilarityTernary
#include <type_traits> // Added for std::is_same_v
#include <vector>      // Added for std::vector

namespace diskann {
namespace core {

/**
 * @brief Converts a vector of a specific type (e.g., int8_t) to float.
 * @tparam T Type of the input vector elements.
 * @param input_vector Pointer to the input vector data.
 * @param output_vector Pointer to the output float vector buffer.
 * @param dimensions The number of dimensions.
 */
template <typename T>
void ConvertToFloat(const T *input_vector, float *output_vector, common::idx_t dimensions) {
	if (!input_vector || !output_vector) {
		// Consider: throw std::invalid_argument("Null pointer provided to ConvertToFloat");
		return; // Or handle error as per project policy
	}
	for (common::idx_t i = 0; i < dimensions; ++i) {
		output_vector[i] = static_cast<float>(input_vector[i]);
	}
}

// --- Exact Distance Calculation --- //

/**
 * @brief Computes the EXACT distance between two full-precision FLOAT vectors.
 * @details Used for final re-ranking in LM-DiskANN search.
 * @param a_ptr Pointer to the first float vector.
 * @param b_ptr Pointer to the second float vector.
 * @param dimensions Dimensionality of the vectors.
 * @param metric_type The metric (L2, COSINE, IP) to use.
 * @return The calculated distance.
 */
inline float ComputeExactDistanceFloat(const float *a_ptr, const float *b_ptr, common::idx_t dimensions,
                                       common::LmDiskannMetricType metric_type) {
	// TODO: Ideally, use optimized DuckDB functions if possible.
	// Placeholder manual implementation.

	switch (metric_type) {
	case common::LmDiskannMetricType::L2: {
		float distance = 0.0f;
		for (common::idx_t i = 0; i < dimensions; ++i) {
			float diff = a_ptr[i] - b_ptr[i];
			distance += diff * diff;
		}
		// Check for potential negative distance due to floating point errors if
		// squaring negative near-zero diffs
		if (distance < 0.0f)
			distance = 0.0f;
		return std::sqrt(distance);
	}
	case common::LmDiskannMetricType::IP: { // Inner Product Distance = -IP
		float dot_product = 0.0f;
		for (common::idx_t i = 0; i < dimensions; ++i) {
			dot_product += a_ptr[i] * b_ptr[i];
		}
		return -dot_product; // Return negative IP as distance
	}
	case common::LmDiskannMetricType::COSINE: { // Cosine Distance = 1 - Cosine Similarity
		float dot_product = 0.0f;
		float norm_a_sq = 0.0f;
		float norm_b_sq = 0.0f;
		for (common::idx_t i = 0; i < dimensions; ++i) {
			dot_product += a_ptr[i] * b_ptr[i];
			norm_a_sq += a_ptr[i] * a_ptr[i];
			norm_b_sq += b_ptr[i] * b_ptr[i];
		}
		// Handle zero vectors or near-zero vectors to avoid division by zero or NaN
		if (norm_a_sq <= 0.0f || norm_b_sq <= 0.0f) {
			return 1.0f; // Max distance for cosine
		}
		float norm_a = std::sqrt(norm_a_sq);
		float norm_b = std::sqrt(norm_b_sq);
		// Avoid division by zero in the unlikely case norms are still zero after
		// check
		if (norm_a == 0.0f || norm_b == 0.0f) {
			return 1.0f;
		}
		float cosine_similarity = dot_product / (norm_a * norm_b);
		// Clamp similarity to [-1, 1] due to potential floating point inaccuracies
		cosine_similarity = std::max(-1.0f, std::min(1.0f, cosine_similarity));
		return 1.0f - cosine_similarity;
	}
	default: {
		throw ::duckdb::Exception(::duckdb::ExceptionType::INVALID_INPUT,
		                          "ComputeExactDistanceFloat: Unsupported metric type");
	}
	}
}

// --- Distance / Similarity Calculation --- //

/**
 * @brief Computes the approximate SIMILARITY between a full query vector
 * (float) and a compressed TERNARY neighbor vector using its separate planes.
 * @details Returns a raw similarity score (higher is better). Used during graph
 * traversal. Implementation requires ternary_quantization.hpp.
 * @param query_float_ptr Pointer to the query float vector.
 * @param neighbor_planes A view containing pointers to the neighbor's ternary
 * planes and dimensions.
 * @param dimensions Dimensionality of the vectors (must match
 * neighbor_planes.dimensions).
 * @return The calculated similarity score.
 */
inline float ComputeApproxSimilarityTernary(const float *query_float_ptr,
                                            const TernaryPlanesView &neighbor_planes, // Use struct view
                                            common::idx_t dimensions) {

	// Validate input struct and dimensions
	if (!query_float_ptr || !neighbor_planes.IsValid() || dimensions == 0) {
		// Using std::runtime_error as in original .cpp, ensure <stdexcept> is included
		throw std::runtime_error("Invalid arguments for ComputeApproxSimilarityTernary");
	}
	// Optional: Check if dimensions matches plane size? words_per_plane should
	// correspond to dimensions. size_t expected_words =
	// WordsPerPlane(dimensions); if (expected_words !=
	// neighbor_planes.words_per_plane) { ... throw ... }

	// 1. Prepare query planes
	const size_t words_per_vector = neighbor_planes.words_per_plane; // Use size from view
	std::vector<uint64_t> query_pos_plane_vec(words_per_vector);
	std::vector<uint64_t> query_neg_plane_vec(words_per_vector);

	EncodeTernary(query_float_ptr, query_pos_plane_vec.data(), query_neg_plane_vec.data(), dimensions);

	// 2. Get the appropriate SIMD kernel
	dot_fun_t dot_kernel = GetDotKernel();

	// 3. Compute the raw ternary dot product score
	// Extract raw pointers from the view struct
	int64_t raw_score = dot_kernel(query_pos_plane_vec.data(), query_neg_plane_vec.data(),
	                               reinterpret_cast<const uint64_t *>(neighbor_planes.positive_plane),
	                               reinterpret_cast<const uint64_t *>(neighbor_planes.negative_plane), words_per_vector);

	// 4. Return the raw score as float (higher is better similarity)
	return static_cast<float>(raw_score);
}

/**
 * @brief Computes the exact distance between two vectors based on the
 * configured metric.
 * @tparam T_QUERY Type of the query vector elements (likely float).
 * @tparam T_NODE Type of the node vector elements (e.g., float, int8_t).
 * @param query_ptr Pointer to the query vector data.
 * @param node_vector_ptr Pointer to the node vector data.
 * @param config The index configuration containing dimensions and metric type.
 * @return The calculated distance.
 */
template <typename T_QUERY, typename T_NODE>
float CalculateDistance(const T_QUERY *query_ptr, const T_NODE *node_vector_ptr, const LmDiskannConfig &config) {
	if (!query_ptr || !node_vector_ptr) {
		throw ::duckdb::InvalidInputException("Null pointer passed to CalculateDistance");
	}
	if (config.dimensions == 0) {
		throw ::duckdb::InvalidInputException("Dimensions cannot be zero in CalculateDistance");
	}

	const float *query_float_ptr_actual = nullptr;
	std::vector<float> temp_query_float_vector;

	if constexpr (std::is_same_v<T_QUERY, float>) {
		query_float_ptr_actual = reinterpret_cast<const float *>(query_ptr);
	} else {
		temp_query_float_vector.resize(config.dimensions);
		ConvertToFloat<T_QUERY>(query_ptr, temp_query_float_vector.data(), config.dimensions);
		query_float_ptr_actual = temp_query_float_vector.data();
	}

	const float *node_float_ptr_actual = nullptr;
	std::vector<float> temp_node_float_vector;

	if constexpr (std::is_same_v<T_NODE, float>) {
		node_float_ptr_actual = reinterpret_cast<const float *>(node_vector_ptr);
	} else {
		temp_node_float_vector.resize(config.dimensions);
		ConvertToFloat<T_NODE>(node_vector_ptr, temp_node_float_vector.data(), config.dimensions);
		node_float_ptr_actual = temp_node_float_vector.data();
	}

	return ComputeExactDistanceFloat(query_float_ptr_actual, node_float_ptr_actual, config.dimensions,
	                                 config.metric_type);
}

/**
 * @brief Computes the approximate distance using the query vector and a
 * compressed neighbor vector.
 * @details Assumes the compressed neighbor is in TERNARY format.
 *          Internally calls ComputeApproxSimilarityTernary and converts
 * similarity to distance if needed.
 * @param query_ptr Pointer to the query vector data (float).
 * @param compressed_neighbor_ptr Pointer to the start of the compressed
 * neighbor data (ternary planes).
 * @param config The index configuration containing dimensions and metric type.
 * @return The calculated approximate distance.
 */
inline float CalculateApproxDistance(const float *query_ptr, ::duckdb::const_data_ptr_t compressed_neighbor_ptr,
                                     const LmDiskannConfig &config) {
	if (!query_ptr || !compressed_neighbor_ptr) {
		throw ::duckdb::InvalidInputException("Null pointer passed to CalculateApproxDistance");
	}
	if (config.dimensions == 0) {
		throw ::duckdb::InvalidInputException("Dimensions cannot be zero in CalculateApproxDistance");
	}

	const size_t words_per_plane = WordsPerPlane(config.dimensions);

	TernaryPlanesView neighbor_planes_view;
	neighbor_planes_view.positive_plane = compressed_neighbor_ptr;
	neighbor_planes_view.negative_plane = compressed_neighbor_ptr + (words_per_plane * sizeof(uint64_t));
	neighbor_planes_view.dimensions = config.dimensions;
	neighbor_planes_view.words_per_plane = words_per_plane;

	float similarity = ComputeApproxSimilarityTernary(query_ptr, neighbor_planes_view, config.dimensions);

	switch (config.metric_type) {
	case common::LmDiskannMetricType::IP:
		return -similarity;
	case common::LmDiskannMetricType::COSINE:
		return 1.0f - similarity;
	case common::LmDiskannMetricType::L2:
		throw ::duckdb::InvalidInputException(
		    "L2 metric is not directly compatible with the current ternary "
		    "CalculateApproxDistance. Ternary approximation is for IP/Cosine-like similarities.");
	default:
		throw ::duckdb::InvalidInputException("Unsupported metric type in CalculateApproxDistance");
	}
}

/**
 * @brief Compresses a vector into the TERNARY format for edge storage.
 * @param input_vector Pointer to the input vector data (float).
 * @param output_compressed_vector Pointer to the output buffer for the
 * compressed data.
 * @param config The index configuration (used for dimensions).
 * @return True if compression was successful, false otherwise.
 */
inline bool CompressVectorForEdge(const float *input_vector, ::duckdb::data_ptr_t output_compressed_vector,
                                  const LmDiskannConfig &config) {
	if (!input_vector || !output_compressed_vector) {
		return false;
	}
	if (config.dimensions == 0) {
		return false;
	}

	const size_t words_per_plane = WordsPerPlane(config.dimensions);

	uint64_t *pos_plane_ptr = reinterpret_cast<uint64_t *>(output_compressed_vector);
	uint64_t *neg_plane_ptr = reinterpret_cast<uint64_t *>(output_compressed_vector + words_per_plane * sizeof(uint64_t));

	EncodeTernary<float>(input_vector, pos_plane_ptr, neg_plane_ptr, config.dimensions);

	return true;
}

} // namespace core
} // namespace diskann
