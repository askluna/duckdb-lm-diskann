/**
 * @file distance.hpp
 * @brief Defines functions for calculating vector distances and similarities.
 */
#pragma once

#include "../common/ann.hpp"
#include "../common/duckdb_types.hpp"
#include "index_config.hpp" // Include config for TernaryPlanesView and LmDiskannMetricType

#include <cstdint>
#include <type_traits> // Added for std::is_same_v
#include <vector>      // Added for std::vector

// Forward declarations no longer needed as index_config.hpp is included
// namespace duckdb {
//     enum class LmDiskannMetricType : uint8_t;
// }

namespace diskann {
namespace core {

// Forward declare config struct if not fully included
// struct LmDiskannConfig;

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
float ComputeExactDistanceFloat(const float *a_ptr, const float *b_ptr, common::idx_t dimensions,
                                common::LmDiskannMetricType metric_type);

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
float ComputeApproxSimilarityTernary(const float *query_float_ptr, const TernaryPlanesView &neighbor_planes,
                                     common::idx_t dimensions);

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
float CalculateDistance(const T_QUERY *query_ptr, const T_NODE *node_vector_ptr, const LmDiskannConfig &config);

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
float CalculateApproxDistance(const float *query_ptr, ::duckdb::const_data_ptr_t compressed_neighbor_ptr,
                              const LmDiskannConfig &config);

/**
 * @brief Compresses a vector into the TERNARY format for edge storage.
 * @param input_vector Pointer to the input vector data (float).
 * @param output_compressed_vector Pointer to the output buffer for the
 * compressed data.
 * @param config The index configuration (used for dimensions).
 * @return True if compression was successful, false otherwise.
 */
bool CompressVectorForEdge(const float *input_vector, ::duckdb::data_ptr_t output_compressed_vector,
                           const LmDiskannConfig &config);

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

// Explicit instantiation for int8_t needed if definition is in .cpp
// extern template void ConvertToFloat<int8_t>(const int8_t* input_vector,
// float* output_vector, idx_t dimensions);

} // namespace core
} // namespace diskann
