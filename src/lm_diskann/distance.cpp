#include "distance.hpp"
#include "config.hpp" // Needed for LMDiskannMetricType enum definition
#include "ternary_quantization.hpp" // Needed for Approx Similarity: WordsPerPlane, EncodeTernary, GetKernel

// #include "ternary_quantization.hpp" // No longer needed here

#include "duckdb/common/exception.hpp"
#include "duckdb/common/limits.hpp" // For NumericLimits
// #include "duckdb/common/types/vector.hpp" // Not needed for current implementation
// #include "duckdb/common/types/value.hpp" // Not needed
// #include "duckdb/common/string_util.hpp" // Not needed

#include <cmath> // For std::sqrt, std::fabs, std::max, std::min
#include <vector> // For temporary query plane buffers in Approx Similarity
#include <stdexcept> // For runtime_error in Approx Similarity

namespace duckdb {

// --- Core Function Implementations --- //

float ComputeExactDistanceFloat(const float *a_ptr, const float *b_ptr, idx_t dimensions, LMDiskannMetricType metric_type) {
    // TODO: Ideally, use optimized DuckDB functions if possible.
    // Placeholder manual implementation.

    switch (metric_type) {
        case LMDiskannMetricType::L2: {
            float distance = 0.0f;
            for (idx_t i = 0; i < dimensions; ++i) {
                float diff = a_ptr[i] - b_ptr[i];
                distance += diff * diff;
            }
            // Check for potential negative distance due to floating point errors if squaring negative near-zero diffs
            if (distance < 0.0f) distance = 0.0f;
            return std::sqrt(distance);
        }
        case LMDiskannMetricType::IP: { // Inner Product Distance = -IP
            float dot_product = 0.0f;
            for (idx_t i = 0; i < dimensions; ++i) {
                dot_product += a_ptr[i] * b_ptr[i];
            }
            return -dot_product; // Return negative IP as distance
        }
        case LMDiskannMetricType::COSINE: { // Cosine Distance = 1 - Cosine Similarity
            float dot_product = 0.0f;
            float norm_a_sq = 0.0f;
            float norm_b_sq = 0.0f;
            for (idx_t i = 0; i < dimensions; ++i) {
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
            // Avoid division by zero in the unlikely case norms are still zero after check
            if (norm_a == 0.0f || norm_b == 0.0f) { 
                return 1.0f;
            }
            float cosine_similarity = dot_product / (norm_a * norm_b);
            // Clamp similarity to [-1, 1] due to potential floating point inaccuracies
            cosine_similarity = std::max(-1.0f, std::min(1.0f, cosine_similarity));
            return 1.0f - cosine_similarity;
        }
        default: {
            throw Exception(ExceptionType::INVALID_INPUT, "ComputeExactDistanceFloat: Unsupported metric type");
        }
    }
}

//! Computes the approximate SIMILARITY between a full query vector (float)
//! and a compressed TERNARY neighbor vector using its separate planes.
//! Returns a raw similarity score (higher is better). Used during graph traversal.
float ComputeApproxSimilarityTernary(const float *query_float_ptr,
                                     const TernaryPlanesView& neighbor_planes, // Use struct view
                                     idx_t dimensions) {

    // Validate input struct and dimensions
    if (!query_float_ptr || !neighbor_planes.IsValid() || dimensions == 0) {
        throw std::runtime_error("Invalid arguments for ComputeApproxSimilarityTernary");
    }
    // Optional: Check if dimensions matches plane size? words_per_plane should correspond to dimensions.
    // size_t expected_words = WordsPerPlane(dimensions);
    // if (expected_words != neighbor_planes.words_per_plane) { ... throw ... }

    // 1. Prepare query planes
    const size_t words_per_vector = neighbor_planes.words_per_plane; // Use size from view
    std::vector<uint64_t> query_pos_plane_vec(words_per_vector);
    std::vector<uint64_t> query_neg_plane_vec(words_per_vector);

    EncodeTernary(query_float_ptr, query_pos_plane_vec.data(), query_neg_plane_vec.data(), dimensions);

    // 2. Get the appropriate SIMD kernel
    dot_fun_t dot_kernel = GetDotKernel();

    // 3. Compute the raw ternary dot product score
    // Extract raw pointers from the view struct
    int64_t raw_score = dot_kernel(
        query_pos_plane_vec.data(),
        query_neg_plane_vec.data(),
        reinterpret_cast<const uint64_t*>(neighbor_planes.positive_plane),
        reinterpret_cast<const uint64_t*>(neighbor_planes.negative_plane),
        words_per_vector
    );

    // 4. Return the raw score as float (higher is better similarity)
    return static_cast<float>(raw_score);
}

} // namespace duckdb
