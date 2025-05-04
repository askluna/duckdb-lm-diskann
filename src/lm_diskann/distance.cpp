#include "distance.hpp"
#include "config.hpp" // Include config for enums
#include "ternary_quantization.hpp" // For ternary encoding and kernels

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/vector.hpp" // For vector access if needed
#include "duckdb/function/scalar/vector_functions.hpp" // For VectorDistance
#include "duckdb/common/types/vector_operations/vector_operations.hpp" // For vector operations if needed directly
#include "duckdb/common/types/value.hpp" // Might define float16_t or related types
#include "duckdb/common/types/uhugeint.hpp" // Includes half.hpp potentially

#include <vector>
#include <cmath>
#include <cstring> // For memcpy
#include <type_traits> // For std::is_same

namespace duckdb {

// Helper to potentially convert vector data to float for calculation
template <typename T>
void ConvertToFloat(const T *src, float *dst, idx_t count) {
    if constexpr (std::is_same<T, float>::value) {
        if (src != dst) { // Avoid self-copy
             memcpy(dst, src, count * sizeof(float));
        }
    } else if constexpr (std::is_same<T, float16_t>::value) {
        for (idx_t i = 0; i < count; ++i) {
            dst[i] = float16_t::ToFloat(src[i]);
        }
    } else if constexpr (std::is_same<T, int8_t>::value) {
        // FIXME: This simple cast/scaling is likely inaccurate.
        // Proper INT8 distance needs scale/offset or specialized kernels.
        // Using this for now as a placeholder.
        for (idx_t i = 0; i < count; ++i) {
            // Example: Scale to [-1, 1] range approximately
            dst[i] = static_cast<float>(src[i]) / 128.0f;
        }
    } else {
        throw NotImplementedException("Unsupported type for ConvertToFloat");
    }
}
// Explicit instantiations
template void ConvertToFloat<float>(const float*, float*, idx_t);
// template void ConvertToFloat<float16_t>(const float16_t*, float*, idx_t);
template void ConvertToFloat<int8_t>(const int8_t*, float*, idx_t);


// Calculates distance between two vectors (potentially different types).
float ComputeDistance(const_data_ptr_t vec_a_ptr, LMDiskannVectorType type_a,
                      const_data_ptr_t vec_b_ptr, LMDiskannVectorType type_b,
                      idx_t dimensions, LMDiskannMetricType metric_type) {

    // Convert both vectors to float for using DuckDB's VectorDistance
    // This is inefficient for non-float types but provides a starting point.
    std::vector<float> vec_a_float(dimensions);
    std::vector<float> vec_b_float(dimensions);

    // Convert A
    switch(type_a) {
        case LMDiskannVectorType::FLOAT32:
            ConvertToFloat(reinterpret_cast<const float*>(vec_a_ptr), vec_a_float.data(), dimensions);
            break;
        case LMDiskannVectorType::FLOAT16:
             // ConvertToFloat(reinterpret_cast<const float16_t*>(vec_a_ptr), vec_a_float.data(), dimensions);
             throw NotImplementedException("FLOAT16 distance support pending float16_t type resolution");
            break;
        case LMDiskannVectorType::INT8:
            ConvertToFloat(reinterpret_cast<const int8_t*>(vec_a_ptr), vec_a_float.data(), dimensions);
            break;
        default: throw InternalException("Unsupported type_a in ComputeDistance");
    }

    // Convert B
    switch(type_b) {
        case LMDiskannVectorType::FLOAT32:
            ConvertToFloat(reinterpret_cast<const float*>(vec_b_ptr), vec_b_float.data(), dimensions);
            break;
        case LMDiskannVectorType::FLOAT16:
            // ConvertToFloat(reinterpret_cast<const float16_t*>(vec_b_ptr), vec_b_float.data(), dimensions);
            throw NotImplementedException("FLOAT16 distance support pending float16_t type resolution");
            break;
        case LMDiskannVectorType::INT8:
            ConvertToFloat(reinterpret_cast<const int8_t*>(vec_b_ptr), vec_b_float.data(), dimensions);
            break;
        default: throw InternalException("Unsupported type_b in ComputeDistance");
    }

    const float *a_ptr = vec_a_float.data();
    const float *b_ptr = vec_b_float.data();

    // Calculate distance based on metric
    switch (metric_type) {
        case LMDiskannMetricType::L2:
             // Returns squared L2 distance
             return VectorDistance::Exec<float, float, float>(a_ptr, b_ptr, dimensions, VectorDistanceType::L2);
        case LMDiskannMetricType::COSINE:
             // Returns 1 - cosine_similarity. Lower is better.
             return VectorDistance::Exec<float, float, float>(a_ptr, b_ptr, dimensions, VectorDistanceType::COSINE);
        case LMDiskannMetricType::IP:
             // Returns -inner_product. Lower is better.
             return -VectorDistance::Exec<float, float, float>(a_ptr, b_ptr, dimensions, VectorDistanceType::IP);
        default:
             throw InternalException("Unknown metric type in ComputeDistance");
    }
}


// Calculates approximate distance between a full query vector (float)
// and a compressed neighbor vector.
float ComputeApproxDistance(const float *query_ptr, const_data_ptr_t compressed_neighbor_ptr,
                            idx_t dimensions, LMDiskannMetricType metric_type,
                            LMDiskannVectorType resolved_edge_vector_type) {

    // Dispatch based on the *resolved* edge type stored in the index
    switch(resolved_edge_vector_type) {
        case LMDiskannVectorType::FLOAT32:
            return ComputeDistance(const_data_ptr_cast(query_ptr), LMDiskannVectorType::FLOAT32,
                                   compressed_neighbor_ptr, LMDiskannVectorType::FLOAT32,
                                   dimensions, metric_type);
        case LMDiskannVectorType::FLOAT16:
            return ComputeDistance(const_data_ptr_cast(query_ptr), LMDiskannVectorType::FLOAT32,
                                   compressed_neighbor_ptr, LMDiskannVectorType::FLOAT16,
                                   dimensions, metric_type);
        case LMDiskannVectorType::INT8:
            // Warning: Accuracy depends heavily on how INT8 was created and how distance is calculated.
            return ComputeDistance(const_data_ptr_cast(query_ptr), LMDiskannVectorType::FLOAT32,
                                   compressed_neighbor_ptr, LMDiskannVectorType::INT8,
                                   dimensions, metric_type);
        case LMDiskannVectorType::UNKNOWN: // Represents FLOAT1BIT
            if (metric_type == LMDiskannMetricType::COSINE) {
                 // FIXME: Implement Hamming distance / popcount for FLOAT1BIT Cosine approximation
                 // Calculate Hamming distance between query (needs binarization) and compressed_neighbor_ptr
                 // Convert Hamming distance to approximate Cosine distance.
                 throw NotImplementedException("Approximate distance for FLOAT1BIT not implemented.");
            } else {
                 throw InternalException("FLOAT1BIT edge type used with non-COSINE metric.");
            }
            break;
        default:
             throw InternalException("Unsupported resolved edge vector type in ComputeApproxDistance.");
    }
}


// Calculates approximate SIMILARITY between a full query vector (float)
// and a compressed TERNARY neighbor vector using its separate planes.
// Returns a raw similarity score (higher is better).
float ComputeApproxSimilarityTernary(const float *query_ptr,
                                     const_data_ptr_t pos_plane_ptr,
                                     const_data_ptr_t neg_plane_ptr,
                                     idx_t dimensions, LMDiskannMetricType metric_type) {

    // Metric validation (Ternary is only valid for COSINE or IP)
    if (metric_type == LMDiskannMetricType::L2) {
        throw InternalException("ComputeApproxSimilarityTernary called with L2 metric.");
    }

    // 1. Calculate size and allocate query planes
    const size_t words_per_plane = WordsPerPlane(dimensions);
    std::vector<uint64_t> query_pos_plane_vec(words_per_plane);
    std::vector<uint64_t> query_neg_plane_vec(words_per_plane);

    // 2. Encode the float query into ternary planes
    EncodeTernary(query_ptr, query_pos_plane_vec.data(), query_neg_plane_vec.data(), dimensions);

    // 3. Get the best available dot product kernel
    dot_fun_t dot_kernel = GetKernel();

    // 4. Call the kernel with query planes and the neighbor's stored planes
    // Note: Need const_cast or reinterpret_cast if kernel expects non-const uint64_t*
    int64_t raw_score = dot_kernel(
        query_pos_plane_vec.data(),
        query_neg_plane_vec.data(),
        reinterpret_cast<const uint64_t*>(pos_plane_ptr),
        reinterpret_cast<const uint64_t*>(neg_plane_ptr),
        words_per_plane
    );

    // 5. Return the raw similarity score
    // Higher score indicates higher similarity (closer vectors for IP/Cosine)
    // The caller needs to handle this (e.g., negate for min-heap, flip comparisons).
    return static_cast<float>(raw_score);
}


// Compresses a float vector into the TERNARY format using separate pos/neg planes.
void CompressVectorToTernary(const float* input_float_ptr,
                             data_ptr_t dest_pos_plane_ptr,
                             data_ptr_t dest_neg_plane_ptr,
                             idx_t dimensions) {

     // Directly call the encoding function from ternary_quantization.hpp
     EncodeTernary(
         input_float_ptr,
         reinterpret_cast<uint64_t*>(dest_pos_plane_ptr),
         reinterpret_cast<uint64_t*>(dest_neg_plane_ptr),
         dimensions
     );
}


} // namespace duckdb
