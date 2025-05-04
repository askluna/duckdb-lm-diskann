
#include "distance.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/vector.hpp" // For vector access if needed

#include <vector>
#include <cmath>
#include <cstring> // For memcpy

namespace duckdb {

// Helper to potentially convert vector data to float for calculation
template <typename T>
void ConvertToFloat(const T *src, float *dst, idx_t count) {
    if constexpr (std::is_same_v<T, float>) {
        if (src != dst) { // Avoid self-copy
             memcpy(dst, src, count * sizeof(float));
        }
    } else if constexpr (std::is_same_v<T, float16_t>) {
        for (idx_t i = 0; i < count; ++i) {
            dst[i] = float16_t::ToFloat(src[i]);
        }
    } else if constexpr (std::is_same_v<T, int8_t>) {
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
            ConvertToFloat(reinterpret_cast<const float16_t*>(vec_a_ptr), vec_a_float.data(), dimensions);
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
            ConvertToFloat(reinterpret_cast<const float16_t*>(vec_b_ptr), vec_b_float.data(), dimensions);
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


// Compresses a float vector into the specified edge format.
bool CompressVectorForEdge(const float* input_float_ptr, data_ptr_t dest_ptr,
                           idx_t dimensions, LMDiskannVectorType resolved_edge_type) {
     switch(resolved_edge_type) {
         case LMDiskannVectorType::FLOAT32:
             memcpy(dest_ptr, input_float_ptr, dimensions * sizeof(float));
             return true;
         case LMDiskannVectorType::FLOAT16:
             {
                 float16_t* dest_f16 = reinterpret_cast<float16_t*>(dest_ptr);
                 for(idx_t i=0; i<dimensions; ++i) {
                     dest_f16[i] = float16_t::FromFloat(input_float_ptr[i]);
                 }
                 return true;
             }
         case LMDiskannVectorType::INT8:
             {
                 // FIXME: Simple cast/scaling. Needs proper quantization scheme.
                 int8_t* dest_i8 = reinterpret_cast<int8_t*>(dest_ptr);
                 for(idx_t i=0; i<dimensions; ++i) {
                     // Example: Scale to [-127, 127] assuming input is roughly [-1, 1]
                     float clamped = std::max(-1.0f, std::min(1.0f, input_float_ptr[i]));
                     dest_i8[i] = static_cast<int8_t>(clamped * 127.0f);
                 }
                 return true;
             }
         case LMDiskannVectorType::UNKNOWN: // FLOAT1BIT
             // FIXME: Implement binarization (e.g., based on sign or threshold)
             throw NotImplementedException("Compression to FLOAT1BIT not implemented.");
             return false; // Unreachable
         default:
             return false; // Unsupported type
     }
}


} // namespace duckdb
