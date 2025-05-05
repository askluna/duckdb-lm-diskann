#include "distance.hpp"
#include "ternary_quantization.hpp" // For EncodeTernary, GetKernel

// #include "config.hpp" // No longer needed here, included via distance.hpp indirectly or types are forward declared

#include "duckdb/common/exception.hpp"
// #include "duckdb/parser/binder/binder_exception.hpp" // Removed: File not found
// #include "duckdb/parser/parser_exception.hpp" // Removed: File not found
#include "duckdb/common/limits.hpp" // For NumericLimits if needed
#include "duckdb/common/types/vector.hpp" // Potentially for VectorOperations or Vector
// #include "duckdb/function/scalar/vector_functions.hpp" // Removed: File not found / Exact function TBD
#include "duckdb/common/types/value.hpp" // Needed for ParseOptions
#include "duckdb/common/string_util.hpp" // Needed for ParseOptions

#include <cmath> // For std::sqrt, std::fabs
#include <vector> // For temporary query plane buffers
#include <stdexcept> // For runtime_error


namespace duckdb {


// --- Core Function Implementations ---

// Helper function to get vector data pointer (adjust if Vector::GetData changes)
// static inline const float* GetFloatDataPtr(const Vector& vec) { // Commented out: Not currently used
// 	return FlatVector::GetData<float>(vec);
// }


float ComputeExactDistanceFloat(const float *a_ptr, const float *b_ptr, idx_t dimensions, LMDiskannMetricType metric_type) {
    // TODO: Ideally, use optimized DuckDB functions if possible, maybe via VectorOperations::Distance.
    // For now, implement manually based on metric type.
    // This is a placeholder and likely needs optimization/replacement.

    switch (metric_type) {
        case LMDiskannMetricType::L2: {
            float distance = 0.0f;
            for (idx_t i = 0; i < dimensions; ++i) {
                float diff = a_ptr[i] - b_ptr[i];
                distance += diff * diff;
            }
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
            float norm_a = 0.0f;
            float norm_b = 0.0f;
            for (idx_t i = 0; i < dimensions; ++i) {
                dot_product += a_ptr[i] * b_ptr[i];
                norm_a += a_ptr[i] * a_ptr[i];
                norm_b += b_ptr[i] * b_ptr[i];
            }
            norm_a = std::sqrt(norm_a);
            norm_b = std::sqrt(norm_b);
            if (norm_a == 0.0f || norm_b == 0.0f) {
                return 1.0f; // Handle zero vectors: max distance
            }
            float cosine_similarity = dot_product / (norm_a * norm_b);
            // Clamp similarity to [-1, 1] due to potential floating point inaccuracies
            cosine_similarity = std::max(-1.0f, std::min(1.0f, cosine_similarity));
            return 1.0f - cosine_similarity;
        }
        default: {
            // Use base Exception as specific types (BinderException, ParserException) couldn't be found.
            throw Exception(ExceptionType::INVALID_INPUT, "ComputeExactDistanceFloat: Unsupported metric type");
        }
    }
}


float ComputeApproxSimilarityTernary(const float *query_float_ptr,
                                     const_data_ptr_t neighbor_pos_plane_ptr,
                                     const_data_ptr_t neighbor_neg_plane_ptr,
                                     idx_t dimensions,
                                     LMDiskannMetricType metric_type) {

    // 1. Prepare query planes
    const size_t words_per_vector = WordsPerPlane(dimensions);
    std::vector<uint64_t> query_pos_plane_vec(words_per_vector);
    std::vector<uint64_t> query_neg_plane_vec(words_per_vector);

    EncodeTernary(query_float_ptr, query_pos_plane_vec.data(), query_neg_plane_vec.data(), dimensions);

    // 2. Get the appropriate SIMD kernel
    dot_fun_t dot_kernel = GetKernel();

    // 3. Compute the raw ternary dot product score
    // Note: The ternary kernel itself doesn't use the metric_type, it always computes the same score.
    // The interpretation (similarity vs. distance proxy) happens outside.
    int64_t raw_score = dot_kernel(
        query_pos_plane_vec.data(),
        query_neg_plane_vec.data(),
        reinterpret_cast<const uint64_t*>(neighbor_pos_plane_ptr),
        reinterpret_cast<const uint64_t*>(neighbor_neg_plane_ptr),
        words_per_vector
    );

    // 4. Return the raw score as float (higher is better similarity)
    return static_cast<float>(raw_score);
}

void CompressVectorToTernary(const float* input_float_ptr,
                             data_ptr_t dest_pos_plane_ptr,
                             data_ptr_t dest_neg_plane_ptr,
                             idx_t dimensions) {

    EncodeTernary(
        input_float_ptr,
        reinterpret_cast<uint64_t*>(dest_pos_plane_ptr),
        reinterpret_cast<uint64_t*>(dest_neg_plane_ptr),
        dimensions
    );
}


// --- Configuration Function Implementations ---

// Parses options from the CREATE INDEX statement's WITH clause.
void ParseOptions(const case_insensitive_map_t<Value> &options,
                  LMDiskannMetricType &metric_type,
                  uint32_t &r, uint32_t &l_insert,
                  float &alpha, uint32_t &l_search) {

    metric_type = LMDISKANN_DEFAULT_METRIC;
    r = LMDISKANN_DEFAULT_R;
    l_insert = LMDISKANN_DEFAULT_L_INSERT;
    alpha = LMDISKANN_DEFAULT_ALPHA;
    l_search = LMDISKANN_DEFAULT_L_SEARCH;

    for (const auto &kv : options) {
        auto loption = StringUtil::Lower(kv.first);
        const auto &value = kv.second;

        if (loption == LMDISKANN_METRIC_OPTION) {
            auto metric_str = StringUtil::Lower(value.ToString());
            if (metric_str == "l2") {
                metric_type = LMDiskannMetricType::L2;
            } else if (metric_str == "cosine") {
                metric_type = LMDiskannMetricType::COSINE;
            } else if (metric_str == "ip") {
                metric_type = LMDiskannMetricType::IP;
            } else {
                // Use base Exception as specific types (BinderException, ParserException) couldn't be found.
                throw Exception(ExceptionType::INVALID_INPUT, StringUtil::Format("Unknown LM-DiskANN metric type: '%s'. Options are l2, cosine, ip", value.ToString()));
            }
        } else if (loption == LMDISKANN_R_OPTION) {
            r = value.GetValue<uint32_t>();
        } else if (loption == LMDISKANN_L_INSERT_OPTION) {
            l_insert = value.GetValue<uint32_t>();
        } else if (loption == LMDISKANN_ALPHA_OPTION) {
            alpha = value.GetValue<float>();
        } else if (loption == LMDISKANN_L_SEARCH_OPTION) {
            l_search = value.GetValue<uint32_t>();
        } else {
             // Use base Exception
            throw Exception(ExceptionType::INVALID_INPUT, StringUtil::Format("Unknown LM-DiskANN option: %s", kv.first));
        }
    }

    // Call validation after parsing
    ValidateParameters(metric_type, r, l_insert, alpha, l_search);
}

// Validates the combination of parameters.
void ValidateParameters(LMDiskannMetricType metric_type,
                        uint32_t r, uint32_t l_insert, float alpha, uint32_t l_search) {
    // Use base Exception
    if (metric_type == LMDiskannMetricType::UNKNOWN) {
         throw Exception(ExceptionType::INVALID_INPUT, "LM-DiskANN metric type cannot be UNKNOWN");
    }
    if (r == 0) {
        throw Exception(ExceptionType::INVALID_INPUT, "LM-DiskANN R must be > 0");
    }
    if (l_insert == 0) {
        throw Exception(ExceptionType::INVALID_INPUT, "LM-DiskANN L_insert must be > 0");
    }
     if (alpha <= 1.0f) {
        throw Exception(ExceptionType::INVALID_INPUT, "LM-DiskANN alpha must be > 1.0");
    }
    if (l_search == 0) {
        throw Exception(ExceptionType::INVALID_INPUT, "LM-DiskANN L_search must be > 0");
    }
    if (l_search < 1) { // Or some other reasonable minimum K for search?
         throw Exception(ExceptionType::INVALID_INPUT, "LM-DiskANN L_search must be at least 1");
    }
    // Add any other cross-parameter validation if needed
}

} // namespace duckdb
