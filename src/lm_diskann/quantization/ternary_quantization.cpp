#define LMDK_TERNARY_IMPL

// This path assumes that the include directories are set up in CMake
// such that "core/" is a recognized path.
// If src/lm_diskann/ is an include directory, this should resolve.
// Otherwise, it might need to be "../core/ternary_quantization.hpp"
#include "../core/ternary_quantization.hpp"

/**
 * @file ternary_quantization_impl.cpp
 * @brief Instantiates the implementations for ternary quantization functions.
 *
 * This file defines LMDK_TERNARY_IMPL before including the
 * ternary_quantization.hpp header. This causes the function bodies
 * within that header to be compiled into this translation unit, making them
 * available to the linker. This resolves undefined reference errors for
 * functions like diskann::core::CalculateApproxDistance,
 * diskann::core::CompressVectorForEdge, and diskann::core::ConvertToFloat,
 * assuming their implementations are guarded by the LMDK_TERNARY_IMPL macro
 * within ternary_quantization.hpp.
 */