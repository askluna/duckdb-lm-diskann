//====================================================================
//  lm_diskann_ternary.hpp  –  Zero‑copy ternary helpers for LM‑DiskANN
//====================================================================
//
//  Purpose:
//  Provides functions to encode floating-point vectors into a ternary
//  representation (+1, 0, -1) stored efficiently in bit-planes, and
//  perform fast Top-K similarity searches using these ternary vectors.
//  Designed for use with memory-mapped index nodes in LM-DiskANN for
//  zero-copy data access during search.
//
//  Layout & Compression:
//  ---------------------
//  • Each original D-dimensional vector is converted into two bit-planes.
//    A "bit-plane" is a sequence of bits representing sign information, where
//    the position of each bit corresponds directly to a dimension in the
//    original vector.
//
//    - Positive Plane (`posPlane`): A sequence of bits where the d-th bit is
//      set to `1` if the d-th element of the original vector was `> 0`,
//      and `0` otherwise.
//
//    - Negative Plane (`negPlane`): A sequence of bits where the d-th bit is
//      set to `1` if the d-th element of the original vector was `< 0`,
//      and `0` otherwise.
//
//    (If the d-th element was exactly `0`, the d-th bit in both `posPlane`
//     and `negPlane` remains `0`).
//
//  • This representation requires exactly 2 bits per original dimension.
//  • Each plane (positive and negative) is stored as a contiguous array of
//    `uint64_t` words. The `d`-th bit overall corresponds to bit `(d % 64)`
//    within word `(d / 64)`.
//  • Total storage per compressed vector: `2 * WordsPerPlane(dims) *
//  sizeof(uint64_t)` bytes,
//    which simplifies to approximately `ceil(2 * D / 8)` or `0.25 * D` bytes.
//  • Data is aligned to 64-bit boundaries (`uint64_t`) for efficient SIMD
//  access.
//
//  SIMD Kernels & Dispatch:
//  ------------------------
//  • The core operation is a specialized "ternary dot product" calculated using
//    bitwise operations (AND) and population counts (popcount).
//    Formula: pop(q⁺∧v⁺) − pop(q⁺∧v⁻) − pop(q⁻∧v⁺) + pop(q⁻∧v⁻)
//  • Uses Google Highway library for portable SIMD, with dynamic dispatch
//    to select the best available instruction set at runtime (e.g., AVX-512,
//    AVX2, NEON, Scalar).
//
//  Public API:
//  -----------
//      EncodeTernary<T>(vec, posPlane, negPlane, dims): Encodes a single
//      vector. EncodeTernaryBatch<T>(srcPtrs, posPlane, negPlane, N, dims):
//      Encodes a batch.
//      GetDotKernel(): Returns a function pointer to the best ternary dot
//      product implementation.
//
//  Usage Notes:
//  ------------
//  • Header-only concepts: The Highway parts require specific compilation setup.
//    Typically, the Highway kernel defined within HWY_NAMESPACE needs to be
//    compiled for multiple targets. This often involves having a .cpp file that
//    includes this header (or a dedicated -inl.h part of it) multiple times
//    under different HWY_TARGET definitions, or by using HWY_TARGET_INCLUDE
//    correctly set up in your CMake/build system.
//  • Zero-Copy: Assumes `posPlaneData` and `negPlaneData` in a search function
//    (not shown here) would point to memory regions containing the bit-planes.
//  • `dims`: Can be any positive integer; SIMD kernels handle tails correctly.
//
//====================================================================
#pragma once

#ifndef LM_DISKANN_TERNARY_HPP
#define LM_DISKANN_TERNARY_HPP

#include <algorithm> // std::min, std::max, std::sort (implicitly via priority_queue)
#include <cassert>   // assert() for debugging checks
#include <cstdint>   // uint64_t, int64_t, uint8_t etc.
#include <cstring>   // std::memset
#include <limits>    // Add for numeric_limits
#include <queue>     // std::priority_queue for Top-K selection
#include <stdexcept> // Add for runtime_error
#include <utility>   // std::pair
#include <vector>    // std::vector (used for query encoding buffer & results)

// C++20 check for std::popcount
#if defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L
#include <bit> // std::popcount
#define LMDK_HAS_STD_POPCOUNT 1
#else
#define LMDK_HAS_STD_POPCOUNT 0
#endif

// Highway specific includes
#include <hwy/highway.h>

// For hwy::contrib::algo::SumsOf (if used directly for byte summation, ensure it's appropriate)
// #include "hwy/contrib/algo/transform-inl.h"
// #include <hwy/contrib/tables/tables-inl.h> // For SetTable etc.

// Required for HWY_TARGET_INCLUDE and HWY_DYNAMIC_DISPATCH
// This definition points to the current file, indicating that this file contains
// the Highway target-specific code sections.
#define HWY_TARGET_INCLUDE "src/lm_diskann/core/ternary_quantization.hpp"

namespace diskann {
namespace core {

//--------------------------------------------------------------------
// Helper Functions
//--------------------------------------------------------------------

/**
 * @brief Calculates the number of 64-bit words required to store 'dims' bits.
 * @param dims The number of dimensions (bits).
 * @return The number of uint64_t elements needed.
 * @details Uses integer arithmetic equivalent to ceil(dims / 64.0).
 * Example: dims=64 -> 1, dims=65 -> 2, dims=128 -> 2.
 */
inline constexpr size_t WordsPerPlane(size_t dims) {
	// Add 63 before dividing by 64 (right shift by 6) to achieve ceiling effect.
	return (dims + 63) >> 6;
}

// Lookup table for popcount of a nibble (0-15)
// This data is used by Highway's SetTable to load into SIMD registers.
const uint8_t nibble_popcount_lut_data[16] = {
    /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
    /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
    /* 8 */ 1, /* 9 */ 2, /* A */ 2, /* B */ 3,
    /* C */ 2, /* D */ 3, /* E */ 3, /* F */ 4};

//--------------------------------------------------------------------
// Ternary Encoding Functions
//--------------------------------------------------------------------
/**
 * @brief Encodes a single dense vector into two ternary bit-planes.
 * @tparam Scalar The data type of the source vector (e.g., float, double).
 * @param src Pointer to the start of the source vector.
 * @param[out] pos Pointer to the start of the positive bit-plane buffer (must
 * have WordsPerPlane(dims) capacity).
 * @param[out] neg Pointer to the start of the negative bit-plane buffer (must
 * have WordsPerPlane(dims) capacity).
 * @param dims The number of dimensions in the source vector.
 * @details Iterates through the source vector. For each dimension 'd', it
 * determines the sign of the value `src[d]`.
 * - Based on the sign, it sets the corresponding d-th bit in either the
 * positive (`pos`) or negative (`neg`) bit-plane.
 * - Uses Highway SIMD for parallel comparisons, then processes the resulting
 *   bitmasks to update the planes.
 */
template <typename Scalar>
inline void EncodeTernary(const Scalar *src, uint64_t *pos, uint64_t *neg, size_t dims) {
	assert(src != nullptr && "Source vector pointer cannot be null");
	assert(pos != nullptr && "Positive plane buffer pointer cannot be null");
	assert(neg != nullptr && "Negative plane buffer pointer cannot be null");
	assert(dims > 0 && "Dimensions must be positive");

	const size_t words = WordsPerPlane(dims);
	if (words > 0) {
		std::memset(pos, 0, words * sizeof(uint64_t));
		std::memset(neg, 0, words * sizeof(uint64_t));
	}

	// Use HWY_STATIC_DISPATCH if you only compile for one target,
	// or ensure this part of code is also target-specific if it uses HWY_NAMESPACE types.
	// For EncodeTernary, which might be called from generic code, it's often simpler
	// to use the hwy:: हाईवे_NAMESPACE directly if types are compatible or
	// make this function also part of the HWY_TARGET_INCLUDE mechanism if it needs
	// to be specialized per target (e.g. if Lanes(d_scalar) behavior differs significantly).
	// For this example, we directly use hwy::HWY_NAMESPACE assuming it resolves correctly
	// in the context where EncodeTernary is called.
	// A more robust approach for a library might be to have EncodeTernary also be a
	// Highway-dispatched function if its performance is critical and varies by target.

	hwy::HWY_NAMESPACE::ScalableTag<Scalar> d_scalar;
	using VScalar = hwy::HWY_NAMESPACE::Vec<decltype(d_scalar)>;
	const size_t lanes = hwy::HWY_NAMESPACE::Lanes(d_scalar);
	auto zero_vec = hwy::HWY_NAMESPACE::Zero(d_scalar);

	for (size_t d_base = 0; d_base < dims; d_base += lanes) {
		const size_t num_elements_to_process = std::min(lanes, dims - d_base);

		auto current_elements_mask = hwy::HWY_NAMESPACE::FirstN(d_scalar, num_elements_to_process);
		// Load only the valid elements for this iteration.
		VScalar scalar_vec = hwy::HWY_NAMESPACE::MaskedLoad(current_elements_mask, d_scalar, src + d_base);

		// Perform comparisons. MaskedLoad ensures out-of-bounds lanes are zero,
		// so they won't satisfy Gt/Lt(zero).
		auto positive_simd_mask = hwy::HWY_NAMESPACE::Gt(scalar_vec, zero_vec);
		auto negative_simd_mask = hwy::HWY_NAMESPACE::Lt(scalar_vec, zero_vec);

		// Ensure masks are only true for valid lanes (already handled by MaskedLoad if it zeros,
		// but explicit And is safer if Gt/Lt behavior on zeroed lanes isn't guaranteed for this).
		// positive_simd_mask = hwy::HWY_NAMESPACE::And(positive_simd_mask, current_elements_mask);
		// negative_simd_mask = hwy::HWY_NAMESPACE::And(negative_simd_mask, current_elements_mask);

		uint64_t pos_bits_for_vector_lanes = hwy::HWY_NAMESPACE::MaskToBits(d_scalar, positive_simd_mask);
		uint64_t neg_bits_for_vector_lanes = hwy::HWY_NAMESPACE::MaskToBits(d_scalar, negative_simd_mask);

		// Iterate through the processed lanes and set bits in the output planes
		for (size_t k = 0; k < num_elements_to_process; ++k) {
			size_t current_dim_abs = d_base + k;          // Absolute dimension index
			const size_t word_idx = current_dim_abs / 64; // Which uint64_t word
			const size_t bit_idx = current_dim_abs % 64;  // Which bit within that word
			const uint64_t bit_mask_in_word = 1ULL << bit_idx;

			if ((pos_bits_for_vector_lanes >> k) & 1) {
				pos[word_idx] |= bit_mask_in_word;
			}
			if ((neg_bits_for_vector_lanes >> k) & 1) {
				neg[word_idx] |= bit_mask_in_word;
			}
		}
	}
}

/**
 * @brief Encodes a batch of dense vectors into concatenated bit-planes.
 * @tparam Scalar Data type of source vectors.
 * @param src Array of pointers, where each element points to a source vector.
 * @param[out] pos Pointer to the buffer for concatenated positive planes (size
 * N * WordsPerPlane(dims)).
 * @param[out] neg Pointer to the buffer for concatenated negative planes (size
 * N * WordsPerPlane(dims)).
 * @param N Number of vectors in the batch.
 * @param dims Dimension of each vector.
 * @details Calls EncodeTernary for each vector, writing the output planes
 * contiguously.
 */
template <typename Scalar>
inline void EncodeTernaryBatch(const Scalar **src, uint64_t *pos, uint64_t *neg, size_t N, size_t dims) {
	// Pre-conditions
	assert(src != nullptr && "Source pointer array cannot be null");
	assert(pos != nullptr && "Positive plane buffer pointer cannot be null");
	assert(neg != nullptr && "Negative plane buffer pointer cannot be null");
	assert(dims > 0 && "Dimensions must be positive");

	if (N == 0)
		return; // Nothing to do for an empty batch.

	const size_t words_per_vector = WordsPerPlane(dims);

	// Iterate through the batch.
	for (size_t i = 0; i < N; ++i) {
		// Ensure the individual vector pointer is valid.
		assert(src[i] != nullptr && "Source vector pointer in batch is null");

		// Calculate the starting position for the current vector's planes in the
		// output buffers.
		uint64_t *current_pos_plane = pos + i * words_per_vector;
		uint64_t *current_neg_plane = neg + i * words_per_vector;

		// Encode the current vector.
		EncodeTernary(src[i], current_pos_plane, current_neg_plane, dims);
	}
}

//--------------------------------------------------------------------
// Ternary Dot Product Kernels (Scalar and Highway SIMD)
//--------------------------------------------------------------------

/**
 * @brief Function pointer type for the ternary dot product kernels.
 * @param qp Pointer to query positive plane (uint64_t array).
 * @param qn Pointer to query negative plane (uint64_t array).
 * @param vp Pointer to database vector positive plane (uint64_t array).
 * @param vn Pointer to database vector negative plane (uint64_t array).
 * @param words The number of 64-bit words in each plane array.
 * @return The raw ternary dot product score (int64_t).
 */
using dot_fun_t = int64_t (*)(const uint64_t *qp, const uint64_t *qn, const uint64_t *vp, const uint64_t *vn,
                              size_t words);

// --- Scalar Kernel (Fallback for tail or non-SIMD paths) ---

/**
 * @brief Scalar implementation of the ternary dot product.
 * @details Calculates: Σ [ popcount(qp[i]&vp[i]) - popcount(qp[i]&vn[i])
 * - popcount(qn[i]&vp[i]) + popcount(qn[i]&vn[i]) ]
 * Uses compiler builtins or C++20 std::popcount for 64-bit popcount.
 * Serves as a fallback and handles tail words for SIMD versions.
 */
static int64_t Dot_scalar(const uint64_t *qp, const uint64_t *qn, const uint64_t *vp, const uint64_t *vn,
                          size_t words) {
	int64_t accumulator = 0; // Use a signed accumulator
	for (size_t i = 0; i < words; ++i) {
		// Calculate the four AND terms between query and vector planes.
		const uint64_t qp_and_vp = qp[i] & vp[i];
		const uint64_t qp_and_vn = qp[i] & vn[i];
		const uint64_t qn_and_vp = qn[i] & vp[i];
		const uint64_t qn_and_vn = qn[i] & vn[i];

// Use std::popcount if available (C++20), otherwise fallback to compiler
// builtin.
#if LMDK_HAS_STD_POPCOUNT
		accumulator += static_cast<int64_t>(std::popcount(qp_and_vp));
		accumulator -= static_cast<int64_t>(std::popcount(qp_and_vn));
		accumulator -= static_cast<int64_t>(std::popcount(qn_and_vp));
		accumulator += static_cast<int64_t>(std::popcount(qn_and_vn));
#else // Fallback to GCC/Clang builtin
		// Note: For MSVC, use __popcnt64 or _mm_popcnt_u64
		accumulator += static_cast<int64_t>(__builtin_popcountll(qp_and_vp));
		accumulator -= static_cast<int64_t>(__builtin_popcountll(qp_and_vn));
		accumulator -= static_cast<int64_t>(__builtin_popcountll(qn_and_vp));
		accumulator += static_cast<int64_t>(__builtin_popcountll(qn_and_vn));
#endif
	}
	return accumulator;
}

// Highway specific code must be within HWY_NAMESPACE for HWY_TARGET_INCLUDE
// This block will be compiled multiple times for different targets.
// Ensure this part of the file is correctly processed by Highway's build system.
} // namespace core
} // namespace diskann
HWY_BEFORE_NAMESPACE(); // Must be outside other namespaces. Marks beginning of a Highway target-specific block.
namespace diskann {
namespace core {
namespace HWY_NAMESPACE { // Switch to Highway's target-specific namespace.

// Import C++ standard library symbols into the HWY_NAMESPACE if needed.
// HWY_USING_STD_MINMAX(); // For std::min/max if used inside kernel

// You can use HWY_ATTR for functions defined within this namespace to apply
// target-specific attributes (like __attribute__((target(...)))).

// Helper to popcount a vector of uint8_t using a nibble lookup table.
// Returns a vector of uint8_t where each lane is the popcount of the input byte lane.
template <typename D_U8> // D_U8 is e.g. hwy::HWY_NAMESPACE::ScalableTag<uint8_t>
HWY_ATTR hwy::HWY_NAMESPACE::Vec<D_U8> PopCountBytesUsingLUT(D_U8 d_u8, hwy::HWY_NAMESPACE::Vec<D_U8> bytes_vec) {
	// Load LUT data into SIMD vector. SetTable is suitable for this.
	// Assumes nibble_popcount_lut_data is accessible.
	auto lut_lanes = hwy::HWY_NAMESPACE::SetTable(d_u8, diskann::core::nibble_popcount_lut_data);

	// Extract low and high nibbles for each byte
	auto low_nibbles = hwy::HWY_NAMESPACE::And(bytes_vec, hwy::HWY_NAMESPACE::Set(d_u8, 0x0F));
	auto high_nibbles = hwy::HWY_NAMESPACE::ShiftRight<4>(bytes_vec);
	// For TableLookupBytes, only the low bits of the index lanes are typically used if the table is small.
	// So, an explicit mask on high_nibbles (e.g., And with 0x0F) might not be needed here
	// if TableLookupBytes handles it or the LUT is indexed appropriately.

	// Perform table lookups for popcounts of low and high nibbles
	auto pop_low = hwy::HWY_NAMESPACE::TableLookupBytes(low_nibbles, lut_lanes);
	auto pop_high = hwy::HWY_NAMESPACE::TableLookupBytes(high_nibbles, lut_lanes);

	// Sum popcounts of low and high nibbles to get popcount per byte
	return hwy::HWY_NAMESPACE::Add(pop_low, pop_high);
}

// Helper to take a vector of N*8 byte popcounts and sum them into N uint64_t popcounts.
// D_U64 is the descriptor for uint64_t vectors (e.g., ScalableTag<uint64_t>).
// D_U8 is the descriptor for uint8_t vectors.
// byte_popcounts_vec contains popcounts for each byte of the original uint64_t data.
template <typename D_U64, typename D_U8>
HWY_ATTR hwy::HWY_NAMESPACE::Vec<D_U64>
SumBytePopcountsToUint64Lanes(D_U64 d_u64, D_U8 d_u8, hwy::HWY_NAMESPACE::Vec<D_U8> byte_popcounts_vec) {
	// --- PRODUCTION NOTE ---
	// The following is a SCALAR, LANE-BY-LANE sum for ILLUSTRATIVE PURPOSES.
	// It correctly describes the logic but is NOT EFFICIENT for a SIMD context.
	// A proper Highway implementation would use a tree of vectorized PromoteEvenTo/OddTo
	// and Add operations, or a specialized utility like hwy::contrib::SumsOf<8>
	// if available and applicable. This section needs to be replaced with such
	// a vectorized approach for actual performance.
	// The complexity of a fully vectorized sum tree is omitted here for brevity
	// but is essential for a real-world high-performance version.

	const size_t u64_lanes = hwy::HWY_NAMESPACE::Lanes(d_u64);
	// Max lanes for uint64_t for storage. HWY_MAX_LANES_D might be an overestimate but safe.
	HWY_ALIGN uint64_t lane_sums_storage[hwy::kMaxLanes / sizeof(uint64_t)]; // Max possible u64 lanes

	// Temporary storage for byte popcounts (to extract scalar values)
	// Max lanes for uint8_t.
	HWY_ALIGN uint8_t byte_counts_storage[hwy::kMaxLanes];
	hwy::HWY_NAMESPACE::Store(byte_popcounts_vec, d_u8, byte_counts_storage);

	for (size_t i = 0; i < u64_lanes; ++i) {
		uint64_t current_u64_lane_popcount = 0;
		for (size_t byte_k = 0; byte_k < 8; ++byte_k) {
			// Assuming byte_popcounts_vec has lanes ordered such that
			// bytes for u64_lane[i] are at byte_counts_storage[i*8 + byte_k]
			current_u64_lane_popcount += byte_counts_storage[i * 8 + byte_k];
		}
		lane_sums_storage[i] = current_u64_lane_popcount;
	}
	return hwy::HWY_NAMESPACE::Load(d_u64, lane_sums_storage);
}

// Vectorized Popcount for a vector of uint64_t.
// Returns a vector of uint64_t where each lane is the popcount of the corresponding input lane.
template <typename D_U64> // D_U64 is e.g. hwy::HWY_NAMESPACE::ScalableTag<uint64_t>
HWY_ATTR hwy::HWY_NAMESPACE::Vec<D_U64> PopCountVectorU64(D_U64 d_u64, hwy::HWY_NAMESPACE::Vec<D_U64> u64_vec) {
	hwy::HWY_NAMESPACE::ScalableTag<uint8_t> d_u8; // Descriptor for uint8_t vectors

	// 1. Bitcast uint64_t vector to uint8_t vector.
	// If u64_vec has L lanes, u8_vec (its bitcast representation) will conceptually cover 8*L bytes.
	auto u8_equivalent_vec = hwy::HWY_NAMESPACE::BitCast(d_u8, u64_vec);

	// 2. Popcount each byte using LUT. This returns a vector of byte popcounts.
	auto byte_popcounts = PopCountBytesUsingLUT(d_u8, u8_equivalent_vec); // Vec<D_U8>

	// 3. Sum 8 byte popcounts to form one uint64_t popcount for each original u64 lane.
	return SumBytePopcountsToUint64Lanes(d_u64, d_u8, byte_popcounts);
}

// Main Highway kernel for ternary dot product.
// This function will be compiled for each enabled SIMD target.
HWY_ATTR int64_t HighwayTernaryDotKernel(const uint64_t *HWY_RESTRICT qp, const uint64_t *HWY_RESTRICT qn,
                                         const uint64_t *HWY_RESTRICT vp, const uint64_t *HWY_RESTRICT vn,
                                         size_t words) {
	hwy::HWY_NAMESPACE::ScalableTag<uint64_t> d_u64;
	using VU64 = hwy::HWY_NAMESPACE::Vec<decltype(d_u64)>;

	const size_t lanes = hwy::HWY_NAMESPACE::Lanes(d_u64);
	VU64 acc_total = hwy::HWY_NAMESPACE::Zero(d_u64); // Vector accumulator for sums

	size_t i = 0;
	// Main loop: process 'lanes' number of uint64_t words at a time.
	for (; i + lanes <= words; i += lanes) {
		VU64 qp_vec = hwy::HWY_NAMESPACE::LoadU(d_u64, qp + i);
		VU64 qn_vec = hwy::HWY_NAMESPACE::LoadU(d_u64, qn + i);
		VU64 vp_vec = hwy::HWY_NAMESPACE::LoadU(d_u64, vp + i);
		VU64 vn_vec = hwy::HWY_NAMESPACE::LoadU(d_u64, vn + i);

		// Perform bitwise AND operations for the four terms
		VU64 term_qp_vp = hwy::HWY_NAMESPACE::And(qp_vec, vp_vec);
		VU64 term_qp_vn = hwy::HWY_NAMESPACE::And(qp_vec, vn_vec);
		VU64 term_qn_vp = hwy::HWY_NAMESPACE::And(qn_vec, vp_vec);
		VU64 term_qn_vn = hwy::HWY_NAMESPACE::And(qn_vec, vn_vec);

		// Calculate population counts for each ANDed result vector
		VU64 pop_qp_vp = PopCountVectorU64(d_u64, term_qp_vp);
		VU64 pop_qp_vn = PopCountVectorU64(d_u64, term_qp_vn);
		VU64 pop_qn_vp = PopCountVectorU64(d_u64, term_qn_vp);
		VU64 pop_qn_vn = PopCountVectorU64(d_u64, term_qn_vn);

		// Combine the popcounts according to the formula:
		// result = pop_qp_vp - pop_qp_vn - pop_qn_vp + pop_qn_vn
		VU64 term_sum = pop_qp_vp;                               // Start with the first positive term
		term_sum = hwy::HWY_NAMESPACE::Sub(term_sum, pop_qp_vn); // Subtract second term
		term_sum = hwy::HWY_NAMESPACE::Sub(term_sum, pop_qn_vp); // Subtract third term
		term_sum = hwy::HWY_NAMESPACE::Add(term_sum, pop_qn_vn); // Add fourth term

		// Add the result for this iteration to the total vector accumulator
		acc_total = hwy::HWY_NAMESPACE::Add(acc_total, term_sum);
	}

	// Horizontal sum: Reduce the lanes of the vector accumulator to a single scalar sum.
	// ReduceSum is appropriate here.
	int64_t total_scalar_sum = hwy::HWY_NAMESPACE::ReduceSum(d_u64, acc_total);

	// Process remaining tail words (0 to lanes-1 words) using the scalar kernel.
	if (i < words) {
		// Call the original Dot_scalar, which is defined outside HWY_NAMESPACE
		total_scalar_sum += diskann::core::Dot_scalar(qp + i, qn + i, vp + i, vn + i, words - i);
	}

	return total_scalar_sum;
}

} // namespace HWY_NAMESPACE
} // namespace core
} // namespace diskann
HWY_AFTER_NAMESPACE(); // Marks end of Highway target-specific block.

// This HWY_ONCE block ensures that the code within is only included once
// by the compiler, even if this header is processed multiple times for
// different Highway targets (which happens with HWY_TARGET_INCLUDE).
#if HWY_ONCE

namespace diskann {
namespace core {

// This macro exports the HighwayTernaryDotKernel function for dynamic dispatch
// and defines a dispatcher function named GetHighwayTernaryDotKernel.
HWY_EXPORT_AND_DYNAMIC_DISPATCH(dot_fun_t, HighwayTernaryDotKernel, HighwayTernaryDotKernel);

/**
 * @brief Gets the function pointer to the best available dot product kernel.
 * @return Function pointer (dot_fun_t).
 * @details Calls Highway's dynamic dispatcher to select the most optimized
 * kernel for the current CPU architecture at runtime.
 */
inline dot_fun_t GetDotKernel() {
	// This calls the dispatcher function generated by HWY_EXPORT_AND_DYNAMIC_DISPATCH
	return GetHighwayTernaryDotKernel();
}

// The rest of the file (e.g., if TopKTernarySearch or other functions were here)
// would use GetDotKernel() to get the appropriate dot product function.
// EncodeTernaryBatch already calls the modified EncodeTernary, so no changes needed there.

} // namespace core
} // namespace diskann

#endif // HWY_ONCE

#endif // LM_DISKANN_TERNARY_HPP
