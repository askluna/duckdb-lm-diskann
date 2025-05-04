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
//  • Each original D-dimensional vector is converted into two bit-planes:
//    - Positive Plane (posPlane): A bit is set if the original element was > 0.
//    - Negative Plane (negPlane): A bit is set if the original element was < 0.
//    (If both bits are 0, the original element was 0).
//  • This requires 2 bits per dimension.
//  • Planes are stored as arrays of `uint64_t` words.
//  • Total storage per vector ≈ (2 * D / 8) bytes = 0.25 * D bytes.
//  • Data is aligned to 64-bit boundaries (`uint64_t`) for efficient SIMD access.
//
//  SIMD Kernels & Dispatch:
//  ------------------------
//  • The core operation is a specialized "ternary dot product" calculated using
//    bitwise operations (AND) and population counts (popcount).
//    Formula: pop(q⁺∧v⁺) − pop(q⁺∧v⁻) − pop(q⁻∧v⁺) + pop(q⁻∧v⁻)
//      where q⁺/q⁻ are query planes, v⁺/v⁻ are database vector planes.
//  • Multiple SIMD implementations are provided for performance:
//      1. AVX‑512 (VPOPCNTDQ): Uses dedicated 64-bit popcount instruction. Best performance.
//      2. AVX2: Simulates popcount using scalar instructions on vector lanes.
//      3. NEON (AArch64): Uses 8-bit vector popcount instruction (`vcntq_u8`).
//      4. Scalar: Uses `__builtin_popcountll` (or `std::popcount` in C++20). Fallback.
//  • `ResolveKernel` function detects CPU capabilities at runtime and `GetKernel`
//    returns a function pointer to the fastest available implementation.
//
//  Public API:
//  -----------
//      EncodeTernary<T>(vec, posPlane, negPlane, dims): Encodes a single vector.
//      EncodeTernaryBatch<T>(srcPtrs, posPlane, negPlane, N, dims): Encodes a batch.
//      TopKTernarySearch(query_fp32, posPlaneData, negPlaneData, N, dims, K, neighIDs, results):
//          Performs the search.
//
//  Usage Notes:
//  ------------
//  • Header-only: Define `LMDK_TERNARY_IMPL` in exactly *one* .cpp file before
//    including this header to compile the function bodies.
//  • Zero-Copy: Assumes `posPlaneData` and `negPlaneData` point to memory regions
//    (potentially memory-mapped) containing the concatenated bit-planes for N vectors.
//  • `dims`: Can be any positive integer; SIMD kernels handle tails correctly.
//
//====================================================================
#ifndef LM_DISKANN_TERNARY_HPP
#define LM_DISKANN_TERNARY_HPP

#include <algorithm> // std::min, std::max, std::sort (implicitly via priority_queue)
#include <cassert>   // assert() for debugging checks
#include <cstdint>   // uint64_t, int64_t, uint8_t etc.
#include <cstring>   // std::memset
#include <queue>     // std::priority_queue for Top-K selection
#include <utility>   // std::pair
#include <vector>    // std::vector (used for query encoding buffer & results)

// Include SIMD headers based on target architecture
#if defined(__x86_64__) || defined(_M_X64) // x86-64 Architecture
  #include <immintrin.h> // AVX, AVX2, AVX-512 intrinsics
#elif defined(__aarch64__) // ARM 64-bit Architecture
  #include <arm_neon.h>  // NEON intrinsics
  // <sys/auxv.h> and <asm/hwcap.h> are not needed for the current NEON detection logic
#endif

// C++20 check for std::popcount
#if defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L
  #include <bit> // std::popcount
  #define LMDK_HAS_STD_POPCOUNT 1
#else
  #define LMDK_HAS_STD_POPCOUNT 0
#endif


namespace duckdb { // Logical namespace for LM-DiskANN components

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

//--------------------------------------------------------------------
// Ternary Encoding Functions
//--------------------------------------------------------------------

/**
 * @brief Encodes a single dense vector into two ternary bit-planes.
 * @tparam Scalar The data type of the source vector (e.g., float, double).
 * @param src Pointer to the start of the source vector.
 * @param[out] pos Pointer to the start of the positive bit-plane buffer (must have WordsPerPlane(dims) capacity).
 * @param[out] neg Pointer to the start of the negative bit-plane buffer (must have WordsPerPlane(dims) capacity).
 * @param dims The number of dimensions in the source vector.
 * @details Iterates through the source vector. For each dimension 'd':
 * - If src[d] > 0, sets the d-th bit in the 'pos' plane.
 * - If src[d] < 0, sets the d-th bit in the 'neg' plane.
 * - If src[d] == 0, both bits remain 0.
 * The bit position is calculated as word_index = d / 64, bit_in_word = d % 64.
 */
template <typename Scalar>
inline void EncodeTernary(const Scalar* src, uint64_t* pos, uint64_t* neg,
                          size_t dims) {
    // Pre-conditions (optional but recommended)
    assert(src != nullptr && "Source vector pointer cannot be null");
    assert(pos != nullptr && "Positive plane buffer pointer cannot be null");
    assert(neg != nullptr && "Negative plane buffer pointer cannot be null");
    assert(dims > 0 && "Dimensions must be positive");

    // Calculate the number of 64-bit words needed for the planes.
    const size_t words = WordsPerPlane(dims);

    // Zero out the destination buffers before setting bits.
    // Check words > 0, although memset with size 0 is usually a no-op.
    if (words > 0) {
        std::memset(pos, 0, words * sizeof(uint64_t));
        std::memset(neg, 0, words * sizeof(uint64_t));
    }

    // Loop through each dimension of the source vector.
    for (size_t d = 0; d < dims; ++d) {
        const Scalar value = src[d];
        // Determine the word index and bit index within that word.
        const size_t word_idx = d >> 6; // Equivalent to d / 64
        const size_t bit_idx  = d & 63;  // Equivalent to d % 64
        // Create a mask with only the target bit set.
        const uint64_t mask = 1ULL << bit_idx;

        // Check the sign of the value and set the corresponding bit.
        // Use a small epsilon for floating-point comparisons against zero, if necessary,
        // although direct comparison is often sufficient for this ternary scheme.
        // static constexpr Scalar zero_threshold = std::numeric_limits<Scalar>::epsilon();
        if (value > Scalar(0)) { // Positive value
            pos[word_idx] |= mask; // Set bit in positive plane
        } else if (value < Scalar(0)) { // Negative value
            neg[word_idx] |= mask; // Set bit in negative plane
        }
        // If value is zero (or close to zero), neither bit is set.
    }
}

/**
 * @brief Encodes a batch of dense vectors into concatenated bit-planes.
 * @tparam Scalar Data type of source vectors.
 * @param src Array of pointers, where each element points to a source vector.
 * @param[out] pos Pointer to the buffer for concatenated positive planes (size N * WordsPerPlane(dims)).
 * @param[out] neg Pointer to the buffer for concatenated negative planes (size N * WordsPerPlane(dims)).
 * @param N Number of vectors in the batch.
 * @param dims Dimension of each vector.
 * @details Calls EncodeTernary for each vector, writing the output planes contiguously.
 */
template <typename Scalar>
inline void EncodeTernaryBatch(const Scalar** src, uint64_t* pos, uint64_t* neg,
                               size_t N, size_t dims) {
    // Pre-conditions
    assert(src != nullptr && "Source pointer array cannot be null");
    assert(pos != nullptr && "Positive plane buffer pointer cannot be null");
    assert(neg != nullptr && "Negative plane buffer pointer cannot be null");
    assert(dims > 0 && "Dimensions must be positive");

    if (N == 0) return; // Nothing to do for an empty batch.

    const size_t words_per_vector = WordsPerPlane(dims);

    // Iterate through the batch.
    for (size_t i = 0; i < N; ++i) {
        // Ensure the individual vector pointer is valid.
        assert(src[i] != nullptr && "Source vector pointer in batch is null");

        // Calculate the starting position for the current vector's planes in the output buffers.
        uint64_t* current_pos_plane = pos + i * words_per_vector;
        uint64_t* current_neg_plane = neg + i * words_per_vector;

        // Encode the current vector.
        EncodeTernary(src[i], current_pos_plane, current_neg_plane, dims);
    }
}


//--------------------------------------------------------------------
// Ternary Dot Product Kernels (Scalar and SIMD)
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
using dot_fun_t = int64_t (*)(const uint64_t* qp, const uint64_t* qn,
                              const uint64_t* vp, const uint64_t* vn,
                              size_t words);

// --- Scalar Kernel (Fallback) ---

/**
 * @brief Scalar implementation of the ternary dot product.
 * @details Calculates: Σ [ popcount(qp[i]&vp[i]) - popcount(qp[i]&vn[i])
 * - popcount(qn[i]&vp[i]) + popcount(qn[i]&vn[i]) ]
 * Uses compiler builtins or C++20 std::popcount for 64-bit popcount.
 * Serves as a fallback and handles tail words for SIMD versions.
 */
static int64_t Dot_scalar(const uint64_t* qp, const uint64_t* qn,
                          const uint64_t* vp, const uint64_t* vn,
                          size_t words){
    int64_t accumulator = 0; // Use a signed accumulator
    for(size_t i = 0; i < words; ++i){
        // Calculate the four AND terms between query and vector planes.
        const uint64_t qp_and_vp = qp[i] & vp[i];
        const uint64_t qp_and_vn = qp[i] & vn[i];
        const uint64_t qn_and_vp = qn[i] & vp[i];
        const uint64_t qn_and_vn = qn[i] & vn[i];

        // Use std::popcount if available (C++20), otherwise fallback to compiler builtin.
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

// --- AVX-512 Kernel ---
// Requires AVX512F (foundation) and AVX512VPOPCNTDQ (vector popcount instruction)
#if defined(__AVX512VPOPCNTDQ__) && defined(__AVX512F__)
/**
 * @brief AVX-512 implementation using VPOPCNTDQ instruction.
 * @details Processes 8 x 64-bit words (512 bits) per loop iteration.
 * Uses the intrinsic _mm512_popcnt_epi64 for direct 64-bit popcount on vectors.
 * Handles tail words using the scalar kernel.
 * @param qp Query positive plane.
 * @param qn Query negative plane.
 * @param vp Vector positive plane.
 * @param vn Vector negative plane.
 * @param words Number of 64-bit words per plane.
 * @return Ternary dot product score.
 */
// Compile this function specifically with AVX-512 support enabled.
static __attribute__((target("avx512vpopcntdq,avx512f")))
int64_t Dot_avx512(const uint64_t* qp, const uint64_t* qn,
                   const uint64_t* vp, const uint64_t* vn,
                   size_t words){
    // Vector accumulator (512 bits, holds 8 x int64 lanes). Initialize to zero.
    __m512i total_acc_vec = _mm512_setzero_si512();
    size_t i = 0;

    // Main loop: process 8 words (512 bits) at a time.
    for(; i + 8 <= words; i += 8){
        // Load 512 bits (8 words) from each plane using unaligned loads (_loadu_).
        // Assumes planes might not be 64-byte aligned, which is safer.
        __m512i qP_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(qp + i));
        __m512i qN_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(qn + i));
        __m512i vP_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(vp + i));
        __m512i vN_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(vn + i));

        // Perform bitwise AND operations for the four terms.
        __m512i qp_and_vp = _mm512_and_si512(qP_vec, vP_vec);
        __m512i qp_and_vn = _mm512_and_si512(qP_vec, vN_vec);
        __m512i qn_and_vp = _mm512_and_si512(qN_vec, vP_vec);
        __m512i qn_and_vn = _mm512_and_si512(qN_vec, vN_vec);

        // Calculate population counts for each AND result using the dedicated instruction.
        __m512i pop_qp_vp = _mm512_popcnt_epi64(qp_and_vp);
        __m512i pop_qp_vn = _mm512_popcnt_epi64(qp_and_vn);
        __m512i pop_qn_vp = _mm512_popcnt_epi64(qn_and_vp);
        __m512i pop_qn_vn = _mm512_popcnt_epi64(qn_and_vn);

        // Combine the popcounts using vector subtraction and addition:
        // term = pop_qp_vp - pop_qp_vn - pop_qn_vp + pop_qn_vn
        __m512i term = _mm512_sub_epi64(pop_qp_vp, pop_qp_vn); // (pop_qp_vp - pop_qp_vn)
        term = _mm512_sub_epi64(term, pop_qn_vp);           // (... - pop_qn_vp)
        term = _mm512_add_epi64(term, pop_qn_vn);           // (... + pop_qn_vn)

        // Add the result for this iteration to the total vector accumulator.
        total_acc_vec = _mm512_add_epi64(total_acc_vec, term);
    }

    // Horizontal sum: Reduce the 8 lanes of the vector accumulator to a single scalar sum.
    int64_t total_acc_scalar = _mm512_reduce_add_epi64(total_acc_vec);

    // Process remaining tail words (0 to 7 words) using the scalar kernel.
    if (i < words) {
        total_acc_scalar += Dot_scalar(qp + i, qn + i, vp + i, vn + i, words - i);
    }

    return total_acc_scalar;
}
#endif // AVX-512 Check

// --- AVX2 Kernel ---
// Requires AVX2. Uses scalar popcount on vector lanes.
#if defined(__AVX2__)
/**
 * @brief AVX2 implementation of the ternary dot product.
 * @details Processes 4 x 64-bit words (256 bits) per loop iteration.
 * Lacks a direct vector popcount instruction. This implementation
 * extracts 64-bit lanes, uses scalar popcount (`__builtin_popcountll` or `std::popcount`),
 * and reassembles the results into a vector. This is functionally correct
 * but potentially slower than optimized AVX2 popcount algorithms (e.g., Harley-Seal).
 * Handles tail words using the scalar kernel.
 * @param qp Query positive plane.
 * @param qn Query negative plane.
 * @param vp Vector positive plane.
 * @param vn Vector negative plane.
 * @param words Number of 64-bit words per plane.
 * @return Ternary dot product score.
 */
// Compile this function specifically with AVX2 support enabled.
// Consider adding popcnt, bmi, bmi2 if __builtin_popcountll relies on them, though usually safe.
static __attribute__((target("avx2")))
int64_t Dot_avx2(const uint64_t* qp, const uint64_t* qn,
                 const uint64_t* vp, const uint64_t* vn,
                 size_t words){
    // Helper lambda to perform popcount on a 256-bit vector by processing scalar lanes.
    auto popcnt256_scalar_lanes = [](__m256i v) -> __m256i {
        // Align temporary storage for performance when storing/loading vector.
        alignas(32) uint64_t tmp[4];
        _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), v); // Store vector to memory

        // Perform scalar popcount on each 64-bit element.
        #if LMDK_HAS_STD_POPCOUNT
            uint64_t p0 = std::popcount(tmp[0]);
            uint64_t p1 = std::popcount(tmp[1]);
            uint64_t p2 = std::popcount(tmp[2]);
            uint64_t p3 = std::popcount(tmp[3]);
        #else
            uint64_t p0 = __builtin_popcountll(tmp[0]);
            uint64_t p1 = __builtin_popcountll(tmp[1]);
            uint64_t p2 = __builtin_popcountll(tmp[2]);
            uint64_t p3 = __builtin_popcountll(tmp[3]);
        #endif

        // Load the scalar results back into a 256-bit vector.
        return _mm256_set_epi64x(static_cast<long long>(p3), static_cast<long long>(p2),
                                 static_cast<long long>(p1), static_cast<long long>(p0));
    };

    // Vector accumulator (256 bits, holds 4 x int64 lanes). Initialize to zero.
    __m256i total_acc_vec = _mm256_setzero_si256();
    size_t i = 0;

    // Main loop: process 4 words (256 bits) at a time.
    for(; i + 4 <= words; i += 4){
        // Load 256 bits (4 words) from each plane using unaligned loads.
        __m256i qP_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qp + i));
        __m256i qN_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qn + i));
        __m256i vP_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vp + i));
        __m256i vN_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vn + i));

        // Perform bitwise AND operations.
        __m256i qp_and_vp = _mm256_and_si256(qP_vec, vP_vec);
        __m256i qp_and_vn = _mm256_and_si256(qP_vec, vN_vec);
        __m256i qn_and_vp = _mm256_and_si256(qN_vec, vP_vec);
        __m256i qn_and_vn = _mm256_and_si256(qN_vec, vN_vec);

        // Calculate population counts using the helper lambda.
        __m256i pop_qp_vp = popcnt256_scalar_lanes(qp_and_vp);
        __m256i pop_qp_vn = popcnt256_scalar_lanes(qp_and_vn);
        __m256i pop_qn_vp = popcnt256_scalar_lanes(qn_and_vp);
        __m256i pop_qn_vn = popcnt256_scalar_lanes(qn_and_vn);

        // Combine the popcounts using vector subtraction and addition.
        __m256i term = _mm256_sub_epi64(pop_qp_vp, pop_qp_vn);
        term = _mm256_sub_epi64(term, pop_qn_vp);
        term = _mm256_add_epi64(term, pop_qn_vn);

        // Add to the total vector accumulator.
        total_acc_vec = _mm256_add_epi64(total_acc_vec, term);
    }

    // Horizontal sum: Extract the 4 lanes and sum them manually.
    // Store the accumulator vector to aligned memory.
    alignas(32) int64_t acc_lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(acc_lanes), total_acc_vec);
    // Sum the scalar lanes.
    int64_t total_acc_scalar = acc_lanes[0] + acc_lanes[1] + acc_lanes[2] + acc_lanes[3];

    // Process remaining tail words (0 to 3 words) using the scalar kernel.
    if (i < words) {
       total_acc_scalar += Dot_scalar(qp + i, qn + i, vp + i, vn + i, words - i);
    }
    return total_acc_scalar;
}
#endif // AVX2 Check

// --- NEON Kernel (AArch64) ---
#if defined(__aarch64__)
/**
 * @brief NEON (AArch64) implementation of the ternary dot product.
 * @details Processes 16 bytes (128 bits / 2 x 64-bit words) per loop iteration.
 * Uses the `vcntq_u8` intrinsic to count set bits within each byte of a 128-bit vector.
 * Crucially, it performs horizontal sums (`vaddlvq_u8`) *before* combining the
 * four popcount terms to avoid issues with unsigned saturation arithmetic in `vsubq_u8`.
 * Handles tail words using the scalar kernel.
 * @param qp Query positive plane.
 * @param qn Query negative plane.
 * @param vp Vector positive plane.
 * @param vn Vector negative plane.
 * @param words Number of 64-bit words per plane.
 * @return Ternary dot product score.
 */
// No specific target attribute needed for baseline NEON on AArch64.
static int64_t Dot_neon(const uint64_t* qp, const uint64_t* qn,
                        const uint64_t* vp, const uint64_t* vn,
                        size_t words){
    int64_t accumulator = 0; // Signed accumulator for the final result.
    size_t i = 0;

    // Main loop: process 2 words (128 bits / 16 bytes) at a time.
    for(; i + 2 <= words; i += 2){
        // Load 128 bits (16 bytes) from each plane.
        // reinterpret_cast is necessary to treat uint64_t* as uint8_t* for vld1q_u8.
        // This is generally safe if alignment is suitable (uint64_t is usually 8-byte aligned).
        uint8x16_t qP_u8 = vld1q_u8(reinterpret_cast<const uint8_t*>(qp + i));
        uint8x16_t qN_u8 = vld1q_u8(reinterpret_cast<const uint8_t*>(qn + i));
        uint8x16_t vP_u8 = vld1q_u8(reinterpret_cast<const uint8_t*>(vp + i));
        uint8x16_t vN_u8 = vld1q_u8(reinterpret_cast<const uint8_t*>(vn + i));

        // Perform bitwise AND operations on the 128-bit vectors (byte-wise).
        uint8x16_t qp_and_vp_u8 = vandq_u8(qP_u8, vP_u8);
        uint8x16_t qp_and_vn_u8 = vandq_u8(qP_u8, vN_u8);
        uint8x16_t qn_and_vp_u8 = vandq_u8(qN_u8, vP_u8);
        uint8x16_t qn_and_vn_u8 = vandq_u8(qN_u8, vN_u8);

        // Calculate popcounts per byte for each AND result vector.
        // vcntq_u8 counts set bits in each of the 16 bytes independently.
        uint8x16_t pop_qp_vp_u8 = vcntq_u8(qp_and_vp_u8);
        uint8x16_t pop_qp_vn_u8 = vcntq_u8(qp_and_vn_u8);
        uint8x16_t pop_qn_vp_u8 = vcntq_u8(qn_and_vp_u8);
        uint8x16_t pop_qn_vn_u8 = vcntq_u8(qn_and_vn_u8);

        // **Critical Step:** Perform horizontal sum *before* combining terms.
        // vaddlvq_u8 sums the 16 uint8_t lanes into a single uint64_t result.
        // This avoids intermediate negative results that would be clamped by
        // unsigned vector subtraction (vsubq_u8).
        uint64_t sum_pop_qp_vp = vaddlvq_u8(pop_qp_vp_u8);
        uint64_t sum_pop_qp_vn = vaddlvq_u8(pop_qp_vn_u8);
        uint64_t sum_pop_qn_vp = vaddlvq_u8(pop_qn_vp_u8);
        uint64_t sum_pop_qn_vn = vaddlvq_u8(pop_qn_vn_u8);

        // Combine the scalar sums using standard signed arithmetic.
        // Cast the unsigned sums to signed int64_t before subtraction.
        accumulator += static_cast<int64_t>(sum_pop_qp_vp);
        accumulator -= static_cast<int64_t>(sum_pop_qp_vn);
        accumulator -= static_cast<int64_t>(sum_pop_qn_vp);
        accumulator += static_cast<int64_t>(sum_pop_qn_vn);
    }

    // Process the remaining tail word (0 or 1 word) using the scalar kernel.
    if (i < words) {
        accumulator += Dot_scalar(qp + i, qn + i, vp + i, vn + i, words - i);
    }
    return accumulator;
}
#endif // AArch64 Check


//--------------------------------------------------------------------
// Runtime Kernel Dispatcher
//--------------------------------------------------------------------

/**
 * @brief Detects CPU features at runtime and selects the best dot product kernel.
 * @return Function pointer (dot_fun_t) to the chosen kernel.
 * @details Checks for CPU instruction set support in order of performance
 * (AVX512_VPOPCNTDQ > AVX2 > NEON > Scalar). Uses compiler-specific
 * builtins like `__builtin_cpu_supports` (GCC/Clang) for detection.
 * Needs adaptation for other compilers (e.g., MSVC's __cpuidex).
 */
static dot_fun_t ResolveKernel() {
    // Check for most performant first.
    // Note: __builtin_cpu_supports is a GCC/Clang extension.
    // MSVC requires using __cpuid / __cpuidex.

#if defined(__AVX512VPOPCNTDQ__) && defined(__AVX512F__) && (defined(__GNUC__) || defined(__clang__))
    if (__builtin_cpu_supports("avx512vpopcntdq")) { // VPOPCNTDQ implies AVX512F
         return &Dot_avx512;
    }
#endif

#if defined(__AVX2__) && (defined(__GNUC__) || defined(__clang__))
    if (__builtin_cpu_supports("avx2")) {
        return &Dot_avx2;
    }
#endif

#if defined(__aarch64__)
    // Basic NEON is assumed to be available if compiled for AArch64.
    // No specific HWCAP check is needed for vcntq_u8, vandq_u8, vld1q_u8, vaddlvq_u8 etc.
    return &Dot_neon;
#endif

    // Fallback to scalar implementation if no suitable SIMD is detected or supported.
    return &Dot_scalar;
}

/**
 * @brief Gets the function pointer to the best available dot product kernel.
 * @return Function pointer (dot_fun_t).
 * @details Calls ResolveKernel() once on the first call and caches the result
 * in a static variable for subsequent calls (thread-safe since C++11).
 */
inline dot_fun_t GetKernel(){
    // Static local variable initialization is thread-safe in C++11 and later.
    static dot_fun_t resolved_kernel_function = ResolveKernel();
    return resolved_kernel_function;
}


//--------------------------------------------------------------------
// Top-K Ternary Search Function
//--------------------------------------------------------------------

/**
 * @brief Performs a Top-K nearest neighbor search using ternary encoded vectors.
 *
 * @param query Pointer to the original floating-point query vector.
 * @param posPlaneData Pointer to the start of the concatenated positive bit-planes
 * for all N database vectors. Data is assumed to be laid out
 * contiguously: [vec0_pos_plane, vec1_pos_plane, ...].
 * @param negPlaneData Pointer to the start of the concatenated negative bit-planes
 * for all N database vectors, laid out similarly.
 * @param N The total number of database vectors represented in the plane data.
 * @param dims The original dimensionality of the vectors.
 * @param K The number of nearest neighbors to retrieve.
 * @param neighIDs Pointer to an array of uint64_t IDs corresponding to the vectors
 * in posPlaneData/negPlaneData (must have size N). The ID at index `i`
 * corresponds to the vector whose planes start at `posPlaneData + i * words`.
 * @param[out] out A std::vector of std::pair<float, uint64_t> which will be filled
 * with the top K results, sorted by score (highest score first).
 * The pair contains (similarity_score, neighbor_ID).
 *
 * @details
 * 1. Encodes the floating-point query vector into its ternary bit-planes.
 * 2. Selects the fastest available ternary dot product kernel (using GetKernel).
 * 3. Iterates through the N database vectors:
 * a. Calculates the pointers to the current database vector's bit-planes.
 * b. Computes the raw ternary dot product score using the selected kernel.
 * c. Normalizes the raw score (divides by dims) as a proxy for similarity.
 * d. Maintains a min-priority queue (min-heap) of size K to keep track of the
 * current top K candidates based on the normalized score.
 * 4. Extracts the results from the priority queue into the output vector, sorted
 * from highest score to lowest score.
 */
inline void TopKTernarySearch(const float* query,
                              const uint64_t* posPlaneData,
                              const uint64_t* negPlaneData,
                              size_t N,                 // Number of database vectors
                              size_t dims,              // Original vector dimension
                              size_t K,                 // Number of neighbors to find
                              const uint64_t* neighIDs, // IDs for database vectors
                              std::vector<std::pair<float, uint64_t>>& out) // Output vector
{
    // --- Input Validation ---
    assert(query != nullptr && "Query vector pointer cannot be null");
    assert(posPlaneData != nullptr && "Positive plane data pointer cannot be null");
    assert(negPlaneData != nullptr && "Negative plane data pointer cannot be null");
    assert(neighIDs != nullptr && "Neighbor IDs pointer cannot be null");
    assert(dims > 0 && "Dimensions must be positive");
    assert(K > 0 && "K must be greater than 0");

    // Handle edge case: No database vectors to search.
    if (N == 0) {
        out.clear(); // Ensure output is empty
        return;
    }
    // Handle edge case: K is larger than the number of database vectors.
    K = std::min(K, N);
    if (K == 0) { // If K became 0 after std::min (e.g., K was initially 0 or N was 0)
        out.clear();
        return;
    }


    // --- Preparation ---
    // Calculate the number of 64-bit words per vector plane.
    const size_t words_per_vector = WordsPerPlane(dims);

    // Allocate temporary buffers for the encoded query vector's planes.
    // Using std::vector ensures proper memory management.
    std::vector<uint64_t> query_pos_plane_vec(words_per_vector);
    std::vector<uint64_t> query_neg_plane_vec(words_per_vector);

    // Encode the float query vector into the temporary ternary planes.
    EncodeTernary(query, query_pos_plane_vec.data(), query_neg_plane_vec.data(), dims);
    // Get pointers to the encoded query data for passing to the kernel.
    const uint64_t* qp = query_pos_plane_vec.data();
    const uint64_t* qn = query_neg_plane_vec.data();

    // Get the best available dot product kernel function pointer.
    dot_fun_t dot_kernel = GetKernel();

    // Define the type for score-ID pairs.
    using ScoreIdPair = std::pair<float, uint64_t>;

    // Define a custom comparator for the priority queue to make it a *min-heap*
    // based on the score (the float element of the pair).
    // `a > b` means `a` has lower priority (larger score goes deeper in min-heap).
    auto min_heap_comparator = [](const ScoreIdPair& a, const ScoreIdPair& b) {
        return a.first > b.first; // Smallest score has highest priority (at the top)
    };

    // Create the min-priority queue. It will store at most K elements.
    std::priority_queue<ScoreIdPair, std::vector<ScoreIdPair>, decltype(min_heap_comparator)>
        min_heap(min_heap_comparator);

    // --- Search Loop ---
    // Iterate through each database vector.
    for(size_t idx = 0; idx < N; ++idx){
        // Calculate pointers to the start of the current database vector's planes.
        const uint64_t* current_vpos = posPlaneData + idx * words_per_vector;
        const uint64_t* current_vneg = negPlaneData + idx * words_per_vector;

        // Compute the raw ternary dot product score using the selected kernel.
        int64_t raw_score = dot_kernel(qp, qn, current_vpos, current_vneg, words_per_vector);

        // Normalize the raw score. Dividing by dims provides a rough measure of
        // similarity per dimension, ranging approximately from -1.0 to +1.0.
        // This is NOT cosine similarity, but serves as a ranking score.
        // Avoid division by zero if dims is somehow 0 (though asserted earlier).
        float normalized_score = (dims > 0)
                               ? static_cast<float>(raw_score) / static_cast<float>(dims)
                               : 0.0f;

        // --- Update Top-K Heap ---
        // If the heap isn't full yet (has less than K elements), add the current result.
        if (min_heap.size() < K){
            min_heap.emplace(normalized_score, neighIDs[idx]);
        }
        // If the heap is full, check if the current score is better (higher) than
        // the worst score currently in the heap (which is at the top of the min-heap).
        else if (normalized_score > min_heap.top().first){
            min_heap.pop(); // Remove the element with the lowest score.
            min_heap.emplace(normalized_score, neighIDs[idx]); // Add the new, better element.
        }
        // Otherwise (current score is not better than the worst in the heap), do nothing.
    }

    // --- Result Extraction ---
    // The min-heap now contains the top K results, but ordered with the lowest score
    // at the top. We need to extract them into the output vector sorted highest-to-lowest.
    size_t result_count = min_heap.size(); // Should be equal to K unless N < K
    out.resize(result_count);

    // Pop elements from the min-heap (smallest score first) and place them
    // into the output vector from back to front to achieve descending score order.
    for (size_t i = 0; i < result_count; ++i) {
        out[result_count - 1 - i] = min_heap.top();
        min_heap.pop();
    }
}

} // namespace lmdk

#endif // LM_DISKANN_TERNARY_HPP
