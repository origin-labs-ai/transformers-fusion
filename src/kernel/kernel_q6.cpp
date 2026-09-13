// kernel_q6.cpp — SIMD GEMM kernel for Q6 (6-bit, 64 centroids)
// 6-bit packed indices: 4 weights per 3 bytes (24 bits = 4 × 6 bits)
// Codebook: 64 × FP32 centroids
#include "quant/kernel.h"
#include "quant/tensor.h"
#include "quant/codebook.h"
#include <cstring>
#include <cmath>
#include <cstdint>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace quant {
namespace kernel {

// Unpack 4 × 6-bit indices from 3 bytes
// 24 bits = 4 × 6 bits: [b0][b1][b2]
// idx0 = bits[5:0], idx1 = bits[11:6], idx2 = bits[17:12], idx3 = bits[23:18]
static inline void unpack_q6_4(const uint8_t* src, uint8_t* dst) {
    uint32_t bits = (uint32_t)src[0] | ((uint32_t)src[1] << 8) | ((uint32_t)src[2] << 16);
    dst[0] = (uint8_t)(bits & 0x3F);
    dst[1] = (uint8_t)((bits >> 6) & 0x3F);
    dst[2] = (uint8_t)((bits >> 12) & 0x3F);
    dst[3] = (uint8_t)((bits >> 18) & 0x3F);
}

[[maybe_unused]] static void q6_gemm_scalar(const uint8_t* packed_indices, const float* codebook,
                             const float* activations, float* output,
                             int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            int packed_row_bytes = (K * 6 + 7) / 8;
            const uint8_t* w_row = packed_indices + ((int64_t)m * N + n) * packed_row_bytes;
            const float* a_row = activations + (int64_t)n * K;

#if defined(_MSC_VER)
            _mm_prefetch(reinterpret_cast<const char*>(w_row + packed_row_bytes), _MM_HINT_T0);
#else
            __builtin_prefetch(w_row + packed_row_bytes, 0, 3);
#endif

            int k = 0;
            // Process 4 weights at a time (3 bytes = 4 × 6-bit indices)
            for (; k + 4 <= K; k += 4) {
                uint8_t indices[4];
                unpack_q6_4(w_row + (k * 6) / 8, indices);
                for (int j = 0; j < 4; j++) {
                    sum += codebook[indices[j]] * a_row[k + j];
                }
            }
            // Handle remaining weights
            if (k < K) {
                uint8_t temp[3] = {};
                int remaining_bytes = ((K - k) * 6 + 7) / 8;
                for (int b = 0; b < remaining_bytes && b < 3; b++) {
                    temp[b] = w_row[(k * 6) / 8 + b];
                }
                uint8_t indices[4] = {};
                unpack_q6_4(temp, indices);
                for (int j = 0; k + j < K && j < 4; j++) {
                    sum += codebook[indices[j]] * a_row[k + j];
                }
            }
            output[m * N + n] = sum;
        }
    }
}

#if defined(__AVX2__)
static void q6_gemm_avx2(const uint8_t* packed_indices, const float* codebook,
                          const float* activations, float* output,
                          int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            __m256 sum8 = _mm256_setzero_ps();
            int packed_row_bytes = (K * 6 + 7) / 8;
            const uint8_t* w_row = packed_indices + ((int64_t)m * N + n) * packed_row_bytes;
            const float* a_row = activations + (int64_t)n * K;

            int k = 0;
            // Process 8 weights at a time: 2 × (4 × 6-bit) = 6 bytes
            for (; k + 8 <= K; k += 8) {
                // Unpack first 4 × 6-bit from bytes [0..2]
                uint8_t idx0[4], idx1[4];
                int byte_offset = (k * 6) / 8;
                unpack_q6_4(w_row + byte_offset, idx0);
                unpack_q6_4(w_row + byte_offset + 3, idx1);

#if defined(_MSC_VER)
                _mm_prefetch(reinterpret_cast<const char*>(w_row + byte_offset + 6), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(a_row + k + 8), _MM_HINT_T0);
#else
                __builtin_prefetch(w_row + byte_offset + 6, 0, 3);
                __builtin_prefetch(a_row + k + 8, 0, 3);
#endif

                // Gather 8 codebook entries
                __m256i idx = _mm256_set_epi32(
                    idx1[3], idx1[2], idx1[1], idx1[0],
                    idx0[3], idx0[2], idx0[1], idx0[0]);
                __m256 w = _mm256_i32gather_ps(codebook, idx, 4);
                __m256 a = _mm256_loadu_ps(a_row + k);
                sum8 = _mm256_fmadd_ps(w, a, sum8);
            }

            // Horizontal sum
            __m128 hi = _mm256_extractf128_ps(sum8, 1);
            __m128 lo = _mm256_castps256_ps128(sum8);
            __m128 sum4 = _mm_add_ps(lo, hi);
            sum4 = _mm_hadd_ps(sum4, sum4);
            sum4 = _mm_hadd_ps(sum4, sum4);
            float result = _mm_cvtss_f32(sum4);

            // Handle tail
            for (; k < K;) {
                uint8_t temp[3] = {};
                int byte_offset = (k * 6) / 8;
                int remaining_bytes = ((K - k) * 6 + 7) / 8;
                for (int b = 0; b < remaining_bytes && b < 3; b++) {
                    temp[b] = w_row[byte_offset + b];
                }
                uint8_t indices[4] = {};
                unpack_q6_4(temp, indices);
                for (int j = 0; k < K && j < 4; j++, k++) {
                    result += codebook[indices[j]] * a_row[k];
                }
            }
            output[m * N + n] = result;
        }
    }
}
#endif

void q6_gemm(const uint8_t* packed_indices, const float* codebook,
             const float* activations, float* output,
             int M, int N, int K) {
#if defined(__AVX2__)
    q6_gemm_avx2(packed_indices, codebook, activations, output, M, N, K);
#else
    q6_gemm_scalar(packed_indices, codebook, activations, output, M, N, K);
#endif
}

} // namespace kernel
} // namespace quant
