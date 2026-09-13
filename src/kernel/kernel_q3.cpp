// kernel_q3.cpp — SIMD GEMM kernel for Q3 (3-bit, 8 centroids)
// 3-bit packed indices: 8 weights per 3 bytes (24 bits = 8 × 3 bits)
// Codebook: 8 × FP32 centroids
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

// Unpack 8 × 3-bit indices from 3 bytes
// Byte layout: [b0][b1][b2]
// idx0 = b0[2:0], idx1 = b0[5:3], idx2 = (b0[7:6] | b1[0]) << shift,
// idx3 = b1[3:1], ... etc.
// Simplified: treat 24 bits as a bit stream, extract 3 bits at a time
static inline void unpack_q3_8(const uint8_t* src, uint8_t* dst) {
    uint32_t bits = (uint32_t)src[0] | ((uint32_t)src[1] << 8) | ((uint32_t)src[2] << 16);
    for (int i = 0; i < 8; i++) {
        dst[i] = (uint8_t)((bits >> (i * 3)) & 0x7);
    }
}

[[maybe_unused]] static void q3_gemm_scalar(const uint8_t* packed_indices, const float* codebook,
                             const float* activations, float* output,
                             int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            // Each weight row: K weights packed as 3-bit indices
            // K weights = (K * 3 + 7) / 8 bytes (3 bits per weight, rounded up)
            // But we process 8 weights at a time from 3 bytes
            int packed_row_bytes = (K * 3 + 7) / 8;
            const uint8_t* w_row = packed_indices + ((int64_t)m * N + n) * packed_row_bytes;
            const float* a_row = activations + (int64_t)n * K;

#if defined(_MSC_VER)
            _mm_prefetch(reinterpret_cast<const char*>(w_row + packed_row_bytes), _MM_HINT_T0);
#else
            __builtin_prefetch(w_row + packed_row_bytes, 0, 3);
#endif

            int k = 0;
            // Process 8 weights at a time (3 bytes = 8 × 3-bit indices)
            for (; k + 8 <= K; k += 8) {
                uint8_t indices[8];
                unpack_q3_8(w_row + (k * 3) / 8, indices);
                for (int j = 0; j < 8; j++) {
                    sum += codebook[indices[j]] * a_row[k + j];
                }
            }
            // Handle remaining weights
            if (k < K) {
                uint8_t indices[8] = {};
                int remaining_bytes = ((K - k) * 3 + 7) / 8;
                uint8_t temp[3] = {};
                for (int b = 0; b < remaining_bytes && b < 3; b++) {
                    temp[b] = w_row[(k * 3) / 8 + b];
                }
                unpack_q3_8(temp, indices);
                for (int j = 0; k + j < K && j < 8; j++) {
                    sum += codebook[indices[j]] * a_row[k + j];
                }
            }
            output[m * N + n] = sum;
        }
    }
}

#if defined(__AVX2__)
static void q3_gemm_avx2(const uint8_t* packed_indices, const float* codebook,
                          const float* activations, float* output,
                          int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            __m256 sum8 = _mm256_setzero_ps();
            int packed_row_bytes = (K * 3 + 7) / 8;
            const uint8_t* w_row = packed_indices + ((int64_t)m * N + n) * packed_row_bytes;
            const float* a_row = activations + (int64_t)n * K;

            int k = 0;
            for (; k + 8 <= K; k += 8) {
                // Unpack 8 × 3-bit indices
                uint8_t indices[8];
                unpack_q3_8(w_row + (k * 3) / 8, indices);

#if defined(_MSC_VER)
                _mm_prefetch(reinterpret_cast<const char*>(w_row + (k * 3) / 8 + 3), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(a_row + k + 8), _MM_HINT_T0);
#else
                __builtin_prefetch(w_row + (k * 3) / 8 + 3, 0, 3);
                __builtin_prefetch(a_row + k + 8, 0, 3);
#endif

                // Gather 8 codebook entries using AVX2
                __m256i idx = _mm256_set_epi32(
                    indices[7], indices[6], indices[5], indices[4],
                    indices[3], indices[2], indices[1], indices[0]);
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
            if (k < K) {
                uint8_t indices[8] = {};
                uint8_t temp[3] = {};
                int remaining_bytes = ((K - k) * 3 + 7) / 8;
                for (int b = 0; b < remaining_bytes && b < 3; b++) {
                    temp[b] = w_row[(k * 3) / 8 + b];
                }
                unpack_q3_8(temp, indices);
                for (int j = 0; k + j < K && j < 8; j++) {
                    result += codebook[indices[j]] * a_row[k + j];
                }
            }
            output[m * N + n] = result;
        }
    }
}
#endif

void q3_gemm(const uint8_t* packed_indices, const float* codebook,
             const float* activations, float* output,
             int M, int N, int K) {
#if defined(__AVX2__)
    q3_gemm_avx2(packed_indices, codebook, activations, output, M, N, K);
#else
    q3_gemm_scalar(packed_indices, codebook, activations, output, M, N, K);
#endif
}

} // namespace kernel
} // namespace quant
