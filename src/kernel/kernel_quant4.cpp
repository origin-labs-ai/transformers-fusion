#include "quant/kernel.h"
#include "quant/tensor.h"
#include "quant/codebook.h"
#include "quant/detail/fp16.h"
#include <cstring>
#include <cmath>
#include <cstdint>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace quant {
namespace kernel {

// FP16 bits -> float — see quant/detail/fp16.h (DEDUP).
using detail::fp16_to_float;

[[maybe_unused]] static void quant4_gemm_scalar(const uint8_t* packed_indices, const uint16_t* codebook,
                              const float* activations, float* output,
                              int M, int N, int K) {
    float f16_centroids[16];
    for (int i = 0; i < 16; i++)
        f16_centroids[i] = fp16_to_float(codebook[i]);

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            const uint8_t* w_row = packed_indices + ((int64_t)m * N + n) * K / 2;
            const float* a_row = activations + (int64_t)n * K;
            // Prefetch next weight row into L1 cache
#if defined(_MSC_VER)
            _mm_prefetch(reinterpret_cast<const char*>(w_row + K / 2), _MM_HINT_T0);
#else
            __builtin_prefetch(w_row + K / 2, 0, 3);
#endif
            for (int k = 0; k < K; k++) {
                uint8_t idx;
                if (k % 2 == 0)
                    idx = w_row[k / 2] & 0xF;
                else
                    idx = (w_row[k / 2] >> 4) & 0xF;
                sum += f16_centroids[idx] * a_row[k];
            }
            output[m * N + n] = sum;
        }
    }
}

#if defined(__AVX2__)
static void quant4_gemm_avx2(const uint8_t* packed_indices, const uint16_t* codebook,
                            const float* activations, float* output,
                            int M, int N, int K) {
    float f16_centroids[16];
    for (int i = 0; i < 16; i++)
        f16_centroids[i] = fp16_to_float(codebook[i]);

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            __m256 sum8 = _mm256_setzero_ps();
            int k = 0;
            for (; k + 16 <= K; k += 16) {
                const uint8_t* w = packed_indices + ((int64_t)m * N + n) * K / 2 + k / 2;
                const float* a = activations + (int64_t)n * K + k;
                // Prefetch next iteration's data into L1
#if defined(_MSC_VER)
                _mm_prefetch(reinterpret_cast<const char*>(w + 8), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(a + 16), _MM_HINT_T0);
#else
                __builtin_prefetch(w + 8, 0, 3);
                __builtin_prefetch(a + 16, 0, 3);
#endif

                __m128i packed = _mm_loadl_epi64((const __m128i*)w);
                __m128i lo = _mm_and_si128(packed, _mm_set1_epi8(0xF));
                __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0xF));
                __m128i idx16 = _mm_unpacklo_epi8(lo, hi);
                __m256i idx0 = _mm256_cvtepu8_epi32(idx16);
                __m256i idx1 = _mm256_cvtepu8_epi32(_mm_srli_si128(idx16, 8));

                __m256 w0 = _mm256_i32gather_ps(f16_centroids, idx0, 4);
                __m256 w1 = _mm256_i32gather_ps(f16_centroids, idx1, 4);

                __m256 a0 = _mm256_loadu_ps(a);
                __m256 a1 = _mm256_loadu_ps(a + 8);

                sum8 = _mm256_fmadd_ps(w0, a0, sum8);
                sum8 = _mm256_fmadd_ps(w1, a1, sum8);
            }
            __m128 hi = _mm256_extractf128_ps(sum8, 1);
            __m128 lo = _mm256_castps256_ps128(sum8);
            __m128 sum4 = _mm_add_ps(lo, hi);
            sum4 = _mm_hadd_ps(sum4, sum4);
            sum4 = _mm_hadd_ps(sum4, sum4);
            float result = _mm_cvtss_f32(sum4);
            for (; k < K; k++) {
                const uint8_t* w = packed_indices + ((int64_t)m * N + n) * K / 2 + k / 2;
                const float* a_row = activations + (int64_t)n * K;
                uint8_t idx = (k % 2 == 0) ? (w[0] & 0xF) : ((w[0] >> 4) & 0xF);
                result += f16_centroids[idx] * a_row[k];
            }
            output[m * N + n] = result;
        }
    }
}
#endif

void quant4_gemm(const uint8_t* packed_indices, const uint16_t* codebook,
               const float* activations, float* output,
               int M, int N, int K) {
#if defined(__AVX2__)
    quant4_gemm_avx2(packed_indices, codebook, activations, output, M, N, K);
#else
    quant4_gemm_scalar(packed_indices, codebook, activations, output, M, N, K);
#endif
}

} // namespace kernel
} // namespace quant
