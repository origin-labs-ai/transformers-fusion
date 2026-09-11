#include "quant/kernel.h"

#include <algorithm>

#if defined(QUANT_HAS_AVX2)
#include <immintrin.h>
#endif

namespace quant {
namespace kernel {

// C[M,N] = A[M,K] x B[K,N] — scalar reference GEMM (overwrites C)
void scalar_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += A[(size_t)m * K + k] * B[(size_t)k * N + n];
            C[(size_t)m * N + n] = acc;
        }
    }
}

// C[M,N] = A[M,K] x B[N,K]^T — B is row-major {N, K} (e.g. Linear weights {out, in})
void scalar_gemm_bt(const float* A, const float* B, float* C,
                    int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        const float* a_row = A + (size_t)m * K;
        for (int n = 0; n < N; n++) {
            const float* b_row = B + (size_t)n * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += a_row[k] * b_row[k];
            C[(size_t)m * N + n] = acc;
        }
    }
}

#if defined(QUANT_HAS_AVX2)
// AVX2 variant of scalar_gemm_bt: 8-wide dot products over K
void avx2_gemm_bt(const float* A, const float* B, float* C,
                  int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        const float* a_row = A + (size_t)m * K;
        for (int n = 0; n < N; n++) {
            const float* b_row = B + (size_t)n * K;
            __m256 acc = _mm256_setzero_ps();
            int k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 av = _mm256_loadu_ps(a_row + k);
                __m256 bv = _mm256_loadu_ps(b_row + k);
                acc = _mm256_fmadd_ps(av, bv, acc);
            }
            float sum = 0.0f;
            alignas(32) float lanes[8]; _mm256_storeu_ps(lanes, acc); for (int r = 0; r < 8; r++) sum += lanes[r];
            for (; k < K; k++) sum += a_row[k] * b_row[k];
            C[(size_t)m * N + n] = sum;
        }
    }
}
#else
void avx2_gemm_bt(const float* A, const float* B, float* C,
                  int M, int N, int K) {
    scalar_gemm_bt(A, B, C, M, N, K);
}
#endif

// C[M,N] = A[M,K] x B[K,N] — 64x64 block-tiled GEMM (cache-friendly)
void tiled_gemm(const float* A, const float* B, float* C,
                int M, int N, int K) {
    const int T = 64;
    for (int m0 = 0; m0 < M; m0 += T) {
        const int m1 = std::min(m0 + T, M);
        for (int n0 = 0; n0 < N; n0 += T) {
            const int n1 = std::min(n0 + T, N);
            for (int m = m0; m < m1; m++)
                for (int n = n0; n < n1; n++)
                    C[(size_t)m * N + n] = 0.0f;
            for (int k0 = 0; k0 < K; k0 += T) {
                const int k1 = std::min(k0 + T, K);
                for (int m = m0; m < m1; m++) {
                    for (int n = n0; n < n1; n++) {
                        float acc = 0.0f;
                        for (int k = k0; k < k1; k++)
                            acc += A[(size_t)m * K + k] * B[(size_t)k * N + n];
                        C[(size_t)m * N + n] += acc;
                    }
                }
            }
        }
    }
}

#if defined(QUANT_HAS_AVX2)
// C[M,N] = A[M,K] x B[K,N] — AVX2 GEMM, 8-wide dot products
void avx2_gemm(const float* A, const float* B, float* C,
               int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        const float* a_row = A + (size_t)m * K;
        for (int n = 0; n < N; n++) {
            __m256 acc = _mm256_setzero_ps();
            int k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 av = _mm256_loadu_ps(a_row + k);
                __m256 bv = _mm256_loadu_ps(B + (size_t)k * N + n);
                acc = _mm256_fmadd_ps(av, bv, acc);
            }
            float sum = 0.0f;
            alignas(32) float lanes[8]; _mm256_storeu_ps(lanes, acc); for (int r = 0; r < 8; r++) sum += lanes[r];
            for (; k < K; k++) sum += a_row[k] * B[(size_t)k * N + n];
            C[(size_t)m * N + n] = sum;
        }
    }
}
#else
void avx2_gemm(const float* A, const float* B, float* C,
               int M, int N, int K) {
    scalar_gemm(A, B, C, M, N, K);
}
#endif

#if defined(QUANT_HAS_AVX2)
// C[M,N] = A[M,K] x B[K,N] — AVX2 with 64x64 tiling for cache reuse
void avx2_tiled_gemm(const float* A, const float* B, float* C,
                     int M, int N, int K) {
    const int T = 64;
    for (int m0 = 0; m0 < M; m0 += T) {
        const int m1 = std::min(m0 + T, M);
        for (int n0 = 0; n0 < N; n0 += T) {
            const int n1 = std::min(n0 + T, N);
            for (int m = m0; m < m1; m++)
                for (int n = n0; n < n1; n++)
                    C[(size_t)m * N + n] = 0.0f;
            for (int k0 = 0; k0 < K; k0 += T) {
                const int k1 = std::min(k0 + T, K);
                for (int m = m0; m < m1; m++) {
                    const float* a_row = A + (size_t)m * K;
                    for (int n = n0; n < n1; n++) {
                        __m256 acc = _mm256_setzero_ps();
                        int k = k0;
                        for (; k + 8 <= k1; k += 8) {
                            __m256 av = _mm256_loadu_ps(a_row + k);
                            __m256 bv = _mm256_loadu_ps(B + (size_t)k * N + n);
                            acc = _mm256_fmadd_ps(av, bv, acc);
                        }
                        float sum = 0.0f;
                        alignas(32) float lanes[8]; _mm256_storeu_ps(lanes, acc); for (int r = 0; r < 8; r++) sum += lanes[r];
                        for (; k < k1; k++) sum += a_row[k] * B[(size_t)k * N + n];
                        C[(size_t)m * N + n] += sum;
                    }
                }
            }
        }
    }
}
#else
void avx2_tiled_gemm(const float* A, const float* B, float* C,
                     int M, int N, int K) {
    tiled_gemm(A, B, C, M, N, K);
}
#endif

} // namespace kernel
} // namespace quant
