#include "quant/math_tiled.h"
#include "quant/math.h"
#include "quant/tensor.h"

#include <cmath>
#include <cstring>
#include <algorithm>

#if defined(QUANT_AVX2)

#include <immintrin.h>
#include "quant/detail/exp_avx2.h"

namespace quant {
namespace math {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
static inline const float* rd(const Tensor& t) { return t.data<float>(); }
static inline float* wr(Tensor& t) { return t.data<float>(); }

// ---------------------------------------------------------------------------
// AVX2 exp approximation — see quant/detail/exp_avx2.h (DEDUP: was
// triplicated byte-identical in math_avx2/_tensor/_tiled).
// ---------------------------------------------------------------------------
using detail::exp_ps;

// ---------------------------------------------------------------------------
// Scalar inner kernels for edge tiles
// ---------------------------------------------------------------------------
static void micro_scalar_ab(const float* A, const float* B, float* C,
                            int64_t M, int64_t N, int64_t K,
                            int64_t lda, int64_t ldb, int64_t ldc) {
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] += sum;
        }
    }
}

static void micro_scalar_atb(const float* A, const float* B, float* C,
                             int64_t M, int64_t N, int64_t K,
                             int64_t lda, int64_t ldb, int64_t ldc) {
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                sum += A[k * lda + i] * B[k * ldb + j];
            }
            C[i * ldc + j] += sum;
        }
    }
}

static void micro_scalar_abt(const float* A, const float* B, float* C,
                             int64_t M, int64_t N, int64_t K,
                             int64_t lda, int64_t ldb, int64_t ldc) {
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                sum += A[i * lda + k] * B[j * ldb + k];
            }
            C[i * ldc + j] += sum;
        }
    }
}

static void micro_scalar_atbt(const float* A, const float* B, float* C,
                              int64_t M, int64_t N, int64_t K,
                              int64_t lda, int64_t ldb, int64_t ldc) {
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                sum += A[k * lda + i] * B[j * ldb + k];
            }
            C[i * ldc + j] += sum;
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2 inner kernel: 6×16 micro-tile (2 × __m256 per column pair)
// ---------------------------------------------------------------------------
// Software prefetch: pull next iteration's B-row and A-column into L1 cache
// while the CPU executes the current FMA. On Haswell-Skylake, this hides
// ~5-10 cycle DRAM-to-L1 latency by overlapping with the 3-cycle FMA.
static void micro_6x16_ab(const float* A, const float* B, float* C,
                           int64_t M, int64_t N, int64_t K,
                           int64_t lda, int64_t ldb, int64_t ldc) {
    __m256 acc[6][2];
    for (int64_t ii = 0; ii < 6; ii++) {
        acc[ii][0] = _mm256_setzero_ps();
        acc[ii][1] = _mm256_setzero_ps();
    }

    for (int64_t k = 0; k < K; k++) {
        __m256 b0 = _mm256_loadu_ps(B + k * ldb);
        __m256 b1 = _mm256_loadu_ps(B + k * ldb + 8);

        // Prefetch next B-row (64 bytes ahead = 2 cache lines)
        if (k + 1 < K) {
            _mm_prefetch((const char*)(B + (k + 1) * ldb), _MM_HINT_T0);
            _mm_prefetch((const char*)(B + (k + 1) * ldb + 8), _MM_HINT_T0);
        }

        for (int64_t ii = 0; ii < M; ii++) {
            // Prefetch next A column for the same row
            if (k + 1 < K)
                _mm_prefetch((const char*)(A + ii * lda + k + 1), _MM_HINT_T0);
            __m256 a_bc = _mm256_set1_ps(A[ii * lda + k]);
            acc[ii][0] = _mm256_fmadd_ps(a_bc, b0, acc[ii][0]);
            acc[ii][1] = _mm256_fmadd_ps(a_bc, b1, acc[ii][1]);
        }
    }

    // Store results using aligned stores where C is aligned, otherwise unaligned
    for (int64_t ii = 0; ii < M; ii++) {
        __m256 c0 = _mm256_loadu_ps(C + ii * ldc);
        __m256 c1 = _mm256_loadu_ps(C + ii * ldc + 8);
        _mm256_storeu_ps(C + ii * ldc,     _mm256_add_ps(c0, acc[ii][0]));
        _mm256_storeu_ps(C + ii * ldc + 8, _mm256_add_ps(c1, acc[ii][1]));
    }
}

static void micro_6x16_atb(const float* A, const float* B, float* C,
                            int64_t M, int64_t N, int64_t K,
                            int64_t lda, int64_t ldb, int64_t ldc) {
    __m256 acc[6][2];
    for (int64_t ii = 0; ii < 6; ii++) {
        acc[ii][0] = _mm256_setzero_ps();
        acc[ii][1] = _mm256_setzero_ps();
    }

    for (int64_t k = 0; k < K; k++) {
        __m256 b0 = _mm256_loadu_ps(B + k * ldb);
        __m256 b1 = _mm256_loadu_ps(B + k * ldb + 8);

        for (int64_t ii = 0; ii < M; ii++) {
            __m256 a_bc = _mm256_set1_ps(A[k * lda + ii]);
            acc[ii][0] = _mm256_fmadd_ps(a_bc, b0, acc[ii][0]);
            acc[ii][1] = _mm256_fmadd_ps(a_bc, b1, acc[ii][1]);
        }
    }

    for (int64_t ii = 0; ii < M; ii++) {
        __m256 c0 = _mm256_loadu_ps(C + ii * ldc);
        __m256 c1 = _mm256_loadu_ps(C + ii * ldc + 8);
        _mm256_storeu_ps(C + ii * ldc,     _mm256_add_ps(c0, acc[ii][0]));
        _mm256_storeu_ps(C + ii * ldc + 8, _mm256_add_ps(c1, acc[ii][1]));
    }
}

static void micro_6x16_abt(const float* A, const float* B, float* C,
                            int64_t M, int64_t N, int64_t K,
                            int64_t lda, int64_t ldb, int64_t ldc) {
    __m256 acc[6][2];
    for (int64_t ii = 0; ii < 6; ii++) {
        acc[ii][0] = _mm256_setzero_ps();
        acc[ii][1] = _mm256_setzero_ps();
    }

    for (int64_t k = 0; k < K; k++) {
        for (int64_t jj = 0; jj < 2; jj++) {
            __m256 b_bc = _mm256_set1_ps(B[jj * 8 + k * ldb]);
            for (int64_t ii = 0; ii < M; ii++) {
                __m256 a_v = _mm256_loadu_ps(A + ii * lda + k);
                if (jj == 0)
                    acc[ii][0] = _mm256_fmadd_ps(a_v, b_bc, acc[ii][0]);
                else
                    acc[ii][1] = _mm256_fmadd_ps(a_v, b_bc, acc[ii][1]);
            }
        }
    }

    for (int64_t ii = 0; ii < M; ii++) {
        __m256 c0 = _mm256_loadu_ps(C + ii * ldc);
        __m256 c1 = _mm256_loadu_ps(C + ii * ldc + 8);
        _mm256_storeu_ps(C + ii * ldc,     _mm256_add_ps(c0, acc[ii][0]));
        _mm256_storeu_ps(C + ii * ldc + 8, _mm256_add_ps(c1, acc[ii][1]));
    }
}

static void micro_6x16_atbt(const float* A, const float* B, float* C,
                             int64_t M, int64_t N, int64_t K,
                             int64_t lda, int64_t ldb, int64_t ldc) {
    __m256 acc[6][2];
    for (int64_t ii = 0; ii < 6; ii++) {
        acc[ii][0] = _mm256_setzero_ps();
        acc[ii][1] = _mm256_setzero_ps();
    }

    for (int64_t k = 0; k < K; k++) {
        for (int64_t jj = 0; jj < 2; jj++) {
            __m256 b_bc = _mm256_set1_ps(B[jj * 8 + k * ldb]);
            for (int64_t ii = 0; ii < M; ii++) {
                __m256 a_v = _mm256_set1_ps(A[k * lda + ii]);
                if (jj == 0)
                    acc[ii][0] = _mm256_fmadd_ps(a_v, b_bc, acc[ii][0]);
                else
                    acc[ii][1] = _mm256_fmadd_ps(a_v, b_bc, acc[ii][1]);
            }
        }
    }

    for (int64_t ii = 0; ii < M; ii++) {
        __m256 c0 = _mm256_loadu_ps(C + ii * ldc);
        __m256 c1 = _mm256_loadu_ps(C + ii * ldc + 8);
        _mm256_storeu_ps(C + ii * ldc,     _mm256_add_ps(c0, acc[ii][0]));
        _mm256_storeu_ps(C + ii * ldc + 8, _mm256_add_ps(c1, acc[ii][1]));
    }
}

// ---------------------------------------------------------------------------
// Micro-kernel dispatch by transpose variant
// ---------------------------------------------------------------------------
typedef void (*MicroFn)(const float*, const float*, float*,
                        int64_t, int64_t, int64_t,
                        int64_t, int64_t, int64_t);

static MicroFn select_micro_ab(Transpose trans) {
    switch (trans) {
        case Transpose::None:   return micro_6x16_ab;
        case Transpose::TransA: return micro_6x16_atb;
        case Transpose::TransB: return micro_6x16_abt;
        case Transpose::TransAB:return micro_6x16_atbt;
    }
    return micro_6x16_ab;
}

static MicroFn select_micro_scalar(Transpose trans) {
    switch (trans) {
        case Transpose::None:   return micro_scalar_ab;
        case Transpose::TransA: return micro_scalar_atb;
        case Transpose::TransB: return micro_scalar_abt;
        case Transpose::TransAB:return micro_scalar_atbt;
    }
    return micro_scalar_ab;
}

// ---------------------------------------------------------------------------
// Panel packing: copy a rectangular panel from A/B into contiguous buffer
// with optional transpose, enabling stride-1 loads in the micro-kernel.
// ---------------------------------------------------------------------------
static void pack_panel_a(const float* A, int64_t lda,
                         float* panel, int64_t rows, int64_t cols,
                         Transpose trans) {
    if (trans == Transpose::None || trans == Transpose::TransB) {
        for (int64_t i = 0; i < rows; i++)
            for (int64_t k = 0; k < cols; k++)
                panel[i * cols + k] = A[i * lda + k];
    } else {
        for (int64_t i = 0; i < rows; i++)
            for (int64_t k = 0; k < cols; k++)
                panel[i * cols + k] = A[k * lda + i];
    }
}

static void pack_panel_b(const float* B, int64_t ldb,
                         float* panel, int64_t rows, int64_t cols,
                         Transpose trans) {
    if (trans == Transpose::None || trans == Transpose::TransA) {
        for (int64_t k = 0; k < rows; k++)
            for (int64_t j = 0; j < cols; j++)
                panel[k * cols + j] = B[k * ldb + j];
    } else {
        for (int64_t k = 0; k < rows; k++)
            for (int64_t j = 0; j < cols; j++)
                panel[k * cols + j] = B[j * ldb + k];
    }
}

// ---------------------------------------------------------------------------
// Core tiled GEMM: C = alpha * op(A) * op(B) + beta * C
//
// Uses 64×64 tiles in M×N, 64 in K. Inner kernel is 6×16 with AVX2 FMA.
// Panels are packed into L1-sized buffers for contiguous access.
// ---------------------------------------------------------------------------
void gemm_tiled(float alpha,
                const float* A, int64_t lda,
                const float* B, int64_t ldb,
                float beta,
                float* C, int64_t ldc,
                int64_t M, int64_t N, int64_t K,
                Transpose trans,
                bool col_major) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    int64_t lda_use = col_major ? K : lda;
    int64_t ldb_use = col_major ? N : ldb;
    int64_t ldc_use = col_major ? M : ldc;
    const float* A_use = col_major ? A : A;
    const float* B_use = col_major ? B : B;
    float* C_use = col_major ? C : C;

    if (beta == 0.0f) {
        for (int64_t i = 0; i < M; i++)
            std::memset(C_use + i * ldc_use, 0, N * sizeof(float));
    } else if (beta != 1.0f) {
        for (int64_t i = 0; i < M; i++)
            for (int64_t j = 0; j < N; j++)
                C_use[i * ldc_use + j] *= beta;
    }

    bool trans_a = (trans == Transpose::TransA || trans == Transpose::TransAB);
    bool trans_b = (trans == Transpose::TransB || trans == Transpose::TransAB);

    MicroFn micro = select_micro_ab(trans);

    // 64-byte cache-line aligned panels for optimal SIMD load/store throughput.
    // Misaligned accesses can cost 1-2 extra cycles per _mm256_loadu_ps on some microarchitectures.
    alignas(64) float panel_A[TILE_M * TILE_K];
    alignas(64) float panel_B[TILE_K * TILE_N];

    for (int64_t i0 = 0; i0 < M; i0 += TILE_M) {
        int64_t i_end = (i0 + TILE_M < M) ? i0 + TILE_M : M;
        int64_t m_tile = i_end - i0;

        for (int64_t j0 = 0; j0 < N; j0 += TILE_N) {
            int64_t j_end = (j0 + TILE_N < N) ? j0 + TILE_N : N;
            int64_t n_tile = j_end - j0;

            for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
                int64_t k_end = (k0 + TILE_K < K) ? k0 + TILE_K : K;
                int64_t k_tile = k_end - k0;

                const float* A_panel = A_use + i0 * lda_use + k0;
                const float* B_panel = B_use + k0 * ldb_use + j0;

                int64_t panel_a_rows = trans_a ? k_tile : m_tile;
                int64_t panel_a_cols = trans_a ? m_tile : k_tile;
                int64_t panel_b_rows = trans_b ? n_tile : k_tile;
                int64_t panel_b_cols = trans_b ? k_tile : n_tile;

                pack_panel_a(A_panel, lda_use, panel_A,
                             panel_a_rows, panel_a_cols,
                             (trans_a) ? Transpose::TransA : Transpose::None);
                pack_panel_b(B_panel, ldb_use, panel_B,
                             panel_b_rows, panel_b_cols,
                             (trans_b) ? Transpose::TransB : Transpose::None);

                if (m_tile >= 6 && n_tile >= 16) {
                    micro(panel_A, panel_B, C_use + i0 * ldc_use + j0,
                          m_tile, n_tile, k_tile,
                          trans_a ? m_tile : k_tile,
                          trans_b ? k_tile : n_tile,
                          ldc_use);
                } else {
                    MicroFn micro_s = select_micro_scalar(trans);
                    micro_s(A_panel, B_panel, C_use + i0 * ldc_use + j0,
                            m_tile, n_tile, k_tile,
                            lda_use, ldb_use, ldc_use);
                }

                if (k0 + TILE_K < K) {
                    const float* next_A = A_use + i0 * lda_use + (k0 + TILE_K);
                    const float* next_B = B_use + (k0 + TILE_K) * ldb_use + j0;
                    _mm_prefetch((const char*)next_A, _MM_HINT_T0);
                    _mm_prefetch((const char*)next_B, _MM_HINT_T0);
                }
            }
        }
    }

    if (alpha != 1.0f) {
        for (int64_t i = 0; i < M; i++)
            for (int64_t j = 0; j < N; j++)
                C_use[i * ldc_use + j] *= alpha;
    }
}

// ---------------------------------------------------------------------------
// Fused bias addition
// ---------------------------------------------------------------------------
void gemm_tiled_bias(float alpha,
                     const float* A, int64_t lda,
                     const float* B, int64_t ldb,
                     float beta,
                     float* C, int64_t ldc,
                     int64_t M, int64_t N, int64_t K,
                     const float* bias,
                     Transpose trans) {
    gemm_tiled(alpha, A, lda, B, ldb, beta, C, ldc, M, N, K, trans);

    if (bias) {
        __m256 alphav = _mm256_set1_ps(alpha);
        for (int64_t i = 0; i < M; i++) {
            int64_t j = 0;
            for (; j + 8 <= N; j += 8) {
                __m256 c = _mm256_loadu_ps(C + i * ldc + j);
                __m256 b = _mm256_loadu_ps(bias + j);
                _mm256_storeu_ps(C + i * ldc + j, _mm256_add_ps(c, b));
            }
            for (; j < N; j++)
                C[i * ldc + j] += bias[j];
        }
    }
}

// ---------------------------------------------------------------------------
// Fused activation kernels (in-place on C tile)
// ---------------------------------------------------------------------------
static void apply_relu(float* C, int64_t count) {
    __m256 zeros = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 v = _mm256_loadu_ps(C + i);
        _mm256_storeu_ps(C + i, _mm256_max_ps(v, zeros));
    }
    for (; i < count; i++) C[i] = C[i] > 0.0f ? C[i] : 0.0f;
}

static void apply_gelu(float* C, int64_t count) {
    const float s = 0.7071067811865475f;
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 sqrt2_inv = _mm256_set1_ps(s);
    int64_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 v = _mm256_loadu_ps(C + i);
        __m256 x = _mm256_mul_ps(v, sqrt2_inv);
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);
        __m256 x5 = _mm256_mul_ps(x3, x2);
        __m256 x7 = _mm256_mul_ps(x5, x2);
        __m256 x9 = _mm256_mul_ps(x7, x2);
        __m256 erv = _mm256_sub_ps(x, _mm256_mul_ps(x3, _mm256_set1_ps(1.0f / 3.0f)));
        erv = _mm256_add_ps(erv, _mm256_mul_ps(x5, _mm256_set1_ps(1.0f / 10.0f)));
        erv = _mm256_sub_ps(erv, _mm256_mul_ps(x7, _mm256_set1_ps(1.0f / 42.0f)));
        erv = _mm256_add_ps(erv, _mm256_mul_ps(x9, _mm256_set1_ps(1.0f / 216.0f)));
        __m256 one_p = _mm256_add_ps(one, erv);
        _mm256_storeu_ps(C + i, _mm256_mul_ps(half, _mm256_mul_ps(v, one_p)));
    }
    for (; i < count; i++)
        C[i] = 0.5f * C[i] * (1.0f + std::erf(C[i] * s));
}

static void apply_silu(float* C, int64_t count) {
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 clamp_hi = _mm256_set1_ps(88.376f);
    __m256 clamp_lo = _mm256_set1_ps(-88.376f);
    int64_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 v = _mm256_loadu_ps(C + i);
        __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), v);
        __m256 cl = _mm256_min_ps(_mm256_max_ps(neg, clamp_lo), clamp_hi);
        __m256 e = exp_ps(cl);
        __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
        _mm256_storeu_ps(C + i, _mm256_mul_ps(v, sig));
    }
    for (; i < count; i++)
        C[i] = C[i] / (1.0f + std::exp(-C[i]));
}

static void apply_activation(float* C, int64_t count, Activation act) {
    switch (act) {
        case Activation::None: break;
        case Activation::ReLU: apply_relu(C, count); break;
        case Activation::GELU: apply_gelu(C, count); break;
        case Activation::SiLU: apply_silu(C, count); break;
    }
}

// ---------------------------------------------------------------------------
// Fused GEMM + activation
// ---------------------------------------------------------------------------
void gemm_tiled_act(float alpha,
                    const float* A, int64_t lda,
                    const float* B, int64_t ldb,
                    float beta,
                    float* C, int64_t ldc,
                    int64_t M, int64_t N, int64_t K,
                    Activation act,
                    Transpose trans) {
    gemm_tiled(alpha, A, lda, B, ldb, beta, C, ldc, M, N, K, trans);
    if (act != Activation::None)
        apply_activation(C, M * ldc, act);
}

// ---------------------------------------------------------------------------
// Fused GEMM + bias + activation
// ---------------------------------------------------------------------------
void gemm_tiled_bias_act(float alpha,
                         const float* A, int64_t lda,
                         const float* B, int64_t ldb,
                         float beta,
                         float* C, int64_t ldc,
                         int64_t M, int64_t N, int64_t K,
                         const float* bias,
                         Activation act,
                         Transpose trans) {
    gemm_tiled_bias(alpha, A, lda, B, ldb, beta, C, ldc, M, N, K, bias, trans);
    if (act != Activation::None)
        apply_activation(C, M * ldc, act);
}

// ---------------------------------------------------------------------------
// Tensor-level convenience wrappers
// ---------------------------------------------------------------------------
void gemm_tiled(float alpha, const Tensor& A, const Tensor& B,
                float beta, Tensor& C, Transpose trans) {
    QUANT_CHECK(A.shape().rank == 2 && B.shape().rank == 2 && C.shape().rank == 2,
              "gemm_tiled: all inputs must be 2D");
    int64_t M = A.shape().dims[0];
    int64_t K = A.shape().dims[1];
    int64_t N = B.shape().dims[1];
    QUANT_CHECK(B.shape().dims[0] == K, "gemm_tiled: A cols != B rows");
    QUANT_CHECK(C.shape().dims[0] == M && C.shape().dims[1] == N,
              "gemm_tiled: C shape mismatch");

    gemm_tiled(alpha,
               rd(A), K,
               rd(B), N,
               beta,
               wr(C), N,
               M, N, K, trans);
}

void gemm_tiled_bias(float alpha, const Tensor& A, const Tensor& B,
                     float beta, Tensor& C, const Tensor& bias,
                     Transpose trans) {
    QUANT_CHECK(C.shape().rank == 2, "gemm_tiled_bias: C must be 2D");
    int64_t N = C.shape().dims[1];
    QUANT_CHECK(bias.numel() == N, "gemm_tiled_bias: bias length != N cols");

    gemm_tiled(alpha, A, B, beta, C, trans);

    int64_t M = C.shape().dims[0];
    float* pc = wr(C);
    const float* pb = rd(bias);
    for (int64_t i = 0; i < M; i++) {
        int64_t j = 0;
        for (; j + 8 <= N; j += 8) {
            __m256 c = _mm256_loadu_ps(pc + i * N + j);
            __m256 b = _mm256_loadu_ps(pb + j);
            _mm256_storeu_ps(pc + i * N + j, _mm256_add_ps(c, b));
        }
        for (; j < N; j++)
            pc[i * N + j] += pb[j];
    }
}

void gemm_tiled_act(float alpha, const Tensor& A, const Tensor& B,
                    float beta, Tensor& C,
                    Activation act, Transpose trans) {
    gemm_tiled(alpha, A, B, beta, C, trans);
    if (act != Activation::None)
        apply_activation(wr(C), C.numel(), act);
}

void gemm_tiled_bias_act(float alpha, const Tensor& A, const Tensor& B,
                         float beta, Tensor& C, const Tensor& bias,
                         Activation act, Transpose trans) {
    gemm_tiled_bias(alpha, A, B, beta, C, bias, trans);
    if (act != Activation::None)
        apply_activation(wr(C), C.numel(), act);
}

} // namespace math
} // namespace quant

#else // !QUANT_AVX2 — scalar fallbacks

namespace quant {
namespace math {

static inline const float* rd(const Tensor& t) { return t.data<float>(); }
static inline float* wr(Tensor& t) { return t.data<float>(); }

void gemm_tiled(float alpha,
                const float* A, int64_t lda,
                const float* B, int64_t ldb,
                float beta,
                float* C, int64_t ldc,
                int64_t M, int64_t N, int64_t K,
                Transpose trans,
                bool col_major) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    if (beta == 0.0f) {
        for (int64_t i = 0; i < M; i++)
            std::memset(C + i * ldc, 0, N * sizeof(float));
    } else if (beta != 1.0f) {
        for (int64_t i = 0; i < M; i++)
            for (int64_t j = 0; j < N; j++)
                C[i * ldc + j] *= beta;
    }

    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                float a_val, b_val;
                switch (trans) {
                    case Transpose::None:
                        a_val = A[i * lda + k];
                        b_val = B[k * ldb + j];
                        break;
                    case Transpose::TransA:
                        a_val = A[k * lda + i];
                        b_val = B[k * ldb + j];
                        break;
                    case Transpose::TransB:
                        a_val = A[i * lda + k];
                        b_val = B[j * ldb + k];
                        break;
                    case Transpose::TransAB:
                        a_val = A[k * lda + i];
                        b_val = B[j * ldb + k];
                        break;
                }
                sum += a_val * b_val;
            }
            C[i * ldc + j] += sum;
        }
    }

    if (alpha != 1.0f) {
        for (int64_t i = 0; i < M; i++)
            for (int64_t j = 0; j < N; j++)
                C[i * ldc + j] *= alpha;
    }
}

void gemm_tiled_bias(float alpha,
                     const float* A, int64_t lda,
                     const float* B, int64_t ldb,
                     float beta,
                     float* C, int64_t ldc,
                     int64_t M, int64_t N, int64_t K,
                     const float* bias,
                     Transpose trans) {
    gemm_tiled(alpha, A, lda, B, ldb, beta, C, ldc, M, N, K, trans);
    if (bias) {
        for (int64_t i = 0; i < M; i++)
            for (int64_t j = 0; j < N; j++)
                C[i * ldc + j] += bias[j];
    }
}

void gemm_tiled_act(float alpha,
                    const float* A, int64_t lda,
                    const float* B, int64_t ldb,
                    float beta,
                    float* C, int64_t ldc,
                    int64_t M, int64_t N, int64_t K,
                    Activation act,
                    Transpose trans) {
    gemm_tiled(alpha, A, lda, B, ldb, beta, C, ldc, M, N, K, trans);
    if (act == Activation::ReLU) {
        for (int64_t i = 0; i < M * ldc; i++)
            C[i] = C[i] > 0.0f ? C[i] : 0.0f;
    } else if (act == Activation::GELU) {
        const float s = 0.7071067811865475f;
        for (int64_t i = 0; i < M * ldc; i++)
            C[i] = 0.5f * C[i] * (1.0f + std::erf(C[i] * s));
    } else if (act == Activation::SiLU) {
        for (int64_t i = 0; i < M * ldc; i++)
            C[i] = C[i] / (1.0f + std::exp(-C[i]));
    }
}

void gemm_tiled_bias_act(float alpha,
                         const float* A, int64_t lda,
                         const float* B, int64_t ldb,
                         float beta,
                         float* C, int64_t ldc,
                         int64_t M, int64_t N, int64_t K,
                         const float* bias,
                         Activation act,
                         Transpose trans) {
    gemm_tiled_bias(alpha, A, lda, B, ldb, beta, C, ldc, M, N, K, bias, trans);
    if (act == Activation::ReLU) {
        for (int64_t i = 0; i < M * ldc; i++)
            C[i] = C[i] > 0.0f ? C[i] : 0.0f;
    } else if (act == Activation::GELU) {
        const float s = 0.7071067811865475f;
        for (int64_t i = 0; i < M * ldc; i++)
            C[i] = 0.5f * C[i] * (1.0f + std::erf(C[i] * s));
    } else if (act == Activation::SiLU) {
        for (int64_t i = 0; i < M * ldc; i++)
            C[i] = C[i] / (1.0f + std::exp(-C[i]));
    }
}

void gemm_tiled(float alpha, const Tensor& A, const Tensor& B,
                float beta, Tensor& C, Transpose trans) {
    QUANT_CHECK(A.shape().rank == 2 && B.shape().rank == 2 && C.shape().rank == 2,
              "gemm_tiled: all inputs must be 2D");
    int64_t M = A.shape().dims[0];
    int64_t K = A.shape().dims[1];
    int64_t N = B.shape().dims[1];
    gemm_tiled(alpha, rd(A), K, rd(B), N, beta, wr(C), N, M, N, K, trans);
}

void gemm_tiled_bias(float alpha, const Tensor& A, const Tensor& B,
                     float beta, Tensor& C, const Tensor& bias,
                     Transpose trans) {
    QUANT_CHECK(C.shape().rank == 2, "gemm_tiled_bias: C must be 2D");
    int64_t N = C.shape().dims[1];
    gemm_tiled(alpha, A, B, beta, C, trans);
    int64_t M = C.shape().dims[0];
    float* pc = wr(C);
    const float* pb = rd(bias);
    for (int64_t i = 0; i < M; i++)
        for (int64_t j = 0; j < N; j++)
            pc[i * N + j] += pb[j];
}

void gemm_tiled_act(float alpha, const Tensor& A, const Tensor& B,
                    float beta, Tensor& C,
                    Activation act, Transpose trans) {
    gemm_tiled(alpha, A, B, beta, C, trans);
    if (act == Activation::ReLU) {
        for (int64_t i = 0; i < C.numel(); i++) {
            float* p = wr(C);
            p[i] = p[i] > 0.0f ? p[i] : 0.0f;
        }
    }
}

void gemm_tiled_bias_act(float alpha, const Tensor& A, const Tensor& B,
                         float beta, Tensor& C, const Tensor& bias,
                         Activation act, Transpose trans) {
    gemm_tiled_bias(alpha, A, B, beta, C, bias, trans);
    if (act == Activation::ReLU) {
        for (int64_t i = 0; i < C.numel(); i++) {
            float* p = wr(C);
            p[i] = p[i] > 0.0f ? p[i] : 0.0f;
        }
    }
}

} // namespace math
} // namespace quant

#endif // QUANT_AVX2
