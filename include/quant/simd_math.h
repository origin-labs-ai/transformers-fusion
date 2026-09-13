#pragma once
// ============================================================================
// PILLAR 2: SIMD Math Kernels — AVX2/AVX-512 for 512+ tok/s inference
// ============================================================================
// WHY: Scalar math on 14B params = unusably slow. SIMD kernels process
// 8 floats (AVX2) or 16 floats (AVX-512) per instruction.
//
// MEMORY ALIGNMENT: All SIMD operations require 64-byte aligned data to
// prevent cache-line splits. Our AlignedAllocator guarantees this.
//
// CACHE STRATEGY: Each kernel is tiled to fit within L1 (32KB) and L2 (256KB)
// cache sizes. The tile dimensions are tuned for Ryzen 5600GT's cache layout.
//
// TARGET: 512+ tokens/sec on CPU for 0.4B dense model inference.
// ============================================================================

#include "quant/tensor.h"
#include <cstdint>
#include <cmath>

#if defined(__AVX2__) || defined(_MSC_VER)
#include <immintrin.h>
#define QUANT_HAS_AVX2 1
#endif

#if defined(__AVX512F__)
#include <immintrin.h>
#define QUANT_HAS_AVX512 1
#endif

namespace quant {
namespace simd {

#if defined(__AVX2__) || defined(_MSC_VER)
static inline __m256 quant_exp_ps(__m256 x) {
    __m256 ln2 = _mm256_set1_ps(1.4426950408889634f);
    __m256 t = _mm256_mul_ps(x, ln2);
    __m256 t_floor = _mm256_floor_ps(t);
    __m256 frac = _mm256_sub_ps(t, t_floor);
    __m256 c1 = _mm256_set1_ps(0.6931471805599453f);
    __m256 c2 = _mm256_set1_ps(0.240226506959101f);
    __m256 c3 = _mm256_set1_ps(0.055504108664672f);
    __m256 c4 = _mm256_set1_ps(0.009618129107628f);
    __m256 c5 = _mm256_set1_ps(0.001333355814643f);
    __m256 p = _mm256_fmadd_ps(c5, frac, c4);
    p = _mm256_fmadd_ps(p, frac, c3);
    p = _mm256_fmadd_ps(p, frac, c2);
    p = _mm256_fmadd_ps(p, frac, c1);
    p = _mm256_fmadd_ps(p, frac, _mm256_set1_ps(1.0f));
    __m256i exp_part = _mm256_slli_epi32(
        _mm256_add_epi32(_mm256_cvttps_epi32(t_floor), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(exp_part));
}
#endif

// ============================================================================
// RMSNorm — Root Mean Square Layer Normalization
// ============================================================================
// y = x * gamma / sqrt(mean(x^2) + eps)
//
// AVX2: Processes 8 floats per iteration.
// Memory: Reads x once (sequential), writes y once. Fits in L1 for small dims.
// ============================================================================

void rms_norm_avx2(const float* x, const float* gamma, float* y,
                   int64_t n, float eps = 1e-5f);

void rms_norm_avx512(const float* x, const float* gamma, float* y,
                     int64_t n, float eps = 1e-5f);

// Scalar fallback
void rms_norm_scalar(const float* x, const float* gamma, float* y,
                     int64_t n, float eps = 1e-5f);

// Auto-dispatch based on CPU features
void rms_norm(const float* x, const float* gamma, float* y,
              int64_t n, float eps = 1e-5f);

// ============================================================================
// SwiGLU — Swish-Gated Linear Unit activation
// ============================================================================
// SwiGLU(x, W_gate, W_up) = silu(x @ W_gate) * (x @ W_up)
//
// In-place variant: silu(gate) *= up, then project down
// AVX2: Uses _mm256_mul_ps + _mm256_fmadd_ps for fused multiply-add
// ============================================================================

void swiglu_avx2(const float* gate, const float* up, float* output,
                 int64_t n);

void swiglu_avx512(const float* gate, const float* up, float* output,
                   int64_t n);

void swiglu_scalar(const float* gate, const float* up, float* output,
                   int64_t n);

void swiglu(const float* gate, const float* up, float* output, int64_t n);

// ============================================================================
// GeGLU — GELU-Gated Linear Unit activation
// ============================================================================
// GeGLU(x, W_gate, W_up) = gelu(x @ W_gate) * (x @ W_up)
//
// Similar to SwiGLU but uses GELU instead of SiLU as the gating function.
// Used in PaLM and Gemma architectures.
// ============================================================================

void geglu_avx2(const float* gate, const float* up, float* output,
                int64_t n);

void geglu_avx512(const float* gate, const float* up, float* output,
                  int64_t n);

void geglu_scalar(const float* gate, const float* up, float* output,
                  int64_t n);

void geglu(const float* gate, const float* up, float* output, int64_t n);

// ============================================================================
// RoPE — Rotary Position Embeddings
// ============================================================================
// Applies rotation matrix to Q and K tensors for position encoding.
// RoPE(x, pos) = x * cos(pos * theta) + rotate_half(x) * sin(pos * theta)
//
// rotate_half: [x0,x1,x2,x3] -> [-x1,x0,-x3,x2]
//
// AVX2: Processes 4 complex pairs (8 floats) per iteration.
// Memory: Read-only on cos/sin cache (fits in L1 for 1M context).
// ============================================================================

void rope_avx2(float* q, float* k,
               const float* cos_cache, const float* sin_cache,
               int64_t head_dim, int64_t seq_len, int64_t pos_offset = 0);

void rope_avx512(float* q, float* k,
                 const float* cos_cache, const float* sin_cache,
                 int64_t head_dim, int64_t seq_len, int64_t pos_offset = 0);

void rope_scalar(float* q, float* k,
                 const float* cos_cache, const float* sin_cache,
                 int64_t head_dim, int64_t seq_len, int64_t pos_offset = 0);

void rope(float* q, float* k,
          const float* cos_cache, const float* sin_cache,
          int64_t head_dim, int64_t seq_len, int64_t pos_offset = 0);

// Precompute RoPE frequency cache: freq[i] = 1 / (theta^(2i/dim))
void rope_precompute_freqs(float* cos_out, float* sin_out,
                           int64_t dim, int64_t max_seq_len,
                           float theta = 10000.0f);

// ============================================================================
// Softmax — Numerically stable softmax along last axis
// ============================================================================
// Uses max-subtract trick for stability.
// AVX2: Parallel max-reduction + exp + sum-reduction.
// ============================================================================

void softmax_avx2(const float* x, float* y, int64_t rows, int64_t cols);
void softmax_avx512(const float* x, float* y, int64_t rows, int64_t cols);
void softmax_scalar(const float* x, float* y, int64_t rows, int64_t cols);
void softmax(const float* x, float* y, int64_t rows, int64_t cols);

// ============================================================================
// GEMV — General Matrix-Vector multiply (for linear layers)
// ============================================================================
// y = alpha * A * x + beta * y
// A: [M, K] row-major, x: [K], y: [M]
//
// AVX2: Each output element is a dot product of a row of A with x.
// We process 8 elements of x per AVX2 iteration.
// ============================================================================

void gemv_avx2(const float* A, const float* x, float* y,
               int64_t M, int64_t K, float alpha = 1.0f, float beta = 0.0f);

void gemv_avx512(const float* A, const float* x, float* y,
                 int64_t M, int64_t K, float alpha = 1.0f, float beta = 0.0f);

void gemv_scalar(const float* A, const float* x, float* y,
                 int64_t M, int64_t K, float alpha = 1.0f, float beta = 0.0f);

void gemv(const float* A, const float* x, float* y,
          int64_t M, int64_t K, float alpha = 1.0f, float beta = 0.0f);

// ============================================================================
// Tiled GEMM — Cache-efficient matrix multiply
// ============================================================================
// C = alpha * A * B + beta * C
// A: [M,K], B: [K,N], C: [M,N] — all row-major
//
// Tile sizes tuned for L1 (32KB) and L2 (256KB) cache:
//   Tile: 64x64 — fits 64*64*4 = 16KB in L1
//   K-tile: 64 — 64*64*4*2 = 32KB for A_tile + B_tile in L1
// ============================================================================

void tiled_gemm_avx2(const float* A, const float* B, float* C,
                     int64_t M, int64_t N, int64_t K,
                     float alpha = 1.0f, float beta = 0.0f);

void tiled_gemm_avx512(const float* A, const float* B, float* C,
                       int64_t M, int64_t N, int64_t K,
                       float alpha = 1.0f, float beta = 0.0f);

void tiled_gemm_scalar(const float* A, const float* B, float* C,
                       int64_t M, int64_t N, int64_t K,
                       float alpha = 1.0f, float beta = 0.0f);

void tiled_gemm(const float* A, const float* B, float* C,
                int64_t M, int64_t N, int64_t K,
                float alpha = 1.0f, float beta = 0.0f);

// ============================================================================
// CPU Feature Detection
// ============================================================================

struct CpuFeatures {
    bool has_avx2 = false;
    bool has_avx512 = false;
    bool has_fma = false;
    bool has_vnni = false;  // Vector Neural Network Instructions (INT8 GEMM)
};

CpuFeatures detect_cpu_features();

} // namespace simd
} // namespace quant

#ifdef QUANT_HAS_AVX2
#undef QUANT_HAS_AVX2
#endif
#ifdef QUANT_HAS_AVX512
#undef QUANT_HAS_AVX512
#endif
