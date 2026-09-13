// ============================================================================
// PILLAR 2: SIMD Math Kernels — AVX2/AVX-512 Implementation
// ============================================================================

#include "quant/simd_math.h"
#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef _MSC_VER
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

namespace quant {
namespace simd {

// ============================================================================
// CPU Feature Detection
// ============================================================================

CpuFeatures detect_cpu_features() {
    CpuFeatures feat{};
#ifdef _MSC_VER
    int cpuInfo[4];
    __cpuid(cpuInfo, 7);
    feat.has_avx2 = (cpuInfo[1] & (1 << 5)) != 0;
    feat.has_fma = (cpuInfo[1] & (1 << 12)) != 0;
    feat.has_avx512 = (cpuInfo[1] & (1 << 16)) != 0;
    feat.has_vnni = (cpuInfo[1] & (1 << 11)) != 0;
#elif defined(__x86_64__) || defined(_M_X64)
    unsigned int cpuInfo[4];
    __get_cpuid_count(7, 0, &cpuInfo[0], &cpuInfo[1], &cpuInfo[2], &cpuInfo[3]);
    feat.has_avx2 = (cpuInfo[1] & (1 << 5)) != 0;
    feat.has_fma = (cpuInfo[1] & (1 << 12)) != 0;
    feat.has_avx512 = (cpuInfo[1] & (1 << 16)) != 0;
    feat.has_vnni = (cpuInfo[1] & (1 << 11)) != 0;
#endif
    return feat;
}

// ============================================================================
// RMSNorm — Root Mean Square Layer Normalization
// ============================================================================

void rms_norm_scalar(const float* x, const float* gamma, float* y,
                     int64_t n, float eps) {
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        sum_sq += x[i] * x[i];
    }
    float rms = std::sqrt(sum_sq / n + eps);
    float inv_rms = 1.0f / rms;
    for (int64_t i = 0; i < n; ++i) {
        y[i] = x[i] * inv_rms * gamma[i];
    }
}

#ifdef QUANT_HAS_AVX2
void rms_norm_avx2(const float* x, const float* gamma, float* y,
                   int64_t n, float eps) {
    int64_t i = 0;

    // Phase 1: Compute sum of squares using AVX2
    __m256 sum_sq = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 xi = _mm256_loadu_ps(x + i);
        sum_sq = _mm256_fmadd_ps(xi, xi, sum_sq);
    }
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum_sq, 1);
    __m128 lo = _mm256_castps256_ps128(sum_sq);
    __m128 s = _mm_add_ps(hi, lo);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float total = _mm_cvtss_f32(s);
    // Handle remainder
    for (; i < n; ++i) total += x[i] * x[i];

    float rms = std::sqrt(total / n + eps);
    float inv_rms = 1.0f / rms;
    __m256 v_inv_rms = _mm256_set1_ps(inv_rms);

    // Phase 2: Multiply x * gamma * inv_rms
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xi = _mm256_loadu_ps(x + i);
        __m256 gi = _mm256_loadu_ps(gamma + i);
        __m256 yi = _mm256_mul_ps(xi, gi);
        yi = _mm256_mul_ps(yi, v_inv_rms);
        _mm256_storeu_ps(y + i, yi);
    }
    for (; i < n; ++i) {
        y[i] = x[i] * gamma[i] * inv_rms;
    }
}
#endif

#ifdef QUANT_HAS_AVX512
void rms_norm_avx512(const float* x, const float* gamma, float* y,
                     int64_t n, float eps) {
    int64_t i = 0;
    __m512 sum_sq = _mm512_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        __m512 xi = _mm512_loadu_ps(x + i);
        sum_sq = _mm512_fmadd_ps(xi, xi, sum_sq);
    }
    float total = _mm512_reduce_add_ps(sum_sq);
    for (; i < n; ++i) total += x[i] * x[i];

    float rms = std::sqrt(total / n + eps);
    float inv_rms = 1.0f / rms;
    __m512 v_inv_rms = _mm512_set1_ps(inv_rms);

    i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 xi = _mm512_loadu_ps(x + i);
        __m512 gi = _mm512_loadu_ps(gamma + i);
        __m512 yi = _mm512_mul_ps(xi, gi);
        yi = _mm512_mul_ps(yi, v_inv_rms);
        _mm512_storeu_ps(y + i, yi);
    }
    for (; i < n; ++i) {
        y[i] = x[i] * gamma[i] * inv_rms;
    }
}
#endif

void rms_norm(const float* x, const float* gamma, float* y,
              int64_t n, float eps) {
    static CpuFeatures feat = detect_cpu_features();
#ifdef QUANT_HAS_AVX512
    if (feat.has_avx512) return rms_norm_avx512(x, gamma, y, n, eps);
#endif
#ifdef QUANT_HAS_AVX2
    if (feat.has_avx2) return rms_norm_avx2(x, gamma, y, n, eps);
#endif
    rms_norm_scalar(x, gamma, y, n, eps);
}

// ============================================================================
// SwiGLU — Swish-Gated Linear Unit
// ============================================================================

void swiglu_scalar(const float* gate, const float* up, float* output, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        float g = gate[i];
        float s = g / (1.0f + std::exp(-g)); // silu
        output[i] = s * up[i];
    }
}

#ifdef QUANT_HAS_AVX2
void swiglu_avx2(const float* gate, const float* up, float* output, int64_t n) {
    __m256 one = _mm256_set1_ps(1.0f);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 u = _mm256_loadu_ps(up + i);
        // silu: g * sigmoid(g) = g / (1 + exp(-g))
        float neg_vals[8], exp_vals[8];
        _mm256_storeu_ps(neg_vals, _mm256_sub_ps(_mm256_setzero_ps(), g));
        for (int ei = 0; ei < 8; ei++) exp_vals[ei] = std::exp(neg_vals[ei]);
        __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, _mm256_loadu_ps(exp_vals)));
        __m256 swish = _mm256_mul_ps(g, sig);
        __m256 out = _mm256_mul_ps(swish, u);
        _mm256_storeu_ps(output + i, out);
    }
    for (; i < n; ++i) {
        float s = gate[i] / (1.0f + std::exp(-gate[i]));
        output[i] = s * up[i];
    }
}
#endif

void swiglu(const float* gate, const float* up, float* output, int64_t n) {
    static CpuFeatures feat = detect_cpu_features();
#ifdef QUANT_HAS_AVX2
    if (feat.has_avx2) return swiglu_avx2(gate, up, output, n);
#endif
    swiglu_scalar(gate, up, output, n);
}

void geglu_scalar(const float* gate, const float* up, float* output, int64_t n) {
    const float s = 0.7071067811865475f;
    for (int64_t i = 0; i < n; ++i) {
        float g = gate[i];
        float gel = 0.5f * g * (1.0f + std::erf(g * s));
        output[i] = gel * up[i];
    }
}

#ifdef QUANT_HAS_AVX2
void geglu_avx2(const float* gate, const float* up, float* output, int64_t n) {
    const float s = 0.7071067811865475f;
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float gv[8], uv[8], rv[8];
        std::memcpy(gv, gate + i, 8 * sizeof(float));
        std::memcpy(uv, up + i, 8 * sizeof(float));
        for (int j = 0; j < 8; j++) rv[j] = 0.5f * gv[j] * (1.0f + std::erf(gv[j] * s)) * uv[j];
        std::memcpy(output + i, rv, 8 * sizeof(float));
    }
    for (; i < n; ++i) { float g = gate[i]; output[i] = 0.5f * g * (1.0f + std::erf(g * 0.7071067811865475f)) * up[i]; }
}
#endif

void geglu(const float* gate, const float* up, float* output, int64_t n) {
    static CpuFeatures feat = detect_cpu_features();
#ifdef QUANT_HAS_AVX2
    if (feat.has_avx2) return geglu_avx2(gate, up, output, n);
#endif
    geglu_scalar(gate, up, output, n);
}

// ============================================================================
// RoPE — Rotary Position Embeddings
// ============================================================================

void rope_precompute_freqs(float* cos_out, float* sin_out,
                           int64_t dim, int64_t max_seq_len, float theta) {
    for (int64_t pos = 0; pos < max_seq_len; ++pos) {
        for (int64_t i = 0; i < dim / 2; ++i) {
            float freq = 1.0f / std::pow(theta, (2.0f * i) / dim);
            float angle = pos * freq;
            cos_out[pos * (dim / 2) + i] = std::cos(angle);
            sin_out[pos * (dim / 2) + i] = std::sin(angle);
        }
    }
}

void rope_scalar(float* q, float* k,
                 const float* cos_cache, const float* sin_cache,
                 int64_t head_dim, int64_t seq_len, int64_t pos_offset) {
    int64_t half_dim = head_dim / 2;
    for (int64_t pos = 0; pos < seq_len; ++pos) {
        int64_t cache_idx = (pos + pos_offset) * half_dim;
        for (int64_t i = 0; i < half_dim; ++i) {
            float c = cos_cache[cache_idx + i];
            float s = sin_cache[cache_idx + i];
            // Q rotation
            float q0 = q[pos * head_dim + i];
            float q1 = q[pos * head_dim + i + half_dim];
            q[pos * head_dim + i] = q0 * c - q1 * s;
            q[pos * head_dim + i + half_dim] = q0 * s + q1 * c;
            // K rotation
            if (k) {
                float k0 = k[pos * head_dim + i];
                float k1 = k[pos * head_dim + i + half_dim];
                k[pos * head_dim + i] = k0 * c - k1 * s;
                k[pos * head_dim + i + half_dim] = k0 * s + k1 * c;
            }
        }
    }
}

#ifdef QUANT_HAS_AVX2
void rope_avx2(float* q, float* k,
               const float* cos_cache, const float* sin_cache,
               int64_t head_dim, int64_t seq_len, int64_t pos_offset) {
    int64_t half_dim = head_dim / 2;
    for (int64_t pos = 0; pos < seq_len; ++pos) {
        int64_t cache_idx = (pos + pos_offset) * half_dim;
        int64_t i = 0;
        for (; i + 4 <= half_dim; i += 4) {
            __m256 c = _mm256_loadu_ps(cos_cache + cache_idx + i);
            __m256 s = _mm256_loadu_ps(sin_cache + cache_idx + i);

            // Q rotation (4 complex pairs = 8 floats)
            __m256 q0123 = _mm256_loadu_ps(q + pos * head_dim + i);
            __m256 q4567 = _mm256_loadu_ps(q + pos * head_dim + i + half_dim);
            // Rotate: q0' = q0*c - q4*s, q4' = q0*s + q4*c
            __m256 q_rot = _mm256_mul_ps(q0123, c);
            q_rot = _mm256_fmsub_ps(q4567, s, q_rot);
            __m256 q_rot2 = _mm256_mul_ps(q0123, s);
            q_rot2 = _mm256_fmadd_ps(q4567, c, q_rot2);
            _mm256_storeu_ps(q + pos * head_dim + i, q_rot);
            _mm256_storeu_ps(q + pos * head_dim + i + half_dim, q_rot2);

            // K rotation (same pattern)
            if (k) {
                __m256 k0123 = _mm256_loadu_ps(k + pos * head_dim + i);
                __m256 k4567 = _mm256_loadu_ps(k + pos * head_dim + i + half_dim);
                __m256 k_rot = _mm256_mul_ps(k0123, c);
                k_rot = _mm256_fmsub_ps(k4567, s, k_rot);
                __m256 k_rot2 = _mm256_mul_ps(k0123, s);
                k_rot2 = _mm256_fmadd_ps(k4567, c, k_rot2);
                _mm256_storeu_ps(k + pos * head_dim + i, k_rot);
                _mm256_storeu_ps(k + pos * head_dim + i + half_dim, k_rot2);
            }
        }
        for (; i < half_dim; ++i) {
            float c = cos_cache[cache_idx + i];
            float s = sin_cache[cache_idx + i];
            float q0 = q[pos * head_dim + i];
            float q1 = q[pos * head_dim + i + half_dim];
            q[pos * head_dim + i] = q0 * c - q1 * s;
            q[pos * head_dim + i + half_dim] = q0 * s + q1 * c;
            if (k) {
                float k0 = k[pos * head_dim + i];
                float k1 = k[pos * head_dim + i + half_dim];
                k[pos * head_dim + i] = k0 * c - k1 * s;
                k[pos * head_dim + i + half_dim] = k0 * s + k1 * c;
            }
        }
    }
}
#endif

void rope(float* q, float* k,
          const float* cos_cache, const float* sin_cache,
          int64_t head_dim, int64_t seq_len, int64_t pos_offset) {
    static CpuFeatures feat = detect_cpu_features();
#ifdef QUANT_HAS_AVX2
    if (feat.has_avx2) return rope_avx2(q, k, cos_cache, sin_cache, head_dim, seq_len, pos_offset);
#endif
    rope_scalar(q, k, cos_cache, sin_cache, head_dim, seq_len, pos_offset);
}

// ============================================================================
// Softmax — Numerically stable
// ============================================================================

void softmax_scalar(const float* x, float* y, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; ++r) {
        float max_val = *std::max_element(x + r * cols, x + (r + 1) * cols);
        float sum = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            y[r * cols + c] = std::exp(x[r * cols + c] - max_val);
            sum += y[r * cols + c];
        }
        for (int64_t c = 0; c < cols; ++c) {
            y[r * cols + c] /= sum;
        }
    }
}

#ifdef QUANT_HAS_AVX2
void softmax_avx2(const float* x, float* y, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; ++r) {
        const float* row = x + r * cols;
        float* out = y + r * cols;

        // Find max
        __m256 v_max = _mm256_set1_ps(-1e30f);
        int64_t c = 0;
        for (; c + 8 <= cols; c += 8) {
            __m256 v = _mm256_loadu_ps(row + c);
            v_max = _mm256_max_ps(v_max, v);
        }
        __m128 hi = _mm256_extractf128_ps(v_max, 1);
        __m128 lo = _mm256_castps256_ps128(v_max);
        __m128 mx = _mm_max_ps(hi, lo);
        mx = _mm_max_ps(mx, _mm_shuffle_ps(mx, mx, 0x4E));
        mx = _mm_max_ps(mx, _mm_shuffle_ps(mx, mx, 0xB1));
        float max_val = _mm_cvtss_f32(mx);
        for (; c < cols; ++c) max_val = std::max(max_val, row[c]);

        // Compute exp and sum
        __m256 v_sum = _mm256_setzero_ps();
        c = 0;
        for (; c + 8 <= cols; c += 8) {
            float tmp[8];
            _mm256_storeu_ps(tmp, _mm256_sub_ps(_mm256_loadu_ps(row + c), _mm256_set1_ps(max_val)));
            for (int ei = 0; ei < 8; ei++) tmp[ei] = std::exp(tmp[ei]);
            __m256 v = _mm256_loadu_ps(tmp);
            _mm256_storeu_ps(out + c, v);
            v_sum = _mm256_add_ps(v_sum, v);
        }
        __m128 v_sum_hi = _mm256_extractf128_ps(v_sum, 1);
        __m128 v_sum_lo = _mm256_castps256_ps128(v_sum);
        __m128 v_sum128 = _mm_add_ps(v_sum_hi, v_sum_lo);
        v_sum128 = _mm_hadd_ps(v_sum128, v_sum128);
        v_sum128 = _mm_hadd_ps(v_sum128, v_sum128);
        float sum = _mm_cvtss_f32(v_sum128);
        for (; c < cols; ++c) {
            out[c] = std::exp(row[c] - max_val);
            sum += out[c];
        }
        // Normalize
        __m256 v_inv_sum = _mm256_set1_ps(1.0f / sum);
        c = 0;
        for (; c + 8 <= cols; c += 8) {
            __m256 v = _mm256_loadu_ps(out + c);
            _mm256_storeu_ps(out + c, _mm256_mul_ps(v, v_inv_sum));
        }
        for (; c < cols; ++c) out[c] /= sum;
    }
}
#endif

void softmax(const float* x, float* y, int64_t rows, int64_t cols) {
    static CpuFeatures feat = detect_cpu_features();
#ifdef QUANT_HAS_AVX2
    if (feat.has_avx2) return softmax_avx2(x, y, rows, cols);
#endif
    softmax_scalar(x, y, rows, cols);
}

// ============================================================================
// GEMV — General Matrix-Vector Multiply
// ============================================================================

void gemv_scalar(const float* A, const float* x, float* y,
                 int64_t M, int64_t K, float alpha, float beta) {
    for (int64_t m = 0; m < M; ++m) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            sum += A[m * K + k] * x[k];
        }
        y[m] = alpha * sum + beta * y[m];
    }
}

#ifdef QUANT_HAS_AVX2
void gemv_avx2(const float* A, const float* x, float* y,
               int64_t M, int64_t K, float alpha, float beta) {
    __m256 v_alpha = _mm256_set1_ps(alpha);
    __m256 v_beta = _mm256_set1_ps(beta);
    for (int64_t m = 0; m < M; ++m) {
        __m256 sum = _mm256_setzero_ps();
        int64_t k = 0;
        for (; k + 8 <= K; k += 8) {
            __m256 a = _mm256_loadu_ps(A + m * K + k);
            __m256 b = _mm256_loadu_ps(x + k);
            sum = _mm256_fmadd_ps(a, b, sum);
        }
        float s = _mm_cvtss_f32(_mm_hadd_ps(
            _mm_hadd_ps(_mm256_castps256_ps128(sum),
                        _mm256_extractf128_ps(sum, 1)),
            _mm256_castps256_ps128(sum)));
        for (; k < K; ++k) s += A[m * K + k] * x[k];
        y[m] = alpha * s + beta * y[m];
    }
}
#endif

void gemv(const float* A, const float* x, float* y,
          int64_t M, int64_t K, float alpha, float beta) {
    static CpuFeatures feat = detect_cpu_features();
#ifdef QUANT_HAS_AVX2
    if (feat.has_avx2) return gemv_avx2(A, x, y, M, K, alpha, beta);
#endif
    gemv_scalar(A, x, y, M, K, alpha, beta);
}

// ============================================================================
// Tiled GEMM — Cache-efficient matrix multiply
// ============================================================================

void tiled_gemm_scalar(const float* A, const float* B, float* C,
                       int64_t M, int64_t N, int64_t K,
                       float alpha, float beta) {
    constexpr int64_t TILE = 64;
    // Pre-scale C
    if (beta != 1.0f) {
        for (int64_t i = 0; i < M * N; ++i) C[i] *= beta;
    }
    for (int64_t mi = 0; mi < M; mi += TILE) {
        for (int64_t ni = 0; ni < N; ni += TILE) {
            for (int64_t ki = 0; ki < K; ki += TILE) {
                int64_t m_end = std::min(mi + TILE, M);
                int64_t n_end = std::min(ni + TILE, N);
                int64_t k_end = std::min(ki + TILE, K);
                for (int64_t m = mi; m < m_end; ++m) {
                    for (int64_t k = ki; k < k_end; ++k) {
                        float a = alpha * A[m * K + k];
                        for (int64_t n = ni; n < n_end; ++n) {
                            C[m * N + n] += a * B[k * N + n];
                        }
                    }
                }
            }
        }
    }
}

void tiled_gemm(const float* A, const float* B, float* C,
                int64_t M, int64_t N, int64_t K,
                float alpha, float beta) {
    static CpuFeatures feat = detect_cpu_features();
#ifdef QUANT_HAS_AVX512
    if (feat.has_avx512) return tiled_gemm_avx512(A, B, C, M, N, K, alpha, beta);
#endif
#ifdef QUANT_HAS_AVX2
    if (feat.has_avx2) return tiled_gemm_avx2(A, B, C, M, N, K, alpha, beta);
#endif
    tiled_gemm_scalar(A, B, C, M, N, K, alpha, beta);
}

#ifdef QUANT_HAS_AVX2
void tiled_gemm_avx2(const float* A, const float* B, float* C,
                     int64_t M, int64_t N, int64_t K,
                     float alpha, float beta) {
    constexpr int64_t TILE = 64;
    if (beta != 1.0f) {
        for (int64_t i = 0; i < M * N; ++i) C[i] *= beta;
    }
    for (int64_t mi = 0; mi < M; mi += TILE) {
        for (int64_t ni = 0; ni < N; ni += TILE) {
            for (int64_t ki = 0; ki < K; ki += TILE) {
                int64_t m_end = std::min(mi + TILE, M);
                int64_t n_end = std::min(ni + TILE, N);
                int64_t k_end = std::min(ki + TILE, K);
                for (int64_t m = mi; m < m_end; ++m) {
                    for (int64_t k = ki; k < k_end; ++k) {
                        __m256 va = _mm256_set1_ps(alpha * A[m * K + k]);
                        int64_t n = ni;
                        for (; n + 8 <= n_end; n += 8) {
                            __m256 vb = _mm256_loadu_ps(B + k * N + n);
                            __m256 vc = _mm256_loadu_ps(C + m * N + n);
                            vc = _mm256_fmadd_ps(va, vb, vc);
                            _mm256_storeu_ps(C + m * N + n, vc);
                        }
                        for (; n < n_end; ++n) {
                            C[m * N + n] += alpha * A[m * K + k] * B[k * N + n];
                        }
                    }
                }
            }
        }
    }
}
#endif

#ifdef QUANT_HAS_AVX512
void tiled_gemm_avx512(const float* A, const float* B, float* C,
                       int64_t M, int64_t N, int64_t K,
                       float alpha, float beta) {
    constexpr int64_t TILE = 64;
    if (beta != 1.0f) {
        for (int64_t i = 0; i < M * N; ++i) C[i] *= beta;
    }
    for (int64_t mi = 0; mi < M; mi += TILE) {
        for (int64_t ni = 0; ni < N; ni += TILE) {
            for (int64_t ki = 0; ki < K; ki += TILE) {
                int64_t m_end = std::min(mi + TILE, M);
                int64_t n_end = std::min(ni + TILE, N);
                int64_t k_end = std::min(ki + TILE, K);
                for (int64_t m = mi; m < m_end; ++m) {
                    for (int64_t k = ki; k < k_end; ++k) {
                        __m512 va = _mm512_set1_ps(alpha * A[m * K + k]);
                        int64_t n = ni;
                        for (; n + 16 <= n_end; n += 16) {
                            __m512 vb = _mm512_loadu_ps(B + k * N + n);
                            __m512 vc = _mm512_loadu_ps(C + m * N + n);
                            vc = _mm512_fmadd_ps(va, vb, vc);
                            _mm512_storeu_ps(C + m * N + n, vc);
                        }
                        for (; n < n_end; ++n) {
                            C[m * N + n] += alpha * A[m * K + k] * B[k * N + n];
                        }
                    }
                }
            }
        }
    }
}
#endif

} // namespace simd
} // namespace quant
