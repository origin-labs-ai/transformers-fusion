#include "quant/math.h"
#include "quant/tensor.h"

#include <cmath>
#include <cstring>
#include <algorithm>

// This file provides AVX2-optimized implementations.
// math.cpp is excluded via #if !defined(QUANT_AVX2) to avoid duplicate symbols.
#if defined(QUANT_AVX2)

#include <immintrin.h>
#include "quant/detail/exp_avx2.h"

namespace quant {
namespace math {

static inline const float* rd(const Tensor& t) { return t.data<float>(); }
static inline float* wr(Tensor& t) { return t.data<float>(); }

// ---------------------------------------------------------------------------
// gemm – 6x16 tiled with _mm256_fmadd_ps
// ---------------------------------------------------------------------------
void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) {
    QUANT_CHECK(A.shape().rank == 2 && B.shape().rank == 2 && C.shape().rank == 2,
              "gemm: all inputs must be 2D");
    int64_t M = A.shape().dims[0];
    int64_t K = A.shape().dims[1];
    int64_t N = B.shape().dims[1];
    QUANT_CHECK(B.shape().dims[0] == K, "gemm A cols != B rows");
    QUANT_CHECK(C.shape().dims[0] == M && C.shape().dims[1] == N, "gemm C shape mismatch");

    const float* pa = rd(A);
    const float* pb = rd(B);
    float* pc = wr(C);

    if (beta == 0.0f) {
        std::memset(pc, 0, (size_t)(M * N) * sizeof(float));
    } else if (beta != 1.0f) {
        for (int64_t i = 0; i < M * N; i++) pc[i] *= beta;
    }

    for (int64_t i = 0; i < M; i += 6) {
        int64_t imax = (i + 6 < M) ? i + 6 : M;
        for (int64_t j = 0; j < N; j += 16) {
            int64_t jmax = (j + 16 < N) ? j + 16 : N;

            __m256 acc[6][2] = {{_mm256_setzero_ps(), _mm256_setzero_ps()},
                                {_mm256_setzero_ps(), _mm256_setzero_ps()},
                                {_mm256_setzero_ps(), _mm256_setzero_ps()},
                                {_mm256_setzero_ps(), _mm256_setzero_ps()},
                                {_mm256_setzero_ps(), _mm256_setzero_ps()},
                                {_mm256_setzero_ps(), _mm256_setzero_ps()}};

            for (int64_t k = 0; k < K; k++) {
                __m256 a_bc[6];
                for (int64_t ii = i; ii < imax; ii++) {
                    a_bc[ii - i] = _mm256_set1_ps(pa[ii * K + k]);
                }

                // P0 fix: edge tile must not read past N. b1 (j+8..j+16)
                // over-reads when j+16 > N — load only the valid half and
                // zero the rest so the FMA stays exact.
                __m256 b0 = _mm256_loadu_ps(&pb[k * N + j]);
                __m256 b1;
                if (j + 16 <= N) {
                    b1 = _mm256_loadu_ps(&pb[k * N + j + 8]);
                } else {
                    alignas(32) float btail[8] = {0,0,0,0,0,0,0,0};
                    int64_t valid = (j + 8 < N) ? (N - j - 8) : 0;
                    for (int64_t t = 0; t < valid; t++)
                        btail[t] = pb[k * N + j + 8 + t];
                    b1 = _mm256_load_ps(btail);
                }

                for (int64_t ii = i; ii < imax; ii++) {
                    int idx = (int)(ii - i);
                    acc[idx][0] = _mm256_fmadd_ps(a_bc[idx], b0, acc[idx][0]);
                    acc[idx][1] = _mm256_fmadd_ps(a_bc[idx], b1, acc[idx][1]);
                }
            }

            __m256 alphav = _mm256_set1_ps(alpha);
            for (int64_t ii = i; ii < imax; ii++) {
                int idx = (int)(ii - i);
                // P0 fix: store only valid columns — the second 8-wide
                // store overran C when N-j < 16. Scalar tail is exact.
                int64_t cols0 = (j + 8 <= N) ? 8 : (N - j);
                int64_t cols1 = (N > j + 8) ? (N - j - 8) : 0;
                {
                    alignas(32) float tmp[8];
                    _mm256_store_ps(tmp, _mm256_mul_ps(alphav, acc[idx][0]));
                    for (int64_t t = 0; t < cols0; t++)
                        pc[ii * N + j + t] = tmp[t];
                }
                if (cols1 > 0) {
                    alignas(32) float tmp[8];
                    _mm256_store_ps(tmp, _mm256_mul_ps(alphav, acc[idx][1]));
                    for (int64_t t = 0; t < cols1; t++)
                        pc[ii * N + j + 8 + t] = tmp[t];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// gemv – vectorized dot per row
// ---------------------------------------------------------------------------
void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) {
    QUANT_CHECK(A.shape().rank == 2 && x.shape().rank == 1 && y.shape().rank == 1,
              "gemv: A 2D, x and y 1D");
    int64_t M = A.shape().dims[0];
    int64_t N = A.shape().dims[1];
    QUANT_CHECK(x.numel() == N, "gemv x size != A cols");
    QUANT_CHECK(y.numel() == M, "gemv y size != A rows");

    const float* pa = rd(A);
    const float* px = rd(x);
    float* py = wr(y);

    for (int64_t i = 0; i < M; i++) {
        __m256 sumv = _mm256_setzero_ps();
        int64_t j = 0;
        for (; j + 8 <= N; j += 8) {
            __m256 av = _mm256_loadu_ps(pa + i * N + j);
            __m256 xv = _mm256_loadu_ps(px + j);
            sumv = _mm256_fmadd_ps(av, xv, sumv);
        }
        __m128 hi = _mm_add_ps(_mm256_castps256_ps128(sumv),
                                _mm256_extractf128_ps(sumv, 1));
        hi = _mm_hadd_ps(hi, hi);
        hi = _mm_hadd_ps(hi, hi);
        float s = _mm_cvtss_f32(hi);
        for (; j < N; j++) s += pa[i * N + j] * px[j];
        py[i] = alpha * s + beta * py[i];
    }
}

// ---------------------------------------------------------------------------
// dot – vectorized reduce
// ---------------------------------------------------------------------------
float dot(const Tensor& a, const Tensor& b) {
    QUANT_CHECK(a.numel() == b.numel(), "dot shape mismatch");
    const float* pa = rd(a);
    const float* pb = rd(b);
    int64_t n = a.numel();
    __m256 sumv = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(pa + i);
        __m256 vb = _mm256_loadu_ps(pb + i);
        sumv = _mm256_fmadd_ps(va, vb, sumv);
    }
    __m128 hi = _mm_add_ps(_mm256_castps256_ps128(sumv),
                            _mm256_extractf128_ps(sumv, 1));
    hi = _mm_hadd_ps(hi, hi);
    hi = _mm_hadd_ps(hi, hi);
    float s = _mm_cvtss_f32(hi);
    for (; i < n; i++) s += pa[i] * pb[i];
    return s;
}

// ---------------------------------------------------------------------------
// axpy – vectorized
// ---------------------------------------------------------------------------
void axpy(float alpha, const Tensor& x, Tensor& y) {
    QUANT_CHECK(x.numel() == y.numel(), "axpy shape mismatch");
    const float* px = rd(x);
    float* py = wr(y);
    int64_t n = x.numel();
    __m256 av = _mm256_set1_ps(alpha);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xv = _mm256_loadu_ps(px + i);
        __m256 yv = _mm256_loadu_ps(py + i);
        _mm256_storeu_ps(py + i, _mm256_fmadd_ps(av, xv, yv));
    }
    for (; i < n; i++) py[i] += alpha * px[i];
}

// ---------------------------------------------------------------------------
// norm – vectorized
// ---------------------------------------------------------------------------
float norm(const Tensor& x) {
    const float* p = rd(x);
    int64_t n = x.numel();
    __m256 sumv = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(p + i);
        sumv = _mm256_fmadd_ps(v, v, sumv);
    }
    __m128 hi = _mm_add_ps(_mm256_castps256_ps128(sumv),
                            _mm256_extractf128_ps(sumv, 1));
    hi = _mm_hadd_ps(hi, hi);
    hi = _mm_hadd_ps(hi, hi);
    double s = (double)_mm_cvtss_f32(hi);
    for (; i < n; i++) s += (double)p[i] * p[i];
    return std::sqrt((float)s);
}

// ---------------------------------------------------------------------------
// asum – vectorized
// ---------------------------------------------------------------------------
float asum(const Tensor& x) {
    const float* p = rd(x);
    int64_t n = x.numel();
    __m256 sumv = _mm256_setzero_ps();
    __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_and_ps(_mm256_loadu_ps(p + i), absmask);
        sumv = _mm256_add_ps(sumv, v);
    }
    __m128 hi = _mm_add_ps(_mm256_castps256_ps128(sumv),
                            _mm256_extractf128_ps(sumv, 1));
    hi = _mm_hadd_ps(hi, hi);
    hi = _mm_hadd_ps(hi, hi);
    double s = (double)_mm_cvtss_f32(hi);
    for (; i < n; i++) s += std::abs((double)p[i]);
    return (float)s;
}

// ---------------------------------------------------------------------------
// Fast exp approximation for AVX2 — see quant/detail/exp_avx2.h (DEDUP:
// was triplicated in math_avx2/_tensor/_tiled). Imported, not redefined.
// ---------------------------------------------------------------------------
using detail::exp_ps;

// ---------------------------------------------------------------------------
// relu
// ---------------------------------------------------------------------------
void relu(const Tensor& x, Tensor& y) {
    QUANT_CHECK(x.numel() == y.numel(), "relu shape mismatch");
    const float* px = rd(x);
    float* py = wr(y);
    int64_t n = x.numel();
    __m256 zeros = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(px + i);
        _mm256_storeu_ps(py + i, _mm256_max_ps(v, zeros));
    }
    for (; i < n; i++) py[i] = px[i] > 0.0f ? px[i] : 0.0f;
}

// ---------------------------------------------------------------------------
// sigmoid – 1/(1+exp(-x)) with exp_ps
// ---------------------------------------------------------------------------
void sigmoid(const Tensor& x, Tensor& y) {
    QUANT_CHECK(x.numel() == y.numel(), "sigmoid shape mismatch");
    const float* px = rd(x);
    float* py = wr(y);
    int64_t n = x.numel();
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 clamp_hi = _mm256_set1_ps(88.376f);
    __m256 clamp_lo = _mm256_set1_ps(-88.376f);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(px + i);
        __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), v);
        __m256 cl = _mm256_min_ps(_mm256_max_ps(neg, clamp_lo), clamp_hi);
        __m256 e = exp_ps(cl);
        _mm256_storeu_ps(py + i, _mm256_div_ps(one, _mm256_add_ps(one, e)));
    }
    for (; i < n; i++) py[i] = 1.0f / (1.0f + std::exp(-px[i]));
}

// ---------------------------------------------------------------------------
// gelu – 0.5 * x * (1 + erf(x/sqrt(2)))
// ---------------------------------------------------------------------------
void gelu(const Tensor& x, Tensor& y) {
    QUANT_CHECK(x.numel() == y.numel(), "gelu shape mismatch");
    const float* px = rd(x);
    float* py = wr(y);
    int64_t n = x.numel();
    const float s = 0.7071067811865475f;
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 sqrt2_inv = _mm256_set1_ps(s);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(px + i);
        __m256 x = _mm256_mul_ps(v, sqrt2_inv);
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);
        __m256 x5 = _mm256_mul_ps(x3, x2);
        __m256 x7 = _mm256_mul_ps(x5, x2);
        __m256 x9 = _mm256_mul_ps(x7, x2);
        __m256 erv = _mm256_sub_ps(x, _mm256_mul_ps(x3, _mm256_set1_ps(1.0f/3.0f)));
        erv = _mm256_add_ps(erv, _mm256_mul_ps(x5, _mm256_set1_ps(1.0f/10.0f)));
        erv = _mm256_sub_ps(erv, _mm256_mul_ps(x7, _mm256_set1_ps(1.0f/42.0f)));
        erv = _mm256_add_ps(erv, _mm256_mul_ps(x9, _mm256_set1_ps(1.0f/216.0f)));
        __m256 one_point = _mm256_add_ps(one, erv);
        __m256 r = _mm256_mul_ps(half, _mm256_mul_ps(v, one_point));
        _mm256_storeu_ps(py + i, r);
    }
    for (; i < n; i++) py[i] = 0.5f * px[i] * (1.0f + std::erf(px[i] * s));
}

// ---------------------------------------------------------------------------
// silu – x * sigmoid(x)
// ---------------------------------------------------------------------------
void silu(const Tensor& x, Tensor& y) {
    QUANT_CHECK(x.numel() == y.numel(), "silu shape mismatch");
    const float* px = rd(x);
    float* py = wr(y);
    int64_t n = x.numel();
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 clamp_hi = _mm256_set1_ps(88.376f);
    __m256 clamp_lo = _mm256_set1_ps(-88.376f);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(px + i);
        __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), v);
        __m256 cl = _mm256_min_ps(_mm256_max_ps(neg, clamp_lo), clamp_hi);
        __m256 e = exp_ps(cl);
        __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
        _mm256_storeu_ps(py + i, _mm256_mul_ps(v, sig));
    }
    for (; i < n; i++) py[i] = px[i] / (1.0f + std::exp(-px[i]));
}

// ---------------------------------------------------------------------------
// tanh – 2*sigmoid(2x) - 1
// ---------------------------------------------------------------------------
void tanh_(const Tensor& x, Tensor& y) {
    QUANT_CHECK(x.numel() == y.numel(), "tanh shape mismatch");
    const float* px = rd(x);
    float* py = wr(y);
    int64_t n = x.numel();
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 clamp_hi = _mm256_set1_ps(88.376f);
    __m256 clamp_lo = _mm256_set1_ps(-88.376f);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(px + i);
        __m256 v2 = _mm256_mul_ps(v, two);
        __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), v2);
        __m256 cl = _mm256_min_ps(_mm256_max_ps(neg, clamp_lo), clamp_hi);
        __m256 e = exp_ps(cl);
        __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
        _mm256_storeu_ps(py + i, _mm256_sub_ps(_mm256_mul_ps(two, sig), one));
    }
    for (; i < n; i++) py[i] = std::tanh(px[i]);
}

// ---------------------------------------------------------------------------
// layer_norm
// ---------------------------------------------------------------------------
void layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps, Tensor& y) {
    QUANT_CHECK(x.numel() == y.numel(), "layer_norm shape mismatch");
    int64_t last_dim = x.shape().dims[x.shape().rank - 1];
    int64_t outer = x.numel() / last_dim;
    QUANT_CHECK(gamma.numel() == last_dim, "layer_norm gamma dim mismatch");
    QUANT_CHECK(beta.numel() == last_dim, "layer_norm beta dim mismatch");

    const float* px = rd(x);
    const float* pg = rd(gamma);
    const float* pb = rd(beta);
    float* py = wr(y);

    __m256 epsv = _mm256_set1_ps(eps);

    for (int64_t i = 0; i < outer; i++) {
        const float* row = px + i * last_dim;

        // Mean
        __m256 sumv = _mm256_setzero_ps();
        int64_t j = 0;
        for (; j + 8 <= last_dim; j += 8)
            sumv = _mm256_add_ps(sumv, _mm256_loadu_ps(row + j));
        __m128 hi = _mm_add_ps(_mm256_castps256_ps128(sumv),
                                _mm256_extractf128_ps(sumv, 1));
        hi = _mm_hadd_ps(hi, hi);
        hi = _mm_hadd_ps(hi, hi);
        double mu = (double)_mm_cvtss_f32(hi);
        for (; j < last_dim; j++) mu += row[j];
        mu /= (double)last_dim;

        // Variance
        __m256 muv = _mm256_set1_ps((float)mu);
        sumv = _mm256_setzero_ps();
        j = 0;
        for (; j + 8 <= last_dim; j += 8) {
            __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), muv);
            sumv = _mm256_fmadd_ps(d, d, sumv);
        }
        hi = _mm_add_ps(_mm256_castps256_ps128(sumv),
                        _mm256_extractf128_ps(sumv, 1));
        hi = _mm_hadd_ps(hi, hi);
        hi = _mm_hadd_ps(hi, hi);
        double var = (double)_mm_cvtss_f32(hi);
        for (; j < last_dim; j++) { double d = row[j] - mu; var += d * d; }
        var /= (double)last_dim;

        double inv_std = 1.0 / std::sqrt(var + (double)eps);
        __m256 invv = _mm256_set1_ps((float)inv_std);

        float* ry = py + i * last_dim;
        j = 0;
        for (; j + 8 <= last_dim; j += 8) {
            __m256 v = _mm256_loadu_ps(row + j);
            __m256 g = _mm256_loadu_ps(pg + j);
            __m256 b = _mm256_loadu_ps(pb + j);
            __m256 nv = _mm256_mul_ps(_mm256_sub_ps(v, muv), invv);
            _mm256_storeu_ps(ry + j, _mm256_fmadd_ps(nv, g, b));
        }
        for (; j < last_dim; j++) ry[j] = (float)(((double)row[j] - mu) * inv_std) * pg[j] + pb[j];
    }
}

// ---------------------------------------------------------------------------
// rms_norm
// ---------------------------------------------------------------------------
void rms_norm(const Tensor& x, const Tensor& gamma, float eps, Tensor& y) {
    QUANT_CHECK(x.numel() == y.numel(), "rms_norm shape mismatch");
    int64_t last_dim = x.shape().dims[x.shape().rank - 1];
    int64_t outer = x.numel() / last_dim;
    QUANT_CHECK(gamma.numel() == last_dim, "rms_norm gamma dim mismatch");

    const float* px = rd(x);
    const float* pg = rd(gamma);
    float* py = wr(y);

    for (int64_t i = 0; i < outer; i++) {
        const float* row = px + i * last_dim;

        __m256 sumv = _mm256_setzero_ps();
        int64_t j = 0;
        for (; j + 8 <= last_dim; j += 8) {
            __m256 v = _mm256_loadu_ps(row + j);
            sumv = _mm256_fmadd_ps(v, v, sumv);
        }
        __m128 hi = _mm_add_ps(_mm256_castps256_ps128(sumv),
                                _mm256_extractf128_ps(sumv, 1));
        hi = _mm_hadd_ps(hi, hi);
        hi = _mm_hadd_ps(hi, hi);
        float ss = _mm_cvtss_f32(hi);
        for (; j < last_dim; j++) ss += row[j] * row[j];
        ss /= (float)last_dim;

        double inv = 1.0 / std::sqrt((double)ss + (double)eps);
        __m256 invv = _mm256_set1_ps((float)inv);

        float* ry = py + i * last_dim;
        j = 0;
        for (; j + 8 <= last_dim; j += 8) {
            __m256 v = _mm256_loadu_ps(row + j);
            __m256 g = _mm256_loadu_ps(pg + j);
            _mm256_storeu_ps(ry + j, _mm256_mul_ps(_mm256_mul_ps(v, invv), g));
        }
        for (; j < last_dim; j++) ry[j] = (float)((double)row[j] * inv) * pg[j];
    }
}

// ---------------------------------------------------------------------------
// softmax (stable: subtract max before exp)
// ---------------------------------------------------------------------------
void softmax(const Tensor& x, Tensor& y, int axis) {
    QUANT_CHECK(x.numel() == y.numel(), "softmax shape mismatch");
    Shape s = x.shape();
    int rank = s.rank;
    if (axis < 0) axis += rank;
    QUANT_CHECK(axis >= 0 && axis < rank, "softmax axis out of range");

    int64_t outer = 1;
    for (int i = 0; i < axis; i++) outer *= s.dims[i];
    int64_t dim = s.dims[axis];
    int64_t inner = 1;
    for (int i = axis + 1; i < rank; i++) inner *= s.dims[i];

    const float* px = rd(x);
    float* py = wr(y);
    int64_t block = dim * inner;

    for (int64_t o = 0; o < outer; o++) {
        for (int64_t i = 0; i < inner; i++) {
            const float* src = px + o * block + i;
            float* dst = py + o * block + i;

            // Find max (vectorized)
            __m256 mxv = _mm256_set1_ps(-INFINITY);
            int64_t d = 0;
            for (; d + 8 <= dim; d += 8) {
                __m256 v = _mm256_loadu_ps(src + d * inner);
                mxv = _mm256_max_ps(mxv, v);
            }
            __m128 hi = _mm_max_ps(_mm256_castps256_ps128(mxv),
                                    _mm256_extractf128_ps(mxv, 1));
            hi = _mm_max_ps(hi, _mm_shuffle_ps(hi, hi, _MM_SHUFFLE(2,3,0,1)));
            hi = _mm_max_ps(hi, _mm_shuffle_ps(hi, hi, _MM_SHUFFLE(1,0,3,2)));
            float mx = _mm_cvtss_f32(hi);
            for (; d < dim; d++) mx = std::max(mx, src[d * inner]);

            // Compute exp and sum
            __m256 sumv = _mm256_setzero_ps();
            __m256 mx_bc = _mm256_set1_ps(mx);
            d = 0;
            for (; d + 8 <= dim; d += 8) {
                __m256 v = _mm256_loadu_ps(src + d * inner);
                __m256 e = exp_ps(_mm256_sub_ps(v, mx_bc));
                _mm256_storeu_ps(dst + d * inner, e);
                sumv = _mm256_add_ps(sumv, e);
            }
            __m128 sum_hi = _mm_add_ps(_mm256_castps256_ps128(sumv),
                                        _mm256_extractf128_ps(sumv, 1));
            sum_hi = _mm_hadd_ps(sum_hi, sum_hi);
            sum_hi = _mm_hadd_ps(sum_hi, sum_hi);
            double total = (double)_mm_cvtss_f32(sum_hi);
            for (; d < dim; d++) {
                float e = std::exp(src[d * inner] - mx);
                dst[d * inner] = e;
                total += e;
            }

            double inv = 1.0 / total;
            __m256 invv = _mm256_set1_ps((float)inv);
            d = 0;
            for (; d + 8 <= dim; d += 8) {
                __m256 e = _mm256_loadu_ps(dst + d * inner);
                _mm256_storeu_ps(dst + d * inner, _mm256_mul_ps(e, invv));
            }
            for (; d < dim; d++) dst[d * inner] = (float)((double)dst[d * inner] * inv);
        }
    }
}

// ---------------------------------------------------------------------------
// Pointwise
// ---------------------------------------------------------------------------
void add(const Tensor& a, const Tensor& b, Tensor& c) {
    QUANT_CHECK(a.numel() == b.numel() && b.numel() == c.numel(), "add shape mismatch");
    const float* pa = rd(a); const float* pb = rd(b); float* pc = wr(c);
    int64_t n = a.numel();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(pc + i, _mm256_add_ps(_mm256_loadu_ps(pa + i), _mm256_loadu_ps(pb + i)));
    }
    for (; i < n; i++) pc[i] = pa[i] + pb[i];
}

void sub(const Tensor& a, const Tensor& b, Tensor& c) {
    QUANT_CHECK(a.numel() == b.numel() && b.numel() == c.numel(), "sub shape mismatch");
    const float* pa = rd(a); const float* pb = rd(b); float* pc = wr(c);
    int64_t n = a.numel();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(pc + i, _mm256_sub_ps(_mm256_loadu_ps(pa + i), _mm256_loadu_ps(pb + i)));
    }
    for (; i < n; i++) pc[i] = pa[i] - pb[i];
}

void mul(const Tensor& a, const Tensor& b, Tensor& c) {
    QUANT_CHECK(a.numel() == b.numel() && b.numel() == c.numel(), "mul shape mismatch");
    const float* pa = rd(a); const float* pb = rd(b); float* pc = wr(c);
    int64_t n = a.numel();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(pc + i, _mm256_mul_ps(_mm256_loadu_ps(pa + i), _mm256_loadu_ps(pb + i)));
    }
    for (; i < n; i++) pc[i] = pa[i] * pb[i];
}

void scale(float s, const Tensor& x, Tensor& y) {
    QUANT_CHECK(x.numel() == y.numel(), "scale shape mismatch");
    const float* px = rd(x); float* py = wr(y);
    int64_t n = x.numel();
    __m256 sv = _mm256_set1_ps(s);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(py + i, _mm256_mul_ps(sv, _mm256_loadu_ps(px + i)));
    for (; i < n; i++) py[i] = s * px[i];
}

// ---------------------------------------------------------------------------
// Reduce
// ---------------------------------------------------------------------------
float mean(const Tensor& x) { return sum(x) / (float)x.numel(); }

float sum(const Tensor& x) {
    const float* p = rd(x);
    int64_t n = x.numel();
    __m256 sumv = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8)
        sumv = _mm256_add_ps(sumv, _mm256_loadu_ps(p + i));
    __m128 hi = _mm_add_ps(_mm256_castps256_ps128(sumv),
                            _mm256_extractf128_ps(sumv, 1));
    hi = _mm_hadd_ps(hi, hi);
    hi = _mm_hadd_ps(hi, hi);
    double s = (double)_mm_cvtss_f32(hi);
    for (; i < n; i++) s += p[i];
    return (float)s;
}

float max(const Tensor& x) {
    const float* p = rd(x);
    QUANT_CHECK(x.numel() > 0, "max on empty tensor");
    int64_t n = x.numel();
    __m256 mxv = _mm256_loadu_ps(p);
    int64_t i = 8;
    for (; i + 8 <= n; i += 8)
        mxv = _mm256_max_ps(mxv, _mm256_loadu_ps(p + i));
    __m128 hi = _mm_max_ps(_mm256_castps256_ps128(mxv),
                            _mm256_extractf128_ps(mxv, 1));
    hi = _mm_max_ps(hi, _mm_shuffle_ps(hi, hi, _MM_SHUFFLE(2,3,0,1)));
    hi = _mm_max_ps(hi, _mm_shuffle_ps(hi, hi, _MM_SHUFFLE(1,0,3,2)));
    float m = _mm_cvtss_f32(hi);
    for (; i < n; i++) m = std::max(m, p[i]);
    return m;
}

void swiglu(const Tensor& gate, const Tensor& up, Tensor& y) {
    QUANT_CHECK(gate.numel() == up.numel() && up.numel() == y.numel(), "swiglu shape mismatch");
    const float* pg = rd(gate); const float* pu = rd(up); float* py = wr(y);
    int64_t n = gate.numel();
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 clamp_hi = _mm256_set1_ps(88.376f);
    __m256 clamp_lo = _mm256_set1_ps(-88.376f);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 g = _mm256_loadu_ps(pg + i);
        __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), g);
        __m256 cl = _mm256_min_ps(_mm256_max_ps(neg, clamp_lo), clamp_hi);
        __m256 e = exp_ps(cl);
        __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
        __m256 silu = _mm256_mul_ps(g, sig);
        _mm256_storeu_ps(py + i, _mm256_mul_ps(silu, _mm256_loadu_ps(pu + i)));
    }
    for (; i < n; i++) { float g = pg[i]; py[i] = g / (1.0f + std::exp(-g)) * pu[i]; }
}

void geglu(const Tensor& gate, const Tensor& up, Tensor& y) {
    QUANT_CHECK(gate.numel() == up.numel() && up.numel() == y.numel(), "geglu shape mismatch");
    const float* pg = rd(gate); const float* pu = rd(up); float* py = wr(y);
    int64_t n = gate.numel();
    const float s = 0.7071067811865475f;
    __m256 sqrt2_inv = _mm256_set1_ps(s);
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 one = _mm256_set1_ps(1.0f);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 g = _mm256_loadu_ps(pg + i);
        __m256 x = _mm256_mul_ps(g, sqrt2_inv);
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);
        __m256 x5 = _mm256_mul_ps(x3, x2);
        __m256 x7 = _mm256_mul_ps(x5, x2);
        __m256 x9 = _mm256_mul_ps(x7, x2);
        __m256 erv = _mm256_sub_ps(x, _mm256_mul_ps(x3, _mm256_set1_ps(1.0f/3.0f)));
        erv = _mm256_add_ps(erv, _mm256_mul_ps(x5, _mm256_set1_ps(1.0f/10.0f)));
        erv = _mm256_sub_ps(erv, _mm256_mul_ps(x7, _mm256_set1_ps(1.0f/42.0f)));
        erv = _mm256_add_ps(erv, _mm256_mul_ps(x9, _mm256_set1_ps(1.0f/216.0f)));
        __m256 gel = _mm256_mul_ps(half, _mm256_mul_ps(g, _mm256_add_ps(one, erv)));
        _mm256_storeu_ps(py + i, _mm256_mul_ps(gel, _mm256_loadu_ps(pu + i)));
    }
    for (; i < n; i++) { float g = pg[i]; float gel = 0.5f * g * (1.0f + std::erf(g * s)); py[i] = gel * pu[i]; }
}

// FP8 dispatch (scalar fallback — AVX2 path loops the scalar per-element which is fine for bulk conversion)
// NOTE: subnormal/saturation semantics mirror src/math/math.cpp exactly (see
// comments there): subnormal E4M3 region uses lround(|f|*512), E5M2 uses
// lround(|f|*65536); E4M3 top bin is finite except mant==7 (NaN).
void vec_fp32_to_fp8_e4m3(uint8_t* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) {
        float f = src[i];
        uint32_t bits; std::memcpy(&bits, &f, sizeof(float));
        int sign = (bits >> 31) & 1;
        int exp = ((int)((bits >> 23) & 0xFF)) - 127;
        int mant = bits & 0x7FFFFF;
        if (f == 0.0f) dst[i] = 0;
        else if (std::isnan(f)) dst[i] = 0x7F;
        else if (std::isinf(f)) dst[i] = f > 0 ? 0x7E : 0xFE;
        else if (exp > 8) dst[i] = (uint8_t)((sign << 7) | 0x7E);
        else if (exp < -6) {
            int m = (int)std::lround(std::fabs(f) * 512.0f);
            if (m <= 0) dst[i] = (uint8_t)(sign << 7);
            else if (m >= 8) dst[i] = (uint8_t)((sign << 7) | (1 << 3));
            else dst[i] = (uint8_t)((sign << 7) | m);
        } else {
            int te = exp + 7; int tm = (mant >> 20) & 0x07;
            if ((mant >> 19) & 1) { tm++; if (tm >= 8) { tm = 0; te++; if (te >= 15) { dst[i] = (uint8_t)((sign << 7) | 0x7E); continue; } } }
            if (te <= 0) dst[i] = (uint8_t)(sign << 7);
            else if (te >= 15) dst[i] = (uint8_t)((sign << 7) | 0x7E);
            else dst[i] = (uint8_t)((sign << 7) | (te << 3) | tm);
        }
    }
}
void vec_fp8_e4m3_to_fp32(float* dst, const uint8_t* src, int n) { for (int i = 0; i < n; i++) { uint8_t v = src[i]; if (v == 0) dst[i] = 0.0f; else { int sign = (v & 0x80) ? -1 : 1; int exp = (v & 0x78) >> 3; int mant = v & 0x07; if (exp == 0) dst[i] = sign * std::pow(2.0f, -6) * (mant / 8.0f); else if (exp == 15) dst[i] = (mant == 0x07) ? std::nanf("") : sign * std::pow(2.0f, 8) * (1.0f + mant / 8.0f); else dst[i] = sign * std::pow(2.0f, exp - 7) * (1.0f + mant / 8.0f); } } }
void vec_fp32_to_fp8_e5m2(uint8_t* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) {
        float f = src[i];
        uint32_t bits; std::memcpy(&bits, &f, sizeof(float));
        int sign = (bits >> 31) & 1;
        int exp = ((int)((bits >> 23) & 0xFF)) - 127;
        int mant = bits & 0x7FFFFF;
        if (f == 0.0f) dst[i] = 0;
        else if (std::isnan(f)) dst[i] = 0x7F;
        else if (std::isinf(f)) dst[i] = f > 0 ? 0x7C : 0xFC;
        else if (exp > 15) dst[i] = (uint8_t)((sign << 7) | 0x7C);
        else if (exp < -14) {
            int m = (int)std::lround(std::fabs(f) * 65536.0f);
            if (m <= 0) dst[i] = (uint8_t)(sign << 7);
            else if (m >= 4) dst[i] = (uint8_t)((sign << 7) | (1 << 2));
            else dst[i] = (uint8_t)((sign << 7) | m);
        } else {
            int te = exp + 15; int tm = (mant >> 21) & 0x03;
            if ((mant >> 20) & 1) { tm++; if (tm >= 4) { tm = 0; te++; if (te >= 31) { dst[i] = (uint8_t)((sign << 7) | 0x7C); continue; } } }
            if (te <= 0) dst[i] = (uint8_t)(sign << 7);
            else if (te >= 31) dst[i] = (uint8_t)((sign << 7) | 0x7C);
            else dst[i] = (uint8_t)((sign << 7) | (te << 2) | tm);
        }
    }
}
void vec_fp8_e5m2_to_fp32(float* dst, const uint8_t* src, int n) { for (int i = 0; i < n; i++) { uint8_t v = src[i]; if ((v & 0x7F) == 0) dst[i] = 0.0f; else { int sign = (v & 0x80) ? -1 : 1; int exp = (v & 0x7C) >> 2; int mant = v & 0x03; if (exp == 0) dst[i] = sign * std::pow(2.0f, -14) * (mant / 4.0f); else if (exp == 31) dst[i] = (mant == 0) ? (sign * INFINITY) : std::nanf(""); else dst[i] = sign * std::pow(2.0f, exp - 15) * (1.0f + mant / 4.0f); } } }
float f32_to_fp8_e4m3_scalar(float val) { uint8_t r; vec_fp32_to_fp8_e4m3(&r, &val, 1); return (float)r; }
float fp8_e4m3_to_f32_scalar(uint8_t val) { float r; vec_fp8_e4m3_to_fp32(&r, &val, 1); return r; }
float f32_to_fp8_e5m2_scalar(float val) { uint8_t r; vec_fp32_to_fp8_e5m2(&r, &val, 1); return (float)r; }
float fp8_e5m2_to_f32_scalar(uint8_t val) { float r; vec_fp8_e5m2_to_fp32(&r, &val, 1); return r; }
void fp8_gemm(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K, bool use_e4m3) { for (int64_t i = 0; i < M * N; i++) C[i] = 0.0f; std::vector<uint8_t> Aq(M * K), Bq(K * N); if (use_e4m3) { vec_fp32_to_fp8_e4m3(Aq.data(), A, (int)(M * K)); vec_fp32_to_fp8_e4m3(Bq.data(), B, (int)(K * N)); } else { vec_fp32_to_fp8_e5m2(Aq.data(), A, (int)(M * K)); vec_fp32_to_fp8_e5m2(Bq.data(), B, (int)(K * N)); } std::vector<float> Ad(M * K), Bd(K * N); if (use_e4m3) { vec_fp8_e4m3_to_fp32(Ad.data(), Aq.data(), (int)(M * K)); vec_fp8_e4m3_to_fp32(Bd.data(), Bq.data(), (int)(K * N)); } else { vec_fp8_e5m2_to_fp32(Ad.data(), Aq.data(), (int)(M * K)); vec_fp8_e5m2_to_fp32(Bd.data(), Bq.data(), (int)(K * N)); } for (int64_t m = 0; m < M; m++) for (int64_t k = 0; k < K; k++) { float aq = Ad[m * K + k]; for (int64_t n2 = 0; n2 < N; n2++) C[m * N + n2] += aq * Bd[k * N + n2]; } }
void vec_geglu(float* dst, const float* gate, const float* up, int n) { const float s = 0.7071067811865475f; for (int i = 0; i < n; i++) { float g = gate[i]; float gel = 0.5f * g * (1.0f + std::erf(g * s)); dst[i] = gel * up[i]; } }
void vec_geglu_avx2(float* dst, const float* gate, const float* up, int n) { vec_geglu(dst, gate, up, n); }

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------
Tensor zeros_like(const Tensor& x) { Tensor t(x.shape(), x.dtype()); t.zero_(); return t; }
Tensor ones_like(const Tensor& x) { Tensor t(x.shape(), x.dtype()); t.fill(1.0f); return t; }

} // namespace math
} // namespace quant

#endif // QUANT_AVX2
