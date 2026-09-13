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

                // P0 fix (mirrors math_avx2.cpp): edge tile must not read
                // past N. b1 (j+8..j+16) over-reads when j+16 > N.
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
                // P0 fix: store only valid columns (N-j < 16 edge).
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
// Fast exp approximation for AVX2 — see quant/detail/exp_avx2.h (DEDUP).
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
        __m256 xv = _mm256_mul_ps(v, sqrt2_inv);
        __m256 x2 = _mm256_mul_ps(xv, xv);
        __m256 x3 = _mm256_mul_ps(x2, xv);
        __m256 x5 = _mm256_mul_ps(x3, x2);
        __m256 x7 = _mm256_mul_ps(x5, x2);
        __m256 x9 = _mm256_mul_ps(x7, x2);
        __m256 erv = _mm256_sub_ps(xv, _mm256_mul_ps(x3, _mm256_set1_ps(1.0f/3.0f)));
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

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------
Tensor zeros_like(const Tensor& x) { Tensor t(x.shape(), x.dtype()); t.zero_(); return t; }
Tensor ones_like(const Tensor& x) { Tensor t(x.shape(), x.dtype()); t.fill(1.0f); return t; }

} // namespace math
} // namespace quant

#endif // QUANT_AVX2
