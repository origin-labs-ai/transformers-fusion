#include "quant/math.h"
#include "quant/math_avx512.h"
#include "quant/tensor.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cfloat>

#if defined(__AVX512F__)

#include <immintrin.h>

namespace quant {
namespace math {

static inline const float* rd(const Tensor& t) { return t.data<float>(); }
static inline float* wr(Tensor& t) { return t.data<float>(); }

// ---------------------------------------------------------------------------
// Fast exp approximation for AVX512 (polynomial, float-precision)
// exp(x) = 2^(x * log2(e)), with polynomial on fractional part
// ---------------------------------------------------------------------------
static inline __m512 exp_ps_avx512(__m512 x) {
    __m512 ln2 = _mm512_set1_ps(1.4426950408889634f);
    __m512 t = _mm512_mul_ps(x, ln2);

    __m512 t_floor = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEG_INF);
    __m512 frac = _mm512_sub_ps(t, t_floor);

    __m512 c1 = _mm512_set1_ps(0.6931471805599453f);
    __m512 c2 = _mm512_set1_ps(0.240226506959101f);
    __m512 c3 = _mm512_set1_ps(0.055504108664672f);
    __m512 c4 = _mm512_set1_ps(0.009618129107628f);
    __m512 c5 = _mm512_set1_ps(0.001333355814643f);

    __m512 p = _mm512_fmadd_ps(c5, frac, c4);
    p = _mm512_fmadd_ps(p, frac, c3);
    p = _mm512_fmadd_ps(p, frac, c2);
    p = _mm512_fmadd_ps(p, frac, c1);
    p = _mm512_fmadd_ps(p, frac, _mm512_set1_ps(1.0f));

    __m512i exp_i = _mm512_cvttps_epi32(t_floor);
    __m512i exp_part = _mm512_slli_epi32(
        _mm512_add_epi32(exp_i, _mm512_set1_epi32(127)), 23);
    return _mm512_mul_ps(p, _mm512_castsi512_ps(exp_part));
}

// ---------------------------------------------------------------------------
// Horizontal reduce helpers for AVX512
// ---------------------------------------------------------------------------

static inline float hmax_ps_avx512(__m512 v) {
    __m256 hi256 = _mm512_extractf32x8_ps(v, 1);
    __m256 lo256 = _mm512_castps512_ps256(v);
    __m256 mx256 = _mm256_max_ps(lo256, hi256);
    __m128 hi128 = _mm256_extractf128_ps(mx256, 1);
    __m128 lo128 = _mm256_castps256_ps128(mx256);
    __m128 mx128 = _mm_max_ps(lo128, hi128);
    mx128 = _mm_max_ps(mx128, _mm_shuffle_ps(mx128, mx128, _MM_SHUFFLE(2,3,0,1)));
    mx128 = _mm_max_ps(mx128, _mm_shuffle_ps(mx128, mx128, _MM_SHUFFLE(1,0,3,2)));
    return _mm_cvtss_f32(mx128);
}

static inline float hadd_ps_avx512(__m512 v) {
    __m256 hi256 = _mm512_extractf32x8_ps(v, 1);
    __m256 lo256 = _mm512_castps512_ps256(v);
    __m256 sum256 = _mm256_add_ps(lo256, hi256);
    __m128 hi128 = _mm256_extractf128_ps(sum256, 1);
    __m128 lo128 = _mm256_castps256_ps128(sum256);
    __m128 sum128 = _mm_add_ps(lo128, hi128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

// ---------------------------------------------------------------------------
// dot_avx512 – 512-bit dot product (16 floats at a time)
// ---------------------------------------------------------------------------
void dot_avx512(float* result, const float* a, const float* b, int64_t n) {
    __m512 sumv = _mm512_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        sumv = _mm512_fmadd_ps(va, vb, sumv);
    }
    double s = (double)hadd_ps_avx512(sumv);
    for (; i < n; i++) s += (double)a[i] * (double)b[i];
    *result = (float)s;
}

// ---------------------------------------------------------------------------
// axpy_avx512 – y = a*x + y
// ---------------------------------------------------------------------------
void axpy_avx512(float alpha, const float* x, float* y, int64_t n) {
    __m512 av = _mm512_set1_ps(alpha);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 xv = _mm512_loadu_ps(x + i);
        __m512 yv = _mm512_loadu_ps(y + i);
        if (__builtin_expect(n > 1024, 1))
            _mm512_stream_ps(y + i, _mm512_fmadd_ps(av, xv, yv));
        else
            _mm512_storeu_ps(y + i, _mm512_fmadd_ps(av, xv, yv));
    }
    for (; i < n; i++) y[i] += alpha * x[i];
}

// ---------------------------------------------------------------------------
// vec_add_avx512
// ---------------------------------------------------------------------------
void vec_add_avx512(float* dst, const float* a, const float* b, int64_t n) {
    const bool nt = (n > 1024);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 r = _mm512_add_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        if (nt) _mm512_stream_ps(dst + i, r);
        else    _mm512_storeu_ps(dst + i, r);
    }
    for (; i < n; i++) dst[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------
// vec_sub_avx512
// ---------------------------------------------------------------------------
void vec_sub_avx512(float* dst, const float* a, const float* b, int64_t n) {
    const bool nt = (n > 1024);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 r = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        if (nt) _mm512_stream_ps(dst + i, r);
        else    _mm512_storeu_ps(dst + i, r);
    }
    for (; i < n; i++) dst[i] = a[i] - b[i];
}

// ---------------------------------------------------------------------------
// vec_mul_avx512
// ---------------------------------------------------------------------------
void vec_mul_avx512(float* dst, const float* a, const float* b, int64_t n) {
    const bool nt = (n > 1024);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 r = _mm512_mul_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        if (nt) _mm512_stream_ps(dst + i, r);
        else    _mm512_storeu_ps(dst + i, r);
    }
    for (; i < n; i++) dst[i] = a[i] * b[i];
}

// ---------------------------------------------------------------------------
// vec_div_avx512
// ---------------------------------------------------------------------------
void vec_div_avx512(float* dst, const float* a, const float* b, int64_t n) {
    const bool nt = (n > 1024);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 r = _mm512_div_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        if (nt) _mm512_stream_ps(dst + i, r);
        else    _mm512_storeu_ps(dst + i, r);
    }
    for (; i < n; i++) dst[i] = a[i] / b[i];
}

// ---------------------------------------------------------------------------
// vec_scale_avx512
// ---------------------------------------------------------------------------
void vec_scale_avx512(float* dst, const float* src, float factor, int64_t n) {
    __m512 fv = _mm512_set1_ps(factor);
    const bool nt = (n > 1024);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 r = _mm512_mul_ps(fv, _mm512_loadu_ps(src + i));
        if (nt) _mm512_stream_ps(dst + i, r);
        else    _mm512_storeu_ps(dst + i, r);
    }
    for (; i < n; i++) dst[i] = src[i] * factor;
}

// ---------------------------------------------------------------------------
// relu_avx512
// ---------------------------------------------------------------------------
void relu_avx512(float* dst, const float* src, int64_t n) {
    __m512 zeros = _mm512_setzero_ps();
    const bool nt = (n > 1024);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        __m512 r = _mm512_max_ps(v, zeros);
        if (nt) _mm512_stream_ps(dst + i, r);
        else    _mm512_storeu_ps(dst + i, r);
    }
    for (; i < n; i++) dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
}

// ---------------------------------------------------------------------------
// gelu_avx512 – 0.5 * x * (1 + erf(x/sqrt(2)))
// ---------------------------------------------------------------------------
void gelu_avx512(float* dst, const float* src, int64_t n) {
    __m512 half = _mm512_set1_ps(0.5f);
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 sqrt2_inv = _mm512_set1_ps(0.7071067811865475f);
    const bool nt = (n > 1024);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        __m512 xv = _mm512_mul_ps(v, sqrt2_inv);
        __m512 x2 = _mm512_mul_ps(xv, xv);
        __m512 x3 = _mm512_mul_ps(x2, xv);
        __m512 x5 = _mm512_mul_ps(x3, x2);
        __m512 x7 = _mm512_mul_ps(x5, x2);
        __m512 x9 = _mm512_mul_ps(x7, x2);
        __m512 erv = _mm512_fmadd_ps(x3, _mm512_set1_ps(-1.0f/3.0f), xv);
        erv = _mm512_fmadd_ps(x5, _mm512_set1_ps(1.0f/10.0f), erv);
        erv = _mm512_fmadd_ps(x7, _mm512_set1_ps(-1.0f/42.0f), erv);
        erv = _mm512_fmadd_ps(x9, _mm512_set1_ps(1.0f/216.0f), erv);
        __m512 r = _mm512_mul_ps(half, _mm512_mul_ps(v, _mm512_add_ps(one, erv)));
        if (nt) _mm512_stream_ps(dst + i, r);
        else    _mm512_storeu_ps(dst + i, r);
    }
    for (; i < n; i++) dst[i] = 0.5f * src[i] * (1.0f + std::erf(src[i] * 0.7071067811865475f));
}

// ---------------------------------------------------------------------------
// silu_avx512 – x * sigmoid(x)
// ---------------------------------------------------------------------------
void silu_avx512(float* dst, const float* src, int64_t n) {
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 clamp_hi = _mm512_set1_ps(88.376f);
    __m512 clamp_lo = _mm512_set1_ps(-88.376f);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        __m512 neg = _mm512_sub_ps(_mm512_setzero_ps(), v);
        __m512 cl = _mm512_min_ps(_mm512_max_ps(neg, clamp_lo), clamp_hi);
        __m512 e = exp_ps_avx512(cl);
        __m512 sig = _mm512_div_ps(one, _mm512_add_ps(one, e));
        _mm512_storeu_ps(dst + i, _mm512_mul_ps(v, sig));
    }
    for (; i < n; i++) dst[i] = src[i] / (1.0f + std::exp(-src[i]));
}

// ---------------------------------------------------------------------------
// sigmoid_avx512 – 1/(1+exp(-x))
// ---------------------------------------------------------------------------
void sigmoid_avx512(float* dst, const float* src, int64_t n) {
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 clamp_hi = _mm512_set1_ps(88.376f);
    __m512 clamp_lo = _mm512_set1_ps(-88.376f);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        __m512 neg = _mm512_sub_ps(_mm512_setzero_ps(), v);
        __m512 cl = _mm512_min_ps(_mm512_max_ps(neg, clamp_lo), clamp_hi);
        __m512 e = exp_ps_avx512(cl);
        _mm512_storeu_ps(dst + i, _mm512_div_ps(one, _mm512_add_ps(one, e)));
    }
    for (; i < n; i++) dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
}

// ---------------------------------------------------------------------------
// tanh_avx512 – 2*sigmoid(2x) - 1
// ---------------------------------------------------------------------------
void tanh_avx512(float* dst, const float* src, int64_t n) {
    __m512 two = _mm512_set1_ps(2.0f);
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 clamp_hi = _mm512_set1_ps(44.188f);
    __m512 clamp_lo = _mm512_set1_ps(-44.188f);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        __m512 v2 = _mm512_mul_ps(v, two);
        __m512 neg = _mm512_sub_ps(_mm512_setzero_ps(), v2);
        __m512 cl = _mm512_min_ps(_mm512_max_ps(neg, clamp_lo), clamp_hi);
        __m512 e = exp_ps_avx512(cl);
        __m512 sig = _mm512_div_ps(one, _mm512_add_ps(one, e));
        _mm512_storeu_ps(dst + i, _mm512_sub_ps(_mm512_mul_ps(two, sig), one));
    }
    for (; i < n; i++) dst[i] = std::tanh(src[i]);
}

// ---------------------------------------------------------------------------
// softmax_avx512 – stable softmax (subtract max before exp)
// ---------------------------------------------------------------------------
void softmax_avx512(float* dst, const float* src, int64_t n) {
    if (n <= 0) return;

    // Pass 1: find max
    __m512 mxv = _mm512_set1_ps(-INFINITY);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        mxv = _mm512_max_ps(mxv, v);
    }
    float max_val = hmax_ps_avx512(mxv);
    for (; i < n; i++) max_val = std::max(max_val, src[i]);

    // Pass 2: compute exp(x - max) and sum
    __m512 mx_bc = _mm512_set1_ps(max_val);
    __m512 sumv = _mm512_setzero_ps();
    i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        __m512 e = exp_ps_avx512(_mm512_sub_ps(v, mx_bc));
        _mm512_storeu_ps(dst + i, e);
        sumv = _mm512_add_ps(sumv, e);
    }
    double total = (double)hadd_ps_avx512(sumv);
    for (; i < n; i++) {
        float e = std::exp(src[i] - max_val);
        dst[i] = e;
        total += e;
    }

    // Pass 3: normalize
    double inv = 1.0 / total;
    __m512 invv = _mm512_set1_ps((float)inv);
    i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 e = _mm512_loadu_ps(dst + i);
        _mm512_storeu_ps(dst + i, _mm512_mul_ps(e, invv));
    }
    for (; i < n; i++) dst[i] = (float)((double)dst[i] * inv);
}

// ---------------------------------------------------------------------------
// layer_norm_avx512
// ---------------------------------------------------------------------------
void layer_norm_avx512(float* dst, const float* src, const float* gamma,
                       const float* beta, int64_t n, float eps) {
    // Compute mean
    __m512 sumv = _mm512_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        sumv = _mm512_add_ps(sumv, _mm512_loadu_ps(src + i));
    }
    double mu = (double)hadd_ps_avx512(sumv);
    for (; i < n; i++) mu += src[i];
    mu /= (double)n;

    // Compute variance
    __m512 muv = _mm512_set1_ps((float)mu);
    sumv = _mm512_setzero_ps();
    i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 d = _mm512_sub_ps(_mm512_loadu_ps(src + i), muv);
        sumv = _mm512_fmadd_ps(d, d, sumv);
    }
    double var = (double)hadd_ps_avx512(sumv);
    for (; i < n; i++) { double d = src[i] - mu; var += d * d; }
    var /= (double)n;

    double inv_std = 1.0 / std::sqrt(var + (double)eps);
    __m512 invv = _mm512_set1_ps((float)inv_std);

    // Normalize and apply gamma/beta
    i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        __m512 g = gamma ? _mm512_loadu_ps(gamma + i) : _mm512_set1_ps(1.0f);
        __m512 b = beta ? _mm512_loadu_ps(beta + i) : _mm512_setzero_ps();
        __m512 nv = _mm512_mul_ps(_mm512_sub_ps(v, muv), invv);
        _mm512_storeu_ps(dst + i, _mm512_fmadd_ps(nv, g, b));
    }
    for (; i < n; i++) {
        dst[i] = (float)(((double)src[i] - mu) * inv_std);
        if (gamma) dst[i] *= gamma[i];
        if (beta) dst[i] += beta[i];
    }
}

// ---------------------------------------------------------------------------
// rms_norm_avx512
// ---------------------------------------------------------------------------
void rms_norm_avx512(float* dst, const float* src, const float* gamma,
                     int64_t n, float eps) {
    __m512 sumv = _mm512_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        sumv = _mm512_fmadd_ps(v, v, sumv);
    }
    float ss = hadd_ps_avx512(sumv);
    for (; i < n; i++) ss += src[i] * src[i];
    ss /= (float)n;

    double inv = 1.0 / std::sqrt((double)ss + (double)eps);
    __m512 invv = _mm512_set1_ps((float)inv);

    i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        __m512 g = gamma ? _mm512_loadu_ps(gamma + i) : _mm512_set1_ps(1.0f);
        _mm512_storeu_ps(dst + i, _mm512_mul_ps(_mm512_mul_ps(v, invv), g));
    }
    for (; i < n; i++) {
        double val = (double)src[i] * inv;
        if (gamma) val *= gamma[i];
        dst[i] = (float)val;
    }
}

// ---------------------------------------------------------------------------
// gemv_avx512 – matrix-vector multiply: y = alpha * A * x + beta * y
// ---------------------------------------------------------------------------
void gemv_avx512(float alpha, const Tensor& A, const Tensor& x,
                 float beta, Tensor& y) {
    QUANT_CHECK(A.shape().rank == 2 && x.shape().rank == 1 && y.shape().rank == 1,
              "gemv_avx512: A 2D, x and y 1D");
    int64_t M = A.shape().dims[0];
    int64_t N = A.shape().dims[1];
    QUANT_CHECK(x.numel() == N, "gemv_avx512 x size != A cols");
    QUANT_CHECK(y.numel() == M, "gemv_avx512 y size != A rows");

    const float* pa = rd(A);
    const float* px = rd(x);
    float* py = wr(y);

    for (int64_t i = 0; i < M; i++) {
        __m512 sumv = _mm512_setzero_ps();
        int64_t j = 0;
        for (; j + 16 <= N; j += 16) {
            __m512 av = _mm512_loadu_ps(pa + i * N + j);
            __m512 xv = _mm512_loadu_ps(px + j);
            sumv = _mm512_fmadd_ps(av, xv, sumv);
        }
        float s = hadd_ps_avx512(sumv);
        for (; j < N; j++) s += pa[i * N + j] * px[j];
        py[i] = alpha * s + beta * py[i];
    }
}

} // namespace math
} // namespace quant

#endif // __AVX512F__
