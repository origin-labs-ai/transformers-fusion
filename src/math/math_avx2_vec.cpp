#include "quant/math.h"
#include "quant/tensor.h"

#include <cmath>
#include <cstring>
#include <algorithm>

#if defined(QUANT_AVX2)

#include <immintrin.h>

namespace quant {
namespace math {
// ===========================================================================
// SIMD vector math functions — raw pointer interface
// ===========================================================================

void vec_exp(float* dst, const float* src, int n) {
    int i = 0;
    __m256 clamp_hi = _mm256_set1_ps(88.376f);
    __m256 clamp_lo = _mm256_set1_ps(-88.376f);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 cl = _mm256_min_ps(_mm256_max_ps(v, clamp_lo), clamp_hi);
        _mm256_storeu_ps(dst + i, exp_ps(cl));
    }
    for (; i < n; i++) dst[i] = std::exp(std::max(-88.376f, std::min(88.376f, src[i])));
}

void vec_log(float* dst, const float* src, int n) {
    int i = 0;
    // log(x) ≈ log2(x) / log2(e) — use native log2 on packed float
    // Use polynomial log approximation: log(x) = log(mantissa) + log(2)*exponent
    // We'll use a simpler approach via reinterpretation + polynomial
    __m256 log2_e = _mm256_set1_ps(1.4426950408889634f);
    __m256 one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_max_ps(_mm256_loadu_ps(src + i), _mm256_set1_ps(1e-38f));
        // Extract exponent: reinterpret bits
        __m256i bits = _mm256_castps_si256(v);
        __m256i exp_bits = _mm256_srli_epi32(bits, 23);
        __m256i exp_int = _mm256_sub_epi32(exp_bits, _mm256_set1_epi32(127));
        __m256 e = _mm256_cvtepi32_ps(exp_int);
        // Mantissa: set exponent to 127 (zero bias)
        __m256i mant_bits = _mm256_and_si256(bits, _mm256_set1_epi32(0x007FFFFF));
        mant_bits = _mm256_or_si256(mant_bits, _mm256_set1_epi32(0x3F800000));
        __m256 m = _mm256_castsi256_ps(mant_bits);
        // log2(m) on [1,2) using polynomial: P(m) ≈ log2(m)
        __m256 t = _mm256_sub_ps(m, one);
        __m256 p = _mm256_set1_ps(0.3333333333f);
        p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-0.5f));
        p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(1.0f));
        // log2(x) = e + log2(m) ≈ e + t * P(t)
        __m256 log2x = _mm256_add_ps(e, _mm256_mul_ps(t, p));
        // Convert to natural log: ln(x) = log2(x) / log2(e)
        _mm256_storeu_ps(dst + i, _mm256_div_ps(log2x, log2_e));
    }
    for (; i < n; i++) dst[i] = std::log(std::max(1e-38f, src[i]));
}

void vec_sigmoid(float* dst, const float* src, int n) {
    int i = 0;
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 clamp_hi = _mm256_set1_ps(88.376f);
    __m256 clamp_lo = _mm256_set1_ps(-88.376f);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), v);
        __m256 cl = _mm256_min_ps(_mm256_max_ps(neg, clamp_lo), clamp_hi);
        __m256 e = exp_ps(cl);
        _mm256_storeu_ps(dst + i, _mm256_div_ps(one, _mm256_add_ps(one, e)));
    }
    for (; i < n; i++) dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
}

void vec_tanh(float* dst, const float* src, int n) {
    int i = 0;
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 clamp_hi = _mm256_set1_ps(44.188f);
    __m256 clamp_lo = _mm256_set1_ps(-44.188f);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 v2 = _mm256_mul_ps(v, two);
        __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), v2);
        __m256 cl = _mm256_min_ps(_mm256_max_ps(neg, clamp_lo), clamp_hi);
        __m256 e = exp_ps(cl);
        __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
        _mm256_storeu_ps(dst + i, _mm256_sub_ps(_mm256_mul_ps(two, sig), one));
    }
    for (; i < n; i++) dst[i] = std::tanh(src[i]);
}

void vec_pow(float* dst, const float* src, float exp_val, int n) {
    if (exp_val == 2.0f) {
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, _mm256_mul_ps(v, v));
        }
        for (; i < n; i++) dst[i] = src[i] * src[i];
    } else if (exp_val == 0.5f) {
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_max_ps(_mm256_loadu_ps(src + i), _mm256_set1_ps(0.0f));
            _mm256_storeu_ps(dst + i, _mm256_sqrt_ps(v));
        }
        for (; i < n; i++) dst[i] = std::sqrt(std::max(0.0f, src[i]));
    } else {
        // Use ln(x)*exp_val then exp
        vec_log(dst, src, n);
        __m256 ev = _mm256_set1_ps(exp_val);
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(dst + i);
            _mm256_storeu_ps(dst + i, _mm256_mul_ps(v, ev));
        }
        for (; i < n; i++) dst[i] *= exp_val;
        vec_exp(dst, dst, n);
    }
}

void vec_erf(float* dst, const float* src, int n) {
    // erf(x) ≈ sign(x) * sqrt(1 - exp(-x^2 * (4/pi + a*x^2)/(1 + a*x^2)))
    // where a ≈ 0.147 — Abramowitz & Stegun approximation
    int i = 0;
    __m256 a = _mm256_set1_ps(0.147f);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 four_over_pi = _mm256_set1_ps(1.2732395447351628f);
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(src + i);
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 numer = _mm256_fmadd_ps(a, x2, four_over_pi);
        __m256 denom = _mm256_fmadd_ps(a, x2, one);
        __m256 ratio = _mm256_div_ps(numer, denom);
        __m256 exp_arg = _mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(x2, ratio));
        // Clamp for stability
        exp_arg = _mm256_max_ps(exp_arg, _mm256_set1_ps(-88.376f));
        __m256 e = exp_ps(exp_arg);
        __m256 erf_abs = _mm256_sqrt_ps(_mm256_sub_ps(one, e));
        // Apply sign of x
        __m256 sign = _mm256_or_ps(_mm256_and_ps(x, _mm256_set1_ps(-0.0f)),
                                    _mm256_set1_ps(1.0f));
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(erf_abs, sign));
    }
    for (; i < n; i++) dst[i] = std::erf(src[i]);
}

void vec_softmax_stable(float* dst, const float* src, int n) {
    if (n <= 0) return;
    __m256 mxv = _mm256_set1_ps(-INFINITY);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        mxv = _mm256_max_ps(mxv, v);
    }
    __m128 hi = _mm_max_ps(_mm256_castps256_ps128(mxv),
                            _mm256_extractf128_ps(mxv, 1));
    hi = _mm_max_ps(hi, _mm_shuffle_ps(hi, hi, _MM_SHUFFLE(2,3,0,1)));
    hi = _mm_max_ps(hi, _mm_shuffle_ps(hi, hi, _MM_SHUFFLE(1,0,3,2)));
    float max_val = _mm_cvtss_f32(hi);
    for (; i < n; i++) max_val = std::max(max_val, src[i]);

    __m256 mx_bc = _mm256_set1_ps(max_val);
    __m256 sumv = _mm256_setzero_ps();
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 e = exp_ps(_mm256_sub_ps(v, mx_bc));
        _mm256_storeu_ps(dst + i, e);
        sumv = _mm256_add_ps(sumv, e);
    }
    __m128 sum_hi = _mm_add_ps(_mm256_castps256_ps128(sumv),
                                _mm256_extractf128_ps(sumv, 1));
    sum_hi = _mm_hadd_ps(sum_hi, sum_hi);
    sum_hi = _mm_hadd_ps(sum_hi, sum_hi);
    double total = (double)_mm_cvtss_f32(sum_hi);
    for (; i < n; i++) {
        float e = std::exp(src[i] - max_val);
        dst[i] = e;
        total += e;
    }
    double inv = 1.0 / total;
    __m256 invv = _mm256_set1_ps((float)inv);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 e = _mm256_loadu_ps(dst + i);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(e, invv));
    }
    for (; i < n; i++) dst[i] = (float)((double)dst[i] * inv);
}

void vec_layer_norm(float* dst, const float* src, const float* gamma,
                    const float* beta, int n, float eps) {
    __m256 sumv = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        sumv = _mm256_add_ps(sumv, _mm256_loadu_ps(src + i));
    }
    __m128 hi = _mm_add_ps(_mm256_castps256_ps128(sumv),
                            _mm256_extractf128_ps(sumv, 1));
    hi = _mm_hadd_ps(hi, hi);
    hi = _mm_hadd_ps(hi, hi);
    double mu = (double)_mm_cvtss_f32(hi);
    for (; i < n; i++) mu += src[i];
    mu /= (double)n;

    __m256 muv = _mm256_set1_ps((float)mu);
    sumv = _mm256_setzero_ps();
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(src + i), muv);
        sumv = _mm256_fmadd_ps(d, d, sumv);
    }
    hi = _mm_add_ps(_mm256_castps256_ps128(sumv),
                    _mm256_extractf128_ps(sumv, 1));
    hi = _mm_hadd_ps(hi, hi);
    hi = _mm_hadd_ps(hi, hi);
    double var = (double)_mm_cvtss_f32(hi);
    for (; i < n; i++) { double d = src[i] - mu; var += d * d; }
    var /= (double)n;

    double inv_std = 1.0 / std::sqrt(var + (double)eps);
    __m256 invv = _mm256_set1_ps((float)inv_std);

    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 g = gamma ? _mm256_loadu_ps(gamma + i) : _mm256_set1_ps(1.0f);
        __m256 b = beta ? _mm256_loadu_ps(beta + i) : _mm256_setzero_ps();
        __m256 nv = _mm256_mul_ps(_mm256_sub_ps(v, muv), invv);
        _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(nv, g, b));
    }
    for (; i < n; i++) {
        dst[i] = (float)(((double)src[i] - mu) * inv_std);
        if (gamma) dst[i] *= gamma[i];
        if (beta) dst[i] += beta[i];
    }
}

void vec_rms_norm(float* dst, const float* src, const float* gamma,
                  int n, float eps) {
    __m256 sumv = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        sumv = _mm256_fmadd_ps(v, v, sumv);
    }
    __m128 hi = _mm_add_ps(_mm256_castps256_ps128(sumv),
                            _mm256_extractf128_ps(sumv, 1));
    hi = _mm_hadd_ps(hi, hi);
    hi = _mm_hadd_ps(hi, hi);
    float ss = _mm_cvtss_f32(hi);
    for (; i < n; i++) ss += src[i] * src[i];
    ss /= (float)n;

    double inv = 1.0 / std::sqrt((double)ss + (double)eps);
    __m256 invv = _mm256_set1_ps((float)inv);

    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 g = gamma ? _mm256_loadu_ps(gamma + i) : _mm256_set1_ps(1.0f);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_mul_ps(v, invv), g));
    }
    for (; i < n; i++) {
        double val = (double)src[i] * inv;
        if (gamma) val *= gamma[i];
        dst[i] = (float)val;
    }
}

// ---------------------------------------------------------------------------
// AVX2-explicit function variants (alias or identical to vec_* above)
// ---------------------------------------------------------------------------
void vec_exp_avx2(float* dst, const float* src, int n) { vec_exp(dst, src, n); }
void vec_log_avx2(float* dst, const float* src, int n) { vec_log(dst, src, n); }
void vec_sigmoid_avx2(float* dst, const float* src, int n) { vec_sigmoid(dst, src, n); }
void vec_tanh_avx2(float* dst, const float* src, int n) { vec_tanh(dst, src, n); }
void vec_pow_avx2(float* dst, const float* src, float exp_val, int n) { vec_pow(dst, src, exp_val, n); }
void vec_erf_avx2(float* dst, const float* src, int n) { vec_erf(dst, src, n); }

// ---------------------------------------------------------------------------
// Additional raw-pointer SIMD functions
// ---------------------------------------------------------------------------

void vec_relu(float* dst, const float* src, int n) {
    int i = 0;
    __m256 zeros = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_max_ps(v, zeros));
    }
    for (; i < n; i++) dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
}

void vec_gelu(float* dst, const float* src, int n) {
    int i = 0;
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 s = _mm256_set1_ps(0.7071067811865475f);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 xv = _mm256_mul_ps(v, s);
        __m256 x2 = _mm256_mul_ps(xv, xv);
        __m256 x3 = _mm256_mul_ps(x2, xv);
        __m256 x5 = _mm256_mul_ps(x3, x2);
        __m256 x7 = _mm256_mul_ps(x5, x2);
        __m256 x9 = _mm256_mul_ps(x7, x2);
        __m256 erv = _mm256_sub_ps(xv, _mm256_mul_ps(x3, _mm256_set1_ps(1.0f/3.0f)));
        erv = _mm256_add_ps(erv, _mm256_mul_ps(x5, _mm256_set1_ps(1.0f/10.0f)));
        erv = _mm256_sub_ps(erv, _mm256_mul_ps(x7, _mm256_set1_ps(1.0f/42.0f)));
        erv = _mm256_add_ps(erv, _mm256_mul_ps(x9, _mm256_set1_ps(1.0f/216.0f)));
        __m256 r = _mm256_mul_ps(half, _mm256_mul_ps(v, _mm256_add_ps(one, erv)));
        _mm256_storeu_ps(dst + i, r);
    }
    for (; i < n; i++) dst[i] = 0.5f * src[i] * (1.0f + std::erf(src[i] * 0.7071067811865475f));
}

void vec_silu(float* dst, const float* src, int n) {
    int i = 0;
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 clamp_hi = _mm256_set1_ps(88.376f);
    __m256 clamp_lo = _mm256_set1_ps(-88.376f);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), v);
        __m256 cl = _mm256_min_ps(_mm256_max_ps(neg, clamp_lo), clamp_hi);
        __m256 e = exp_ps(cl);
        __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(v, sig));
    }
    for (; i < n; i++) dst[i] = src[i] / (1.0f + std::exp(-src[i]));
}

void vec_add(float* dst, const float* a, const float* b, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    for (; i < n; i++) dst[i] = a[i] + b[i];
}

void vec_sub(float* dst, const float* a, const float* b, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    for (; i < n; i++) dst[i] = a[i] - b[i];
}

void vec_mul(float* dst, const float* a, const float* b, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    for (; i < n; i++) dst[i] = a[i] * b[i];
}

void vec_scale(float* dst, const float* src, float factor, int n) {
    int i = 0;
    __m256 fv = _mm256_set1_ps(factor);
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(fv, _mm256_loadu_ps(src + i)));
    }
    for (; i < n; i++) dst[i] = src[i] * factor;
}

void vec_negate(float* dst, const float* src, int n) {
    int i = 0;
    __m256 neg_mask = _mm256_set1_ps(-0.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_xor_ps(v, neg_mask));
    }
    for (; i < n; i++) dst[i] = -src[i];
}

void vec_abs(float* dst, const float* src, int n) {
    int i = 0;
    __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_and_ps(v, abs_mask));
    }
    for (; i < n; i++) dst[i] = std::abs(src[i]);
}

void vec_sqrt(float* dst, const float* src, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_max_ps(_mm256_loadu_ps(src + i), _mm256_set1_ps(0.0f));
        _mm256_storeu_ps(dst + i, _mm256_sqrt_ps(v));
    }
    for (; i < n; i++) dst[i] = std::sqrt(std::max(0.0f, src[i]));
}

void vec_dot(float* result, const float* a, const float* b, int n) {
    __m256 sumv = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sumv = _mm256_fmadd_ps(va, vb, sumv);
    }
    __m128 hi = _mm_add_ps(_mm256_castps256_ps128(sumv), _mm256_extractf128_ps(sumv, 1));
    hi = _mm_hadd_ps(hi, hi);
    hi = _mm_hadd_ps(hi, hi);
    *result = _mm_cvtss_f32(hi);
    for (; i < n; i++) *result += a[i] * b[i];
}

void vec_norm(float* result, const float* src, int n) {
    __m256 sumv = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        sumv = _mm256_fmadd_ps(v, v, sumv);
    }
    __m128 hi = _mm_add_ps(_mm256_castps256_ps128(sumv), _mm256_extractf128_ps(sumv, 1));
    hi = _mm_hadd_ps(hi, hi);
    hi = _mm_hadd_ps(hi, hi);
    double s = (double)_mm_cvtss_f32(hi);
    for (; i < n; i++) s += (double)src[i] * src[i];
    *result = std::sqrt((float)s);
}

void vec_sum(float* result, const float* src, int n) {
    __m256 sumv = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        sumv = _mm256_add_ps(sumv, _mm256_loadu_ps(src + i));
    }
    __m128 hi = _mm_add_ps(_mm256_castps256_ps128(sumv), _mm256_extractf128_ps(sumv, 1));
    hi = _mm_hadd_ps(hi, hi);
    hi = _mm_hadd_ps(hi, hi);
    double s = (double)_mm_cvtss_f32(hi);
    for (; i < n; i++) s += src[i];
    *result = (float)s;
}

void vec_maximum(float* dst, const float* a, const float* b, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_max_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    for (; i < n; i++) dst[i] = std::max(a[i], b[i]);
}

void vec_minimum(float* dst, const float* a, const float* b, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_min_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    for (; i < n; i++) dst[i] = std::min(a[i], b[i]);
}

void vec_clip(float* dst, const float* src, float lo, float hi, int n) {
    int i = 0;
    __m256 lov = _mm256_set1_ps(lo);
    __m256 hiv = _mm256_set1_ps(hi);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_min_ps(_mm256_max_ps(v, lov), hiv));
    }
    for (; i < n; i++) dst[i] = std::max(lo, std::min(hi, src[i]));
}

void vec_lerp(float* dst, const float* a, const float* b, float t, int n) {
    int i = 0;
    __m256 tv = _mm256_set1_ps(t);
    __m256 one_minus_t = _mm256_set1_ps(1.0f - t);
    for (; i + 8 <= n; i += 8) {
        __m256 av = _mm256_loadu_ps(a + i);
        __m256 bv = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_mul_ps(av, one_minus_t), _mm256_mul_ps(bv, tv)));
    }
    for (; i < n; i++) dst[i] = a[i] * (1.0f - t) + b[i] * t;
}

void vec_smoothstep(float* dst, const float* src, float edge0, float edge1, int n) {
    int i = 0;
    __m256 e0 = _mm256_set1_ps(edge0);
    __m256 e1 = _mm256_set1_ps(edge1);
    __m256 zero = _mm256_setzero_ps();
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 three = _mm256_set1_ps(3.0f);
    __m256 two = _mm256_set1_ps(-2.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 tval = _mm256_div_ps(_mm256_sub_ps(v, e0), _mm256_sub_ps(e1, e0));
        tval = _mm256_min_ps(_mm256_max_ps(tval, zero), one);
        // t*t*(3-2*t)
        __m256 tsq = _mm256_mul_ps(tval, tval);
        __m256 tcube = _mm256_mul_ps(tsq, tval);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_mul_ps(three, tsq), _mm256_mul_ps(two, tcube)));
    }
    for (; i < n; i++) {
        float tval = std::max(0.0f, std::min(1.0f, (src[i] - edge0) / (edge1 - edge0)));
        dst[i] = tval * tval * (3.0f - 2.0f * tval);
    }
}

void vec_fp32_to_fp16(uint16_t* dst, const float* src, int n) {
    // AVX2: convert 8 floats to 8 half-floats using F16C
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), h);
    }
    // Fallback for remainder and non-F16C: use scalar
    for (; i < n; i++) {
        float f = src[i];
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        uint16_t sign = (uint16_t)((bits >> 16) & 0x8000);
        int32_t exp = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = bits & 0x007FFFFF;
        if (exp <= 0) {
            if (exp < -10) { dst[i] = sign; }
            else { mant = (mant | 0x00800000) >> (1 - exp); dst[i] = (uint16_t)(sign | (mant >> 13)); }
        } else if (exp >= 31) {
            dst[i] = (uint16_t)(sign | 0x7C00 | (mant != 0 ? 0x0200 : 0));
        } else {
            dst[i] = (uint16_t)(sign | ((uint16_t)exp << 10) | (mant >> 13));
        }
    }
}

void vec_fp16_to_fp32(float* dst, const uint16_t* src, int n) {
    // AVX2: convert 8 half-floats to 8 floats using F16C
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256 v = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(dst + i, v);
    }
    for (; i < n; i++) {
        uint16_t h = src[i];
        uint32_t sign = (uint32_t)(h >> 15) << 31;
        int32_t exp = (int32_t)((h >> 10) & 0x1F);
        uint32_t mant = h & 0x03FF;
        if (exp == 0) {
            if (mant == 0) { dst[i] = 0.0f; }
            else { float m = (float)mant / 1024.0f; dst[i] = (sign ? -1.0f : 1.0f) * m * 1.5258789e-5f; }
        } else if (exp == 31) {
            uint32_t f32 = sign | 0x7F800000 | (mant << 13);
            std::memcpy(&dst[i], &f32, sizeof(float));
        } else {
            uint32_t f32 = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
            std::memcpy(&dst[i], &f32, sizeof(float));
        }
    }
}

void vec_softmax_stable_avx2(float* dst, const float* src, int n) {
    vec_softmax_stable(dst, src, n);
}

void vec_layer_norm_avx2(float* dst, const float* src, const float* gamma, const float* beta, int n, float eps) {
    vec_layer_norm(dst, src, gamma, beta, n, eps);
}

void vec_rms_norm_avx2(float* dst, const float* src, const float* gamma, int n, float eps) {
    vec_rms_norm(dst, src, gamma, n, eps);
}

} // namespace math
} // namespace quant

#endif // QUANT_AVX2