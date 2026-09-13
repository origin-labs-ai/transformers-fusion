#include "quant/math.h"
#include "quant/tensor.h"

#include <cmath>
#include <cstring>
#include <algorithm>

#if !defined(QUANT_AVX2)

namespace quant {
namespace math {

static inline const float* rd(const Tensor& t) { return t.data<float>(); }
static inline float* wr(Tensor& t) { return t.data<float>(); }

float dot(const Tensor& a, const Tensor& b) {
    QUANT_CHECK(a.numel() == b.numel(), "dot shape mismatch");
    const float* pa = rd(a);
    const float* pb = rd(b);
    int64_t n = a.numel();
    float s = 0.0f;
    for (int64_t i = 0; i < n; i++) s += pa[i] * pb[i];
    return s;
}

void axpy(float alpha, const Tensor& x, Tensor& y) {
    QUANT_CHECK(x.numel() == y.numel(), "axpy shape mismatch");
    const float* px = rd(x);
    float* py = wr(y);
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; i++) py[i] += alpha * px[i];
}

float norm(const Tensor& x) {
    const float* p = rd(x);
    int64_t n = x.numel();
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) s += (double)p[i] * p[i];
    return (float)std::sqrt(s);
}

float asum(const Tensor& x) {
    const float* p = rd(x);
    int64_t n = x.numel();
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) s += std::abs((double)p[i]);
    return (float)s;
}

void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) {
    QUANT_CHECK(A.shape().rank == 2, "gemv A must be 2D");
    QUANT_CHECK(x.shape().rank == 1, "gemv x must be 1D");
    QUANT_CHECK(y.shape().rank == 1, "gemv y must be 1D");
    int64_t M = A.shape().dims[0];
    int64_t N = A.shape().dims[1];
    QUANT_CHECK(x.numel() == N, "gemv x size != A cols");
    QUANT_CHECK(y.numel() == M, "gemv y size != A rows");

    const float* pa = rd(A);
    const float* px = rd(x);
    float* py = wr(y);

    for (int64_t i = 0; i < M; i++) {
        float s = 0.0f;
        for (int64_t j = 0; j < N; j++) {
            s += pa[i * N + j] * px[j];
        }
        py[i] = alpha * s + beta * py[i];
    }
}

void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) {
    QUANT_CHECK(A.shape().rank == 2, "gemm A must be 2D");
    QUANT_CHECK(B.shape().rank == 2, "gemm B must be 2D");
    QUANT_CHECK(C.shape().rank == 2, "gemm C must be 2D");
    int64_t M = A.shape().dims[0];
    int64_t K = A.shape().dims[1];
    int64_t N = B.shape().dims[1];
    QUANT_CHECK(B.shape().dims[0] == K, "gemm A cols != B rows");
    QUANT_CHECK(C.shape().dims[0] == M && C.shape().dims[1] == N, "gemm C shape mismatch");

    const float* pa = rd(A);
    const float* pb = rd(B);
    float* pc = wr(C);

    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float s = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                s += pa[i * K + k] * pb[k * N + j];
            }
            pc[i * N + j] = alpha * s + beta * pc[i * N + j];
        }
    }
}

void relu(const Tensor& x, Tensor& y) {
    QUANT_CHECK(x.numel() == y.numel(), "relu shape mismatch");
    const float* px = rd(x);
    float* py = wr(y);
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; i++) py[i] = px[i] > 0.0f ? px[i] : 0.0f;
}

void gelu(const Tensor& x, Tensor& y) {
    QUANT_CHECK(x.numel() == y.numel(), "gelu shape mismatch");
    const float* px = rd(x);
    float* py = wr(y);
    int64_t n = x.numel();
    const float s = 0.7071067811865475f;
    for (int64_t i = 0; i < n; i++) {
        float v = px[i];
        py[i] = 0.5f * v * (1.0f + std::erf(v * s));
    }
}

void silu(const Tensor& x, Tensor& y) {
    QUANT_CHECK(x.numel() == y.numel(), "silu shape mismatch");
    const float* px = rd(x);
    float* py = wr(y);
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; i++) {
        float v = px[i];
        py[i] = v / (1.0f + std::exp(-v));
    }
}

void sigmoid(const Tensor& x, Tensor& y) {
    QUANT_CHECK(x.numel() == y.numel(), "sigmoid shape mismatch");
    const float* px = rd(x);
    float* py = wr(y);
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; i++) {
        py[i] = 1.0f / (1.0f + std::exp(-px[i]));
    }
}

void tanh_(const Tensor& x, Tensor& y) {
    QUANT_CHECK(x.numel() == y.numel(), "tanh shape mismatch");
    const float* px = rd(x);
    float* py = wr(y);
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; i++) py[i] = std::tanh(px[i]);
}

void swiglu(const Tensor& gate, const Tensor& up, Tensor& y) {
    QUANT_CHECK(gate.numel() == up.numel() && up.numel() == y.numel(), "swiglu shape mismatch");
    const float* pg = rd(gate);
    const float* pu = rd(up);
    float* py = wr(y);
    int64_t n = gate.numel();
    for (int64_t i = 0; i < n; i++) {
        float g = pg[i];
        float silu = g / (1.0f + std::exp(-g));
        py[i] = silu * pu[i];
    }
}

void geglu(const Tensor& gate, const Tensor& up, Tensor& y) {
    QUANT_CHECK(gate.numel() == up.numel() && up.numel() == y.numel(), "geglu shape mismatch");
    const float* pg = rd(gate);
    const float* pu = rd(up);
    float* py = wr(y);
    int64_t n = gate.numel();
    const float s = 0.7071067811865475f;
    for (int64_t i = 0; i < n; i++) {
        float g = pg[i];
        float gel = 0.5f * g * (1.0f + std::erf(g * s));
        py[i] = gel * pu[i];
    }
}

void layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps, Tensor& y) {
    QUANT_CHECK(x.numel() == y.numel(), "layer_norm shape mismatch");
    QUANT_CHECK(gamma.numel() == x.shape().dims[x.shape().rank-1], "layer_norm gamma dim mismatch");
    QUANT_CHECK(beta.numel() == x.shape().dims[x.shape().rank-1], "layer_norm beta dim mismatch");

    int64_t last_dim = x.shape().dims[x.shape().rank - 1];
    int64_t outer = x.numel() / last_dim;

    const float* px = rd(x);
    const float* pg = rd(gamma);
    const float* pb = rd(beta);
    float* py = wr(y);

    for (int64_t i = 0; i < outer; i++) {
        const float* row = px + i * last_dim;
        double mu = 0.0;
        for (int64_t j = 0; j < last_dim; j++) mu += row[j];
        mu /= (double)last_dim;

        double var = 0.0;
        for (int64_t j = 0; j < last_dim; j++) {
            double d = row[j] - mu;
            var += d * d;
        }
        var /= (double)last_dim;

        float* ry = py + i * last_dim;
        double inv_std = 1.0 / std::sqrt(var + (double)eps);
        for (int64_t j = 0; j < last_dim; j++) {
            ry[j] = (float)(((double)row[j] - mu) * inv_std) * pg[j] + pb[j];
        }
    }
}

void rms_norm(const Tensor& x, const Tensor& gamma, float eps, Tensor& y) {
    QUANT_CHECK(x.numel() == y.numel(), "rms_norm shape mismatch");
    QUANT_CHECK(gamma.numel() == x.shape().dims[x.shape().rank-1], "rms_norm gamma dim mismatch");

    int64_t last_dim = x.shape().dims[x.shape().rank - 1];
    int64_t outer = x.numel() / last_dim;

    const float* px = rd(x);
    const float* pg = rd(gamma);
    float* py = wr(y);

    for (int64_t i = 0; i < outer; i++) {
        const float* row = px + i * last_dim;
        double ss = 0.0;
        for (int64_t j = 0; j < last_dim; j++) ss += (double)row[j] * row[j];
        ss /= (double)last_dim;

        double inv = 1.0 / std::sqrt(ss + (double)eps);
        float* ry = py + i * last_dim;
        for (int64_t j = 0; j < last_dim; j++) {
            ry[j] = (float)((double)row[j] * inv) * pg[j];
        }
    }
}

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

            float mx = -INFINITY;
            for (int64_t d = 0; d < dim; d++) {
                mx = std::max(mx, src[d * inner]);
            }

            double sum = 0.0;
            for (int64_t d = 0; d < dim; d++) {
                float v = std::exp(src[d * inner] - mx);
                dst[d * inner] = v;
                sum += v;
            }

            double inv = 1.0 / sum;
            for (int64_t d = 0; d < dim; d++) {
                dst[d * inner] = (float)((double)dst[d * inner] * inv);
            }
        }
    }
}

void add(const Tensor& a, const Tensor& b, Tensor& c) {
    QUANT_CHECK(a.numel() == b.numel() && b.numel() == c.numel(), "add shape mismatch");
    const float* pa = rd(a);
    const float* pb = rd(b);
    float* pc = wr(c);
    int64_t n = a.numel();
    for (int64_t i = 0; i < n; i++) pc[i] = pa[i] + pb[i];
}

void sub(const Tensor& a, const Tensor& b, Tensor& c) {
    QUANT_CHECK(a.numel() == b.numel() && b.numel() == c.numel(), "sub shape mismatch");
    const float* pa = rd(a);
    const float* pb = rd(b);
    float* pc = wr(c);
    int64_t n = a.numel();
    for (int64_t i = 0; i < n; i++) pc[i] = pa[i] - pb[i];
}

void mul(const Tensor& a, const Tensor& b, Tensor& c) {
    QUANT_CHECK(a.numel() == b.numel() && b.numel() == c.numel(), "mul shape mismatch");
    const float* pa = rd(a);
    const float* pb = rd(b);
    float* pc = wr(c);
    int64_t n = a.numel();
    for (int64_t i = 0; i < n; i++) pc[i] = pa[i] * pb[i];
}

void scale(float s, const Tensor& x, Tensor& y) {
    QUANT_CHECK(x.numel() == y.numel(), "scale shape mismatch");
    const float* px = rd(x);
    float* py = wr(y);
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; i++) py[i] = s * px[i];
}

float mean(const Tensor& x) {
    float s = sum(x);
    return s / (float)x.numel();
}

float sum(const Tensor& x) {
    const float* p = rd(x);
    int64_t n = x.numel();
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) s += p[i];
    return (float)s;
}

float max(const Tensor& x) {
    const float* p = rd(x);
    int64_t n = x.numel();
    QUANT_CHECK(n > 0, "max on empty tensor");
    float m = p[0];
    for (int64_t i = 1; i < n; i++) m = std::max(m, p[i]);
    return m;
}

Tensor zeros_like(const Tensor& x) {
    Tensor t(x.shape(), x.dtype());
    t.zero_();
    return t;
}

Tensor ones_like(const Tensor& x) {
    Tensor t(x.shape(), x.dtype());
    t.fill(1.0f);
    return t;
}

// ===========================================================================
// SIMD vector math functions — scalar fallbacks
// ===========================================================================

void vec_exp(float* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) dst[i] = std::exp(std::max(-88.376f, std::min(88.376f, src[i])));
}

void vec_log(float* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) dst[i] = std::log(std::max(1e-38f, src[i]));
}

void vec_sigmoid(float* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
}

void vec_tanh(float* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) dst[i] = std::tanh(src[i]);
}

void vec_pow(float* dst, const float* src, float exp_val, int n) {
    for (int i = 0; i < n; i++) dst[i] = std::pow(src[i], exp_val);
}

void vec_erf(float* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) dst[i] = std::erf(src[i]);
}

void vec_softmax_stable(float* dst, const float* src, int n) {
    if (n <= 0) return;
    float max_val = src[0];
    for (int i = 1; i < n; i++) max_val = std::max(max_val, src[i]);
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        float e = std::exp(src[i] - max_val);
        dst[i] = e;
        sum += e;
    }
    double inv = 1.0 / sum;
    for (int i = 0; i < n; i++) dst[i] = (float)((double)dst[i] * inv);
}

void vec_layer_norm(float* dst, const float* src, const float* gamma,
                    const float* beta, int n, float eps) {
    double mu = 0.0;
    for (int i = 0; i < n; i++) mu += src[i];
    mu /= (double)n;

    double var = 0.0;
    for (int i = 0; i < n; i++) { double d = src[i] - mu; var += d * d; }
    var /= (double)n;

    double inv_std = 1.0 / std::sqrt(var + (double)eps);
    for (int i = 0; i < n; i++) {
        double val = ((double)src[i] - mu) * inv_std;
        if (gamma) val *= gamma[i];
        if (beta) val += beta[i];
        dst[i] = (float)val;
    }
}

void vec_rms_norm(float* dst, const float* src, const float* gamma,
                  int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)src[i] * src[i];
    ss /= (double)n;

    double inv = 1.0 / std::sqrt(ss + (double)eps);
    for (int i = 0; i < n; i++) {
        double val = (double)src[i] * inv;
        if (gamma) val *= gamma[i];
        dst[i] = (float)val;
    }
}

// Scalar fallback — real AVX2 in math_avx2.cpp
void vec_exp_avx2(float* dst, const float* src, int n) { vec_exp(dst, src, n); }
void vec_log_avx2(float* dst, const float* src, int n) { vec_log(dst, src, n); }
void vec_sigmoid_avx2(float* dst, const float* src, int n) { vec_sigmoid(dst, src, n); }
void vec_tanh_avx2(float* dst, const float* src, int n) { vec_tanh(dst, src, n); }
void vec_pow_avx2(float* dst, const float* src, float exp_val, int n) { vec_pow(dst, src, exp_val, n); }
void vec_erf_avx2(float* dst, const float* src, int n) { vec_erf(dst, src, n); }

// ---------------------------------------------------------------------------
// Additional scalar vector functions
// ---------------------------------------------------------------------------

void vec_relu(float* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
}

void vec_gelu(float* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = 0.5f * src[i] * (1.0f + std::erf(src[i] * 0.7071067811865475f));
    }
}

void vec_silu(float* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = src[i] / (1.0f + std::exp(-src[i]));
    }
}

void vec_add(float* dst, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) dst[i] = a[i] + b[i];
}

void vec_sub(float* dst, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) dst[i] = a[i] - b[i];
}

void vec_mul(float* dst, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) dst[i] = a[i] * b[i];
}

void vec_scale(float* dst, const float* src, float factor, int n) {
    for (int i = 0; i < n; i++) dst[i] = src[i] * factor;
}

void vec_negate(float* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) dst[i] = -src[i];
}

void vec_abs(float* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) dst[i] = std::abs(src[i]);
}

void vec_sqrt(float* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) dst[i] = std::sqrt(std::max(0.0f, src[i]));
}

void vec_dot(float* result, const float* a, const float* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)a[i] * b[i];
    *result = (float)s;
}

void vec_norm(float* result, const float* src, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)src[i] * src[i];
    *result = (float)std::sqrt(s);
}

void vec_sum(float* result, const float* src, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += src[i];
    *result = (float)s;
}

void vec_maximum(float* dst, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) dst[i] = std::max(a[i], b[i]);
}

void vec_minimum(float* dst, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) dst[i] = std::min(a[i], b[i]);
}

void vec_clip(float* dst, const float* src, float lo, float hi, int n) {
    for (int i = 0; i < n; i++) dst[i] = std::max(lo, std::min(hi, src[i]));
}

void vec_lerp(float* dst, const float* a, const float* b, float t, int n) {
    for (int i = 0; i < n; i++) dst[i] = a[i] * (1.0f - t) + b[i] * t;
}

void vec_smoothstep(float* dst, const float* src, float edge0, float edge1, int n) {
    for (int i = 0; i < n; i++) {
        float tval = std::max(0.0f, std::min(1.0f, (src[i] - edge0) / (edge1 - edge0)));
        dst[i] = tval * tval * (3.0f - 2.0f * tval);
    }
}

void vec_fp32_to_fp16(uint16_t* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) {
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
    for (int i = 0; i < n; i++) {
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

// Scalar fallback — real AVX2 in math_avx2.cpp
void vec_softmax_stable_avx2(float* dst, const float* src, int n) { vec_softmax_stable(dst, src, n); }
void vec_layer_norm_avx2(float* dst, const float* src, const float* gamma, const float* beta, int n, float eps) { vec_layer_norm(dst, src, gamma, beta, n, eps); }
void vec_rms_norm_avx2(float* dst, const float* src, const float* gamma, int n, float eps) { vec_rms_norm(dst, src, gamma, n, eps); }

// ===========================================================================
// FP8 E4M3 / E5M2 — scalar implementations (AVX2 in math_avx2.cpp)
// ===========================================================================

static float fp8_e4m3_bits_to_f32(uint8_t v) {
    if (v == 0) return 0.0f;
    int sign = (v & 0x80) ? -1 : 1;
    int exp = (v & 0x78) >> 3;
    int mant = v & 0x07;
    if (exp == 0) return sign * std::pow(2.0f, -6) * (mant / 8.0f);
    // OCP E4M3: only mant==7 at exp==15 is NaN; mant 0..6 are finite
    // (0x78..0x7E = 256..448). Treating them as NaN collapsed real
    // small-weight GEMMs to NaN once the encoder could emit them.
    if (exp == 15) {
        if (mant == 0x07) return std::nanf("");
        return sign * std::pow(2.0f, 8) * (1.0f + mant / 8.0f);
    }
    return sign * std::pow(2.0f, exp - 7) * (1.0f + mant / 8.0f);
}

static float fp8_e5m2_bits_to_f32(uint8_t v) {
    if ((v & 0x7F) == 0) return 0.0f;
    int sign = (v & 0x80) ? -1 : 1;
    int exp = (v & 0x7C) >> 2;
    int mant = v & 0x03;
    if (exp == 0) return sign * std::pow(2.0f, -14) * (mant / 4.0f);
    if (exp == 31) return (mant == 0) ? (sign * INFINITY) : std::nanf("");
    return sign * std::pow(2.0f, exp - 15) * (1.0f + mant / 4.0f);
}

static uint8_t f32_to_fp8_e4m3_bits(float f) {
    if (f == 0.0f) return 0;
    if (std::isnan(f)) return 0x7F;
    if (std::isinf(f)) return f > 0 ? 0x7E : 0xFE;
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    int sign = (int)((bits >> 31) & 1);
    int exp = ((int)((bits >> 23) & 0xFF)) - 127;
    int mant = (int)(bits & 0x7FFFFF);
    if (exp > 8) return (uint8_t)((sign << 7) | 0x7E);  // saturate to 448
    if (exp < -6) {
        // Subnormal region (|f| < 2^-6): value = 2^-6 * (m/8).
        // The old code fell through with target_exp <= 0, shifting a
        // negative exponent into the code — small weights (e.g. 0.0044)
        // encoded to 0xF9-class patterns that decode as NaN, poisoning
        // every FP8 attention projection. Encode subnormals properly.
        float af = std::fabs(f);
        int m = (int)std::lround(af * 512.0f);  // 2^9
        if (m <= 0) return (uint8_t)(sign << 7);
        if (m >= 8) return (uint8_t)((sign << 7) | (1 << 3));  // normal min
        return (uint8_t)((sign << 7) | m);
    }
    int target_exp = exp + 7;
    int target_mant = (mant >> 20) & 0x07;
    if ((mant >> 19) & 1) {
        target_mant++;
        if (target_mant >= 8) { target_mant = 0; target_exp++; if (target_exp >= 15) return (uint8_t)((sign << 7) | 0x7E); }
    }
    if (target_exp <= 0) return (uint8_t)(sign << 7);  // unreachable given exp >= -6, kept as guard
    if (target_exp >= 15) return (uint8_t)((sign << 7) | 0x7E);
    return (uint8_t)((sign << 7) | (target_exp << 3) | target_mant);
}

static uint8_t f32_to_fp8_e5m2_bits(float f) {
    if (f == 0.0f) return 0;
    if (std::isnan(f)) return 0x7F;
    if (std::isinf(f)) return f > 0 ? 0x7C : 0xFC;
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    int sign = (int)((bits >> 31) & 1);
    int exp = ((int)((bits >> 23) & 0xFF)) - 127;
    int mant = (int)(bits & 0x7FFFFF);
    if (exp > 15) return (uint8_t)((sign << 7) | 0x7C);
    if (exp < -14) {
        // Subnormal region (|f| < 2^-14): value = 2^-14 * (m/4).
        // Same negative-shift hazard as E4M3 (exp=-16 gave te=-1 garbage
        // that decoded as Inf/NaN); encode subnormals explicitly.
        float af = std::fabs(f);
        int m = (int)std::lround(af * 65536.0f);  // 2^16
        if (m <= 0) return (uint8_t)(sign << 7);
        if (m >= 4) return (uint8_t)((sign << 7) | (1 << 2));  // normal min
        return (uint8_t)((sign << 7) | m);
    }
    int target_exp = exp + 15;
    int target_mant = (mant >> 21) & 0x03;
    if ((mant >> 20) & 1) {
        target_mant++;
        if (target_mant >= 4) { target_mant = 0; target_exp++; if (target_exp >= 31) return (uint8_t)((sign << 7) | 0x7C); }
    }
    if (target_exp <= 0) return (uint8_t)(sign << 7);  // unreachable given exp >= -14, kept as guard
    if (target_exp >= 31) return (uint8_t)((sign << 7) | 0x7C);
    return (uint8_t)((sign << 7) | (target_exp << 2) | target_mant);
}

float f32_to_fp8_e4m3_scalar(float val) { return (float)f32_to_fp8_e4m3_bits(val); }
float fp8_e4m3_to_f32_scalar(uint8_t val) { return fp8_e4m3_bits_to_f32(val); }
void vec_fp32_to_fp8_e4m3(uint8_t* dst, const float* src, int n) { for (int i = 0; i < n; i++) dst[i] = f32_to_fp8_e4m3_bits(src[i]); }
void vec_fp8_e4m3_to_fp32(float* dst, const uint8_t* src, int n) { for (int i = 0; i < n; i++) dst[i] = fp8_e4m3_bits_to_f32(src[i]); }
float f32_to_fp8_e5m2_scalar(float val) { return (float)f32_to_fp8_e5m2_bits(val); }
float fp8_e5m2_to_f32_scalar(uint8_t val) { return fp8_e5m2_bits_to_f32(val); }
void vec_fp32_to_fp8_e5m2(uint8_t* dst, const float* src, int n) { for (int i = 0; i < n; i++) dst[i] = f32_to_fp8_e5m2_bits(src[i]); }
void vec_fp8_e5m2_to_fp32(float* dst, const uint8_t* src, int n) { for (int i = 0; i < n; i++) dst[i] = fp8_e5m2_bits_to_f32(src[i]); }

void fp8_gemm(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K, bool use_e4m3) {
    for (int64_t i = 0; i < M * N; i++) C[i] = 0.0f;
    for (int64_t m = 0; m < M; m++) {
        for (int64_t k = 0; k < K; k++) {
            float a_q = use_e4m3 ? fp8_e4m3_bits_to_f32(f32_to_fp8_e4m3_bits(A[m * K + k]))
                                 : fp8_e5m2_bits_to_f32(f32_to_fp8_e5m2_bits(A[m * K + k]));
            for (int64_t n_ = 0; n_ < N; n_++) {
                float b_q = use_e4m3 ? fp8_e4m3_bits_to_f32(f32_to_fp8_e4m3_bits(B[k * N + n_]))
                                     : fp8_e5m2_bits_to_f32(f32_to_fp8_e5m2_bits(B[k * N + n_]));
                C[m * N + n_] += a_q * b_q;
            }
        }
    }
}

void vec_geglu(float* dst, const float* gate, const float* up, int n) {
    const float s = 0.7071067811865475f;
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        float gel = 0.5f * g * (1.0f + std::erf(g * s));
        dst[i] = gel * up[i];
    }
}

void vec_geglu_avx2(float* dst, const float* gate, const float* up, int n) { vec_geglu(dst, gate, up, n); }

} // namespace math
} // namespace quant

#endif // !QUANT_AVX2
