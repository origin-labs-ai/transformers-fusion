#include "quant/quant_engines.h"
#include "quant/math.h"
#include "quant/codebook.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cfloat>

namespace quant {
namespace engines {

// ===========================================================================
// FP8 E4M3: 1 sign + 4 exponent + 3 mantissa bits
// Range: [-448, 448], no inf/nan (uses all 8-bit patterns for values)
// ===========================================================================

float fp8_e4m3_dequantize(uint8_t bits) {
    int sign = (bits >> 7) & 1;
    int exp_raw = (bits >> 3) & 0xF;
    int mant = bits & 0x7;
    float val;
    if (exp_raw == 0) {
        val = (float)mant * (1.0f / 64.0f);
    } else if (exp_raw == 0xF && mant == 0x7) {
        val = 0.0f;
    } else {
        int exp = exp_raw - 7;
        val = ldexpf(1.0f + (float)mant / 8.0f, exp);
    }
    return sign ? -val : val;
}

uint8_t fp8_e4m3_quantize(float val) {
    if (val == 0.0f) return 0;
    int sign = 0;
    if (val < 0) { sign = 1; val = -val; }
    if (val > 448.0f) val = 448.0f;
    int exp = 0;
    float mant = val;
    while (mant >= 2.0f && exp < 8) { mant /= 2.0f; exp++; }
    while (mant < 1.0f && exp > -7) { mant *= 2.0f; exp--; }
    if (exp < -7) {
        int m = (int)std::round(mant * 64.0f);
        if (m > 7) m = 7;
        return (uint8_t)((sign << 7) | m);
    }
    int m = (int)std::round((mant - 1.0f) * 8.0f);
    if (m > 7) m = 7;
    if (m < 0) m = 0;
    return (uint8_t)((sign << 7) | ((exp + 7) << 3) | m);
}

Tensor fp8_e4m3_dequant_tensor(const uint8_t* data, int64_t n) {
    Tensor out({n});
    float* od = out.data<float>();
#ifdef QUANT_HAS_AVX2
    if (n >= 16) {
        dequant_tensor_fp8_avx2(data, od, n, false);
        return out;
    }
#endif
    for (int64_t i = 0; i < n; ++i)
        od[i] = fp8_e4m3_dequantize(data[i]);
    return out;
}

// ===========================================================================
// FP8 E5M2: 1 sign + 5 exponent + 2 mantissa bits
// Range: [-57344, 57344], supports inf/nan
// ===========================================================================

float fp8_e5m2_dequantize(uint8_t bits) {
    int sign = (bits >> 7) & 1;
    int exp_raw = (bits >> 2) & 0x1F;
    int mant = bits & 0x3;
    float val;
    if (exp_raw == 0) {
        val = (float)mant * (1.0f / 16384.0f);
    } else if (exp_raw == 0x1F) {
        val = (mant == 0) ? INFINITY : NAN;
    } else {
        int exp = exp_raw - 15;
        val = ldexpf(1.0f + (float)mant / 4.0f, exp);
    }
    return sign ? -val : val;
}

uint8_t fp8_e5m2_quantize(float val) {
    if (val == 0.0f) return 0;
    int sign = 0;
    if (val < 0) { sign = 1; val = -val; }
    if (val > 57344.0f) val = 57344.0f;
    int exp = 0;
    float mant = val;
    while (mant >= 2.0f && exp < 16) { mant /= 2.0f; exp++; }
    while (mant < 1.0f && exp > -15) { mant *= 2.0f; exp--; }
    if (exp < -15) {
        int m = (int)std::round(mant * 16384.0f);
        if (m > 3) m = 3;
        return (uint8_t)((sign << 7) | m);
    }
    int m = (int)std::round((mant - 1.0f) * 4.0f);
    if (m > 3) m = 3;
    if (m < 0) m = 0;
    return (uint8_t)((sign << 7) | ((exp + 15) << 2) | m);
}

Tensor fp8_e5m2_dequant_tensor(const uint8_t* data, int64_t n) {
    Tensor out({n});
    float* od = out.data<float>();
#ifdef QUANT_HAS_AVX2
    if (n >= 16) {
        dequant_tensor_fp8_avx2(data, od, n, true);
        return out;
    }
#endif
    for (int64_t i = 0; i < n; ++i)
        od[i] = fp8_e5m2_dequantize(data[i]);
    return out;
}

// ===========================================================================
// Roundtrip test helpers
// ===========================================================================

float compute_quant_error(const Tensor& original, const Tensor& dequantized) {
    int64_t n = original.numel();
    float max_err = 0;
    const float* od = original.data<float>();
    const float* dd = dequantized.data<float>();
    for (int64_t i = 0; i < n; ++i) {
        float err = std::abs(od[i] - dd[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// ===========================================================================
// FP8 E4M3: Batch operations and extensions
// ===========================================================================

Tensor fp8_e4m3_quantize_tensor(const float* data, int64_t n) {
    Tensor out({n}, quant::DType::U8);
    uint8_t* od = out.data<uint8_t>();
    for (int64_t i = 0; i < n; ++i)
        od[i] = fp8_e4m3_quantize(data[i]);
    return out;
}

Tensor fp8_e4m3_quant_gemm(const Tensor& a, const uint8_t* b_q,
                           int64_t M, int64_t N, int64_t K) {
    Tensor C({M, N});
    C.zero_();
    const float* ad = a.data<float>();
    float* cd = C.data<float>();
#ifdef QUANT_HAS_AVX2
    quant_gemm_fp8_avx2(ad, cd, b_q, M, N, K, false);
#else
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = ad[m * K + k];
            if (a_val == 0.0f) continue;
            for (int64_t n = 0; n < N; ++n)
                cd[m * N + n] += a_val * fp8_e4m3_dequantize(b_q[k * N + n]);
        }
    }
#endif
    return C;
}

void fp8_e4m3_quantize_per_channel(const Tensor& t, int channel_dim,
                                    Tensor& q, Tensor& scales) {
    QUANT_CHECK(t.rank() == 2, "fp8_e4m3_quantize_per_channel expects 2D tensor");
    int64_t d0 = t.dim(0), d1 = t.dim(1);
    int64_t channels = (channel_dim == 0) ? d0 : d1;
    int64_t other = (channel_dim == 0) ? d1 : d0;
    q = Tensor(t.shape(), quant::DType::U8);
    scales = Tensor({channels});
    const float* td = t.data<float>();
    uint8_t* qd = q.data<uint8_t>();
    float* sd = scales.data<float>();
    for (int64_t c = 0; c < channels; ++c) {
        float max_abs = 0;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            max_abs = std::max(max_abs, std::abs(td[idx]));
        }
        if (max_abs < 1e-10f) max_abs = 1.0f;
        sd[c] = max_abs;
        float inv_scale = 448.0f / max_abs;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            qd[idx] = fp8_e4m3_quantize(td[idx] * inv_scale);
        }
    }
}

void fp8_e4m3_dequantize_per_channel(const Tensor& q, const Tensor& scales,
                                      int channel_dim, Tensor& out) {
    QUANT_CHECK(q.rank() == 2, "fp8_e4m3_dequantize_per_channel expects 2D q tensor");
    out = Tensor(q.shape());
    int64_t d0 = q.dim(0), d1 = q.dim(1);
    int64_t channels = (channel_dim == 0) ? d0 : d1;
    int64_t other = (channel_dim == 0) ? d1 : d0;
    const uint8_t* qd = q.data<uint8_t>();
    const float* sd = scales.data<float>();
    float* od = out.data<float>();
    for (int64_t c = 0; c < channels; ++c) {
        float mul = sd[c] / 448.0f;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            od[idx] = fp8_e4m3_dequantize(qd[idx]) * mul;
        }
    }
}

float fp8_e4m3_quant_error(const Tensor& original, const Tensor& reconstructed) {
    return compute_quant_mse(original, reconstructed);
}

float fp8_e4m3_quant_snr(const Tensor& original, const Tensor& reconstructed) {
    return compute_quant_snr(original, reconstructed);
}

// ===========================================================================
// FP8 E5M2: Batch operations and extensions
// ===========================================================================

Tensor fp8_e5m2_quantize_tensor(const float* data, int64_t n) {
    Tensor out({n}, quant::DType::U8);
    uint8_t* od = out.data<uint8_t>();
    for (int64_t i = 0; i < n; ++i)
        od[i] = fp8_e5m2_quantize(data[i]);
    return out;
}

Tensor fp8_e5m2_quant_gemm(const Tensor& a, const uint8_t* b_q,
                           int64_t M, int64_t N, int64_t K) {
    Tensor C({M, N});
    C.zero_();
    const float* ad = a.data<float>();
    float* cd = C.data<float>();
#ifdef QUANT_HAS_AVX2
    quant_gemm_fp8_avx2(ad, cd, b_q, M, N, K, true);
#else
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = ad[m * K + k];
            if (a_val == 0.0f) continue;
            for (int64_t n = 0; n < N; ++n)
                cd[m * N + n] += a_val * fp8_e5m2_dequantize(b_q[k * N + n]);
        }
    }
#endif
    return C;
}

void fp8_e5m2_quantize_per_channel(const Tensor& t, int channel_dim,
                                    Tensor& q, Tensor& scales) {
    QUANT_CHECK(t.rank() == 2, "fp8_e5m2_quantize_per_channel expects 2D tensor");
    int64_t d0 = t.dim(0), d1 = t.dim(1);
    int64_t channels = (channel_dim == 0) ? d0 : d1;
    int64_t other = (channel_dim == 0) ? d1 : d0;
    q = Tensor(t.shape(), quant::DType::U8);
    scales = Tensor({channels});
    const float* td = t.data<float>();
    uint8_t* qd = q.data<uint8_t>();
    float* sd = scales.data<float>();
    for (int64_t c = 0; c < channels; ++c) {
        float max_abs = 0;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            max_abs = std::max(max_abs, std::abs(td[idx]));
        }
        if (max_abs < 1e-10f) max_abs = 1.0f;
        sd[c] = max_abs;
        float inv_scale = 57344.0f / max_abs;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            qd[idx] = fp8_e5m2_quantize(td[idx] * inv_scale);
        }
    }
}

void fp8_e5m2_dequantize_per_channel(const Tensor& q, const Tensor& scales,
                                      int channel_dim, Tensor& out) {
    QUANT_CHECK(q.rank() == 2, "fp8_e5m2_dequantize_per_channel expects 2D q tensor");
    out = Tensor(q.shape());
    int64_t d0 = q.dim(0), d1 = q.dim(1);
    int64_t channels = (channel_dim == 0) ? d0 : d1;
    int64_t other = (channel_dim == 0) ? d1 : d0;
    const uint8_t* qd = q.data<uint8_t>();
    const float* sd = scales.data<float>();
    float* od = out.data<float>();
    for (int64_t c = 0; c < channels; ++c) {
        float mul = sd[c] / 57344.0f;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            od[idx] = fp8_e5m2_dequantize(qd[idx]) * mul;
        }
    }
}

float fp8_e5m2_quant_error(const Tensor& original, const Tensor& reconstructed) {
    return compute_quant_mse(original, reconstructed);
}

float fp8_e5m2_quant_snr(const Tensor& original, const Tensor& reconstructed) {
    return compute_quant_snr(original, reconstructed);
}

// ===========================================================================
// Roundtrip test helpers
// ===========================================================================

float compute_quant_mse(const Tensor& original, const Tensor& reconstructed) {
    int64_t n = original.numel();
    float mse = 0;
    const float* od = original.data<float>();
    const float* rd = reconstructed.data<float>();
    for (int64_t i = 0; i < n; ++i) {
        float diff = od[i] - rd[i];
        mse += diff * diff;
    }
    return mse / (float)n;
}

float compute_quant_snr(const Tensor& original, const Tensor& reconstructed) {
    int64_t n = original.numel();
    float signal = 0, noise = 0;
    const float* od = original.data<float>();
    const float* rd = reconstructed.data<float>();
    for (int64_t i = 0; i < n; ++i) {
        signal += od[i] * od[i];
        float diff = od[i] - rd[i];
        noise += diff * diff;
    }
    if (noise < 1e-30f) return 100.0f;
    return 10.0f * std::log10(signal / noise);
}

// ===========================================================================
// AVX2 SIMD-accelerated quantize/dequantize batch operations
// ===========================================================================
#ifdef QUANT_HAS_AVX2

static void dequant_tensor_fp8_avx2(const uint8_t* data, float* out,
                                     int64_t n, bool e5m2) {
    static float lut_e4m3[256];
    static float lut_e5m2[256];
    static std::once_flag lut_flag;
    std::call_once(lut_flag, []() {
        for (int i = 0; i < 256; ++i) {
            lut_e4m3[i] = fp8_e4m3_dequantize((uint8_t)i);
            lut_e5m2[i] = fp8_e5m2_dequantize((uint8_t)i);
        }
    });
    const float* lut = e5m2 ? lut_e5m2 : lut_e4m3;
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i idx8 = _mm_loadl_epi64((const __m128i*)(data + i));
        __m256i idx32 = _mm256_cvtepu8_epi32(idx8);
        __m256 val = _mm256_i32gather_ps(lut, idx32, 4);
        _mm256_storeu_ps(out + i, val);
    }
    for (; i < n; ++i)
        out[i] = lut[data[i]];
}

// AVX2 quant_gemm: FP8 E4M3 / E5M2 (256-entry LUT)
static void quant_gemm_fp8_avx2(const float* ad, float* cd,
                                 const uint8_t* b_q,
                                 int64_t M, int64_t N, int64_t K,
                                 bool e5m2) {
    static float lut_e4m3[256];
    static float lut_e5m2[256];
    static std::once_flag lut_flag;
    std::call_once(lut_flag, []() {
        for (int i = 0; i < 256; ++i) {
            lut_e4m3[i] = fp8_e4m3_dequantize((uint8_t)i);
            lut_e5m2[i] = fp8_e5m2_dequantize((uint8_t)i);
        }
    });
    const float* lut = e5m2 ? lut_e5m2 : lut_e4m3;
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = ad[m * K + k];
            if (a_val == 0.0f) continue;
            __m256 a_v = _mm256_set1_ps(a_val);
            int64_t n = 0;
            for (; n + 8 <= N; n += 8) {
                __m128i idx8 = _mm_loadl_epi64(
                    (const __m128i*)(b_q + k * N + n));
                __m256i idx32 = _mm256_cvtepu8_epi32(idx8);
                __m256 b_v = _mm256_i32gather_ps(lut, idx32, 4);
                __m256 c_v = _mm256_loadu_ps(cd + m * N + n);
                _mm256_storeu_ps(cd + m * N + n,
                                 _mm256_fmadd_ps(a_v, b_v, c_v));
            }
            for (; n < N; ++n)
                cd[m * N + n] += a_val * lut[b_q[k * N + n]];
        }
    }
}

#endif // QUANT_HAS_AVX2

// ===========================================================================
// QUANT16 Engine: FP16 storage (2 bytes per weight), no codebook
// ===========================================================================

Tensor QUANT16Engine::quantize(const Tensor& weight) const {
    int64_t n = weight.numel();
    Tensor out({n}, quant::DType::F16);
    math::vec_fp32_to_fp16(out.data<uint16_t>(), weight.data<float>(), (int)n);
    return out;
}

Tensor QUANT16Engine::dequantize(const Tensor& packed, int64_t n) const {
    Tensor out({n});
    math::vec_fp16_to_fp32(out.data<float>(), packed.data<uint16_t>(), (int)n);
    return out;
}

Tensor QUANT16Engine::quantize_batch(const Tensor& t) const {
    return quantize(t);
}

Tensor QUANT16Engine::dequantize_batch(const Tensor& q) const {
    return dequantize(q, q.numel());
}

Tensor QUANT16Engine::quant_gemm(const Tensor& a, const Tensor& b_packed,
                                int64_t M, int64_t N, int64_t K) const {
    Tensor C({M, N});
    C.zero_();
    const float* ad = a.data<float>();
    float* cd = C.data<float>();
    const uint16_t* bd = b_packed.data<uint16_t>();
    std::vector<float> b_deq((size_t)(K * N));
    math::vec_fp16_to_fp32(b_deq.data(), bd, (int)(K * N));
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = ad[m * K + k];
            if (a_val == 0.0f) continue;
            for (int64_t n = 0; n < N; ++n)
                cd[m * N + n] += a_val * b_deq[(size_t)(k * N + n)];
        }
    }
    return C;
}

void QUANT16Engine::quantize_per_channel(const Tensor& t, int channel_dim,
                                        Tensor& q, Tensor& scales) const {
    QUANT_CHECK(t.rank() == 2, "QUANT16 per-channel expects 2D tensor");
    int64_t d0 = t.dim(0), d1 = t.dim(1);
    int64_t channels = (channel_dim == 0) ? d0 : d1;
    int64_t other = (channel_dim == 0) ? d1 : d0;
    q = Tensor(t.shape(), quant::DType::F16);
    scales = Tensor({channels});
    const float* td = t.data<float>();
    uint16_t* qd = q.data<uint16_t>();
    float* sd = scales.data<float>();
    for (int64_t c = 0; c < channels; ++c) {
        float max_abs = 0;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            max_abs = std::max(max_abs, std::abs(td[idx]));
        }
        sd[c] = max_abs;
        if (max_abs < 1e-10f) max_abs = 1.0f;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            qd[idx] = CodebookQUANT4::float_to_half(td[idx] / max_abs);
        }
    }
}

void QUANT16Engine::dequantize_per_channel(const Tensor& q, const Tensor& scales,
                                          int channel_dim, Tensor& out) const {
    int64_t total = scales.numel();
    int64_t d0 = total, d1 = 1;
    if (channel_dim == 0) { d0 = total; d1 = q.numel() / total; }
    else { d1 = total; d0 = q.numel() / total; }
    out = Tensor({d0, d1});
    const uint16_t* qd = q.data<uint16_t>();
    const float* sd = scales.data<float>();
    float* od = out.data<float>();
    for (int64_t c = 0; c < total; ++c) {
        float scale = sd[c];
        int64_t other = out.numel() / total;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * total + c;
            od[idx] = CodebookQUANT4::half_to_float(qd[idx]) * scale;
        }
    }
}

float QUANT16Engine::quant_error(const Tensor& original, const Tensor& reconstructed) const {
    return compute_quant_mse(original, reconstructed);
}

float QUANT16Engine::quant_snr(const Tensor& original, const Tensor& reconstructed) const {
    return compute_quant_snr(original, reconstructed);
}

// ===========================================================================
// QUANT32 Engine: FP32 identity (lossless) — just copies data
// ===========================================================================

Tensor QUANT32Engine::quantize(const Tensor& weight) const {
    Tensor out(weight.shape(), quant::DType::F32);
    std::memcpy(out.data<float>(), weight.data<float>(), (size_t)weight.numel() * sizeof(float));
    return out;
}

Tensor QUANT32Engine::dequantize(const Tensor& packed, int64_t n) const {
    Tensor out({n});
    std::memcpy(out.data<float>(), packed.data<float>(), (size_t)n * sizeof(float));
    return out;
}

Tensor QUANT32Engine::quantize_batch(const Tensor& t) const {
    return quantize(t);
}

Tensor QUANT32Engine::dequantize_batch(const Tensor& q) const {
    return dequantize(q, q.numel());
}

Tensor QUANT32Engine::quant_gemm(const Tensor& a, const Tensor& b_packed,
                                int64_t M, int64_t N, int64_t K) const {
    Tensor C({M, N});
    C.zero_();
    const float* ad = a.data<float>();
    const float* bd = b_packed.data<float>();
    float* cd = C.data<float>();
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = ad[m * K + k];
            if (a_val == 0.0f) continue;
            for (int64_t n = 0; n < N; ++n)
                cd[m * N + n] += a_val * bd[k * N + n];
        }
    }
    return C;
}

void QUANT32Engine::quantize_per_channel(const Tensor& t, int channel_dim,
                                        Tensor& q, Tensor& scales) const {
    QUANT_CHECK(t.rank() == 2, "QUANT32 per-channel expects 2D tensor");
    int64_t d0 = t.dim(0), d1 = t.dim(1);
    int64_t channels = (channel_dim == 0) ? d0 : d1;
    q = Tensor(t.shape(), quant::DType::F32);
    scales = Tensor({channels});
    std::memcpy(q.data<float>(), t.data<float>(), (size_t)t.numel() * sizeof(float));
    std::fill(scales.data<float>(), scales.data<float>() + channels, 1.0f);
}

void QUANT32Engine::dequantize_per_channel(const Tensor& q, const Tensor& scales,
                                          int channel_dim, Tensor& out) const {
    (void)scales; (void)channel_dim;
    out = Tensor(q.shape(), quant::DType::F32);
    std::memcpy(out.data<float>(), q.data<float>(), (size_t)q.numel() * sizeof(float));
}

float QUANT32Engine::quant_error(const Tensor& original, const Tensor& reconstructed) const {
    return compute_quant_mse(original, reconstructed);
}

float QUANT32Engine::quant_snr(const Tensor& original, const Tensor& reconstructed) const {
    return compute_quant_snr(original, reconstructed);
}

} // namespace engines
} // namespace quant
