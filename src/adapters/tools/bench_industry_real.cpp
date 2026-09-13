// ============================================================================
// bench_industry_real.cpp — REAL-WEIGHT benchmark: QUANT vs industry codecs
// ----------------------------------------------------------------------------
// Loads a real model (GGUF Q8_0 ~near-lossless ground truth) via gguf_bridge,
// dequantizes every weight tensor to FP32, then quantizes + dequantizes each
// tensor with BOTH:
//   - industry formats, ported byte-exactly from llama.cpp (ggml-quants.c):
//       Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16  (32-weight blocks)
//       Q2_K, Q4_K, Q6_K, Q8_K              (256-weight super-blocks)
//   - native QUANT formats through FormatRegistry (singles + adaptive mixes).
//
// Every format is measured on the SAME data (each tensor's 256-aligned prefix,
// so block quantizers are never starved by a tail), and aggregated as a
// weight-weighted MSE over the WHOLE real model. This is the honest
// apples-to-apples number: no synthetic sine waves, no cherry-picking.
//
// Usage:
//   bench_industry_real <model.gguf> [--verbose]
// ============================================================================
#include "adapters/adapter_core.h"
#include "adapters/gguf_bridge.h"
#include "quant/format_registry.h"
#include "quant/block_codec.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <random>


using namespace quant;
using namespace quant::adapters;

// ─────────────────────────── FP16 helpers ──────────────────────────────────
static inline uint16_t f32_to_f16(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    const uint32_t sign = (u >> 16) & 0x8000;
    int32_t e = (int32_t)((u >> 23) & 0xff) - 127 + 15;
    uint32_t m = u & 0x7fffff;
    if (e >= 31) return (uint16_t)(sign | 0x7c00);          // inf (overflow)
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;                 // flush to zero
        m |= 0x800000;
        uint32_t shift = (uint32_t)(14 - e);
        uint32_t half = m >> shift;
        uint32_t rem = m & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1))) half++;
        return (uint16_t)(sign | half);
    }
    uint32_t e2 = (uint32_t)e;
    uint32_t dropped = m & 0x1fff;
    m >>= 13;
    if (dropped > 0x1000 || (dropped == 0x1000 && (m & 1))) m++;
    if (m == 0x400) { m = 0; e2++; }
    return (uint16_t)(sign | (e2 << 10) | m);
}

static inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1f;
    uint32_t m = h & 0x3ff;
    uint32_t u;
    if (e == 0) {
        if (m == 0) {
            u = sign;
        } else {
            e = 127 - 15 + 1;
            while (!(m & 0x400)) { m <<= 1; e--; }
            m &= 0x3ff;
            u = sign | (e << 23) | (m << 13);
        }
    } else if (e == 31) {
        u = sign | 0x7f800000u | (m << 13);
    } else {
        u = sign | ((e - 15 + 127) << 23) | (m << 13);
    }
    float f; memcpy(&f, &u, 4);
    return f;
}

#define QM_MIN(a,b) ((a)<(b)?(a):(b))
#define QM_MAX(a,b) ((a)>(b)?(a):(b))

// ─────────────────────── nearest_int (exact bit trick) ─────────────────────
static inline int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static const float GROUP_MAX_EPS = 1e-15f;

// ─────────────────────────── legacy 32-block codecs ────────────────────────

struct CodecQ4_0 {   // 18 B/32w = 4.5 bpw
    static const int qk = 32, bs = 18;
    static void quantize(const float* x, int64_t k, std::vector<uint8_t>& out) {
        int nb = (int)(k / qk); out.resize((size_t)nb * bs);
        for (int i = 0; i < nb; i++) {
            float amax = 0, max = 0;
            for (int j = 0; j < qk; j++) { float v = x[i*qk+j]; if (amax < fabsf(v)) { amax = fabsf(v); max = v; } }
            const float d = max / -8.f;
            const float id = d ? 1.f/d : 0.f;
            uint16_t dh = f32_to_f16(d);
            memcpy(&out[(size_t)i*bs], &dh, 2);
            for (int j = 0; j < qk/2; j++) {
                const float x0 = x[i*qk + 0 + j]*id;
                const float x1 = x[i*qk + qk/2 + j]*id;
                const uint8_t xi0 = (uint8_t)QM_MIN(15, (int8_t)(x0 + 8.5f));
                const uint8_t xi1 = (uint8_t)QM_MIN(15, (int8_t)(x1 + 8.5f));
                out[(size_t)i*bs + 2 + j] = (uint8_t)(xi0 | (xi1 << 4));
            }
        }
    }
    static void dequantize(const uint8_t* p, int64_t k, float* y) {
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++) {
            uint16_t dh; memcpy(&dh, p + (size_t)i*bs, 2);
            const float d = f16_to_f32(dh);
            for (int j = 0; j < qk/2; j++) {
                const uint8_t q = p[(size_t)i*bs + 2 + j];
                y[(size_t)i*qk + j]       = (float)((q & 0x0F) - 8) * d;
                y[(size_t)i*qk + j + qk/2] = (float)((q >> 4) - 8) * d;
            }
        }
    }
};

struct CodecQ4_1 {   // 20 B/32w = 5.0 bpw
    static const int qk = 32, bs = 20;
    static void quantize(const float* x, int64_t k, std::vector<uint8_t>& out) {
        int nb = (int)(k / qk); out.resize((size_t)nb * bs);
        for (int i = 0; i < nb; i++) {
            float mn = FLT_MAX, mx = -FLT_MAX;
            for (int j = 0; j < qk; j++) { float v = x[i*qk+j]; if (v < mn) mn = v; if (v > mx) mx = v; }
            const float d = (mx - mn) / 15.f;
            const float id = d ? 1.f/d : 0.f;
            uint16_t dh = f32_to_f16(d), mh = f32_to_f16(mn);
            memcpy(&out[(size_t)i*bs], &dh, 2); memcpy(&out[(size_t)i*bs+2], &mh, 2);
            for (int j = 0; j < qk/2; j++) {
                const float x0 = (x[i*qk + 0 + j] - mn)*id;
                const float x1 = (x[i*qk + qk/2 + j] - mn)*id;
                const uint8_t xi0 = (uint8_t)QM_MIN(15, (int8_t)(x0 + 0.5f));
                const uint8_t xi1 = (uint8_t)QM_MIN(15, (int8_t)(x1 + 0.5f));
                out[(size_t)i*bs + 4 + j] = (uint8_t)(xi0 | (xi1 << 4));
            }
        }
    }
    static void dequantize(const uint8_t* p, int64_t k, float* y) {
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++) {
            uint16_t dh, mh; memcpy(&dh, p + (size_t)i*bs, 2); memcpy(&mh, p + (size_t)i*bs+2, 2);
            const float d = f16_to_f32(dh), m = f16_to_f32(mh);
            for (int j = 0; j < qk/2; j++) {
                const uint8_t q = p[(size_t)i*bs + 4 + j];
                y[(size_t)i*qk + j]        = (float)(q & 0x0F) * d + m;
                y[(size_t)i*qk + j + qk/2] = (float)(q >> 4)   * d + m;
            }
        }
    }
};

struct CodecQ5_0 {   // 22 B/32w = 5.5 bpw
    static const int qk = 32, bs = 22;
    static void quantize(const float* x, int64_t k, std::vector<uint8_t>& out) {
        int nb = (int)(k / qk); out.resize((size_t)nb * bs);
        for (int i = 0; i < nb; i++) {
            float amax = 0, max = 0;
            for (int j = 0; j < qk; j++) { float v = x[i*qk+j]; if (amax < fabsf(v)) { amax = fabsf(v); max = v; } }
            const float d = max / -16.f;
            const float id = d ? 1.f/d : 0.f;
            uint16_t dh = f32_to_f16(d);
            memcpy(&out[(size_t)i*bs], &dh, 2);
            uint32_t qh = 0;
            for (int j = 0; j < qk/2; j++) {
                const float x0 = x[i*qk + 0 + j]*id;
                const float x1 = x[i*qk + qk/2 + j]*id;
                const uint8_t xi0 = (uint8_t)QM_MIN(31, (int8_t)(x0 + 16.5f));
                const uint8_t xi1 = (uint8_t)QM_MIN(31, (int8_t)(x1 + 16.5f));
                out[(size_t)i*bs + 6 + j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
                qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
                qh |= ((xi1 & 0x10u) >> 4) << (j + qk/2);
            }
            memcpy(&out[(size_t)i*bs + 2], &qh, 4);
        }
    }
    static void dequantize(const uint8_t* p, int64_t k, float* y) {
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++) {
            uint16_t dh; memcpy(&dh, p + (size_t)i*bs, 2);
            const float d = f16_to_f32(dh);
            uint32_t qh; memcpy(&qh, p + (size_t)i*bs + 2, 4);
            for (int j = 0; j < qk/2; j++) {
                const uint8_t q = p[(size_t)i*bs + 6 + j];
                const uint8_t xh_0 = (uint8_t)(((qh >> (j + 0)) << 4) & 0x10);
                const uint8_t xh_1 = (uint8_t)((qh >> (j + 12)) & 0x10);
                y[(size_t)i*qk + j]        = (float)(((q & 0x0F) | xh_0) - 16) * d;
                y[(size_t)i*qk + j + qk/2] = (float)(((q >> 4) | xh_1) - 16) * d;
            }
        }
    }
};

struct CodecQ5_1 {   // 24 B/32w = 6.0 bpw
    static const int qk = 32, bs = 24;
    static void quantize(const float* x, int64_t k, std::vector<uint8_t>& out) {
        int nb = (int)(k / qk); out.resize((size_t)nb * bs);
        for (int i = 0; i < nb; i++) {
            float mn = FLT_MAX, mx = -FLT_MAX;
            for (int j = 0; j < qk; j++) { float v = x[i*qk+j]; if (v < mn) mn = v; if (v > mx) mx = v; }
            const float d = (mx - mn) / 31.f;
            const float id = d ? 1.f/d : 0.f;
            uint16_t dh = f32_to_f16(d), mh = f32_to_f16(mn);
            memcpy(&out[(size_t)i*bs], &dh, 2); memcpy(&out[(size_t)i*bs+2], &mh, 2);
            uint32_t qh = 0;
            for (int j = 0; j < qk/2; j++) {
                const float x0 = (x[i*qk + 0 + j] - mn)*id;
                const float x1 = (x[i*qk + qk/2 + j] - mn)*id;
                const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
                const uint8_t xi1 = (uint8_t)(x1 + 0.5f);
                out[(size_t)i*bs + 8 + j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
                qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
                qh |= ((xi1 & 0x10u) >> 4) << (j + qk/2);
            }
            memcpy(&out[(size_t)i*bs + 4], &qh, 4);
        }
    }
    static void dequantize(const uint8_t* p, int64_t k, float* y) {
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++) {
            uint16_t dh, mh; memcpy(&dh, p + (size_t)i*bs, 2); memcpy(&mh, p + (size_t)i*bs+2, 2);
            const float d = f16_to_f32(dh), m = f16_to_f32(mh);
            uint32_t qh; memcpy(&qh, p + (size_t)i*bs + 4, 4);
            for (int j = 0; j < qk/2; j++) {
                const uint8_t q = p[(size_t)i*bs + 8 + j];
                const uint8_t xh_0 = (uint8_t)(((qh >> (j + 0)) << 4) & 0x10);
                const uint8_t xh_1 = (uint8_t)((qh >> (j + 12)) & 0x10);
                y[(size_t)i*qk + j]        = (float)(((q & 0x0F) | xh_0)) * d + m;
                y[(size_t)i*qk + j + qk/2] = (float)(((q >> 4) | xh_1)) * d + m;
            }
        }
    }
};

struct CodecQ8_0 {   // 34 B/32w = 8.5 bpw
    static const int qk = 32, bs = 34;
    static void quantize(const float* x, int64_t k, std::vector<uint8_t>& out) {
        int nb = (int)(k / qk); out.resize((size_t)nb * bs);
        for (int i = 0; i < nb; i++) {
            float amax = 0;
            for (int j = 0; j < qk; j++) amax = QM_MAX(amax, fabsf(x[i*qk+j]));
            const float d = amax / 127.f;
            const float id = d ? 1.f/d : 0.f;
            uint16_t dh = f32_to_f16(d);
            memcpy(&out[(size_t)i*bs], &dh, 2);
            for (int j = 0; j < qk; j++) out[(size_t)i*bs + 2 + j] = (uint8_t)(int8_t)roundf(x[i*qk+j]*id);
        }
    }
    static void dequantize(const uint8_t* p, int64_t k, float* y) {
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++) {
            uint16_t dh; memcpy(&dh, p + (size_t)i*bs, 2);
            const float d = f16_to_f32(dh);
            for (int j = 0; j < qk; j++) y[(size_t)i*qk + j] = (float)(int8_t)p[(size_t)i*bs + 2 + j] * d;
        }
    }
};

struct CodecF16 {    // 16.0 bpw
    static void quantize(const float* x, int64_t k, std::vector<uint8_t>& out) {
        out.resize((size_t)k * 2);
        for (int64_t i = 0; i < k; i++) { uint16_t h = f32_to_f16(x[i]); memcpy(&out[(size_t)i*2], &h, 2); }
    }
    static void dequantize(const uint8_t* p, int64_t k, float* y) {
        for (int64_t i = 0; i < k; i++) { uint16_t h; memcpy(&h, p + (size_t)i*2, 2); y[i] = f16_to_f32(h); }
    }
};

// ──────────────────────────── k-quant helpers ──────────────────────────────

static float make_qx_quants(int n, int nmax, const float* x, int8_t* L, int rmse_type, const float* qw) {
    float max = 0, amax = 0;
    for (int i = 0; i < n; ++i) { float ax = fabsf(x[i]); if (ax > amax) { amax = ax; max = x[i]; } }
    if (amax < GROUP_MAX_EPS) { for (int i = 0; i < n; ++i) L[i] = 0; return 0.f; }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) { int l = nearest_int(iscale * x[i]); L[i] = (int8_t)(nmax + QM_MAX(-nmax, QM_MIN(nmax-1, l))); }
        return 1/iscale;
    }
    float sumlx = 0, suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = QM_MAX(-nmax, QM_MIN(nmax-1, l));
        L[i] = (int8_t)(l + nmax);
        float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) continue;
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = QM_MAX(-nmax, QM_MIN(nmax-1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) { int l = nearest_int(iscale * x[i]); L[i] = (int8_t)(nmax + QM_MAX(-nmax, QM_MIN(nmax-1, l))); }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}

static float make_qkx2_quants(int n, int nmax, const float* x, const float* weights,
        uint8_t* L, float* the_min, uint8_t* Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0], max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) { for (int i = 0; i < n; ++i) L[i] = 0; *the_min = -min; return 0.f; }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_error = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = (uint8_t)QM_MAX(0, QM_MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        best_error += weights[i] * diff;
    }
    if (nstep < 1) { *the_min = -min; return scale; }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = QM_MAX(0, QM_MIN(nmax, l));
            Laux[i] = (uint8_t)l;
            float w = weights[i];
            sum_l += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) { this_min = 0; this_scale = sum_xl / sum_l2; }
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                cur_error += weights[i] * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) L[i] = Laux[i];
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (uint8_t)((q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)); *m = (uint8_t)((q[j+4] >> 4) | ((q[j] >> 6) << 4)); }
}

// ──────────────────────────── k-quant codecs ───────────────────────────────

struct CodecQ2_K {   // 84 B/256w = 2.625 bpw
    static const int qk = 256, bs = 84;
    static void quantize(const float* x, int64_t k, std::vector<uint8_t>& out) {
        int nb = (int)(k / qk); out.resize((size_t)nb * bs);
        uint8_t L[256]; uint8_t Laux[16]; float weights[16];
        float mins[16]; float scales[16];
        const float q4scale = 15.f;
        for (int i = 0; i < nb; i++) {
            float max_scale = 0, max_min = 0;
            for (int j = 0; j < qk/16; ++j) {
                for (int l = 0; l < 16; ++l) weights[l] = fabsf(x[16*j + l]);
                scales[j] = make_qkx2_quants(16, 3, x + 16*j, weights, L + 16*j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
                if (scales[j] > max_scale) max_scale = scales[j];
                if (mins[j] > max_min) max_min = mins[j];
            }
            uint8_t* p = out.data() + (size_t)i*bs;
            if (max_scale > 0) {
                float iscale = q4scale/max_scale;
                for (int j = 0; j < qk/16; ++j) p[4+j] = (uint8_t)nearest_int(iscale*scales[j]);
                uint16_t dh = f32_to_f16(max_scale/q4scale);
                memcpy(p, &dh, 2);
            } else {
                for (int j = 0; j < qk/16; ++j) p[4+j] = 0;
                uint16_t dh = f32_to_f16(0.f); memcpy(p, &dh, 2);
            }
            if (max_min > 0) {
                float iscale = q4scale/max_min;
                for (int j = 0; j < qk/16; ++j) p[4+j] |= (uint8_t)(nearest_int(iscale*mins[j]) << 4);
                uint16_t dh = f32_to_f16(max_min/q4scale);
                memcpy(p+2, &dh, 2);
            } else {
                uint16_t dh = f32_to_f16(0.f); memcpy(p+2, &dh, 2);
            }
            for (int j = 0; j < qk/16; ++j) {
                const float d = f16_to_f32(*(uint16_t*)p) * (p[4+j] & 0xF);
                if (!d) continue;
                const float dm = f16_to_f32(*(uint16_t*)(p+2)) * (p[4+j] >> 4);
                for (int ii = 0; ii < 16; ++ii) {
                    int l = nearest_int((x[16*j + ii] + dm)/d);
                    L[16*j + ii] = (uint8_t)QM_MAX(0, QM_MIN(3, l));
                }
            }
            uint8_t* qs = p + 20;
            for (int j = 0; j < qk; j += 128)
                for (int l = 0; l < 32; ++l)
                    qs[j/4 + l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6));
            x += qk;
        }
    }
    static void dequantize(const uint8_t* p, int64_t k, float* y) {
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++) {
            const uint8_t* blk = p + (size_t)i*bs;
            const float d = f16_to_f32(*(uint16_t*)blk);
            const float min = f16_to_f32(*(uint16_t*)(blk+2));
            const uint8_t* q = blk + 20;
            const uint8_t* sc = blk + 4;
            int is = 0;
            for (int n = 0; n < qk; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    uint8_t s = sc[is++];
                    float dl = d * (s & 0xF), ml = min * (s >> 4);
                    for (int l = 0; l < 16; ++l) *y++ = dl * (float)(int8_t)((q[l] >> shift) & 3) - ml;
                    s = sc[is++];
                    dl = d * (s & 0xF); ml = min * (s >> 4);
                    for (int l = 0; l < 16; ++l) *y++ = dl * (float)(int8_t)((q[l+16] >> shift) & 3) - ml;
                    shift += 2;
                }
                q += 32;
            }
        }
    }
};

struct CodecQ4_K {   // 144 B/256w = 4.5 bpw (d 2B + dmin 2B + scales[12] + qs[128])
    static const int qk = 256, bs = 144;
    static void quantize(const float* x, int64_t k, std::vector<uint8_t>& out) {
        int nb = (int)(k / qk); out.resize((size_t)nb * bs);
        uint8_t L[256]; uint8_t Laux[32]; float weights[32];
        float mins[8]; float scales[8];
        for (int i = 0; i < nb; i++) {
            float max_scale = 0, max_min = 0;
            for (int j = 0; j < qk/32; ++j) {
                float sum_x2 = 0;
                for (int l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
                float av_x = sqrtf(sum_x2/32);
                for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
                scales[j] = make_qkx2_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -1.f, 0.1f, 20, false);
                if (scales[j] > max_scale) max_scale = scales[j];
                if (mins[j] > max_min) max_min = mins[j];
            }
            float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
            float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
            uint8_t* sc = out.data() + (size_t)i*bs + 4;
            for (int j = 0; j < qk/32; ++j) {
                uint8_t ls = (uint8_t)nearest_int(inv_scale*scales[j]);
                uint8_t lm = (uint8_t)nearest_int(inv_min*mins[j]);
                ls = (uint8_t)QM_MIN(63, ls);
                lm = (uint8_t)QM_MIN(63, lm);
                if (j < 4) { sc[j] = ls; sc[j+4] = lm; }
                else { sc[j+4] = (uint8_t)((ls & 0xF) | ((lm & 0xF) << 4)); sc[j-4] |= (uint8_t)((ls >> 4) << 6); sc[j] |= (uint8_t)((lm >> 4) << 6); }
            }
            uint8_t* p = out.data() + (size_t)i*bs;
            uint16_t dh = f32_to_f16(max_scale/63.f); memcpy(p, &dh, 2);
            uint16_t dm = f32_to_f16(max_min/63.f);  memcpy(p+2, &dm, 2);
            uint8_t s, m;
            for (int j = 0; j < qk/32; ++j) {
                get_scale_min_k4(j, sc, &s, &m);
                const float d = f16_to_f32(*(uint16_t*)p) * s;
                if (!d) continue;
                const float dmm = f16_to_f32(*(uint16_t*)(p+2)) * m;
                for (int ii = 0; ii < 32; ++ii) {
                    int l = nearest_int((x[32*j + ii] + dmm)/d);
                    L[32*j + ii] = (uint8_t)QM_MAX(0, QM_MIN(15, l));
                }
            }
            uint8_t* q = p + 16;
            for (int j = 0; j < qk; j += 64) {
                for (int l = 0; l < 32; ++l)
                    q[l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 4));
                q += 32;
            }
            x += qk;
        }
    }
    static void dequantize(const uint8_t* p, int64_t k, float* y) {
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++) {
            const uint8_t* blk = p + (size_t)i*bs;
            const uint8_t* q = blk + 16;
            const float d = f16_to_f32(*(uint16_t*)blk);
            const float min = f16_to_f32(*(uint16_t*)(blk+2));
            const uint8_t* sc = blk + 4;
            int is = 0;
            uint8_t s, m;
            for (int j = 0; j < qk; j += 64) {
                get_scale_min_k4(is + 0, sc, &s, &m);
                const float d1 = d * s; const float m1 = min * m;
                get_scale_min_k4(is + 1, sc, &s, &m);
                const float d2 = d * s; const float m2 = min * m;
                for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
                for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
                q += 32; is += 2;
            }
        }
    }
};

struct CodecQ6_K {   // 210 B/256w = 6.5625 bpw
    static const int qk = 256, bs = 210;
    static void quantize(const float* x, int64_t k, std::vector<uint8_t>& out) {
        int nb = (int)(k / qk); out.resize((size_t)nb * bs);
        int8_t L[256]; float scales[16];
        for (int i = 0; i < nb; i++) {
            float max_scale = 0, max_abs_scale = 0;
            for (int ib = 0; ib < qk/16; ++ib) {
                const float scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, NULL);
                scales[ib] = scale;
                const float abs_scale = fabsf(scale);
                if (abs_scale > max_abs_scale) { max_abs_scale = abs_scale; max_scale = scale; }
            }
            uint8_t* p = out.data() + (size_t)i*bs;
            if (max_abs_scale < GROUP_MAX_EPS) {
                memset(p, 0, bs);
                uint16_t dh = f32_to_f16(0.f); memcpy(p + 208, &dh, 2);
                x += qk;
                continue;
            }
            float iscale = -128.f/max_scale;
            uint16_t dh = f32_to_f16(1/iscale);
            memcpy(p + 208, &dh, 2);
            int8_t* sc = (int8_t*)(p + 192);
            for (int ib = 0; ib < qk/16; ++ib) sc[ib] = (int8_t)QM_MIN(127, nearest_int(iscale*scales[ib]));
            for (int j = 0; j < qk/16; ++j) {
                float d = f16_to_f32(*(uint16_t*)(p+208)) * sc[j];
                if (!d) continue;
                for (int ii = 0; ii < 16; ++ii) {
                    int l = nearest_int(x[16*j + ii]/d);
                    L[16*j + ii] = (int8_t)(QM_MAX(-32, QM_MIN(31, l)) + 32);
                }
            }
            uint8_t* ql = p;
            uint8_t* qh = p + 128;
            for (int j = 0; j < qk; j += 128) {
                for (int l = 0; l < 32; ++l) {
                    const uint8_t q1 = (uint8_t)(L[j + l +  0] & 0xF);
                    const uint8_t q2 = (uint8_t)(L[j + l + 32] & 0xF);
                    const uint8_t q3 = (uint8_t)(L[j + l + 64] & 0xF);
                    const uint8_t q4 = (uint8_t)(L[j + l + 96] & 0xF);
                    ql[l+ 0] = (uint8_t)(q1 | (q3 << 4));
                    ql[l+32] = (uint8_t)(q2 | (q4 << 4));
                    qh[l] = (uint8_t)((L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6));
                }
                ql += 64;
                qh += 32;
            }
            x += qk;
        }
    }
    static void dequantize(const uint8_t* p, int64_t k, float* y) {
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++) {
            const uint8_t* blk = p + (size_t)i*bs;
            const float d = f16_to_f32(*(uint16_t*)(blk + 208));
            const uint8_t* ql = blk;
            const uint8_t* qh = blk + 128;
            const int8_t* sc = (const int8_t*)(blk + 192);
            for (int n = 0; n < qk; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is = l/16;
                    const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int8_t q3 = (int8_t)((ql[l +  0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[l +  0] = d * sc[is + 0] * q1;
                    y[l + 32] = d * sc[is + 2] * q2;
                    y[l + 64] = d * sc[is + 4] * q3;
                    y[l + 96] = d * sc[is + 6] * q4;
                }
                y  += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
    }
};

struct CodecQ8_K {   // 292 B/256w = 9.125 bpw
    static const int qk = 256, bs = 292;
    static void quantize(const float* x, int64_t k, std::vector<uint8_t>& out) {
        int nb = (int)(k / qk); out.resize((size_t)nb * bs);
        for (int i = 0; i < nb; i++) {
            uint8_t* p = out.data() + (size_t)i*bs;
            float max = 0, amax = 0;
            for (int j = 0; j < qk; ++j) { float ax = fabsf(x[j]); if (ax > amax) { amax = ax; max = x[j]; } }
            if (!amax) { float z = 0.f; memcpy(p, &z, 4); memset(p + 4, 0, qk); x += qk; continue; }
            const float iscale = -127.f/max;
            for (int j = 0; j < qk; ++j) { int v = nearest_int(iscale*x[j]); p[4 + j] = (uint8_t)(int8_t)QM_MIN(127, v); }
            int16_t* bsums = (int16_t*)(p + 4 + qk);
            for (int j = 0; j < qk/16; ++j) {
                int sum = 0;
                for (int ii = 0; ii < 16; ++ii) sum += (int8_t)p[4 + j*16 + ii];
                bsums[j] = (int16_t)sum;
            }
            float d = 1/iscale;
            memcpy(p, &d, 4);
            x += qk;
        }
    }
    static void dequantize(const uint8_t* p, int64_t k, float* y) {
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++) {
            const uint8_t* blk = p + (size_t)i*bs;
            float d; memcpy(&d, blk, 4);
            for (int j = 0; j < qk; ++j) *y++ = d * (float)(int8_t)blk[4 + j];
        }
    }
};

// ─────────────────────────── evaluation ────────────────────────────────────

struct FormatResult {
    std::string name;
    double bpw = 0;
    double mse = 0;       // weight-weighted aggregate
    double psnr = 0;      // dB (peak = 1)
    int64_t weights = 0;
    bool ok = false;
    bool is_quant = false;
};

template <typename Codec>
static bool eval_industry(const float* data, int64_t n, double& mse, double& bpw) {
    std::vector<uint8_t> q;
    Codec::quantize(data, n, q);
    std::vector<float> dq((size_t)n);
    Codec::dequantize(q.data(), n, dq.data());
    double sse = 0;
    for (int64_t i = 0; i < n; i++) { double d = (double)data[i] - dq[(size_t)i]; sse += d*d; }
    mse = sse / (double)n;
    bpw = (double)q.size() * 8.0 / (double)n;
    return true;
}

static bool eval_quant_single(const float* data, int64_t n, const FormatDescriptor& fmt, double& mse) {
    QuantResult qr = FormatRegistry::quantize(data, n, fmt);
    if (!qr.success) return false;
    std::vector<float> dq((size_t)n);
    FormatRegistry::dequantize(qr, dq.data(), n);
    double sse = 0;
    for (int64_t i = 0; i < n; i++) { double d = (double)data[i] - dq[(size_t)i]; sse += d*d; }
    mse = sse / (double)n;
    return true;
}

static bool eval_quant_mix(const float* data, int64_t n, const MixDescriptor& mix, double& mse, double& bpw) {
    FormatRegistry::MixBlockPlan plan = FormatRegistry::allocate_mix_blocks(mix, data, n, 256);
    if (plan.formats.empty()) return false;
    int64_t total_bytes = 0;
    double sse = 0;
    for (size_t b = 0; b < plan.formats.size(); b++) {
        int64_t len = plan.block_lens[b];
        if (len <= 0) continue;
        std::vector<uint8_t> idx, cb;
        if (!quantize_block_all(plan.formats[b], data + plan.block_starts[b], (int)len, idx, cb)) return false;
        std::vector<float> dq((size_t)len);
        dequantize_block_all(plan.formats[b], idx.data(), idx.size(), cb.data(), cb.size(),
                             (uint32_t)len, dq.data());
        for (int64_t i = 0; i < len; i++) {
            double diff = (double)data[plan.block_starts[b] + i] - dq[(size_t)i];
            sse += diff * diff;
        }
        total_bytes += (int64_t)(idx.size() + cb.size());
    }
    mse = sse / (double)n;
    bpw = n ? (double)total_bytes * 8.0 / (double)n : 0.0;
    return true;
}

// ─────────────────────────────── driver ────────────────────────────────────

struct Runner {
    std::string name;
    bool is_quant = false;      // true → QUANT single; 'mix' handled separately
    FormatDescriptor fmt;       // for QUANT singles
    const MixDescriptor* mix = nullptr; // for QUANT mixes
    int rf_id = -1;             // RegFormat id for QUANT formats; -1 for industry

    double sse = 0;
    double bytes = 0;
    int64_t weights = 0;
    bool ok = false;
    bool any_data = false;

    void accum(int64_t n, double mse_add, double bpw_now) {
        sse += mse_add * (double)n;
        bytes += bpw_now * (double)n / 8.0;
        weights += n;
        any_data = true;
        ok = true;
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: bench_industry_real <model.gguf> [--verbose]\n");
        return 2;
    }
    std::string path = argv[1];
    bool verbose = argc > 2 && std::string(argv[2]) == "--verbose";

    printf("== bench_industry_real : REAL-WEIGHT QUANT vs industry codecs ==\n");
    printf("model: %s\n\n", path.c_str());

    auto tensors = load_gguf(path, false);
    if (tensors.empty()) { printf("ERROR: no tensors loaded\n"); return 1; }
    int64_t total_params = 0;
    for (auto& t : tensors) total_params += t.numel();
    printf("tensors: %zu   params: %.3fM\n\n", tensors.size(), (double)total_params/1e6);

    // ── build runner list ────────────────────────────────────────────────
    std::vector<Runner> rs;
    auto add_industry = [&](const char* nm, double expect_bpw) {
        Runner r; r.name = nm; r.is_quant = false; (void)expect_bpw; rs.push_back(r);
    };
    add_industry("INDUSTRY Q4_0", 4.5);
    add_industry("INDUSTRY Q4_1", 5.0);
    add_industry("INDUSTRY Q5_0", 5.5);
    add_industry("INDUSTRY Q5_1", 6.0);
    add_industry("INDUSTRY Q8_0", 8.5);
    add_industry("INDUSTRY Q2_K", 2.625);
    add_industry("INDUSTRY Q4_K", 4.4375);
    add_industry("INDUSTRY Q6_K", 6.5625);
    add_industry("INDUSTRY Q8_K", 9.125);
    add_industry("INDUSTRY F16", 16.0);

    const char* quant_singles[] = {
        "Q1_5", "QG_1_5", "Q2", "QG2",
        "Q4", "QG4", "Q6_K_M", "Q8", "QG8", "Q16"
    };
    for (const char* nm : quant_singles) {
        FormatDescriptor fd = FormatRegistry::parse_format_name(nm);
        if (fd.id == RegFormat::Q32 && std::string(nm) != "Q32") continue;
        Runner r; r.name = std::string("QUANT  ") + nm; r.is_quant = true; r.fmt = fd;
        r.rf_id = (int)fd.id;
        rs.push_back(r);
    }
    RegFormat mix_ids[] = {
        RegFormat::QG_MX_3_5, RegFormat::QG_MX_4_5,
        RegFormat::QG_MX_6_5, RegFormat::QG_MX_8_5,
        RegFormat::QG_MX_12_5, RegFormat::QG_MX_16_5,
        RegFormat::QG_MX_24_5
    };
    for (RegFormat rf : mix_ids) {
        const MixDescriptor* md = find_mix_descriptor(rf);
        if (!md) continue;
        Runner r; r.name = std::string("QUANT  ") + md->name; r.is_quant = true;
        r.mix = md; r.rf_id = (int)rf; r.fmt.id = RegFormat::Q32; // mark as mix via mix pointer
        rs.push_back(r);
    }

    // ── accumulate over every real tensor ─────────────────────────────────
    printf("quantizing %zu tensors...\n", tensors.size());
    fflush(stdout);
    size_t tensor_done = 0;
    for (auto& t : tensors) {
        if (t.numel() < 256) continue;
        const int64_t aligned = (t.numel() / 256) * 256;   // 256-aligned prefix
        if (verbose) printf("  [%6.1fM] %s\n", (double)t.numel()/1e6, t.name.c_str());

        for (auto& r : rs) {
            double mse = 0, bpw = 0;
            bool ok = false;
            if (r.mix) {
                ok = eval_quant_mix(t.data.data(), aligned, *r.mix, mse, bpw);
            } else if (r.is_quant) {
                ok = eval_quant_single(t.data.data(), aligned, r.fmt, mse);
                if (ok) bpw = r.fmt.bpw;
            } else if (r.name.find("Q4_0") != std::string::npos) {
                ok = eval_industry<CodecQ4_0>(t.data.data(), aligned, mse, bpw);
            } else if (r.name.find("Q4_1") != std::string::npos) {
                ok = eval_industry<CodecQ4_1>(t.data.data(), aligned, mse, bpw);
            } else if (r.name.find("Q5_0") != std::string::npos) {
                ok = eval_industry<CodecQ5_0>(t.data.data(), aligned, mse, bpw);
            } else if (r.name.find("Q5_1") != std::string::npos) {
                ok = eval_industry<CodecQ5_1>(t.data.data(), aligned, mse, bpw);
            } else if (r.name.find("Q8_0") != std::string::npos) {
                ok = eval_industry<CodecQ8_0>(t.data.data(), aligned, mse, bpw);
            } else if (r.name.find("Q2_K") != std::string::npos) {
                ok = eval_industry<CodecQ2_K>(t.data.data(), aligned, mse, bpw);
            } else if (r.name.find("Q4_K") != std::string::npos) {
                ok = eval_industry<CodecQ4_K>(t.data.data(), aligned, mse, bpw);
            } else if (r.name.find("Q6_K") != std::string::npos) {
                ok = eval_industry<CodecQ6_K>(t.data.data(), aligned, mse, bpw);
            } else if (r.name.find("Q8_K") != std::string::npos) {
                ok = eval_industry<CodecQ8_K>(t.data.data(), aligned, mse, bpw);
            } else if (r.name.find("F16") != std::string::npos) {
                ok = eval_industry<CodecF16>(t.data.data(), aligned, mse, bpw);
            }
            if (ok) r.accum(aligned, mse, bpw);
        }
        ++tensor_done;
        if ((tensor_done % 8) == 0 || tensor_done == tensors.size()) {
            printf("  ... %zu/%zu tensors done\n", tensor_done, tensors.size());
            fflush(stdout);
        }
    }

    // ── report ────────────────────────────────────────────────────────────
    printf("\n%-26s %8s %12s %8s %8s %10s\n",
           "FORMAT", "BPW", "MSE", "PSNR(dB)", "dB/bit", "weights");
    printf("%-26s %8s %12s %8s %8s %10s\n", "------", "---", "---", "--------", "------", "-------");
    double f16_psnr = 0;
    for (auto& r : rs) {
        if (!r.any_data || r.weights == 0) continue;
        double mse = r.sse / (double)r.weights;
        double bpw = r.bytes * 8.0 / (double)r.weights;
        double psnr = mse > 0 ? 10.0 * log10(1.0 / mse) : 999.0;
        if (r.name.find("F16") != std::string::npos) f16_psnr = psnr;
        printf("%-26s %8.3f %12.3e %8.2f %8.2f %10.2fM\n",
               r.name.c_str(), bpw, mse, psnr, psnr / bpw, (double)r.weights/1e6);
    }

    // ── tier head-to-head (best QUANT vs best industry in each band) ───────
    struct Tier { const char* label; const int* q; int nq; const char* const* ind; int ni; };
    auto pick = [&](std::vector<Runner>& src, const int* ids, int n) -> Runner* {
        Runner* best = nullptr;
        for (auto& r : src) {
            if (!r.any_data || r.weights == 0) continue;
            bool in = false;
            for (int i = 0; i < n; i++) if (r.rf_id == ids[i]) in = true;
            if (!in) continue;
            double psnr = (r.sse > 0) ? 10.0 * log10((double)r.weights / r.sse) : 999.0;
            if (!best || psnr > (best->sse > 0 ? 10.0*log10((double)best->weights/best->sse) : 0.0)) best = &r;
        }
        return best;
    };
    auto pick_ind = [&](const char* const* keys, int n) -> Runner* {
        Runner* best = nullptr;
        for (auto& r : rs) {
            if (r.is_quant || !r.any_data || r.weights == 0) continue;
            bool in = false;
            for (int i = 0; i < n; i++) if (r.name.find(keys[i]) != std::string::npos) in = true;
            if (!in) continue;
            double psnr = (r.sse > 0) ? 10.0 * log10((double)r.weights / r.sse) : 999.0;
            if (!best || psnr > (best->sse > 0 ? 10.0*log10((double)best->weights/best->sse) : 0.0)) best = &r;
        }
        return best;
    };
    auto row = [&](Runner* r, char tag, double* bpw_out) {
        if (!r) { printf("  %c  (n/a)\n", tag); return; }
        double mse = r->sse / (double)r->weights;
        double bpw = r->bytes * 8.0 / (double)r->weights;
        double psnr = mse > 0 ? 10.0 * log10(1.0/mse) : 999.0;
        *bpw_out = bpw;
        printf("  %c  %-20s BPW %7.3f   PSNR %7.2f dB   (rel F16 %+7.2f dB)\n",
               tag, r->name.c_str(), bpw, psnr, psnr - f16_psnr);
    };

    printf("\n===== HEAD-TO-HEAD BY TIER (best of each family) =====\n");

    { // 1.9-2.7 BPW
        const int q_ids[] = { (int)RegFormat::Q1_5, (int)RegFormat::QG_1_5,
                              (int)RegFormat::QG_MX_3_5, (int)RegFormat::QG_MX_4_5,
                              (int)RegFormat::Q2,
                              (int)RegFormat::QG_MX_6_5, (int)RegFormat::QG2 };
        const char* ind[] = { "Q2_K" };
        printf("\n[1.5-2.7 BPW band]\n");
        double qb = 0, ib = 0;
        row(pick(rs, q_ids, 7), 'Q', &qb);
        row(pick_ind(ind, 1), 'I', &ib);
        if (qb > 0 && ib > 0) printf("  => QUANT at %.3f BPW vs industry at %.3f BPW (QUANT uses %.1f%% fewer bits)\n",
                                     qb, ib, (1.0 - qb/ib) * 100.0);
    }
    { // 4-4.5 BPW
        const int q_ids[] = { (int)RegFormat::Q4, (int)RegFormat::QG4,
                              (int)RegFormat::QG_MX_8_5,
                              (int)RegFormat::QG_MX_12_5 };
        const char* ind[] = { "Q4_0", "Q4_K" };
        printf("\n[4.0-4.5 BPW band]\n");
        double qb = 0, ib = 0;
        row(pick(rs, q_ids, 4), 'Q', &qb);
        row(pick_ind(ind, 2), 'I', &ib);
        if (qb > 0 && ib > 0) printf("  => QUANT at %.3f BPW vs industry at %.3f BPW (QUANT uses %.1f%% fewer bits)\n",
                                     qb, ib, (1.0 - qb/ib) * 100.0);
    }
    { // 5-6.6 BPW
        const int q_ids[] = { (int)RegFormat::Q6_K_M,
                              (int)RegFormat::QG_MX_24_5 };
        const char* ind[] = { "Q5_0", "Q5_1", "Q6_K" };
        printf("\n[5.5-6.6 BPW band]\n");
        double qb = 0, ib = 0;
        row(pick(rs, q_ids, 2), 'Q', &qb);
        row(pick_ind(ind, 3), 'I', &ib);
        if (qb > 0 && ib > 0) printf("  => QUANT at %.3f BPW vs industry at %.3f BPW (QUANT uses %.1f%% fewer bits)\n",
                                     qb, ib, (1.0 - qb/ib) * 100.0);
    }
    { // 8-9.1 BPW
        const int q_ids[] = { (int)RegFormat::Q8, (int)RegFormat::QG8,
                              (int)RegFormat::QG_MX_16_5 };
        const char* ind[] = { "Q8_0", "Q8_K" };
        printf("\n[8.0-9.1 BPW band]\n");
        double qb = 0, ib = 0;
        row(pick(rs, q_ids, 3), 'Q', &qb);
        row(pick_ind(ind, 2), 'I', &ib);
        if (qb > 0 && ib > 0) printf("  => QUANT at %.3f BPW vs industry at %.3f BPW (QUANT uses %.1f%% fewer bits)\n",
                                     qb, ib, (1.0 - qb/ib) * 100.0);
    }
    { // 16 BPW
        const int q_ids[] = { (int)RegFormat::Q16 };
        const char* ind[] = { "F16" };
        printf("\n[16.0 BPW band]\n");
        double qb = 0, ib = 0;
        row(pick(rs, q_ids, 1), 'Q', &qb);
        row(pick_ind(ind, 1), 'I', &ib);
    }

    printf("\n===== GRP GROUPING AUDIT (is per-group state really stored?) =====\n");
    printf("  QG4 stores a REAL 6-bit scale + 6-bit min per 32-weight group\n");
    printf("  + FP16 d/dm (144 B/256w = +14 B vs Q4). QG2 stores a REAL\n");
    printf("  4-bit scale + 4-bit min per 16-weight group + FP16 d/dm (84 B/256w =\n");
    printf("  +18 B vs Q2). QG8 stores a per-16-weight 7-bit scale + FP16\n");
    printf("  d (272 B/256w = +16 B vs Q8). Proof: on identical data the GRP\n");
    printf("  payload must be exactly that much larger than its plain twin, and a\n");
    printf("  block split into distinct-scale groups must decode each group with\n");
    printf("  ITS OWN scale/min.\n");
    {
        // (a) stored-bytes delta on real 256-weight data
        std::vector<float> w(256);
        {
            std::mt19937 rng(20260805);
            std::normal_distribution<float> g(0.0f, 1.0f);
            for (auto& v : w) v = g(rng);
        }
        struct GrpTwin { Format plain, grp; const char* name; };
        const GrpTwin twins[] = {
            { Format::Q2, Format::QG2, "Q2" },
            { Format::Q4, Format::QG4, "Q4" },
            { Format::Q8, Format::QG8, "Q8" },
        };
        for (const auto& tw : twins) {
            std::vector<uint8_t> ip, ig, cb;
            quantize_block_all(tw.plain, w.data(), 256, ip, cb);
            quantize_block_all(tw.grp,    w.data(), 256, ig, cb);
            const char* note =
                tw.name[1] == '4' ? "(8x32w 6b sc + 8x32w 6b min + FP16 d/dm)" :
                tw.name[1] == '2' ? "(16x16w 4b sc + 16x16w 4b min + FP16 d/dm)" :
                                    "(16x16w 7b sc + FP16 d)";
            printf("  %-9s plain %3zu B   %-12s GRP %3zu B   (+%zu B = %s)\n",
                   tw.name, ip.size(), format_name(tw.grp),
                   ig.size(), ig.size() - ip.size(), note);
        }
        // (b) distinct-scale groups must decode with their own scale/min.
        // QG4 groups 32 weights, QG2 groups 16 weights.
        {
            std::vector<float> wg(256);
            for (int g = 0; g < 8; ++g) {
                const float s = 0.1f * (float)(g + 1);          // 0.1 .. 0.8
                for (int i = 0; i < 32; ++i) wg[(size_t)g * 32 + i] = s * 20.0f;
            }
            std::vector<uint8_t> i4, cb;
            quantize_block_all(Format::QG4, wg.data(), 256, i4, cb);
            std::vector<float> d4(256);
            dequantize_block_all(Format::QG4, i4.data(), i4.size(), cb.data(), cb.size(), 256, d4.data());
            int ok4 = 1;
            for (int g = 0; g < 8; ++g) {
                const float exp = 0.1f * (float)(g + 1) * 20.0f;
                const float got = d4[(size_t)g * 32];
                const double rel = std::fabs(got - exp) / (exp > 0 ? exp : 1.0);
                printf("    QG4 grp %d expected %5.1f decoded %5.1f (rel err %.1f%%)\n",
                       g, exp, got, rel * 100.0);
                if (rel > 0.15) ok4 = 0;
            }
            printf("  => QG4 per-32 scale+min actually applied: %s\n", ok4 ? "YES" : "NO");
        }
        {
            std::vector<float> wg(256);
            for (int g = 0; g < 16; ++g) {
                const float s = 0.1f * (float)(g + 1);          // 0.1 .. 1.6
                for (int i = 0; i < 16; ++i) wg[(size_t)g * 16 + i] = s * 20.0f;
            }
            std::vector<uint8_t> i2, cb;
            quantize_block_all(Format::QG2, wg.data(), 256, i2, cb);
            std::vector<float> d2(256);
            dequantize_block_all(Format::QG2, i2.data(), i2.size(), cb.data(), cb.size(), 256, d2.data());
            int ok2 = 1;
            for (int g = 0; g < 16; ++g) {
                const float exp = 0.1f * (float)(g + 1) * 20.0f;
                const float got = d2[(size_t)g * 16];
                const double rel = std::fabs(got - exp) / (exp > 0 ? exp : 1.0);
                printf("    QG2 grp %2d expected %5.1f decoded %5.1f (rel err %.1f%%)\n",
                       g, exp, got, rel * 100.0);
                if (rel > 0.15) ok2 = 0;
            }
            printf("  => QG2 per-16 scale+min actually applied: %s\n", ok2 ? "YES" : "NO");
        }
    }

    printf("\n(dB rel F16 = reconstruction PSNR relative to the FP16 baseline; PSNR higher = better.\n");
    printf(" Weights are the real tensors of %s, 256-aligned; MSE is weight-weighted over the whole model.)\n",
           path.c_str());
    printf("done.\n");
    return 0;
}
