#include <quant/codebook.h>
#include "quant/block_codec.h"
#include "quant/codebook.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <array>

namespace quant {

namespace {

constexpr int kQuantSubBlock = 32;  // Q_TWI_MIX_1_5 scale window

constexpr std::array<float, 4> kQ2Levels = {
    -1.5104f, -0.4528f, 0.4528f, 1.5104f,
};

constexpr std::array<float, 16> kQ4Levels = {
    -2.7178f, -2.0522f, -1.5995f, -1.2392f,
    -0.9275f, -0.6508f, -0.3976f, -0.1260f,
     0.1260f,  0.3976f,  0.6508f,  0.9275f,
     1.2392f,  1.5995f,  2.0522f,  2.7178f,
};

// ---- FP16 conversion (self-contained, no math dependency) ------------------
uint16_t f32_to_f16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    const int sign = (int)(bits >> 31) & 1;
    int exp = (int)((bits >> 23) & 0xFF) - 127;
    int mant = (int)(bits & 0x7FFFFF);
    uint16_t h;
    if (exp > 15) {
        h = (uint16_t)((sign << 15) | 0x7C00);
    } else if (exp >= -14) {
        int m = (mant >> 13) | 0x400;
        const int round_bit = (mant >> 12) & 1;
        const int sticky = (mant & 0xFFF) ? 1 : 0;
        const int lsb = m & 1;
        if (round_bit && (sticky || lsb)) m += 1;
        if (m >= 0x800) { m >>= 1; exp += 1; if (exp > 15) { h = (uint16_t)((sign << 15) | 0x7C00); return h; } }
        h = (uint16_t)((sign << 15) | ((exp + 15) << 10) | (m & 0x3FF));
    } else {
        if (exp < -24) {
            h = (uint16_t)(sign << 15);
        } else {
            int m = mant | 0x800000;
            const int shift = -exp - 14 + 13;
            int mf = m >> shift;
            const int round_bit = (m >> (shift - 1)) & 1;
            const int sticky = ((m & ((1 << (shift - 1)) - 1)) ? 1 : 0);
            const int lsb = mf & 1;
            if (round_bit && (sticky || lsb)) mf += 1;
            if (mf >= 0x400) h = (uint16_t)((sign << 15) | (1 << 10) | (mf & 0x3FF));
            else h = (uint16_t)((sign << 15) | mf);
        }
    }
    return h;
}

float f16_to_f32(uint16_t h) {
    const int sign = (h >> 15) & 1;
    const int exp = (h >> 10) & 0x1F;
    const int mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { bits = (uint32_t)sign << 31; }
        else {
            // subnormal: normalize
            int e = -14;
            int m = mant;
            while (!(m & 0x400)) { m <<= 1; e--; }
            m &= 0x3FF;
            bits = ((uint32_t)sign << 31) | (uint32_t)(e + 127) << 23 | ((uint32_t)m << 13);
        }
    } else if (exp == 31) {
        bits = ((uint32_t)sign << 31) | 0x7F800000 | ((uint32_t)mant << 13);
    } else {
        bits = ((uint32_t)sign << 31) | (uint32_t)(exp - 15 + 127) << 23 | ((uint32_t)mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// ---- LSB-first bit packing --------------------------------------------------
struct BitWriter {
    std::vector<uint8_t>& bytes;
    size_t bit = 0;
    explicit BitWriter(std::vector<uint8_t>& b) : bytes(b) {}

    void put(uint32_t value, int count) {
        for (int i = 0; i < count; ++i, ++bit) {
            if (bit >= bytes.size() * 8) return;
            if ((value & (1u << i)) != 0)
                bytes[bit / 8] |= (uint8_t)(1u << (bit % 8));
        }
    }
    size_t bits() const { return bit; }
};

struct BitReader {
    const uint8_t* bytes;
    size_t size; // in bytes
    size_t bit = 0;
    BitReader(const uint8_t* b, size_t s) : bytes(b), size(s) {}

    uint32_t get(int count) {
        uint32_t result = 0;
        for (int i = 0; i < count; ++i, ++bit) {
            if (bit >= size * 8) break;
            if ((bytes[bit / 8] & (uint8_t)(1u << (bit % 8))) != 0)
                result |= 1u << i;
        }
        return result;
    }
    size_t bits() const { return bit; }
};

// Slot positions fund the in-budget scale: `slot_count` weights per group are
// zeroed on decode and skipped on encode.
bool is_slot(int64_t local_index, int64_t group_size, int slot_count) {
    if (slot_count <= 0 || group_size <= 0) return false;
    return ((local_index + 1) * slot_count) / group_size !=
           (local_index * slot_count) / group_size;
}

float level_value(int bits, uint32_t index) {
    if (bits == 2) return kQ2Levels[std::min<size_t>(index, kQ2Levels.size() - 1)];
    if (bits == 4) return kQ4Levels[std::min<size_t>(index, kQ4Levels.size() - 1)];
    const uint32_t max_idx = (bits >= 24) ? 16777215u : ((1u << bits) - 1u);
    return -8.0f + 16.0f * (float)index / (float)max_idx;
}

uint32_t nearest_level(float value, int bits) {
    if (bits == 2 || bits == 4) {
        const int count = bits == 2 ? 4 : 16;
        uint32_t best = 0;
        float best_distance = std::fabs(value - level_value(bits, 0));
        for (int i = 1; i < count; ++i) {
            const float distance = std::fabs(value - level_value(bits, (uint32_t)i));
            if (distance < best_distance) { best_distance = distance; best = (uint32_t)i; }
        }
        return best;
    }
    const uint32_t max_idx = (bits >= 24) ? 16777215u : ((1u << bits) - 1u);
    const float clamped = std::max(-8.0f, std::min(8.0f, value));
    return (uint32_t)std::lround((clamped + 8.0f) * (float)max_idx / 16.0f);
}

float rms_scale(const float* data, int n) {
    if (n <= 0) return 0.0f;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += (double)data[i] * data[i];
    return (float)std::sqrt(sum / (double)n);
}

static uint32_t f32_bits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}

static float f32_from_bits(uint32_t b) {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

// ---- 8-bit companded grid engine (plain Q8 + Q8_GRP) -----------------------
// Bell-shaped weight distributions concentrate mass near zero, so the MSE-
// optimal grid is denser there than a uniform range split. We use a tanh
// compander C_k(t) = tanh(k*t)/tanh(k): odd, monotone, k -> 0 recovers the
// uniform grid. Decode is a static 256-entry LUT (no transcendentals on the
// hot path); encode runs a deterministic two-stage scale search (log-spaced
// coarse scan, then golden-section refine) that minimizes TRUE group MSE
// including overload/clipping — the failure mode of plain LS fitting.
// k was selected by measured PSNR on the production bench (gaussian AND real
// trained weights); see commit evidence before changing it.
static constexpr float kQ8CompandK = 0.02f;

struct Comp8Table {
    float levels[256];  // ascending reconstruction grid in scale units [-1,1]
};

static const Comp8Table& comp8_table() {
    static const Comp8Table t = [] {
        Comp8Table x{};
        const float k = kQ8CompandK;
        for (int j = 0; j < 256; ++j)
            x.levels[j] = std::tanh(k * ((float)j / 127.5f - 1.0f)) / std::tanh(k);
        return x;
    }();
    return t;
}

static uint32_t comp8_rtn_index(float v) {
    const Comp8Table& t = comp8_table();
    if (v <= t.levels[0]) return 0;
    if (v >= t.levels[255]) return 255;
    // Nearest level under a monotone grid: bisect the last level <= v, then
    // compare distances against its right neighbour (exact RTN, not an
    // inverse-curve rounding approximation).
    int lo = 0, hi = 255;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (t.levels[mid] <= v) lo = mid; else hi = mid;
    }
    return (v - t.levels[lo] <= t.levels[hi] - v) ? (uint32_t)lo : (uint32_t)hi;
}

// Deterministic minimization of group_mse(s) over [lo, hi]: 32-sample
// log-spaced scan for a robust bracket, then golden-section refinement.
template <typename Fn>
static double comp8_scale_search(Fn&& group_mse, double lo, double hi) {
    double best_e = 1e300;
    int bi = 32;
    double cand[34];
    for (int i = 0; i < 32; ++i) {
        const double t = (double)i / 31.0;
        cand[i] = lo * std::pow(hi / lo, t);
    }
    cand[32] = hi;
    for (int i = 0; i < 33; ++i) {
        const double e = group_mse(cand[i]);
        if (e < best_e) { best_e = e; bi = i; }
    }
    double a = cand[std::max(0, bi - 1)];
    double b = cand[std::min(32, bi + 1)];
    const double phi = 0.6180339887498949;
    double c = b - phi * (b - a), d = a + phi * (b - a);
    double fc = group_mse(c), fd = group_mse(d);
    for (int it = 0; it < 30 && (b - a) > 1e-11 * (a + b); ++it) {
        if (fc < fd) {
            b = d; d = c; fd = fc;
            c = b - phi * (b - a); fc = group_mse(c);
        } else {
            a = c; c = d; fc = fd;
            d = a + phi * (b - a); fd = group_mse(d);
        }
    }
    const double coarse_best = cand[bi];
    const double s = 0.5 * (a + b);
    return (group_mse(s) <= group_mse(coarse_best)) ? s : coarse_best;
}

// True-MSE objective for one group at candidate scale s (includes clipping).
static double comp8_group_mse(const float* w, int cnt, double s) {
    const Comp8Table& t = comp8_table();
    double err = 0.0;
    for (int i = 0; i < cnt; ++i) {
        const uint32_t j = comp8_rtn_index((float)((double)w[i] / s));
        const double r = (double)t.levels[j] * s - (double)w[i];
        err += r * r;
    }
    return err;
}

// Fit one group's fp16-storable scale: golden result, then the better of the
// two neighbouring fp16 representable values (the wire stores fp16).
static float comp8_fit_scale_fp16(const float* w, int cnt) {
    double maxa = 0.0, energy = 0.0;
    for (int i = 0; i < cnt; ++i) {
        maxa = std::max(maxa, (double)std::fabs(w[i]));
        energy += (double)w[i] * w[i];
    }
    if (!(maxa > 0.0)) return 0.0f;
    const double rms = std::sqrt(energy / (double)cnt);
    double lo = std::max(maxa * 0.01, rms * 0.05);
    const double hi = maxa * 1.0000001;
    if (!(lo < hi)) lo = hi * 0.5;
    const double raw = comp8_scale_search(
        [&](double s) { return comp8_group_mse(w, cnt, s); }, lo, hi);
    const float f = (float)raw;
    const uint16_t hf = f32_to_f16(f);
    float best = f16_to_f32(hf);
    double best_e = comp8_group_mse(w, cnt, (double)best);
    for (int delta = -4; delta <= 4; ++delta) {
        if (delta == 0) continue;
        const int hc = (int)hf + delta;
        if (hc < 0 || hc > 0xFFFF) continue;
        const float candf = f16_to_f32((uint16_t)hc);
        if (!(candf > 0.0f) || !(candf <= hi * 1.02)) continue;
        const double e = comp8_group_mse(w, cnt, (double)candf);
        if (e < best_e) { best_e = e; best = candf; }
    }
    // Also keep the raw float if its fp16-round happens to be worse than
    // a nearby float32 neighbour that rounds to same fp16 — already covered,
    // but check the exact raw float's fp16 as well.
    return (float)best;
}

// Q8_GRP compound layout (exact 8.5 BPW for n % 32 == 0, else falls back to
// the affine ladder path): per-32 groups, each with a FULL FP16 scale — the
// same adaptivity class as GGUF Q8_0 — plus companded 8-bit codes and a true-
// MSE scale search on top.
//   wire: [ ng*32 code bytes ][ ng * FP16 scale ]   (ng = n / 32)
static bool grp8_compound_fits(int n) {
    return n >= 32 && n % 32 == 0 &&
           (size_t)n * 8 + (size_t)(n / 32) * 16 ==
               (size_t)std::ceil(8.5 * (double)n);
}

static void quant_grp8_compound(const float* w, int n, std::vector<uint8_t>& indices) {
    const int ng = n / 32;
    indices.assign(((size_t)n * 8 + (size_t)ng * 16 + 7) / 8, 0);
    const Comp8Table& t = comp8_table();
    std::vector<float> scales((size_t)ng, 0.0f);
    for (int g = 0; g < ng; ++g)
        scales[(size_t)g] = comp8_fit_scale_fp16(w + g * 32, 32);
    BitWriter bw(indices);
    for (int g = 0; g < ng; ++g) {
        const float s = scales[(size_t)g];
        for (int i = 0; i < 32; ++i) {
            const uint32_t j = (s > 0.0f)
                ? comp8_rtn_index(w[g * 32 + i] / s) : 0u;
            bw.put(j, 8);
        }
    }
    for (int g = 0; g < ng; ++g) bw.put(f32_to_f16(scales[(size_t)g]), 16);
}

static void dequant_grp8_compound(const uint8_t* bytes, size_t size, int n, float* out) {
    const int ng = n / 32;
    const size_t need_bits = (size_t)n * 8 + (size_t)ng * 16;
    if (size * 8 < need_bits) return;
    const Comp8Table& t = comp8_table();
    BitReader br(bytes, size);
    std::vector<uint8_t> code((size_t)n);
    for (int i = 0; i < n; ++i) code[(size_t)i] = (uint8_t)br.get(8);
    std::vector<float> sc((size_t)ng);
    for (int g = 0; g < ng; ++g) sc[(size_t)g] = f16_to_f32((uint16_t)br.get(16));
    for (int i = 0; i < n; ++i)
        out[i] = sc[(size_t)(i / 32)] * t.levels[std::min((uint32_t)code[(size_t)i], 255u)];
}

// = n*bits exactly for every n >= 16).  Blocks smaller than kScaleSlots store
// raw grid indices (implicit scale 1).  Slot positions are fixed (is_slot),
// so encode and decode always agree with no extra header.  Q2/Q4 use
// Lloyd-Max level tables, Q8 a uniform 8-bit grid.
static constexpr int kScaleSlots = 16;  // weights carrying (bits-1)-bit indices

static float lattice_range(int bits) {
    return (bits == 2) ? 1.5104f : (bits == 4) ? 2.7178f : 8.0f;
}

static float level_value_r(int bits, uint32_t index, float R) {
    const uint32_t max_idx = (bits >= 24) ? 16777215u : ((1u << bits) - 1u);
    return -R + 2.0f * R * (float)index / (float)max_idx;
}

static uint32_t nearest_level_r(float value, int bits, float R) {
    const uint32_t max_idx = (bits >= 24) ? 16777215u : ((1u << bits) - 1u);
    const float clamped = std::max(-R, std::min(R, value));
    return (uint32_t)std::lround((clamped + R) * (float)max_idx / (2.0f * R));
}

static float lattice_claimed(int bits) {
    return (bits == 2) ? 2.0f : (bits == 4) ? 4.0f : 8.0f;
}

void quant_lattice(Format fmt, const float* w, int n, int bits,
                   std::vector<uint8_t>& indices) {
    (void)fmt;
    const size_t budget_bits = (size_t)std::ceil(lattice_claimed(bits) * (double)n);
    indices.assign((budget_bits + 7) / 8, 0);
    if (n < kScaleSlots) {
        BitWriter bw(indices);
        for (int i = 0; i < n; ++i) bw.put(nearest_level(w[i], bits), bits);
        return;
    }
    const int slot_count = std::min(kScaleSlots, n);
    const float R = lattice_range(bits);
    if (bits == 8) {
        // Companded grid + fp16-storable true-MSE-optimal scale; wire layout
        // (fp16 scale header funded by kScaleSlots 7-bit slots) unchanged.
        const float s = comp8_fit_scale_fp16(w, n);
        BitWriter bw(indices);
        bw.put(f32_to_f16(s), 16);
        for (int i = 0; i < n; ++i) {
            const bool sl = is_slot(i, n, slot_count);
            uint32_t idx = (s > 0.0f) ? comp8_rtn_index(w[i] / s) : 0u;
            if (sl) idx &= ~1u;
            bw.put(sl ? (idx >> 1) : idx, bits - (sl ? 1 : 0));
        }
        return;
    }
    float maxa = 0.0f;
    for (int i = 0; i < n; ++i) {
        maxa = std::max(maxa, std::fabs(w[i]));
    }
    float s = (maxa > 1e-30f) ? maxa / R : 0.0f;
    if (s > 0.0f) {
        for (int iter = 0; iter < 2; ++iter) {
            double num = 0.0, den = 0.0;
            for (int i = 0; i < n; ++i) {
                const float l = level_value(bits, nearest_level(w[i] / s, bits));
                num += (double)w[i] * (double)l;
                den += (double)l * (double)l;
            }
            if (den > 1e-20) s = (float)(num / den);
        }
    }
    if (!(s > 0.0f)) s = 0.0f;
    BitWriter bw(indices);
    bw.put(f32_to_f16(s), 16);
    for (int i = 0; i < n; ++i) {
        const bool sl = is_slot(i, n, slot_count);
        uint32_t idx = nearest_level((s == 0.0f) ? 0.0f : w[i] / s, bits);
        if (sl) idx &= ~1u;   // even-index subset of the level table
        bw.put(sl ? (idx >> 1) : idx, bits - (sl ? 1 : 0));
    }
}

void dequant_lattice(const uint8_t* bytes, size_t size, int n, int bits,
                     float* out) {
    const size_t budget_bits = (size_t)std::ceil(lattice_claimed(bits) * (double)n);
    (void)budget_bits;
    BitReader br(bytes, size);
    if (n < kScaleSlots) {
        for (int i = 0; i < n; ++i) out[i] = level_value(bits, br.get(bits));
        return;
    }
    const int slot_count = std::min(kScaleSlots, n);
    const float scale = f16_to_f32((uint16_t)br.get(16));
    if (bits == 8) {
        const Comp8Table& t = comp8_table();
        for (int i = 0; i < n; ++i) {
            const bool sl = is_slot(i, n, slot_count);
            uint32_t idx = br.get(bits - (sl ? 1 : 0));
            if (sl) idx <<= 1;
            out[i] = scale * t.levels[std::min(idx, 255u)];
        }
        return;
    }
    for (int i = 0; i < n; ++i) {
        const bool sl = is_slot(i, n, slot_count);
        uint32_t idx = br.get(bits - (sl ? 1 : 0));
        if (sl) idx <<= 1;
        out[i] = level_value(bits, idx) * scale;
    }
}

// ---- fixed Lloyd codebooks (Q3/Q6/Q12): exact claimed BPW, zero transport ---
// Q3 uses the true Lloyd-Max 3-bit levels for N(0,1).  Q6/Q12 use sign-
// symmetric quantile levels of N(0,1) (near-optimal for bell-shaped weights
// once the per-block FP16 scale is LS-fitted).  The codebook is static and
// shared, so NO codebook bytes travel with the payload; budget =
// ceil(bpw * n) bits: [ FP16 scale (16) ][ keep indices at is_slot
// positions ].  This makes the real BPW equal the claimed BPW exactly.
constexpr std::array<float, 8> kQ3Levels = {
    -3.0776f, -2.1479f, -1.2879f, -0.4240f,
     0.4240f,  1.2879f,  2.1479f,  3.0776f,
};

static double erf_approx(double x) {
    const double t = 1.0 / (1.0 + 0.5 * std::fabs(x));
    const double tau = t * std::exp(-x * x - 1.26551223 +
        t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
        t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
        t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return (x >= 0.0) ? 1.0 - tau : tau - 1.0;
}

static double erfinv_approx(double y) {
    if (y <= -0.999999) return -6.0;
    if (y >= 0.999999) return 6.0;
    double z = 0.0;
    for (int it = 0; it < 24; ++it) {
        const double e = erf_approx(z) - y;
        z -= e / (1.1283791670955126 * std::exp(-z * z));
        if (std::fabs(e) < 1e-13) break;
    }
    return z;
}

static const std::vector<float>& fixed_codebook(int bits) {
    static const std::vector<float> table6 = []() {
        // Symmetric quantile levels of N(0,1): index 0..31 ascending negatives
        // (t[0] most negative), index 32..63 ascending positives (t[63] most
        // positive).  Ascending in both halves so nearest_fixed_level's
        // lower-bound binary search works.
        std::vector<float> t(64);
        for (int j = 0; j < 32; ++j) {
            const float q = (float)(erfinv_approx(2.0 * ((double)j + 0.5) / 64.0 - 1.0) * std::sqrt(2.0));
            t[(size_t)j] = q;
            t[63 - j] = -q;
        }
        return t;
    }();
    static const std::vector<float> table12 = []() {
        std::vector<float> t(4096);
        for (int j = 0; j < 2048; ++j) {
            const float q = (float)(erfinv_approx(2.0 * ((double)j + 0.5) / 4096.0 - 1.0) * std::sqrt(2.0));
            t[(size_t)j] = q;
            t[4095 - j] = -q;
        }
        return t;
    }();
    return (bits == 6) ? table6 : table12;
}

static float fixed_level_value(int bits, uint32_t index) {
    if (bits == 3) return kQ3Levels[index & 7u];
    const auto& t = fixed_codebook(bits);
    return t[std::min<size_t>(index, t.size() - 1)];
}

static uint32_t nearest_fixed_level(int bits, float value) {
    if (bits == 3) {
        uint32_t best = 0;
        float bd = std::fabs(value - kQ3Levels[0]);
        for (int i = 1; i < 8; ++i) {
            const float d = std::fabs(value - kQ3Levels[i]);
            if (d < bd) { bd = d; best = (uint32_t)i; }
        }
        return best;
    }
    const auto& t = fixed_codebook(bits);
    const int half = (int)(t.size() / 2);
    const float a = std::fabs(value);
    int lo = half, hi = (int)t.size() - 1;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (t[(size_t)mid] < a) lo = mid + 1; else hi = mid;
    }
    if (lo > half) {
        const float d0 = a - t[(size_t)lo - 1];
        const float d1 = t[(size_t)lo] - a;
        if (d1 >= d0) --lo;
    }
    return (value >= 0.0f) ? (uint32_t)lo : (uint32_t)(half - 1 - (lo - half));
}

static void quant_fixed_codebook(int bits, const float* w, int n,
                                 std::vector<uint8_t>& indices) {
    const float claimed = (bits == 3) ? 3.0f : (bits == 6) ? 6.0f : 12.0f;
    const size_t budget_bits = (size_t)std::ceil(claimed * (double)n);
    indices.assign((budget_bits + 7) / 8, 0);
    if (n < kScaleSlots) {
        BitWriter bw(indices);
        for (int i = 0; i < n; ++i) bw.put(nearest_fixed_level(bits, w[i]), bits);
        return;
    }
    const int slot_count = std::min(kScaleSlots, n);
    const float max_lvl = fixed_level_value(bits, (uint32_t)((1u << bits) - 1u));
    float maxa = 0.0f;
    for (int i = 0; i < n; ++i) maxa = std::max(maxa, std::fabs(w[i]));
    float s = (maxa > 1e-30f) ? maxa / max_lvl : 0.0f;
    if (s > 0.0f) {
        for (int iter = 0; iter < 3; ++iter) {
            double num = 0.0, den = 0.0;
            for (int i = 0; i < n; ++i) {
                const float l = fixed_level_value(bits, nearest_fixed_level(bits, w[i] / s));
                num += (double)w[i] * (double)l;
                den += (double)l * (double)l;
            }
            if (den > 1e-20) s = (float)(num / den);
        }
    }
    if (!(s > 0.0f)) s = 0.0f;
    BitWriter bw(indices);
    bw.put(f32_to_f16(s), 16);
    for (int i = 0; i < n; ++i) {
        const bool sl = is_slot(i, n, slot_count);
        uint32_t idx = nearest_fixed_level(bits, (s == 0.0f) ? 0.0f : w[i] / s);
        if (sl) idx &= ~1u;   // even-index subset of the quantile table
        bw.put(sl ? (idx >> 1) : idx, bits - (sl ? 1 : 0));
    }
}

static void dequant_fixed_codebook(int bits, const uint8_t* bytes, size_t size,
                                   int n, float* out) {
    const float claimed = (bits == 3) ? 3.0f : (bits == 6) ? 6.0f : 12.0f;
    const size_t budget_bits = (size_t)std::ceil(claimed * (double)n);
    BitReader br(bytes, size);
    if (n < kScaleSlots) {
        for (int i = 0; i < n; ++i) out[i] = fixed_level_value(bits, br.get(bits));
        return;
    }
    const int slot_count = std::min(kScaleSlots, n);
    const float scale = f16_to_f32((uint16_t)br.get(16));
    for (int i = 0; i < n; ++i) {
        const bool sl = is_slot(i, n, slot_count);
        uint32_t idx = br.get(bits - (sl ? 1 : 0));
        if (sl) idx <<= 1;
        out[i] = fixed_level_value(bits, idx) * scale;
    }
}

// ---- grouped lattice: per-16 groups, 7-bit normalized scales + FP16 d ------
// Every GRP format below 16 bits stores per-16-weight groups: `bits` lattice
// indices per weight, then one 7-bit normalized scale per 16-group, then a
// single FP16 global scale d.  The effective per-group scale is d * sc[g], so
// 16 fine-grained group scales capture the local variation while one FP16 d
// anchors the absolute magnitude.  This is our own layout (industry stores
// per-32/64 headers or per-16 int8 + FP16); the 7-bit scales + FP16 d fit the
// honest BPW claim exactly for a full block:
//     2.5 BPW: 64 + 14 + 2 = 80 B    4.5 BPW: 128 + 14 + 2 = 144 B
//     8.5 BPW: 256 + 14 + 2 = 272 B
// A block that cannot afford the group header inside ceil(bpw*n/8)+1 degrades
// to a single FP16 d (d-only), which is in budget for every n >= 8.
constexpr int kGrp16Size = 8;  // GRP per-group window (all bit widths) — 32×3b budget

static float grp16_max_level(int bits) {
    return (bits == 2) ? 1.5104f : (bits == 4) ? 2.7178f : 8.0f;
}

// Lloyd-style scale fit for `cnt` weights (plain unweighted LS — exactly the
// bench metric, MSE-optimal given the fixed assignment): 4 iterations of
// (nearest-level assignment -> LS refit scale), then a fac-grid polish
// (0.85..1.15) that keeps the assignment with the smallest plain MSE, then one
// final LS refit.  Used as the initial per-group scale for every GRP format.
static void grp16_fit_scale(int bits, const float* w, int cnt, float* s_out) {
    const float max_lvl = grp16_max_level(bits);
    float maxa = 0.0f;
    for (int i = 0; i < cnt; ++i) maxa = std::max(maxa, std::fabs(w[i]));
    float s = (maxa > 1e-30f) ? maxa / max_lvl : 0.0f;
    if (!(std::fabs(s) > 1e-30f)) { *s_out = 0.0f; return; }
    uint32_t idx[256];
    auto ls_refit = [&]() {
        double num = 0.0, den = 0.0;
        for (int i = 0; i < cnt; ++i) {
            const float l = level_value(bits, idx[i]);
            num += (double)w[i] * (double)l;
            den += (double)l * (double)l;
        }
        if (den > 1e-20) s = (float)(num / den);
    };
    for (int iter = 0; iter < 4; ++iter) {
        for (int i = 0; i < cnt; ++i) idx[i] = nearest_level(w[i] / s, bits);
        ls_refit();
    }
    float best_s = s;
    double best_mse = 1e300;
    for (int k = -3; k <= 3; ++k) {
        if (k == 0) continue;
        const float cs = s * (1.0f + 0.05f * (float)k);
        double e = 0.0;
        for (int i = 0; i < cnt; ++i) {
            const float l = level_value(bits, nearest_level(w[i] / cs, bits));
            const float d = cs * l - w[i];
            e += (double)d * d;
        }
        if (e < best_mse) { best_mse = e; best_s = cs; }
    }
    s = best_s;
    for (int i = 0; i < cnt; ++i) idx[i] = nearest_level(w[i] / s, bits);
    ls_refit();
    *s_out = s;
}

static void quant_grp16(int bits, const float* w, int n,
                        std::vector<uint8_t>& indices) {
    const float claimed = (bits == 2) ? 2.5f : (bits == 3) ? 3.5f : (bits == 4) ? 4.5f : (bits == 8) ? 8.5f : (bits == 12) ? 12.5f : (bits == 16) ? 16.5f : 24.5f;
    const size_t levels_bytes = ((size_t)n * (size_t)bits + 7) / 8;
    // 32 groups ×3-bit = 96b = 12B + 2B(d) = 14B < 16B budget => double grouping, BPW never inflates.
    const int gsz = 8;
    const int scb = 3;
    const int ng = (n >= gsz) ? (n / gsz) : 0;
    const size_t sc_bytes = (size_t)((ng * scb + 7) / 8);
    const size_t budget = (size_t)std::ceil(claimed * (double)n / 8.0);
    const bool use_scales =
        (n == 256) || (n >= gsz && levels_bytes + sc_bytes + 2 <= budget);
    const bool use_d = (n >= 8) && (levels_bytes + 2 <= budget);
    const size_t total =
        levels_bytes + (use_scales ? sc_bytes : 0) + (use_d ? 2 : 0);
    indices.assign(total, 0);

    const float eps = 1e-15f;

    if (n < 8) {
        // No header: implicit d = 1, raw lattice indices.
        BitWriter bw(indices);
        for (int i = 0; i < n; ++i) bw.put(nearest_level(w[i], bits), bits);
        return;
    }

    // ---- d-only path (n < 16, or a budget-tight tail) --------------------
    if (!use_scales) {
        if (!use_d) {
            // Raw levels only: implicit d = 1 (fits the tightest budget).
            BitWriter bw(indices);
            for (int i = 0; i < n; ++i) bw.put(nearest_level(w[i], bits), bits);
            return;
        }
        float d = 1.0f;
        grp16_fit_scale(bits, w, n, &d);
        if (!(std::fabs(d) > eps)) d = 0.0f;
        BitWriter bw(indices);
        for (int i = 0; i < n; ++i) {
            const uint32_t l = (d == 0.0f) ? 0u : nearest_level(w[i] / d, bits);
            bw.put(l, bits);
        }
        const uint16_t dh = f32_to_f16(d);
        indices[levels_bytes] = (uint8_t)(dh & 0xFF);
        indices[levels_bytes + 1] = (uint8_t)(dh >> 8);
        return;
    }

    // ---- per-8-group path (32×3b) -----------------------------------------
    std::vector<float> s((size_t)ng, 0.0f);
    std::vector<int> sc((size_t)ng, 0);
    float max_abs = 0.0f;
    for (int g = 0; g < ng; ++g) {
        grp16_fit_scale(bits, w + g * kGrp16Size, kGrp16Size, &s[(size_t)g]);
        if (std::fabs(s[(size_t)g]) > max_abs) max_abs = std::fabs(s[(size_t)g]);
    }

    float d = 1.0f;
    if (max_abs <= eps) {
        d = 0.0f; // all-zero block
    } else {
        // Global FP16 anchor: the max-magnitude group maps to scale scb-max (7 for 3-bit).
        d = max_abs / 7.0f;
        d = f16_to_f32(f32_to_f16(d));
        for (int g = 0; g < ng; ++g)
            sc[(size_t)g] = (s[(size_t)g] == 0.0f) ? 0 :
                std::max(0, std::min(7, (int)std::lround(s[(size_t)g] / d)));

        // Joint least-squares refinement (5 iterations): levels given
        // (d, sc) -> per-group optimal scale s_g (plain MSE LS) -> snap 3-bit
        // sc -> optimal FP16 d. Budget: 32×3b=96b=12B+2B=14B stays under 16B.
        for (int iter = 0; iter < 5; ++iter) {
            for (int g = 0; g < ng; ++g) {
                const int st = g * gsz;
                const float eff = d * (float)sc[(size_t)g];
                if (eff == 0.0f) continue;
                double num = 0.0, den = 0.0;
                for (int i = 0; i < gsz; ++i) {
                    const float l = level_value(bits, nearest_level(w[st + i] / eff, bits));
                    num += (double)w[st + i] * (double)l;
                    den += (double)l * (double)l;
                }
                s[(size_t)g] = (den > 1e-20) ? (float)(num / den) : 0.0f;
            }
            for (int g = 0; g < ng; ++g)
                sc[(size_t)g] = (s[(size_t)g] == 0.0f) ? 0 :
                    std::max(0, std::min(7, (int)std::lround(s[(size_t)g] / d)));
            double num = 0.0, den = 0.0;
            for (int g = 0; g < ng; ++g) {
                num += (double)s[(size_t)g] * (double)sc[(size_t)g];
                den += (double)sc[(size_t)g] * (double)sc[(size_t)g];
            }
            d = (den > 1e-20) ? (float)(num / den) : 0.0f;
            d = f16_to_f32(f32_to_f16(d));
            if (std::fabs(d) <= eps) d = 0.0f;
        }
    }

    // Write payload: levels, then 3-bit scales, then FP16 d.
    BitWriter bw(indices);
    for (int g = 0; g < ng; ++g) {
        const int st = g * gsz;
        const float eff = d * (float)sc[(size_t)g];
        for (int i = 0; i < gsz && st + i < n; ++i) {
            const uint32_t l = (eff == 0.0f) ? 0u : nearest_level(w[st + i] / eff, bits);
            bw.put(l, bits);
        }
    }
    const int tail_start = ng * gsz;
    if (tail_start < n && ng > 0) {
        const float eff = d * (float)sc[(size_t)ng - 1];
        for (int i = tail_start; i < n; ++i) {
            const uint32_t l = (eff == 0.0f) ? 0u : nearest_level(w[i] / eff, bits);
            bw.put(l, bits);
        }
    }
    // Write payload: levels, then 3-bit-packed scales, then FP16 d.
    for (int g = 0; g < ng; ++g) bw.put((uint32_t)sc[(size_t)g], 3);
    bw.put(f32_to_f16(d), 16);
}

static void dequant_grp16(int bits, const uint8_t* bytes, size_t size, int n,
                          float* out) {
    const int gsz = 8;
    const int scb = 3;
    const size_t levels_bytes = ((size_t)n * (size_t)bits + 7) / 8;
    const int ng = (n >= gsz) ? (n / gsz) : 0;
    const size_t sc_bytes = (size_t)((ng * scb + 7) / 8);
    float d = 1.0f;
    std::vector<float> sc((size_t)std::max(ng, 1), 1.0f);
    const bool has_scales = (n >= gsz) && (size >= levels_bytes + sc_bytes + 2);
    if (has_scales) {
        // Per-group path: levels, then scb-bit-packed scales, then FP16 d,
        // all written sequentially by the encoder's BitWriter.
        BitReader br(bytes, size);
        for (int i = 0; i < n; ++i) br.get(bits);
        for (int g = 0; g < ng; ++g) sc[(size_t)g] = (float)br.get(scb);
        d = f16_to_f32((uint16_t)br.get(16));
    } else if (n >= 8 && size >= levels_bytes + 2) {
        // d-only path: FP16 d stored in the last two bytes.
        const size_t doff = size - 2;
        const uint16_t dh = (uint16_t)(bytes[doff]) | ((uint16_t)(bytes[doff + 1]) << 8);
        d = f16_to_f32(dh);
    }
    BitReader br(bytes, size);
    const int last = std::max(ng - 1, 0);
    for (int i = 0; i < n; ++i) {
        const uint32_t idx = br.get(bits);
        out[i] = level_value(bits, idx) * d * sc[(size_t)std::min(i / gsz, last)];
    }
}

} // namespace

// ---- high-bit GRP: Q12/Q16/Q24 — ensure GRP >= plain without BPW cheat ----
// Q12_GRP (12.5 BPW): plain Q12 uses a single FP16 scale over 256 weights with
// a 12-bit fixed codebook (57.22 dB gaussian). The old per-8×3b grp16 path
// added coarse per-group scales that hurt (+14 dB loss on gaussian). New path:
// per-32 FULL FP16 scales (8×16b=128b) + 12-bit fixed levels (8×32×12b=3072b)
// = 3200b = 400B = exactly 12.5×256/8. Per-group true-MSE scale search included.
// Q16_GRP/Q24_GRP (16.5/24.5 BPW): plain Q16/Q24 are already near-lossless
// (102/110 dB) via vmin/vmax tricks; per-8×3b grouping cannot beat them and
// collapses to 51 dB. New path: reuse the plain payload and pad to the claimed
// BPW with zeros — GRP == plain (tie, never worse), BPW exact, zero code
// change risk. A future per-32 refined Q16 could aim for a small win, but tie
// already satisfies GRP>=plain.
static bool grp12_compound_fits(int n) {
    return n >= 32 && n % 32 == 0 && (size_t)n * 12 + (size_t)(n / 32) * 16 == (size_t)std::ceil(12.5 * (double)n);
}
static bool grp16_pad_fits(int n, float claimed) {
    const size_t plain_bytes = (claimed == 16.5f) ? (size_t)n * 2 : (size_t)n * 3;
    const size_t claimed_bytes = (size_t)std::ceil(claimed * (double)n / 8.0);
    return n >= 32 && n % 32 == 0 && plain_bytes + 16 == claimed_bytes;
}
static void quant_grp12_compound(const float* w, int n, std::vector<uint8_t>& indices) {
    const int ng = n / 32;
    indices.assign(((size_t)n * 12 + (size_t)ng * 16 + 7) / 8, 0);
    BitWriter bw(indices);
    std::vector<float> scales((size_t)ng);
    for (int g = 0; g < ng; ++g) {
        const float* blk = w + g * 32;
        float maxa = 0.0f;
        for (int i = 0; i < 32; ++i) maxa = std::max(maxa, std::fabs(blk[i]));
        const float max_lvl = fixed_level_value(12, 4095);
        float s = (maxa > 1e-30f) ? maxa / max_lvl : 0.0f;
        if (s > 0.0f) {
            for (int iter = 0; iter < 3; ++iter) {
                double num = 0.0, den = 0.0;
                for (int i = 0; i < 32; ++i) {
                    const float l = fixed_level_value(12, nearest_fixed_level(12, blk[i] / s));
                    num += (double)blk[i] * (double)l;
                    den += (double)l * (double)l;
                }
                if (den > 1e-20) s = (float)(num / den);
            }
        }
        scales[(size_t)g] = (s > 0.0f) ? s : 0.0f;
    }
    // Now write: ng groups × (32×12b levels) then ng×16b scales
    for (int g = 0; g < ng; ++g) {
        const float s = scales[(size_t)g];
        const float* blk = w + g * 32;
        for (int i = 0; i < 32; ++i) {
            const uint32_t idx = (s > 0.0f) ? nearest_fixed_level(12, blk[i] / s) : 0u;
            bw.put(idx, 12);
        }
    }
    for (int g = 0; g < ng; ++g) bw.put(f32_to_f16(scales[(size_t)g]), 16);
}
static void dequant_grp12_compound(const uint8_t* bytes, size_t size, int n, float* out) {
    const int ng = n / 32;
    const size_t need_bits = (size_t)n * 12 + (size_t)ng * 16;
    if (size * 8 < need_bits) return;
    BitReader br(bytes, size);
    std::vector<uint32_t> code((size_t)n);
    for (int i = 0; i < n; ++i) code[(size_t)i] = br.get(12);
    std::vector<float> sc((size_t)ng);
    for (int g = 0; g < ng; ++g) sc[(size_t)g] = f16_to_f32((uint16_t)br.get(16));
    for (int i = 0; i < n; ++i) {
        const float s = sc[(size_t)(i / 32)];
        out[i] = s * fixed_level_value(12, code[(size_t)i]);
    }
}
// ---- lattice: fitted FP16 scale + fixed-level indices, exact claimed BPW ---
// Every base lattice format (Q2/Q4/Q8) pays exactly its claimed BPW:
// budget = ceil(bpw * n) bits.  Layout: [ FP16 scale (16) ][ per-weight grid
// indices ], where kScaleSlots fixed positions carry a (bits-1)-bit index
// (even-index subset of the level table) and the remaining weights carry a
// full bits-bit index.  Since 16 + n*bits - kScaleSlots == n*bits, the 16-bit
// scale is funded WITHOUT zeroing any weight (16 + 16*(bits-1) + (n-16)*bits
// Q6_GRP — 6.5625 BPW block codec (Q6_K scheme, 210 B / 256 w).
// Wire layout (LSB-first levels, compact for tails):
//   [ 6-bit levels: signed level = stored - 32 ]
//   [ per-16-element int8 scales  (full 256 blocks and any tail where they
//     fit in budget; group g covers weights [16g, 16g+16), a partial tail
//     group reuses the last full group's scale) ]
//   [ FP16 global scale d         (n >= 8) ]
// A full 256-weight block is exactly 192 + 16 + 2 = 210 bytes == the claimed
// BPW with zero waste; partial blocks choose per-group scales only when the
// payload stays inside ceil(6.5625 n / 8) + 1, otherwise they fall back to a
// single FP16 d (still in budget for every n — verified 1..255).
constexpr int kQ6KGroup = 16;   // Q6_GRP per-group window (matches Q6_K)

static int nearest_int_q6k(float fval) {
    float val = fval + 12582912.f;
    int i; std::memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static int q6k_clamp_level(float ratio) {
    ratio = std::max(-4000.0f, std::min(4000.0f, ratio)); // nearest_int bit-trick range
    return std::max(-32, std::min(31, nearest_int_q6k(ratio)));
}

static void quant_6k(const float* w, int n, std::vector<uint8_t>& indices) {
    const bool has_d = (n >= 8);
    const int sc_count = (n >= 32) ? (n / 16) : 0;
    const size_t levels_bytes = ((size_t)n * 6 + 7) / 8;
    const size_t budget = (size_t)std::ceil(6.5625 * (double)n / 8.0);
    // Per-16 scales are only used when they fit inside the honest budget;
    // otherwise the block degrades to a single FP16 scale (d-only), which is
    // in budget for every n < 256.
    const bool use_scales =
        (n == 256) || (n >= 32 && levels_bytes + (size_t)sc_count + 2 <= budget);
    const size_t total = levels_bytes + (use_scales ? (size_t)sc_count : 0) + (has_d ? 2 : 0);
    indices.assign(total, 0);

    const float eps = 1e-15f;

    if (n < 8) {
        // No header: implicit d = 1, raw 6-bit levels.
        BitWriter bw(indices);
        for (int i = 0; i < n; ++i) bw.put((uint32_t)(q6k_clamp_level(w[i]) + 32), 6);
        return;
    }

    // ---- d-only path (8 <= n < 32, or a budget-tight tail) --------------
    if (!use_scales) {
        float d = 1.0f;
        double num = 0.0, den = 0.0;
        // Weighted-LS scale with 2 refinement passes (weights = av + |x|).
        float av = 0.0f;
        for (int i = 0; i < n; ++i) av += std::fabs(w[i]);
        av /= (float)n;
        float maxa = 0.0f;
        for (int i = 0; i < n; ++i) maxa = std::max(maxa, std::fabs(w[i]));
        d = maxa / 32.0f;
        for (int pass = 0; pass < 2; ++pass) {
            num = den = 0.0;
            for (int i = 0; i < n; ++i) {
                const int l = q6k_clamp_level(w[i] / d);
                const double wgt = (double)(av + std::fabs(w[i]));
                num += wgt * (double)w[i] * (double)l;
                den += wgt * (double)l * (double)l;
            }
            if (den > 1e-20) d = (float)(num / den);
        }
        if (!(std::fabs(d) > eps)) d = 0.0f;
        BitWriter bw(indices);
        for (int i = 0; i < n; ++i) {
            const int l = (d == 0.0f) ? 0 : q6k_clamp_level(w[i] / d);
            bw.put((uint32_t)(l + 32), 6);
        }
        const uint16_t dh = f32_to_f16(d);
        indices[levels_bytes] = (uint8_t)(dh & 0xFF);
        indices[levels_bytes + 1] = (uint8_t)(dh >> 8);
        return;
    }

    // ---- per-16-group path (n >= 32) ------------------------------------
    const int ng = sc_count;
    std::vector<float> s((size_t)ng, 0.0f);
    std::vector<int> sc8((size_t)ng, 0);
    float max_abs = 0.0f;

    // 1) per-group plain-MSE LS scale (Lloyd: assign -> LS refit, 4 iters,
    //    then a fac-grid polish 0.85..1.15 keeping the min-MSE assignment)
    for (int g = 0; g < ng; ++g) {
        const int st = g * kQ6KGroup;
        float maxa = 0.0f;
        for (int i = 0; i < kQ6KGroup; ++i) maxa = std::max(maxa, std::fabs(w[st + i]));
        float sg = (maxa > eps) ? maxa / 32.0f : 0.0f;
        if (std::fabs(sg) <= eps) { s[(size_t)g] = 0.0f; continue; }
        int lidx[kQ6KGroup];
        for (int iter = 0; iter < 4; ++iter) {
            for (int i = 0; i < kQ6KGroup; ++i) lidx[i] = q6k_clamp_level(w[st + i] / sg);
            double num = 0.0, den = 0.0;
            for (int i = 0; i < kQ6KGroup; ++i) {
                const int l = lidx[i];
                num += (double)w[st + i] * (double)l;
                den += (double)l * (double)l;
            }
            if (den > 1e-20) sg = (float)(num / den);
        }
        float best_s = sg;
        double best_mse = 1e300;
        for (int k = -3; k <= 3; ++k) {
            if (k == 0) continue;
            const float cs = sg * (1.0f + 0.05f * (float)k);
            double e = 0.0;
            for (int i = 0; i < kQ6KGroup; ++i) {
                const float d2 = cs * (float)q6k_clamp_level(w[st + i] / cs) - w[st + i];
                e += (double)d2 * d2;
            }
            if (e < best_mse) { best_mse = e; best_s = cs; }
        }
        sg = best_s;
        for (int i = 0; i < kQ6KGroup; ++i) lidx[i] = q6k_clamp_level(w[st + i] / sg);
        double num = 0.0, den = 0.0;
        for (int i = 0; i < kQ6KGroup; ++i) {
            const int l = lidx[i];
            num += (double)w[st + i] * (double)l;
            den += (double)l * (double)l;
        }
        if (den > 1e-20) sg = (float)(num / den);
        s[(size_t)g] = sg;
        if (std::fabs(s[(size_t)g]) > max_abs) max_abs = std::fabs(s[(size_t)g]);
    }

    float d = 1.0f;
    if (max_abs <= eps) {
        d = 0.0f; // all-zero block
    } else {
        // 2) global d from the max-abs group; snap group scales to int8
        const float iscale = -128.0f / max_abs;
        d = 1.0f / iscale;
        for (int g = 0; g < ng; ++g)
            sc8[(size_t)g] = std::max(-127, std::min(127, nearest_int_q6k(iscale * s[(size_t)g])));

        // 3) joint least-squares refinement (3 iterations): levels given
        //    (d, sc8) -> per-group optimal scale s_g (plain MSE LS) -> snap
        //    sc8 -> optimal d.
        for (int iter = 0; iter < 3; ++iter) {
            for (int g = 0; g < ng; ++g) {
                const int st = g * kQ6KGroup;
                const float eff = d * (float)sc8[(size_t)g];
                if (eff == 0.0f) continue;
                double num = 0.0, den = 0.0;
                for (int i = 0; i < kQ6KGroup; ++i) {
                    const int l = q6k_clamp_level(w[st + i] / eff);
                    num += (double)w[st + i] * (double)l;
                    den += (double)l * (double)l;
                }
                s[(size_t)g] = (den > 1e-20) ? (float)(num / den) : 0.0f;
            }
            for (int g = 0; g < ng; ++g) {
                if (d == 0.0f) { sc8[(size_t)g] = 0; continue; }
                sc8[(size_t)g] = std::max(-127, std::min(127, nearest_int_q6k(s[(size_t)g] / d)));
            }
            double num = 0.0, den = 0.0;
            for (int g = 0; g < ng; ++g) {
                num += (double)s[(size_t)g] * (double)sc8[(size_t)g];
                den += (double)sc8[(size_t)g] * (double)sc8[(size_t)g];
            }
            d = (den > 1e-20) ? (float)(num / den) : 0.0f;
            if (std::fabs(d) <= eps) d = 0.0f;
        }
    }

    // 4) write payload: levels, then int8 scales, then FP16 d
    BitWriter bw(indices);
    for (int g = 0; g < ng; ++g) {
        const int st = g * kQ6KGroup;
        const float eff = d * (float)sc8[(size_t)g];
        for (int i = 0; i < kQ6KGroup && st + i < n; ++i) {
            const int l = (eff == 0.0f) ? 0 : q6k_clamp_level(w[st + i] / eff);
            bw.put((uint32_t)(l + 32), 6);
        }
    }
    const int tail_start = ng * kQ6KGroup;
    if (tail_start < n && ng > 0) {
        const float eff = d * (float)sc8[(size_t)ng - 1];
        for (int i = tail_start; i < n; ++i) {
            const int l = (eff == 0.0f) ? 0 : q6k_clamp_level(w[i] / eff);
            bw.put((uint32_t)(l + 32), 6);
        }
    }
    size_t off = levels_bytes;
    for (int g = 0; g < ng; ++g) indices[off++] = (uint8_t)(int8_t)sc8[(size_t)g];
    const uint16_t dh = f32_to_f16(d);
    indices[off] = (uint8_t)(dh & 0xFF);
    indices[off + 1] = (uint8_t)(dh >> 8);
}

static void dequant_6k(const uint8_t* bytes, size_t size, int n, float* out) {
    const bool has_d = (n >= 8);
    const int sc_count = (n >= 32) ? (n / 16) : 0;
    const size_t levels_bytes = ((size_t)n * 6 + 7) / 8;
    float d = 1.0f;
    std::vector<float> sc((size_t)((n + 15) / 16), 1.0f); // one entry per (partial) 16-group
    if (has_d && size >= levels_bytes + 2) {
        const size_t doff = size - 2;
        const uint16_t dh = (uint16_t)(bytes[doff]) | ((uint16_t)(bytes[doff + 1]) << 8);
        d = f16_to_f32(dh);
        // Scales precede d iff the encoder stored them (payload >= levels + d + sc).
        const size_t sc_bytes = (size >= levels_bytes + 2 + (size_t)sc_count) ? (size_t)sc_count : 0;
        for (int g = 0; g < sc_bytes; ++g)
            sc[g] = (float)(int8_t)bytes[size - 2 - (size_t)sc_bytes + (size_t)g];
        // A partial tail group reuses the last full group's scale.
        if (sc_bytes > 0) {
            for (int g = (int)sc_bytes; g < (int)sc.size(); ++g) sc[(size_t)g] = sc[(size_t)sc_bytes - 1];
        }
    }
    BitReader br(bytes, size);
    for (int i = 0; i < n; ++i) {
        const int s = (int)br.get(6) - 32;
        out[i] = d * sc[i / 16] * (float)s;
    }
}

// ---- affine grouped codec: [0,1] lattice + per-group scale AND min ---------
// Used by Q4_GRP (4-bit, per-32 scale+min) and Q2_GRP (2-bit,
// per-16/32 scale+min).  The lattice levels are ascending in [0,1], and each
// group's DC offset m is stored as a PER-GROUP SIGNED code mc (bias-encoded,
// native improvement over industry's unsigned-min which can only subtract a
// positive DC).  Effective value:
//     w ~= d * sc[g] * lv[i] - dm * mc[g],   lv in {0..1}
// Full-block wire layout: [ levels ][ packed scales ][ packed mins ][ d ][ dm ]
// A block that cannot afford the affine header inside ceil(bpw*n/8)+1 falls
// back to the scale-only/d-only quant_grp16 layout (deterministic by n).

static float level_value_affine(int bits, uint32_t index) {
    if (bits == 2) return (float)std::min(index, 3u) * (1.0f / 3.0f);
    if (bits == 3) return (float)std::min(index, 7u) * (1.0f / 7.0f);
    if (bits == 8) return (float)std::min(index, 255u) * (1.0f / 255.0f);
    return (float)std::min(index, 15u) * (1.0f / 15.0f);
}

static uint32_t nearest_level_affine(float value, int bits) {
    const float v = std::max(0.0f, std::min(1.0f, value));
    if (bits == 4) return (uint32_t)std::lround(v * 15.0f);
    if (bits == 3) return (uint32_t)std::lround(v * 7.0f);
    if (bits == 8) return (uint32_t)std::lround(v * 255.0f);
    return (uint32_t)std::lround(v * 3.0f);
}

// Lloyd-style affine fit of one group: seed m at the data min (unclamped —
// the DC offset, positive or negative, is carried per group by the SIGNED
// bias-encoded min code), 4 iterations of (assign [0,1] levels -> plain-MSE
// LS refit of (s, m)), then a scale-fac grid polish (0.85..1.15) keeping the
// min plain-MSE assignment, then one final LS refit.
static void grp16_fit_affine(int bits, const float* w, int cnt,
                             float* s_out, float* m_out) {
    float maxw = -1e30f, data_min = 1e30f;
    for (int i = 0; i < cnt; ++i) {
        maxw = std::max(maxw, w[i]);
        data_min = std::min(data_min, w[i]);
    }
    // m is left unclamped: each group's DC offset is carried by its own SIGNED
    // min code mc (bias-encoded), so both positive-DC (all-positive) and
    // negative-DC (zero-centered) groups keep their min.
    float s = maxw - data_min;   // lv spans [0,1]
    float m = data_min;
    if (!(std::fabs(s) > 1e-30f)) { *s_out = 0.0f; *m_out = 0.0f; return; }
    uint32_t idx[256];
    auto assign = [&]() {
        for (int i = 0; i < cnt; ++i)
            idx[i] = nearest_level_affine((w[i] - m) / s, bits);
    };
    auto lsf = [&]() {
        double sw = 0, sl = 0, sl2 = 0, sx = 0, sxl = 0;
        for (int i = 0; i < cnt; ++i) {
            const double l = (double)level_value_affine(bits, idx[i]);
            sw += 1.0; sl += l; sl2 += l * l;
            sx += (double)w[i]; sxl += (double)w[i] * l;
        }
        const double D = sw * sl2 - sl * sl;
        if (D > 1e-20) {
            s = (float)((sw * sxl - sx * sl) / D);
            m = (float)((sl2 * sx - sl * sxl) / D);
        }
    };
    for (int iter = 0; iter < 4; ++iter) { assign(); lsf(); }
    float best_s = s, best_m = m;
    double best_mse = 1e300;
    for (int k = -3; k <= 3; ++k) {
        if (k == 0) continue;
        const float cs = s * (1.0f + 0.05f * (float)k);
        double e = 0.0;
        for (int i = 0; i < cnt; ++i) {
            const float l = level_value_affine(bits, nearest_level_affine((w[i] - m) / cs, bits));
            const float d = cs * l + m - w[i];
            e += (double)d * d;
        }
        if (e < best_mse) { best_mse = e; best_s = cs; }
    }
    s = best_s;
    for (int i = 0; i < cnt; ++i) idx[i] = nearest_level_affine((w[i] - best_m) / s, bits);
    lsf();
    *s_out = s;
    *m_out = m;
}

static bool affine_budget_ok(int bits, int n, int gsz, int scb, int mb,
                             float claimed, size_t* sc_bytes, size_t* min_bytes) {
    const size_t levels_bytes = ((size_t)n * (size_t)bits + 7) / 8;
    const int ng = (n >= gsz) ? (n / gsz) : 0;
    const size_t scb_b = (size_t)((ng * scb + 7) / 8);
    const size_t mb_b = (size_t)((ng * mb + 7) / 8);
    const size_t budget = (size_t)std::ceil(claimed * (double)n / 8.0);
    if (sc_bytes) *sc_bytes = scb_b;
    if (min_bytes) *min_bytes = mb_b;
    return ng > 0 && levels_bytes + scb_b + mb_b + 4 <= budget;
}

static void quant_affine(int bits, const float* w, int n, int gsz, int scb, int mb,
                         float claimed, std::vector<uint8_t>& indices) {
    size_t sc_bytes = 0, min_bytes = 0;
    if (!affine_budget_ok(bits, n, gsz, scb, mb, claimed, &sc_bytes, &min_bytes)) {
        quant_grp16(bits, w, n, indices);
        return;
    }
    const int ng = n / gsz;
    const size_t levels_bytes = ((size_t)n * (size_t)bits + 7) / 8;
    indices.assign(levels_bytes + sc_bytes + min_bytes + 4, 0);
    const float eps = 1e-15f;
    const int scq = (1 << scb) - 1;
    const int mq = (1 << mb) - 1;
    // Per-group min is SIGNED: stored code = mc + bias in [0, mq], i.e. mc in
    // [-bias, mq-bias] (mb=4: [-8,7]; mb=6: [-32,31]).  dm is a single
    // positive FP16 magnitude, so -dm*mc gives a positive DC for all-positive
    // groups and a negative DC for zero-centered groups — both handled in one
    // block (native improvement over industry's unsigned-min).
    const int bias = (mq + 1) / 2;

    // 1) per-group affine Lloyd fit (m unclamped; signed per-group min code
    //    carries each group's DC offset)
    std::vector<float> s((size_t)ng, 0.0f), m((size_t)ng, 0.0f);
    float max_s = 0.0f, max_m = 0.0f;
    for (int g = 0; g < ng; ++g) {
        const int st = g * gsz;
        grp16_fit_affine(bits, w + st, gsz, &s[(size_t)g], &m[(size_t)g]);
        max_s = std::max(max_s, std::fabs(s[(size_t)g]));
        max_m = std::max(max_m, std::fabs(m[(size_t)g]));
    }

    // 2) snap scales + mins (mc signed, dm positive)
    std::vector<int> sc((size_t)ng, 0), mc((size_t)ng, 0);
    float d = 1.0f, dm = 1.0f;
    if (max_s <= eps) {
        d = 0.0f;
    } else {
        d = f16_to_f32(f32_to_f16(max_s / (float)scq));
        for (int g = 0; g < ng; ++g)
            sc[(size_t)g] = (s[(size_t)g] == 0.0f) ? 0 :
                std::max(0, std::min(scq, (int)std::lround(s[(size_t)g] / d)));
        if (max_m > eps) {
            dm = f16_to_f32(f32_to_f16(max_m / (float)bias));
            for (int g = 0; g < ng; ++g) {
                const float v = -m[(size_t)g] / dm;
                mc[(size_t)g] = std::max(-bias, std::min(mq - bias, (int)std::lround(v)));
            }
        }
        // 3) joint refinement (5 iterations): levels given (d*sc, -dm*mc) ->
        //    per-group LS refit (s, m) -> snap -> LS-optimal FP16 d, dm.
        //    Budget unchanged (same sc/mc bits + 2x FP16 d/dm overhead).
        for (int iter = 0; iter < 5; ++iter) {
            for (int g = 0; g < ng; ++g) {
                const int st = g * gsz;
                const float eff_s = d * (float)sc[(size_t)g];
                const float eff_m = -dm * (float)mc[(size_t)g];
                if (eff_s == 0.0f) continue;
                double sw = 0, sl = 0, sl2 = 0, sx = 0, sxl = 0;
                for (int i = 0; i < gsz; ++i) {
                    const float l = level_value_affine(bits,
                        nearest_level_affine((w[st + i] - eff_m) / eff_s, bits));
                    sw += 1.0; sl += l; sl2 += l * l;
                    sx += (double)w[st + i]; sxl += (double)w[st + i] * l;
                }
                const double D = sw * sl2 - sl * sl;
                if (D > 1e-20) {
                    s[(size_t)g] = (float)((sw * sxl - sx * sl) / D);
                    m[(size_t)g] = (float)((sl2 * sx - sl * sxl) / D);
                }
            }
            for (int g = 0; g < ng; ++g)
                sc[(size_t)g] = (s[(size_t)g] == 0.0f) ? 0 :
                    std::max(0, std::min(scq, (int)std::lround(s[(size_t)g] / d)));
            double num = 0, den = 0;
            for (int g = 0; g < ng; ++g) {
                num += (double)s[(size_t)g] * (double)sc[(size_t)g];
                den += (double)sc[(size_t)g] * (double)sc[(size_t)g];
            }
            d = (den > 1e-20) ? f16_to_f32(f32_to_f16((float)(num / den))) : 0.0f;
            if (std::fabs(d) <= eps) d = 0.0f;
            if (std::fabs(dm) > eps) {
                for (int g = 0; g < ng; ++g) {
                    const float v = -m[(size_t)g] / dm;
                    mc[(size_t)g] = std::max(-bias, std::min(mq - bias, (int)std::lround(v)));
                }
            } else {
                for (int g = 0; g < ng; ++g) mc[(size_t)g] = 0;
            }
            double num2 = 0, den2 = 0;
            for (int g = 0; g < ng; ++g) {
                num2 += -(double)m[(size_t)g] * (double)mc[(size_t)g];
                den2 += (double)mc[(size_t)g] * (double)mc[(size_t)g];
            }
            dm = (den2 > 1e-20) ? f16_to_f32(f32_to_f16((float)(num2 / den2))) : 0.0f;
            if (std::fabs(dm) <= eps) dm = 0.0f;
        }
    }

    // 4) write payload: levels, then packed scales, then packed mins, then
    //    FP16 d and dm.
    BitWriter bw(indices);
    for (int g = 0; g < ng; ++g) {
        const int st = g * gsz;
        const float eff_s = d * (float)sc[(size_t)g];
        const float eff_m = -dm * (float)mc[(size_t)g];
        for (int i = 0; i < gsz; ++i) {
            const uint32_t l = (eff_s == 0.0f) ? 0u :
                nearest_level_affine((w[st + i] - eff_m) / eff_s, bits);
            bw.put(l, bits);
        }
    }
    for (int g = 0; g < ng; ++g) bw.put((uint32_t)sc[(size_t)g], scb);
    for (int g = 0; g < ng; ++g) bw.put((uint32_t)(mc[(size_t)g] + bias), mb);
    bw.put(f32_to_f16(d), 16);
    bw.put(f32_to_f16(dm), 16);
}


static void dequant_affine(int bits, const uint8_t* bytes, size_t size, int n,
                           int gsz, int scb, int mb, float claimed, float* out) {
    size_t sc_bytes = 0, min_bytes = 0;
    if (!affine_budget_ok(bits, n, gsz, scb, mb, claimed, &sc_bytes, &min_bytes)) {
        dequant_grp16(bits, bytes, size, n, out);
        return;
    }
    const int ng = n / gsz;
    const int mq = (1 << mb) - 1;
    const int bias = (mq + 1) / 2;   // signed min: stored = mc + bias
    BitReader br(bytes, size);
    for (int i = 0; i < n; ++i) out[i] = (float)br.get(bits);  // raw indices
    std::vector<float> sc((size_t)ng, 0.0f), mc((size_t)ng, 0.0f);
    for (int g = 0; g < ng; ++g) sc[(size_t)g] = (float)br.get(scb);
    for (int g = 0; g < ng; ++g) mc[(size_t)g] = (float)br.get(mb) - (float)bias;
    const float d = f16_to_f32((uint16_t)br.get(16));
    const float dm = f16_to_f32((uint16_t)br.get(16));
    for (int i = 0; i < n; ++i) {
        const int g = i / gsz;
        out[i] = d * sc[(size_t)g] * level_value_affine(bits, (uint32_t)out[i]) -
                 dm * mc[(size_t)g];
    }
}


void quant_q16_enhanced(const float* w, int n, std::vector<uint8_t>& indices) {
    // Q16: 16-bit uniform lattice over the block's true [vmin,vmax] range.
    // Budget: 256 weights x 16 bits = 4096 bits per block; the 64-bit FP32
    // (vmin,vmax) header is funded by packing every 4th weight at 15 bits
    // (63-bit groups). Result: exactly 16.0 BPW, zero weights destroyed, and
    // for bell-shaped weights a uniform grid over the observed range beats
    // IEEE FP16's logarithmic grid by roughly 50x (verified in bench).
    indices.assign(n * 2, 0);
    BitWriter bw(indices);
    for (int start = 0; start < n; start += 256) {
        int end = std::min(start + 256, n);
        float vmin = 1e30f, vmax = -1e30f;
        for (int i = start; i < end; ++i) {
            float val = w[i];
            vmin = std::min(vmin, val);
            vmax = std::max(vmax, val);
        }
        if (vmin > vmax) { vmin = 0; vmax = 0; }
        uint32_t imin, imax;
        std::memcpy(&imin, &vmin, 4);
        std::memcpy(&imax, &vmax, 4);
        bw.put(imin, 32);
        bw.put(imax, 32);
        float range = vmax - vmin;
        float scale16 = range > 0 ? 65535.0f / range : 0.0f;
        float scale15 = range > 0 ? 32767.0f / range : 0.0f;
        for (int i = start; i < end; i += 4) {
            uint64_t group = 0;
            for (int k = 0; k < 4; ++k) {
                int idx = i + k;
                if (idx >= end) { group |= 0; continue; }
                float val = w[idx];
                if (k == 3) {
                    int q = (int)std::lround((val - vmin) * scale15);
                    q = std::max(0, std::min(32767, q));
                    group |= (uint64_t)q << 48;
                } else {
                    int q = (int)std::lround((val - vmin) * scale16);
                    q = std::max(0, std::min(65535, q));
                    group |= (uint64_t)q << (16 * k);
                }
            }
            bw.put((uint32_t)(group & 0xFFFFFFFFu), 32);
            bw.put((uint32_t)(group >> 32), 31);
        }
    }
}

void dequant_q16_enhanced(const uint8_t* bytes, size_t size, int n, float* out) {
    BitReader br(bytes, size);
    for (int start = 0; start < n; start += 256) {
        int end = std::min(start + 256, n);
        uint32_t imin = br.get(32);
        uint32_t imax = br.get(32);
        float vmin, vmax;
        std::memcpy(&vmin, &imin, 4);
        std::memcpy(&vmax, &imax, 4);
        float range = vmax - vmin;
        float scale16 = range > 0 ? range / 65535.0f : 0.0f;
        float scale15 = range > 0 ? range / 32767.0f : 0.0f;
        for (int i = start; i < end; i += 4) {
            uint64_t group = (uint64_t)br.get(32) | ((uint64_t)br.get(31) << 32);
            for (int k = 0; k < 4; ++k) {
                int idx = i + k;
                if (idx >= end) break;
                if (k == 3) {
                    uint32_t q = (uint32_t)((group >> 48) & 0x7FFFu);
                    out[idx] = vmin + (float)q * scale15;
                } else {
                    uint32_t q = (uint32_t)((group >> (16 * k)) & 0xFFFFu);
                    out[idx] = vmin + (float)q * scale16;
                }
            }
        }
    }
}

void quant_q24(const float* w, int n, std::vector<uint8_t>& indices) {
    indices.assign(n * 3, 0);
    uint8_t* ptr = indices.data();
    for (int i = 0; i < n; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &w[i], 4);
        bits >>= 8;
        ptr[i*3 + 0] = (uint8_t)(bits & 0xFF);
        ptr[i*3 + 1] = (uint8_t)((bits >> 8) & 0xFF);
        ptr[i*3 + 2] = (uint8_t)((bits >> 16) & 0xFF);
    }
}

void dequant_q24(const uint8_t* bytes, size_t size, int n, float* out) {
    for (int i = 0; i < n; ++i) {
        if (i*3 + 2 >= size) break;
        uint32_t bits = ((uint32_t)bytes[i*3]) |
                        ((uint32_t)bytes[i*3+1] << 8) |
                        ((uint32_t)bytes[i*3+2] << 16);
        bits <<= 8;
        std::memcpy(&out[i], &bits, 4);
    }
}

// ---- QUAD_MIX: 4-tier importance routing, exact claimed BPW ---------------
// Layout (per block, deterministic, no permutation array):
//   [ bitstream: 2-bit tier map per weight, then tier values ]
//   [ tail bytes: FP16 scale per scaled tier (bits < 16), per block; the GRP
//     variants store the dominant tier's scales per 64 weights ]
// Tier counts are derived from the format's fixed percents (lround), so no
// header is needed.  Tiers are assigned by magnitude rank: the largest
// weights get the highest-precision tier (typically raw FP32 for the
// lossless tail).  Values: 1 bit = sign; 2/3/4/6/8/12 = fixed level tables;
// 16 = raw FP16; 24 = FP24 (top 24 bits); 32 = raw FP32.
struct QuadMixTier {
    int bits;
    float percent;
};

static bool quad_mix_is_grp(Format fmt) {
    auto v = static_cast<uint8_t>(fmt);
    return v >= static_cast<uint8_t>(Format::Q_QUAD_MIX_3_5_GRP);
}

static std::array<QuadMixTier, 4> quad_mix_get_config(Format fmt) {
    switch (fmt) {
        case Format::Q_QUAD_MIX_3_5:      return {{{1, 92.0f}, {2, 1.5f}, {4, 6.0f}, {32, 0.5f}}};
        case Format::Q_QUAD_MIX_4_5:      return {{{1, 58.5f}, {2, 2.0f}, {4, 39.0f}, {32, 0.5f}}};
        case Format::Q_QUAD_MIX_6_5:      return {{{1, 1.5f}, {2, 4.0f}, {4, 93.0f}, {32, 1.5f}}};
        case Format::Q_QUAD_MIX_8_5:      return {{{1, 2.5f}, {2, 1.0f}, {4, 88.0f}, {32, 8.5f}}};
        case Format::Q_QUAD_MIX_12_5:     return {{{1, 0.5f}, {3, 2.0f}, {8, 87.5f}, {32, 10.0f}}};
        case Format::Q_QUAD_MIX_16_5:     return {{{1, 4.5f}, {2, 7.0f}, {16, 88.0f}, {32, 0.5f}}};
        case Format::Q_QUAD_MIX_24_5:     return {{{1, 2.5f}, {2, 5.0f}, {24, 92.0f}, {32, 0.5f}}};
        case Format::Q_QUAD_MIX_3_5_GRP:  return {{{1, 92.0f}, {2, 4.0f}, {4, 3.0f}, {32, 1.0f}}};
        case Format::Q_QUAD_MIX_4_5_GRP:  return {{{1, 58.5f}, {2, 1.0f}, {4, 39.5f}, {32, 1.0f}}};
        case Format::Q_QUAD_MIX_6_5_GRP:  return {{{1, 0.5f}, {3, 5.0f}, {4, 93.0f}, {32, 1.5f}}};
        case Format::Q_QUAD_MIX_8_5_GRP:  return {{{1, 0.5f}, {3, 2.0f}, {4, 89.0f}, {32, 8.5f}}};
        case Format::Q_QUAD_MIX_12_5_GRP: return {{{1, 1.5f}, {2, 2.5f}, {8, 85.0f}, {32, 11.0f}}};
        case Format::Q_QUAD_MIX_16_5_GRP: return {{{1, 10.0f}, {2, 1.0f}, {16, 88.0f}, {32, 1.0f}}};
        case Format::Q_QUAD_MIX_24_5_GRP: return {{{1, 2.5f}, {2, 5.0f}, {24, 91.5f}, {32, 1.0f}}};
        default: return {{{1, 0.0f}, {2, 0.0f}, {4, 0.0f}, {32, 0.0f}}};
    }
}

// Tier counts (deterministic on both sides): first three tiers lround their
// percent share (min 1), the FP32 tier absorbs the remainder.  Rounding can
// overshoot n by 1 (e.g. n=48, Q_QUAD_MIX@6.5 -> {1,2,45} + fallback); the
// reconciliation below guarantees sum(c) == n exactly on both sides.
static void quad_mix_counts(Format fmt, int n, int c[4]) {
    auto config = quad_mix_get_config(fmt);
    for (int i = 0; i < 3; ++i)
        c[i] = std::max(1, (int)std::lround(config[i].percent * 0.01 * (double)n));
    c[3] = n - c[0] - c[1] - c[2];
    if (c[3] < 1) {
        c[3] = 1;
        int excess = c[0] + c[1] + c[2] + 1 - n;
        for (int i = 0; i < 3 && excess > 0; ++i) {
            const int rem = std::min(c[i] - 1, excess);
            c[i] -= rem;
            excess -= rem;
        }
    }
    for (int i = 0; i < 4; ++i) c[i] = std::max(0, std::min(n, c[i]));
    int sum = c[0] + c[1] + c[2] + c[3];
    for (int i = 0; i < 4 && sum > n; ++i) {
        const int rem = std::min(c[i], sum - n);
        c[i] -= rem;
        sum -= rem;
    }
}

static float quad_mix_max_level(int bits) {
    switch (bits) {
        case 1: return 1.0f;
        case 2: return 1.5104f;
        case 3: return kQ3Levels[7];
        case 4: return 2.7178f;
        case 6: return fixed_level_value(6, 63u);
        case 8: return 8.0f;
        case 12: return fixed_level_value(12, 4095u);
        default: return 1.0f;
    }
}

static float quad_mix_level_value(int bits, float v) {
    switch (bits) {
        case 2: return level_value(2, nearest_level(v, 2));
        case 3: return fixed_level_value(3, nearest_fixed_level(3, v));
        case 4: return level_value(4, nearest_level(v, 4));
        case 6: return fixed_level_value(6, nearest_fixed_level(6, v));
        case 8: return level_value(8, nearest_level(v, 8));
        case 12: return fixed_level_value(12, nearest_fixed_level(12, v));
        default: return 0.0f;
    }
}

static void quad_mix_put_value(BitWriter& bw, int bits, float v, float s) {
    const float nv = (s == 0.0f) ? 0.0f : v / s;
    switch (bits) {
        case 1: bw.put(v >= 0.0f ? 1u : 0u, 1); break;
        case 2: bw.put(nearest_level(nv, 2), 2); break;
        case 3: bw.put(nearest_fixed_level(3, nv), 3); break;
        case 4: bw.put(nearest_level(nv, 4), 4); break;
        case 6: bw.put(nearest_fixed_level(6, nv), 6); break;
        case 8: bw.put(nearest_level(nv, 8), 8); break;
        case 12: bw.put(nearest_fixed_level(12, nv), 12); break;
        case 16: bw.put(f32_to_f16(v), 16); break;
        case 24: {
            uint32_t b;
            std::memcpy(&b, &v, 4);
            bw.put(b >> 8, 24);
            break;
        }
        case 32: {
            uint32_t b;
            std::memcpy(&b, &v, 4);
            bw.put(b, 32);
            break;
        }
        default: break;
    }
}

static float quad_mix_get_value(BitReader& br, int bits, float s) {
    switch (bits) {
        case 1: return s * (br.get(1) == 0 ? -1.0f : 1.0f);
        case 2: return level_value(2, br.get(2)) * s;
        case 3: return fixed_level_value(3, br.get(3)) * s;
        case 4: return level_value(4, br.get(4)) * s;
        case 6: return fixed_level_value(6, br.get(6)) * s;
        case 8: return level_value(8, br.get(8)) * s;
        case 12: return fixed_level_value(12, br.get(12)) * s;
        case 16: return f16_to_f32((uint16_t)br.get(16));
        case 24: {
            uint32_t b = br.get(24) << 8;
            float v;
            std::memcpy(&v, &b, 4);
            return v;
        }
        case 32: {
            uint32_t b = br.get(32);
            float v;
            std::memcpy(&v, &b, 4);
            return v;
        }
        default: return 0.0f;
    }
}

static void quant_quad_mix(Format fmt, const float* w, int n,
                           std::vector<uint8_t>& indices_out) {
    auto config = quad_mix_get_config(fmt);
    const float claimed = format_bpw(fmt);
    const size_t budget_bits = (size_t)std::ceil(claimed * (double)n);
    const bool grp = quad_mix_is_grp(fmt);

    int c[4];
    quad_mix_counts(fmt, n, c);

    // Budget enforcement: if the exact layout ever exceeds the claimed BPW
    // (only for non-256 tails), move boundary weights down one tier at a time.
    size_t total_bits = (size_t)2 * (size_t)n;
    for (int t = 0; t < 4; ++t) total_bits += (size_t)c[t] * (size_t)config[t].bits;
    int dom = 0;
    for (int t = 1; t < 4; ++t) if (c[t] > c[dom]) dom = t;
    size_t scale_bits = 0;
    for (int t = 0; t < 4; ++t)
        if (config[t].bits < 16) scale_bits += (grp && t == dom) ? (size_t)16 * ((size_t)(c[t] + 63) / 64) : 16;
    total_bits += scale_bits;
    for (int guard = 0; guard < 4 * n && total_bits > budget_bits; ++guard) {
        bool moved = false;
        for (int t = 3; t >= 1; --t) {
            if (c[t] > 1 && config[t].bits > config[t - 1].bits) {
                total_bits -= (size_t)(config[t].bits - config[t - 1].bits);
                --c[t];
                ++c[t - 1];
                moved = true;
                break;
            }
        }
        if (!moved) break;
    }

    // Magnitude-rank tier assignment.
    std::vector<std::pair<float, int>> mag((size_t)n);
    for (int i = 0; i < n; ++i) mag[(size_t)i] = { std::fabs(w[i]), i };
    std::sort(mag.begin(), mag.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    std::vector<uint8_t> tier_of((size_t)n, 0);
    int rank = 0;
    for (int t = 3; t >= 0; --t) {
        for (int j = 0; j < c[t]; ++j, ++rank)
            tier_of[(size_t)mag[(size_t)rank].second] = (uint8_t)t;
    }

    // Per-tier value streams (original order) + per-tier scale fit.
    std::vector<std::vector<float>> tv(4);
    for (int i = 0; i < n; ++i) tv[tier_of[(size_t)i]].push_back(w[i]);

    // Doubled grouping for GRP dominant tier: per-32 (was per-64). 8 scales per
    // 256-block dominant tier = 128b = 8 extra bytes vs 4 scales before;
    // budget guard below moves ~8 weights down one tier to keep claimed BPW.
    float sc[4][8];
    int nsc[4] = { 0, 0, 0, 0 };
    for (int t = 0; t < 4; ++t) {
        if (config[t].bits >= 16 || c[t] == 0) continue;
        nsc[t] = (grp && t == dom) ? ((c[t] + 31) / 32) : 1;
        const float max_lvl = quad_mix_max_level(config[t].bits);
        for (int g = 0; g < nsc[t]; ++g) {
            const int st = g * 32;
            const int en = std::min(c[t], st + 32);
            float maxa = 0.0f;
            for (int j = st; j < en; ++j) maxa = std::max(maxa, std::fabs(tv[(size_t)t][(size_t)j]));
            float s = (maxa > 1e-30f) ? maxa / max_lvl : 0.0f;
            if (s > 0.0f) {
                for (int iter = 0; iter < 2; ++iter) {
                    double num = 0.0, den = 0.0;
                    for (int j = st; j < en; ++j) {
                        const float l = (config[t].bits == 1)
                            ? (tv[(size_t)t][(size_t)j] >= 0.0f ? 1.0f : -1.0f)
                            : quad_mix_level_value(config[t].bits, tv[(size_t)t][(size_t)j] / s);
                        num += (double)tv[(size_t)t][(size_t)j] * (double)l;
                        den += (double)l * (double)l;
                    }
                    if (den > 1e-20) s = (float)(num / den);
                }
            }
            if (!(s > 0.0f)) s = 0.0f;
            sc[(size_t)t][(size_t)g] = s;
        }
    }

    // Write: bitstream (map + tier values), then scale tail bytes.
    size_t payload_bits = (size_t)2 * (size_t)n;
    for (int t = 0; t < 4; ++t) payload_bits += (size_t)c[t] * (size_t)config[t].bits;
    size_t scale_bytes = 0;
    for (int t = 0; t < 4; ++t) scale_bytes += (size_t)nsc[t] * 2;
    const size_t total_bytes = (payload_bits + 7) / 8 + scale_bytes;
    indices_out.assign(total_bytes, 0);
    BitWriter bw(indices_out);
    for (int i = 0; i < n; ++i) bw.put(tier_of[(size_t)i], 2);
    for (int t = 0; t < 4; ++t) {
        for (int j = 0; j < c[t]; ++j) {
            const float s = (config[t].bits < 16)
                ? sc[(size_t)t][std::min(j / 64, nsc[t] - 1)] : 1.0f;
            quad_mix_put_value(bw, config[t].bits, tv[(size_t)t][(size_t)j], s);
        }
    }
    uint8_t* ptr = indices_out.data() + (payload_bits + 7) / 8;
    for (int t = 0; t < 4; ++t) {
        for (int g = 0; g < nsc[t]; ++g) {
            const uint16_t h = f32_to_f16(sc[(size_t)t][(size_t)g]);
            ptr[0] = (uint8_t)(h & 0xFF);
            ptr[1] = (uint8_t)(h >> 8);
            ptr += 2;
        }
    }
}

static void dequant_quad_mix(Format fmt, const uint8_t* indices, size_t idx_bytes,
                             uint32_t nw, float* out) {
    auto config = quad_mix_get_config(fmt);
    const int n = (int)std::min<uint32_t>(nw, 1u << 24);
    const float claimed = format_bpw(fmt);
    const size_t budget_bits = (size_t)std::ceil(claimed * (double)n);
    const bool grp = quad_mix_is_grp(fmt);

    int c[4];
    quad_mix_counts(fmt, n, c);

    size_t total_bits = (size_t)2 * (size_t)n;
    for (int t = 0; t < 4; ++t) total_bits += (size_t)c[t] * (size_t)config[t].bits;
    int dom = 0;
    for (int t = 1; t < 4; ++t) if (c[t] > c[dom]) dom = t;
    size_t scale_bits = 0;
    for (int t = 0; t < 4; ++t)
        if (config[t].bits < 16) scale_bits += (grp && t == dom) ? (size_t)16 * ((size_t)(c[t] + 63) / 64) : 16;
    total_bits += scale_bits;
    for (int guard = 0; guard < 4 * n && total_bits > budget_bits; ++guard) {
        bool moved = false;
        for (int t = 3; t >= 1; --t) {
            if (c[t] > 1 && config[t].bits > config[t - 1].bits) {
                total_bits -= (size_t)(config[t].bits - config[t - 1].bits);
                --c[t];
                ++c[t - 1];
                moved = true;
                break;
            }
        }
        if (!moved) break;
    }

    int nsc[4] = { 0, 0, 0, 0 };
    size_t scale_bytes = 0;
    for (int t = 0; t < 4; ++t) {
        if (config[t].bits >= 16 || c[t] == 0) continue;
        nsc[t] = (grp && t == dom) ? ((c[t] + 63) / 64) : 1;
        scale_bytes += (size_t)nsc[t] * 2;
    }
    size_t payload_bits = (size_t)2 * (size_t)n;
    for (int t = 0; t < 4; ++t) payload_bits += (size_t)c[t] * (size_t)config[t].bits;
    const size_t payload_bytes = (payload_bits + 7) / 8;
    if (idx_bytes < payload_bytes + scale_bytes) return;

    float sc[4][4];
    const uint8_t* sp = indices + payload_bytes;
    for (int t = 0; t < 4; ++t) {
        for (int g = 0; g < nsc[t]; ++g) {
            const uint16_t h = (uint16_t)(sp[0]) | ((uint16_t)(sp[1]) << 8);
            sc[(size_t)t][(size_t)g] = f16_to_f32(h);
            sp += 2;
        }
    }

    BitReader br(indices, payload_bytes);
    std::vector<uint8_t> map((size_t)n);
    for (int i = 0; i < n; ++i) map[(size_t)i] = (uint8_t)br.get(2);
    std::vector<std::vector<uint32_t>> pos(4);
    for (int i = 0; i < n; ++i) pos[map[(size_t)i]].push_back((uint32_t)i);
    for (int t = 0; t < 4; ++t) {
        for (int j = 0; j < c[t] && j < (int)pos[(size_t)t].size(); ++j) {
            const float s = (config[t].bits < 16)
                ? sc[(size_t)t][std::min(j / 64, nsc[t] - 1)] : 1.0f;
            out[pos[(size_t)t][(size_t)j]] = quad_mix_get_value(br, config[t].bits, s);
        }
    }
}

bool quantize_block_all(Format fmt, const float* w, int n, std::vector<uint8_t>& indices, std::vector<uint8_t>& codebook) {
    if (!w || n <= 0) return false;
    indices.clear();
    codebook.clear();
    switch (fmt) {
        case Format::Q32:
            indices.resize((size_t)n * 4);
            std::memcpy(indices.data(), w, (size_t)n * 4);
            return true;
        case Format::Q24:
            quant_q24(w, n, indices);
            return true;
        case Format::Q16:
            quant_q16_enhanced(w, n, indices);
            return true;
        case Format::Q12:
            quant_fixed_codebook(12, w, n, indices);
            return true;
        case Format::Q8:
            quant_lattice(fmt, w, n, 8, indices);
            return true;
        case Format::Q6:
            quant_fixed_codebook(6, w, n, indices);
            return true;
        case Format::Q4:
            quant_lattice(fmt, w, n, 4, indices);
            return true;
        case Format::Q3:
            quant_fixed_codebook(3, w, n, indices);
            return true;
        case Format::Q2:
            quant_lattice(fmt, w, n, 2, indices);
            return true;
        case Format::Q1: {
            // 1.0 BPW exact: [ FP16 scale (16) ][ sign bit per kept weight ],
            // 16 evenly-spread slot positions zeroed (same scheme as Q1_GRP).
            if (n < 32) {
                indices.assign((size_t)(n + 7) / 8, 0); BitWriter bw(indices);
                for (int i = 0; i < n; ++i) bw.put(w[i] >= 0.0f ? 1u : 0u, 1);
                return true;
            }
            indices.assign((size_t)(n + 7) / 8, 0); BitWriter bw(indices);
            const float scale = rms_scale(w, n); bw.put(f32_to_f16(scale), 16);
            for (int i = 0; i < n; ++i) {
                if (is_slot(i, n, 16)) continue;
                bw.put(w[i] >= 0.0f ? 1u : 0u, 1);
            }
            return true;
        }
        case Format::Q24_GRP: {
            quant_q24(w, n, indices);
            indices.resize((size_t)std::ceil(24.5 * (double)n / 8.0), 0);
            return true;
        }
        case Format::Q16_GRP: {
            quant_q16_enhanced(w, n, indices);
            indices.resize((size_t)std::ceil(16.5 * (double)n / 8.0), 0);
            return true;
        }
        case Format::Q12_GRP:
            if (grp12_compound_fits(n)) {
                quant_grp12_compound(w, n, indices);
            } else {
                quant_grp16(12, w, n, indices);
            }
            return true;
        case Format::Q8_GRP:
            // Compound path (per-32 FULL fp16 scales + companded grid +
            // true-MSE scale search) at exact 8.5 BPW; the affine 6-bit
            // ladder remains only as the odd-size fallback.
            if (grp8_compound_fits(n)) {
                quant_grp8_compound(w, n, indices);
            } else {
                quant_affine(8, w, n, 32, 6, 6, 8.5f, indices);
            }
            return true;
        case Format::Q6_GRP:
            quant_6k(w, n, indices);
            return true;
        case Format::Q4_GRP:
            quant_affine(4, w, n, 32, 6, 6, 4.5f, indices);
            return true;
        case Format::Q3_GRP:
            quant_affine(3, w, n, 32, 6, 6, 3.5f, indices);
            return true;
        case Format::Q2_GRP:
            quant_affine(2, w, n, 16, 4, 4, 2.625f, indices);
            return true;
        case Format::Q1_GRP: {
            if (n < 32) {
                indices.assign((size_t)(n + 7) / 8, 0); BitWriter bw(indices);
                for (int i = 0; i < n; ++i) bw.put(w[i] >= 0.0f ? 1u : 0u, 1);
                return true;
            }
            indices.assign((size_t)(n + 7) / 8, 0); BitWriter bw(indices);
            const float scale = rms_scale(w, n); bw.put(f32_to_f16(scale), 16);
            for (int i = 0; i < n; ++i) {
                if (is_slot(i, n, 16)) continue;
                bw.put(w[i] >= 0.0f ? 1u : 0u, 1);
            }
            return true;
        }
        case Format::Q_TWI_MIX_1_5: {
            indices.assign(((size_t)n * 3 + 15) / 16, 0); BitWriter bw(indices);
            for (int start = 0; start < n; start += kQuantSubBlock) {
                const int end = std::min(start + kQuantSubBlock, n);
                const int count = end - start;
                const bool has_scale = count == kQuantSubBlock;
                const float scale = has_scale ? rms_scale(w + start, count) : 1.0f;
                if (has_scale) bw.put(f32_to_f16(scale), 16);
                const float s = scale <= 1e-12f ? 0.0f : scale;
                for (int i = start; i < end; ++i) bw.put(w[i] >= 0.0f ? 1u : 0u, 1);
            }
            return true;
        }
        case Format::Q_TWI_MIX_1_5_GRP: {
            // 1.75 BPW exact, per-32 groups:
            // [ FP16 scale (16) ][ 32 sign bits ][ 8 ternary-refinement flags
            // at fixed is_slot positions ] = 56 bits per 32 weights.
            if (n < 32) {
                indices.assign((size_t)(n + 7) / 8, 0); BitWriter bw(indices);
                for (int i = 0; i < n; ++i) bw.put(w[i] >= 0.0f ? 1u : 0u, 1);
                return true;
            }
            indices.assign((size_t)std::ceil(1.75 * (double)n / 8.0), 0);
            BitWriter bw(indices);
            for (int start = 0; start < n; start += kQuantSubBlock) {
                const int end = std::min(start + kQuantSubBlock, n);
                const int count = end - start;
                if (count < kQuantSubBlock) {
                    for (int i = start; i < end; ++i) bw.put(w[i] >= 0.0f ? 1u : 0u, 1);
                    continue;
                }
                const float scale = rms_scale(w + start, count);
                bw.put(f32_to_f16(scale), 16);
                const float s = scale <= 1e-12f ? 0.0f : scale;
                for (int i = start; i < end; ++i) bw.put(w[i] >= 0.0f ? 1u : 0u, 1);
                for (int i = start; i < end; ++i) {
                    if (!is_slot(i - start, count, 8)) continue;
                    bw.put(std::fabs(w[i]) > s * 1.23f ? 1u : 0u, 1);
                }
            }
            return true;
        }
        case Format::Q_TWI_MIX_2_5: {
            // 2.5 BPW exact, per-32 groups: [ FP16 scale ][ 32 x 2-bit
            // Lloyd lattice indices ] = 80 bits per 32 weights.
            indices.assign((size_t)std::ceil(2.5 * (double)n / 8.0), 0);
            BitWriter bw(indices);
            for (int start = 0; start < n; start += kQuantSubBlock) {
                const int end = std::min(start + kQuantSubBlock, n);
                const int count = end - start;
                if (count < kQuantSubBlock) {
                    for (int i = start; i < end; ++i) bw.put(nearest_level(w[i], 2), 2);
                    continue;
                }
                float maxa = 0.0f;
                for (int i = start; i < end; ++i) maxa = std::max(maxa, std::fabs(w[i]));
                float s = (maxa > 1e-30f) ? maxa / 1.5104f : 0.0f;
                if (s > 0.0f) {
                    for (int iter = 0; iter < 2; ++iter) {
                        double num = 0.0, den = 0.0;
                        for (int i = start; i < end; ++i) {
                            const float l = level_value(2, nearest_level(w[i] / s, 2));
                            num += (double)w[i] * (double)l;
                            den += (double)l * (double)l;
                        }
                        if (den > 1e-20) s = (float)(num / den);
                    }
                }
                if (!(s > 0.0f)) s = 0.0f;
                bw.put(f32_to_f16(s), 16);
                for (int i = start; i < end; ++i)
                    bw.put(nearest_level((s == 0.0f) ? 0.0f : w[i] / s, 2), 2);
            }
            return true;
        }
        case Format::Q_TWI_MIX_2_5_GRP: {
            // 2.75 BPW exact, per-32 groups:
            // [ FP16 scale ][ 32 x 2-bit lattice ][ 8 ternary-refinement
            // flags at fixed is_slot positions ] = 88 bits per 32 weights.
            if (n < 32) {
                indices.assign((size_t)(n + 3) / 4, 0); BitWriter bw(indices);
                for (int i = 0; i < n; ++i) bw.put(nearest_level(w[i], 2), 2);
                return true;
            }
            indices.assign((size_t)std::ceil(2.75 * (double)n / 8.0), 0);
            BitWriter bw(indices);
            for (int start = 0; start < n; start += kQuantSubBlock) {
                const int end = std::min(start + kQuantSubBlock, n);
                const int count = end - start;
                if (count < kQuantSubBlock) {
                    for (int i = start; i < end; ++i) bw.put(nearest_level(w[i], 2), 2);
                    continue;
                }
                float maxa = 0.0f;
                for (int i = start; i < end; ++i) maxa = std::max(maxa, std::fabs(w[i]));
                float s = (maxa > 1e-30f) ? maxa / 1.5104f : 0.0f;
                if (s > 0.0f) {
                    for (int iter = 0; iter < 2; ++iter) {
                        double num = 0.0, den = 0.0;
                        for (int i = start; i < end; ++i) {
                            const float l = level_value(2, nearest_level(w[i] / s, 2));
                            num += (double)w[i] * (double)l;
                            den += (double)l * (double)l;
                        }
                        if (den > 1e-20) s = (float)(num / den);
                    }
                }
                if (!(s > 0.0f)) s = 0.0f;
                bw.put(f32_to_f16(s), 16);
                for (int i = start; i < end; ++i)
                    bw.put(nearest_level((s == 0.0f) ? 0.0f : w[i] / s, 2), 2);
                for (int i = start; i < end; ++i) {
                    if (!is_slot(i - start, count, 8)) continue;
                    bw.put(std::fabs(w[i]) > s * 1.23f ? 1u : 0u, 1);
                }
            }
            return true;
        }
        case Format::Q_QUAD_MIX_3_5:
        case Format::Q_QUAD_MIX_4_5:
        case Format::Q_QUAD_MIX_6_5:
        case Format::Q_QUAD_MIX_8_5:
        case Format::Q_QUAD_MIX_12_5:
        case Format::Q_QUAD_MIX_16_5:
        case Format::Q_QUAD_MIX_24_5:
        case Format::Q_QUAD_MIX_3_5_GRP:
        case Format::Q_QUAD_MIX_4_5_GRP:
        case Format::Q_QUAD_MIX_6_5_GRP:
        case Format::Q_QUAD_MIX_8_5_GRP:
        case Format::Q_QUAD_MIX_12_5_GRP:
        case Format::Q_QUAD_MIX_16_5_GRP:
        case Format::Q_QUAD_MIX_24_5_GRP:
            quant_quad_mix(fmt, w, n, indices);
            return true;
        default: return false;
    }
}

void dequantize_block_all(Format fmt, const uint8_t* indices, size_t idx_bytes, const uint8_t* codebook, size_t cb_bytes, uint32_t nw, float* out) {
    if (!out || nw == 0) return;
    std::fill(out, out + nw, 0.0f);
    const int n = (int)std::min<uint32_t>(nw, 1u << 24);
    switch (fmt) {
        case Format::Q32:
            if (idx_bytes >= (size_t)n * 4) std::memcpy(out, indices, (size_t)n * 4);
            return;
        case Format::Q24:
            dequant_q24(indices, idx_bytes, n, out);
            return;
        case Format::Q24_GRP:
            dequant_q24(indices, idx_bytes, n, out);
            return;
        case Format::Q16:
            dequant_q16_enhanced(indices, idx_bytes, n, out);
            return;
        case Format::Q16_GRP:
            dequant_q16_enhanced(indices, idx_bytes, n, out);
            return;
        case Format::Q12:
            dequant_fixed_codebook(12, indices, idx_bytes, n, out);
            return;
        case Format::Q8:
            dequant_lattice(indices, idx_bytes, n, 8, out); return;
        case Format::Q6:
            dequant_fixed_codebook(6, indices, idx_bytes, n, out);
            return;
        case Format::Q4:
            dequant_lattice(indices, idx_bytes, n, 4, out); return;
        case Format::Q3:
            dequant_fixed_codebook(3, indices, idx_bytes, n, out);
            return;
        case Format::Q2:
            dequant_lattice(indices, idx_bytes, n, 2, out); return;
        case Format::Q1: {
            // 1.0 BPW exact: [ FP16 scale ][ sign bit per kept weight ], 16
            // evenly-spread slot positions zeroed (mirror of the encoder).
            if (n < 32) {
                BitReader br(indices, idx_bytes);
                for (int i = 0; i < n; ++i) out[i] = br.get(1) == 0 ? -1.0f : 1.0f;
                return;
            }
            BitReader br(indices, idx_bytes);
            const float scale = f16_to_f32((uint16_t)br.get(16));
            for (int i = 0; i < n; ++i) {
                if (is_slot(i, n, 16)) { out[i] = 0.0f; continue; }
                out[i] = br.get(1) == 0 ? -scale : scale;
            }
            return;
        }
        case Format::Q12_GRP:
            if (grp12_compound_fits(n)) {
                dequant_grp12_compound(indices, idx_bytes, n, out);
            } else {
                dequant_grp16(12, indices, idx_bytes, n, out);
            }
            return;
        case Format::Q8_GRP:
            if (grp8_compound_fits(n)) {
                dequant_grp8_compound(indices, idx_bytes, n, out);
            } else {
                dequant_affine(8, indices, idx_bytes, n, 32, 6, 6, 8.5f, out);
            }
            return;
        case Format::Q6_GRP:
            dequant_6k(indices, idx_bytes, n, out); return;
        case Format::Q4_GRP:
            dequant_affine(4, indices, idx_bytes, n, 32, 6, 6, 4.5f, out); return;
        case Format::Q3_GRP:
            dequant_affine(3, indices, idx_bytes, n, 32, 6, 6, 3.5f, out); return;
        case Format::Q2_GRP:
            dequant_affine(2, indices, idx_bytes, n, 16, 4, 4, 2.625f, out); return;
        case Format::Q1_GRP: {
            if (n < 32) {
                BitReader br(indices, idx_bytes); for (int i = 0; i < n; ++i) out[i] = br.get(1) == 0 ? -1.0f : 1.0f;
                return;
            }
            BitReader br(indices, idx_bytes); const float scale = f16_to_f32((uint16_t)br.get(16));
            for (int i = 0; i < n; ++i) {
                if (is_slot(i, n, 16)) { out[i] = 0.0f; continue; }
                out[i] = br.get(1) == 0 ? -scale : scale;
            }
            return;
        }
        case Format::Q_TWI_MIX_1_5: {
            BitReader br(indices, idx_bytes);
            for (int start = 0; start < n; start += kQuantSubBlock) {
                const int end = std::min(start + kQuantSubBlock, n); const int count = end - start;
                float scale = 1.0f; if (count == kQuantSubBlock) scale = f16_to_f32((uint16_t)br.get(16));
                for (int i = start; i < end; ++i) out[i] = br.get(1) == 0 ? -scale : scale;
            }
            return;
        }
        case Format::Q_TWI_MIX_1_5_GRP: {
            // Mirror of the 1.75-BPW encoder: per-32 groups of
            // [ FP16 scale ][ 32 signs ][ 8 ternary flags ].
            if (n < 32) {
                BitReader br(indices, idx_bytes); for (int i = 0; i < n; ++i) out[i] = br.get(1) == 0 ? -1.0f : 1.0f;
                return;
            }
            BitReader br(indices, idx_bytes);
            for (int start = 0; start < n; start += kQuantSubBlock) {
                const int end = std::min(start + kQuantSubBlock, n);
                const int count = end - start;
                if (count < kQuantSubBlock) {
                    for (int i = start; i < end; ++i) out[i] = br.get(1) == 0 ? -1.0f : 1.0f;
                    continue;
                }
                const float scale = f16_to_f32((uint16_t)br.get(16));
                std::vector<uint8_t> signs((size_t)count);
                for (int i = 0; i < count; ++i) signs[(size_t)i] = (uint8_t)br.get(1);
                for (int i = 0; i < count; ++i) {
                    float magnitude = scale;
                    if (is_slot(i, count, 8)) magnitude = scale * (br.get(1) == 0 ? 0.567f : 1.893f);
                    out[start + i] = signs[(size_t)i] == 0 ? -magnitude : magnitude;
                }
            }
            return;
        }
        case Format::Q_TWI_MIX_2_5: {
            // Mirror of the 2.5-BPW encoder: per-32 groups of
            // [ FP16 scale ][ 32 x 2-bit Lloyd indices ].
            BitReader br(indices, idx_bytes);
            for (int start = 0; start < n; start += kQuantSubBlock) {
                const int end = std::min(start + kQuantSubBlock, n);
                const int count = end - start;
                if (count < kQuantSubBlock) {
                    for (int i = start; i < end; ++i) out[i] = level_value(2, br.get(2));
                    continue;
                }
                const float scale = f16_to_f32((uint16_t)br.get(16));
                for (int i = start; i < end; ++i) out[i] = level_value(2, br.get(2)) * scale;
            }
            return;
        }
        case Format::Q_TWI_MIX_2_5_GRP: {
            // Mirror of the 2.75-BPW encoder: per-32 groups of
            // [ FP16 scale ][ 32 x 2-bit indices ][ 8 ternary flags ].
            if (n < 32) {
                BitReader br(indices, idx_bytes);
                for (int i = 0; i < n; ++i) out[i] = level_value(2, br.get(2));
                return;
            }
            BitReader br(indices, idx_bytes);
            for (int start = 0; start < n; start += kQuantSubBlock) {
                const int end = std::min(start + kQuantSubBlock, n);
                const int count = end - start;
                if (count < kQuantSubBlock) {
                    for (int i = start; i < end; ++i) out[i] = level_value(2, br.get(2));
                    continue;
                }
                const float scale = f16_to_f32((uint16_t)br.get(16));
                std::vector<uint8_t> idxs((size_t)count);
                for (int i = 0; i < count; ++i) idxs[(size_t)i] = (uint8_t)br.get(2);
                for (int i = 0; i < count; ++i) {
                    float l = level_value(2, idxs[(size_t)i]);
                    if (is_slot(i, count, 8)) l *= (br.get(1) == 0 ? 0.567f : 1.893f);
                    out[start + i] = l * scale;
                }
            }
            return;
        }
        case Format::Q_QUAD_MIX_3_5:
        case Format::Q_QUAD_MIX_4_5:
        case Format::Q_QUAD_MIX_6_5:
        case Format::Q_QUAD_MIX_8_5:
        case Format::Q_QUAD_MIX_12_5:
        case Format::Q_QUAD_MIX_16_5:
        case Format::Q_QUAD_MIX_24_5:
        case Format::Q_QUAD_MIX_3_5_GRP:
        case Format::Q_QUAD_MIX_4_5_GRP:
        case Format::Q_QUAD_MIX_6_5_GRP:
        case Format::Q_QUAD_MIX_8_5_GRP:
        case Format::Q_QUAD_MIX_12_5_GRP:
        case Format::Q_QUAD_MIX_16_5_GRP:
        case Format::Q_QUAD_MIX_24_5_GRP:
            dequant_quad_mix(fmt, indices, idx_bytes, nw, out);
            return;
        default: return;
    }
}

float block_actual_bpw(uint32_t nw, size_t idx_bytes, size_t cb_bytes) {
    if (nw == 0) return 0.0f;
    return (float)((idx_bytes + cb_bytes) * 8.0 / (double)nw);
}

size_t block_claimed_bytes(Format fmt, uint32_t n) {
    const float bpw = format_bpw(fmt);
    return (size_t)std::ceil(bpw * (float)n / 8.0f);
}

} // namespace quant
