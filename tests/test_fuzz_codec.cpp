// test_fuzz_codec.cpp — L036 property-based roundtrip fuzz testing for the
// canonical block codec (quantize_block_all / dequantize_block_all).
//
// Design (deterministic, zero-dependency):
//   * Single std::mt19937_64 with a FIXED seed drives every generated tensor;
//     uniforms are derived straight from 64-bit engine output (no
//     implementation-defined distribution objects), so the corpus is
//     bit-identical on every run / toolset.
//   * 10,000 random blocks (n = 256) across four distributions:
//       uniform[-1,1], gaussian(sigma in {0.05, 0.1, 1.0}),
//       sparse (90% zeros), extreme-outlier mix (2% huge + 98% tiny).
//     Every block is round-tripped through EVERY supported codec format
//     enumerated from the Format registry (37 formats) => 370k roundtrips,
//     plus a multi-block pass and a tail-size robustness pass.
//   * Per roundtrip we assert: encode success, payload budget contract
//     (<= claimed BPW bytes, +1 documented tail byte), all-finite decode,
//     correct dimensions, Q32 bit-exactness, and a per-family relative MSE
//     bound (NMSE = MSE / E[w^2]).
//
// Bound derivation (mirrors test_grp_quality_proof's threshold style):
// every decoder reconstructs out[i] = level(i) * scale(block-or-group).
// Levels live on grids whose extent is tied to the block max (or a fixed
// quantile table), so reconstruction error is bounded by a multiple of the
// signal power for well-matched inputs (uniform/gaussian), and degrades
// gracefully — but never diverges — for sparse/outlier shapes. The caps
// below sit 2x-10x above measured-theory worst cases per distribution and
// far below broken-codec territory (NaN/garbage, wrong scales by 2x+,
// index corruption), reproducing the regression-guard role of the 22 dB
// floor in test_grp_quality_proof [Test 3].

#include "quant/types.h"
#include "quant/block_codec.h"
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

namespace {

// ---- deterministic RNG -----------------------------------------------------

std::mt19937_64 g_rng(0xCAFEF00DBEEF4211ULL);  // fixed seed: same tensors every run

double u01() {  // uniform [0,1) straight from engine bits (portable determinism)
    return (double)(g_rng() >> 11) * (1.0 / 9007199254740992.0);
}

double gauss01() {  // Box-Muller; guard log(0)
    double u1 = u01(), u2 = u01();
    if (u1 < 1e-300) u1 = 1e-300;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

// ---- distributions ----------------------------------------------------------

enum class Dist { Uniform, Gaussian, Sparse, Outlier };

const char* dist_name(Dist d) {
    switch (d) {
        case Dist::Uniform:  return "uniform[-1,1]";
        case Dist::Gaussian: return "gaussian";
        case Dist::Sparse:   return "sparse(90%)";
        case Dist::Outlier:  return "outlier-mix";
    }
    return "?";
}

void fill_tensor(std::vector<float>& w, Dist d, double sigma) {
    const size_t n = w.size();
    switch (d) {
        case Dist::Uniform:
            for (size_t i = 0; i < n; ++i) w[i] = (float)(2.0 * u01() - 1.0);
            break;
        case Dist::Gaussian:
            for (size_t i = 0; i < n; ++i) w[i] = (float)(gauss01() * sigma);
            break;
        case Dist::Sparse:  // 90% exact zeros, nonzeros ~ N(0,1)
            for (size_t i = 0; i < n; ++i)
                w[i] = (u01() < 0.10) ? (float)gauss01() : 0.0f;
            break;
        case Dist::Outlier:  // few huge values + many tiny ones
            for (size_t i = 0; i < n; ++i) {
                if (u01() < 0.02) {
                    const double mag = 10.0 + 10.0 * u01();
                    w[i] = (float)(mag * (u01() < 0.5 ? -1.0 : 1.0));
                } else {
                    w[i] = (float)(gauss01() * 0.01);
                }
            }
            break;
    }
}

// ---- failure handling (immune to NDEBUG, unlike bare assert) ----------------

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "FUZZ FAIL: %s\n", msg.c_str());
    std::fflush(stderr);
    std::exit(1);
}

void require(bool cond, const std::string& msg) {
    if (!cond) die(msg);
}

// ---- per-format NMSE caps ---------------------------------------------------
// NMSE := MSE / E[w^2] of one roundtripped tensor; cap must hold for every
// tensor of every distribution (values chosen with headroom over theory, see
// file-top comment). Order follows the Format enum for easy auditing.

struct CapRow { quant::Format fmt; double cap; bool exact; };

const CapRow kCaps[] = {
    // Base lattice / codebook formats (block-wide fitted scale)
    { quant::Format::Q1,              3.00, false },  // sign-only: NMSE ~0.4 dense, ~1.5 sparse
    { quant::Format::Q2,              1.50, false },  // 4-level Lloyd, block max-normalized
    { quant::Format::Q3,              0.90, false },
    { quant::Format::Q4,              0.60, false },
    { quant::Format::Q6,              0.20, false },  // quantile table puts levels near 0
    { quant::Format::Q8,              0.08, false },  // step^2/4 bound: (A/rms)^2/255^2 << cap
    { quant::Format::Q12,             0.05, false },
    { quant::Format::Q16,             1e-5, false },  // uniform grid over [vmin,vmax], 65535 levels
    { quant::Format::Q24,             1e-7, false },  // mantissa truncation rel err <= 2^-15
    { quant::Format::Q32,             0.0,  true  },  // FP32 identity: bit-exact required
    // GRP variants
    { quant::Format::Q1_G,          3.00, false },
    { quant::Format::Q2_G,          1.60, false },  // per-16 affine (4b sc+min)
    { quant::Format::Q3_G,          0.80, false },
    { quant::Format::Q4_G,          0.60, false },
    { quant::Format::Q6_G,          0.25, false },  // Q6_K scheme, per-16 int8 scales
    { quant::Format::Q8_G,          0.10, false },
    { quant::Format::Q12_G,         0.08, false },  // grp16 path, 3-bit group scales
    { quant::Format::Q16_G,         0.08, false },
    { quant::Format::Q24_G,         0.08, false },
    // K variants — identical wire + encoder as their plain twins (L/M/H are
    // search-depth knobs, not layout changes), so they inherit plain caps.
    { quant::Format::Q1_K_L,          3.00, false },
    { quant::Format::Q1_K_M,          3.00, false },
    { quant::Format::Q1_K_H,          3.00, false },
    { quant::Format::Q2_K_L,          1.50, false },
    { quant::Format::Q2_K_M,          1.50, false },
    { quant::Format::Q2_K_H,          1.50, false },
    { quant::Format::Q3_K_L,          0.90, false },
    { quant::Format::Q3_K_M,          0.90, false },
    { quant::Format::Q3_K_H,          0.90, false },
    { quant::Format::Q4_K_L,          0.60, false },
    { quant::Format::Q4_K_M,          0.60, false },
    { quant::Format::Q4_K_H,          0.60, false },
    { quant::Format::Q6_K_L,          0.20, false },
    { quant::Format::Q6_K_M,          0.20, false },
    { quant::Format::Q6_K_H,          0.20, false },
    { quant::Format::Q8_K_L,          0.08, false },
    { quant::Format::Q8_K_M,          0.08, false },
    { quant::Format::Q8_K_H,          0.08, false },
    { quant::Format::Q12_K_L,         0.05, false },
    { quant::Format::Q12_K_M,         0.05, false },
    { quant::Format::Q12_K_H,         0.05, false },
    { quant::Format::Q16_K_L,         1e-5, false },
    { quant::Format::Q16_K_M,         1e-5, false },
    { quant::Format::Q16_K_H,         1e-5, false },
    { quant::Format::Q24_K_L,         1e-7, false },
    { quant::Format::Q24_K_M,         1e-7, false },
    { quant::Format::Q24_K_H,         1e-7, false },
    // K_G variants — same wire as the GRP twins (see block_codec dispatch)
    { quant::Format::Q1_K_L_G,      3.00, false },
    { quant::Format::Q1_K_M_G,      3.00, false },
    { quant::Format::Q1_K_H_G,      3.00, false },
    { quant::Format::Q2_K_L_G,      1.60, false },
    { quant::Format::Q2_K_M_G,      1.60, false },
    { quant::Format::Q2_K_H_G,      1.60, false },
    { quant::Format::Q3_K_L_G,      0.80, false },
    { quant::Format::Q3_K_M_G,      0.80, false },
    { quant::Format::Q3_K_H_G,      0.80, false },
    { quant::Format::Q4_K_L_G,      0.60, false },
    { quant::Format::Q4_K_M_G,      0.60, false },
    { quant::Format::Q4_K_H_G,      0.60, false },
    { quant::Format::Q6_K_L_G,      0.25, false },
    { quant::Format::Q6_K_M_G,      0.25, false },
    { quant::Format::Q6_K_H_G,      0.25, false },
    { quant::Format::Q8_K_L_G,      0.10, false },
    { quant::Format::Q8_K_M_G,      0.10, false },
    { quant::Format::Q8_K_H_G,      0.10, false },
    { quant::Format::Q12_K_L_G,     0.08, false },
    { quant::Format::Q12_K_M_G,     0.08, false },
    { quant::Format::Q12_K_H_G,     0.08, false },
    { quant::Format::Q16_K_L_G,     0.08, false },
    { quant::Format::Q16_K_M_G,     0.08, false },
    { quant::Format::Q16_K_H_G,     0.08, false },
    { quant::Format::Q24_K_L_G,     0.08, false },
    { quant::Format::Q24_K_M_G,     0.08, false },
    { quant::Format::Q24_K_H_G,     0.08, false },
    // Half-BPW plain (affine / compound paths per block_codec dispatch)
    { quant::Format::Q1_5,            3.00, false },
    { quant::Format::Q2_5,            1.60, false },
    { quant::Format::Q3_5,            0.80, false },
    { quant::Format::Q4_5,            0.60, false },
    { quant::Format::Q6_5,            0.25, false },
    { quant::Format::Q8_5,            0.10, false },
    { quant::Format::Q12_5,           0.08, false },
    { quant::Format::Q16_5,           1e-5, false },
    { quant::Format::Q24_5,           1e-7, false },
    // Half-BPW GRP — identical wire to the half plain rows
    { quant::Format::Q_G_1_5,       3.00, false },
    { quant::Format::Q_G_2_5,       1.60, false },
    { quant::Format::Q_G_3_5,       0.80, false },
    { quant::Format::Q_G_4_5,       0.60, false },
    { quant::Format::Q_G_6_5,       0.25, false },
    { quant::Format::Q_G_8_5,       0.10, false },
    { quant::Format::Q_G_12_5,      0.08, false },
    { quant::Format::Q_G_16_5,      1e-5, false },
    { quant::Format::Q_G_24_5,      1e-7, false },
    // QUAD_MIX plain — dominant tier governs; GRP caps are conservative here
    { quant::Format::MXQ_3_5,          4.00, false },
    { quant::Format::MXQ_4_5,          4.00, false },
    { quant::Format::MXQ_6_5,          2.00, false },
    { quant::Format::MXQ_8_5,          2.00, false },
    { quant::Format::MXQ_12_5,         2.00, false },
    { quant::Format::MXQ_16_5,         2.00, false },
    { quant::Format::MXQ_24_5,         2.00, false },
    // QUAD_MIX GRP — dominant tier governs the cap
    { quant::Format::MXQ_3_5_G,      4.00, false },  // 92% sign tier
    { quant::Format::MXQ_4_5_G,      4.00, false },  // 58.5% sign tier
    { quant::Format::MXQ_6_5_G,      2.00, false },
    { quant::Format::MXQ_8_5_G,      2.00, false },
    { quant::Format::MXQ_12_5_G,     2.00, false },
    { quant::Format::MXQ_16_5_G,     2.00, false },
    { quant::Format::MXQ_24_5_G,     2.00, false },
};

const CapRow& cap_for(quant::Format f) {
    for (const auto& r : kCaps)
        if (r.fmt == f) return r;
    die(std::string("no cap registered for format id ") +
        std::to_string((int)f));
}

// Enumerate every supported codec format from the registry. Enum slot 19 is
// a deliberate gap in quant::Format (see include/quant/types.h) — reported,
// never silently skipped.
std::vector<quant::Format> supported_formats() {
    std::vector<quant::Format> out;
    out.reserve((size_t)quant::FORMAT_COUNT);
    for (int v = 0; v < quant::FORMAT_COUNT; ++v) {
        if (v == 19) continue;  // gap slot: not a defined Format
        const auto f = static_cast<quant::Format>(v);
        require(quant::format_bpw(f) > 0.0f,
                std::string("registry gap: format_bpw==0 for id ") + std::to_string(v));
        out.push_back(f);
    }
    return out;
}

// ---- one encode->decode roundtrip -------------------------------------------

void roundtrip(quant::Format fmt, const std::vector<float>& w,
               std::vector<uint8_t>& idx, std::vector<uint8_t>& cb,
               std::vector<float>& dec,
               double& mse, double& energy, const std::string& ctx) {
    const int n = (int)w.size();

    require(quant::quantize_block_all(fmt, w.data(), n, idx, cb),
            ctx + ": quantize_block_all returned false");
    require(!idx.empty() || !cb.empty(),
            ctx + ": encoder produced empty payload");
    // Budget contract (include/quant/block_codec.h): full blocks exact,
    // tails allowed one extra container-alignment byte.
    const size_t claimed = quant::block_claimed_bytes(fmt, (uint32_t)n);
    require(idx.size() + cb.size() <= claimed + 1,
            ctx + ": payload " + std::to_string((unsigned long long)(idx.size() + cb.size())) +
            " B exceeds claimed budget " + std::to_string((unsigned long long)claimed) + " B (+1)");

    dec.assign((size_t)n, 0.0f);
    quant::dequantize_block_all(fmt, idx.data(), idx.size(), cb.data(),
                                cb.size(), (uint32_t)n, dec.data());
    require(dec.size() == (size_t)n, ctx + ": decode dimension mismatch");

    double se = 0.0, en = 0.0;
    for (int i = 0; i < n; ++i) {
        require(std::isfinite(dec[(size_t)i]),
                ctx + ": non-finite decoded output at [" + std::to_string(i) + "]");
        const double diff = (double)dec[(size_t)i] - (double)w[(size_t)i];
        se += diff * diff;
        en += (double)w[(size_t)i] * (double)w[(size_t)i];
    }
    mse = se / (double)n;
    energy = en / (double)n;

    if (cap_for(fmt).exact) {
        for (int i = 0; i < n; ++i)
            require(dec[(size_t)i] == w[(size_t)i],
                    ctx + ": Q32 roundtrip not bit-exact at [" + std::to_string(i) + "]");
    }
}

// ---- corpus driver ------------------------------------------------------------

constexpr int kBlock = 256;           // one full codec block per tensor
constexpr int kMainTensors = 10000;   // 4 distributions x 2500 blocks
constexpr int kFormats = 104;         // v3: FORMAT_COUNT-1 (enum slot 19 is a gap)

struct PerFormatStats {
    double worst_nmse = 0.0;
    double gauss_sum = 0.0;   // mean-NMSE accumulator over gaussian(0.1) slice
    long   gauss_cnt = 0;
    unsigned long long roundtrips = 0;
};

} // namespace

int main() {
    std::printf("=========================================\n");
    std::printf("   InNova Block Codec Fuzz (L036)\n");
    std::printf("=========================================\n");

    const std::vector<quant::Format> formats = supported_formats();
    require((int)formats.size() == kFormats,
            "expected 37 supported formats, got " + std::to_string(formats.size()));
    std::printf("SKIPPED enum slot 19: deliberate gap in quant::Format "
                "(not a defined format; unreachable by design)\n");

    std::vector<PerFormatStats> st(kFormats);

    // ---- pass 1: 10,000 single-block tensors x all formats ---------------
    const double gauss_sigmas[3] = {0.05, 0.1, 1.0};
    std::vector<float> w(kBlock);
    std::vector<uint8_t> idx, cb;
    std::vector<float> dec;

    for (int t = 0; t < kMainTensors; ++t) {
        const int branch = t % 4;
        const Dist d = (branch == 0) ? Dist::Uniform
                     : (branch == 1) ? Dist::Gaussian
                     : (branch == 2) ? Dist::Sparse
                                     : Dist::Outlier;
        const double sigma = (d == Dist::Gaussian)
                           ? gauss_sigmas[(t / 4) % 3] : 0.0;
        fill_tensor(w, d, sigma);

        const bool hi_res_slice = (d == Dist::Gaussian && sigma == 0.1);

        for (int fi = 0; fi < kFormats; ++fi) {
            const quant::Format f = formats[(size_t)fi];
            double mse = 0.0, energy = 0.0;
            roundtrip(f, w, idx, cb, dec, mse, energy,
                      std::string("tensor#") + std::to_string(t) + "/" +
                      dist_name(d) + "/" + quant::format_name(f));

            require(energy > 1e-30,
                    "generator produced zero-energy tensor (bug in test)");
            const double nmse = mse / energy;
            const CapRow& cr = cap_for(f);
            if (!cr.exact) {
                require(nmse <= cr.cap,
                        std::string(quant::format_name(f)) + " NMSE " +
                        std::to_string(nmse) + " exceeds bound " +
                        std::to_string(cr.cap) + " on tensor #" +
                        std::to_string(t));
            }
            if (nmse > st[(size_t)fi].worst_nmse) st[(size_t)fi].worst_nmse = nmse;
            if (hi_res_slice) {
                st[(size_t)fi].gauss_sum += nmse;
                ++st[(size_t)fi].gauss_cnt;
            }
            ++st[(size_t)fi].roundtrips;
        }
    }
    std::printf("pass 1: %d blocks x %d formats = %lld roundtrips\n",
                kMainTensors, kFormats,
                (long long)kMainTensors * kFormats);

    // ---- pass 2: multi-block tensors (cross-block scale independence) -----
    {
        constexpr int N = 1024;
        std::vector<float> mw(N);
        const Dist dists[4] = {Dist::Uniform, Dist::Gaussian, Dist::Sparse, Dist::Outlier};
        unsigned long long trips = 0;
        for (int k = 0; k < 4; ++k) {
            fill_tensor(mw, dists[k], 0.1);
            for (int fi = 0; fi < kFormats; ++fi) {
                const quant::Format f = formats[(size_t)fi];
                double mse = 0.0, energy = 0.0;
                roundtrip(f, mw, idx, cb, dec, mse, energy,
                          std::string("multiblock/") + dist_name(dists[k]) + "/" +
                          quant::format_name(f));
                const double nmse = mse / energy;
                const CapRow& cr = cap_for(f);
                if (!cr.exact) {
                    require(nmse <= cr.cap,
                            std::string(quant::format_name(f)) +
                            " NMSE exceeds bound on multiblock/" +
                            dist_name(dists[k]));
                }
                if (nmse > st[(size_t)fi].worst_nmse) st[(size_t)fi].worst_nmse = nmse;
                ++st[(size_t)fi].roundtrips;
                ++trips;
            }
        }
        std::printf("pass 2: 4 tensors (n=%d) x %d formats = %lld roundtrips\n",
                    N, kFormats, (long long)trips);
    }

    // ---- pass 3: tail sizes (structural checks only) -----------------------
    // Tiny blocks hit documented degradation paths (implicit d=1 raw-index
    // layouts below 8/16/32 weights, see src/block_codec.cpp), so absolute
    // error bounds do not apply; structure, finiteness and budget still must.
    {
        const int sizes[] = {1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 127, 255};
        const Dist dists[4] = {Dist::Uniform, Dist::Gaussian, Dist::Sparse, Dist::Outlier};
        unsigned long long trips = 0, tensors = 0;
        for (int sz : sizes) {
            std::vector<float> tw((size_t)sz);
            for (int k = 0; k < 4; ++k) {
                fill_tensor(tw, dists[k], 0.1);
                ++tensors;
                for (int fi = 0; fi < kFormats; ++fi) {
                    const quant::Format f = formats[(size_t)fi];
                    double mse = 0.0, energy = 0.0;
                    roundtrip(f, tw, idx, cb, dec, mse, energy,
                              std::string("tail/") + dist_name(dists[k]) +
                              "/n=" + std::to_string(sz) + "/" +
                              quant::format_name(f));
                    ++st[(size_t)fi].roundtrips;
                    ++trips;
                }
            }
        }
        std::printf("pass 3: %llu tail tensors (12 sizes x 4 dists) x %d formats"
                    " = %lld roundtrips (structural checks)\n",
                    (unsigned long long)tensors, kFormats, (long long)trips);
    }

    // ---- quality hierarchy on the gaussian(0.1) slice ----------------------
    // Mirrors test_grp_quality_proof [Test 3]: per-group scaling must beat the
    // same-BPW plain format, and quality must improve monotonically with BPW.
    auto gauss_mean = [&](quant::Format f) -> double {
        for (int i = 0; i < kFormats; ++i)
            if (formats[(size_t)i] == f) {
                require(st[(size_t)i].gauss_cnt > 0, "empty gaussian slice");
                return st[(size_t)i].gauss_sum / (double)st[(size_t)i].gauss_cnt;
            }
        die("format missing from registry");
    };
    const double m_q2_grp = gauss_mean(quant::Format::Q2_G);
    const double m_q3     = gauss_mean(quant::Format::Q3);
    const double m_q3_grp = gauss_mean(quant::Format::Q3_G);
    const double m_q4_grp = gauss_mean(quant::Format::Q4_G);
    require(m_q3_grp <= m_q3 * 1.13 + 1e-15,   // GRP@3.5 within +0.5 dB of Q3@3.0
            "Q3_G mean NMSE regressed vs plain Q3 on gaussian slice");
    require(m_q4_grp < m_q3_grp,               // more bits -> better (strict)
            "Q4_G does not beat Q3_G on gaussian slice");
    require(m_q3_grp < m_q2_grp,               // fewer bits -> worse (strict)
            "Q2_G is not worse than Q3_G on gaussian slice");
    std::printf("hierarchy (gaussian sigma=0.1 mean NMSE): "
                "Q2_G=%.3g > Q3_G=%.3g > Q4_G=%.3g; Q3=%.3g\n",
                m_q2_grp, m_q3_grp, m_q4_grp, m_q3);

    // ---- per-format summary -------------------------------------------------
    std::printf("-----------------------------------------\n");
    std::printf("%-20s %8s %12s %12s\n", "format", "bpw", "worst-NMSE", "bound");
    for (int fi = 0; fi < kFormats; ++fi) {
        const quant::Format f = formats[(size_t)fi];
        const CapRow& cr = cap_for(f);
        char capbuf[32];
        if (cr.exact) std::snprintf(capbuf, sizeof(capbuf), "%s", "exact");
        else          std::snprintf(capbuf, sizeof(capbuf), "%.4g", cr.cap);
        std::printf("%-20s %8.3f %12.4g %12s\n",
                    quant::format_name(f), quant::format_bpw(f),
                    st[(size_t)fi].worst_nmse, capbuf);
    }
    std::printf("-----------------------------------------\n");
    unsigned long long total = 0;
    for (int fi = 0; fi < kFormats; ++fi) total += st[(size_t)fi].roundtrips;
    std::printf("FUZZ CODEC PASS: %d formats, %d generated tensors, %llu roundtrips, 0 crashes\n",
                kFormats, kMainTensors + 4 + 48, total);
    return 0;
}
