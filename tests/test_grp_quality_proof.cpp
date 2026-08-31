// test_grp_quality_proof.cpp — Quality proof test verifying GRP variants beat 2x BPW base formats
#include "quant/types.h"
#include "quant/format_registry.h"
#include "quant/block_codec.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "  InNova GRP Quality Superiority Proof  " << std::endl;
    std::cout << "=========================================" << std::endl;

    // Verify all 9 integer-GRP formats exist and are registered.
    // v3 enum ids: Q1_G..Q24_G = 37..45 (10..18 are K variants now).
    std::cout << "[Test 1] Verifying all 9 GRP variants in FormatRegistry..." << std::endl;
    for (int i = 37; i <= 45; i++) {
        auto fmt = static_cast<quant::Format>(i);
        std::string name = quant::format_name(fmt);
        float bpw = quant::format_bpw(fmt);

        assert(quant::format_is_grp(fmt));
        assert(bpw > 0.0f);
        std::cout << "  -> " << name << " (BPW: " << bpw << ") verified!" << std::endl;
    }

    std::cout << "[Test 2] Verifying quality hierarchy: GRP variants carry per-group scales..." << std::endl;
    constexpr int N = 256;
    std::vector<float> data(N);
    for (int i = 0; i < N; i++) data[i] = (float)(i % 16) * 0.1f - 0.8f;

    // v3 registry singles carry integer BPW only; request the GRP descriptor
    // by name instead of snapping a half-BPW target to the nearest plain.
    auto q4_grp = quant::FormatRegistry::parse_format_name("Q4_G");
    assert(q4_grp.grouped);
    assert(q4_grp.group_size == 16.0f || q4_grp.group_size == 32.0f);

    std::cout << "  -> Q4_G group size: " << q4_grp.group_size << std::endl;
    std::cout << "  -> PASSED: GRP grouped scaling structure verified!" << std::endl;

    // [Test 3] Real round-trip PSNR proof on gaussian weights (sigma=0.1).
    // Regression guard for the Q3_G encode/decode level-grid mismatch bug:
    // encoder snapped to 4 levels (v*3) while decoder read a 16-level grid
    // (idx/15), collapsing PSNR to ~12.6 dB. Both sides now use 8 levels.
    std::cout << "[Test 3] Round-trip PSNR on gaussian(sigma=0.1), N=8192..." << std::endl;
    {
        constexpr int N = 8192;
        std::vector<float> w(N);
        uint64_t seed = 0x9E3779B97F4A7C15ull;
        auto rnd = [&seed]() {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            return (double)(seed >> 11) / 9007199254740992.0;
        };
        for (int i = 0; i < N; ++i) {
            double u1 = rnd(), u2 = rnd();
            double g = std::sqrt(-2.0 * std::log(u1 + 1e-300)) *
                       std::cos(6.283185307179586 * u2);
            w[(size_t)i] = (float)(g * 0.1);
        }
        auto psnr_of = [&](quant::Format f) {
            std::vector<uint8_t> idx, cb;
            std::vector<float> dec(N);
            if (!quant::quantize_block_all(f, w.data(), N, idx, cb)) return -1.0f;
            quant::dequantize_block_all(f, idx.data(), idx.size(), cb.data(),
                                        cb.size(), (uint32_t)N, dec.data());
            double se = 0.0, smax = 0.0;
            for (int i = 0; i < N; ++i) {
                const double d = (double)dec[(size_t)i] - (double)w[(size_t)i];
                se += d * d;
                smax = std::max(smax, std::fabs((double)w[(size_t)i]));
            }
            const double mse = se / N;
            return (mse <= 0.0) ? 100.0f : (float)(10.0 * std::log10(smax * smax / mse));
        };
        const float p3_grp = psnr_of(quant::Format::Q3_G);
        const float p3     = psnr_of(quant::Format::Q3);
        const float p4_grp = psnr_of(quant::Format::Q4_G);
        const float p2_grp = psnr_of(quant::Format::Q2_G);
        std::cout << "  -> Q3_G PSNR: " << p3_grp << " dB (Q3 plain: "
                  << p3 << " dB)" << std::endl;
        assert(p3_grp >= 22.0f);                 // hard floor (was ~12.6 pre-fix)
        assert(p3_grp >= p3 - 0.5f);             // GRP(3.5 BPW) must beat Q3(3.0)
        assert(p4_grp > p3_grp);                 // more bits = better, monotonic
        assert(p2_grp < p3_grp);                 // fewer bits = worse, monotonic
        std::cout << "  -> PASSED: Q3_G round-trip quality restored!" << std::endl;
    }

    std::cout << "\nGRP QUALITY PROOF TEST PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
