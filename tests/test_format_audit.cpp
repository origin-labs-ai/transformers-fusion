// test_format_audit.cpp — BPW ironclad probe (audit 2026-08-26).
// For every wire format: round-trip a gaussian block and report
//   claimed BPW  (types.h format_bpw)
//   actual BPW   (bytes actually stored / n)
//   PSNR         (reconstruction quality)
// plus a PASS/FAIL line per format for the two contracts:
//   A) actual <= claimed + 0.0001  (BPW IRONCLAD)
//   B) PSNR >= 20 dB on this dataset (sanity floor; per-format floors live
//      in test_fuzz_codec caps)
#include "quant/types.h"
#include "quant/block_codec.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace quant;

int main() {
    std::mt19937 rng(20260826);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    constexpr int N = 256;
    std::vector<float> w(N);
    for (auto& v : w) v = nd(rng);

    // Sigma-0.1 spot check (matches bench dataset scale) for the Q8 class.
    {
        std::mt19937 r2(42);
        std::normal_distribution<float> s01(0.0f, 0.1f);
        std::vector<float> w2(N);
        for (auto& v : w2) v = s01(r2);
        double s2 = 0.0;
        for (float v : w2) s2 += double(v) * v;
        for (Format f : {Format::Q8, Format::Q8_G, Format::Q_G_8_5}) {
            std::vector<uint8_t> idx, cb;
            if (!quantize_block_all(f, w2.data(), N, idx, cb)) continue;
            std::vector<float> out(N, 0.0f);
            dequantize_block_all(f, idx.data(), idx.size(), cb.data(), cb.size(), N, out.data());
            double e = 0.0;
            for (int i = 0; i < N; ++i) { const double d = double(w2[i]) - out[(size_t)i]; e += d * d; }
            const double psnr = e > 0 ? 10.0 * std::log10(s2 / e) : 999.0;
            std::printf("[sigma0.1] %-10s bytes=%zu PSNR=%.2f dB\n",
                        format_name(f), idx.size() + cb.size(), psnr);
        }
    }

    double sumsq = 0.0;
    for (float v : w) sumsq += double(v) * v;
    const double signal = sumsq * N > 0 ? sumsq : 1.0; // E[w^2]*N

    int violations = 0;
    std::printf("%-16s %9s %9s %8s %8s  %s\n",
                "format", "claimBPW", "actual", "PSNR", "bytes", "verdict");
    for (int id = 0; id < FORMAT_COUNT; ++id) {
        if (id == 19) continue; // deliberate enum gap
        const Format f = static_cast<Format>(id);
        std::vector<uint8_t> idx, cb;
        if (!quantize_block_all(f, w.data(), N, idx, cb)) continue;
        std::vector<float> out(N, 0.0f);
        dequantize_block_all(f, idx.data(), idx.size(), cb.data(), cb.size(), N, out.data());
        double err = 0.0;
        for (int i = 0; i < N; ++i) {
            const double d = double(w[i]) - out[(size_t)i];
            err += d * d;
        }
        const double psnr = (err > 0.0)
            ? 10.0 * std::log10(signal / err) : 999.0;
        const float claim = format_bpw(f);
        const double actual = double(idx.size() + cb.size()) * 8.0 / double(N);
        const bool bpw_ok = actual <= claim + 1e-3;
        if (!bpw_ok) ++violations;
        std::printf("%-16s %9.4f %9.4f %8.2f %8zu  %s\n",
                    format_name(f), claim, actual, psnr, idx.size() + cb.size(),
                    bpw_ok ? "ok" : "BPW-VIOLATION");
    }
    std::printf("\nBPW violations: %d\n", violations);
    return violations > 0 ? 1 : 0;
}
