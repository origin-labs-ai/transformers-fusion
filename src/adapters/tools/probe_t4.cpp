// probe_t4.cpp — replicate test_quant_mix Test 4 data + measure per-block
// format MSE to find why QUANT2_G (affine) underperforms QUANT1_G on the
// smooth sine blocks (negative benefit) after the sign-aware-min change.
#include "quant/format_registry.h"
#include "quant/block_codec.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace quant;

static double block_mse_fmt(Format f, const float* blk, int n) {
    std::vector<uint8_t> idx, cb;
    quantize_block_all(f, blk, n, idx, cb);
    std::vector<float> dec((size_t)n, 0.0f);
    dequantize_block_all(f, idx.data(), idx.size(), cb.data(), cb.size(), (uint32_t)n, dec.data());
    double e = 0.0;
    for (int j = 0; j < n; j++) { const double d = (double)blk[j] - (double)dec[(size_t)j]; e += d * d; }
    return e / (double)n;
}

int main() {
    const int nb = 64;
    std::vector<float> data((size_t)nb * 256);
    for (int b = 0; b < nb; b++) {
        const bool spikey = b >= 32;
        const double phase = (double)b;
        for (int j = 0; j < 256; j++) {
            double v;
            if (spikey) {
                v = 0.5 * (std::sin(2.0 * 3.14159265358979 * 8.0 * (double)j / 256.0 + phase)
                           + 0.6 * std::sin(2.0 * 3.14159265358979 * (double)j / 7.0 + phase));
            } else {
                v = std::sin(2.0 * 3.14159265358979 * (double)j / 256.0 + phase);
            }
            data[(size_t)b * 256 + j] = (float)v;
        }
    }
    printf("block type : QUANT1_G  QUANT2_G  benefit   QUANT2    QUANT_Q0\n");
    for (int b = 0; b < 8; b++) {
        const float* blk = data.data() + (size_t)b * 256;
        const double q1 = block_mse_fmt(Format::QUANT1_G, blk, 256);
        const double q2g = block_mse_fmt(Format::QUANT2_G, blk, 256);
        const double q2 = block_mse_fmt(Format::QUANT2, blk, 256);
        const double q0 = block_mse_fmt(Format::QUANT_Q0, blk, 256);
        printf("smooth %2d  : %.6f  %.6f  %+.6f  %.6f  %.6f\n",
               b, q1, q2g, q1 - q2g, q2, q0);
    }
    for (int b = 32; b < 40; b++) {
        const float* blk = data.data() + (size_t)b * 256;
        const double q1 = block_mse_fmt(Format::QUANT1_G, blk, 256);
        const double q2g = block_mse_fmt(Format::QUANT2_G, blk, 256);
        const double q2 = block_mse_fmt(Format::QUANT2, blk, 256);
        const double q0 = block_mse_fmt(Format::QUANT_Q0, blk, 256);
        printf("spikey %2d  : %.6f  %.6f  %+.6f  %.6f  %.6f\n",
               b, q1, q2g, q1 - q2g, q2, q0);
    }
    return 0;
}
