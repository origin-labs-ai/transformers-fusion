// test_native_weight.cpp — NativeQWeightStore roundtrip + CID allocation
#include "quant/native_weight.h"
#include "quant/test.h"
#include <vector>
#include <cmath>
#include <cstdio>

using namespace quant::native;

int main() {
    TEST_SUITE("native_weight");
    printf("=== NativeQWeightStore test ===\n\n");

    // 4 blocks x 128
    const size_t N = 512;
    const size_t B = 128;

    NativeQWeightStore store(N, B);
    TEST_CHECK(store.size() == N, "store size == N");
    TEST_CHECK(store.block_size() == B, "block size == 128");
    TEST_CHECK(store.num_blocks() == N / B, "4 blocks");

    // Synthetic weights: mostly small, a few large (sensitivity distribution)
    std::vector<float> w(N), sens(N), dec(N);
    unsigned int seed = 7;
    for (size_t i = 0; i < N; i++) {
        seed = seed * 1103515245u + 12345u;
        float v = (float)((seed >> 16) & 0xFFFF) / 65535.0f - 0.5f;
        w[i] = v;
        sens[i] = std::fabs(v);
        // amplify sensitivity of first block so CID picks Q8 there
        if (i < B) { w[i] *= 8.0f; sens[i] *= 8.0f; }
    }

    store.initialize(w.data(), sens.data(), 0.25f, 0.5f);

    // Allocation fractions: 4 blocks -> 1x Q8 (top 25% sensitivity),
    // 2x Q1 (next 50%), 1x Q4 (rest).
    size_t n_q8 = 0, n_q1 = 0, n_q4 = 0;
    for (size_t b = 0; b < store.num_blocks(); b++) {
        switch (store.get_format(b * B)) {
            case NativeFormat::Q8: n_q8++; break;
            case NativeFormat::Q1: n_q1++; break;
            case NativeFormat::Q4: n_q4++; break;
        }
    }
    TEST_CHECK(n_q8 == 1 && n_q1 == 2 && n_q4 == 1,
               "allocation = 1x Q8, 2x Q1, 1x Q4");
    TEST_CHECK(store.get_format(0) == NativeFormat::Q8,
               "block 0 allocated Q8 (high sensitivity)");

    // Dequantized error: Q8 block must be tighter than Q1 blocks
    store.dequantize(dec.data());
    float err8 = 0.0f, err1 = 0.0f;
    bool has_q8 = false, has_q1 = false;
    for (size_t b = 0; b < store.num_blocks(); b++) {
        float err = 0.0f;
        for (size_t i = b * B; i < (b + 1) * B; i++) err += std::fabs(w[i] - dec[i]);
        if (store.get_format(b * B) == NativeFormat::Q8) { err8 += err; has_q8 = true; }
        if (store.get_format(b * B) == NativeFormat::Q1) { err1 += err; has_q1 = true; }
    }
    TEST_CHECK(has_q8 && has_q1, "both Q8 and Q1 blocks present");
    TEST_CHECK(err1 > err8, "Q1 block error > Q8 block error");
    TEST_CHECK(store.get_weight(3) == dec[3], "get_weight matches dequantize");

    // Apply a gradient update (two-timescale SGD) — no crash, weights remain finite
    std::vector<float> grad(N, 0.0f);
    for (size_t i = 0; i < N; i++) grad[i] = dec[i] - w[i];
    store.apply_quant_update(grad.data(), 0.1f, 0.5f);
    store.dequantize(dec.data());
    bool finite = true;
    for (size_t i = 0; i < N; i++)
        if (!std::isfinite(dec[i])) { finite = false; break; }
    TEST_CHECK(finite, "weights finite after quant update");

    printf("\n");
    return TEST_REPORT() > 0 ? 1 : 0;
}
