#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
// test_block_codec.cpp — Unit tests for Q3, Q6, Q12, Q24, Q16 enhanced, and GRP codecs
#include "quant/types.h"
#include "quant/codebook.h"
#include "quant/kernel.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

void test_q24_codec() {
    std::cout << "[Test 1] Testing Q24 direct FP24 encode/decode..." << std::endl;
    constexpr int N = 256;
    std::vector<float> original(N);
    std::vector<uint8_t> packed(N * 3);
    std::vector<float> decoded(N);

    for (int i = 0; i < N; i++) {
        original[i] = (float)(i - 128) * 0.05f;
    }

    quant::kernel::q24_encode(original.data(), packed.data(), N);
    quant::kernel::q24_decode(packed.data(), decoded.data(), N);

    double max_err = 0.0;
    for (int i = 0; i < N; i++) {
        double err = std::abs((double)original[i] - (double)decoded[i]);
        if (err > max_err) max_err = err;
    }

    std::cout << "  -> Max Q24 error: " << max_err << std::endl;
    // FP24 has 15-bit mantissa (1 part in 32768 precision)
    assert(max_err < 0.01);
    std::cout << "  -> PASSED: Q24 encode/decode verified!" << std::endl;
}

void test_q3_kernel() {
    std::cout << "[Test 2] Testing Q3 SIMD GEMM kernel..." << std::endl;
    constexpr int M = 2, N = 2, K = 32;
    float codebook[8] = {-1.0f, -0.7f, -0.4f, -0.1f, 0.1f, 0.4f, 0.7f, 1.0f};
    int packed_bytes = M * N * ((K * 3 + 7) / 8);
    std::vector<uint8_t> packed(packed_bytes, 0x42);
    std::vector<float> activations(N * K, 0.5f);
    std::vector<float> output(M * N, 0.0f);

    quant::kernel::q3_gemm(packed.data(), codebook, activations.data(), output.data(), M, N, K);

    for (int i = 0; i < M * N; i++) {
        assert(!std::isnan(output[i]));
    }
    std::cout << "  -> PASSED: Q3 SIMD GEMM kernel executed cleanly!" << std::endl;
}

void test_q6_kernel() {
    std::cout << "[Test 3] Testing Q6 SIMD GEMM kernel..." << std::endl;
    constexpr int M = 2, N = 2, K = 32;
    float codebook[64];
    for (int i = 0; i < 64; i++) codebook[i] = -1.0f + 2.0f * i / 63.0f;
    int packed_bytes = M * N * ((K * 6 + 7) / 8);
    std::vector<uint8_t> packed(packed_bytes, 0x1F);
    std::vector<float> activations(N * K, 0.5f);
    std::vector<float> output(M * N, 0.0f);

    quant::kernel::q6_gemm(packed.data(), codebook, activations.data(), output.data(), M, N, K);

    for (int i = 0; i < M * N; i++) {
        assert(!std::isnan(output[i]));
    }
    std::cout << "  -> PASSED: Q6 SIMD GEMM kernel executed cleanly!" << std::endl;
}

void test_q12_kernel() {
    std::cout << "[Test 4] Testing Q12 SIMD GEMM kernel..." << std::endl;
    constexpr int M = 2, N = 2, K = 16;
    uint16_t codebook_fp16[4096] = {};
    for (int i = 0; i < 4096; i++) codebook_fp16[i] = 0x3C00; // FP16 1.0f
    int packed_bytes = M * N * ((K * 12 + 7) / 8);
    std::vector<uint8_t> packed(packed_bytes, 0x12);
    std::vector<float> activations(N * K, 0.5f);
    std::vector<float> output(M * N, 0.0f);

    quant::kernel::q12_gemm(packed.data(), codebook_fp16, activations.data(), output.data(), M, N, K);

    for (int i = 0; i < M * N; i++) {
        assert(!std::isnan(output[i]));
    }
    std::cout << "  -> PASSED: Q12 SIMD GEMM kernel executed cleanly!" << std::endl;
}

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "      Transcender Block Codec Unit Tests      " << std::endl;
    std::cout << "=========================================" << std::endl;

    test_q24_codec();
    test_q3_kernel();
    test_q6_kernel();
    test_q12_kernel();

    std::cout << "\nALL BLOCK CODEC TESTS PASSED!" << std::endl;
    return 0;
}
