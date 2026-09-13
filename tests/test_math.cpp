#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
// test_math.cpp — Unit test for Transcender math library (activations, norms, BLAS)
#include "quant/math.h"
#include "quant/tensor.h"
#include "quant/test.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "      Transcender Math Library Unit Test      " << std::endl;
    std::cout << "=========================================" << std::endl;

    constexpr int N = 256;
    std::vector<float> input(N);
    std::vector<float> output(N);

    for (int i = 0; i < N; i++) input[i] = (float)i * 0.05f - 6.4f;

    std::cout << "[Test 1] Testing ReLU activation..." << std::endl;
    quant::math::vec_relu(output.data(), input.data(), N);
    for (int i = 0; i < N; i++) {
        float expected = std::max(0.0f, input[i]);
        assert(std::abs(output[i] - expected) < 1e-5f);
    }
    std::cout << "  -> PASSED: ReLU verified!" << std::endl;

    std::cout << "[Test 2] Testing SiLU activation..." << std::endl;
    quant::math::vec_silu(output.data(), input.data(), N);
    for (int i = 0; i < N; i++) {
        float sig = 1.0f / (1.0f + std::exp(-input[i]));
        float expected = input[i] * sig;
        assert(std::abs(output[i] - expected) < 1e-4f);
    }
    std::cout << "  -> PASSED: SiLU verified!" << std::endl;

    std::cout << "[Test 3] Testing RMSNorm..." << std::endl;
    std::vector<float> weight(N, 1.0f);
    quant::math::vec_rms_norm(output.data(), input.data(), weight.data(), N, 1e-5f);
    double sq_sum = 0.0;
    for (int i = 0; i < N; i++) sq_sum += output[i] * output[i];
    double rms = std::sqrt(sq_sum / N);
    std::cout << "  -> RMS of normalized output: " << rms << std::endl;
    assert(std::abs(rms - 1.0) < 1e-3);
    std::cout << "  -> PASSED: RMSNorm verified!" << std::endl;

    std::cout << "\nALL MATH TESTS PASSED SUCCESSFULLY!" << std::endl;

    // P0 regression: AVX2 gemm edge-tile OOB (math_avx2.cpp:60-74).
    // Old code loaded/stored 16 B/C floats even when N-j < 16 — over-read
    // past B and over-wrote past C. Odd sizes hit the edge path.
    {
        TEST_SUITE("P0: gemm edge tiles (odd N)");
        const int64_t M = 7, K = 9;
        const int64_t Ns[] = {1, 3, 7, 8, 9, 15, 16, 17, 23, 31};
        for (int64_t N : Ns) {
            quant::Tensor A(quant::Shape(M, K)), B(quant::Shape(K, N)), C(quant::Shape(M, N));
            float* pa = A.data<float>();
            float* pb = B.data<float>();
            for (int64_t i = 0; i < M * K; i++) pa[i] = (float)(i % 7) * 0.25f - 0.5f;
            for (int64_t i = 0; i < K * N; i++) pb[i] = (float)(i % 5) * 0.2f - 0.3f;
            quant::math::gemm(1.0f, A, B, 0.0f, C);
            const float* pc = C.data<float>();
            bool ok = true;
            for (int64_t m = 0; m < M && ok; m++)
                for (int64_t n = 0; n < N && ok; n++) {
                    double ref = 0;
                    for (int64_t k = 0; k < K; k++)
                        ref += (double)pa[m * K + k] * pb[k * N + n];
                    if (std::abs(pc[m * N + n] - (float)ref) > 1e-3f) ok = false;
                }
            TEST_CHECK(ok, "gemm exact vs scalar reference");
        }
    }

    return TEST_REPORT() > 0 ? 1 : 0;
}
