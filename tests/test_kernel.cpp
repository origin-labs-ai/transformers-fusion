#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
// test_kernel.cpp — Unit test for Transcender SIMD kernels
#include "quant/kernel.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "     Transcender SIMD Kernel Unit Tests      " << std::endl;
    std::cout << "=========================================" << std::endl;

    constexpr int M = 4, N = 4, K = 64;
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 2.0f);
    std::vector<float> C(M * N, 0.0f);

    std::cout << "[Test 1] Testing scalar_gemm..." << std::endl;
    quant::kernel::scalar_gemm(A.data(), B.data(), C.data(), M, N, K);
    for (int i = 0; i < M * N; i++) {
        assert(std::abs(C[i] - 128.0f) < 1e-4f);
    }
    std::cout << "  -> PASSED: scalar_gemm verified!" << std::endl;

    std::cout << "[Test 2] Testing tiled_gemm..." << std::endl;
    std::fill(C.begin(), C.end(), 0.0f);
    quant::kernel::tiled_gemm(A.data(), B.data(), C.data(), M, N, K);
    for (int i = 0; i < M * N; i++) {
        assert(std::abs(C[i] - 128.0f) < 1e-4f);
    }
    std::cout << "  -> PASSED: tiled_gemm verified!" << std::endl;

    std::cout << "[Test 3] Testing avx2_gemm..." << std::endl;
    std::fill(C.begin(), C.end(), 0.0f);
    quant::kernel::avx2_gemm(A.data(), B.data(), C.data(), M, N, K);
    for (int i = 0; i < M * N; i++) {
        assert(std::abs(C[i] - 128.0f) < 1e-4f);
    }
    std::cout << "  -> PASSED: avx2_gemm verified!" << std::endl;

    std::cout << "\nALL SIMD KERNEL TESTS PASSED!" << std::endl;
    return 0;
}
