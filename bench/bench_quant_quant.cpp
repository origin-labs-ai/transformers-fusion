#include "quant/tensor.h"
#include "quant/math.h"
#include "quant/kernel.h"
#include "quant/random.h"
#include "quant/types.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <cstring>
#include <iomanip>

using namespace quant;

struct QuantBenchResult {
    double fp32_gflops;
    double quant8_gflops;
    double quant4_gflops;
    double tl1_gflops;
};

static double measure_gflops(int64_t M, int64_t N, int64_t K, int iters,
                             const std::function<void()>& fn) {
    for (int w = 0; w < 3; w++) fn();
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) fn();
    auto end = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(end - start).count() / iters;
    double ops = 2.0 * M * N * K;
    return ops / (sec * 1e9);
}

static QuantBenchResult bench_all_quant_gemm(int64_t M, int64_t N, int64_t K) {
    QuantBenchResult r = {0,0,0,0};
    RNG rng(42);
    Tensor A({M, K}), B({K, N}), C({M, N});
    for (int64_t i = 0; i < A.numel(); i++) A.data<float>()[i] = rng.uniform() * 2.0f - 1.0f;
    for (int64_t i = 0; i < B.numel(); i++) B.data<float>()[i] = rng.uniform() * 2.0f - 1.0f;

    int iters = 20;
    r.fp32_gflops = measure_gflops(M, N, K, iters, [&]() {
        math::gemm(1.0f, A, B, 0.0f, C);
    });
    std::cerr << "  FP32 done: " << r.fp32_gflops << " GFLOPS\n";

    // Q8: uint8 indices + float codebook[256]
    {
        float codebook[256];
        for (int i = 0; i < 256; i++) codebook[i] = (float)(i - 128) / 128.0f;
        std::vector<uint8_t> indices((size_t)M * N * K);
        for (size_t i = 0; i < indices.size(); i++)
            indices[i] = (uint8_t)(rng.uniform() * 255.0f);
        std::vector<float> act_buf((size_t)N * K);
        std::memcpy(act_buf.data(), B.data<float>(), (size_t)N * K * sizeof(float));
        std::vector<float> out_buf((size_t)M * N);
        iters = 10;
        r.quant8_gflops = measure_gflops(M, N, K, iters, [&]() {
            kernel::quant8_gemm(indices.data(), codebook, act_buf.data(),
                              out_buf.data(), (int)M, (int)N, (int)K);
        });
    }
    std::cerr << "  Q8 done: " << r.quant8_gflops << " GFLOPS\n";

    // TL1: QUANT packed ternary weights — the kernel's layout is one
    // ceil(K/4)-byte packed row per (m,n) pair, i.e. M*N*ceil(K/4) bytes.
    {
        Tensor w(Shape{M * N * ((K + 3) / 4)}, DType::U8);
        Tensor a(Shape{K, N}, DType::F32);
        Tensor out(Shape{M, N}, DType::F32);
        for (int64_t i = 0; i < w.numel(); i++) w.data<uint8_t>()[i] = (uint8_t)(rng.uniform() * 3.0f);
        std::memcpy(a.data<float>(), B.data<float>(), (size_t)(K * N) * sizeof(float));
        iters = 10;
        r.tl1_gflops = measure_gflops(M, N, K, iters, [&]() {
            kernel::tl1_gemm(w, a, out, (int)M, (int)N, (int)K);
        });
    }
    std::cerr << "  TL1 done: " << r.tl1_gflops << " GFLOPS\n";

    return r;
}

int main(int argc, char** argv) {
    int64_t M = 256, N = 256, K = 256;
    if (argc > 3) { M = std::atol(argv[1]); N = std::atol(argv[2]); K = std::atol(argv[3]); }
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "=== QUANT Quantized GEMM Benchmarks (" << M << "x" << N << "x" << K << ") ===\n";
    QuantBenchResult r = bench_all_quant_gemm(M, N, K);
    std::cout << "FP32:  " << r.fp32_gflops << " GFLOPS\n";
    std::cout << "Q8:  " << r.quant8_gflops << " GFLOPS";
    if (r.fp32_gflops > 0 && r.quant8_gflops > 0)
        std::cout << "  (" << (r.quant8_gflops / r.fp32_gflops) << "x vs FP32)";
    std::cout << "\n";
    std::cout << "TL1:   " << r.tl1_gflops << " GFLOPS";
    if (r.fp32_gflops > 0 && r.tl1_gflops > 0)
        std::cout << "  (" << (r.tl1_gflops / r.fp32_gflops) << "x vs FP32)";
    std::cout << "\n";
    std::cout << "=== Done ===\n";
    return 0;
}
