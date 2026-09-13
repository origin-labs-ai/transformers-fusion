// P35 bench — tiled GEMM parity + timing (REAL quant/math_tiled.h).
// Verifies gemm_tiled == naive within 1e-4 on 64x64 (one tile), then times
// both with warmup + median-of-N. Run Release for meaningful numbers.
// Orchestrator wiring:
//   add_executable(bench_p1p73_gemm p1p73/bench/bench_p1p73_gemm.cpp)
//   target_link_libraries(bench_p1p73_gemm PRIVATE quant_math quant_core)
#include "quant/math_tiled.h"
#include "quant/tensor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

static double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main() {
    const int64_t M = 64, N = 64, K = 64;
    quant::Tensor A(quant::Shape(M, K)), B(quant::Shape(K, N)), C(quant::Shape(M, N));
    for (int64_t i = 0; i < M * K; ++i) A.data<float>()[i] = (float)(i % 13) * 0.05f - 0.3f;
    for (int64_t i = 0; i < K * N; ++i) B.data<float>()[i] = (float)(i % 11) * 0.04f - 0.2f;

    // parity
    C.zero_();
    quant::math::gemm_tiled(1.0f, A, B, 0.0f, C);
    double max_err = 0.0;
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            double ref = 0.0;
            for (int64_t k = 0; k < K; ++k)
                ref += (double)A.data<float>()[m * K + k] * (double)B.data<float>()[k * N + n];
            max_err = std::max(max_err, std::fabs(ref - (double)C.data<float>()[m * N + n]));
        }
    std::printf("bench=tiled_gemm M=%lld N=%lld K=%lld max_abs_err=%.6f %s\n", (long long)M,
                (long long)N, (long long)K, max_err, max_err < 1e-4 ? "PARITY_OK" : "PARITY_FAIL");

    // timing: tiled vs naive-triple-loop
    const int warmup = 2, reps = 5;
    auto tiled = [&]() {
        C.zero_();
        auto t0 = std::chrono::steady_clock::now();
        quant::math::gemm_tiled(1.0f, A, B, 0.0f, C);
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    };
    std::vector<float> naive(M * N, 0.0f);
    auto naive_fn = [&]() {
        auto t0 = std::chrono::steady_clock::now();
        for (int64_t m = 0; m < M; ++m)
            for (int64_t n = 0; n < N; ++n) {
                double s = 0.0;
                for (int64_t k = 0; k < K; ++k)
                    s += (double)A.data<float>()[m * K + k] * (double)B.data<float>()[k * N + n];
                naive[m * N + n] = (float)s;
            }
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    };
    for (int i = 0; i < warmup; ++i) { tiled(); naive_fn(); }
    std::vector<double> tt, tn;
    for (int i = 0; i < reps; ++i) { tt.push_back(tiled()); tn.push_back(naive_fn()); }
    double mt = median_of(tt), mn = median_of(tn);
    std::printf("tiled_median_us=%.2f naive_median_us=%.2f speedup=%.3f\n", mt, mn, mn / mt);
    std::printf("STATUS=UNVERIFIED until Release run attached (parity gate is build-independent)\n");
    return max_err < 1e-4 ? 0 : 1;
}
