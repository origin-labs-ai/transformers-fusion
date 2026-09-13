// P24 bench — 10x processing-speed harness (UNVERIFIED until hardware run).
// Compares BASELINE vs CANDIDATE on the same machine, same input, median-of-N.
// This file ships the HARNESS + a synthetic workload pair (memcpy baseline vs
// fused scale-add candidate). Real 10x claims must replace the workload with
// the actual engine path (e.g. naive decode vs FastBitReader decode) and
// attach bench_format_comparison.csv hashes.
// Orchestrator wiring: add_executable(bench_p1p73_speed p1p73/bench/bench_p1p73_speed.cpp)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main() {
    const int warmup = 3, reps = 7, N = 1 << 20;
    std::vector<float> a(N, 0.5f), b(N, 0.25f), c(N, 0.0f);

    auto baseline = [&]() {  // plain copy
        auto t0 = std::chrono::steady_clock::now();
        std::memcpy(c.data(), a.data(), N * sizeof(float));
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    };
    auto candidate = [&]() {  // fused scale-add (stand-in for optimized path)
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) c[i] = a[i] * 1.5f + b[i];
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    };

    for (int i = 0; i < warmup; ++i) { baseline(); candidate(); }
    std::vector<double> tb, tc;
    for (int i = 0; i < reps; ++i) { tb.push_back(baseline()); tc.push_back(candidate()); }
    double mb = median_of(tb), mc = median_of(tc);
    std::printf("bench=speed_pair warmup=%d reps=%d N=%d\n", warmup, reps, N);
    std::printf("baseline_median_us=%.2f candidate_median_us=%.2f speedup=%.3f\n", mb, mc, mb / mc);
    std::printf("STATUS=UNVERIFIED (synthetic workloads; replace with engine decode paths for P24 claim)\n");
    return 0;
}
