// ============================================================================
// inferno_bench.cpp — Inferno Benchmark: Type 1 (FMA stress) + Type 2 (GEMV)
// ============================================================================
// Build: cmake --build . --target bench_inferno
// Run:   ./bench_inferno
// ============================================================================

#include "quant/inferno.h"
#include "quant/inferno_kernel.h"
#include "quant/backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>

#if defined(_MSC_VER)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

using namespace quant;

// ── Thread pinning ────────────────────────────────────────────────────────

static void pin_thread_to_core(int core_id) {
#if defined(_MSC_VER)
    HANDLE proc = GetCurrentProcess();
    DWORD_PTR mask = 1ULL << core_id;
    SetThreadAffinityMask(GetCurrentThread(), mask);
#elif defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
    (void)core_id;
#endif
}

// ── CPU feature detection ─────────────────────────────────────────────────

static void detect_cpu_features() {
    printf("  ISA detection:\n");
#if defined(__AVX512F__)
    printf("    AVX-512:  YES\n");
#elif defined(__AVX2__)
    printf("    AVX-512:  NO (compile with -mavx512f)\n");
#else
    printf("    AVX-512:  NO\n");
#endif

#if defined(__AVX2__)
    printf("    AVX2:     YES\n");
#else
    printf("    AVX2:     NO\n");
#endif

#if defined(__SSE4_1__)
    printf("    SSE4.1:   YES\n");
#else
    printf("    SSE4.1:   NO\n");
#endif

    printf("    Default:  %s\n\n", inferno_isa_name(INFERNO_DEFAULT_ISA));
}

// ── Type 1: Synthetic FMA stress ──────────────────────────────────────────

static std::atomic<int64_t> g_fma_ops{0};

static void fma_stress_worker(int core_id, int64_t fma_count, InfernoCounter* out) {
    pin_thread_to_core(core_id);

    InfernoCounter sc;
    sc.cpu_ghz = 3.0;
    sc.start();

#if defined(__AVX2__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 ones = _mm256_set1_ps(1.0f);
    int64_t batch = fma_count / 32;
    for (int64_t i = 0; i < batch; i++) {
        acc0 = _mm256_fmadd_ps(ones, ones, acc0);
        acc1 = _mm256_fmadd_ps(ones, ones, acc1);
        acc2 = _mm256_fmadd_ps(ones, ones, acc2);
        acc3 = _mm256_fmadd_ps(ones, ones, acc3);
        acc0 = _mm256_fmadd_ps(ones, ones, acc0);
        acc1 = _mm256_fmadd_ps(ones, ones, acc1);
        acc2 = _mm256_fmadd_ps(ones, ones, acc2);
        acc3 = _mm256_fmadd_ps(ones, ones, acc3);
    }
    volatile float sink = _mm_cvtss_f32(_mm256_castps256_ps128(acc0));
    (void)sink;
    int64_t actual = batch * 32;
#else
    volatile float acc = 0.0f;
    for (int64_t i = 0; i < fma_count; i++) acc += 1.0f;
    int64_t actual = fma_count;
#endif

    sc.stop();
    g_fma_ops.fetch_add(actual);

    out->start_tsc = sc.start_tsc;
    out->end_tsc = sc.end_tsc;
}

// ── Type 2: Memory-bound GEMV ────────────────────────────────────────────

static void gemv_worker(int core_id, const float* weights, float* output,
                         const float* input, int64_t rows, int64_t cols,
                         int64_t reps, InfernoCounter* out) {
    pin_thread_to_core(core_id);

    InfernoCounter sc;
    sc.cpu_ghz = 3.0;
    sc.start();

    for (int64_t rep = 0; rep < reps; rep++) {
        for (int64_t r = 0; r < rows; r++) {
            float sum = 0.0f;
            int64_t base = r * cols;
#if defined(__AVX2__)
            __m256 vsum = _mm256_setzero_ps();
            int64_t j = 0;
            int64_t cols8 = cols & ~7LL;
            for (; j < cols8; j += 8) {
                __m256 vw = _mm256_loadu_ps(weights + base + j);
                __m256 vin = _mm256_loadu_ps(input + j);
                vsum = _mm256_fmadd_ps(vw, vin, vsum);
            }
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, vsum);
            sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                  tmp[4] + tmp[5] + tmp[6] + tmp[7];
            for (; j < cols; j++) {
                sum += weights[base + j] * input[j];
            }
#else
            for (int64_t j = 0; j < cols; j++) {
                sum += weights[base + j] * input[j];
            }
#endif
            output[r] = sum;
        }
    }

    sc.stop();
    out->start_tsc = sc.start_tsc;
    out->end_tsc = sc.end_tsc;
    out->total_elements.store((uint64_t)(rows * cols * reps));
    out->total_info_ops.store((uint64_t)((double)(rows * cols * reps) * 8.0));
}

// ── Format Inferno table ────────────────────────────────────────────────────

static void print_format_inferno_table(int64_t model_elements) {
    printf("  %-18s  %6s  %8s  %12s  %12s  %10s\n",
           "Format", "BPW", "IW", "Model Size", "Info Ops", "vs FP32");
    printf("  %-18s  %6s  %8s  %12s  %12s  %10s\n",
           "------------------", "------", "--------", "------------", "------------", "----------");

    for (int i = 0; i < INFERNO_NUM_FORMATS; i++) {
        auto& f = inferno_formats[i];
        double model_bytes = model_elements * (f.bpw / 8.0);
        double info_ops = (double)model_elements * f.info_weight;

        const char* unit = "B";
        double disp = model_bytes;
        if (disp > 1e9) { disp /= 1e9; unit = "GB"; }
        else if (disp > 1e6) { disp /= 1e6; unit = "MB"; }
        else if (disp > 1e3) { disp /= 1e3; unit = "KB"; }

        printf("  %-18s  %6.2f  %6.2fx  %8.1f %-2s  %12.0f  %6.1fx\n",
               f.name, f.bpw, f.info_weight, disp, unit, info_ops, f.info_weight);
    }

    printf("\n  Mix formats:\n\n");
    printf("  %-22s  %6s  %8s  %12s  %10s\n",
           "Mix Format", "Eff BPW", "IW", "Info Ops", "vs FP32");
    printf("  %-22s  %6s  %8s  %12s  %10s\n",
           "----------------------", "------", "--------", "------------", "----------");

    for (int i = 0; i < INFERNO_NUM_MIXES; i++) {
        auto& f = inferno_mix_formats[i];
        double info_ops = (double)model_elements * f.info_weight;
        printf("  %-22s  %6.2f  %6.2fx  %12.0f  %6.1fx\n",
               f.name, f.bpw, f.info_weight, info_ops, f.info_weight);
    }
}

// ── Theoretical max Inferno ─────────────────────────────────────────────────

static void print_theoretical_max(int ncores) {
    double cpu_ghz = 3.0;
    int fma_units = 2;
    int fma_width = 8;
    int fma_ops = 2;

    double raw_flops = (double)ncores * cpu_ghz * 1e9 * fma_units * fma_width * fma_ops;

    printf("  Cores:          %d\n", ncores);
    printf("  Est. clock:     %.1f GHz\n", cpu_ghz);
    printf("  FMA units:      %d per core\n", fma_units);
    printf("  FMA width:      %d FP32 (AVX2)\n", fma_width);
    printf("  Ops per FMA:    %d (mul+add)\n", fma_ops);
    printf("  Raw FLOPS:      %.2e\n\n", raw_flops);

    printf("  %-18s  %8s  %14s  %10s  %12s\n",
           "Format", "IW", "Theoretical", "Scale", "Gap to 1");
    printf("  %-18s  %8s  %14s  %10s  %12s\n",
           "------------------", "--------", "--------------", "----------", "------------");

    for (int i = 0; i < INFERNO_NUM_FORMATS; i++) {
        auto& f = inferno_formats[i];
        double eff_ops = raw_flops * f.info_weight / 2.0;
        double inferno_val = eff_ops / 1e21;
        double gap = 1.0 / (inferno_val > 0 ? inferno_val : 1e-30);
        printf("  %-18s  %6.2fx  %14.4e  %10s  %10.2e x\n",
               f.name, f.info_weight, inferno_val, inferno_unit_name(inferno_val), gap);
    }
}

// ── Main ──────────────────────────────────────────────────────────────────

int main() {
    int ncores = (int)std::thread::hardware_concurrency();
    if (ncores < 1) ncores = 1;

    printf("================================================================\n");
    printf("  Inferno BENCHMARK TOOL — Sextillion Operations Per Second\n");
    printf("  Type 1: Synthetic FMA Stress | Type 2: Memory-Bound GEMV\n");
    printf("================================================================\n\n");

    // ── CPU detection ────────────────────────────────────────────────

    detect_cpu_features();
    printf("  Cores: %d\n\n", ncores);

    // ── Base format reference ────────────────────────────────────────

    printf("================================================================\n");
    printf("  FORMAT REFERENCE — All Q formats + mixes\n");
    printf("================================================================\n\n");

    int64_t model_elements = 64000000LL;
    print_format_inferno_table(model_elements);

    // ── Theoretical max ──────────────────────────────────────────────

    printf("\n================================================================\n");
    printf("  THEORETICAL MAX Inferno (%d cores, AVX2 dual FMA)\n", ncores);
    printf("================================================================\n\n");

    print_theoretical_max(ncores);

    // ── Type 1: FMA stress test ──────────────────────────────────────

    printf("\n================================================================\n");
    printf("  TYPE 1: SYNTHETIC FMA STRESS TEST (%d threads)\n", ncores);
    printf("================================================================\n\n");

    int64_t fmas_per_thread = 10000000000LL / ncores;
    printf("  FMAs per thread: %.2e\n", (double)fmas_per_thread);
    printf("  Total FMAs:      %.2e\n\n", (double)(fmas_per_thread * ncores));

    g_fma_ops.store(0);

    std::vector<InfernoCounter> thread_counters(ncores);
    for (auto& c : thread_counters) {
        c.cpu_ghz = 3.0;
        c.reset();
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    threads.reserve(ncores);
    for (int t = 0; t < ncores; t++) {
        threads.emplace_back(fma_stress_worker, t, fmas_per_thread, &thread_counters[t]);
    }
    for (auto& th : threads) th.join();

    auto t1 = std::chrono::high_resolution_clock::now();
    double wall_dt = std::chrono::duration<double>(t1 - t0).count();

    int64_t total_fmas = g_fma_ops.load();
    double raw_flops = (double)total_fmas * 2.0 / wall_dt;
    double raw_gflops = raw_flops / 1e9;

    double theoretical_gflops = (double)ncores * 3.0e9 * 2.0 * 8.0 * 2.0 / 1e9;
    double fma_efficiency = raw_gflops / theoretical_gflops * 100.0;

    printf("  Wall time:          %.3f sec\n", wall_dt);
    printf("  Actual raw GFLOPS:  %.2f\n", raw_gflops);
    printf("  Theoretical:        %.2f\n", theoretical_gflops);
    printf("  FMA efficiency:     %.1f%%\n\n", fma_efficiency);

    printf("  Per-format Inferno from FLOPS (Type 1, FMA compute-bound):\n\n");
    printf("  %-18s  %8s  %14s  %10s  %12s\n",
           "Format", "IW", "Inferno", "Scale", "Gap to 1");
    printf("  %-18s  %8s  %14s  %10s  %12s\n",
           "------------------", "--------", "--------------", "----------", "------------");

    double best_fma_inferno = 0.0;
    double best_fma_gflops = raw_gflops;
    for (int i = 0; i < INFERNO_NUM_FORMATS; i++) {
        auto& f = inferno_formats[i];
        double flops = raw_gflops * 1e9;
        double eff_ops = flops * f.info_weight / 2.0;
        double inferno_val = eff_ops / 1e21;
        double gap = 1.0 / (inferno_val > 0 ? inferno_val : 1e-30);
        if (inferno_val > best_fma_inferno) best_fma_inferno = inferno_val;
        printf("  %-18s  %6.2fx  %14.4e  %10s  %10.2e x\n",
               f.name, f.info_weight, inferno_val, inferno_unit_name(inferno_val), gap);
    }

    printf("\n  Best Inferno: %.4e %s (from %.2f GFLOPS)\n", best_fma_inferno, inferno_unit_name(best_fma_inferno), best_fma_gflops);

    // ── Type 2: Memory-bound GEMV ────────────────────────────────────

    printf("\n================================================================\n");
    printf("  TYPE 2: MEMORY-BOUND GEMV (%d threads)\n", ncores);
    printf("================================================================\n\n");

    int64_t gemv_rows = 4096;
    int64_t gemv_cols = 15360;
    int64_t gemv_reps = 16;
    int64_t gemv_elements = gemv_rows * gemv_cols;

    double gemv_weight_mb = (double)(gemv_rows * gemv_cols) * 4.0 / 1e6;
    printf("  Matrix: %lld x %lld (FP32 weights)\n", (long long)gemv_rows, (long long)gemv_cols);
    printf("  Weight memory: %.1f MB\n", gemv_weight_mb);
    printf("  Repetitions:   %lld\n", (long long)gemv_reps);
    printf("  Total elements per step: %.2e\n\n", (double)(gemv_elements * gemv_reps));

    std::vector<float> weights(gemv_rows * gemv_cols);
    std::vector<float> input(gemv_cols);
    std::vector<float> output(gemv_rows);

    for (int64_t i = 0; i < gemv_rows * gemv_cols; i++) {
        weights[i] = (float)(i % 1000) * 0.001f;
    }
    for (int64_t i = 0; i < gemv_cols; i++) {
        input[i] = (float)(i % 1000) * 0.001f;
    }

    t0 = std::chrono::high_resolution_clock::now();

    std::vector<InfernoCounter> gemv_counters(ncores);
    for (auto& c : gemv_counters) {
        c.cpu_ghz = 3.0;
        c.reset();
    }

    threads.clear();
    for (int t = 0; t < ncores; t++) {
        threads.emplace_back(gemv_worker, t, weights.data(), output.data(),
                             input.data(), gemv_rows, gemv_cols, gemv_reps,
                             &gemv_counters[t]);
    }
    for (auto& th : threads) th.join();

    t1 = std::chrono::high_resolution_clock::now();
    wall_dt = std::chrono::duration<double>(t1 - t0).count();

    double gemv_total_flops = 2.0 * gemv_elements * gemv_reps;
    double gemv_gflops = gemv_total_flops / wall_dt / 1e9;

    double bytes_read = (double)(gemv_rows * gemv_cols * 4 + gemv_cols * 4) * gemv_reps;
    double bytes_written = (double)gemv_rows * 4 * gemv_reps;
    double effective_bw = (bytes_read + bytes_written) / wall_dt / 1e9;

    printf("  Wall time:     %.3f sec\n", wall_dt);
    printf("  Raw GFLOPS:    %.2f\n", gemv_gflops);
    printf("  Effective BW:  %.2f GB/s\n\n", effective_bw);

    printf("  Per-format Inferno from FLOPS (Type 2, memory-bound):\n\n");
    printf("  %-18s  %8s  %14s  %10s  %12s\n",
           "Format", "IW", "Inferno", "Scale", "Gap to 1");
    printf("  %-18s  %8s  %14s  %10s  %12s\n",
           "------------------", "--------", "--------------", "----------", "------------");

    double best_gemv_inferno = 0.0;
    for (int i = 0; i < INFERNO_NUM_FORMATS; i++) {
        auto& f = inferno_formats[i];
        double flops = gemv_gflops * 1e9;
        double eff_ops = flops * f.info_weight / 2.0;
        double inferno_val = eff_ops / 1e21;
        double gap = 1.0 / (inferno_val > 0 ? inferno_val : 1e-30);
        if (inferno_val > best_gemv_inferno) best_gemv_inferno = inferno_val;
        printf("  %-18s  %6.2fx  %14.4e  %10s  %10.2e x\n",
               f.name, f.info_weight, inferno_val, inferno_unit_name(inferno_val), gap);
    }

    printf("\n  Mix format Inferno from FLOPS (Type 2):\n\n");
    printf("  %-22s  %8s  %14s  %10s  %12s\n",
           "Mix Format", "IW", "Inferno", "Scale", "Gap to 1");
    printf("  %-22s  %8s  %14s  %10s  %12s\n",
           "----------------------", "--------", "--------------", "----------", "------------");

    for (int i = 0; i < INFERNO_NUM_MIXES; i++) {
        auto& f = inferno_mix_formats[i];
        double flops = gemv_gflops * 1e9;
        double eff_ops = flops * f.info_weight / 2.0;
        double inferno_val = eff_ops / 1e21;
        double gap = 1.0 / (inferno_val > 0 ? inferno_val : 1e-30);
        printf("  %-22s  %6.2fx  %14.4e  %10s  %10.2e x\n",
               f.name, f.info_weight, inferno_val, inferno_unit_name(inferno_val), gap);
    }

    printf("\n  Best GEMV Inferno: %.4e %s\n", best_gemv_inferno, inferno_unit_name(best_gemv_inferno));

    // ── Summary ──────────────────────────────────────────────────────

    printf("\n================================================================\n");
    printf("  SUMMARY\n");
    printf("================================================================\n\n");

    printf("  CPU:              %d cores\n", ncores);
    printf("  ISA:              %s\n", inferno_isa_name(INFERNO_DEFAULT_ISA));
    printf("  FMA efficiency:   %.1f%%\n", fma_efficiency);
    printf("  Raw GFLOPS:       %.2f (FMA) / %.2f (GEMV)\n", raw_gflops, gemv_gflops);
    printf("  Best FMA Inferno:    %.4e %s\n", best_fma_inferno, inferno_unit_name(best_fma_inferno));
    printf("  Best GEMV Inferno:   %.4e %s\n", best_gemv_inferno, inferno_unit_name(best_gemv_inferno));

    double best_overall = std::max(best_fma_inferno, best_gemv_inferno);
    double gap = 1.0 / (best_overall > 0 ? best_overall : 1e-30);
    printf("  Gap to 1 Inferno:    %.2e x\n\n", gap);

    printf("  -- Backend interconnect sweep (REAL-ONLY, honest) --\n\n");
    printf("  %-14s  %-9s  %10s  %14s\n", "Backend", "Available", "GFLOPS", "Inferno");
    printf("  %-14s  %-9s  %10s  %14s\n", "--------------", "---------", "----------", "--------------");
    auto scores = quant::inferno_kernel::sweep_backends(512, 512, 512);
    double best_be_gflops = 0.0;
    const char* best_be_name = "none";
    for (auto& s : scores) {
        printf("  %-14s  %-9s  %10.2f  %14.4e\n",
               quant::backend::backend_name(s.type),
               s.available ? "yes" : "no",
               s.gflops, s.inferno);
        if (s.gflops > best_be_gflops) { best_be_gflops = s.gflops; best_be_name = quant::backend::backend_name(s.type); }
    }
    printf("\n  Best live backend: %s at %.2f GFLOPS\n", best_be_name, best_be_gflops);
    double cuda_g = 0.0;
    for (auto& s : scores)
        if (s.type == quant::backend::BackendType::GPU_CUDA) cuda_g = s.gflops;
    if (cuda_g > 0 && best_be_gflops > 0)
        printf("  Best/CUDA ratio: %.2fx (99x claim needs this >= 99 on live CUDA)\n\n", best_be_gflops / cuda_g);
    else
        printf("  Best/CUDA ratio: UNMEASURABLE (CUDA offline here — 99x not verifiable on this host)\n\n");

    printf("  ── Gap Analysis: Path to 1 Inferno ──\n\n");

    struct ScalingStep {
        double multiplier;
        const char* label;
    };

    ScalingStep path[] = {
        {1.0, "Current (all cores, AVX2)"},
        {(double)ncores, nullptr},
        {(double)ncores * 2.0, "+ AVX512 (2x SIMD)"},
        {(double)ncores * 2.0 * 2.5, "+ Q2 (IW 16 vs 8)"},
        {(double)ncores * 2.0 * 2.5 * 4.0, "+ 4-node cluster"},
        {(double)ncores * 2.0 * 2.5 * 4.0 * 16.0, "+ 16-node cluster"},
        {(double)ncores * 2.0 * 2.5 * 4.0 * 16.0 * 64.0, "+ 1024-node supercomputer"},
    };

    for (size_t i = 0; i < sizeof(path) / sizeof(path[0]); i++) {
        double inferno_now = best_overall * path[i].multiplier;
        double gap_now = 1.0 / (inferno_now > 0 ? inferno_now : 1e-30);
        printf("  %zu. ", i + 1);
        if (i == 1)
            printf("Full utilization (all %d cores)", ncores);
        else
            printf("%s", path[i].label);
        printf("\n     %.4e Inferno  (gap: %.2e x)\n\n", inferno_now, gap_now);
    }

    printf("================================================================\n");

    return 0;
}
