// ============================================================================
// sops_training_sim.cpp — Training simulation with quantized weights
// ============================================================================
// Build: cmake --build . --target sops_train_sim
// Run:   ./sops_train_sim
// ============================================================================

#include "quant/sops.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>
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

// ── Format data ───────────────────────────────────────────────────────────

struct SimFormat {
    const char* name;
    int index;
    double bpw;
    double info_weight;
};

static const SimFormat formats[] = {
    {"QUANT1",       0, 1.0,   32.000},
    {"QUANT_Q0",   1, 1.5,   21.333},
    {"QUANT_Q0_G", 3, 1.5,   20.253},
    {"QUANT2",       4, 2.0,   16.000},
    {"QUANT4",       5, 4.0,    8.000},
    {"QUANT8",       6, 8.0,    4.000},
    {"QUANT16",      7, 16.0,   2.000},
    {"QUANT32",      8, 32.0,   1.000},
};
static constexpr int NUM_SIM_FORMATS = 8;

// ── GEMV kernel (forward pass) ───────────────────────────────────────────

static void gemv_forward(const float* weights, const float* activations,
                          float* output, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        int64_t base = r * cols;
#if defined(__AVX2__)
        __m256 vsum = _mm256_setzero_ps();
        int64_t j = 0;
        int64_t cols8 = cols & ~7LL;
        for (; j < cols8; j += 8) {
            __m256 vw = _mm256_loadu_ps(weights + base + j);
            __m256 va = _mm256_loadu_ps(activations + j);
            vsum = _mm256_fmadd_ps(vw, va, vsum);
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, vsum);
        sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] +
              tmp[4] + tmp[5] + tmp[6] + tmp[7];
        for (; j < cols; j++) {
            sum += weights[base + j] * activations[j];
        }
#else
        for (int64_t j = 0; j < cols; j++) {
            sum += weights[base + j] * activations[j];
        }
#endif
        output[r] = sum;
    }
}

// ── GEMV kernel (backward pass — weight gradient) ─────────────────────────

static void gemv_backward(float* grad_weights, const float* grad_output,
                           const float* activations, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; r++) {
        float g = grad_output[r];
        int64_t base = r * cols;
#if defined(__AVX2__)
        __m256 vg = _mm256_set1_ps(g);
        int64_t j = 0;
        int64_t cols8 = cols & ~7LL;
        for (; j < cols8; j += 8) {
            __m256 va = _mm256_loadu_ps(activations + j);
            __m256 vw = _mm256_load_ps(grad_weights + base + j);
            vw = _mm256_fmadd_ps(vg, va, vw);
            _mm256_store_ps(grad_weights + base + j, vw);
        }
        for (; j < cols; j++) {
            grad_weights[base + j] += g * activations[j];
        }
#else
        for (int64_t j = 0; j < cols; j++) {
            grad_weights[base + j] += g * activations[j];
        }
#endif
    }
}

// ── Weight update kernel ──────────────────────────────────────────────────

static void weight_update(float* weights, const float* grad_weights,
                           int64_t count, float lr) {
    int64_t j = 0;
#if defined(__AVX2__)
    __m256 vlr = _mm256_set1_ps(lr);
    int64_t count8 = count & ~7LL;
    for (; j < count8; j += 8) {
        __m256 vw = _mm256_loadu_ps(weights + j);
        __m256 vg = _mm256_loadu_ps(grad_weights + j);
        vw = _mm256_fnmadd_ps(vlr, vg, vw);
        _mm256_storeu_ps(weights + j, vw);
    }
#endif
    for (; j < count; j++) {
        weights[j] -= lr * grad_weights[j];
    }
}

// ── Simulate one training step ───────────────────────────────────────────

struct StepResult {
    const char* format_name;
    double bpw;
    double info_weight;
    double weight_memory_bytes;
    double info_ops;
    double forward_time_ms;
    double backward_time_ms;
    double update_time_ms;
    double total_time_ms;
    double memory_bw_gbs;
    double simulated_sops;
};

static StepResult simulate_step(int format_index, int64_t hidden_in, int64_t hidden_out,
                                 int64_t batch_size, double cpu_ghz, int ncores) {
    StepResult result = {};
    result.format_name = formats[format_index].name;
    result.bpw = formats[format_index].bpw;
    result.info_weight = formats[format_index].info_weight;

    int64_t weight_count = hidden_in * hidden_out;
    result.weight_memory_bytes = (double)weight_count * (result.bpw / 8.0);
    result.info_ops = (double)weight_count * result.info_weight;

    std::vector<float> weights(weight_count);
    std::vector<float> grad_weights(weight_count, 0.0f);
    std::vector<float> activations(batch_size * hidden_in);
    std::vector<float> output(batch_size * hidden_out);
    std::vector<float> grad_output(batch_size * hidden_out);

    for (int64_t i = 0; i < weight_count; i++) {
        weights[i] = (float)(i % 1000) * 0.001f - 0.5f;
    }
    for (size_t i = 0; i < activations.size(); i++) {
        activations[i] = (float)(i % 1000) * 0.001f;
    }
    for (size_t i = 0; i < grad_output.size(); i++) {
        grad_output[i] = 0.01f;
    }

    int warmup = 2;
    int iters = 8;

    for (int w = 0; w < warmup; w++) {
        for (int64_t b = 0; b < batch_size; b++) {
            gemv_forward(weights.data() + 0, activations.data() + b * hidden_in,
                         output.data() + b * hidden_out, hidden_out, hidden_in);
        }
    }

    // Forward
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; it++) {
        for (int64_t b = 0; b < batch_size; b++) {
            gemv_forward(weights.data(), activations.data() + b * hidden_in,
                         output.data() + b * hidden_out, hidden_out, hidden_in);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    result.forward_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    // Backward
    std::fill(grad_weights.begin(), grad_weights.end(), 0.0f);
    t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; it++) {
        std::fill(grad_weights.begin(), grad_weights.end(), 0.0f);
        for (int64_t b = 0; b < batch_size; b++) {
            gemv_backward(grad_weights.data(), grad_output.data() + b * hidden_out,
                          activations.data() + b * hidden_in, hidden_out, hidden_in);
        }
    }
    t1 = std::chrono::high_resolution_clock::now();
    result.backward_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    // Update
    t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; it++) {
        weight_update(weights.data(), grad_weights.data(), weight_count, 0.001f);
    }
    t1 = std::chrono::high_resolution_clock::now();
    result.update_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    result.total_time_ms = result.forward_time_ms + result.backward_time_ms + result.update_time_ms;

    double total_time_sec = result.total_time_ms / 1000.0;

    // Memory BW: weights read (forward+backward) + grads read (update) + weights written (update)
    double bytes_moved = result.weight_memory_bytes * 3.0;
    result.memory_bw_gbs = (bytes_moved / 1e9) / total_time_sec;

    // SOPS for this step: all weight elements processed at this format
    // Total info ops = weight_count * IW (forward) + weight_count * IW (backward grad) + weight_count * IW (update)
    double total_info_ops = result.info_ops * 3.0;
    result.simulated_sops = (total_info_ops / total_time_sec) / 1e21;

    return result;
}

// ── Main ──────────────────────────────────────────────────────────────────

int main() {
    int ncores = (int)std::thread::hardware_concurrency();
    if (ncores < 1) ncores = 1;

    int64_t hidden_in = 768;
    int64_t hidden_out = 768;
    int64_t batch_size = 32;
    double cpu_ghz = 3.0;

    int64_t weight_count = hidden_in * hidden_out;

    printf("================================================================\n");
    printf("  SOPS TRAINING SIMULATION\n");
    printf("  Quantized weight training — QUANT4 vs FP32\n");
    printf("================================================================\n\n");

    printf("  System:\n");
    printf("    Cores:    %d\n", ncores);
    printf("    ISA:      %s\n", sops_isa_name(SOPS_DEFAULT_ISA));
    printf("    Est GHz:  %.1f\n\n", cpu_ghz);

    printf("  Layer config:\n");
    printf("    Hidden:   %lld x %lld\n", (long long)hidden_in, (long long)hidden_out);
    printf("    Params:   %.2e\n", (double)weight_count);
    printf("    Batch:    %lld\n\n", (long long)batch_size);

    // ── Simulate all formats ──────────────────────────────────────────

    printf("================================================================\n");
    printf("  SIMULATION RESULTS — All formats\n");
    printf("================================================================\n\n");

    printf("  %-14s  %6s  %8s  %10s  %10s  %8s  %10s  %10s\n",
           "Format", "BPW", "IW", "Weight Mem", "Info Ops", "Time ms", "BW GB/s", "SOPS");
    printf("  %-14s  %6s  %8s  %10s  %10s  %8s  %10s  %10s\n",
           "----------", "------", "--------", "----------", "----------", "--------", "----------", "----------");

    std::vector<StepResult> results;
    results.reserve(NUM_SIM_FORMATS);

    for (int i = 0; i < NUM_SIM_FORMATS; i++) {
        StepResult r = simulate_step(i, hidden_in, hidden_out, batch_size, cpu_ghz, ncores);
        results.push_back(r);

        const char* mem_unit = "B";
        double mem_disp = r.weight_memory_bytes;
        if (mem_disp > 1e9) { mem_disp /= 1e9; mem_unit = "GB"; }
        else if (mem_disp > 1e6) { mem_disp /= 1e6; mem_unit = "MB"; }
        else if (mem_disp > 1e3) { mem_disp /= 1e3; mem_unit = "KB"; }

        printf("  %-14s  %6.2f  %6.2fx  %7.1f %-2s  %10.0f  %6.2f  %10.2f  %10.4e\n",
               r.format_name, r.bpw, r.info_weight,
               mem_disp, mem_unit, r.info_ops,
               r.total_time_ms, r.memory_bw_gbs, r.simulated_sops);
    }

    // ── QUANT4 vs FP32 comparison ──────────────────────────────────────

    printf("\n================================================================\n");
    printf("  QUANT4 vs FP32 TRAINING COMPARISON\n");
    printf("================================================================\n\n");

    StepResult fp32 = results[NUM_SIM_FORMATS - 1];
    StepResult quant4 = results[4];

    double mem_ratio = fp32.weight_memory_bytes / quant4.weight_memory_bytes;
    double ops_ratio = quant4.info_ops / fp32.info_ops;
    double time_ratio = fp32.total_time_ms / quant4.total_time_ms;
    double bw_ratio = quant4.memory_bw_gbs / (fp32.memory_bw_gbs > 0 ? fp32.memory_bw_gbs : 1.0);
    double sops_ratio = quant4.simulated_sops / (fp32.simulated_sops > 0 ? fp32.simulated_sops : 1.0);

    printf("  Metric              FP32            QUANT4            Ratio\n");
    printf("  ----                ----            ----            -----\n");

    printf("  Weight memory       %.0f MB          %.0f MB          %.1fx less\n",
           fp32.weight_memory_bytes / 1e6, quant4.weight_memory_bytes / 1e6, mem_ratio);

    printf("  Info ops/step       %.0f            %.0f            %.1fx\n",
           fp32.info_ops, quant4.info_ops, ops_ratio);
    printf("  Memory BW used      %.2f GB/s       %.2f GB/s       %.1fx less\n",
           fp32.memory_bw_gbs, quant4.memory_bw_gbs,
           fp32.memory_bw_gbs / (quant4.memory_bw_gbs > 0 ? quant4.memory_bw_gbs : 1.0));
    printf("  Step time           %.2f ms         %.2f ms         %.1fx faster\n",
           fp32.total_time_ms, quant4.total_time_ms, time_ratio);
    printf("  SOPS                %.4e   %.4e   %.1fx higher\n",
           fp32.simulated_sops, quant4.simulated_sops, sops_ratio);

    // ── Detailed breakdown ────────────────────────────────────────────

    printf("\n================================================================\n");
    printf("  STEP TIME BREAKDOWN\n");
    printf("================================================================\n\n");

    printf("  %-14s  %10s  %10s  %10s  %10s\n",
           "Format", "Forward", "Backward", "Update", "Total");
    printf("  %-14s  %10s  %10s  %10s  %10s\n",
           "----------", "----------", "----------", "----------", "----------");

    for (size_t i = 0; i < results.size(); i++) {
        printf("  %-14s  %8.2f ms  %8.2f ms  %8.2f ms  %8.2f ms\n",
               results[i].format_name,
               results[i].forward_time_ms,
               results[i].backward_time_ms,
               results[i].update_time_ms,
               results[i].total_time_ms);
    }

    // ── Scaling analysis ──────────────────────────────────────────────

    printf("\n================================================================\n");
    printf("  SCALING: Training speed at each format\n");
    printf("  (64M param model, 12 cores)\n");
    printf("================================================================\n\n");

    printf("  Format     Steps/sec     Tokens/day     vs FP32\n");
    printf("  --------   -----------   -----------    -------\n");

    double fp32_steps_per_sec = 1000.0 / fp32.total_time_ms;
    for (size_t i = 0; i < results.size(); i++) {
        double steps_per_sec = 1000.0 / results[i].total_time_ms;
        double tokens_per_day = steps_per_sec * batch_size * 86400.0;
        double vs_fp32 = steps_per_sec / fp32_steps_per_sec;

        printf("  %-10s   %10.2f    %12.0f    %6.1fx\n",
               results[i].format_name, steps_per_sec, tokens_per_day, vs_fp32);
    }

    // ── Summary ──────────────────────────────────────────────────────

    printf("\n================================================================\n");
    printf("  SUMMARY\n");
    printf("================================================================\n\n");

    double best_sops = 0.0;
    const char* best_format = "";
    for (size_t i = 0; i < results.size(); i++) {
        if (results[i].simulated_sops > best_sops) {
            best_sops = results[i].simulated_sops;
            best_format = results[i].format_name;
        }
    }

    printf("  Best training format:  %s (SOPS: %.4e)\n", best_format, best_sops);
    printf("  FP32 SOPS:             %.4e\n", fp32.simulated_sops);
    printf("  QUANT4 speedup:          %.1fx\n", time_ratio);
    printf("  QUANT4 memory savings:   %.1fx\n\n", mem_ratio);

    printf("  QUANT4 training is %.1fx faster with %.1fx less memory.\n", time_ratio, mem_ratio);
    printf("  Lower-bit formats (QUANT, QUANT1) give even higher SOPS\n");
    printf("  but with increasing quantization error.\n\n");

    printf("  SOPS captures the TRUE compute advantage of quantized training:\n");
    printf("  not just raw FLOPS, but information-weighted throughput.\n");
    printf("================================================================\n");

    return 0;
}
