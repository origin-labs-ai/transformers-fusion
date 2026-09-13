#include "quant/model.h"
#include "quant/tensor.h"
#include "quant/math.h"
#include "quant/moe_variants.h"
#include "quant/backend.h"
#include "quant/autograd.h"
#include "quant/codebook.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

namespace quant {

static int64_t bench_gemm_fp32(int64_t M, int64_t N, int64_t K, int iters) {
    Tensor A({M, K}); float* ad = A.data<float>();
    Tensor B({K, N}); float* bd = B.data<float>();
    Tensor C({M, N});
    for (int64_t i = 0; i < M * K; i++) ad[i] = (float)(i % 100) / 100.0f;
    for (int64_t i = 0; i < K * N; i++) bd[i] = (float)(i % 50) / 50.0f;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) math::gemm(1.0f, A, B, 0.0f, C);
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / iters;
}

double bench_latency() {
    TransformerConfig cfg;
    cfg.hidden_size = 64; cfg.num_layers = 2; cfg.num_heads = 4;
    cfg.head_dim = 16; cfg.ffn_hidden_size = 128; cfg.vocab_size = 100; cfg.max_seq_len = 64;
    DenseModel model(cfg);
    Tensor input({1, 16}); Tensor pos({1, 16});
    float* id = input.data<float>(); float* pd = pos.data<float>();
    for (int i = 0; i < 16; i++) { id[i] = (float)(i % 10); pd[i] = (float)i; }
    auto start = std::chrono::steady_clock::now();
    int runs = 20;
    for (int i = 0; i < runs; i++) model.forward(input, pos);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / runs;
    return ms;
}

void run_all_benchmarks() {
    std::cout << "=== Transcender Benchmarks ===\n\n";

    printf("[GEMM] 256x256x256 FP32...\n");
    double ops = 2.0 * 256 * 256 * 256;
    double us = (double)bench_gemm_fp32(256, 256, 256, 20);
    printf("  Latency: %.0f us\n", us);
    if (us > 0) printf("  GFLOPS:  %.2f\n", ops / (us * 1e3));

    printf("\n[Latency] 64-dim 2-layer Transformer forward (16 tokens)...\n");
    double lat = bench_latency();
    printf("  Average: %.3f ms\n", lat);

    printf("\n=== Done ===\n");
}

} // namespace quant

int main() { quant::run_all_benchmarks(); return 0; }
