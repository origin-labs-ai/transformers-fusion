#include "quant/model.h"
#include "quant/tensor.h"
#include "quant/types.h"
#include "quant/math.h"
#include "quant/quant_format.h"
#include "quant/codebook.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <numeric>

static double now_sec() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t.time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Compute MSE between two float arrays
// ---------------------------------------------------------------------------
static double compute_mse(const float* a, const float* b, int64_t n) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        sum += d * d;
    }
    return sum / (double)n;
}

// ---------------------------------------------------------------------------
// Compute cosine similarity
// ---------------------------------------------------------------------------
static double cosine_sim(const float* a, const float* b, int64_t n) {
    double dot_ab = 0.0, dot_aa = 0.0, dot_bb = 0.0;
    for (int64_t i = 0; i < n; i++) {
        dot_ab += (double)a[i] * b[i];
        dot_aa += (double)a[i] * a[i];
        dot_bb += (double)b[i] * b[i];
    }
    double denom = std::sqrt(dot_aa * dot_bb);
    return denom > 1e-12 ? dot_ab / denom : 0.0;
}

// ---------------------------------------------------------------------------
// Quantize float array to QUANT8 (8-bit codebook lookup)
// ---------------------------------------------------------------------------
static void quantize_quant8(const float* src, uint8_t* dst,
                           const float* codebook, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        float val = src[i];
        int best = 0;
        float best_dist = 1e30f;
        for (int c = 0; c < 256; c++) {
            float d = val - codebook[c];
            if (d * d < best_dist) {
                best_dist = d * d;
                best = c;
            }
        }
        dst[i] = (uint8_t)best;
    }
}

// ---------------------------------------------------------------------------
// Quantize float array to QUANT4 (4-bit codebook lookup, 16 entries)
// ---------------------------------------------------------------------------
static void quantize_quant4(const float* src, uint8_t* dst,
                           const float* codebook, int64_t n) {
    for (int64_t i = 0; i < n; i += 2) {
        float v0 = src[i];
        float v1 = (i + 1 < n) ? src[i + 1] : 0.0f;
        int best0 = 0, best1 = 0;
        float best_d0 = 1e30f, best_d1 = 1e30f;
        for (int c = 0; c < 16; c++) {
            float d0 = v0 - codebook[c];
            if (d0 * d0 < best_d0) { best_d0 = d0 * d0; best0 = c; }
            float d1 = v1 - codebook[c];
            if (d1 * d1 < best_d1) { best_d1 = d1 * d1; best1 = c; }
        }
        dst[i / 2] = (uint8_t)((best1 << 4) | best0);
    }
}

// ---------------------------------------------------------------------------
// BitNet b1.58-style ternary {-1, 0, +1} baseline. NOT the canonical Q1_5
// codec (per-32 FP16 scale + sign bits) — labeled BITNET_158 below.
// ---------------------------------------------------------------------------
static void quantize_quant(const float* src, int8_t* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        float v = src[i];
        if (v > 0.33f) dst[i] = 1;
        else if (v < -0.33f) dst[i] = -1;
        else dst[i] = 0;
    }
}

// ---------------------------------------------------------------------------
// Quantize to QUANT1: {-1, +1}
// ---------------------------------------------------------------------------
static void quantize_quant1(const float* src, int8_t* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = src[i] >= 0.0f ? 1 : -1;
    }
}

// ---------------------------------------------------------------------------
// Dequantize ternary back to float
// ---------------------------------------------------------------------------
static void dequantize_quant(const int8_t* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) dst[i] = (float)src[i];
}

// ---------------------------------------------------------------------------
// Dequantize QUANT1 back to float
// ---------------------------------------------------------------------------
static void dequantize_quant1(const int8_t* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) dst[i] = (float)src[i];
}

// ---------------------------------------------------------------------------
// Simple perplexity estimation using forward pass on random model
// ---------------------------------------------------------------------------
static double estimate_perplexity(quant::Model& model, int64_t seq_len, int64_t vocab_size) {
    quant::Tensor input(quant::Shape{1, seq_len}, quant::DType::F32);
    quant::Tensor positions(quant::Shape{1, seq_len}, quant::DType::F32);
    float* id = input.data<float>();
    float* pd = positions.data<float>();
    for (int64_t i = 0; i < seq_len; i++) {
        id[i] = (float)(i % vocab_size);
        pd[i] = (float)i;
    }

    auto logits = model.forward(input, positions);
    float* ld = logits.data<float>();
    int64_t V = logits.shape().dims[2];

    double nll = 0.0;
    for (int64_t i = 0; i < seq_len; i++) {
        int target = (int)((i + 7) % vocab_size);
        float* row = ld + i * V;
        float mx = -INFINITY;
        for (int64_t j = 0; j < V; j++)
            if (row[j] > mx) mx = row[j];
        double sum = 0.0;
        for (int64_t j = 0; j < V; j++)
            sum += std::exp((double)row[j] - (double)mx);
        double prob = std::exp((double)row[target] - (double)mx) / sum;
        nll += -std::log(prob + 1e-10);
    }
    return std::exp(nll / (double)seq_len);
}

// ---------------------------------------------------------------------------
// QualityResult for table output
// ---------------------------------------------------------------------------
struct QualityResult {
    std::string format;
    int64_t param_count;
    double mse;
    double cosine;
    double perplexity;
    double quantize_us;
    double dequantize_us;
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== QUANT Quality / Perplexity Benchmarks ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);

    // --- Part 1: Weight Quantization Quality ---
    std::cout << "\n=== Weight Quantization Quality ===" << std::endl;
    std::cout << "Testing on random FP32 weights (4096 elements)" << std::endl;

    const int64_t N = 4096;
    std::vector<float> ref(N);
    for (int64_t i = 0; i < N; i++)
        ref[i] = (float)(i % 1000) / 500.0f - 1.0f;

    // Generate codebooks (simple uniform for benchmarking)
    float codebook8[256];
    float codebook4[16];
    for (int i = 0; i < 256; i++)
        codebook8[i] = -1.0f + 2.0f * i / 255.0f;
    for (int i = 0; i < 16; i++)
        codebook4[i] = -1.0f + 2.0f * i / 15.0f;

    std::vector<QualityResult> results;

    {
        // FP32 (reference – identity)
        QualityResult r;
        r.format = "FP32";
        r.param_count = N;
        r.mse = 0.0;
        r.cosine = 1.0;
        r.perplexity = 0.0;
        r.quantize_us = 0.0;
        r.dequantize_us = 0.0;
        results.push_back(r);
    }

    {
        // QUANT8
        std::vector<uint8_t> indices(N);
        std::vector<float> decoded(N);

        double t0 = now_sec();
        quantize_quant8(ref.data(), indices.data(), codebook8, N);
        double q_us = (now_sec() - t0) * 1e6;

        t0 = now_sec();
        for (int64_t i = 0; i < N; i++) decoded[i] = codebook8[indices[i]];
        double dq_us = (now_sec() - t0) * 1e6;

        QualityResult r;
        r.format = "QUANT8";
        r.param_count = N;
        r.mse = compute_mse(ref.data(), decoded.data(), N);
        r.cosine = cosine_sim(ref.data(), decoded.data(), N);
        r.perplexity = 0.0;
        r.quantize_us = q_us;
        r.dequantize_us = dq_us;
        results.push_back(r);
    }

    {
        // QUANT4
        std::vector<uint8_t> packed((N + 1) / 2, 0);
        std::vector<float> decoded(N, 0.0f);

        double t0 = now_sec();
        quantize_quant4(ref.data(), packed.data(), codebook4, N);
        double q_us = (now_sec() - t0) * 1e6;

        t0 = now_sec();
        for (int64_t i = 0; i < N; i += 2) {
            decoded[i] = codebook4[packed[i / 2] & 0x0F];
            if (i + 1 < N) decoded[i + 1] = codebook4[(packed[i / 2] >> 4) & 0x0F];
        }
        double dq_us = (now_sec() - t0) * 1e6;

        QualityResult r;
        r.format = "QUANT4";
        r.param_count = N;
        r.mse = compute_mse(ref.data(), decoded.data(), N);
        r.cosine = cosine_sim(ref.data(), decoded.data(), N);
        r.perplexity = 0.0;
        r.quantize_us = q_us;
        r.dequantize_us = dq_us;
        results.push_back(r);
    }

    {
        // BITNET_158: BitNet b1.58-style ternary baseline (log2(3) = 1.585 bpw)
        std::vector<int8_t> indices(N);
        std::vector<float> decoded(N);

        double t0 = now_sec();
        quantize_quant(ref.data(), indices.data(), N);
        double q_us = (now_sec() - t0) * 1e6;

        t0 = now_sec();
        dequantize_quant(indices.data(), decoded.data(), N);
        double dq_us = (now_sec() - t0) * 1e6;

        QualityResult r;
        r.format = "BITNET_158";
        r.param_count = N;
        r.mse = compute_mse(ref.data(), decoded.data(), N);
        r.cosine = cosine_sim(ref.data(), decoded.data(), N);
        r.perplexity = 0.0;
        r.quantize_us = q_us;
        r.dequantize_us = dq_us;
        results.push_back(r);
    }

    {
        // QUANT1
        std::vector<int8_t> indices(N);
        std::vector<float> decoded(N);

        double t0 = now_sec();
        quantize_quant1(ref.data(), indices.data(), N);
        double q_us = (now_sec() - t0) * 1e6;

        t0 = now_sec();
        dequantize_quant1(indices.data(), decoded.data(), N);
        double dq_us = (now_sec() - t0) * 1e6;

        QualityResult r;
        r.format = "QUANT1";
        r.param_count = N;
        r.mse = compute_mse(ref.data(), decoded.data(), N);
        r.cosine = cosine_sim(ref.data(), decoded.data(), N);
        r.perplexity = 0.0;
        r.quantize_us = q_us;
        r.dequantize_us = dq_us;
        results.push_back(r);
    }

    // Print quality table
    std::cout << "\n";
    std::cout << std::left
              << std::setw(12) << "Format"
              << std::setw(14) << "MSE"
              << std::setw(14) << "Cosine Sim"
              << std::setw(14) << "Quant (us)"
              << std::setw(14) << "Dequant (us)"
              << std::setw(14) << "BPW" << std::endl;
    std::cout << std::string(82, '-') << std::endl;

    for (auto& r : results) {
        float bpw = 0.0f;
        if (r.format == "FP32")        bpw = 32.0f;
        if (r.format == "QUANT8")        bpw = 8.0f;
        if (r.format == "QUANT4")        bpw = 4.0f;
        if (r.format == "BITNET_158")  bpw = 1.585f;  // log2(3) for ternary
        if (r.format == "QUANT1")        bpw = 1.0f;

        std::cout << std::left
                  << std::setw(12) << r.format
                  << std::setw(14) << std::setprecision(8) << r.mse
                  << std::setw(14) << std::setprecision(6) << r.cosine
                  << std::setw(14) << std::setprecision(2) << r.quantize_us
                  << std::setw(14) << std::setprecision(2) << r.dequantize_us
                  << std::setw(14) << std::setprecision(2) << bpw
                  << std::endl;
    }

    // --- Part 2: Model-level perplexity estimation ---
    std::cout << "\n=== Model-Level Quality (Perplexity Estimation) ===" << std::endl;

    quant::TransformerConfig cfg;
    cfg.vocab_size = 1000;
    cfg.hidden_size = 128;
    cfg.num_layers = 4;
    cfg.num_heads = 4;
    cfg.head_dim = 32;
    cfg.ffn_hidden_size = 256;
    cfg.max_seq_len = 512;

    quant::DenseModel model(cfg);
    int64_t params = model.param_count();
    std::cout << "  Model: " << (params / 1000) << "K params" << std::endl;

    const int64_t seq_len = 128;
    std::cout << "  Sequence length: " << seq_len << std::endl;

    double ppl = estimate_perplexity(model, seq_len, cfg.vocab_size);
    std::cout << "  Perplexity: " << std::setprecision(4) << ppl << std::endl;

    // --- Part 3: Comparison chart data ---
    std::cout << "\n=== Quality Comparison Chart Data ===" << std::endl;
    std::cout << "Format,BPW,MSE,CosineSimilarity,CompressionRatio" << std::endl;
    for (auto& r : results) {
        float bpw = 0.0f;
        if (r.format == "FP32")        bpw = 32.0f;
        if (r.format == "QUANT8")        bpw = 8.0f;
        if (r.format == "QUANT4")        bpw = 4.0f;
        if (r.format == "BITNET_158")  bpw = 1.585f;
        if (r.format == "QUANT1")        bpw = 1.0f;
        double compression = 32.0 / bpw;
        std::cout << r.format << ","
                  << std::setprecision(2) << bpw << ","
                  << std::setprecision(8) << r.mse << ","
                  << std::setprecision(6) << r.cosine << ","
                  << std::setprecision(1) << compression << std::endl;
    }

    std::cout << "\nDone." << std::endl;
    return 0;
}
