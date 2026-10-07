#include "quant/model.h"
#include "quant/transformer.h"
#include "quant/tensor.h"
#include "quant/types.h"
#include "quant/math.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>
#include <cstring>

static double now_sec() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t.time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Memory estimation for transformer parameters (bytes)
// ---------------------------------------------------------------------------
static int64_t estimate_model_bytes(const quant::TransformerConfig& cfg) {
    int64_t H = cfg.hidden_size;
    int64_t L = cfg.num_layers;
    int64_t V = cfg.vocab_size;
    int64_t FF = cfg.ffn_hidden_size;
    int64_t NH = cfg.num_heads;

    int64_t emb = V * H;
    int64_t per_layer = 4 * H * H + 2 * H * FF + 4 * H;
    int64_t head_dim = H / NH;
    int64_t attn = 4 * H * H + 2 * H * H;
    int64_t total = emb + L * (per_layer + attn) + 2 * H;
    return total * 4;
}

// ---------------------------------------------------------------------------
// bench_prefill: measure tokens/sec for prefill (processing S tokens at once)
// ---------------------------------------------------------------------------
static void bench_prefill(const quant::TransformerConfig& cfg, int64_t S) {
    quant::DenseModel model(cfg);
    int64_t B = 1;

    quant::Tensor input_ids(quant::Shape{B, S}, quant::DType::F32);
    quant::Tensor positions(quant::Shape{B, S}, quant::DType::F32);
    float* id = input_ids.data<float>();
    float* pd = positions.data<float>();
    for (int64_t i = 0; i < S; i++) {
        id[i] = (float)(i % cfg.vocab_size);
        pd[i] = (float)i;
    }

    int warmup = 3;
    for (int i = 0; i < warmup; i++)
        model.forward(input_ids, positions);

    int iters = 20;
    double t0 = now_sec();
    for (int i = 0; i < iters; i++)
        model.forward(input_ids, positions);
    double dt = (now_sec() - t0) / iters;

    double tok_per_sec = S / dt;
    std::cout << "    S=" << std::setw(5) << S
              << "  latency: " << std::fixed << std::setprecision(3) << (dt * 1000) << " ms"
              << "  throughput: " << std::fixed << std::setprecision(0) << tok_per_sec << " tok/s"
              << std::endl;
}

// ---------------------------------------------------------------------------
// bench_decode: measure single-token decode latency
// ---------------------------------------------------------------------------
static void bench_decode(const quant::TransformerConfig& cfg) {
    quant::DenseModel model(cfg);

    quant::Tensor single_in(quant::Shape{1, 1}, quant::DType::F32);
    quant::Tensor single_pos(quant::Shape{1, 1}, quant::DType::F32);
    single_in.data<float>()[0] = 0.0f;
    single_pos.data<float>()[0] = 0.0f;

    int warmup = 20;
    for (int i = 0; i < warmup; i++)
        model.forward(single_in, single_pos);

    int iters = 500;
    double t0 = now_sec();
    for (int i = 0; i < iters; i++)
        model.forward(single_in, single_pos);
    double dt = (now_sec() - t0) / iters;

    std::cout << "    decode: " << std::fixed << std::setprecision(3) << (dt * 1000) << " ms/token"
              << "  (" << std::fixed << std::setprecision(0) << (1.0 / dt) << " tok/s)" << std::endl;
}

// ---------------------------------------------------------------------------
// bench_model_size: report model memory at different quantization levels
// ---------------------------------------------------------------------------
static void bench_model_size(const quant::TransformerConfig& cfg) {
    std::cout << "\n=== Model Memory Report ===" << std::endl;
    std::cout << "  Config: hidden=" << cfg.hidden_size
              << " layers=" << cfg.num_layers
              << " heads=" << cfg.num_heads
              << " ffn=" << cfg.ffn_hidden_size << std::endl;

    int64_t fp32_bytes = estimate_model_bytes(cfg);

    struct { const char* name; float bpw; } formats[] = {
        {"FP32",   32.0f},
        {"Q8",    8.0f},
        {"Q4",    4.0f},
        {"QUANT",  1.50f},
        {"Q1",   1.0f},
    };

    std::cout << std::left
              << std::setw(12) << "Format"
              << std::setw(14) << "Size (MB)"
              << std::setw(14) << "Compression" << std::endl;
    std::cout << std::string(40, '-') << std::endl;

    for (auto& f : formats) {
        double ratio = 32.0f / f.bpw;
        double bytes = (double)fp32_bytes / ratio;
        double mb = bytes / (1024.0 * 1024.0);
        std::cout << std::left
                  << std::setw(12) << f.name
                  << std::setw(14) << std::fixed << std::setprecision(2) << mb
                  << std::setw(14) << std::fixed << std::setprecision(1) << ratio << "x"
                  << std::endl;
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== QUANT Inference Benchmarks ===" << std::endl;

    // Standard transformer configs
    struct TestCase {
        const char* name;
        quant::TransformerConfig cfg;
    };

    quant::TransformerConfig small;
    small.vocab_size = 32000;
    small.hidden_size = 256;
    small.num_layers = 4;
    small.num_heads = 4;
    small.head_dim = 64;
    small.ffn_hidden_size = 1024;
    small.max_seq_len = 2048;

    quant::TransformerConfig medium;
    medium.vocab_size = 32000;
    medium.hidden_size = 768;
    medium.num_layers = 12;
    medium.num_heads = 12;
    medium.head_dim = 64;
    medium.ffn_hidden_size = 3072;
    medium.max_seq_len = 2048;

    quant::TransformerConfig large;
    large.vocab_size = 32000;
    large.hidden_size = 1024;
    large.num_layers = 24;
    large.num_heads = 16;
    large.head_dim = 64;
    large.ffn_hidden_size = 4096;
    large.max_seq_len = 2048;

    TestCase cases[] = {
        {"Small  (256/4)",  small},
        {"Medium (768/12)", medium},
        {"Large  (1024/24)", large},
    };

    for (auto& tc : cases) {
        std::cout << "\n=== Model: " << tc.name << " ===" << std::endl;
        std::cout << "  Params: ~" << (quant::DenseModel(tc.cfg).param_count() / 1000000) << "M" << std::endl;

        std::cout << "\n  --- Prefill ---" << std::endl;
        for (int64_t S : {16, 32, 64, 128}) {
            if (S > tc.cfg.max_seq_len) continue;
            bench_prefill(tc.cfg, S);
        }

        std::cout << "\n  --- Decode ---" << std::endl;
        bench_decode(tc.cfg);
    }

    bench_model_size(medium);

    std::cout << "\nDone." << std::endl;
    return 0;
}
