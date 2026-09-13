#include "quant/kernel.h"
#include "quant/model.h"
#include "quant/tokenizer.h"
#include "quant/generator.h"
#include "quant/tensor.h"
#include "quant/math.h"

#include "quant/detail/cli_parse.h"
#include <iostream>
#include <string>
#include <cstring>
#include <chrono>
#include <fstream>
#include <vector>
#include <cmath>

struct BenchArgs {
    std::string kernel;
    int size = 1024;
    std::string model_path;
    std::string prompt = "test";
    bool run_inference = false;
    bool run_kernels = false;
};

static BenchArgs parse_args(int argc, char** argv) {
    BenchArgs args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            args.kernel = argv[++i];
            args.run_kernels = true;
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            args.size = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
            // BUGFIX (bug census): unbounded --size drove adjacent Tensor
            // allocs (size^3 floats ×3 tensors → OOM/hang on typos like
            // --size 1000000). Clamp to a sane bench range.
            if (args.size <= 0 || args.size > 4096) {
                std::cerr << "Error: --size needs 1..4096\n";
                exit(2);
            }
        } else if (strcmp(argv[i], "--inference") == 0) {
            args.run_inference = true;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: quant_bench [--kernel <name> --size N] [--inference --model m.quant --prompt p]\n";
            exit(0);
        }
    }
    return args;
}

static double now_sec() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t.time_since_epoch()).count();
}

static void bench_scalar_gemm(int M, int N, int K) {
    quant::Tensor A(quant::Shape{M, K}, quant::DType::F32);
    quant::Tensor B(quant::Shape{K, N}, quant::DType::F32);
    quant::Tensor C(quant::Shape{M, N}, quant::DType::F32);
    A.fill(1.0f);
    B.fill(2.0f);

    int warmup = 3;
    int iters = 20;
    for (int i = 0; i < warmup; i++)
        quant::kernel::scalar_gemm(A.data<float>(), B.data<float>(), C.data<float>(), M, N, K);

    double t0 = now_sec();
    for (int i = 0; i < iters; i++)
        quant::kernel::scalar_gemm(A.data<float>(), B.data<float>(), C.data<float>(), M, N, K);
    double dt = (now_sec() - t0) / iters;

    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = flops / dt * 1e-9;
    std::cout << "scalar_gemm M=" << M << " N=" << N << " K=" << K
              << ": " << dt * 1e6 << " us, " << gflops << " GFLOPS\n";
}

static void bench_avx2_gemm(int M, int N, int K) {
#if defined(QUANT_AVX2)
    quant::Tensor A(quant::Shape{M, K}, quant::DType::F32);
    quant::Tensor B(quant::Shape{K, N}, quant::DType::F32);
    quant::Tensor C(quant::Shape{M, N}, quant::DType::F32);
    A.fill(1.0f);
    B.fill(2.0f);

    int warmup = 3;
    int iters = 20;
    for (int i = 0; i < warmup; i++)
        quant::kernel::avx2_gemm(A.data<float>(), B.data<float>(), C.data<float>(), M, N, K);

    double t0 = now_sec();
    for (int i = 0; i < iters; i++)
        quant::kernel::avx2_gemm(A.data<float>(), B.data<float>(), C.data<float>(), M, N, K);
    double dt = (now_sec() - t0) / iters;

    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = flops / dt * 1e-9;
    std::cout << "avx2_gemm   M=" << M << " N=" << N << " K=" << K
              << ": " << dt * 1e6 << " us, " << gflops << " GFLOPS\n";
#else
    std::cout << "avx2_gemm   not available (QUANT_AVX2 not defined)\n";
#endif
}

static void bench_inference(const std::string& model_path, const std::string& prompt) {
    quant::DenseModel model;
    try {
        model.load(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return;
    }

    quant::BPETokenizer tokenizer;
    // BUGFIX (bug census): default-constructed BPETokenizer with no vocab
    // load — inference bench measured empty-vocab output. Try the model dir's
    // .vocab next to the model; warn when missing (degraded measurement).
    {
        std::string vp = model_path;
        size_t dot = vp.rfind('.');
        if (dot != std::string::npos) vp = vp.substr(0, dot);
        vp += ".vocab";
        std::ifstream vf(vp, std::ios::binary);
        if (vf.good()) {
            vf.close();
            try { tokenizer.load(vp); }
            catch (const std::exception& e) {
                std::cerr << "[Warning] tokenizer load failed: " << e.what()
                          << " (empty-vocab bench)\n";
            }
        } else {
            std::cerr << "[Warning] no .vocab next to model (looked for " << vp
                      << "); bench runs with empty vocab\n";
        }
    }
    quant::Generator gen(&model, &tokenizer);

    quant::SamplerConfig cfg;
    cfg.temperature = 0.0f;
    cfg.max_tokens = 128;

    double t0 = now_sec();
    auto result = gen.generate_full(prompt, cfg);
    double dt = now_sec() - t0;

    std::cout << "Inference: " << result.text.length() << " chars, "
              << dt << "s, "
              << result.tokens_per_sec << " tok/s\n";
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);
    std::cout << "QUANT Benchmark Suite\n";

    if (args.run_kernels) {
        if (args.kernel == "matmul" || args.kernel == "all") {
            bench_scalar_gemm(args.size, args.size, args.size);
            bench_avx2_gemm(args.size, args.size, args.size);
        }
        if (args.kernel == "all") {
            for (int s : {128, 256, 512, 1024}) {
                bench_scalar_gemm(s, s, s);
                bench_avx2_gemm(s, s, s);
            }
        }
    }

    if (args.run_inference) {
        if (args.model_path.empty()) {
            std::cerr << "Error: --model required for inference benchmark\n";
            return 1;
        }
        bench_inference(args.model_path, args.prompt);
    }

    if (!args.run_kernels && !args.run_inference) {
        std::cout << "No benchmark selected. Use --kernel or --inference.\n";
    }
    return 0;
}
