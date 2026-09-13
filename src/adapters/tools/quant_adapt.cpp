// ============================================================================
// quant_adapt.cpp — CLI: Convert any external format -> QUANT mixed-precision
// ============================================================================
// Usage:
//   quant_adapt --input <model.{gguf,safetensors,fp32,fp16,bin}> \
//             --output <output.quant> [--bpw 2.0] [--block-size 256] [--verbose]
// ============================================================================
#include "adapters/adapter_core.h"
#include "adapters/gguf_bridge.h"
#include "adapters/safetensors_bridge.h"
#include "adapters/ptq_bridge.h"
#include "quant/detail/cli_parse.h"

#include <iostream>
#include <cstring>
#include <cstdio>

using namespace quant::adapters;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "QUANT ADAPT — Convert any external format to QUANT mixed-precision\n\n"
            "Usage:\n"
            "  quant_adapt --input <model.{gguf,safetensors,fp32,fp16,bin}> \\\n"
            "            --output <output.quant> [--bpw 2.0] [--block-size 256] [--verbose]\n\n"
            "Options:\n"
            "  --input <path>       Input model (GGUF, Safetensors, FP32, FP16, FP8, INT8)\n"
            "  --output <path>      Output QUANT mixed-precision file\n"
            "  --bpw <float>        Target bits-per-weight (default: 2.0)\n"
            "  --block-size <N>     Block size for mixed allocation (default: 256)\n"
            "  --adaptive           Use activation-based importance scoring\n"
            "  --verbose            Print conversion details\n"
            "  -h, --help           Show this help\n\n"
            "Supported input formats:\n"
            "  GGUF (.gguf)         llama.cpp / GGML format\n"
            "  Safetensors (.safetensors)  HuggingFace format\n"
            "  FP32/FP16/FP8 raw    Raw weight dumps\n\n"
            "Examples:\n"
            "  quant_adapt --input model.safetensors --output model.quant --bpw 2.0\n"
            "  quant_adapt --input model.gguf --output model.quant --bpw 4.0 --verbose\n");
        return 0;
    }

    BridgeConfig cfg;
    cfg.target_bpw = 2.0f;
    cfg.block_size = 256;
    cfg.adaptive = false;
    std::string input_path;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)   input_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) cfg.output_path = argv[++i];
        else if (strcmp(argv[i], "--bpw") == 0 && i + 1 < argc) {
            float v = quant::cli_parse::parse_float("--bpw", argv[++i]);
            if (!(v >= 0.5f) || !(v <= 32.0f)) {
                std::fprintf(stderr, "Error: --bpw must be in [0.5, 32], got '%s'\n", argv[i]);
                return 2;
            }
            cfg.target_bpw = v;
        }
        else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
            int bs = quant::cli_parse::parse_int("--block-size", argv[++i]);
            if (bs < 32 || bs > 8192) {
                std::fprintf(stderr, "Error: --block-size must be in [32, 8192], got '%s'\n", argv[i]);
                return 2;
            }
            cfg.block_size = bs;
        }
        else if (strcmp(argv[i], "--adaptive") == 0)              cfg.adaptive = true;
        else if (strcmp(argv[i], "--verbose") == 0)               cfg.verbose = true;
    }

    if (input_path.empty() || cfg.output_path.empty()) {
        std::fprintf(stderr, "Error: --input and --output are required.\n");
        return 1;
    }

    ExternalFormat fmt = detect_format(input_path);
    std::fprintf(stdout, "QUANT ADAPT: input=%s format=%s -> %s bpw=%.2f\n",
                 input_path.c_str(), external_format_name(fmt),
                 cfg.output_path.c_str(), cfg.target_bpw);

    if (fmt == ExternalFormat::QUANT) {
        std::fprintf(stderr, "Error: input is already QUANT format. Nothing to convert.\n");
        return 1;
    }

    std::vector<AdapterTensor> tensors;

    if (fmt == ExternalFormat::GGUF) {
        tensors = load_gguf(input_path, cfg.verbose);
    } else if (fmt == ExternalFormat::SAFETENSORS) {
        tensors = load_safetensors(input_path, cfg.verbose);
    } else {
        AdapterTensor raw = load_raw_blob(input_path, fmt);
        if (!raw.empty()) tensors.push_back(std::move(raw));
    }

    if (tensors.empty()) {
        std::fprintf(stderr, "Error: failed to load tensors from %s\n", input_path.c_str());
        return 1;
    }

    std::fprintf(stdout, "Loaded %zu tensors\n", tensors.size());

    if (cfg.verbose) {
        int64_t total_params = 0;
        for (const auto& t : tensors) total_params += t.numel();
        std::fprintf(stdout, "Total parameters: %lld\n", (long long)total_params);
    }

    bool ok = write_quant_mixed(tensors, cfg);
    if (!ok) {
        std::fprintf(stderr, "Error: QUANT write failed.\n");
        return 1;
    }

    std::fprintf(stdout, "Success: QUANT model written to %s (target bpw=%.2f)\n",
                 cfg.output_path.c_str(), cfg.target_bpw);
    return 0;
}
