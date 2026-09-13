// ============================================================================
// quant_ptq.cpp — CLI: Post-Training Quantization -> QUANT mixed-precision
// ============================================================================
// Usage:
//   quant_ptq --input <path> [--format raw_fp32|raw_fp16|raw_fp8_e4m3|raw_fp8_e5m2|gguf|safetensors|quant] \\
//           --output <out.quant> [--bpw 1.50] [--block-size 256] [--verbose]
// ============================================================================
#include "adapters/ptq_bridge.h"
#include "adapters/adapter_core.h"
#include "adapters/gguf_bridge.h"
#include "adapters/safetensors_bridge.h"
#include "quant/detail/cli_parse.h"

#include <iostream>
#include <cstring>
#include <cstdio>

using namespace quant::adapters;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "QUANT PTQ — Post-Training Quantization to QUANT mixed-precision\n\n"
            "Usage:\n"
            "  quant_ptq --input <path> --output <out.quant> [options]\n\n"
            "Options:\n"
            "  --format <f>        Input format (default: auto-detect by magic)\n"
            "                      raw_fp32 | raw_fp16 | raw_fp8_e4m3 | raw_fp8_e5m2\n"
            "                      gguf | safetensors | quant\n"
            "  --bpw <float>       Target bits-per-weight (default: 1.50 = Q1_5)\n"
            "                      1.00 = all Q1, 1.50 = all Q1_5, 4.0 = all Q4,\n"
            "                       8.0 = all Q8, 16.0 = fp16, 32.0 = fp32\n"
            "  --block-size <N>    Block size for mixed allocation (default: 256)\n"
            "  --verbose           Print per-tensor stats\n"
            "  -h, --help          Show this help\n\n"
            "Examples:\n"
            "  quant_ptq --input model.gguf --output model.quant --bpw 1.50 --verbose\n"
            "  quant_ptq --input weights.fp32 --output model.quant --format raw_fp32 --bpw 4.0\n"
            "  quant_ptq --input model.quant --output model_quant.quant --bpw 1.0  (re-quantize)\n");
        return 0;
    }

    BridgeConfig cfg;
    cfg.target_bpw = 1.50f;
    cfg.block_size = 256;
    std::string input_path, fmt_str;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)
            input_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            cfg.output_path = argv[++i];
        else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc)
            fmt_str = argv[++i];
        else if (strcmp(argv[i], "--bpw") == 0 && i + 1 < argc) {
            // BUGFIX (bug census): bare atof fail-open (garbage → 1.5 default
            // silently kept... actually garbage → 0.0). Validated.
            cfg.target_bpw = quant::cli_parse::parse_float(argv[i-1], argv[++i]);
            if (!(cfg.target_bpw > 0.0f) || cfg.target_bpw > 32.0f) {
                std::fprintf(stderr, "Error: --bpw needs 0(exclusive)..32\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
            cfg.block_size = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
            if (cfg.block_size <= 0 || cfg.block_size > (1 << 20)) {
                std::fprintf(stderr, "Error: --block-size needs 1..1048576\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--verbose") == 0)
            cfg.verbose = true;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            // Already printed help above.
            return 0;
        }
    }

    if (input_path.empty() || cfg.output_path.empty()) {
        std::fprintf(stderr, "Error: --input and --output are required.\n");
        return 1;
    }

    ExternalFormat fmt;
    if (fmt_str.empty()) {
        fmt = detect_format(input_path);
    } else {
        if (fmt_str == "raw_fp32" || fmt_str == "fp32")      fmt = ExternalFormat::RAW_FP32;
        else if (fmt_str == "raw_fp16" || fmt_str == "fp16")  fmt = ExternalFormat::RAW_FP16;
        else if (fmt_str == "raw_fp8_e4m3")                    fmt = ExternalFormat::RAW_FP8_E4M3;
        else if (fmt_str == "raw_fp8_e5m2")                    fmt = ExternalFormat::RAW_FP8_E5M2;
        else if (fmt_str == "gguf")                            fmt = ExternalFormat::GGUF;
        else if (fmt_str == "safetensors")                      fmt = ExternalFormat::SAFETENSORS;
        else if (fmt_str == "quant")                             fmt = ExternalFormat::QUANT;
        else { std::fprintf(stderr, "Error: unknown format '%s'.\n", fmt_str.c_str()); return 1; }
    }

    std::fprintf(stdout, "QUANT PTQ: input=%s fmt=%s output=%s bpw=%.2f\n",
                 input_path.c_str(), external_format_name(fmt),
                 cfg.output_path.c_str(), cfg.target_bpw);

    bool ok = false;
    if (fmt == ExternalFormat::QUANT) {
        ok = ptq_requant_quant(input_path, cfg);
    } else if (fmt == ExternalFormat::GGUF) {
        ok = gguf_to_quant(input_path, cfg);
    } else if (fmt == ExternalFormat::SAFETENSORS) {
        ok = safetensors_to_quant(input_path, cfg);
    } else {
        ok = ptq_raw(input_path, fmt, cfg);
    }

    if (!ok) {
        std::fprintf(stderr, "Error: PTQ failed for %s\n", input_path.c_str());
        return 1;
    }
    std::fprintf(stdout, "Success: wrote QUANT mixed-precision model to %s\n", cfg.output_path.c_str());
    return 0;
}
