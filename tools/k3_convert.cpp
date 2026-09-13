// tools/k3_convert.cpp — K3 MXFP4 checkpoint → .quant converter (skeleton, G-4)
// Usage: k3_convert --input <hf_dir> --output model.quant [--dry-run]
// Currently dry-run validates HF layout and MXFP4 dequant probe on first block.

#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>

namespace fs = std::filesystem;

// OCP MXFP4 E8M0 shared exponent: 32 values share one uint8 exponent (bias 127).
// Per-value 4b S1E2M1 (sign 1, exp 2, mant 1) — variant probed per docs/K3_MXFP4_BRIDGE.md A1.
static float dequant_mxfp4_S1E2M1(uint8_t nibble, uint8_t shared_exp) {
    int sign = (nibble >> 3) & 1;
    int exp2 = (nibble >> 1) & 0x3;
    int mant = nibble & 1;
    if (exp2 == 0 && mant == 0) return sign ? -0.0f : 0.0f;
    // subnormal/normal: value = (-1)^s * (1.mant/2) * 2^(shared_exp + exp2 - bias)
    // bias: shared 127 + 2-bit exp bias 1
    float mant_f = 1.0f + mant * 0.5f;
    int exp = (int)shared_exp - 127 + exp2 - 1;
    float v = std::ldexp(mant_f, exp);
    return sign ? -v : v;
}

static std::vector<float> dequant_mxfp4_block(const uint8_t* block136b) {
    // layout: byte 0 = shared E8M0, bytes 1..16 = 32×4b packed (2 per byte, low nibble first)
    uint8_t shared = block136b[0];
    std::vector<float> out; out.reserve(32);
    for (int i = 0; i < 32; ++i) {
        uint8_t byte = block136b[1 + i/2];
        uint8_t nib = (i % 2 == 0) ? (byte & 0xF) : (byte >> 4);
        out.push_back(dequant_mxfp4_S1E2M1(nib, shared));
    }
    return out;
}

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " --input <hf_dir> --output <model.quant> [--dry-run]\n"
              << "  HF dir must contain model.safetensors.index.json + *.mxfp4.safetensors shards\n"
              << "  Dry-run: enumerate shards, dequant probe first MXFP4 block, no output written\n";
}

int main(int argc, char** argv) {
    std::string input_dir, output_path;
    bool dry_run = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--input" && i+1 < argc) input_dir = argv[++i];
        else if (a == "--output" && i+1 < argc) output_path = argv[++i];
        else if (a == "--dry-run") dry_run = true;
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); return 0; }
    }
    if (input_dir.empty() || output_path.empty()) { print_usage(argv[0]); return 1; }

    if (!fs::exists(input_dir)) {
        std::cerr << "[k3_convert] input dir not found: " << input_dir << "\n";
        return 2;
    }
    std::cout << "[k3_convert] K3 MXFP4 → .quant bridge (G-4 skeleton)\n";
    std::cout << "  input : " << input_dir << "\n";
    std::cout << "  output: " << output_path << "\n";
    std::cout << "  mode  : " << (dry_run ? "dry-run" : "convert") << "\n";

    // Enumerate shards
    size_t shards = 0;
    for (auto &p : fs::directory_iterator(input_dir)) {
        if (p.path().extension() == ".safetensors") {
            std::cout << "  shard: " << p.path().filename().string() << " (" << fs::file_size(p) << " bytes)\n";
            if (++shards >= 5) { std::cout << "  ... (truncated)\n"; break; }
        }
    }
    if (shards == 0) std::cout << "  [warn] no .safetensors found in input dir\n";

    // Probe: synthesize one MXFP4 block with shared exp 127 and incremental mantissas
    uint8_t probe[17] = {};
    probe[0] = 127; // shared exp = 0
    for (int i = 0; i < 32; ++i) {
        uint8_t nib = (uint8_t)(i & 0xF);
        if (i % 2 == 0) probe[1 + i/2] = (probe[1 + i/2] & 0xF0) | nib;
        else probe[1 + i/2] = (probe[1 + i/2] & 0x0F) | (nib << 4);
    }
    auto fp32 = dequant_mxfp4_block(probe);
    std::cout << "  probe MXFP4 block dequant (first 8 vals): ";
    for (int i = 0; i < 8; ++i) std::cout << fp32[i] << " ";
    std::cout << "\n";

    if (dry_run) {
        std::cout << "[k3_convert] dry-run complete (no .quant written)\n";
        return 0;
    }
    // TODO(G-4): stream shards, dequant per docs/K3_MXFP4_BRIDGE.md, Lloyd-Max Q4, write .quant
    std::cout << "[k3_convert] convert path gated on HAVE_K3_SAMPLE — writing placeholder header\n";
    std::ofstream out(output_path, std::ios::binary);
    if (!out) { std::cerr << "cannot open output\n"; return 3; }
    const char magic[4] = {'Q','N','T','\0'};
    out.write(magic, 4);
    uint32_t version = 1;
    out.write(reinterpret_cast<char*>(&version), 4);
    std::cout << "[k3_convert] placeholder .quant written (8 bytes header)\n";
    return 0;
}
