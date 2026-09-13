// ============================================================================
// safetensors_bridge.h — Safetensors loader -> AdapterTensors (FP32) -> QUANT mixed
// ============================================================================
#pragma once
#include "adapters/adapter_core.h"
#include <string>
#include <vector>

namespace quant {
namespace adapters {

// Parse a safetensors file and dequantize every tensor to FP32 AdapterTensors.
// Supports dtypes: F32, F16, BF16, I64, I32, I16, I8, U8, BOOL, F8_E4M3, F8_E5M2.
std::vector<AdapterTensor> load_safetensors(const std::string& path, bool verbose = false);

// Safetensors -> QUANT mixed-precision file.
bool safetensors_to_quant(const std::string& input_path, const BridgeConfig& cfg);

// L059: split block-FP8 expert tensors (Qwen3.8 class: grouped BF16/F32 work
// alongside per-expert FP8 blocks).
//
// Layout: weight tensor "<weight_key>" stored as raw F8_E4M3 (or F8_E5M2)
// bytes, plus a scale tensor "<scale_key>" holding one FP32/BF16/F16 scale
// per `block` consecutive weight elements (row-major). Dequant:
//   fp32[i] = fp8_direct(weight[i]) * scale[i / block]
// Returns an empty group (ok=false) on missing keys, dtype mismatch, or
// scale-count mismatch — honest failure, never synthetic data.
struct BlockFp8Group {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> data; // FP32, row-major
    int64_t block_size = 128;
    bool use_e4m3 = true;
    bool ok = false;
};

BlockFp8Group load_block_fp8_pair(const std::string& path,
                                  const std::string& weight_key,
                                  const std::string& scale_key,
                                  int64_t block = 128);

// Block-FP8 pair -> QUANT file through the standard mixed funnel.
bool block_fp8_safetensors_to_quant(const std::string& input_path,
                                    const std::string& weight_key,
                                    const std::string& scale_key,
                                    const BridgeConfig& cfg,
                                    int64_t block = 128);

} // namespace adapters
} // namespace quant
