// ============================================================================
// adapter_core.h — Common types + QUANT mixed-precision funnel for ADAPTER-EDITION
// ----------------------------------------------------------------------------
// Every external-format bridge (PTQ, GGUF, Safetensors) loads weights into
// AdapterTensor (always FP32), then funnels through write_quant_mixed() which
// via the parent project's FormatRegistry + codebooks and writes a .quant file.
//
// Design rule: NOTHING leaves the adapter edition in a foreign format. The only
// on-disk product is an QUANT mixed-precision (.quant) file — so a user who shows
// up with GGUF, safetensors, FP32/FP16/FP8 model always gets a native QUANT model.
// ============================================================================
#pragma once

#include "quant/types.h"
#include "quant/format_registry.h"

#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>

namespace quant {
namespace adapters {

// A weight tensor loaded from any external format, held as FP32 internally.
struct AdapterTensor {
    std::string name;
    std::vector<float> data;       // FP32 weights, row-major, flattened
    std::vector<int64_t> shape;    // e.g. {out_features, in_features} for Linear

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
    bool empty() const { return data.empty(); }
};

// Configuration for the QUANT conversion funnel.
struct BridgeConfig {
    // Single on-disk format for ALL blocks (default: Q2_GRP at
    // 2.625 BPW, highly recommended).
    // Note: this is ignored if compound != Q2_GRP
    Format format = Format::Q2_GRP;

    // Compound format (TWI_MIX / QUAD): when set to a MIX_*/QUAD_* RegFormat,
    // blocks are routed to the member formats by importance with the exact
    // registry ratios, so the file's average BPW equals the claimed
    // effective_bpw. Defaults to the single format above.
    RegFormat compound = RegFormat::Q2_GRP;

    // Kept for CLI/API compatibility; write_quant_mixed ignores it in favor of
    // `format`. The nearest matching format is chosen when set.
    float target_bpw = 2.0f;

    // Weights per block (block-shared scale / codebook granularity).
    int block_size = 256;

    // Adaptive mode: accepted for compatibility, unused by single-format writes.
    bool adaptive = false;

    // Verbose per-tensor stats.
    bool verbose = false;

    // Output .quant path (write_quant_mixed writes here).
    std::string output_path;
};

// Result of writing one tensor as mixed-precision blocks.
struct MixedWriteResult {
    uint32_t block_start = 0;
    uint32_t num_blocks = 0;
    float achieved_bpw = 0.0f;
};

// Single conversion funnel: take FP32 AdapterTensors, allocate a per-block
// mixed-precision format by importance (magnitude), quantize each block with
// the matching codebook/QUANT format logic, and write a .quant file.
// Returns true on success. This is the ONLY exit point of the adapter edition.
bool write_quant_mixed(const std::vector<AdapterTensor>& tensors,
                     const BridgeConfig& cfg);

// Quantize one FP32 block into the QUANT on-disk representation of `fmt`
// (codebook + indices). Exposed for streaming/sharded importers that build
// blocks incrementally and write the .quant file themselves.
bool quantize_block(Format fmt, const float* w, int n,
                    std::vector<uint8_t>& indices, std::vector<uint8_t>& codebook);

// Quality-first per-tensor format routing:
//  - Critical small tensors (A_log, dt_bias, norms, conv1d, biases) stay
//    lossless Q32 — low-bit quantization destroys them.
//  - Embedding tables (embed_tokens, lm_head) must NOT be sparsified: the
//    2.0 BPW sparse format keeps only ~8% of weights (92% zeros) which
//    breaks the model. They are routed to the best DENSE GRP format within
//    the claimed BPW (the native "GRP wins at 2x BPW" ladder).
Format select_tensor_format(const std::string& name, int64_t numel, Format base);

// Per-block format allocation for one tensor. Single formats pass through
// select_tensor_format(); compound (TWI_MIX/QUAD) formats route blocks to
// their member formats by importance quantiles using the exact registry
// ratios, so the file average BPW equals the claimed effective_bpw.
// Adaptive mixes (Q_MIX) allocate via FormatRegistry::allocate_mix_blocks
// (measured benefit-per-byte under the hard claimed-BPW budget).
// `mix` may be null for single-format writes. `plan_out` (optional) receives
// the per-block layout (starts/lens/formats); `shape` (optional) is the
// tensor shape used for row/column-aligned block segmentation. Returns one
// Format per block.
std::vector<Format> allocate_tensor_formats(const std::string& name, int64_t numel,
                                            const float* data, int block_size,
                                            Format base, const MixDescriptor* mix,
                                            FormatRegistry::MixBlockPlan* plan_out = nullptr,
                                            const std::vector<int64_t>* shape = nullptr);

// Look up a compound (MIX_*/QUAD_*) descriptor by RegFormat id; returns null
// when `rf` is a single format or unknown.
const MixDescriptor* find_mix_descriptor(RegFormat rf);

// Estimate the achieved bpw for a tensor at a given target (pre-write preview).
float estimate_mixed_bpw(int64_t num_weights, int block_size, float target_bpw);

// ── External-format dequantization helpers (foreign dtype -> FP32) ─────────

float fp16_to_float(uint16_t h);     // IEEE 754 binary16
float bf16_to_float(uint16_t b);     // bfloat16
float fp8_e4m3_to_float(uint8_t x);  // FP8 E4M3 (OCP)
float fp8_e5m2_to_float(uint8_t x);  // FP8 E5M2 (OCP)

// ── Format detection (by magic bytes / extension) ───────────────────────────

enum class ExternalFormat {
    RAW_FP32,
    RAW_FP16,
    RAW_FP8_E4M3,
    RAW_FP8_E5M2,
    GGUF,
    SAFETENSORS,
    QUANT,
    UNKNOWN
};

ExternalFormat detect_format(const std::string& path);
const char* external_format_name(ExternalFormat f);

} // namespace adapters
} // namespace quant
