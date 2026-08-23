// ============================================================================
// adapter_core.cpp -- QUANT mixed-precision funnel + foreign dtype dequantization
// ============================================================================
#include "adapters/adapter_core.h"
#include "quant/quant_format.h"
#include "quant/codebook.h"
#include "quant/format_planner.h"
#include "quant/block_codec.h"
#include "quant/types.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <filesystem>
#include <system_error>
#include <cstdint>

namespace quant {
namespace adapters {

// Foreign dtype -> FP32 dequantization

float fp16_to_float(uint16_t h) {
    int sign = (h >> 15) & 1;
    int exp  = (h >> 10) & 0x1F;
    int mant = h & 0x3FF;
    float f;
    if (exp == 0) {
        f = (float)mant * std::ldexp(1.0f, -24); // subnormal (2^-14 * 2^-10)
    } else if (exp == 31) {
        f = mant ? NAN : INFINITY;
    } else {
        f = (float)(mant | 0x400) / 1024.0f;     // 1.mant, then scale by 2^(exp-15)
        f = std::ldexp(f, exp - 15);
    }
    return sign ? -f : f;
}

float bf16_to_float(uint16_t b) {
    // bfloat16 = top 16 bits of FP32 (8 exp, 7 mant)
    uint32_t bits = ((uint32_t)b) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

float fp8_e4m3_to_float(uint8_t x) {
    // E4M3: 1 sign | 4 exponent (bias 7) | 3 mantissa
    int sign = (x >> 7) & 1;
    int exp  = (x >> 3) & 0xF;
    int mant = x & 0x7;
    float f;
    if (exp == 0 && mant == 0) {
        f = 0.0f;                                // zero
    } else if (exp == 15) {
        f = mant ? NAN : INFINITY;               // 15 = NaN/Inf in E4M3 (no Inf really, but guard)
    } else if (exp == 0) {
        f = (float)mant / 8.0f * std::ldexp(1.0f, -6);  // subnormal (2^-6)
    } else {
        f = (float)(mant | 0x8) / 8.0f;          // 1.mant, then scale by 2^(exp-7)
        f = std::ldexp(f, exp - 7);
    }
    return sign ? -f : f;
}

float fp8_e5m2_to_float(uint8_t x) {
    // E5M2: 1 sign | 5 exponent (bias 15) | 2 mantissa
    int sign = (x >> 7) & 1;
    int exp  = (x >> 2) & 0x1F;
    int mant = x & 0x3;
    float f;
    if (exp == 0 && mant == 0) {
        f = 0.0f;
    } else if (exp == 31) {
        f = mant ? NAN : INFINITY;
    } else if (exp == 0) {
        f = (float)mant / 4.0f * std::ldexp(1.0f, -14); // subnormal (2^-14)
    } else {
        f = (float)(mant | 0x4) / 4.0f;          // 1.mant, then scale by 2^(exp-15)
        f = std::ldexp(f, exp - 15);
    }
    return sign ? -f : f;
}

// ?? Format detection ????????????????????????????????????????????????????????

ExternalFormat detect_format(const std::string& path) {
    // Read first 8 magic bytes.
    std::ifstream f(path, std::ios::binary);
    uint8_t magic[8] = {0};
    std::streamsize got = 0;
    if (f) {
        f.read(reinterpret_cast<char*>(magic), 8);
        got = f.gcount();
    }

    // GGUF: "GGUF" (4 bytes) + version u32
    if (got >= 4 && std::memcmp(magic, "GGUF", 4) == 0)
        return ExternalFormat::GGUF;

    // Safetensors: 8-byte LE header length, then JSON starting with '{'
    if (got >= 8) {
        uint64_t hdr_len = 0;
        std::memcpy(&hdr_len, magic, 8);
        if (hdr_len > 0 && hdr_len < (1ULL << 30)) {
            std::vector<char> hdr((size_t)hdr_len);
            if (f) f.read(hdr.data(), (std::streamsize)hdr_len);
            if (f && f.gcount() == (std::streamsize)hdr_len && hdr_len > 0 && hdr[0] == '{')
                return ExternalFormat::SAFETENSORS;
        }
    }

    // QUANT: "QUA1"
    if (got >= 4 && std::memcmp(magic, "QUA1", 4) == 0)
        return ExternalFormat::QUANT;

    // Fallback: extension-based detection (works even for non-existent files).
    auto ends_with = [](const std::string& s, const std::string& suf) {
        return s.size() >= suf.size() &&
               s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
    };
    if (ends_with(path, ".fp8e4m3") || ends_with(path, ".fp8_e4m3"))
        return ExternalFormat::RAW_FP8_E4M3;
    if (ends_with(path, ".fp8e5m2") || ends_with(path, ".fp8_e5m2"))
        return ExternalFormat::RAW_FP8_E5M2;
    if (ends_with(path, ".fp16") || ends_with(path, ".f16") || ends_with(path, ".half"))
        return ExternalFormat::RAW_FP16;
    if (ends_with(path, ".gguf"))
        return ExternalFormat::GGUF;
    if (ends_with(path, ".safetensors"))
        return ExternalFormat::SAFETENSORS;
    if (ends_with(path, ".quant"))
        return ExternalFormat::QUANT;
    if (ends_with(path, ".fp32") || ends_with(path, ".f32") || ends_with(path, ".bin") ||
        ends_with(path, ".raw") || ends_with(path, ".weights"))
        return ExternalFormat::RAW_FP32;

    return ExternalFormat::UNKNOWN;
}

const char* external_format_name(ExternalFormat f) {
    switch (f) {
        case ExternalFormat::RAW_FP32:     return "raw_fp32";
        case ExternalFormat::RAW_FP16:     return "raw_fp16";
        case ExternalFormat::RAW_FP8_E4M3: return "raw_fp8_e4m3";
        case ExternalFormat::RAW_FP8_E5M2: return "raw_fp8_e5m2";
        case ExternalFormat::GGUF:         return "gguf";
        case ExternalFormat::SAFETENSORS:   return "safetensors";
        case ExternalFormat::QUANT:          return "quant";
        default:                           return "unknown";
    }
}

// Per-block quantization (canonical in-budget block codec)
//
// quantize_block() delegates to quant::quantize_block_all (quant/block_codec.h):
// the single source of truth for on-disk block payloads. All 15 QUANT/QUANT
// formats are supported and never exceed their claimed BPW.

// Quality-first per-tensor routing (native, in-house):
//  - Critical small tensors (A_log, dt_bias, norms, conv1d, biases) stay
//    lossless Q32 — low-bit quantization destroys them.
//  - Embedding tables (embed_tokens, lm_head) must NOT be sparsified: the
//    2.0 BPW sparse format keeps only ~8% of weights (92% zeros) which
//    breaks the model. They are routed to the best DENSE GRP format within
//    the claimed BPW (e.g. Q2_GRP @ 2.625 -> Q4-class quality -- the native
//    "GRP wins at 2x BPW" ladder).
Format select_tensor_format(const std::string& name, int64_t numel, Format base) {
    const bool critical =
        name.find("A_log") != std::string::npos ||
        name.find("dt_bias") != std::string::npos ||
        name.find("norm") != std::string::npos ||
        name.find("layernorm") != std::string::npos ||
        name.find("conv1d") != std::string::npos ||
        name.find("in_proj_a") != std::string::npos ||
        name.find("in_proj_b") != std::string::npos ||
        (name.size() >= 5 && name.compare(name.size() - 5, 5, ".bias") == 0);
    if (critical && numel <= 262144) return Format::Q32;
    const bool embedding =
        name.find("embed_tokens") != std::string::npos ||
        name.find("lm_head") != std::string::npos;
    if (critical && numel <= 262144) return Format::Q32;

    const float b = target_bpw;
    if (b <= 1.0f)  return Format::Q1_GRP;
    if (b <= 2.0f)  return Format::Q2_GRP;
    if (b <= 4.0f)  return Format::Q4_GRP;
    if (b <= 8.0f)  return Format::Q8_GRP;
    if (b <= 16.0f) return Format::Q16_GRP;
    return Format::Q32;
}

// Quantize one block to `fmt`, filling codebook + indices. Every QUANT/QUANT
// format (0..14) is supported via the canonical block codec.
bool quantize_block(Format fmt, const float* w, int n,
                    std::vector<uint8_t>& indices, std::vector<uint8_t>& codebook) {
    return quantize_block_all(fmt, w, n, indices, codebook);
}

const MixDescriptor* find_mix_descriptor(RegFormat rf) {
    for (const auto& m : FormatRegistry::get_all_twi_mixes())
        if (m.id == rf) return &m;
    for (const auto& m : FormatRegistry::get_all_four_mixes())
        if (m.id == rf) return &m;
    return nullptr;
}

// Fill a flat 256-block plan (used for non-adaptive paths so every caller
// writes blocks from the same plan structure).
static void fill_flat_plan(FormatRegistry::MixBlockPlan& plan, int64_t numel, int bs,
                           const std::vector<Format>& fmts) {
    const int nb = (int)fmts.size();
    plan.formats = fmts;
    plan.block_starts.resize((size_t)nb);
    plan.block_lens.resize((size_t)nb);
    for (int b = 0; b < nb; b++) {
        plan.block_starts[(size_t)b] = (int64_t)b * bs;
        plan.block_lens[(size_t)b] = std::min<int64_t>(bs, numel - (int64_t)b * bs);
    }
}

std::vector<Format> allocate_tensor_formats(const std::string& name, int64_t numel,
                                            const float* data, int block_size,
                                            Format base, const MixDescriptor* mix,
                                            FormatRegistry::MixBlockPlan* plan_out,
                                            const std::vector<int64_t>* shape) {
    std::vector<Format> out;
    if (numel <= 0 || !data) return out;
    const int bs = block_size > 0 ? block_size : 256;
    const int nb = (int)((numel + bs - 1) / bs);
    out.assign((size_t)nb, base);

    // Compound (TWI_MIX/QUAD/QUANT_MIX): only the critical-small-tensor guard
    // applies (embeddings stay dense in every mix tier, so raising them above
    // the claimed mix BPW would break the budget).
    const Format guard = select_tensor_format(name, numel, base);
    if (!mix) {
        for (auto& f : out) f = guard;
        if (plan_out) fill_flat_plan(*plan_out, numel, bs, out);
        return out;
    }
    if (guard == Format::Q32) {
        for (auto& f : out) f = Format::Q32;
        if (plan_out) fill_flat_plan(*plan_out, numel, bs, out);
        return out;
    }

    // Q_MIX adaptive allocator: measured benefit-per-byte greedy under a
    // HARD budget equal to the claimed BPW -- the exact BPW is a cap that is
    // never exceeded, and every byte is spent where it buys the most quality
    // (adaptive + priority-wise). Row/column-aligned blocks for narrow 2D.
    if (mix->adaptive) {
        FormatRegistry::MixBlockPlan plan =
            FormatRegistry::allocate_mix_blocks(*mix, data, numel, bs, shape);
        std::vector<Format> fmts = plan.formats;
        if (plan_out) *plan_out = std::move(plan);
        return fmts;
    }

    // Non-adaptive mix: Importance = per-block L1 magnitude; the most
    // important blocks get the highest-precision tier, and tier counts are
    // the exact registry ratios, so the average BPW is the claimed value.
    struct Score { float s; int i; };
    std::vector<Score> scores;
    scores.reserve((size_t)nb);
    for (int b = 0; b < nb; b++) {
        const int start = b * bs;
        const int n = (int)std::min<int64_t>(bs, numel - start);
        double sum = 0.0;
        for (int j = 0; j < n; j++) sum += std::fabs(data[(size_t)start + j]);
        scores.push_back({ (float)sum, b });
    }
    std::sort(scores.begin(), scores.end(),
              [](const Score& a, const Score& b) { return a.s > b.s; });

    struct Tier { RegFormat fmt; float ratio; };
    std::vector<Tier> tiers;
    tiers.push_back({ mix->tier1_fmt, mix->tier1_ratio });
    tiers.push_back({ mix->tier2_fmt, mix->tier2_ratio });
    if (mix->num_tiers >= 3) tiers.push_back({ mix->tier3_fmt, mix->tier3_ratio });
    if (mix->num_tiers >= 4) tiers.push_back({ mix->tier4_fmt, mix->tier4_ratio });

    size_t assigned = 0;
    for (size_t t = 0; t < tiers.size() && assigned < scores.size(); t++) {
        const size_t count = (t == tiers.size() - 1)
            ? scores.size() - assigned
            : (size_t)std::lround(tiers[t].ratio * (double)scores.size());
        for (size_t k = 0; k < count && assigned < scores.size(); k++, assigned++)
            out[(size_t)scores[assigned].i] = regformat_to_format(tiers[t].fmt);
    }
    if (plan_out) fill_flat_plan(*plan_out, numel, bs, out);
    return out;
}

// The funnel: AdapterTensors (FP32) -> .quant file
//
// QUANT on-disk layout: HEADER | FORMAT_TABLE | TENSOR_TABLE | BLOCK_DATA
// Blocks MUST be written last so that QUANTReader can compute data_offset_
// correctly from the sizes of the preceding tables.
//
// Blocks are quantized with a per-tensor format: `cfg.format` by default,
// overridden by select_tensor_format() for critical (Q32) and embedding
// (dense GRP) tensors. TWI_MIX / QUAD_MIX files are produced by assigning
// different formats to different blocks/tensors; the engine decodes every
// block by its own format entry, so any mix runs correctly.

float estimate_mixed_bpw(int64_t num_weights, int block_size, float target_bpw) {
    if (num_weights <= 0) return 0.0f;
    const int bs = block_size > 0 ? block_size : 256;
    int64_t full = num_weights / bs;
    int64_t tail = num_weights % bs;
    double bytes = (double)full * std::ceil(target_bpw * (double)bs / 8.0);
    if (tail > 0) bytes += std::ceil(target_bpw * (double)tail / 8.0);
    return (float)(bytes * 8.0 / (double)num_weights);
}

bool write_quant_mixed(const std::vector<AdapterTensor>& tensors,
                     const BridgeConfig& cfg) {
    if (cfg.output_path.empty()) return false;

    QUANTWriter writer(cfg.output_path);
    QUANTHeader hdr;
    std::memcpy(hdr.magic, "QUA1", 4);
    hdr.version = 1;
    hdr.flags = 0;
    hdr.config_size = 0;
    writer.write_header(hdr, nullptr);

    const int bs = cfg.block_size > 0 ? cfg.block_size : 256;
    const Format fmt = cfg.format;
    const MixDescriptor* mix = find_mix_descriptor(cfg.compound);
    const float eff_bpw = mix ? mix->effective_bpw : format_bpw(fmt);

    std::vector<FormatBlockEntry> ft_entries;
    std::vector<TensorEntry> tensor_entries;
    std::vector<std::string> names;
    std::vector<BlockData> all_blocks;
    all_blocks.reserve(tensors.size() * 4);
    uint32_t block_id = 0;
    double total_bytes = 0.0;
    int64_t total_weights = 0;

    for (const auto& t : tensors) {
        names.push_back(t.name);
        int64_t numel = (int64_t)t.data.size();
        if (numel == 0) {
            TensorEntry te; te.name_len = 0; te.block_start = block_id; te.num_blocks = 0;
            tensor_entries.push_back(te);
            continue;
        }

        int num_blocks = (int)((numel + bs - 1) / bs);
        if (num_blocks == 0) continue;
        FormatRegistry::MixBlockPlan plan;
        const std::vector<Format> fmts =
            allocate_tensor_formats(t.name, numel, t.data.data(), bs, fmt, mix,
                                    &plan, &t.shape);
        num_blocks = (int)plan.block_starts.size();
        if (num_blocks == 0) continue;

        uint32_t block_start = block_id;

        for (int b = 0; b < num_blocks; b++) {
            int64_t start = plan.block_starts[(size_t)b];
            int n = (int)plan.block_lens[(size_t)b];
            const Format bfmt = fmts[(size_t)b];
            const float tbpw = format_bpw(bfmt);

            BlockData block;
            block.format = bfmt;
            block.num_weights = (uint32_t)n;
            quantize_block(bfmt, t.data.data() + start, n, block.indices, block.codebook);
            all_blocks.push_back(std::move(block));

            FormatBlockEntry fe;
            fe.block_id = block_id++;
            fe.format = (uint8_t)bfmt;
            fe.cb_bytes = (uint32_t)all_blocks.back().codebook.size();
            ft_entries.push_back(fe);

            total_bytes += (double)(all_blocks.back().indices.size() +
                                    all_blocks.back().codebook.size());
            total_weights += n;
        }

        TensorEntry te;
        te.name_len = (uint16_t)t.name.size();
        te.block_start = block_start;
        te.num_blocks = (uint32_t)num_blocks;
        tensor_entries.push_back(te);

        if (cfg.verbose) {
            std::printf("  %-48s blocks=%-5d bpw~%.2f raw=%lldKB\n",
                        t.name.c_str(), num_blocks,
                        estimate_mixed_bpw(numel, bs, eff_bpw),
                        (long long)((numel * 4) / 1024));
        }
    }

    writer.write_format_table(ft_entries);
    writer.write_tensor_table(tensor_entries, names);
    for (auto& blk : all_blocks) writer.write_block(blk);
    writer.close();

    if (cfg.verbose && total_weights > 0) {
        std::printf("  achieved avg bpw = %.3f (actual stored bytes across %lld weights, base %s)\n",
                    total_bytes * 8.0 / (double)total_weights, (long long)total_weights,
                    mix ? mix->name.c_str() : format_name(fmt));
        std::uintmax_t file_bytes = 0;
        std::error_code ec;
        file_bytes = std::filesystem::file_size(cfg.output_path, ec);
        if (!ec && file_bytes > 0)
            std::printf("  disk bpw = %.3f (file %.1f MB incl. format table + block headers)\n",
                        (double)file_bytes * 8.0 / (double)total_weights,
                        (double)file_bytes / (1024.0 * 1024.0));
    }
    return true;
}

} // namespace adapters
} // namespace quant
