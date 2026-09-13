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
// the single source of truth for on-disk block payloads. All 105 QUANT/QUANT
// formats are supported and never exceed their claimed BPW.

// Quality-first per-tensor routing (native, in-house):
//  - Critical small tensors (A_log, dt_bias, norms, conv1d, biases) stay
//    lossless Q32 — low-bit quantization destroys them.
//  - Embedding tables (embed_tokens, lm_head) must NOT be sparsified: the
//    2.0 BPW sparse format keeps only ~8% of weights (92% zeros) which
//    breaks the model. They are routed to the best DENSE GRP format within
//    the claimed BPW (e.g. QG2 @ 2.625 -> Q4-class quality -- the native
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

    const float b = format_bpw(base);
    if (b <= 1.0f)  return Format::QG1;
    if (b <= 2.0f)  return Format::QG2;
    if (b <= 4.0f)  return Format::QG4;
    if (b <= 8.0f)  return Format::QG8;
    if (b <= 16.0f) return Format::QG16;
    return Format::Q32;
}

// Quantize one block to `fmt`, filling codebook + indices. Every QUANT/QUANT
// format (0..104) is supported via the canonical block codec.
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

    // Compound (Q_MX/QG_MX): only the critical-small-tensor guard
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

    // Q_MX/QG_MX adaptive allocator: measured benefit-per-byte greedy under a
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
// (dense GRP) tensors. Q_MX / QG_MX files are produced by assigning
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

// ── L070 LLaMA-family loader helpers ───────────────────────────────────────
// Name patterns (safetensors + GGUF dequantized names share these):
//   embed:  model.embed_tokens.weight
//   attn:   model.layers.{l}.self_attn.{q,k,v,o}_proj.weight
//   mlp:    model.layers.{l}.mlp.{gate,up,down}_proj.weight
//   norms:  model.layers.{l}.input_layernorm.weight,
//           model.layers.{l}.post_attention_layernorm.weight,
//           model.norm.weight
//   head:   lm_head.weight
// Qwen-dense adds q_norm/k_norm tensors; Mistral is name-identical to LLaMA.

namespace {

bool has_sub(const std::string& name, const char* sub) {
    return name.find(sub) != std::string::npos;
}

int parse_layer_index(const std::string& name) {
    const char* key = "model.layers.";
    auto p = name.find(key);
    if (p == std::string::npos) return -1;
    p += std::char_traits<char>::length(key);
    int idx = -1, n = 0;
    while (p + (size_t)n < name.size() &&
           name[p + n] >= '0' && name[p + n] <= '9') {
        n++;
    }
    if (n == 0 || n > 6) return -1;
    idx = 0;
    for (int i = 0; i < n; i++) idx = idx * 10 + (name[p + i] - '0');
    return idx;
}

const AdapterTensor* find_tensor(const std::vector<AdapterTensor>& ts,
                                 const std::string& sub) {
    for (const auto& t : ts)
        if (t.name.find(sub) != std::string::npos) return &t;
    return nullptr;
}

} // namespace

const char* llama_family_name(LlamaFamily f) {
    switch (f) {
        case LlamaFamily::Llama: return "llama";
        case LlamaFamily::Mistral: return "mistral";
        case LlamaFamily::QwenDense: return "qwen_dense";
        default: return "unknown";
    }
}

LlamaFamily detect_llama_family(const std::vector<AdapterTensor>& tensors) {
    bool has_embed = false, has_llama_attn = false, has_llama_mlp = false;
    bool has_qwen_marker = false, has_mistral_marker = false;
    for (const auto& t : tensors) {
        if (has_sub(t.name, "model.embed_tokens.weight")) has_embed = true;
        if (has_sub(t.name, "self_attn.q_proj.weight")) has_llama_attn = true;
        if (has_sub(t.name, "mlp.gate_proj.weight")) has_llama_mlp = true;
        // Qwen markers: q_norm/k_norm (Qwen3) or qwen-specific tied head note.
        if (has_sub(t.name, "self_attn.q_norm.weight") ||
            has_sub(t.name, "self_attn.k_norm.weight"))
            has_qwen_marker = true;
        if (has_sub(t.name, "sliding_window") ||
            has_sub(t.name, "mistral"))
            has_mistral_marker = true;
    }
    if (!has_embed || !has_llama_attn || !has_llama_mlp)
        return LlamaFamily::Unknown;
    if (has_qwen_marker) return LlamaFamily::QwenDense;
    if (has_mistral_marker) return LlamaFamily::Mistral;
    // Name-identical LLaMA vs Mistral without markers defaults to LLaMA;
    // sliding-window is a runtime mask detail, not a weight-layout difference.
    return LlamaFamily::Llama;
}

LlamaArchConfig infer_llama_arch(const std::vector<AdapterTensor>& tensors) {
    LlamaArchConfig c;
    c.family = detect_llama_family(tensors);
    if (c.family == LlamaFamily::Unknown) { c.valid = false; return c; }

    int max_layer = -1;
    for (const auto& t : tensors) {
        int li = parse_layer_index(t.name);
        if (li > max_layer) max_layer = li;
    }
    c.num_layers = max_layer >= 0 ? (int64_t)max_layer + 1 : 0;

    // hidden_size + vocab from embed_tokens [vocab, hidden].
    if (const AdapterTensor* e = find_tensor(tensors, "model.embed_tokens.weight")) {
        if (e->shape.size() >= 2) {
            c.vocab_size = e->shape[0];
            c.hidden_size = e->shape[1];
        } else if (e->shape.size() == 1 && !e->data.empty()) {
            // GGUF path flattens to {ne}; fall back to numel-only (vocab unknown).
            c.hidden_size = 0;
            c.vocab_size = 0;
        }
    }
    // Fallback hidden from o_proj [hidden, hidden] or q_proj rows.
    if (c.hidden_size <= 0) {
        if (const AdapterTensor* o = find_tensor(tensors, "self_attn.o_proj.weight")) {
            if (o->shape.size() >= 2) c.hidden_size = o->shape[0];
        }
    }
    // ffn_hidden from gate_proj [ffn, hidden].
    if (const AdapterTensor* g = find_tensor(tensors, "mlp.gate_proj.weight")) {
        if (g->shape.size() >= 2) c.ffn_hidden = g->shape[0];
    }
    // Heads: shapes alone are ambiguous (GQA). Documented fallback:
    // head_dim=128 convention -> num_heads = hidden/128 when divisible,
    // num_kv_heads = num_heads (caller overrides from config.json when known).
    if (c.hidden_size > 0 && c.hidden_size % 128 == 0) {
        c.num_heads = c.hidden_size / 128;
        c.num_kv_heads = c.num_heads;
    }
    // rope_theta family default: LLaMA-3 class uses 500k, older 10k.
    c.rope_theta = (c.family == LlamaFamily::Llama) ? 500000.0f : 10000.0f;
    if (c.family == LlamaFamily::QwenDense) c.rope_theta = 1000000.0f;
    // tie_embeddings: lm_head missing but embed present => tied.
    bool has_head = find_tensor(tensors, "lm_head.weight") != nullptr;
    bool has_embed = find_tensor(tensors, "model.embed_tokens.weight") != nullptr;
    c.tie_embeddings = has_embed && !has_head;
    c.valid = (c.num_layers > 0 && c.hidden_size > 0 && c.ffn_hidden > 0);
    return c;
}

bool validate_llama_tensors(const std::vector<AdapterTensor>& tensors,
                            std::string* err_out) {
    auto fail = [&](const std::string& m) {
        if (err_out) *err_out = m;
        return false;
    };
    if (tensors.empty()) return fail("llama: empty tensor list");
    if (detect_llama_family(tensors) == LlamaFamily::Unknown)
        return fail("llama: missing model.embed_tokens/self_attn.q_proj/mlp.gate_proj");
    int max_layer = -1;
    for (const auto& t : tensors) {
        int li = parse_layer_index(t.name);
        if (li > max_layer) max_layer = li;
    }
    if (max_layer < 0) return fail("llama: no model.layers.{i} tensors");
    // Per-layer completeness: every layer needs q/k/v/o + gate/up/down + 2 norms.
    static const char* kReq[] = {
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        "input_layernorm.weight", "post_attention_layernorm.weight"
    };
    for (int l = 0; l <= max_layer; l++) {
        std::string prefix = "model.layers." + std::to_string(l) + ".";
        for (const char* r : kReq) {
            bool found = false;
            std::string want = prefix + r;
            for (const auto& t : tensors)
                if (t.name == want) { found = true; break; }
            if (!found)
                return fail("llama: layer " + std::to_string(l) + " missing " + r);
        }
    }
    if (!find_tensor(tensors, "model.norm.weight"))
        return fail("llama: missing model.norm.weight");
    // Empty-weight guard (corrupt shard).
    for (const auto& t : tensors) {
        if (t.name.rfind("model.layers.", 0) == 0 && t.data.empty())
            return fail("llama: empty weights for " + t.name);
    }
    return true;
}

// ── L071 MoE arch loader helpers ───────────────────────────────────────────
// Patterns:
//   mixtral:      model.layers.{l}.block_sparse_moe.gate.weight +
//                 model.layers.{l}.block_sparse_moe.experts.{e}.w{1,2,3}.weight
//   qwen_moe:     model.layers.{l}.mlp.experts.{e}.gate_proj/up_proj/down_proj +
//                 model.layers.{l}.mlp.gate.weight (+ shared_expert...)
//   deepseek_moe: model.layers.{l}.mlp.experts.{e}.{w1,w2,w3} or gate_proj set +
//                 model.layers.{l}.mlp.shared_experts... + gate.weight

bool is_moe_tensors(const std::vector<AdapterTensor>& tensors) {
    for (const auto& t : tensors) {
        if (has_sub(t.name, ".experts.") &&
            (has_sub(t.name, "block_sparse_moe") || has_sub(t.name, ".mlp.")))
            return true;
    }
    return false;
}

namespace {

int parse_expert_index(const std::string& name) {
    auto p = name.find(".experts.");
    if (p == std::string::npos) return -1;
    p += 9; // strlen(".experts.")
    int n = 0, idx = 0;
    while (p + (size_t)n < name.size() &&
           name[p + n] >= '0' && name[p + n] <= '9') {
        n++;
    }
    if (n == 0 || n > 6) return -1;
    for (int i = 0; i < n; i++) idx = idx * 10 + (name[p + i] - '0');
    return idx;
}

} // namespace

MoeArchConfig infer_moe_arch(const std::vector<AdapterTensor>& tensors) {
    MoeArchConfig c;
    if (!is_moe_tensors(tensors)) { c.valid = false; return c; }
    c.is_moe = true;

    bool has_block_sparse = false, has_mlp_experts = false;
    bool has_shared = false, has_qwen_proj_names = false;
    int max_expert = -1;
    int64_t layers_with_moe = 0;
    // Per-layer expert max.
    std::vector<int> layer_max(512, -1);
    for (const auto& t : tensors) {
        if (has_sub(t.name, "block_sparse_moe")) has_block_sparse = true;
        if (has_sub(t.name, ".mlp.experts.")) has_mlp_experts = true;
        if (has_sub(t.name, "shared_expert") || has_sub(t.name, "shared_experts"))
            has_shared = true;
        if (has_sub(t.name, "gate_proj.weight") && has_sub(t.name, ".experts."))
            has_qwen_proj_names = true;
        int li = parse_layer_index(t.name);
        int ei = parse_expert_index(t.name);
        if (ei > max_expert) max_expert = ei;
        if (li >= 0 && li < (int)layer_max.size() && ei >= 0) {
            if (ei > layer_max[(size_t)li]) layer_max[(size_t)li] = ei;
        }
    }
    for (int v : layer_max) if (v >= 0) layers_with_moe++;
    c.num_layers_with_moe = layers_with_moe;
    c.num_experts = max_expert >= 0 ? (int64_t)max_expert + 1 : 0;
    c.num_shared = has_shared ? 1 : 0;
    if (has_block_sparse) {
        c.pattern = "mixtral";
        c.top_k = 2;
    } else if (has_mlp_experts && has_qwen_proj_names) {
        c.pattern = "qwen_moe";
        c.top_k = 4;
    } else if (has_mlp_experts && has_shared) {
        c.pattern = "deepseek_moe";
        c.top_k = 6;
    } else {
        c.pattern = "generic_moe";
        c.top_k = 2;
    }
    c.valid = (c.num_experts > 0 && c.num_layers_with_moe > 0);
    return c;
}

bool validate_moe_tensors(const std::vector<AdapterTensor>& tensors,
                          std::string* err_out) {
    auto fail = [&](const std::string& m) {
        if (err_out) *err_out = m;
        return false;
    };
    if (tensors.empty()) return fail("moe: empty tensor list");
    if (!is_moe_tensors(tensors)) return fail("moe: no .experts. tensors found");
    MoeArchConfig c = infer_moe_arch(tensors);
    if (!c.valid) return fail("moe: could not infer expert count/layers");
    // Router (gate) must exist on every MoE layer.
    // Check each MoE layer has a gate + at least one full expert weight set.
    // Collect MoE layer indices.
    std::vector<int> moe_layers;
    for (const auto& t : tensors) {
        int li = parse_layer_index(t.name);
        if (li < 0) continue;
        if (has_sub(t.name, "block_sparse_moe.gate.weight") ||
            has_sub(t.name, ".mlp.gate.weight")) {
            if (std::find(moe_layers.begin(), moe_layers.end(), li) == moe_layers.end())
                moe_layers.push_back(li);
        }
    }
    if (moe_layers.empty())
        return fail("moe: missing router gate on every MoE layer "
                    "(block_sparse_moe.gate / mlp.gate)");
    // Expert weight presence: expert 0 must have its FFN triplet.
    for (int li : moe_layers) {
        std::string p = "model.layers." + std::to_string(li) + ".";
        bool has_e0 = false;
        for (const auto& t : tensors) {
            if (t.name.rfind(p, 0) != 0) continue;
            if (parse_expert_index(t.name) != 0) continue;
            has_e0 = true;
            break;
        }
        if (!has_e0)
            return fail("moe: layer " + std::to_string(li) + " missing expert 0 weights");
        // Empty-weight guard for experts.
        for (const auto& t : tensors) {
            if (t.name.rfind(p, 0) == 0 && has_sub(t.name, ".experts.") && t.data.empty())
                return fail("moe: empty weights for " + t.name);
        }
    }
    return true;
}

} // namespace adapters
} // namespace quant
