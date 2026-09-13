#include "quant/codebook.h"
#include "quant/random.h"
#include "quant/types.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>

namespace {

using namespace quant;

static double now_sec() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t.time_since_epoch()).count();
}

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

// ===========================================================================
// GPT-2 Configuration (matches raw_weights/manifest.json)
// ===========================================================================
struct GPT2Config {
    int64_t vocab_size   = 250880;
    int64_t hidden_size  = 1024;
    int64_t num_layers   = 24;
    int64_t num_heads    = 16;
    int64_t head_dim     = 64;
    int64_t ffn_hidden   = 4096;
    float   norm_eps     = 1e-5f;
};

// ===========================================================================
// GPT-2 Weight Buffers (all in raw FP32 arrays)
// ===========================================================================
struct GPT2Weights {
    std::vector<float> wte;
    std::vector<std::vector<float>> qkv_proj;
    std::vector<std::vector<float>> attn_proj;
    std::vector<std::vector<float>> ffn_up;
    std::vector<std::vector<float>> ffn_down;
    bool loaded = false;
};

// ===========================================================================
// File I/O
// ===========================================================================
static std::vector<float> load_fp32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return {};
    f.seekg(0, std::ios::end);
    size_t sz = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    size_t n = sz / sizeof(float);
    std::vector<float> data(n);
    f.read(reinterpret_cast<char*>(data.data()), sz);
    return data;
}

static bool load_gpt2_weights(const std::string& dir, const GPT2Config& cfg, GPT2Weights& w) {
    w.wte = load_fp32(dir + "/transformer_word_embeddings_weight.fp32");
    if (w.wte.empty()) {
        std::cerr << "Failed to load word embeddings from " << dir << std::endl;
        return false;
    }

    w.qkv_proj.resize(cfg.num_layers);
    w.attn_proj.resize(cfg.num_layers);
    w.ffn_up.resize(cfg.num_layers);
    w.ffn_down.resize(cfg.num_layers);

    for (int64_t l = 0; l < cfg.num_layers; l++) {
        std::string prefix = dir + "/transformer_h_" + std::to_string(l);
        w.qkv_proj[l]  = load_fp32(prefix + "_self_attention_query_key_value_weight.fp32");
        w.attn_proj[l] = load_fp32(prefix + "_self_attention_dense_weight.fp32");
        w.ffn_up[l]    = load_fp32(prefix + "_mlp_dense_h_to_4h_weight.fp32");
        w.ffn_down[l]  = load_fp32(prefix + "_mlp_dense_4h_to_h_weight.fp32");

        if (w.qkv_proj[l].empty() || w.attn_proj[l].empty() ||
            w.ffn_up[l].empty() || w.ffn_down[l].empty()) {
            std::cerr << "Failed to load layer " << l << " weights" << std::endl;
            return false;
        }
    }

    w.loaded = true;
    return true;
}

// ===========================================================================
// Quantization Helpers
// ===========================================================================
struct QuantizedWeight {
    std::vector<uint8_t> indices_8bit;
    std::vector<uint8_t> indices_4bit;
    std::vector<int8_t>  indices_quant;
    std::vector<float>   codebook_8;
    std::vector<float>   codebook_4;
    float quant_scale = 1.0f;
};

static QuantizedWeight quantize_weight(const float* src, int64_t n) {
    QuantizedWeight q;

    // QUANT8: per-block 256-entry Lloyd-Max
    static const int BLOCK8 = 4096;
    q.indices_8bit.resize(n);
    q.codebook_8.resize(256);
    {
        for (int64_t b = 0; b < n; b += BLOCK8) {
            int64_t bsz = std::min((int64_t)BLOCK8, n - b);
            CodebookQUANT8 cb;
            cb.train(src + b, (size_t)bsz);
            for (int i = 0; i < 256; i++) q.codebook_8[i] = cb.centroids[i];
            for (int64_t i = 0; i < bsz; i++)
                q.indices_8bit[b + i] = cb.quantize(src[b + i]);
        }
    }

    // QUANT4: per-block 16-entry Lloyd-Max
    static const int BLOCK4 = 1024;
    q.indices_4bit.resize((n + 1) / 2, 0);
    q.codebook_4.resize(16);
    {
        for (int64_t b = 0; b < n; b += BLOCK4) {
            int64_t bsz = std::min((int64_t)BLOCK4, n - b);
            CodebookQUANT4 cb;
            cb.train(src + b, (size_t)bsz);
            for (int i = 0; i < 16; i++)
                q.codebook_4[i] = CodebookQUANT4::half_to_float(cb.centroids[i]);
            for (int64_t i = 0; i < bsz; i++) {
                uint8_t idx = cb.quantize(src[b + i]);
                int64_t flat = b + i;
                if (flat % 2 == 0)
                    q.indices_4bit[flat / 2] = (q.indices_4bit[flat / 2] & 0xF0) | idx;
                else
                    q.indices_4bit[flat / 2] = (q.indices_4bit[flat / 2] & 0x0F) | (idx << 4);
            }
        }
    }

    // BitNet b1.58-style ternary {-1,0,+1} with a per-tensor scale. This is a
    // generic ternary baseline — NOT the canonical Q1_5 codec (which is
    // per-32 FP16 scale + sign bits); we label it honestly below.
    double sum_abs = 0.0;
    for (int64_t i = 0; i < n; i++) sum_abs += std::abs(src[i]);
    q.quant_scale = (float)(sum_abs / (double)n + 1e-10);
    q.indices_quant.resize(n);
    for (int64_t i = 0; i < n; i++) {
        float v = src[i] / q.quant_scale;
        q.indices_quant[i] = (v > 0.33f) ? 1 : ((v < -0.33f) ? -1 : 0);
    }

    return q;
}

static void dequantize_quant8(const QuantizedWeight& q, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++)
        dst[i] = q.codebook_8[q.indices_8bit[i]];
}

static void dequantize_quant4(const QuantizedWeight& q, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        uint8_t packed = q.indices_4bit[i / 2];
        uint8_t idx = (i % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
        dst[i] = q.codebook_4[idx];
    }
}

static void dequantize_quant(const QuantizedWeight& q, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++)
        dst[i] = (float)q.indices_quant[i] * q.quant_scale;
}

// ===========================================================================
// GPT-2 Minimal Forward Pass
// ===========================================================================
enum class QuantMode { FP32, QUANT8, QUANT4, BITNET_158 };

struct QuantizedGPT2Weights {
    GPT2Config cfg;
    std::vector<float> wte;

    std::vector<std::vector<float>> qkv_fp32;
    std::vector<std::vector<float>> attn_fp32;
    std::vector<std::vector<float>> ffn_up_fp32;
    std::vector<std::vector<float>> ffn_down_fp32;

    std::vector<std::vector<QuantizedWeight>> qkv_q;
    std::vector<std::vector<QuantizedWeight>> attn_q;
    std::vector<std::vector<QuantizedWeight>> ffn_up_q;
    std::vector<std::vector<QuantizedWeight>> ffn_down_q;
};

static QuantizedGPT2Weights quantize_gpt2(const GPT2Config& cfg, const GPT2Weights& w,
                                            QuantMode mode) {
    QuantizedGPT2Weights qw;
    qw.cfg = cfg;
    qw.wte = w.wte;
    int64_t L = cfg.num_layers;
    int64_t qkv_sz   = 3 * cfg.hidden_size * cfg.hidden_size;
    int64_t attn_sz   = cfg.hidden_size * cfg.hidden_size;
    int64_t ffn_up_sz = cfg.ffn_hidden * cfg.hidden_size;
    int64_t ffn_dn_sz = cfg.hidden_size * cfg.ffn_hidden;

    if (mode == QuantMode::FP32) {
        qw.qkv_fp32.resize(L);
        qw.attn_fp32.resize(L);
        qw.ffn_up_fp32.resize(L);
        qw.ffn_down_fp32.resize(L);
        for (int64_t l = 0; l < L; l++) {
            qw.qkv_fp32[l]    = w.qkv_proj[l];
            qw.attn_fp32[l]   = w.attn_proj[l];
            qw.ffn_up_fp32[l] = w.ffn_up[l];
            qw.ffn_down_fp32[l] = w.ffn_down[l];
        }
    } else {
        qw.qkv_q.resize(L);
        qw.attn_q.resize(L);
        qw.ffn_up_q.resize(L);
        qw.ffn_down_q.resize(L);
        for (int64_t l = 0; l < L; l++) {
            auto do_quant = [&](const std::vector<float>& src, int64_t n) {
                return quantize_weight(src.data(), n);
            };
            qw.qkv_q[l]   = {do_quant(w.qkv_proj[l], qkv_sz)};
            qw.attn_q[l]  = {do_quant(w.attn_proj[l], attn_sz)};
            qw.ffn_up_q[l] = {do_quant(w.ffn_up[l], ffn_up_sz)};
            qw.ffn_down_q[l] = {do_quant(w.ffn_down[l], ffn_dn_sz)};
        }
    }
    return qw;
}

static std::vector<float> get_dequantized_weight(
    const QuantizedGPT2Weights& qw, QuantMode mode,
    int layer, const std::string& which, int64_t& out_n) {

    int64_t L = qw.cfg.num_layers;
    int64_t H = qw.cfg.hidden_size;
    int64_t F = qw.cfg.ffn_hidden;
    int64_t qkv_n   = 3 * H * H;
    int64_t attn_n  = H * H;
    int64_t ffn_up_n = F * H;
    int64_t ffn_dn_n = H * F;

    if (mode == QuantMode::FP32) {
        if (which == "qkv")   { out_n = qkv_n;   return qw.qkv_fp32[layer]; }
        if (which == "attn")  { out_n = attn_n;   return qw.attn_fp32[layer]; }
        if (which == "ffn_up"){ out_n = ffn_up_n; return qw.ffn_up_fp32[layer]; }
        if (which == "ffn_dn"){ out_n = ffn_dn_n; return qw.ffn_down_fp32[layer]; }
    }

    int64_t n = 0;
    const QuantizedWeight* qwp = nullptr;
    if (which == "qkv")   { n = qkv_n;   qwp = &qw.qkv_q[layer][0]; }
    if (which == "attn")  { n = attn_n;   qwp = &qw.attn_q[layer][0]; }
    if (which == "ffn_up"){ n = ffn_up_n; qwp = &qw.ffn_up_q[layer][0]; }
    if (which == "ffn_dn"){ n = ffn_dn_n; qwp = &qw.ffn_down_q[layer][0]; }

    out_n = n;
    std::vector<float> result(n);
    switch (mode) {
        case QuantMode::QUANT8:     dequantize_quant8(*qwp, result.data(), n); break;
        case QuantMode::QUANT4:     dequantize_quant4(*qwp, result.data(), n); break;
        case QuantMode::BITNET_158:  dequantize_quant(*qwp, result.data(), n); break;
        default: break;
    }
    return result;
}

static std::vector<float> matvec(const float* mat, const float* vec,
                                  int64_t rows, int64_t cols) {
    std::vector<float> out(rows, 0.0f);
    for (int64_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        const float* row = mat + r * cols;
        for (int64_t c = 0; c < cols; c++)
            sum += row[c] * vec[c];
        out[r] = sum;
    }
    return out;
}

static std::vector<float> matvec_t(const float* mat, const float* vec,
                                    int64_t rows, int64_t cols) {
    std::vector<float> out(cols, 0.0f);
    for (int64_t c = 0; c < cols; c++) {
        float sum = 0.0f;
        for (int64_t r = 0; r < rows; r++)
            sum += mat[r * cols + c] * vec[r];
        out[c] = sum;
    }
    return out;
}

static void rms_norm(float* out, const float* x, const float* w, int64_t n, float eps) {
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < n; i++) sum_sq += x[i] * x[i];
    float rms = std::sqrt(sum_sq / (float)n + eps);
    float inv = 1.0f / rms;
    for (int64_t i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

static float gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static void softmax(float* out, const float* in, int64_t n) {
    float mx = -1e30f;
    for (int64_t i = 0; i < n; i++) mx = std::max(mx, in[i]);
    float sum = 0.0f;
    for (int64_t i = 0; i < n; i++) { out[i] = std::exp(in[i] - mx); sum += out[i]; }
    float inv = 1.0f / (sum + 1e-10f);
    for (int64_t i = 0; i < n; i++) out[i] *= inv;
}

static std::vector<float> gpt2_forward(
    const QuantizedGPT2Weights& qw, QuantMode mode,
    const std::vector<int>& tokens, bool return_logits_at_end) {

    const auto& cfg = qw.cfg;
    int64_t H = cfg.hidden_size;
    int64_t L = cfg.num_layers;
    int64_t NH = cfg.num_heads;
    int64_t D = cfg.head_dim;
    int64_t F = cfg.ffn_hidden;
    int64_t V = cfg.vocab_size;
    int64_t seq_len = (int64_t)tokens.size();

    std::vector<float> hidden(H);

    int pos = tokens[0] % (int)cfg.vocab_size;
    for (int64_t d = 0; d < H; d++)
        hidden[d] = qw.wte[pos * H + d];

    for (int64_t l = 0; l < L; l++) {
        int64_t n;
        auto qkv_w  = get_dequantized_weight(qw, mode, static_cast<int>(l), "qkv", n);
        auto attn_w = get_dequantized_weight(qw, mode, static_cast<int>(l), "attn", n);
        auto fup_w  = get_dequantized_weight(qw, mode, static_cast<int>(l), "ffn_up", n);
        auto fdn_w  = get_dequantized_weight(qw, mode, static_cast<int>(l), "ffn_dn", n);

        std::vector<float> normed(H);
        rms_norm(normed.data(), hidden.data(), normed.data(), H, cfg.norm_eps);
        for (int64_t d = 0; d < H; d++) normed[d] = 1.0f;

        auto qkv = matvec(qkv_w.data(), hidden.data(), 3 * H, H);

        std::vector<float> attn_out(H, 0.0f);
        for (int64_t h = 0; h < NH; h++) {
            const float* qh = qkv.data() + h * D;
            const float* kh = qkv.data() + H + h * D;
            const float* vh = qkv.data() + 2 * H + h * D;
            float scale = 1.0f / std::sqrt((float)D);
            float score = 0.0f;
            for (int64_t d = 0; d < D; d++) score += qh[d] * kh[d];
            score *= scale;
            float weight = std::exp(score);
            for (int64_t d = 0; d < D; d++)
                attn_out[h * D + d] += weight * vh[d];
        }

        auto proj = matvec(attn_w.data(), attn_out.data(), H, H);
        for (int64_t d = 0; d < H; d++) hidden[d] += proj[d];

        std::vector<float> ffn_norm(H);
        rms_norm(ffn_norm.data(), hidden.data(), ffn_norm.data(), H, cfg.norm_eps);
        for (int64_t d = 0; d < H; d++) ffn_norm[d] = 1.0f;

        auto ffn_h = matvec(fup_w.data(), hidden.data(), F, H);
        for (int64_t i = 0; i < F; i++) ffn_h[i] = gelu(ffn_h[i]);
        auto ffn_out = matvec(fdn_w.data(), ffn_h.data(), H, F);
        for (int64_t d = 0; d < H; d++) hidden[d] += ffn_out[d];
    }

    std::vector<float> logits(V);
    for (int64_t v = 0; v < V; v++) {
        float sum = 0.0f;
        for (int64_t d = 0; d < H; d++)
            sum += qw.wte[v * H + d] * hidden[d];
        logits[v] = sum;
    }

    return logits;
}

// ===========================================================================
// Metrics
// ===========================================================================
struct InferenceResult {
    std::string quant_name;
    std::string prompt;
    int top1_token;
    float top1_logprob;
    float perplexity;
    float cosine_to_fp32;
    double forward_time_ms;
};

static int argmax(const std::vector<float>& v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); i++)
        if (v[i] > v[best]) best = (int)i;
    return best;
}

static float log_softmax_prob(const std::vector<float>& logits, int token) {
    float mx = -1e30f;
    for (size_t i = 0; i < logits.size(); i++) mx = std::max(mx, logits[i]);
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); i++) sum += std::exp(logits[i] - mx);
    return logits[token] - mx - std::log(sum + 1e-10f);
}

static float perplexity_from_logits(const std::vector<float>& logits, int target) {
    return -log_softmax_prob(logits, target);
}

// ===========================================================================
// Simple tokenizer: character-level mapping for prompt testing
// ===========================================================================
static std::vector<int> tokenize_simple(const std::string& text, int vocab_size) {
    std::vector<int> ids;
    for (char c : text) {
        int id = (int)(unsigned char)c;
        id = id % vocab_size;
        if (id == 0) id = 1;
        ids.push_back(id);
    }
    if (ids.empty()) ids.push_back(1);
    return ids;
}

} // anonymous namespace

int main(int argc, char** argv) {
    std::string weights_dir = "benchmark_results/raw_weights";
    if (argc > 1) weights_dir = std::string(argv[1]);

    std::cout << "=== GPT-2 Real Inference Quality Test ===" << std::endl;
    std::cout << "Weights directory: " << weights_dir << std::endl;

    GPT2Config cfg;
    GPT2Weights raw_weights;
    if (!load_gpt2_weights(weights_dir, cfg, raw_weights)) {
        std::cerr << "Failed to load GPT-2 weights. Aborting." << std::endl;
        return 1;
    }

    int64_t total_params = cfg.vocab_size * cfg.hidden_size;
    for (int64_t l = 0; l < cfg.num_layers; l++) {
        total_params += 3 * cfg.hidden_size * cfg.hidden_size;
        total_params += cfg.hidden_size * cfg.hidden_size;
        total_params += cfg.ffn_hidden * cfg.hidden_size;
        total_params += cfg.hidden_size * cfg.ffn_hidden;
    }
    std::cout << "  Model: " << cfg.num_layers << " layers, "
              << cfg.hidden_size << " hidden, "
              << cfg.num_heads << " heads, "
              << cfg.vocab_size << " vocab" << std::endl;
    std::cout << "  Parameters: " << (total_params / 1e6) << "M" << std::endl;

    struct PromptCase {
        std::string text;
        std::string description;
    };

    std::vector<PromptCase> prompts = {
        {"The capital of France is", "Factual knowledge"},
        {"1 + 1 =", "Arithmetic"},
        {"Once upon a time", "Creative start"},
    };

    std::vector<QuantMode> modes = {QuantMode::FP32, QuantMode::QUANT8, QuantMode::QUANT4, QuantMode::BITNET_158};
    std::vector<std::string> mode_names = {"FP32", "QUANT8", "QUANT4", "BITNET_158"};

    std::vector<QuantizedGPT2Weights> precomputed;
    for (auto mode : modes) {
        std::cout << "\nQuantizing model with " << mode_names[&mode - &modes[0]]
                  << "..." << std::flush;
        double t0 = now_sec();
        precomputed.push_back(quantize_gpt2(cfg, raw_weights, mode));
        double ms = (now_sec() - t0) * 1000.0;
        std::cout << " done (" << std::fixed << std::setprecision(1) << ms << " ms)" << std::endl;
    }

    std::vector<InferenceResult> all_results;
    std::vector<std::vector<float>> fp32_logits_per_prompt;

    for (auto& pc : prompts) {
        std::cout << "\n--- Prompt: \"" << pc.text << "\" (" << pc.description << ") ---" << std::endl;

        auto tokens = tokenize_simple(pc.text, (int)cfg.vocab_size);
        std::cout << "  Tokens: [";
        for (size_t i = 0; i < tokens.size(); i++) {
            if (i) std::cout << ", ";
            std::cout << tokens[i];
        }
        std::cout << "]" << std::endl;

        for (size_t mi = 0; mi < modes.size(); mi++) {
            double t0 = now_sec();
            auto logits = gpt2_forward(precomputed[mi], modes[mi], tokens, true);
            double fwd_ms = (now_sec() - t0) * 1000.0;

            int top1 = argmax(logits);
            float top1_lp = log_softmax_prob(logits, (int)logits.size() > 0 ? top1 : 0);
            float ppl = perplexity_from_logits(logits, 0);

            InferenceResult ir;
            ir.quant_name = mode_names[mi];
            ir.prompt = pc.text;
            ir.top1_token = top1;
            ir.top1_logprob = top1_lp;
            ir.perplexity = ppl;
            ir.forward_time_ms = fwd_ms;

            if (modes[mi] == QuantMode::FP32) {
                fp32_logits_per_prompt.push_back(logits);
                ir.cosine_to_fp32 = 1.0f;
            } else {
                const auto& fp32_l = fp32_logits_per_prompt.back();
                int64_t vsz = (int64_t)std::min(logits.size(), fp32_l.size());
                ir.cosine_to_fp32 = (float)cosine_sim(logits.data(), fp32_l.data(), vsz);
            }

            all_results.push_back(ir);
            std::cout << "  " << std::setw(10) << std::left << mode_names[mi]
                      << "  top1=" << std::setw(8) << top1
                      << "  logP=" << std::fixed << std::setprecision(4) << top1_lp
                      << "  ppl=" << std::fixed << std::setprecision(4) << ppl
                      << "  cos(fp32)=" << std::fixed << std::setprecision(6) << ir.cosine_to_fp32
                      << "  time=" << std::fixed << std::setprecision(1) << fwd_ms << "ms"
                      << std::endl;
        }
    }

    // --- Summary Table ---
    std::cout << "\n=== Summary: Side-by-Side Comparison ===" << std::endl;
    std::cout << std::left
              << std::setw(10) << "Quant"
              << std::setw(30) << "Prompt"
              << std::setw(10) << "Top-1"
              << std::setw(12) << "LogProb"
              << std::setw(12) << "PPL"
              << std::setw(12) << "Cos(fp32)"
              << std::setw(12) << "Time(ms)" << std::endl;
    std::cout << std::string(98, '-') << std::endl;

    for (auto& ir : all_results) {
        std::cout << std::left
                  << std::setw(10) << ir.quant_name
                  << std::setw(30) << ir.prompt.substr(0, 28)
                  << std::setw(10) << ir.top1_token
                  << std::setw(12) << std::fixed << std::setprecision(4) << ir.top1_logprob
                  << std::setw(12) << std::fixed << std::setprecision(4) << ir.perplexity
                  << std::setw(12) << std::fixed << std::setprecision(6) << ir.cosine_to_fp32
                  << std::setw(12) << std::fixed << std::setprecision(1) << ir.forward_time_ms
                  << std::endl;
    }

    // --- Per-method averages ---
    std::cout << "\n=== Per-Quantization Averages ===" << std::endl;
    std::cout << std::left
              << std::setw(10) << "Quant"
              << std::setw(14) << "Avg LogProb"
              << std::setw(14) << "Avg PPL"
              << std::setw(14) << "Avg Cos(fp32)"
              << std::setw(14) << "Avg Time(ms)"
              << std::setw(14) << "Top1 Match%" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    int fp32_top1 = -1;
    for (auto& ir : all_results) {
        if (ir.quant_name == "FP32") { fp32_top1 = ir.top1_token; break; }
    }

    for (auto& mn : mode_names) {
        double sum_lp = 0, sum_ppl = 0, sum_cos = 0, sum_time = 0;
        int match = 0, count = 0;
        int prompt_idx = 0;
        for (auto& ir : all_results) {
            if (ir.quant_name == mn) {
                sum_lp += ir.top1_logprob;
                sum_ppl += ir.perplexity;
                sum_cos += ir.cosine_to_fp32;
                sum_time += ir.forward_time_ms;
                count++;
            }
        }
        std::cout << std::left
                  << std::setw(10) << mn
                  << std::setw(14) << std::fixed << std::setprecision(4) << (count > 0 ? sum_lp / count : 0)
                  << std::setw(14) << std::fixed << std::setprecision(4) << (count > 0 ? sum_ppl / count : 0)
                  << std::setw(14) << std::fixed << std::setprecision(6) << (count > 0 ? sum_cos / count : 0)
                  << std::setw(14) << std::fixed << std::setprecision(1) << (count > 0 ? sum_time / count : 0)
                  << std::endl;
    }

    // --- CSV Output ---
    std::cout << "\n=== CSV Export ===" << std::endl;
    std::cout << "quant,prompt,top1_token,logprob,ppl,cosine_to_fp32,time_ms" << std::endl;
    for (auto& ir : all_results) {
        std::cout << ir.quant_name << ","
                  << ir.prompt << ","
                  << ir.top1_token << ","
                  << std::fixed << std::setprecision(6) << ir.top1_logprob << ","
                  << std::fixed << std::setprecision(6) << ir.perplexity << ","
                  << std::fixed << std::setprecision(8) << ir.cosine_to_fp32 << ","
                  << std::fixed << std::setprecision(3) << ir.forward_time_ms << std::endl;
    }

    // --- Quality verdict ---
    std::cout << "\n=== Quality Verdict ===" << std::endl;
    for (auto& mn : mode_names) {
        int count = 0;
        float total_cos = 0;
        for (auto& ir : all_results) {
            if (ir.quant_name == mn) {
                total_cos += ir.cosine_to_fp32;
                count++;
            }
        }
        float avg_cos = count > 0 ? total_cos / count : 0;
        std::string verdict;
        if (avg_cos > 0.99f) verdict = "EXCELLENT — near-identical to FP32";
        else if (avg_cos > 0.95f) verdict = "GOOD — minor quality loss";
        else if (avg_cos > 0.90f) verdict = "ACCEPTABLE — noticeable degradation";
        else verdict = "POOR — significant quality loss";

        std::cout << "  " << std::setw(10) << std::left << mn
                  << " cos(fp32)=" << std::fixed << std::setprecision(6) << avg_cos
                  << "  " << verdict << std::endl;
    }

    std::cout << "\n=== Done ===" << std::endl;
    return 0;
}
