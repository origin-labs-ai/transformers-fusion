#pragma once
#include "quant/types.h"
#include "quant/tensor.h"
#include "quant/math.h"
#include "quant/kernel.h"
#include "quant/kv_cache.h"
#include <vector>
#include <memory>

namespace quant {

struct TransformerConfig {
    int64_t vocab_size = 32000;
    int64_t hidden_size = 768;
    int64_t num_layers = 12;
    int64_t num_heads = 12;
    int64_t head_dim = 64;
    int64_t ffn_hidden_size = 3072;
    float norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    int64_t max_seq_len = 2048;
    Activation activation = Activation::SiLU;
    int64_t num_kv_heads = 0;
    bool use_parallel_residual = false;

    // MTP (Multi-Token Prediction) — predict multiple future tokens simultaneously
    int mtp_num_heads = 0;        // 0 = disabled; >0 = number of future-token heads
    float mtp_loss_weight = 0.1f; // weight of MTP auxiliary loss relative to CE loss

    // RoPE scaling for long-context (YARN / NTK-aware interpolation)
    RoPEScalingMode rope_scaling_mode = RoPEScalingMode::None;
    float rope_scaling_factor = 1.0f;      // linear/NTK scale factor (e.g., 4.0 for 4× context)
    int64_t rope_original_max_seq_len = 0; // original context window (0 = use max_seq_len)

    // YARN-specific parameters
    float yarn_attn_factor = 1.0f;         // attention scaling factor
    float yarn_beta_fast = 32.0f;          // frequency boundary for fast-rotating dims
    float yarn_beta_slow = 1.0f;           // frequency boundary for slow-rotating dims

    // FP8 mixed precision
    bool use_fp8 = false;
    bool fp8_use_e4m3 = true;  // true = E4M3 (activations), false = E5M2 (gradients)
};

class Embedding {
public:
    Tensor weight;
    Embedding() = default;
    explicit Embedding(int64_t vocab_size, int64_t dim);
    Tensor forward(const Tensor& input_ids) const;
    size_t param_count() const;
};

class Linear {
public:
    Tensor weight;
    Tensor bias;
    Format weight_format = Format::Q32;
    Linear() = default;
    Linear(int64_t in_features, int64_t out_features);
    Tensor forward(const Tensor& input) const;
    size_t param_count() const;
};

class RMSNorm {
public:
    Tensor weight;
    float eps;
    RMSNorm() : eps(1e-5f) {}
    explicit RMSNorm(int64_t size, float eps = 1e-5f);
    Tensor forward(const Tensor& input) const;
};

class RotaryEmbedding {
public:
    Tensor cos_cached;
    Tensor sin_cached;
    int64_t head_dim;
    float theta;
    RoPEScalingMode scaling_mode = RoPEScalingMode::None;
    float scaling_factor = 1.0f;
    float mscale = 1.0f;
    float yarn_attn_factor = 1.0f;
    float attn_scale_mult() const { return mscale * yarn_attn_factor; }
    RotaryEmbedding() = default;
    RotaryEmbedding(int64_t head_dim, int64_t max_seq_len, float theta = 10000.0f);
    RotaryEmbedding(int64_t head_dim, int64_t max_seq_len, float theta,
                    RoPEScalingMode mode, float factor,
                    int64_t original_max_seq_len,
                    float yarn_beta_fast = 32.0f,
                    float yarn_beta_slow = 1.0f,
                    float yarn_attn_factor = 1.0f);
    void apply(Tensor& x, int64_t seq_start, int64_t seq_len) const;
};

class Attention {
public:
    Linear q_proj, k_proj, v_proj, o_proj;
    int64_t num_heads, num_kv_heads, head_dim;
    RotaryEmbedding rope;
    // L052: FP8 attention path (inference-only; training falls back to FP32).
    // Wired from TransformerConfig::use_fp8 / fp8_use_e4m3.
    bool use_fp8 = false;
    bool fp8_e4m3 = true;
    Attention() = default;
    explicit Attention(const TransformerConfig& cfg);
    Tensor forward(const Tensor& x, const Tensor& positions,
                   const Tensor& mask, KVCache& cache, int layer_idx) const;
};

// L052: FP8 linear GEMM (inference-only slow path).
// Quantizes A (activations) and B (weights) to E4M3/E5M2 via math::fp8_gemm
// (elementwise quantize, FP32 accumulate). Training must use Linear::forward
// (autograd graph); this path is for the use_fp8 inference switch.
// Max roundtrip error bounds (measured, see tests/test_wave6_model_feats.cpp):
//   E4M3: <= 0.07 abs on [-1, 1]; E5M2: <= 0.16 abs on [-1, 1].
Tensor linear_forward_fp8(const Linear& lin, const Tensor& input, bool use_e4m3);

// NOTE (owner purge 2026-09-07): attention-variant layer classes REMOVED.
// Standard Attention + TransformerBlock remain the only sequence mixers until
// the Cender/ formula (Phase 19-23, owner-gated) proves parity. No variant
// lineage of any name lives here.

class FFN {
public:
    Linear gate_proj, up_proj, down_proj;
    Activation activation;
    FFN() : activation(Activation::SiLU) {}
    explicit FFN(const TransformerConfig& cfg);
    FFN(int64_t hidden, int64_t ffn_hidden)
        : gate_proj(hidden, ffn_hidden), up_proj(hidden, ffn_hidden),
          down_proj(ffn_hidden, hidden), activation(Activation::SiLU) {}
    Tensor forward(const Tensor& x) const;
};

class TransformerBlock {
public:
    RMSNorm attention_norm;
    Attention attention;
    RMSNorm ffn_norm;
    FFN ffn;
    bool use_parallel_residual = false;  // GPT-NeoX style: attn+ffn run on same pre-norm input
    TransformerBlock() = default;
    explicit TransformerBlock(const TransformerConfig& cfg);
    Tensor forward(const Tensor& x, const Tensor& positions,
                   const Tensor& mask, KVCache& cache, int layer_idx) const;
};

} // namespace quant
