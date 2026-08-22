// ============================================================================
// MLA — Multi-head Latent Attention (gated low-rank KV compression)
// ============================================================================
// DeepSeek V4 Flash / Kimi K3 style MLA used for 24 of 93 K3 layers.
// Instead of caching full K/V per head (2·H·d·S), compress into a shared
// latent c_kv of rank r = kv_lora_rank (plus a decoupled RoPE slice of
// dimension rope_dim). The latent is cached; K/V are re-materialized on
// the fly via absorbed up-projections W_UK / W_UV. Compression ratio:
//
//   standard_bytes = 2 · num_heads · head_dim · S · sizeof(float)
//   mla_bytes      = (kv_lora_rank + rope_dim) · S · sizeof(float)
//   saving ≈ 1 − mla_bytes / standard_bytes  (target ≈ 75–93% per paper)
//
// This is the REFERENCE implementation: sequential full-recompute forward.
// Incremental latent caching is exposed but verified separately.
// Wire-compatible with Attention for the hybrid scheduler (G-3).
#pragma once

#include "quant/transformer.h"
#include "quant/tensor.h"
#include "quant/types.h"

namespace quant {

class MLAAttention {
public:
    // Compression projections
    Linear w_dq, w_uq;    // hidden → q_lora, q_lora → H·d
    Linear w_dkv, w_uk, w_uv; // hidden → latent, latent → H·d (K/V)
    Linear o_proj;
    // Decoupled RoPE slices for position-aware part
    Linear q_pe_proj, k_pe_proj;
    RMSNorm q_norm, kv_norm;

    int64_t num_heads = 1;
    int64_t head_dim = 0;
    int64_t q_lora_rank = 0;
    int64_t kv_lora_rank = 0;
    int64_t rope_dim = 0;
    int64_t hidden_size = 0;

    MLAAttention() = default;
    explicit MLAAttention(const TransformerConfig& cfg);

    // Reference full-sequence forward: x {B,S,hidden} → {B,S,hidden}
    Tensor forward_naive(const Tensor& x) const;
    Tensor forward(const Tensor& x, const Tensor& positions,
                   const Tensor& mask, KVCache& cache, int layer_idx) const {
        (void)positions; (void)mask; (void)cache; (void)layer_idx;
        return forward_naive(x);
    }

    // Bytes cached per token at this layer (standard vs latent).
    static int64_t cache_bytes_per_token_mla(int64_t kv_lora_rank,
                                             int64_t rope_dim);
    static int64_t cache_bytes_per_token_standard(int64_t num_heads,
                                                  int64_t head_dim);

    int64_t param_count() const;
};

} // namespace quant
