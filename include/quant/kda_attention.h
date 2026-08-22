// ============================================================================
// KDA — Kimi Delta Attention (gated delta-rule linear attention)
// ============================================================================
// Linear-attention block used by Kimi-K3-class architectures (69 of 93 layers
// in K3 are KDA; the rest are gated MLA). Replaces O(S^2) softmax attention
// with an O(S) recurrent state update per head:
//
//   state  S ∈ R^{dk × dv}          (per batch, per head)
//   decay  d_h = exp(log_decay[h])  (learnable, per head)
//   write  err = v_t − Sᵀ k_t                       (delta: erase old value)
//          S   ← d_h · S + k_t ⊗ err                 (then write new value)
//   read   o_t = Sᵀ q_t                              (post-update readout)
//   norm   o_t /= max(cumulative_decay_weight, eps)  (optional scale control)
//
// This is the REFERENCE implementation: sequential recurrence, plain loops.
// Correctness first (delta-overwrite property is unit-tested); a chunked/
// SIMD path can replace the inner loop later without changing semantics.
//
// Wire interface mirrors Attention so the hybrid scheduler (G-3) can swap
// blocks freely: forward(x, positions, mask, cache, layer_idx).
#pragma once

#include "quant/transformer.h"
#include "quant/tensor.h"
#include "quant/types.h"

namespace quant {

class KDAAttention {
public:
    Linear q_proj, k_proj, v_proj, o_proj;
    Tensor log_decay;      // Shape{num_heads}, learnable log-decay base
    int64_t num_heads = 1;
    int64_t key_dim = 0;   // per-head key dimension (dk)
    int64_t value_dim = 0; // per-head value dimension (dv)
    bool normalize = true;

    KDAAttention() = default;
    explicit KDAAttention(const TransformerConfig& cfg);

    // Reference recurrence over the full sequence. x: {B,S,hidden} → {B,S,hidden}
    Tensor forward_naive(const Tensor& x) const;

    // Interface-compatible with Attention (positions/mask/cache accepted for
    // scheduler compatibility; reference path recomputes full sequence).
    Tensor forward(const Tensor& x, const Tensor& positions,
                   const Tensor& mask, KVCache& cache, int layer_idx) const {
        (void)positions; (void)mask; (void)cache; (void)layer_idx;
        return forward_naive(x);
    }

    // Exposed for unit tests: one delta-rule state update on raw vectors.
    // state: {dk × dv} row-major; k: {dk}; v: {dv}. Returns written error norm.
    static float delta_step(float* state, const float* k, const float* v,
                            float decay, int64_t dk, int64_t dv);

    int64_t param_count() const;
};

} // namespace quant
