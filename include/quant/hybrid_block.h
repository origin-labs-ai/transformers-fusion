#pragma once
#include "quant/transformer.h"
#include "quant/hybrid_scheduler.h"

namespace quant {

// HybridBlock — per-layer dispatch, STD Attention only (owner purge 2026-09-07).
// Mirrors TransformerBlock's norm/FFN wiring.
class HybridBlock {
public:
    RMSNorm attention_norm;
    RMSNorm ffn_norm;
    FFN ffn;

    HybridAttnKind kind = HybridAttnKind::STD;
    // Standard full-attention core.
    Attention attn;

    HybridBlock() = default;
    HybridBlock(const TransformerConfig& cfg, HybridAttnKind k);

    Tensor forward(const Tensor& x, const Tensor& positions,
                   const Tensor& mask, KVCache& cache, int layer_idx) const;

    int64_t param_count() const;
};

class HybridModel {
public:
    explicit HybridModel(const TransformerConfig& cfg, HybridSchedule sched);

    Tensor forward(const Tensor& x, const Tensor& positions,
                   KVCache* cache = nullptr) const;

    int64_t param_count() const;
    const HybridSchedule& schedule() const { return schedule_; }

    TransformerConfig config;
private:
    HybridSchedule schedule_;
    std::vector<HybridBlock> blocks_;
    RMSNorm final_norm;
};

} // namespace quant
