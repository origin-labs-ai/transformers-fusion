#pragma once
#include "quant/transformer.h"
#include "quant/kda_attention.h"
#include "quant/mla_attention.h"
#include "quant/hybrid_scheduler.h"

namespace quant {

// HybridBlock — per-layer dispatch between KDA and MLA per HybridSchedule.
// Mirrors TransformerBlock's norm/FFN wiring but swaps the attention core.
class HybridBlock {
public:
    RMSNorm attention_norm;
    RMSNorm ffn_norm;
    FFN ffn;

    HybridAttnKind kind = HybridAttnKind::KDA;
    // Only the active branch is usable; the other stays default-constructed.
    KDAAttention kda;
    MLAAttention mla;

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
