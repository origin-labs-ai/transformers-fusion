#include "quant/hybrid_block.h"
#include "quant/math.h"
#include <cmath>

namespace quant {

HybridBlock::HybridBlock(const TransformerConfig& cfg, HybridAttnKind k)
    : attention_norm(cfg.hidden_size, cfg.norm_eps),
      ffn_norm(cfg.hidden_size, cfg.norm_eps),
      ffn(cfg),
      kind(k) {
    attn = Attention(cfg);
}

Tensor HybridBlock::forward(const Tensor& x, const Tensor& positions,
                            const Tensor& mask, KVCache& cache, int layer_idx) const {
    Tensor h_norm = attention_norm.forward(x);
    Tensor attn_out = attn.forward(h_norm, positions, mask, cache, layer_idx);
    Tensor h1(Shape{x.shape()}, DType::F32);
    for (int64_t i = 0; i < x.numel(); ++i) h1.data<float>()[i] = x.data<float>()[i] + attn_out.data<float>()[i];
    Tensor h1_norm = ffn_norm.forward(h1);
    Tensor ffn_out = ffn.forward(h1_norm);
    Tensor out(Shape{x.shape()}, DType::F32);
    for (int64_t i = 0; i < x.numel(); ++i) out.data<float>()[i] = h1.data<float>()[i] + ffn_out.data<float>()[i];
    return out;
}

int64_t HybridBlock::param_count() const {
    int64_t n = attention_norm.weight.numel() + ffn_norm.weight.numel() + ffn.gate_proj.param_count() + ffn.up_proj.param_count() + ffn.down_proj.param_count();
    n += (int64_t)attn.q_proj.param_count() + (int64_t)attn.k_proj.param_count() + (int64_t)attn.v_proj.param_count() + (int64_t)attn.o_proj.param_count();
    return n;
}

HybridModel::HybridModel(const TransformerConfig& cfg, HybridSchedule sched)
    : config(cfg), schedule_(std::move(sched)), final_norm(cfg.hidden_size, cfg.norm_eps) {
    blocks_.reserve(schedule_.total());
    for (auto k : schedule_.layers) blocks_.emplace_back(cfg, k);
}

Tensor HybridModel::forward(const Tensor& x, const Tensor& positions, KVCache* cache) const {
    Tensor h = x;
    KVCache local;
    KVCache* c = cache ? cache : &local;
    if (!cache) c->init((int)blocks_.size(), config.max_seq_len, config.num_heads, config.head_dim);
    Tensor dummy_mask;
    for (size_t i = 0; i < blocks_.size(); ++i) h = blocks_[i].forward(h, positions, dummy_mask, *c, (int)i);
    h = final_norm.forward(h);
    return h;
}

int64_t HybridModel::param_count() const {
    int64_t s = final_norm.weight.numel();
    for (auto &b : blocks_) s += b.param_count();
    return s;
}

} // namespace quant
