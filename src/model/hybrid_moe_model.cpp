#include "quant/hybrid_moe_model.h"

namespace quant {

HybridMoeBlock::HybridMoeBlock(const TransformerConfig& cfg,
                               const moe::MoEAllConfig& mcfg,
                               HybridAttnKind k)
    : attention_norm(cfg.hidden_size, cfg.norm_eps),
      ffn_norm(cfg.hidden_size, cfg.norm_eps),
      kind(k),
      moe_cfg(mcfg) {
    attn = Attention(cfg);
    moe = std::make_unique<moe::SparseMoE>(cfg.hidden_size, mcfg);
    if (mcfg.use_shared_expert) shared_expert = std::make_unique<moe::ExpertFFN>(cfg.hidden_size, mcfg.expert_hidden_size);
}

Tensor HybridMoeBlock::forward(const Tensor& x, const Tensor& positions,
                               const Tensor& mask, KVCache& cache, int layer_idx) const {
    Tensor h_norm = attention_norm.forward(x);
    Tensor attn_out = attn.forward(h_norm, positions, mask, cache, layer_idx);
    Tensor h1(x.shape(), DType::F32);
    for (int64_t i = 0; i < x.numel(); ++i) h1.data<float>()[i] = x.data<float>()[i] + attn_out.data<float>()[i];
    Tensor h1n = ffn_norm.forward(h1);
    auto moe_out = moe->forward(h1n);
    Tensor moe_res = moe_out.output;
    if (shared_expert) {
        Tensor shared = shared_expert->forward(h1n);
        Tensor combined(moe_res.shape(), DType::F32);
        for (int64_t i = 0; i < moe_res.numel(); ++i) combined.data<float>()[i] = moe_res.data<float>()[i] + shared.data<float>()[i];
        moe_res = combined;
    }
    Tensor out(x.shape(), DType::F32);
    for (int64_t i = 0; i < x.numel(); ++i) out.data<float>()[i] = h1.data<float>()[i] + moe_res.data<float>()[i];
    return out;
}

int64_t HybridMoeBlock::param_count() const {
    int64_t n = attention_norm.weight.numel() + ffn_norm.weight.numel();
    n += (int64_t)attn.q_proj.param_count() + (int64_t)attn.k_proj.param_count() + (int64_t)attn.v_proj.param_count() + (int64_t)attn.o_proj.param_count();
    if (moe) {
        for (auto &e : moe->experts) n += e.gate_proj.param_count() + e.up_proj.param_count() + e.down_proj.param_count();
        n += moe->router_weight.param_count();
    }
    if (shared_expert) n += shared_expert->gate_proj.param_count() + shared_expert->up_proj.param_count() + shared_expert->down_proj.param_count();
    return n;
}

HybridMoeModel::HybridMoeModel(const TransformerConfig& cfg, const moe::MoEAllConfig& mcfg,
                               HybridSchedule sched)
    : schedule_(std::move(sched)), moe_config(mcfg) {
    this->config = cfg;
    tok_embeddings = std::make_unique<Embedding>(cfg.vocab_size, cfg.hidden_size);
    for (auto k : schedule_.layers) blocks.emplace_back(cfg, mcfg, k);
    norm = std::make_unique<RMSNorm>(cfg.hidden_size, cfg.norm_eps);
    lm_head = std::make_unique<Linear>(cfg.hidden_size, cfg.vocab_size);
}

Tensor HybridMoeModel::forward(const Tensor& input_ids, const Tensor& positions, KVCache* cache) {
    int64_t B = input_ids.dim(0), S = input_ids.dim(1);
    Tensor h = tok_embeddings->forward(input_ids.reshape(Shape{B * S}));
    h = h.reshape(Shape{B, S, this->config.hidden_size});
    KVCache local;
    KVCache* c = cache ? cache : &local;
    if (!cache) c->init((int)this->blocks.size(), this->config.max_seq_len, this->config.num_heads, this->config.head_dim);
    Tensor pos = positions.numel() ? positions : Tensor();
    Tensor mask;
    for (size_t i = 0; i < this->blocks.size(); ++i) h = this->blocks[i].forward(h, pos, mask, *c, (int)i);
    h = norm->forward(h);
    return lm_head->forward(h.reshape(Shape{B * S, this->config.hidden_size})).reshape(Shape{B, S, this->config.vocab_size});
}

int64_t HybridMoeModel::param_count() const {
    int64_t s = tok_embeddings->weight.numel() + norm->weight.numel() + lm_head->param_count();
    for (auto &b : this->blocks) s += b.param_count();
    return s;
}

} // namespace quant
