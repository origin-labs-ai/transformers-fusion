#include "quant/native_quant_moe.h"
#include "quant/transformer.h"
#include "quant/kv_cache.h"
#include "quant/autograd.h"
#include "quant/math.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <memory>
#include <cstdio>

namespace quant {
namespace native {

using moe::MoEAllConfig;
using moe::MoEVariant;
using moe::SparseMoE;
using moe::MoEOutput;

NativeQUANTMoEModel::NativeQUANTMoEModel(const NativeMoEConfig& cfg)
    : cfg_(cfg), total_params_(0), emb_params_(0), head_params_(0)
{
#ifdef QUANT_DEBUG
#endif
    emb_ = new Embedding(cfg_.vocab_size, cfg_.hidden_size);
#ifdef QUANT_DEBUG
#endif
    emb_params_ = (size_t)cfg_.vocab_size * (size_t)cfg_.hidden_size;

    layers_.resize(cfg_.num_layers);
#ifdef QUANT_DEBUG
#endif
    TransformerConfig tcfg;
    tcfg.hidden_size = cfg_.hidden_size;
    tcfg.num_heads = cfg_.num_heads;
    tcfg.head_dim = cfg_.hidden_size / cfg_.num_heads;
    tcfg.ffn_hidden_size = cfg_.ffn_hidden;
    tcfg.vocab_size = cfg_.vocab_size;
    tcfg.max_seq_len = cfg_.max_seq_len;
    for (int64_t i = 0; i < cfg_.num_layers; i++) {
#ifdef QUANT_DEBUG
#endif
        layers_[i] = std::make_unique<TransformerBlock>(tcfg);
    }

#ifdef QUANT_DEBUG
#endif
    MoEAllConfig mc;
    mc.variant = MoEVariant::SPARSE_TOPK;
    mc.num_experts = cfg_.num_experts;
    mc.top_k = cfg_.top_k;
    mc.expert_hidden_size = cfg_.ffn_hidden;
    mc.load_balance_coef = cfg_.load_balance_coef;
    mc.z_loss_coef = cfg_.z_loss_coef;
#ifdef QUANT_DEBUG
#endif
    moe_ = new SparseMoE(cfg_.hidden_size, mc);
#ifdef QUANT_DEBUG
#endif

    norm_ = new RMSNorm(cfg_.hidden_size);
    lm_head_ = new Linear(cfg_.hidden_size, cfg_.vocab_size);
    head_params_ = (size_t)cfg_.hidden_size * (size_t)cfg_.vocab_size;

#ifdef QUANT_DEBUG
#endif
    total_params_ = emb_params_ + head_params_;
    for (auto& l : layers_) {
        total_params_ += (size_t)cfg_.hidden_size;
        total_params_ += 4 * (size_t)cfg_.hidden_size * (cfg_.hidden_size / cfg_.num_heads) * cfg_.num_heads;
        total_params_ += 4 * (cfg_.hidden_size / cfg_.num_heads) * cfg_.num_heads;
        total_params_ += (size_t)cfg_.hidden_size;
    }
    total_params_ += (size_t)cfg_.num_experts * 3 * (size_t)cfg_.hidden_size * (size_t)cfg_.ffn_hidden;
    total_params_ += (size_t)cfg_.hidden_size * (size_t)cfg_.num_experts;
//DBG
//DBG#ifdef QUANT_DEBUG
//DBG           (unsigned long long)(total_params_ + 4096),
//DBG           (unsigned long long)((total_params_ + 4096) * sizeof(float)));
//DBG    fflush(stdout);
//DBG#endif
    temp_deq_ = std::make_unique<float[]>(total_params_ + 4096);
#ifdef QUANT_DEBUG
#endif

    printf("[NativeQUANTMoE] Config: hidden=%lld layers=%lld experts=%lld top_k=%lld\n",
           (long long)cfg_.hidden_size, (long long)cfg_.num_layers,
           (long long)cfg_.num_experts, (long long)cfg_.top_k);
    printf("[NativeQUANTMoE] Total params: %lld (%.2fM)\n",
           (long long)total_params_, total_params_ / 1e6);
#ifdef QUANT_DEBUG
#endif
}

NativeQUANTMoEModel::~NativeQUANTMoEModel() {
    delete emb_;
    delete moe_;
    delete norm_;
    delete lm_head_;
}

void NativeQUANTMoEModel::collect_all_params(std::vector<Tensor*>& params) {
    params.push_back(&emb_->weight);
    for (auto& l : layers_) {
        params.push_back(&l->attention_norm.weight);
        params.push_back(&l->attention.q_proj.weight);
        if (l->attention.q_proj.bias.numel() > 0)
            params.push_back(&l->attention.q_proj.bias);
        params.push_back(&l->attention.k_proj.weight);
        if (l->attention.k_proj.bias.numel() > 0)
            params.push_back(&l->attention.k_proj.bias);
        params.push_back(&l->attention.v_proj.weight);
        if (l->attention.v_proj.bias.numel() > 0)
            params.push_back(&l->attention.v_proj.bias);
        params.push_back(&l->attention.o_proj.weight);
        if (l->attention.o_proj.bias.numel() > 0)
            params.push_back(&l->attention.o_proj.bias);
        params.push_back(&l->ffn_norm.weight);
    }
    params.push_back(&norm_->weight);
    params.push_back(&lm_head_->weight);
    if (lm_head_->bias.numel() > 0)
        params.push_back(&lm_head_->bias);
}

Tensor NativeQUANTMoEModel::forward(const Tensor& input_ids) {
#ifdef QUANT_DEBUG
    printf("[DBG:FWD] begin B=%lld S=%lld\n", (long long)input_ids.dim(0), (long long)input_ids.dim(1)); fflush(stdout);
#endif
    int64_t B = input_ids.dim(0), S = input_ids.dim(1);
    Tensor h = emb_->forward(input_ids);
    h = h.reshape(Shape{B, S, cfg_.hidden_size});
#ifdef QUANT_DEBUG
    printf("[DBG:FWD] embedding done, h.rank=%d, h.numel=%lld\n", h.rank(), (long long)h.numel()); fflush(stdout);
#endif
    Tensor positions(Shape{B, S});
    for (int64_t i = 0; i < B * S; i++) positions.data<float>()[i] = (float)(i % S);
    int64_t head_dim = cfg_.hidden_size / cfg_.num_heads;
    Tensor mask;
    KVCache cache(static_cast<int>(cfg_.num_layers), cfg_.max_seq_len, cfg_.num_heads, head_dim);
    for (auto& l : layers_) {
        Tensor n = l->attention_norm.forward(h);
        Tensor a = l->attention.forward(n, positions, mask, cache, 0);
        a = AutogradEngine::add_op(a, h);
        Tensor fn = l->ffn_norm.forward(a);
        Tensor f = l->ffn.forward(fn);
        h = AutogradEngine::add_op(f, a);
    }
    moe::MoEOutput moe_out = moe_->forward(h);
    h = moe_out.output;
    h = norm_->forward(h);
    Tensor logits = lm_head_->forward(h);
    return logits;
}

void NativeQUANTMoEModel::backward(const Tensor& loss) {
    Tensor& loss_mut = const_cast<Tensor&>(loss);
    auto& engine = AutogradEngine::instance();
    engine.backward(loss_mut);
}

void NativeQUANTMoEModel::apply_update(float lr_scale, float lr_weight) {
    std::vector<Tensor*> params;
    collect_all_params(params);
    for (Tensor* t : params) {
        if (!t || t->numel() == 0) continue;
        if (!t->has_grad()) continue;
        float* pd = t->data<float>();
        const float* gd = t->grad().data<float>();
        int64_t n = t->numel();
        float lr = lr_scale * lr_weight;
        for (int64_t i = 0; i < n; i++) {
            pd[i] -= lr * gd[i];
        }
    }
}

void NativeQUANTMoEModel::init_from_fp32() {
    std::vector<Tensor*> params;
    collect_all_params(params);
    for (Tensor* t : params) {
        if (!t || t->numel() == 0) continue;
        float* d = t->data<float>();
        int64_t n = t->numel();
        for (int64_t i = 0; i < n; i++) {
            d[i] = d[i];
        }
    }
    printf("[NativeQUANTMoE] init_from_fp32: %zu params initialized\n", params.size());
}

NativeMoEMetrics NativeQUANTMoEModel::step(const float* input_ids, const float* targets,
                                           size_t B, size_t S, float lr_s, float lr_w) {
    (void)lr_s;
    (void)lr_w;
    NativeMoEMetrics m;
    m.total_params = total_params_;
    m.active_params = (size_t)(cfg_.top_k * cfg_.ffn_hidden * 3 * 2);

    Tensor inp(Shape{(int64_t)B, (int64_t)S});
    std::memcpy(inp.data<float>(), input_ids, B * S * sizeof(float));
    Tensor tgt(Shape{(int64_t)B, (int64_t)S});
    std::memcpy(tgt.data<float>(), targets, B * S * sizeof(float));

    AutogradEngine::set_enabled(true);
    Tensor logits = forward(inp);
    Tensor loss_t = AutogradEngine::cross_entropy_op(logits, tgt);
    Tensor& loss_mut = const_cast<Tensor&>(loss_t);
    backward(loss_mut);
    auto& engine = AutogradEngine::instance();
    engine.clear();
    AutogradEngine::set_enabled(false);

    m.loss = *(const float*)loss_t.data();
    return m;
}

void NativeQUANTMoEModel::push_weights() {
    std::vector<Tensor*> params;
    collect_all_params(params);
    for (Tensor* t : params) {
        if (!t || t->numel() == 0) continue;
        const float* src = t->data<float>();
        float* dst = temp_deq_.get();
        int64_t n = t->numel();
        std::memcpy(dst, src, n * sizeof(float));
        dst += n;
    }
}

void NativeQUANTMoEModel::pull_gradients() {
    std::vector<Tensor*> params;
    collect_all_params(params);
    for (Tensor* t : params) {
        if (!t || t->numel() == 0) continue;
        if (!t->has_grad()) {
            Tensor g(t->shape());
            g.zero_();
            t->set_grad(g);
        }
    }
}

void NativeQUANTMoEModel::print_memory_report() const {
    size_t weight_bytes = total_params_ * sizeof(float);
    size_t quant_bytes = total_params_ / 8 * 3;
    printf("[MEMORY] FP32 weights: %.2f MB\n", weight_bytes / 1e6);
    printf("[MEMORY] QUANT 1.50 BPW:  %.2f MB\n", quant_bytes / 1e6);
    printf("[MEMORY] Savings:       %.1fx\n", (float)weight_bytes / quant_bytes);
}

} // namespace native
} // namespace quant
