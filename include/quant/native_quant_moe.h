#pragma once
#include "quant/moe_variants.h"
#include "quant/native_weight.h"
#include "quant/transformer.h"
#include "quant/autograd.h"
#include <vector>

namespace quant {
namespace native {

struct NativeMoEConfig {
    int64_t hidden_size = 512;
    int64_t num_heads = 8;
    int64_t ffn_hidden = 1024;
    int64_t num_layers = 4;
    int64_t num_experts = 8;
    int64_t top_k = 2;
    float load_balance_coef = 0.01f;
    float z_loss_coef = 0.001f;
    size_t block_size = 128;
    float frac_quant8 = 0.01f;
    float frac_quant = 0.95f;
    int64_t vocab_size = 256;
    int64_t max_seq_len = 128;
};

struct NativeMoEMetrics {
    float loss = 0.0f;
    float lb_loss = 0.0f;
    float z_loss = 0.0f;
    float grad_norm = 0.0f;
    size_t frozen_pct = 0;
    size_t quant8 = 0;
    size_t quant = 0;
    size_t quant1 = 0;
    size_t total_params = 0;
    size_t active_params = 0;
};

class NativeQUANTMoEModel {
public:
    explicit NativeQUANTMoEModel(const NativeMoEConfig& cfg);
    ~NativeQUANTMoEModel();

    Tensor forward(const Tensor& input_ids);
    void backward(const Tensor& loss);
    void apply_update(float lr_scale, float lr_weight);
    void init_from_fp32();
    NativeMoEMetrics step(const float* input_ids, const float* targets,
                          size_t B, size_t S, float lr_s, float lr_w);
    size_t total_params() const { return total_params_; }
    void print_memory_report() const;

private:
    NativeMoEConfig cfg_;
    Embedding* emb_;
    std::vector<std::unique_ptr<TransformerBlock>> layers_;
    RMSNorm* norm_;
    Linear* lm_head_;
    moe::SparseMoE* moe_;

    size_t num_moe_stores_;
    std::unique_ptr<NativeQWeightStore> emb_store_;
    std::unique_ptr<NativeQWeightStore> head_store_;
    std::vector<std::unique_ptr<NativeQWeightStore>> moe_expert_stores_;

    std::unique_ptr<float[]> temp_deq_;
    size_t total_params_;
    size_t emb_params_;
    size_t head_params_;

    void collect_all_params(std::vector<Tensor*>& params);
    void push_weights();
    void pull_gradients();
};

} // namespace native
} // namespace quant
