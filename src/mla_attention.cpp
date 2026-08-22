#include "quant/mla_attention.h"
#include "quant/random.h"
#include <cmath>
#include <vector>
#include <cstring>

namespace quant {
namespace {
void mla_init_uniform(Tensor& t, float bound, int block_idx) {
    RNG rng(42 + block_idx * 7919);
    float* d = t.data<float>();
    for (int64_t i = 0; i < t.numel(); ++i) d[i] = (rng.uniform() * 2.0f - 1.0f) * bound;
}
}

MLAAttention::MLAAttention(const TransformerConfig& cfg)
    : num_heads(std::max<int64_t>(1, cfg.num_heads)),
      head_dim(cfg.head_dim > 0 ? cfg.head_dim : 32),
      q_lora_rank(cfg.q_lora_rank > 0 ? cfg.q_lora_rank : std::max<int64_t>(8, cfg.hidden_size / 4)),
      kv_lora_rank(cfg.kv_lora_rank > 0 ? cfg.kv_lora_rank : std::max<int64_t>(8, cfg.hidden_size / 4)),
      rope_dim(cfg.mla_rope_dim > 0 ? cfg.mla_rope_dim : std::max<int64_t>(4, cfg.head_dim / 2)),
      hidden_size(cfg.hidden_size) {
    const int64_t H = cfg.hidden_size;
    w_dq = Linear(H, q_lora_rank);
    w_uq = Linear(q_lora_rank, num_heads * head_dim);
    w_dkv = Linear(H, kv_lora_rank);
    w_uk = Linear(kv_lora_rank, num_heads * head_dim);
    w_uv = Linear(kv_lora_rank, num_heads * head_dim);
    o_proj = Linear(num_heads * head_dim, H);
    q_pe_proj = Linear(q_lora_rank, rope_dim);
    k_pe_proj = Linear(kv_lora_rank, rope_dim);
    q_norm = RMSNorm(q_lora_rank);
    kv_norm = RMSNorm(kv_lora_rank);
    float b = 1.0f / std::sqrt((float)H);
    mla_init_uniform(w_dq.weight, b, 11);
    mla_init_uniform(w_uq.weight, 1.0f / std::sqrt((float)q_lora_rank), 12);
    mla_init_uniform(w_dkv.weight, b, 13);
    mla_init_uniform(w_uk.weight, 1.0f / std::sqrt((float)kv_lora_rank), 14);
    mla_init_uniform(w_uv.weight, 1.0f / std::sqrt((float)kv_lora_rank), 15);
    mla_init_uniform(o_proj.weight, 1.0f / std::sqrt((float)(num_heads * head_dim)), 16);
    mla_init_uniform(q_pe_proj.weight, b, 17);
    mla_init_uniform(k_pe_proj.weight, b, 18);
}

Tensor MLAAttention::forward_naive(const Tensor& x) const {
    QUANT_CHECK(x.rank() == 3, "MLAAttention expects {B,S,hidden}");
    const int64_t B = x.dim(0), S = x.dim(1);
    Tensor c_q = w_dq.forward(x);
    c_q = q_norm.forward(c_q);
    Tensor q = w_uq.forward(c_q);
    q = q.reshape(Shape{B, S, num_heads, head_dim});
    Tensor c_kv = w_dkv.forward(x);
    c_kv = kv_norm.forward(c_kv);
    Tensor k = w_uk.forward(c_kv);
    k = k.reshape(Shape{B, S, num_heads, head_dim});
    Tensor v = w_uv.forward(c_kv);
    v = v.reshape(Shape{B, S, num_heads, head_dim});
    Tensor q_pe = q_pe_proj.forward(c_q);
    Tensor k_pe = k_pe_proj.forward(c_kv);
    Tensor q_t = q.transpose(1, 2);
    Tensor k_t = k.transpose(1, 2);
    Tensor v_t = v.transpose(1, 2);
    const float scale = 1.0f / std::sqrt((float)head_dim);
    Tensor out(Shape{B, num_heads, S, head_dim}, DType::F32);
    out.zero_();
    const float* qd = q_t.data<float>();
    const float* kd = k_t.data<float>();
    const float* vd = v_t.data<float>();
    float* od = out.data<float>();
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < num_heads; ++h) {
            for (int64_t s = 0; s < S; ++s) {
                std::vector<float> scores((size_t)S, 0.0f);
                float mx = -1e9f;
                for (int64_t t = 0; t <= s; ++t) {
                    float sc = 0.0f;
                    for (int64_t d = 0; d < head_dim; ++d)
                        sc += qd[((b * num_heads + h) * S + s) * head_dim + d] * kd[((b * num_heads + h) * S + t) * head_dim + d];
                    sc *= scale;
                    if (rope_dim > 0 && q_pe.numel() > 0) {
                        float pe = 0.0f;
                        const float* qp = q_pe.data<float>() + (b * S + s) * rope_dim;
                        const float* kp = k_pe.data<float>() + (b * S + t) * rope_dim;
                        for (int64_t d = 0; d < rope_dim; ++d) pe += qp[d] * kp[d];
                        sc += pe * scale * 0.5f;
                    }
                    scores[(size_t)t] = sc;
                    if (sc > mx) mx = sc;
                }
                float se = 0.0f;
                for (int64_t t = 0; t <= s; ++t) { scores[(size_t)t] = std::exp(scores[(size_t)t] - mx); se += scores[(size_t)t]; }
                float inv = se > 0 ? 1.0f / se : 0.0f;
                for (int64_t t = 0; t <= s; ++t) {
                    float w = scores[(size_t)t] * inv;
                    for (int64_t d = 0; d < head_dim; ++d)
                        od[((b * num_heads + h) * S + s) * head_dim + d] += w * vd[((b * num_heads + h) * S + t) * head_dim + d];
                }
            }
        }
    }
    Tensor flat = out.transpose(1, 2).reshape(Shape{B * S, num_heads * head_dim});
    return o_proj.forward(flat).reshape(Shape{B, S, hidden_size});
}

int64_t MLAAttention::cache_bytes_per_token_mla(int64_t kv_lora_rank, int64_t rope_dim) { return (kv_lora_rank + rope_dim) * (int64_t)sizeof(float); }
int64_t MLAAttention::cache_bytes_per_token_standard(int64_t num_heads, int64_t head_dim) { return 2 * num_heads * head_dim * (int64_t)sizeof(float); }
int64_t MLAAttention::param_count() const { return w_dq.param_count() + w_uq.param_count() + w_dkv.param_count() + w_uk.param_count() + w_uv.param_count() + o_proj.param_count() + q_pe_proj.param_count() + k_pe_proj.param_count() + q_norm.weight.numel() + kv_norm.weight.numel(); }

} // namespace quant
