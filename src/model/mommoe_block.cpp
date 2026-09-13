// ============================================================================
// PILLAR 5: MoMMoE Block — Mixture of Multimodal Experts Implementation
// ============================================================================

#include "quant/mommoe_block.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace quant {
namespace moe {

// ============================================================================
// TopKRouter — Modality-aware Top-2 expert routing
// ============================================================================

TopKRouter::TopKRouter(int64_t hidden_size, int64_t num_experts, int64_t top_k,
                       int64_t num_modalities)
    : hidden_size_(hidden_size), num_experts_(num_experts), top_k_(top_k),
      num_modalities_(num_modalities) {
    gate_weight_ = Tensor(Shape{hidden_size, num_experts}, DType::F32);
    modality_head_ = Tensor(Shape{hidden_size, num_modalities}, DType::F32);
    // Xavier init
    float scale = std::sqrt(2.0f / hidden_size);
    float* gw = gate_weight_.data<float>();
    float* mh = modality_head_.data<float>();
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, scale);
    for (int64_t i = 0; i < hidden_size * num_experts; ++i) gw[i] = dist(rng);
    for (int64_t i = 0; i < hidden_size * num_modalities; ++i) mh[i] = dist(rng);
    expert_counts_.resize(num_experts, 0);
}

RouterOutput TopKRouter::forward(const Tensor& x, const Tensor* modality_hints) {
    int64_t B = x.dim(0);
    int64_t S = x.rank() > 1 ? x.dim(1) : 1;
    int64_t T = B * S;

    // Compute gate logits: x @ gate_weight
    Tensor logits(Shape{T, num_experts_}, DType::F32);
    logits.zero_();
    const float* x_data = x.data<float>();
    const float* gw = gate_weight_.data<float>();
    float* log_data = logits.data<float>();
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t e = 0; e < num_experts_; ++e) {
            float sum = 0.0f;
            for (int64_t d = 0; d < hidden_size_; ++d) {
                sum += x_data[t * hidden_size_ + d] * gw[d * num_experts_ + e];
            }
            log_data[t * num_experts_ + e] = sum;
        }
    }

    // Softmax
    Tensor probs(Shape{T, num_experts_}, DType::F32);
    math::softmax(logits, probs);

    // Top-K selection
    Tensor indices(Shape{T, top_k_}, DType::I64);
    Tensor weights(Shape{T, top_k_}, DType::F32);
    indices.zero_();
    weights.zero_();
    int64_t* idx_data = indices.data<int64_t>();
    float* wt_data = weights.data<float>();
    const float* prob_data = probs.data<float>();

    for (int64_t t = 0; t < T; ++t) {
        // Find top-K
        std::vector<int64_t> sorted_idx(num_experts_);
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::partial_sort(sorted_idx.begin(), sorted_idx.begin() + top_k_, sorted_idx.end(),
                          [&](int64_t a, int64_t b) {
                              return prob_data[t * num_experts_ + a] >
                                     prob_data[t * num_experts_ + b];
                          });
        float weight_sum = 0.0f;
        for (int64_t k = 0; k < top_k_; ++k) {
            idx_data[t * top_k_ + k] = sorted_idx[k];
            wt_data[t * top_k_ + k] = prob_data[t * num_experts_ + sorted_idx[k]];
            weight_sum += wt_data[t * top_k_ + k];
            expert_counts_[sorted_idx[k]]++;
        }
        total_tokens_routed_++;
        // Normalize weights
        for (int64_t k = 0; k < top_k_; ++k) {
            wt_data[t * top_k_ + k] /= weight_sum;
        }
    }

    // Modality classification
    Tensor mod_probs(Shape{T, num_modalities_}, DType::F32);
    mod_probs.zero_();
    float* mp = mod_probs.data<float>();
    const float* mh = modality_head_.data<float>();
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t m = 0; m < num_modalities_; ++m) {
            float sum = 0.0f;
            for (int64_t d = 0; d < hidden_size_; ++d) {
                sum += x_data[t * hidden_size_ + d] * mh[d * num_modalities_ + m];
            }
            mp[t * num_modalities_ + m] = sum;
        }
    }
    // Softmax modality probs
    Tensor mod_softmax(Shape{T, num_modalities_}, DType::F32);
    math::softmax(mod_probs, mod_softmax);

    // Determine dominant modality
    Modality dominant = Modality::TEXT;
    float max_mod_prob = 0.0f;
    for (int64_t m = 0; m < num_modalities_; ++m) {
        float avg = 0.0f;
        for (int64_t t = 0; t < T; ++t) avg += mp[t * num_modalities_ + m];
        avg /= T;
        if (avg > max_mod_prob) { max_mod_prob = avg; dominant = static_cast<Modality>(m); }
    }

    RouterOutput out;
    out.gates = std::move(probs);
    out.indices = std::move(indices);
    out.weights = std::move(weights);
    out.modality_probs = std::move(mod_softmax);
    out.dominant_modality = dominant;
    out.load_balance_loss = load_balance_loss(out.gates, out.indices);
    out.z_loss = z_loss(logits);
    out.num_activated_experts = top_k_;
    return out;
}

float TopKRouter::load_balance_loss(const Tensor& gates, const Tensor& indices) const {
    int64_t T = gates.dim(0);
    int64_t E = num_experts_;
    // f_i = fraction of tokens routed to expert i
    std::vector<float> f_i(E, 0.0f);
    const int64_t* idx = indices.data<int64_t>();
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t k = 0; k < top_k_; ++k) {
            f_i[idx[t * top_k_ + k]] += 1.0f;
        }
    }
    float scale = 1.0f / (T * top_k_);
    for (auto& f : f_i) f *= scale;

    // P_i = average gate probability for expert i
    std::vector<float> P_i(E, 0.0f);
    const float* g = gates.data<float>();
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t e = 0; e < E; ++e) {
            P_i[e] += g[t * E + e];
        }
    }
    for (auto& p : P_i) p /= T;

    float loss = 0.0f;
    for (int64_t e = 0; e < E; ++e) {
        loss += f_i[e] * P_i[e];
    }
    return E * loss;
}

float TopKRouter::z_loss(const Tensor& logits) const {
    int64_t T = logits.dim(0);
    int64_t E = num_experts_;
    const float* l = logits.data<float>();
    float z = 0.0f;
    for (int64_t t = 0; t < T; ++t) {
        float max_l = *std::max_element(l + t * E, l + (t + 1) * E);
        float sum_exp = 0.0f;
        for (int64_t e = 0; e < E; ++e) {
            sum_exp += std::exp(l[t * E + e] - max_l);
        }
        float log_sum = max_l + std::log(sum_exp);
        z += log_sum * log_sum;
    }
    return z / T;
}

Modality TopKRouter::classify_modality(const Tensor& hidden) const {
    int64_t T = hidden.dim(0);
    Modality best = Modality::TEXT;
    float max_prob = -1e30f;
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t m = 0; m < num_modalities_; ++m) {
            float sum = 0.0f;
            const float* h = hidden.data<float>() + t * hidden_size_;
            const float* mh = modality_head_.data<float>();
            for (int64_t d = 0; d < hidden_size_; ++d) {
                sum += h[d] * mh[d * num_modalities_ + m];
            }
            if (sum > max_prob) { max_prob = sum; best = static_cast<Modality>(m); }
        }
    }
    return best;
}

// ============================================================================
// CrossModalAttention
// ============================================================================

CrossModalAttention::CrossModalAttention(int64_t hidden_size, int64_t num_heads,
                                         int64_t head_dim)
    : hidden_size_(hidden_size), num_heads_(num_heads),
      head_dim_(head_dim > 0 ? head_dim : hidden_size / num_heads) {
    q_proj_ = Tensor(Shape{hidden_size, hidden_size}, DType::F32);
    k_proj_ = Tensor(Shape{hidden_size, hidden_size}, DType::F32);
    v_proj_ = Tensor(Shape{hidden_size, hidden_size}, DType::F32);
    o_proj_ = Tensor(Shape{hidden_size, hidden_size}, DType::F32);
    float scale = std::sqrt(2.0f / hidden_size);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, scale);
    for (auto* t : {&q_proj_, &k_proj_, &v_proj_, &o_proj_}) {
        float* d = t->data<float>();
        for (int64_t i = 0; i < t->numel(); ++i) d[i] = dist(rng);
    }
}

Tensor CrossModalAttention::forward(const Tensor& query_tokens,
                                    const Tensor& key_value_tokens,
                                    const Tensor* attention_mask) {
    int64_t Tq = query_tokens.dim(0);
    int64_t Tk = key_value_tokens.dim(0);
    float scale = 1.0f / std::sqrt((float)head_dim_);

    Tensor output(Shape{Tq, hidden_size_}, DType::F32);
    output.zero_();

    // Q = query @ q_proj, K = key_value @ k_proj, V = key_value @ v_proj
    // Simplified: direct matmul without batched attention
    for (int64_t t = 0; t < Tq; ++t) {
        for (int64_t h = 0; h < num_heads_; ++h) {
            float attn_max = -1e30f;
            // Compute attention scores
            std::vector<float> scores(Tk);
            for (int64_t s = 0; s < Tk; ++s) {
                float score = 0.0f;
                for (int64_t d = 0; d < head_dim_; ++d) {
                    int64_t q_idx = t * hidden_size_ + h * head_dim_ + d;
                    int64_t k_idx = s * hidden_size_ + h * head_dim_ + d;
                    score += query_tokens.data<float>()[q_idx] *
                             key_value_tokens.data<float>()[k_idx];
                }
                scores[s] = score * scale;
                attn_max = std::max(attn_max, scores[s]);
            }
            // Softmax
            float sum_exp = 0.0f;
            for (auto& s : scores) { s = std::exp(s - attn_max); sum_exp += s; }
            for (auto& s : scores) s /= sum_exp;
            // Weighted sum of values
            for (int64_t d = 0; d < head_dim_; ++d) {
                float val = 0.0f;
                for (int64_t s = 0; s < Tk; ++s) {
                    val += scores[s] * key_value_tokens.data<float>()[s * hidden_size_ + h * head_dim_ + d];
                }
                output.data<float>()[t * hidden_size_ + h * head_dim_ + d] = val;
            }
        }
    }
    return output;
}

// ============================================================================
// ModalityExpertFFN
// ============================================================================

ModalityExpertFFN::ModalityExpertFFN() : specialization_(Modality::TEXT) {}

ModalityExpertFFN::ModalityExpertFFN(int64_t hidden_size, int64_t ffn_hidden)
    : specialization_(Modality::TEXT) {
    gate_proj_ = Tensor(Shape{hidden_size, ffn_hidden}, DType::F32);
    up_proj_ = Tensor(Shape{hidden_size, ffn_hidden}, DType::F32);
    down_proj_ = Tensor(Shape{ffn_hidden, hidden_size}, DType::F32);
    float scale = std::sqrt(2.0f / hidden_size);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, scale);
    for (auto* t : {&gate_proj_, &up_proj_, &down_proj_}) {
        float* d = t->data<float>();
        for (int64_t i = 0; i < t->numel(); ++i) d[i] = dist(rng);
    }
}

ModalityExpertFFN::ModalityExpertFFN(int64_t hidden_size, int64_t ffn_hidden, Modality specialization)
    : ModalityExpertFFN(hidden_size, ffn_hidden) {
    specialization_ = specialization;
}

Tensor ModalityExpertFFN::forward(const Tensor& x) const {
    int64_t T = x.dim(0);
    int64_t D = gate_proj_.dim(0);
    int64_t F = gate_proj_.dim(1);

    // gate = x @ gate_proj
    Tensor gate(Shape{T, F}, DType::F32);
    // up = x @ up_proj
    Tensor up(Shape{T, F}, DType::F32);
    // output = down_proj(silu(gate) * up)

    const float* xd = x.data<float>();
    const float* gd = gate_proj_.data<float>();
    const float* ud = up_proj_.data<float>();

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t f = 0; f < F; ++f) {
            float g = 0.0f, u = 0.0f;
            for (int64_t d = 0; d < D; ++d) {
                g += xd[t * D + d] * gd[d * F + f];
                u += xd[t * D + d] * ud[d * F + f];
            }
            gate.data<float>()[t * F + f] = g;
            up.data<float>()[t * F + f] = u;
        }
    }

    // SwiGLU: silu(gate) * up
    Tensor gated(Shape{T, F}, DType::F32);
    math::silu(gate, gate);
    math::mul(gate, up, gated);

    // output = gated @ down_proj
    Tensor output(Shape{T, D}, DType::F32);
    output.zero_();
    const float* gtd = gated.data<float>();
    const float* dd = down_proj_.data<float>();
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t d = 0; d < D; ++d) {
            float sum = 0.0f;
            for (int64_t f = 0; f < F; ++f) {
                sum += gtd[t * F + f] * dd[f * D + d];
            }
            output.data<float>()[t * D + d] = sum;
        }
    }
    return output;
}

// ============================================================================
// MoMBlock — Single Multimodal MoE Transformer Block
// ============================================================================

MoMBlock::MoMBlock(int64_t hidden_size, int64_t num_heads, int64_t ffn_hidden,
                   int64_t num_experts, int64_t top_k, int64_t num_modalities)
    : hidden_size_(hidden_size), num_experts_(num_experts),
      attention_norm_(hidden_size),
      cross_modal_attn_(hidden_size, num_heads),
      moe_norm_(hidden_size),
      router_(hidden_size, num_experts, top_k, num_modalities) {
    TransformerConfig attn_cfg;
    attn_cfg.hidden_size = hidden_size;
    attn_cfg.num_heads = num_heads;
    attn_cfg.head_dim = hidden_size / num_heads;
    attn_cfg.max_seq_len = 2048;
    self_attention_ = Attention(attn_cfg);
    // Create experts with different modalities
    Modality mods[] = {Modality::TEXT, Modality::TEXT, Modality::TEXT, Modality::TEXT,
                       Modality::VISION, Modality::AUDIO, Modality::IMAGE_GEN, Modality::VIDEO_GEN};
    for (int64_t i = 0; i < num_experts; ++i) {
        experts_.emplace_back(hidden_size, ffn_hidden, mods[i % static_cast<int>(Modality::COUNT)]);
    }
}

MoMBlock::MoMOutput MoMBlock::forward(const Tensor& x, const Tensor* modality_hints,
                                       KVCache* cache, int64_t layer_idx) {
    MoMOutput out;

    // 1. Attention Norm + Self-Attention
    Tensor normed_x = attention_norm_.forward(x);
    int64_t T_attn = normed_x.dim(0);
    Tensor positions({T_attn}, DType::F32);
    {
        float* pd = positions.data<float>();
        for (int64_t i = 0; i < T_attn; i++) pd[i] = (float)i;
    }
    Tensor mask({T_attn, T_attn}, DType::F32);
    mask.zero_();
    static KVCache dummy_cache;
    KVCache& cache_ref = cache ? *cache : dummy_cache;
    out.hidden = self_attention_.forward(normed_x, positions, mask, cache_ref, (int)layer_idx);
    // Residual
    for (int64_t i = 0; i < x.numel(); ++i) {
        out.hidden.data<float>()[i] += x.data<float>()[i];
    }

    // 2. Cross-Modal Attention (if modality hints provided)
    if (modality_hints) {
        out.cross_modal_output = cross_modal_attn_.forward(out.hidden, *modality_hints);
        for (int64_t i = 0; i < out.hidden.numel(); ++i) {
            out.hidden.data<float>()[i] += out.cross_modal_output.data<float>()[i];
        }
    }

    // 3. MoE Norm + Router + Expert FFNs
    Tensor moe_input = moe_norm_.forward(out.hidden);
    out.routing = router_.forward(moe_input, modality_hints);

    // Dispatch to experts and combine
    Tensor moe_output(Shape{out.hidden.dim(0), hidden_size_}, DType::F32);
    moe_output.zero_();

    int64_t T = moe_input.dim(0);
    const int64_t* idx = out.routing.indices.data<int64_t>();
    const float* wt = out.routing.weights.data<float>();

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t k = 0; k < router_.top_k(); ++k) {
            int64_t expert_id = idx[t * router_.top_k() + k];
            float weight = wt[t * router_.top_k() + k];
            // Forward single token through expert
            Tensor token(Shape{1, hidden_size_}, DType::F32);
            std::memcpy(token.data<float>(), moe_input.data<float>() + t * hidden_size_,
                        hidden_size_ * sizeof(float));
            Tensor expert_out = experts_[expert_id].forward(token);
            // Weighted combine
            for (int64_t d = 0; d < hidden_size_; ++d) {
                moe_output.data<float>()[t * hidden_size_ + d] +=
                    weight * expert_out.data<float>()[d];
            }
        }
    }

    // Residual
    for (int64_t i = 0; i < out.hidden.numel(); ++i) {
        out.hidden.data<float>()[i] += moe_output.data<float>()[i];
    }

    return out;
}

// ============================================================================
// MoMMoE — Full Model
// ============================================================================

MoMMoE::MoMMoE(const MoMMoEConfig& cfg) : cfg_(cfg) {
    embeddings_ = Tensor(Shape{cfg.vocab_size, cfg.hidden_size}, DType::F32);
    float scale = std::sqrt(2.0f / cfg.hidden_size);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, scale);
    float* ed = embeddings_.data<float>();
    for (int64_t i = 0; i < cfg.vocab_size * cfg.hidden_size; ++i) ed[i] = dist(rng);

    for (int64_t i = 0; i < cfg.num_layers; ++i) {
        layers_.emplace_back(cfg.hidden_size, cfg.num_heads, cfg.ffn_hidden,
                             cfg.num_experts, cfg.top_k, cfg.num_modalities);
    }

    final_norm_ = Tensor(Shape{cfg.hidden_size}, DType::F32);
    float* nd = final_norm_.data<float>();
    for (int64_t i = 0; i < cfg.hidden_size; ++i) nd[i] = 1.0f;

    lm_head_ = Tensor(Shape{cfg.hidden_size, cfg.vocab_size}, DType::F32);
    float* ld = lm_head_.data<float>();
    for (int64_t i = 0; i < cfg.hidden_size * cfg.vocab_size; ++i) ld[i] = dist(rng);
}

Tensor MoMMoE::forward(const Tensor& input_ids, const Tensor* modality_hints) {
    int64_t B = input_ids.dim(0);
    int64_t S = input_ids.dim(1);

    // Embed tokens
    Tensor x(Shape{B * S, cfg_.hidden_size}, DType::F32);
    const int64_t* ids = input_ids.data<int64_t>();
    const float* emb = embeddings_.data<float>();
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s < S; ++s) {
            int64_t id = ids[b * S + s];
            std::memcpy(x.data<float>() + (b * S + s) * cfg_.hidden_size,
                        emb + id * cfg_.hidden_size,
                        cfg_.hidden_size * sizeof(float));
        }
    }

    // Forward through layers
    for (auto& layer : layers_) {
        auto out = layer.forward(x, modality_hints);
        x = std::move(out.hidden);
    }

    // Final norm + LM head
    int64_t B_rms = x.dim(0);
    int64_t D_rms = cfg_.hidden_size;
    const float* xd_rms = x.data<float>();
    const float* w_rms = final_norm_.data<float>();
    Tensor normed({B_rms, D_rms}, DType::F32);
    float* nd_out = normed.data<float>();
    const float eps_rms = 1e-5f;
    for (int64_t b = 0; b < B_rms; b++) {
        float ss = 0.0f;
        for (int64_t d = 0; d < D_rms; d++) {
            float v = xd_rms[b * D_rms + d];
            ss += v * v;
        }
        float inv = 1.0f / std::sqrt(ss / D_rms + eps_rms);
        for (int64_t d = 0; d < D_rms; d++) {
            nd_out[b * D_rms + d] = w_rms[d] * inv * xd_rms[b * D_rms + d];
        }
    }

    Tensor logits(Shape{B * S, cfg_.vocab_size}, DType::F32);
    logits.zero_();
    float* out = logits.data<float>();
    const float* xd = x.data<float>();
    const float* hd = lm_head_.data<float>();
    for (int64_t t = 0; t < B * S; ++t) {
        for (int64_t v = 0; v < cfg_.vocab_size; ++v) {
            float sum = 0.0f;
            for (int64_t d = 0; d < cfg_.hidden_size; ++d) {
                sum += xd[t * cfg_.hidden_size + d] * hd[d * cfg_.vocab_size + v];
            }
            out[t * cfg_.vocab_size + v] = sum;
        }
    }

    return logits;
}

Tensor MoMMoE::generate(const Tensor& prompt_ids, int64_t max_new_tokens,
                         float temperature, int64_t top_k) {
    // Simple greedy generation
    Tensor current = prompt_ids.clone();
    for (int64_t step = 0; step < max_new_tokens; ++step) {
        Tensor logits = forward(current);
        int64_t T = logits.dim(0);
        int64_t V = logits.dim(1);
        // Get last token logits
        const float* last = logits.data<float>() + (T - 1) * V;
        // Greedy argmax
        int64_t best = 0;
        float best_val = last[0];
        for (int64_t v = 1; v < V; ++v) {
            if (last[v] > best_val) { best_val = last[v]; best = v; }
        }
        // Append
        Tensor new_token(Shape{1, 1}, DType::I64);
        new_token.data<int64_t>()[0] = best;
        // Concat (simplified)
        Tensor expanded(Shape{current.dim(0) + 1}, DType::I64);
        std::memcpy(expanded.data<int64_t>(), current.data<int64_t>(), current.numel() * sizeof(int64_t));
        expanded.data<int64_t>()[current.numel()] = best;
        current = expanded;
    }
    return current;
}

int64_t MoMMoE::total_parameters() const {
    int64_t total = embeddings_.numel() + lm_head_.numel() + final_norm_.numel();
    for (auto& layer : layers_) {
        total += layer.attention_norm().weight.numel();
        total += layer.self_attention().q_proj.weight.numel()
               + layer.self_attention().k_proj.weight.numel()
               + layer.self_attention().v_proj.weight.numel();
        total += layer.cross_modal_attn().q_proj().numel()
               + layer.cross_modal_attn().k_proj().numel()
               + layer.cross_modal_attn().v_proj().numel()
               + layer.cross_modal_attn().o_proj().numel();
        total += layer.moe_norm().weight.numel();
        total += layer.router().gate_weight().numel();
        for (auto& expert : layer.experts()) {
            total += expert.gate_proj().numel() + expert.up_proj().numel() + expert.down_proj().numel();
        }
    }
    return total;
}

int64_t MoMMoE::active_parameters() const {
    // Top-2 routing: each token activates 2 experts out of 8
    // Active = (2/8) * expert_params + attention_params + embeddings
    int64_t expert_params = 0;
    for (auto& expert : layers_[0].experts()) {
        expert_params += expert.gate_proj().numel() + expert.up_proj().numel() + expert.down_proj().numel();
    }
    int64_t active_per_layer = expert_params * cfg_.top_k / cfg_.num_experts;
    int64_t attention_per_layer = layers_[0].self_attention().q_proj.weight.numel()
                                + layers_[0].self_attention().k_proj.weight.numel()
                                + layers_[0].self_attention().v_proj.weight.numel();
    return (active_per_layer + attention_per_layer) * cfg_.num_layers + embeddings_.numel();
}

std::vector<float> MoMMoE::expert_utilization() const {
    // Uniform expert utilization estimate — refined when router stats are tracked
    return std::vector<float>(cfg_.num_experts, 1.0f / cfg_.num_experts);
}

} // namespace moe
} // namespace quant
