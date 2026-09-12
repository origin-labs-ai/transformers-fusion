#include "moe.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace quant {
namespace moe {

// ============= ModalityClassifier =============

ModalityClassifier::ModalityClassifier(int64_t hidden, int64_t num_mod)
    : hidden_size(hidden), num_modalities(num_mod)
{
    modality_head = Tensor({hidden, num_mod});
    modality_head.zero_();
}

Tensor ModalityClassifier::forward(const Tensor& x) {
    int64_t T = x.numel() / hidden_size;
    Tensor logits({T, num_modalities});
    math::gemm(1.0f, x.reshape({T, hidden_size}), modality_head, 0.0f, logits);
    return logits;
}

ModalityType ModalityClassifier::predict_modality(const Tensor& logits) {
    const float* data = logits.data<float>();
    int64_t num_mod = logits.dim(1);
    int64_t max_idx = 0;
    float max_val = data[0];
    for (int64_t i = 1; i < num_mod; ++i) {
        if (data[i] > max_val) { max_val = data[i]; max_idx = i; }
    }
    return static_cast<ModalityType>(max_idx);
}

// ============= MoERouter =============

MoERouter::MoERouter(int64_t hidden_size, const MoEConfig& cfg)
    : config_(cfg),
      modality_cls(hidden_size, 9)  // 9 modality types
{
    gate_weight = Tensor({hidden_size, cfg.num_experts});
    gate_weight.zero_();
}

RouterOutput MoERouter::forward(const Tensor& x, const Tensor& modality_hints) {
    int64_t B = x.dim(0);
    int64_t S = x.dim(1);
    int64_t hidden = x.dim(2);
    int64_t E = config_.num_experts;
    // BUGFIX (bug census): unvalidated top_k drove partial_sort(begin,
    // begin+K) OOB when K > E (UB) or K <= 0 (empty top-k → garbage routes).
    int64_t K = config_.top_k;
    if (E <= 0) return RouterOutput{};
    if (K < 1) K = 1;
    if (K > E) K = E;
    int64_t T = B * S;
    int64_t num_mod = 9;

    // 1. Classify modality of each token
    Tensor mod_logits = modality_hints;
    Tensor mod_probs({T, num_mod});
    math::softmax(mod_logits, mod_probs, 1);

    ModalityType dominant = modality_cls.predict_modality(mod_logits);

    // 2. Route: each expert gets bias based on modality
    Tensor gating_logits({T, E});
    math::gemm(1.0f, x.reshape({T, hidden}), gate_weight, 0.0f, gating_logits);

    // Add modality bias — experts are grouped by modality preference
    float* gl = gating_logits.data<float>();
    const float* mp = mod_probs.data<float>();
    // Expert 0..text_experts-1 prefer TEXT, text_experts..+vision_experts prefer VISION, etc.
    int64_t expert_offset = 0;
    struct ModRange { ModalityType mod; int64_t start; int64_t count; };
    ModRange ranges[] = {
        {ModalityType::TEXT, 0, config_.text_experts},
        {ModalityType::VISION, config_.text_experts, config_.vision_experts},
        {ModalityType::IMAGE_GEN, config_.text_experts + config_.vision_experts, config_.image_gen_experts},
        {ModalityType::VIDEO_GEN, config_.text_experts + config_.vision_experts + config_.image_gen_experts, config_.video_gen_experts},
        {ModalityType::AUDIO, config_.text_experts + config_.vision_experts + config_.image_gen_experts + config_.video_gen_experts, config_.audio_experts},
        {ModalityType::OCR, config_.text_experts + config_.vision_experts + config_.image_gen_experts + config_.video_gen_experts + config_.audio_experts, config_.ocr_experts},
        {ModalityType::CROSS_MODAL, config_.text_experts + config_.vision_experts + config_.image_gen_experts + config_.video_gen_experts + config_.audio_experts + config_.ocr_experts, config_.cross_modal_experts}
    };

    for (int64_t t = 0; t < T; ++t) {
        for (auto& r : ranges) {
            float mod_score = mp[t * num_mod + (int64_t)r.mod];
            for (int64_t e = r.start; e < r.start + r.count && e < E; ++e)
                gl[t * E + e] += 10.0f * mod_score;  // modality bias
        }
    }

    // 3. Top-K routing
    Tensor probs({T, E});
    math::softmax(gating_logits, probs, 1);

    Tensor indices({T, K}, DType::I64);
    Tensor weights({T, K});
    float* p_ptr = probs.data<float>();
    float* w_ptr = weights.data<float>();
    int64_t* idx_ptr = indices.data<int64_t>();

    for (int64_t t = 0; t < T; ++t) {
        float* row = p_ptr + t * E;
        std::vector<std::pair<float, int64_t>> scored;
        scored.reserve(E);
        for (int64_t e = 0; e < E; ++e)
            scored.push_back({row[e], e});
        std::partial_sort(scored.begin(), scored.begin() + K, scored.end(),
            [](auto& a, auto& b) { return a.first > b.first; });
        for (int64_t k = 0; k < K; ++k) {
            idx_ptr[t * K + k] = scored[k].second;
            w_ptr[t * K + k] = scored[k].first;
        }
    }

    // 4. Load balance calculation
    std::vector<double> f_i(E, 0.0);
    std::vector<double> P_i(E, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t k = 0; k < K; ++k) {
            int64_t e = idx_ptr[t * K + k];
            f_i[e] += 1.0;
            P_i[e] += w_ptr[t * K + k];
        }
    }
    double load_balance = 0.0;
    for (int64_t e = 0; e < E; ++e) {
        f_i[e] /= (double)T;
        P_i[e] /= (double)T;
        load_balance += f_i[e] * P_i[e];
    }
    load_balance *= (double)E;

    RouterOutput out;
    out.gates = probs.reshape({B, S, E});
    out.indices = indices.reshape({B, S, K});
    out.weights = weights.reshape({B, S, K});
    out.modality_probs = mod_probs.reshape({B, S, num_mod});
    out.dominant_modality = dominant;
    out.load_balance_loss = (float)load_balance;
    double z_lse = 0.0;
    for (int64_t t = 0; t < T; ++t) {
        float row_max = gl[t * E];
        for (int64_t e = 1; e < E; ++e)
            if (gl[t * E + e] > row_max) row_max = gl[t * E + e];
        double lse = 0.0;
        for (int64_t e = 0; e < E; ++e)
            lse += std::exp((double)(gl[t * E + e] - row_max));
        lse = (double)row_max + std::log(lse);
        z_lse += lse * lse;
    }
    out.z_loss = static_cast<float>(config_.z_loss_coef * z_lse / (float)T);
    return out;
}

// ============= MoEFFN =============

MoEFFN::MoEFFN(const MoEConfig& cfg, const TransformerConfig& tcfg)
    : config_(cfg)
{
    int64_t total = cfg.text_experts + cfg.vision_experts + cfg.image_gen_experts
                  + cfg.video_gen_experts + cfg.audio_experts + cfg.ocr_experts
                  + cfg.cross_modal_experts;
    experts.reserve(total);
    expert_modalities.reserve(total);

    auto add_experts = [&](int64_t count, ModalityType mod) {
        for (int64_t i = 0; i < count; ++i) {
            experts.emplace_back(tcfg);
            expert_modalities.push_back(mod);
        }
    };

    add_experts(cfg.text_experts, ModalityType::TEXT);
    add_experts(cfg.vision_experts, ModalityType::VISION);
    add_experts(cfg.image_gen_experts, ModalityType::IMAGE_GEN);
    add_experts(cfg.video_gen_experts, ModalityType::VIDEO_GEN);
    add_experts(cfg.audio_experts, ModalityType::AUDIO);
    add_experts(cfg.ocr_experts, ModalityType::OCR);
    add_experts(cfg.cross_modal_experts, ModalityType::CROSS_MODAL);
}

Tensor MoEFFN::forward(const Tensor& x, const RouterOutput& routing) {
    int64_t B = x.dim(0);
    int64_t S = x.dim(1);
    int64_t hidden = x.dim(2);
    int64_t T = B * S;
    int64_t K = routing.indices.dim(2);
    int64_t E = (int64_t)experts.size();

    Tensor output({T, hidden});
    output.zero_();

    const int64_t* idx_ptr = routing.indices.data<int64_t>();
    const float* w_ptr = routing.weights.data<float>();
    Tensor x_flat = x.reshape({T, hidden});
    const float* x_ptr = x_flat.data<float>();
    float* o_ptr = output.data<float>();

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t k = 0; k < K; ++k) {
            int64_t e = idx_ptr[t * K + k];
            float weight = w_ptr[t * K + k];
            if (e < 0 || e >= E) continue;

            Tensor token_in({1, hidden});
            std::memcpy(token_in.data<float>(), x_ptr + t * hidden, hidden * sizeof(float));

            Tensor expert_out = experts[e].forward(token_in);

            for (int64_t h = 0; h < hidden; ++h)
                o_ptr[t * hidden + h] += weight * expert_out.data<float>()[h];
        }
    }
    last_z_loss = routing.z_loss;

    Tensor out = output.reshape({B, S, hidden});
    return out;
}

// ============= CrossModalAttention =============

CrossModalAttention::CrossModalAttention(int64_t hidden, int64_t n_heads)
    : hidden_size(hidden), num_heads(n_heads), head_dim(hidden / n_heads)
{
    q_proj = Tensor({hidden, hidden}); q_proj.zero_();
    k_proj = Tensor({hidden, hidden}); k_proj.zero_();
    v_proj = Tensor({hidden, hidden}); v_proj.zero_();
    o_proj = Tensor({hidden, hidden}); o_proj.zero_();
}

Tensor CrossModalAttention::forward(const Tensor& query_mod, const Tensor& kv_mod) {
    int64_t B = query_mod.dim(0);
    int64_t Sq = query_mod.dim(1);
    int64_t Skv = kv_mod.dim(1);
    int64_t D = hidden_size;
    int64_t H = num_heads;
    int64_t Dh = head_dim;

    int64_t Tq = B * Sq, Tkv = B * Skv;

    // Project to Q, K, V
    auto project = [&](const Tensor& proj_w, const Tensor& inp, int64_t N) -> Tensor {
        Tensor out({N, D});
        math::gemm(1.0f, inp.reshape({N, D}), proj_w, 0.0f, out);
        return out;
    };
    Tensor q_all = project(q_proj, query_mod, Tq).reshape({B, Sq, H, Dh});
    Tensor k_all = project(k_proj, kv_mod, Tkv).reshape({B, Skv, H, Dh});
    Tensor v_all = project(v_proj, kv_mod, Tkv).reshape({B, Skv, H, Dh});

    // Attention per batch/head
    Tensor context({B, Sq, H, Dh});
    float* ctx = context.data<float>();
    const float* qd = q_all.data<float>();
    const float* kd = k_all.data<float>();
    const float* vd = v_all.data<float>();
    float scale = 1.0f / std::sqrt((float)Dh);

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            // Pointers for this head
            const float* qh = qd + ((b * Sq) * H + h) * Dh;
            const float* kh = kd + ((b * Skv) * H + h) * Dh;
            const float* vh = vd + ((b * Skv) * H + h) * Dh;
            float* ch = ctx + ((b * Sq) * H + h) * Dh;

            // scores[seq_i][seq_j] = dot(q[seq_i], k[seq_j]) * scale
            Tensor scores({Sq, Skv});
            float* sc = scores.data<float>();
            for (int64_t i = 0; i < Sq; ++i)
                for (int64_t j = 0; j < Skv; ++j) {
                    float dot = 0.0f;
                    for (int64_t d = 0; d < Dh; ++d)
                        dot += qh[i * H * Dh + d] * kh[j * H * Dh + d];
                    sc[i * Skv + j] = dot * scale;
                }

            // softmax over j (Skv dim)
            for (int64_t i = 0; i < Sq; ++i) {
                float row_max = sc[i * Skv];
                for (int64_t j = 1; j < Skv; ++j)
                    if (sc[i * Skv + j] > row_max) row_max = sc[i * Skv + j];
                float sum = 0.0f;
                for (int64_t j = 0; j < Skv; ++j) {
                    sc[i * Skv + j] = std::exp(sc[i * Skv + j] - row_max);
                    sum += sc[i * Skv + j];
                }
                for (int64_t j = 0; j < Skv; ++j)
                    sc[i * Skv + j] /= sum;
            }

            // context = scores @ v
            std::memset(ch, 0, Sq * Dh * sizeof(float));
            for (int64_t i = 0; i < Sq; ++i)
                for (int64_t j = 0; j < Skv; ++j)
                    for (int64_t d = 0; d < Dh; ++d)
                        ch[i * H * Dh + d] += sc[i * Skv + j] * vh[j * H * Dh + d];
        }
    }

    Tensor out({Tq, D});
    math::gemm(1.0f, context.reshape({Tq, D}), o_proj, 0.0f, out);
    return out.reshape({B, Sq, D});
}

// ============= MoMBlock =============

MoMBlock::MoMBlock(const MoEConfig& moe_cfg, const TransformerConfig& tcfg)
    : attention_norm(tcfg.hidden_size, tcfg.norm_eps),
      attention(tcfg),
      moe_norm(tcfg.hidden_size, tcfg.norm_eps),
      router(tcfg.hidden_size, moe_cfg),
      moe_ffn(moe_cfg, tcfg),
      cross_attn(tcfg.hidden_size, tcfg.num_heads)
{
}

Tensor MoMBlock::forward(const Tensor& x, const Tensor& positions,
                          const Tensor& mask, KVCache& cache, int layer_idx) {
    // Self-attention
    Tensor attn_normed = attention_norm.forward(x);
    Tensor attn_out = attention.forward(attn_normed, positions, mask, cache, layer_idx);
    Tensor residual({x.shape()});
    math::add(x, attn_out, residual);

    // Modality-aware MoE
    Tensor moe_normed = moe_norm.forward(residual);
    Tensor modality_hints = router.modality_cls.forward(moe_normed);
    RouterOutput rout = router.forward(moe_normed, modality_hints);
    Tensor moe_out = moe_ffn.forward(moe_normed, rout);

    // Cross-modal attention between pre-MoE and post-MoE features
    Tensor cross_out = cross_attn.forward(moe_normed, moe_out);

    Tensor result({residual.shape()});
    math::add(residual, moe_out, result);
    Tensor final_result({result.shape()});
    math::add(result, cross_out, final_result);
    return final_result;
}

} // namespace moe
} // namespace quant
