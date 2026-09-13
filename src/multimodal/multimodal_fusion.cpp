#include "quant/multimodal.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace quant {

// ============================================================================
// H22: ModalityProjection — per-modality linear projection heads
// ============================================================================
ModalityProjection::ModalityProjection(int64_t hidden)
    : vision_proj(hidden, hidden), audio_proj(hidden, hidden),
      text_proj(hidden, hidden), hidden_size(hidden) {}

Tensor ModalityProjection::project_vision(const Tensor& vision_features) const {
    return vision_proj.forward(vision_features);
}

Tensor ModalityProjection::project_audio(const Tensor& audio_features) const {
    return audio_proj.forward(audio_features);
}

Tensor ModalityProjection::project_text(const Tensor& text_features) const {
    return text_proj.forward(text_features);
}

Tensor ModalityProjection::project(const Tensor& features, const std::string& modality) const {
    if (modality == "vision") return project_vision(features);
    if (modality == "audio") return project_audio(features);
    if (modality == "text") return project_text(features);
    return features.clone();
}

// ============================================================================
// H23: CrossAttentionFusion — fused cross-attention between two modalities
// ============================================================================
CrossAttentionFusion::CrossAttentionFusion(int64_t hidden)
    : q_proj(hidden, hidden), k_proj(hidden, hidden),
      v_proj(hidden, hidden), out_proj(hidden, hidden),
      hidden_size(hidden), scale(1.0f / std::sqrt((float)hidden)) {}

Tensor CrossAttentionFusion::forward(const Tensor& query, const Tensor& key_value,
                                       const Tensor& mask) const {
    int64_t B = query.dim(0);
    int64_t Q = query.dim(1);
    int64_t KV = key_value.dim(1);
    int64_t D = hidden_size;

    int64_t BQ = B * Q;
    int64_t BKV = B * KV;

    Tensor q = q_proj.forward(query.reshape({BQ, D}));
    Tensor k = k_proj.forward(key_value.reshape({BKV, D}));
    Tensor v = v_proj.forward(key_value.reshape({BKV, D}));

    Tensor kt({D, BKV});
    const float* kd = k.data<float>();
    float* ktd = kt.data<float>();
    for (int64_t i = 0; i < BKV; i++)
        for (int64_t j = 0; j < D; j++)
            ktd[j * BKV + i] = kd[i * D + j];

    Tensor attn_scores({BQ, BKV});
    math::gemm(1.0f, q, kt, 0.0f, attn_scores);
    float* ad = attn_scores.data<float>();
    for (int64_t i = 0; i < BQ * BKV; i++)
        ad[i] *= scale;

    if (mask.data() != nullptr && mask.numel() > 0) {
        const float* md = mask.data<float>();
        for (int64_t b = 0; b < B; b++)
            for (int64_t q_idx = 0; q_idx < Q; q_idx++)
                for (int64_t kv_idx = 0; kv_idx < KV; kv_idx++) {
                    float m = md[b * KV + kv_idx];
                    if (m < 0.5f)
                        ad[(b * Q + q_idx) * BKV + b * KV + kv_idx] = -1e10f;
                }
    }

    for (int64_t i = 0; i < BQ; i++) {
        float max_val = ad[i * BKV];
        for (int64_t j = 1; j < BKV; j++)
            if (ad[i * BKV + j] > max_val) max_val = ad[i * BKV + j];
        float sum_exp = 0;
        for (int64_t j = 0; j < BKV; j++) {
            ad[i * BKV + j] = std::exp(ad[i * BKV + j] - max_val);
            sum_exp += ad[i * BKV + j];
        }
        if (sum_exp > 1e-10f) {
            float inv_sum = 1.0f / sum_exp;
            for (int64_t j = 0; j < BKV; j++)
                ad[i * BKV + j] *= inv_sum;
        }
    }

    Tensor context({BQ, D});
    math::gemm(1.0f, attn_scores, v, 0.0f, context);
    Tensor output = out_proj.forward(context);
    return output.reshape(Shape{B, Q, D});
}

// ============================================================================
// H24: MultimodalFusion — layered cross-attention fusion across modalities
// ============================================================================
MultimodalFusion::MultimodalFusion(const FusionConfig& cfg)
    : proj(cfg.hidden_size), hidden_size(cfg.hidden_size) {
    for (int64_t i = 0; i < cfg.num_fusion_layers; i++) {
        vision_audio_layers.emplace_back(cfg.hidden_size);
        vision_text_layers.emplace_back(cfg.hidden_size);
        audio_text_layers.emplace_back(cfg.hidden_size);
    }
}

Tensor MultimodalFusion::fuse_vision_audio(const Tensor& vision, const Tensor& audio) const {
    int64_t B = vision.dim(0), D = hidden_size;
    Tensor v_proj = this->proj.project_vision(vision);
    Tensor a_proj = this->proj.project_audio(audio);
    // Linear flattens {B,S,D} to {B*S,D}; restore 3D for cross-attention layers.
    Tensor out_v = v_proj.reshape(Shape{B, v_proj.numel() / (B * D), D});
    Tensor out_a = a_proj.reshape(Shape{B, a_proj.numel() / (B * D), D});
    for (const auto& layer : vision_audio_layers) {
        Tensor v_attended = layer.forward(out_v, out_a);
        Tensor a_attended = layer.forward(out_a, out_v);
        const float* va = v_attended.data<float>();
        const float* aa = a_attended.data<float>();
        float* ov = out_v.data<float>();
        float* oa = out_a.data<float>();
        int64_t nv = out_v.numel();
        int64_t na = out_a.numel();
        for (int64_t i = 0; i < nv; i++) ov[i] = ov[i] * 0.5f + va[i] * 0.5f;
        for (int64_t i = 0; i < na; i++) oa[i] = oa[i] * 0.5f + aa[i] * 0.5f;
    }
    int64_t V = out_v.dim(1), A = out_a.dim(1);
    Tensor fused({B, V + A, D});
    float* fd = fused.data<float>();
    std::memcpy(fd, out_v.data<float>(), (size_t)(B * V * D) * sizeof(float));
    std::memcpy(fd + B * V * D, out_a.data<float>(), (size_t)(B * A * D) * sizeof(float));
    return fused;
}

Tensor MultimodalFusion::fuse_vision_text(const Tensor& vision, const Tensor& text) const {
    int64_t B = vision.dim(0), D = hidden_size;
    Tensor v_proj = this->proj.project_vision(vision);
    Tensor t_proj = this->proj.project_text(text);
    Tensor out_v = v_proj;
    Tensor out_t = t_proj;
    for (const auto& layer : vision_text_layers) {
        Tensor v_attended = layer.forward(out_v, out_t);
        Tensor t_attended = layer.forward(out_t, out_v);
        const float* va = v_attended.data<float>();
        const float* ta = t_attended.data<float>();
        float* ov = out_v.data<float>();
        float* ot = out_t.data<float>();
        int64_t nv = out_v.numel();
        int64_t nt = out_t.numel();
        for (int64_t i = 0; i < nv; i++) ov[i] = ov[i] * 0.5f + va[i] * 0.5f;
        for (int64_t i = 0; i < nt; i++) ot[i] = ot[i] * 0.5f + ta[i] * 0.5f;
    }
    int64_t V = out_v.dim(1), T = out_t.dim(1);
    Tensor fused({B, V + T, D});
    float* fd = fused.data<float>();
    std::memcpy(fd, out_v.data<float>(), (size_t)(B * V * D) * sizeof(float));
    std::memcpy(fd + B * V * D, out_t.data<float>(), (size_t)(B * T * D) * sizeof(float));
    return fused;
}

Tensor MultimodalFusion::fuse_audio_text(const Tensor& audio, const Tensor& text) const {
    int64_t B = audio.dim(0), D = hidden_size;
    Tensor a_proj = this->proj.project_audio(audio);
    Tensor t_proj = this->proj.project_text(text);
    Tensor out_a = a_proj;
    Tensor out_t = t_proj;
    for (const auto& layer : audio_text_layers) {
        Tensor a_attended = layer.forward(out_a, out_t);
        Tensor t_attended = layer.forward(out_t, out_a);
        const float* aa = a_attended.data<float>();
        const float* ta = t_attended.data<float>();
        float* oa = out_a.data<float>();
        float* ot = out_t.data<float>();
        int64_t na = out_a.numel();
        int64_t nt = out_t.numel();
        for (int64_t i = 0; i < na; i++) oa[i] = oa[i] * 0.5f + aa[i] * 0.5f;
        for (int64_t i = 0; i < nt; i++) ot[i] = ot[i] * 0.5f + ta[i] * 0.5f;
    }
    int64_t A = out_a.dim(1), T = out_t.dim(1);
    Tensor fused({B, A + T, D});
    float* fd = fused.data<float>();
    std::memcpy(fd, out_a.data<float>(), (size_t)(B * A * D) * sizeof(float));
    std::memcpy(fd + B * A * D, out_t.data<float>(), (size_t)(B * T * D) * sizeof(float));
    return fused;
}

Tensor MultimodalFusion::fuse_all(const Tensor& vision, const Tensor& audio,
                                    const Tensor& text) const {
    int64_t B = vision.dim(0), D = hidden_size;
    Tensor v_proj = this->proj.project_vision(vision);
    Tensor a_proj = this->proj.project_audio(audio);
    Tensor t_proj = this->proj.project_text(text);
    Tensor out_v = v_proj;
    Tensor out_a = a_proj;
    Tensor out_t = t_proj;
    for (size_t l = 0; l < vision_audio_layers.size(); l++) {
        Tensor v_a = vision_audio_layers[l].forward(out_v, out_a);
        Tensor a_v = vision_audio_layers[l].forward(out_a, out_v);
        Tensor v_t = vision_text_layers[l].forward(out_v, out_t);
        Tensor t_v = vision_text_layers[l].forward(out_t, out_v);
        Tensor a_t = audio_text_layers[l].forward(out_a, out_t);
        Tensor t_a = audio_text_layers[l].forward(out_t, out_a);
        const float* va = v_a.data<float>();
        const float* av = a_v.data<float>();
        const float* vt = v_t.data<float>();
        const float* tv = t_v.data<float>();
        const float* at = a_t.data<float>();
        const float* ta = t_a.data<float>();
        float* ov = out_v.data<float>();
        float* oa = out_a.data<float>();
        float* ot = out_t.data<float>();
        int64_t nv = out_v.numel();
        int64_t na = out_a.numel();
        int64_t nt = out_t.numel();
        float inv3 = 1.0f / 3.0f;
        for (int64_t i = 0; i < nv; i++)
            ov[i] = ov[i] * inv3 + (va[i] + vt[i]) * inv3;
        for (int64_t i = 0; i < na; i++)
            oa[i] = oa[i] * inv3 + (av[i] + at[i]) * inv3;
        for (int64_t i = 0; i < nt; i++)
            ot[i] = ot[i] * inv3 + (tv[i] + ta[i]) * inv3;
    }
    int64_t V = out_v.dim(1), A = out_a.dim(1), T = out_t.dim(1);
    Tensor fused({B, V + A + T, D});
    float* fd = fused.data<float>();
    std::memcpy(fd, out_v.data<float>(), (size_t)(B * V * D) * sizeof(float));
    std::memcpy(fd + B * V * D, out_a.data<float>(), (size_t)(B * A * D) * sizeof(float));
    std::memcpy(fd + B * (V + A) * D, out_t.data<float>(), (size_t)(B * T * D) * sizeof(float));
    return fused;
}

Tensor MultimodalFusion::fuse_all_multistream(const Tensor& vision, const Tensor& audio,
                                                const Tensor& text) const {
    int64_t B = vision.dim(0), D = hidden_size;
    Tensor v_proj = this->proj.project_vision(vision);
    Tensor a_proj = this->proj.project_audio(audio);
    Tensor t_proj = this->proj.project_text(text);
    Tensor out_v = v_proj;
    Tensor out_a = a_proj;
    Tensor out_t = t_proj;
    for (size_t l = 0; l < vision_audio_layers.size(); l++) {
        Tensor new_v(out_v.shape());
        Tensor new_a(out_a.shape());
        Tensor new_t(out_t.shape());
        new_v.zero_();
        new_a.zero_();
        new_t.zero_();
        const auto& va_layer = vision_audio_layers[l];
        const auto& vt_layer = vision_text_layers[l];
        const auto& at_layer = audio_text_layers[l];
        Tensor v_from_a = va_layer.forward(out_v, out_a);
        Tensor v_from_t = vt_layer.forward(out_v, out_t);
        Tensor a_from_v = va_layer.forward(out_a, out_v);
        Tensor a_from_t = at_layer.forward(out_a, out_t);
        Tensor t_from_v = vt_layer.forward(out_t, out_v);
        Tensor t_from_a = at_layer.forward(out_t, out_a);
        float* nvd = new_v.data<float>();
        float* nad = new_a.data<float>();
        float* ntd = new_t.data<float>();
        const float* ovd = out_v.data<float>();
        const float* oad = out_a.data<float>();
        const float* otd = out_t.data<float>();
        const float* vfd = v_from_a.data<float>();
        const float* vtd = v_from_t.data<float>();
        const float* afd = a_from_v.data<float>();
        const float* atd = a_from_t.data<float>();
        const float* tfd = t_from_v.data<float>();
        const float* tad = t_from_a.data<float>();
        int64_t nv = out_v.numel(), na = out_a.numel(), nt = out_t.numel();
        float inv3 = 1.0f / 3.0f;
        for (int64_t i = 0; i < nv; i++)
            nvd[i] = ovd[i] * 0.5f + (vfd[i] + vtd[i]) * 0.25f;
        for (int64_t i = 0; i < na; i++)
            nad[i] = oad[i] * 0.5f + (afd[i] + atd[i]) * 0.25f;
        for (int64_t i = 0; i < nt; i++)
            ntd[i] = otd[i] * 0.5f + (tfd[i] + tad[i]) * 0.25f;
        out_v = new_v;
        out_a = new_a;
        out_t = new_t;
    }
    int64_t V = out_v.dim(1), A = out_a.dim(1), T = out_t.dim(1);
    Tensor fused({B, V + A + T, D});
    float* fd = fused.data<float>();
    std::memcpy(fd, out_v.data<float>(), (size_t)(B * V * D) * sizeof(float));
    std::memcpy(fd + B * V * D, out_a.data<float>(), (size_t)(B * A * D) * sizeof(float));
    std::memcpy(fd + B * (V + A) * D, out_t.data<float>(), (size_t)(B * T * D) * sizeof(float));
    return fused;
}

Tensor MultimodalFusion::compute_fusion_weights(const Tensor& vision, const Tensor& audio,
                                                  const Tensor& text) const {
    int64_t B = vision.dim(0), D = hidden_size;
    Tensor v_proj = this->proj.project_vision(vision);
    Tensor a_proj = this->proj.project_audio(audio);
    Tensor t_proj = this->proj.project_text(text);
    int64_t V = v_proj.dim(1), A = a_proj.dim(1), T = t_proj.dim(1);
    Tensor weights({B, V + A + T, 3});
    weights.fill(0.5f);
    float* wd = weights.data<float>();
    for (int64_t b = 0; b < B; b++) {
        int64_t offset = 0;
        for (int64_t v = 0; v < V; v++, offset++)
            wd[(b * (V + A + T) + offset) * 3 + 0] = 1.0f;
        for (int64_t a = 0; a < A; a++, offset++)
            wd[(b * (V + A + T) + offset) * 3 + 1] = 1.0f;
        for (int64_t t = 0; t < T; t++, offset++)
            wd[(b * (V + A + T) + offset) * 3 + 2] = 1.0f;
    }
    return weights;
}

size_t MultimodalFusion::param_count() const {
    size_t count = 0;
    count += (size_t)(proj.vision_proj.param_count() + proj.audio_proj.param_count() + proj.text_proj.param_count());
    for (const auto& l : vision_audio_layers)
        count += (size_t)(l.q_proj.param_count() + l.k_proj.param_count() + l.v_proj.param_count() + l.out_proj.param_count());
    for (const auto& l : vision_text_layers)
        count += (size_t)(l.q_proj.param_count() + l.k_proj.param_count() + l.v_proj.param_count() + l.out_proj.param_count());
    for (const auto& l : audio_text_layers)
        count += (size_t)(l.q_proj.param_count() + l.k_proj.param_count() + l.v_proj.param_count() + l.out_proj.param_count());
    return count;
}

Tensor MultimodalFusion::fuse_weighted(const Tensor& vision, const Tensor& audio,
                                         const Tensor& text, float vw, float aw, float tw) const {
    int64_t B = vision.dim(0), D = hidden_size;
    Tensor v_proj = this->proj.project_vision(vision);
    Tensor a_proj = this->proj.project_audio(audio);
    Tensor t_proj = this->proj.project_text(text);
    float total = vw + aw + tw;
    if (total < 0.001f) { vw = 1.0f; aw = 1.0f; tw = 1.0f; total = 3.0f; }
    float inv_total = 1.0f / total;
    int64_t V = v_proj.dim(1), A = a_proj.dim(1), T = t_proj.dim(1);
    Tensor fused({B, V + A + T, D});
    float* fd = fused.data<float>();
    for (int64_t b = 0; b < B; b++) {
        int64_t offset = 0;
        for (int64_t v = 0; v < V; v++, offset++)
            for (int64_t d = 0; d < D; d++)
                fd[(b * (V + A + T) + offset) * D + d] = v_proj.data<float>()[(b * V + v) * D + d] * vw * inv_total;
        for (int64_t a = 0; a < A; a++, offset++)
            for (int64_t d = 0; d < D; d++)
                fd[(b * (V + A + T) + offset) * D + d] = a_proj.data<float>()[(b * A + a) * D + d] * aw * inv_total;
        for (int64_t t = 0; t < T; t++, offset++)
            for (int64_t d = 0; d < D; d++)
                fd[(b * (V + A + T) + offset) * D + d] = t_proj.data<float>()[(b * T + t) * D + d] * tw * inv_total;
    }
    return fused;
}

} // namespace quant
