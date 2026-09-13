#include "quant/multimodal_cross_attn.h"
#include "quant/math.h"
#include "quant/tokenizer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <sstream>

namespace quant {

static float dot_product(const float* a, const float* b, int64_t n) {
    float sum = 0;
    for (int64_t i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static void softmax_inplace(float* data, int64_t n) {
    float mx = data[0];
    for (int64_t i = 1; i < n; i++)
        if (data[i] > mx) mx = data[i];
    float s = 0;
    for (int64_t i = 0; i < n; i++) { data[i] = std::exp(data[i] - mx); s += data[i]; }
    float inv = 1.0f / (s + 1e-10f);
    for (int64_t i = 0; i < n; i++) data[i] *= inv;
}

static void xavier_init(Tensor& w, int64_t fan_in, int64_t fan_out, std::mt19937& rng) {
    float std_dev = std::sqrt(2.0f / (float)(fan_in + fan_out));
    std::normal_distribution<float> dist(0, std_dev);
    float* d = w.data<float>();
    for (int64_t i = 0; i < w.numel(); i++)
        d[i] = dist(rng);
}

PatchEmbedding::PatchEmbedding(int64_t image_size, int64_t patch_size,
                                 int64_t in_channels, int64_t hidden_dim)
    : hidden_dim(hidden_dim) {
    num_patches = (image_size / patch_size) * (image_size / patch_size);
    patch_proj_weight = Tensor({(int64_t)(patch_size * patch_size * in_channels), hidden_dim});
    pos_embed = Tensor({num_patches + 1, hidden_dim});
    std::mt19937 rng(42);
    xavier_init(patch_proj_weight, patch_proj_weight.dim(0), patch_proj_weight.dim(1), rng);
    xavier_init(pos_embed, pos_embed.dim(0), pos_embed.dim(1), rng);
}

Tensor PatchEmbedding::forward(const Tensor& images) {
    int64_t B = images.dim(0);
    int64_t C = images.dim(1);
    int64_t H = images.dim(2);
    int64_t W = images.dim(3);
    int64_t ps = (int64_t)std::sqrt((float)patch_proj_weight.dim(0) / (float)C);
    int64_t n_h = H / ps;
    int64_t n_w = W / ps;
    int64_t np = n_h * n_w;

    Tensor patches({B, np, patch_proj_weight.dim(1)});
    patches.zero_();
    const float* id = images.data<float>();
    const float* pw = patch_proj_weight.data<float>();
    float* pd = patches.data<float>();

    for (int64_t b = 0; b < B; b++) {
        for (int64_t ph = 0; ph < n_h; ph++) {
            for (int64_t pw_i = 0; pw_i < n_w; pw_i++) {
                int64_t patch_idx = ph * n_w + pw_i;
                for (int64_t d = 0; d < hidden_dim; d++) {
                    float sum = 0;
                    for (int64_t c = 0; c < C; c++) {
                        for (int64_t pi = 0; pi < ps; pi++) {
                            for (int64_t pj = 0; pj < ps; pj++) {
                                int64_t ih = ph * ps + pi;
                                int64_t iw = pw_i * ps + pj;
                                if (ih < H && iw < W) {
                                    float pixel = id[b * C * H * W + c * H * W + ih * W + iw];
                                    int64_t w_idx = c * ps * ps + pi * ps + pj;
                                    sum += pixel * pw[w_idx * hidden_dim + d];
                                }
                            }
                        }
                    }
                    pd[b * np * hidden_dim + patch_idx * hidden_dim + d] = sum;
                }
            }
        }
    }

    Tensor cls_token = Tensor({1, 1, hidden_dim});
    cls_token.zero_();

    Tensor combined({B, np + 1, hidden_dim});
    float* cd = combined.data<float>();
    const float* pd2 = patches.data<float>();
    for (int64_t b = 0; b < B; b++) {
        for (int64_t d = 0; d < hidden_dim; d++)
            cd[b * (np + 1) * hidden_dim + d] = 0;
        std::memcpy(cd + b * (np + 1) * hidden_dim + hidden_dim,
                    pd2 + b * np * hidden_dim,
                    (size_t)(np * hidden_dim) * sizeof(float));
    }

    for (int64_t b = 0; b < B; b++) {
        for (int64_t p = 0; p < np + 1 && p < pos_embed.dim(0); p++) {
            const float* pe = pos_embed.data<float>() + p * hidden_dim;
            float* target = cd + b * (np + 1) * hidden_dim + p * hidden_dim;
            for (int64_t d = 0; d < hidden_dim; d++)
                target[d] += pe[d];
        }
    }
    return combined;
}

struct VisionTransformer::Impl {
    std::mt19937 rng{42};
};

VisionTransformer::VisionTransformer(const ViTConfig& cfg)
    : config(cfg),
      patch_embed(cfg.image_size, cfg.patch_size, cfg.in_channels, cfg.hidden_dim),
      final_norm(cfg.hidden_dim) {
    impl_ = new Impl();

    for (int64_t i = 0; i < cfg.num_layers; i++) {
        attn_qkv.emplace_back(cfg.hidden_dim, cfg.hidden_dim * 3);
        attn_out.emplace_back(cfg.hidden_dim, cfg.hidden_dim);
        mlp_fc1.emplace_back(cfg.hidden_dim, cfg.mlp_dim);
        mlp_fc2.emplace_back(cfg.mlp_dim, cfg.hidden_dim);
        norm1.emplace_back(cfg.hidden_dim);
        norm2.emplace_back(cfg.hidden_dim);

        xavier_init(attn_qkv.back().weight, cfg.hidden_dim, cfg.hidden_dim * 3, impl_->rng);
        xavier_init(attn_out.back().weight, cfg.hidden_dim, cfg.hidden_dim, impl_->rng);
        xavier_init(mlp_fc1.back().weight, cfg.hidden_dim, cfg.mlp_dim, impl_->rng);
        xavier_init(mlp_fc2.back().weight, cfg.mlp_dim, cfg.hidden_dim, impl_->rng);
    }
}

VisionTransformer::~VisionTransformer() { delete impl_; }

Tensor VisionTransformer::encode(const Tensor& images) { return forward(images); }

std::vector<Tensor> VisionTransformer::get_patch_features(const Tensor& images) {
    int64_t B = images.dim(0);
    Tensor x = patch_embed.forward(images);
    std::vector<Tensor> per_patch;
    int64_t np1 = x.dim(1);
    int64_t D = x.dim(2);

    for (int64_t b = 0; b < B; b++) {
        Tensor single({np1, D});
        std::memcpy(single.data<float>(),
                    x.data<float>() + b * np1 * D,
                    (size_t)(np1 * D) * sizeof(float));
        per_patch.push_back(std::move(single));
    }
    return per_patch;
}

Tensor VisionTransformer::forward(const Tensor& images) {
    int64_t B = images.dim(0);
    Tensor x = patch_embed.forward(images);
    int64_t S = x.dim(1);
    int64_t D = x.dim(2);

    for (size_t l = 0; l < norm1.size(); l++) {
        Tensor normed_x({S, D});
        for (int64_t b = 0; b < B; b++) {
            Tensor xb({S, D});
            const float* xd = x.data<float>() + b * S * D;
            float* xbd = xb.data<float>();
            std::memcpy(xbd, xd, (size_t)(S * D) * sizeof(float));

            Tensor normed = norm1[l].forward(xb);

            int64_t H = config.num_heads;
            int64_t hd = D / H;
            Tensor qkv = attn_qkv[l].forward(normed);
            const float* qkvd = qkv.data<float>();

            Tensor attn_out_t({S, D});
            attn_out_t.zero_();
            float* aod = attn_out_t.data<float>();

            for (int64_t h = 0; h < H; h++) {
                for (int64_t q = 0; q < S; q++) {
                    float scores[512];
                    int64_t seq = std::min(S, (int64_t)512);
                    for (int64_t k = 0; k < seq; k++) {
                        float s = 0;
                        for (int64_t d = 0; d < hd; d++)
                            s += qkvd[q * 3 * D + h * hd + d] *
                                 qkvd[k * 3 * D + D + h * hd + d];
                        scores[k] = s / std::sqrt((float)hd);
                    }
                    softmax_inplace(scores, seq);
                    for (int64_t k = 0; k < seq; k++) {
                        for (int64_t d = 0; d < hd; d++)
                            aod[q * D + h * hd + d] += scores[k] *
                                qkvd[k * 3 * D + 2 * D + h * hd + d];
                    }
                }
            }
            Tensor proj_out = attn_out[l].forward(attn_out_t);

            float* xbd2 = xb.data<float>();
            const float* pod = proj_out.data<float>();
            for (int64_t i = 0; i < S * D; i++)
                xbd2[i] += pod[i];

            Tensor normed2 = norm2[l].forward(xb);
            Tensor ffn1 = mlp_fc1[l].forward(normed2);
            float* f1d = ffn1.data<float>();
            for (int64_t i = 0; i < ffn1.numel(); i++)
                f1d[i] = f1d[i] * (1.0f / (1.0f + std::exp(-f1d[i])));
            Tensor ffn2 = mlp_fc2[l].forward(ffn1);

            float* xbd3 = xb.data<float>();
            const float* f2d = ffn2.data<float>();
            for (int64_t i = 0; i < S * D; i++)
                xbd3[i] += f2d[i];

            for (int64_t i = 0; i < S * D; i++)
                normed_x.data<float>()[i] = xbd3[i];
        }
        x = normed_x;
    }

    Tensor cls_out({B, D});
    float* cod = cls_out.data<float>();
    const float* xd = x.data<float>();
    for (int64_t b = 0; b < B; b++)
        std::memcpy(cod + b * D, xd + b * x.dim(1) * D, D * sizeof(float));

    return final_norm.forward(cls_out);
}

Conv1DExtractor::Conv1DExtractor(int64_t in_dim, int64_t out_dim, int64_t kernel_size, int64_t stride)
    : stride_(stride) {
    weight = Tensor({out_dim, in_dim, kernel_size});
    bias = Tensor({out_dim});
    bias.zero_();
    std::mt19937 rng(42);
    xavier_init(weight, in_dim * kernel_size, out_dim, rng);
}

Tensor Conv1DExtractor::forward(const Tensor& input) {
    int64_t T = input.dim(0);
    int64_t D = input.dim(1);
    int64_t F = weight.dim(0);
    int64_t K = weight.dim(2);
    int64_t out_T = std::max((int64_t)1, (T - K) / stride_ + 1);

    Tensor output({out_T, F});
    output.zero_();
    float* od = output.data<float>();
    const float* id = input.data<float>();
    const float* wd = weight.data<float>();
    const float* bd = bias.data<float>();

    for (int64_t t = 0; t < out_T; t++) {
        for (int64_t f = 0; f < F; f++) {
            float sum = bd[f];
            for (int64_t k = 0; k < K; k++) {
                int64_t in_t = t * stride_ + k;
                if (in_t < T) {
                    for (int64_t d = 0; d < D; d++)
                        sum += id[in_t * D + d] * wd[f * D * K + d * K + k];
                }
            }
            od[t * F + f] = sum;
        }
    }
    return output;
}

struct AudioTransformer::Impl {};

AudioTransformer::AudioTransformer(const AudioEncoderConfig& cfg)
    : config(cfg),
      conv1(cfg.input_dim, cfg.hidden_dim, cfg.kernel_size, 2),
      conv2(cfg.hidden_dim, cfg.hidden_dim, cfg.kernel_size, 2),
      final_norm(cfg.hidden_dim) {
    for (int64_t i = 0; i < cfg.num_layers; i++) {
        attn_qkv.emplace_back(cfg.hidden_dim, cfg.hidden_dim * 3);
        attn_out.emplace_back(cfg.hidden_dim, cfg.hidden_dim);
        ffn_fc1.emplace_back(cfg.hidden_dim, cfg.hidden_dim * 4);
        ffn_fc2.emplace_back(cfg.hidden_dim * 4, cfg.hidden_dim);
        norm1.emplace_back(cfg.hidden_dim);
        norm2.emplace_back(cfg.hidden_dim);
    }
    impl_ = new Impl();
}

AudioTransformer::~AudioTransformer() { delete impl_; }

Tensor AudioTransformer::encode(const Tensor& spectrograms) { return forward(spectrograms); }

Tensor AudioTransformer::forward(const Tensor& spectrograms) {
    int64_t T = spectrograms.dim(0);
    int64_t D = spectrograms.dim(1);

    Tensor x = conv1.forward(spectrograms);
    for (int64_t i = 0; i < x.numel(); i++) {
        float* d = x.data<float>();
        d[i] = d[i] * (1.0f / (1.0f + std::exp(-d[i])));
    }
    x = conv2.forward(x);
    for (int64_t i = 0; i < x.numel(); i++) {
        float* d = x.data<float>();
        d[i] = d[i] * (1.0f / (1.0f + std::exp(-d[i])));
    }

    int64_t S = x.dim(0);
    int64_t H = config.num_heads;
    int64_t hd = config.hidden_dim / H;

    for (size_t l = 0; l < norm1.size(); l++) {
        Tensor normed = norm1[l].forward(x);
        Tensor qkv_t = attn_qkv[l].forward(normed);
        const float* qkvd = qkv_t.data<float>();

        Tensor attn_out_t({S, config.hidden_dim});
        attn_out_t.zero_();
        float* aod = attn_out_t.data<float>();

        for (int64_t h = 0; h < H; h++) {
            for (int64_t q = 0; q < S; q++) {
                std::vector<float> scores((size_t)S);
                for (int64_t k = 0; k < S; k++) {
                    float s = 0;
                    for (int64_t d = 0; d < hd; d++)
                        s += qkvd[q * 3 * config.hidden_dim + h * hd + d] *
                             qkvd[k * 3 * config.hidden_dim + config.hidden_dim + h * hd + d];
                    scores[(size_t)k] = s / std::sqrt((float)hd);
                }
                softmax_inplace(scores.data(), S);
                for (int64_t k = 0; k < S; k++) {
                    for (int64_t d = 0; d < hd; d++)
                        aod[q * config.hidden_dim + h * hd + d] += scores[(size_t)k] *
                            qkvd[k * 3 * config.hidden_dim + 2 * config.hidden_dim + h * hd + d];
                }
            }
        }
        Tensor proj = attn_out[l].forward(attn_out_t);
        float* xd = x.data<float>();
        const float* pd = proj.data<float>();
        for (int64_t i = 0; i < S * config.hidden_dim; i++)
            xd[i] += pd[i];

        Tensor normed2 = norm2[l].forward(x);
        Tensor fc1 = ffn_fc1[l].forward(normed2);
        float* f1d = fc1.data<float>();
        for (int64_t i = 0; i < fc1.numel(); i++)
            f1d[i] = f1d[i] * (1.0f / (1.0f + std::exp(-f1d[i])));
        Tensor fc2 = ffn_fc2[l].forward(fc1);
        float* xd2 = x.data<float>();
        const float* f2d = fc2.data<float>();
        for (int64_t i = 0; i < S * config.hidden_dim; i++)
            xd2[i] += f2d[i];
    }
    return final_norm.forward(x);
}

struct CrossAttentionBlock::Impl {};

CrossAttentionBlock::CrossAttentionBlock(const CrossAttentionLayerConfig& cfg)
    : config(cfg),
      q_proj(cfg.hidden_size, cfg.hidden_size),
      k_proj(cfg.hidden_size, cfg.hidden_size),
      v_proj(cfg.hidden_size, cfg.hidden_size),
      out_proj(cfg.hidden_size, cfg.hidden_size),
      ffn_fc1(cfg.hidden_size, cfg.hidden_size * 4),
      ffn_fc2(cfg.hidden_size * 4, cfg.hidden_size),
      norm1(cfg.hidden_size),
      norm2(cfg.hidden_size),
      cross_norm(cfg.hidden_size) {
    impl_ = new Impl();
    std::mt19937 rng(42);
    xavier_init(q_proj.weight, cfg.hidden_size, cfg.hidden_size, rng);
    xavier_init(k_proj.weight, cfg.hidden_size, cfg.hidden_size, rng);
    xavier_init(v_proj.weight, cfg.hidden_size, cfg.hidden_size, rng);
    xavier_init(out_proj.weight, cfg.hidden_size, cfg.hidden_size, rng);
    xavier_init(ffn_fc1.weight, cfg.hidden_size, cfg.hidden_size * 4, rng);
    xavier_init(ffn_fc2.weight, cfg.hidden_size * 4, cfg.hidden_size, rng);
}

CrossAttentionBlock::~CrossAttentionBlock() { delete impl_; }

Tensor CrossAttentionBlock::forward_self_attn(const Tensor& x) {
    // Real self-attention pass: attend x against itself. Self-attention is
    // just cross-attention with query == key == value == x, so reuse the
    // block's full cross-attention forward (q/k/v projections, scaled
    // dot-product attention, output projection, residual + norms + FFN).
    return forward(x, x, "");
}

Tensor CrossAttentionBlock::forward(const Tensor& query, const Tensor& key_value,
                                     const std::string& modality_tag) {
    (void)modality_tag;
    int64_t Tq = query.dim(0);
    int64_t D = config.hidden_size;
    int64_t Tk = key_value.dim(0);

    Tensor q = q_proj.forward(query);
    Tensor k = k_proj.forward(key_value);
    Tensor v = v_proj.forward(key_value);

    int64_t H = config.num_heads;
    int64_t hd = D / H;
    float scale = 1.0f / std::sqrt((float)hd);

    Tensor output({Tq, D});
    output.zero_();
    float* od = output.data<float>();
    const float* qd = q.data<float>();
    const float* kd = k.data<float>();
    const float* vd = v.data<float>();

    for (int64_t h = 0; h < H; h++) {
        for (int64_t tq = 0; tq < Tq; tq++) {
            std::vector<float> scores((size_t)Tk);
            for (int64_t tk = 0; tk < Tk; tk++) {
                float s = 0;
                for (int64_t d = 0; d < hd; d++)
                    s += qd[tq * D + h * hd + d] * kd[tk * D + h * hd + d];
                scores[(size_t)tk] = s * scale;
            }
            softmax_inplace(scores.data(), Tk);
            for (int64_t tk = 0; tk < Tk; tk++) {
                for (int64_t d = 0; d < hd; d++)
                    od[tq * D + h * hd + d] += scores[(size_t)tk] * vd[tk * D + h * hd + d];
            }
        }
    }

    Tensor proj_out = out_proj.forward(output);

    float* od2 = new float[Tq * D]();
    const float* pd = proj_out.data<float>();
    const float* qd2 = query.data<float>();
    for (int64_t i = 0; i < Tq * D; i++)
        od2[i] = qd2[i] + pd[i];

    Tensor residual({Tq, D});
    std::memcpy(residual.data<float>(), od2, (size_t)(Tq * D) * sizeof(float));
    delete[] od2;

    Tensor normed = norm1.forward(residual);

    Tensor ff1 = ffn_fc1.forward(normed);
    float* f1d = ff1.data<float>();
    for (int64_t i = 0; i < ff1.numel(); i++)
        f1d[i] = f1d[i] * (1.0f / (1.0f + std::exp(-f1d[i])));
    Tensor ff2 = ffn_fc2.forward(ff1);

    float* rd = residual.data<float>();
    const float* f2d = ff2.data<float>();
    for (int64_t i = 0; i < Tq * D; i++)
        rd[i] += f2d[i];

    return norm2.forward(residual);
}

struct ModalityFusionEncoder::Impl {};

ModalityFusionEncoder::ModalityFusionEncoder(const MultimodalEncoderConfig& cfg)
    : config(cfg),
      text_proj(cfg.text_embed_dim, cfg.shared_hidden),
      image_proj(cfg.image_embed_dim, cfg.shared_hidden),
      audio_proj(cfg.audio_embed_dim, cfg.shared_hidden),
      output_proj(cfg.shared_hidden, cfg.shared_hidden),
      output_norm(cfg.shared_hidden) {
    impl_ = new Impl();

    CrossAttentionLayerConfig xcfg;
    xcfg.hidden_size = cfg.shared_hidden;
    xcfg.num_heads = cfg.shared_num_heads;

    for (int64_t i = 0; i < cfg.shared_num_layers; i++) {
        text_cross_attn_layers.emplace_back(xcfg);
        image_cross_attn_layers.emplace_back(xcfg);
        audio_cross_attn_layers.emplace_back(xcfg);
        shared_fusion_layers.emplace_back(xcfg);
    }
}

ModalityFusionEncoder::~ModalityFusionEncoder() { delete impl_; }

void ModalityFusionEncoder::apply_modality_dropout(Tensor& text_emb, Tensor& image_emb,
                                                     Tensor& audio_emb,
                                                     float drop_prob, bool training) {
    if (!training || drop_prob <= 0) return;

    std::mt19937 rng((unsigned)std::random_device{}());
    std::uniform_real_distribution<float> dist(0, 1);

    float r_text = dist(rng);
    float r_image = dist(rng);
    float r_audio = dist(rng);

    if (r_text < drop_prob && text_emb.numel() > 0) text_emb.zero_();
    if (r_image < drop_prob && image_emb.numel() > 0) image_emb.zero_();
    if (r_audio < drop_prob && audio_emb.numel() > 0) audio_emb.zero_();

    int active = 0;
    if (text_emb.numel() > 0) active++;
    if (image_emb.numel() > 0) active++;
    if (audio_emb.numel() > 0) active++;

    if (active == 0) {
        if (text_emb.numel() > 0) text_emb.zero_();
    }
}

std::vector<Tensor> ModalityFusionEncoder::get_per_modality_features(
    const Tensor& text_emb, const Tensor& image_emb, const Tensor& audio_emb) {
    std::vector<Tensor> features;
    features.push_back(text_emb);
    features.push_back(image_emb);
    features.push_back(audio_emb);
    return features;
}

Tensor ModalityFusionEncoder::fuse_early(const Tensor& text_emb, const Tensor& image_emb,
                                           const Tensor& audio_emb) {
    int64_t D = config.shared_hidden;
    int64_t total = 0;
    if (text_emb.numel() > 0) total += text_emb.dim(0);
    if (image_emb.numel() > 0) total += image_emb.dim(0);
    if (audio_emb.numel() > 0) total += audio_emb.dim(0);

    if (total == 0) return Tensor({0, D});

    Tensor combined({total, D});
    combined.zero_();
    float* cd = combined.data<float>();
    int64_t offset = 0;

    if (text_emb.numel() > 0) {
        int64_t n = text_emb.dim(0);
        int64_t dd = text_emb.numel() / std::max((int64_t)1, n);
        Tensor proj_in({n, dd});
        std::memcpy(proj_in.data<float>(), text_emb.data<float>(), (size_t)(n * dd) * sizeof(float));
        Tensor proj_out = text_proj.forward(proj_in);
        const float* pod = proj_out.data<float>();
        int64_t pd = std::min(D, proj_out.numel() / std::max((int64_t)1, n));
        for (int64_t i = 0; i < n; i++)
            std::memcpy(cd + (offset + i) * D, pod + i * pd, (size_t)pd * sizeof(float));
        offset += n;
    }

    if (image_emb.numel() > 0) {
        int64_t n = image_emb.dim(0);
        int64_t dd = image_emb.numel() / std::max((int64_t)1, n);
        Tensor proj_in({n, dd});
        std::memcpy(proj_in.data<float>(), image_emb.data<float>(), (size_t)(n * dd) * sizeof(float));
        Tensor proj_out = image_proj.forward(proj_in);
        const float* pod = proj_out.data<float>();
        int64_t pd = std::min(D, proj_out.numel() / std::max((int64_t)1, n));
        for (int64_t i = 0; i < n; i++)
            std::memcpy(cd + (offset + i) * D, pod + i * pd, (size_t)pd * sizeof(float));
        offset += n;
    }

    if (audio_emb.numel() > 0) {
        int64_t n = audio_emb.dim(0);
        int64_t dd = audio_emb.numel() / std::max((int64_t)1, n);
        Tensor proj_in({n, dd});
        std::memcpy(proj_in.data<float>(), audio_emb.data<float>(), (size_t)(n * dd) * sizeof(float));
        Tensor proj_out = audio_proj.forward(proj_in);
        const float* pod = proj_out.data<float>();
        int64_t pd = std::min(D, proj_out.numel() / std::max((int64_t)1, n));
        for (int64_t i = 0; i < n; i++)
            std::memcpy(cd + (offset + i) * D, pod + i * pd, (size_t)pd * sizeof(float));
        offset += n;
    }

    return combined;
}

Tensor ModalityFusionEncoder::fuse_late(const Tensor& text_emb, const Tensor& image_emb,
                                          const Tensor& audio_emb) {
    int64_t D = config.shared_hidden;

    Tensor text_p = text_emb.numel() > 0 ? text_proj.forward(text_emb) : Tensor({0, D});
    Tensor image_p = image_emb.numel() > 0 ? image_proj.forward(image_emb) : Tensor({0, D});
    Tensor audio_p = audio_emb.numel() > 0 ? audio_proj.forward(audio_emb) : Tensor({0, D});

    int64_t n_text = text_p.numel() > 0 ? text_p.dim(0) : 0;
    int64_t n_image = image_p.numel() > 0 ? image_p.dim(0) : 0;
    int64_t n_audio = audio_p.numel() > 0 ? audio_p.dim(0) : 0;

    if (n_text == 0 && n_image == 0 && n_audio == 0)
        return Tensor({0, D});

    Tensor pooled({3, D});
    pooled.zero_();
    float* pd = pooled.data<float>();

    for (int64_t d = 0; d < D; d++) {
        float sum = 0;
        int count = 0;
        if (n_text > 0) { sum += pd[d]; count++; }
        if (n_image > 0) { sum += pd[d]; count++; }
        if (n_audio > 0) { sum += pd[d]; count++; }
        (void)count;
    }

    Tensor combined({1, D});
    combined.zero_();
    return combined;
}

Tensor ModalityFusionEncoder::fuse(const Tensor& text_emb, const Tensor& image_emb,
                                    const Tensor& audio_emb) {
    int64_t D = config.shared_hidden;

    Tensor text_p = text_emb.numel() > 0 ? text_proj.forward(text_emb) : Tensor({0, D});
    Tensor image_p = image_emb.numel() > 0 ? image_proj.forward(image_emb) : Tensor({0, D});
    Tensor audio_p = audio_emb.numel() > 0 ? audio_proj.forward(audio_emb) : Tensor({0, D});

    int64_t n_text = text_p.numel() > 0 ? text_p.dim(0) : 0;
    int64_t n_image = image_p.numel() > 0 ? image_p.dim(0) : 0;
    int64_t n_audio = audio_p.numel() > 0 ? audio_p.dim(0) : 0;
    int64_t total = n_text + n_image + n_audio;

    if (total == 0) return Tensor({0, D});

    Tensor combined({total, D});
    combined.zero_();
    float* cd = combined.data<float>();
    int64_t offset = 0;

    if (n_text > 0) {
        std::memcpy(cd + offset * D, text_p.data<float>(), (size_t)(n_text * D) * sizeof(float));
        offset += n_text;
    }
    if (n_image > 0) {
        std::memcpy(cd + offset * D, image_p.data<float>(), (size_t)(n_image * D) * sizeof(float));
        offset += n_image;
    }
    if (n_audio > 0) {
        std::memcpy(cd + offset * D, audio_p.data<float>(), (size_t)(n_audio * D) * sizeof(float));
        offset += n_audio;
    }

    for (size_t l = 0; l < shared_fusion_layers.size(); l++) {
        if (n_text > 0 && (n_image > 0 || n_audio > 0)) {
            Tensor kv;
            if (n_image > 0 && n_audio > 0) {
                kv = Tensor({n_image + n_audio, D});
                float* kvd = kv.data<float>();
                std::memcpy(kvd, cd + n_text * D, (size_t)(n_image * D) * sizeof(float));
                std::memcpy(kvd + n_image * D, cd + (n_text + n_image) * D,
                            (size_t)(n_audio * D) * sizeof(float));
            } else if (n_image > 0) {
                kv = Tensor({n_image, D});
                std::memcpy(kv.data<float>(), cd + n_text * D,
                            (size_t)(n_image * D) * sizeof(float));
            } else {
                kv = Tensor({n_audio, D});
                std::memcpy(kv.data<float>(), cd + n_text * D,
                            (size_t)(n_audio * D) * sizeof(float));
            }

            Tensor text_q({n_text, D});
            std::memcpy(text_q.data<float>(), cd, (size_t)(n_text * D) * sizeof(float));

            Tensor fused_text = shared_fusion_layers[l].forward(text_q, kv);
            std::memcpy(cd, fused_text.data<float>(), (size_t)(n_text * D) * sizeof(float));
        }
    }

    return output_norm.forward(combined);
}

ContrastiveAlignmentHead::ContrastiveAlignmentHead(int64_t hidden_size, float temperature)
    : text_projector(hidden_size, hidden_size),
      image_projector(hidden_size, hidden_size),
      audio_projector(hidden_size, hidden_size),
      temperature_(temperature > 0 ? temperature : 0.07f),
      hidden_size_(hidden_size) {}

float ContrastiveAlignmentHead::get_temperature() const { return temperature_; }
void ContrastiveAlignmentHead::set_temperature(float t) { temperature_ = t > 0 ? t : 0.07f; }

std::pair<Tensor, Tensor> ContrastiveAlignmentHead::project(const Tensor& text_raw,
                                                             const Tensor& image_raw) {
    Tensor t_proj = text_raw.numel() > 0 ? text_projector.forward(text_raw) : Tensor({0, hidden_size_});
    Tensor i_proj = image_raw.numel() > 0 ? image_projector.forward(image_raw) : Tensor({0, hidden_size_});

    if (t_proj.numel() > 0) {
        float* td = t_proj.data<float>();
        float norm = 0;
        for (int64_t i = 0; i < t_proj.numel(); i++) norm += td[i] * td[i];
        norm = std::sqrt(norm + 1e-10f);
        for (int64_t i = 0; i < t_proj.numel(); i++) td[i] /= norm;
    }
    if (i_proj.numel() > 0) {
        float* id = i_proj.data<float>();
        float norm = 0;
        for (int64_t i = 0; i < i_proj.numel(); i++) norm += id[i] * id[i];
        norm = std::sqrt(norm + 1e-10f);
        for (int64_t i = 0; i < i_proj.numel(); i++) id[i] /= norm;
    }
    return {t_proj, i_proj};
}

Tensor ContrastiveAlignmentHead::align_features(const Tensor& text_features,
                                                 const Tensor& image_features) {
    if (text_features.numel() == 0 || image_features.numel() == 0)
        return Tensor({0});

    int64_t B = text_features.dim(0);
    int64_t D = (text_features.numel() > 0 && B > 0) ? text_features.numel() / B : hidden_size_;

    Tensor aligned({B, D});
    float* ad = aligned.data<float>();
    const float* td = text_features.data<float>();
    const float* id = image_features.data<float>();

    for (int64_t b = 0; b < B; b++) {
        float tn = 0, in2 = 0;
        for (int64_t d = 0; d < D; d++) {
            tn += td[b * D + d] * td[b * D + d];
            in2 += id[b * D + d] * id[b * D + d];
        }
        tn = std::sqrt(tn + 1e-10f);
        in2 = std::sqrt(in2 + 1e-10f);
        float alpha = tn / (tn + in2 + 1e-10f);
        for (int64_t d = 0; d < D; d++)
            ad[b * D + d] = alpha * td[b * D + d] + (1.0f - alpha) * id[b * D + d];
    }
    return aligned;
}

float ContrastiveAlignmentHead::compute_clip_loss(const Tensor& text_features,
                                                    const Tensor& image_features) {
    if (text_features.numel() == 0 || image_features.numel() == 0) return 0;
    return compute_infonce_loss(text_features, image_features);
}

float ContrastiveAlignmentHead::compute_infonce_loss(const Tensor& features_a,
                                                       const Tensor& features_b) {
    if (features_a.numel() == 0 || features_b.numel() == 0) return 0;

    int64_t B = features_a.dim(0);
    int64_t D = features_a.numel() / std::max((int64_t)1, B);
    if (B <= 0 || D <= 0) return 0;

    const float* ad = features_a.data<float>();
    const float* bd = features_b.data<float>();

    std::vector<float> sim((size_t)B * (size_t)B, 0.0f);
    for (int64_t i = 0; i < B; i++)
        for (int64_t j = 0; j < B; j++)
            sim[(size_t)i * (size_t)B + (size_t)j] =
                dot_product(ad + i * D, bd + j * D, D) / temperature_;

    float loss = 0;
    for (int64_t i = 0; i < B; i++) {
        float mx = sim[(size_t)i * (size_t)B];
        for (int64_t j = 1; j < B; j++)
            if (sim[(size_t)i * (size_t)B + (size_t)j] > mx)
                mx = sim[(size_t)i * (size_t)B + (size_t)j];
        float s = 0;
        for (int64_t j = 0; j < B; j++)
            s += std::exp(sim[(size_t)i * (size_t)B + (size_t)j] - mx);
        loss -= std::exp(sim[(size_t)i * (size_t)B + (size_t)i] - mx) / (s + 1e-10f);
    }
    return loss / (float)B;
}

struct MultimodalCrossAttention::Impl {
    std::mt19937 rng{42};
};

MultimodalCrossAttention::MultimodalCrossAttention(const MultimodalEncoderConfig& cfg)
    : config(cfg),
      vision_encoder({cfg.image_size, cfg.image_patch_size, cfg.image_channels,
                      cfg.image_embed_dim, cfg.image_num_heads, cfg.image_num_layers,
                      cfg.image_embed_dim * 4}),
      audio_encoder({cfg.audio_n_mels, cfg.audio_embed_dim, cfg.audio_num_heads,
                     cfg.audio_num_layers}),
      text_embedding(cfg.text_vocab_size, cfg.text_embed_dim),
      fusion(cfg),
      contrastive_head(cfg.shared_hidden) {
    impl_ = new Impl();
}

MultimodalCrossAttention::~MultimodalCrossAttention() { delete impl_; }

Tensor MultimodalCrossAttention::encode_image(const Tensor& images) {
    return vision_encoder.forward(images);
}

Tensor MultimodalCrossAttention::encode_audio(const Tensor& spectrograms) {
    return audio_encoder.forward(spectrograms);
}

Tensor MultimodalCrossAttention::encode_text(const Tensor& text_ids) {
    if (text_ids.numel() == 0) return Tensor({0, config.text_embed_dim});
    int64_t T = (text_ids.rank() >= 2) ? text_ids.dim(1) : text_ids.numel();
    int64_t B = (text_ids.rank() >= 2) ? text_ids.dim(0) : 1;
    Tensor flat = (text_ids.rank() >= 2) ? text_ids.reshape({B * T}) : text_ids;
    return text_embedding.forward(flat);
}

void MultimodalCrossAttention::add_text_tokens(const Tensor& text_features,
                                                 Tensor& combined, int64_t& offset) {
    if (text_features.numel() == 0) return;
    int64_t n = text_features.dim(0);
    int64_t D = text_features.numel() / std::max((int64_t)1, n);
    std::memcpy(combined.data<float>() + offset * D,
                text_features.data<float>(),
                (size_t)(n * D) * sizeof(float));
    offset += n;
}

void MultimodalCrossAttention::add_image_tokens(const Tensor& image_features,
                                                  Tensor& combined, int64_t& offset) {
    if (image_features.numel() == 0) return;
    int64_t n = image_features.dim(0);
    int64_t D = image_features.numel() / std::max((int64_t)1, n);
    std::memcpy(combined.data<float>() + offset * D,
                image_features.data<float>(),
                (size_t)(n * D) * sizeof(float));
    offset += n;
}

void MultimodalCrossAttention::add_audio_tokens(const Tensor& audio_features,
                                                  Tensor& combined, int64_t& offset) {
    if (audio_features.numel() == 0) return;
    int64_t n = audio_features.dim(0);
    int64_t D = audio_features.numel() / std::max((int64_t)1, n);
    std::memcpy(combined.data<float>() + offset * D,
                audio_features.data<float>(),
                (size_t)(n * D) * sizeof(float));
    offset += n;
}

Tensor MultimodalCrossAttention::forward(const Tensor& text_tokens, const Tensor& image_tokens,
                                          const Tensor& audio_tokens, bool training) {
    int64_t D = config.shared_hidden;

    Tensor text_emb = encode_text(text_tokens);
    Tensor image_emb = image_tokens.numel() > 0 ? encode_image(image_tokens) : Tensor({0, D});
    Tensor audio_emb = audio_tokens.numel() > 0 ? encode_audio(audio_tokens) : Tensor({0, D});

    fusion.apply_modality_dropout(text_emb, image_emb, audio_emb,
                                   config.modality_dropout, training);

    return fusion.fuse(text_emb, image_emb, audio_emb);
}

Tensor MultimodalCrossAttention::forward_text_image(const Tensor& text_tokens,
                                                      const Tensor& image_tokens) {
    Tensor empty_audio({0});
    return forward(text_tokens, image_tokens, empty_audio);
}

Tensor MultimodalCrossAttention::forward_text_audio(const Tensor& text_tokens,
                                                      const Tensor& audio_tokens) {
    Tensor empty_image({0});
    return forward(text_tokens, empty_image, audio_tokens);
}

std::string MultimodalCrossAttention::generate_caption(const Tensor& image, int max_tokens) {
    if (image.numel() == 0) return "";

    Tensor image_feat = encode_image(image);
    static BPETokenizer bpe;
    std::vector<int> tokens = {bpe.bos_id()};

    int64_t V = config.text_vocab_size;

    for (int i = 0; i < max_tokens; i++) {
        Tensor input({1, (int64_t)tokens.size()});
        float* id = input.data<float>();
        for (size_t t = 0; t < tokens.size(); t++)
            id[t] = (float)tokens[t];

        Tensor text_emb = encode_text(input);

        int64_t Tt = text_emb.dim(0);
        int64_t Ti = image_feat.dim(0);

        Tensor q({Tt, config.shared_hidden});
        std::memcpy(q.data<float>(), text_emb.data<float>(),
                    (size_t)(Tt * config.shared_hidden) * sizeof(float));

        Tensor kv({Ti, config.shared_hidden});
        std::memcpy(kv.data<float>(), image_feat.data<float>(),
                    (size_t)(Ti * config.shared_hidden) * sizeof(float));

        if (fusion.shared_fusion_layers.size() > 0) {
            Tensor fused = fusion.shared_fusion_layers[0].forward(q, kv);
            (void)fused;
        }

        int next = (int)(impl_->rng() % (unsigned)V);
        if (next == bpe.eos_id()) break;
        tokens.push_back(next);
    }

    return bpe.decode(tokens);
}

std::vector<std::pair<int64_t, float>> MultimodalCrossAttention::retrieve_images(
    const Tensor& text_embedding, int top_k) {
    std::vector<std::pair<int64_t, float>> results;
    if (text_embedding.numel() == 0 || top_k <= 0) return results;

    int64_t D = text_embedding.numel();
    const float* td = text_embedding.data<float>();
    float tn = 0;
    for (int64_t i = 0; i < D; i++) tn += td[i] * td[i];
    tn = std::sqrt(tn + 1e-10f);

    for (int64_t i = 0; i < top_k; i++) {
        float score = (float)(impl_->rng()) / (float)UINT32_MAX;
        results.push_back({i, score});
    }
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    return results;
}

std::vector<std::pair<int64_t, float>> MultimodalCrossAttention::retrieve_text(
    const Tensor& image_embedding, int top_k) {
    std::vector<std::pair<int64_t, float>> results;
    if (image_embedding.numel() == 0 || top_k <= 0) return results;

    for (int64_t i = 0; i < top_k; i++) {
        float score = (float)(impl_->rng()) / (float)UINT32_MAX;
        results.push_back({i, score});
    }
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    return results;
}

Tensor MultimodalCrossAttention::visual_question_answer(const Tensor& image,
                                                          const Tensor& question_tokens) {
    Tensor empty_audio({0});
    return forward(question_tokens, image, empty_audio);
}

} // namespace quant
