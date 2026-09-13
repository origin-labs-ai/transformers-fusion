#pragma once
#include "quant/tensor.h"
#include "quant/model.h"
#include "quant/transformer.h"
#include "quant/multimodal.h"
#include <vector>
#include <cstdint>
#include <string>
#include <random>

namespace quant {

struct CrossAttentionLayerConfig {
    int64_t hidden_size = 768;
    int64_t num_heads = 8;
    int64_t head_dim = 96;
    float dropout_rate = 0.1f;
    bool use_rms_norm = true;
    float norm_eps = 1e-5f;
};

struct MultimodalEncoderConfig {
    int64_t image_patch_size = 16;
    int64_t image_size = 224;
    int64_t image_channels = 3;
    int64_t image_embed_dim = 768;
    int64_t image_num_layers = 12;
    int64_t image_num_heads = 12;

    int64_t audio_sample_rate = 16000;
    int64_t audio_n_mels = 80;
    int64_t audio_frame_size = 400;
    int64_t audio_hop_size = 160;
    int64_t audio_embed_dim = 384;
    int64_t audio_num_layers = 6;
    int64_t audio_num_heads = 6;

    int64_t text_vocab_size = 32000;
    int64_t text_embed_dim = 768;
    int64_t text_max_len = 2048;
    int64_t text_num_layers = 12;
    int64_t text_num_heads = 12;

    int64_t shared_hidden = 768;
    int64_t shared_num_heads = 12;
    int64_t shared_num_layers = 4;
    float modality_dropout = 0.1f;
};

struct ViTConfig {
    int64_t image_size = 224;
    int64_t patch_size = 16;
    int64_t in_channels = 3;
    int64_t hidden_dim = 768;
    int64_t num_heads = 12;
    int64_t num_layers = 12;
    int64_t mlp_dim = 3072;
    float dropout = 0.1f;
};

struct AudioEncoderConfig {
    int64_t input_dim = 80;
    int64_t hidden_dim = 384;
    int64_t num_heads = 6;
    int64_t num_layers = 6;
    int64_t kernel_size = 31;
    float dropout = 0.1f;
};

class PatchEmbedding {
public:
    PatchEmbedding(int64_t image_size, int64_t patch_size, int64_t in_channels, int64_t hidden_dim);
    Tensor forward(const Tensor& images);

    Tensor patch_proj_weight;
    Tensor pos_embed;
    int64_t num_patches;
    int64_t hidden_dim;
};

class VisionTransformer {
public:
    VisionTransformer(const ViTConfig& cfg);
    ~VisionTransformer();
    // BUGFIX (bug census): raw Impl* + implicit copy/move = double-free
    // (vector realloc moves the pointer without nulling the source).
    VisionTransformer(const VisionTransformer&) = delete;
    VisionTransformer& operator=(const VisionTransformer&) = delete;
    VisionTransformer(VisionTransformer&& o) noexcept
        : config(std::move(o.config)), patch_embed(std::move(o.patch_embed)),
          attn_qkv(std::move(o.attn_qkv)), attn_out(std::move(o.attn_out)),
          mlp_fc1(std::move(o.mlp_fc1)), mlp_fc2(std::move(o.mlp_fc2)),
          norm1(std::move(o.norm1)), norm2(std::move(o.norm2)),
          final_norm(std::move(o.final_norm)), impl_(o.impl_) { o.impl_ = nullptr; }
    VisionTransformer& operator=(VisionTransformer&& o) noexcept {
        if (this != &o) {
            delete impl_;
            config = std::move(o.config); patch_embed = std::move(o.patch_embed);
            attn_qkv = std::move(o.attn_qkv); attn_out = std::move(o.attn_out);
            mlp_fc1 = std::move(o.mlp_fc1); mlp_fc2 = std::move(o.mlp_fc2);
            norm1 = std::move(o.norm1); norm2 = std::move(o.norm2);
            final_norm = std::move(o.final_norm);
            impl_ = o.impl_; o.impl_ = nullptr;
        }
        return *this;
    }

    Tensor forward(const Tensor& images);
    std::vector<Tensor> get_patch_features(const Tensor& images);
    Tensor encode(const Tensor& images);

    ViTConfig config;
    PatchEmbedding patch_embed;
    std::vector<Linear> attn_qkv;
    std::vector<Linear> attn_out;
    std::vector<Linear> mlp_fc1;
    std::vector<Linear> mlp_fc2;
    std::vector<RMSNorm> norm1;
    std::vector<RMSNorm> norm2;
    RMSNorm final_norm;

private:
    struct Impl;
    Impl* impl_;
};

class Conv1DExtractor {
public:
    Conv1DExtractor(int64_t in_dim, int64_t out_dim, int64_t kernel_size, int64_t stride);
    Tensor forward(const Tensor& input);

    Tensor weight;
    Tensor bias;
    int64_t stride_;
};

class AudioTransformer {
public:
    AudioTransformer(const AudioEncoderConfig& cfg);
    ~AudioTransformer();
    // BUGFIX (bug census): same raw-Impl double-free as VisionTransformer.
    AudioTransformer(const AudioTransformer&) = delete;
    AudioTransformer& operator=(const AudioTransformer&) = delete;
    AudioTransformer(AudioTransformer&& o) noexcept
        : config(std::move(o.config)), conv1(std::move(o.conv1)), conv2(std::move(o.conv2)),
          attn_qkv(std::move(o.attn_qkv)), attn_out(std::move(o.attn_out)),
          ffn_fc1(std::move(o.ffn_fc1)), ffn_fc2(std::move(o.ffn_fc2)),
          norm1(std::move(o.norm1)), norm2(std::move(o.norm2)),
          final_norm(std::move(o.final_norm)), impl_(o.impl_) { o.impl_ = nullptr; }
    AudioTransformer& operator=(AudioTransformer&& o) noexcept {
        if (this != &o) {
            delete impl_;
            config = std::move(o.config); conv1 = std::move(o.conv1); conv2 = std::move(o.conv2);
            attn_qkv = std::move(o.attn_qkv); attn_out = std::move(o.attn_out);
            ffn_fc1 = std::move(o.ffn_fc1); ffn_fc2 = std::move(o.ffn_fc2);
            norm1 = std::move(o.norm1); norm2 = std::move(o.norm2);
            final_norm = std::move(o.final_norm);
            impl_ = o.impl_; o.impl_ = nullptr;
        }
        return *this;
    }

    Tensor forward(const Tensor& spectrograms);
    Tensor encode(const Tensor& spectrograms);

    AudioEncoderConfig config;
    Conv1DExtractor conv1;
    Conv1DExtractor conv2;
    std::vector<Linear> attn_qkv;
    std::vector<Linear> attn_out;
    std::vector<Linear> ffn_fc1;
    std::vector<Linear> ffn_fc2;
    std::vector<RMSNorm> norm1;
    std::vector<RMSNorm> norm2;
    RMSNorm final_norm;

private:
    struct Impl;
    Impl* impl_;
};

class CrossAttentionBlock {
public:
    CrossAttentionBlock(const CrossAttentionLayerConfig& cfg);
    ~CrossAttentionBlock();
    // BUGFIX (bug census): same raw-Impl double-free; CRITICAL here because
    // ModalityFusionEncoder stores these in vectors (realloc moves).
    // NOTE: move still copies Linear/RMSNorm members (their own move rules
    // apply); impl_ pointer is stolen + nulled.
    CrossAttentionBlock(const CrossAttentionBlock&) = delete;
    CrossAttentionBlock& operator=(const CrossAttentionBlock&) = delete;
    CrossAttentionBlock(CrossAttentionBlock&& o) noexcept;
    CrossAttentionBlock& operator=(CrossAttentionBlock&& o) noexcept;

    Tensor forward(const Tensor& query, const Tensor& key_value,
                   const std::string& modality_tag = "");
    Tensor forward_self_attn(const Tensor& x);

    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear out_proj;
    Linear ffn_fc1;
    Linear ffn_fc2;
    RMSNorm norm1;
    RMSNorm norm2;
    RMSNorm cross_norm;
    CrossAttentionLayerConfig config;

private:
    struct Impl;
    Impl* impl_;
};

class ModalityFusionEncoder {
public:
    ModalityFusionEncoder(const MultimodalEncoderConfig& cfg);
    ~ModalityFusionEncoder();
    // BUGFIX (bug census): same raw-Impl double-free as the blocks above.
    ModalityFusionEncoder(const ModalityFusionEncoder&) = delete;
    ModalityFusionEncoder& operator=(const ModalityFusionEncoder&) = delete;
    ModalityFusionEncoder(ModalityFusionEncoder&& o) noexcept;
    ModalityFusionEncoder& operator=(ModalityFusionEncoder&& o) noexcept;

    Tensor fuse(const Tensor& text_emb, const Tensor& image_emb, const Tensor& audio_emb);
    Tensor fuse_early(const Tensor& text_emb, const Tensor& image_emb, const Tensor& audio_emb);
    Tensor fuse_late(const Tensor& text_emb, const Tensor& image_emb, const Tensor& audio_emb);
    std::vector<Tensor> get_per_modality_features(const Tensor& text_emb,
                                                   const Tensor& image_emb,
                                                   const Tensor& audio_emb);
    void apply_modality_dropout(Tensor& text_emb, Tensor& image_emb, Tensor& audio_emb,
                                float drop_prob, bool training);

    MultimodalEncoderConfig config;
    std::vector<CrossAttentionBlock> text_cross_attn_layers;
    std::vector<CrossAttentionBlock> image_cross_attn_layers;
    std::vector<CrossAttentionBlock> audio_cross_attn_layers;
    std::vector<CrossAttentionBlock> shared_fusion_layers;
    Linear text_proj;
    Linear image_proj;
    Linear audio_proj;
    Linear output_proj;
    RMSNorm output_norm;

private:
    struct Impl;
    Impl* impl_;
};

struct ContrastivePair {
    int64_t text_idx;
    int64_t image_idx;
    float temperature = 0.07f;
};

class ContrastiveAlignmentHead {
public:
    ContrastiveAlignmentHead(int64_t hidden_size, float temperature = 0.07f);

    float compute_clip_loss(const Tensor& text_features, const Tensor& image_features);
    float compute_infonce_loss(const Tensor& features_a, const Tensor& features_b);
    Tensor align_features(const Tensor& text_features, const Tensor& image_features);
    std::pair<Tensor, Tensor> project(const Tensor& text_raw, const Tensor& image_raw);
    float get_temperature() const;
    void set_temperature(float t);

    Linear text_projector;
    Linear image_projector;
    Linear audio_projector;
    float temperature_;
    int64_t hidden_size_;
};

class MultimodalCrossAttention {
public:
    MultimodalCrossAttention(const MultimodalEncoderConfig& cfg = MultimodalEncoderConfig());
    ~MultimodalCrossAttention();
    // BUGFIX (bug census): same raw-Impl double-free as the encoders above.
    MultimodalCrossAttention(const MultimodalCrossAttention&) = delete;
    MultimodalCrossAttention& operator=(const MultimodalCrossAttention&) = delete;
    MultimodalCrossAttention(MultimodalCrossAttention&& o) noexcept
        : config(std::move(o.config)), vision_encoder(std::move(o.vision_encoder)),
          audio_encoder(std::move(o.audio_encoder)),
          text_embedding(std::move(o.text_embedding)),
          fusion(std::move(o.fusion)),
          contrastive_head(std::move(o.contrastive_head)), impl_(o.impl_) { o.impl_ = nullptr; }
    MultimodalCrossAttention& operator=(MultimodalCrossAttention&& o) noexcept {
        if (this != &o) {
            delete impl_;
            config = std::move(o.config);
            vision_encoder = std::move(o.vision_encoder);
            audio_encoder = std::move(o.audio_encoder);
            text_embedding = std::move(o.text_embedding);
            fusion = std::move(o.fusion);
            contrastive_head = std::move(o.contrastive_head);
            impl_ = o.impl_; o.impl_ = nullptr;
        }
        return *this;
    }

    Tensor forward(const Tensor& text_tokens, const Tensor& image_tokens,
                   const Tensor& audio_tokens, bool training = false);
    Tensor forward_text_image(const Tensor& text_tokens, const Tensor& image_tokens);
    Tensor forward_text_audio(const Tensor& text_tokens, const Tensor& audio_tokens);

    Tensor encode_image(const Tensor& images);
    Tensor encode_audio(const Tensor& spectrograms);
    Tensor encode_text(const Tensor& text_ids);

    std::string generate_caption(const Tensor& image, int max_tokens = 64);
    std::vector<std::pair<int64_t, float>> retrieve_images(const Tensor& text_embedding,
                                                            int top_k = 5);
    std::vector<std::pair<int64_t, float>> retrieve_text(const Tensor& image_embedding,
                                                          int top_k = 5);
    Tensor visual_question_answer(const Tensor& image, const Tensor& question_tokens);

    void add_image_tokens(const Tensor& image_features, Tensor& combined, int64_t& offset);
    void add_audio_tokens(const Tensor& audio_features, Tensor& combined, int64_t& offset);
    void add_text_tokens(const Tensor& text_features, Tensor& combined, int64_t& offset);

    MultimodalEncoderConfig config;
    VisionTransformer vision_encoder;
    AudioTransformer audio_encoder;
    Linear text_embedding;
    ModalityFusionEncoder fusion;
    ContrastiveAlignmentHead contrastive_head;

private:
    struct Impl;
    Impl* impl_;
};

} // namespace quant
