#pragma once
#include "quant/tensor.h"
#include <cstdint>
#include <vector>

namespace quant {
namespace dense_engine {

// Full loss suite with real forward+backward math on host tensors.
// Every loss returns value AND writes dlogits gradient.
struct LossOut {
    float loss = 0.0f;
    float acc = 0.0f;
};

// Cross-entropy with ignore_index, real softmax + dlogits.
LossOut ce_loss(const Tensor& logits, const Tensor& labels, Tensor& dlogits,
                int64_t ignore_index = -100);
// Label-smoothed CE: real uniform mix.
LossOut label_smooth_ce(const Tensor& logits, const Tensor& labels, Tensor& dlogits,
                        float smoothing = 0.1f, int64_t ignore_index = -100);
// Focal loss: real (1-pt)^gamma weighting.
LossOut focal_loss(const Tensor& logits, const Tensor& labels, Tensor& dlogits,
                   float gamma = 2.0f, float alpha = 0.25f, int64_t ignore_index = -100);
// R-Drop: symmetric KL between two passes + CE, real math.
struct RDropOut {
    float ce = 0.0f;
    float kl = 0.0f;
    float total = 0.0f;
};
RDropOut rdrop_loss(const Tensor& logits1, const Tensor& logits2, const Tensor& labels,
                    float alpha = 1.0f, int64_t ignore_index = -100);
// Contrastive InfoNCE over row-normalized embeddings, real math.
float infonce_loss(const Tensor& emb, const Tensor& pos, Tensor& demb, Tensor& dpos,
                   float temperature = 0.07f);
// MSE with real dlogits.
LossOut mse_loss_fn(const Tensor& pred, const Tensor& target, Tensor& dpred);
// Z-loss helper shared with MoE path.
float seq_z_loss(const Tensor& logits);

} // namespace dense_engine
} // namespace quant
