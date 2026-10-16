#pragma once
#include "quant/multimodal.h"
#include "quant/moe_trainer.h"
#include <chrono>
#include <cstdint>
#include <string>

namespace quant {
namespace multimodal_moe {

// Multimodal MoE path: VisionMoE/AudioMoE/VisionTextMoE/AudioTextMoE/
// AllModalityMoE driven with real forward + aux-loss + utilization.
struct MoEModalityStats {
    int64_t steps = 0;
    int64_t tokens = 0;
    double tok_per_sec = 0.0;
    float last_loss = 0.0f;
    float expert_util = 0.0f;
};

class VisionMoEEngine {
public:
    VisionMoEEngine(VisionMoE* model, Tokenizer* tok);
    float train_step(const Tensor& image, const Tensor& labels);
    float expert_utilization() const;
    const MoEModalityStats& stats() const { return stats_; }
private:
    VisionMoE* model_ = nullptr;
    Tokenizer* tok_ = nullptr;
    MoEModalityStats stats_;
    int64_t t0_us_ = 0;
    static int64_t now_us();
};

class AudioMoEEngine {
public:
    AudioMoEEngine(AudioMoE* model, Tokenizer* tok);
    float train_step(const Tensor& audio, const Tensor& labels);
    const MoEModalityStats& stats() const { return stats_; }
private:
    AudioMoE* model_ = nullptr;
    Tokenizer* tok_ = nullptr;
    MoEModalityStats stats_;
    int64_t t0_us_ = 0;
    static int64_t now_us();
};

class JointMoEEngine {
public:
    JointMoEEngine(AllModalityMoE* model, Tokenizer* tok);
    float train_step(const Tensor& text, const Tensor& image, const Tensor& audio,
                     const Tensor& labels);
    float eval_step(const Tensor& text, const Tensor& image, const Tensor& audio,
                    const Tensor& labels);
    float run_eval(const std::vector<Tensor>& texts, const std::vector<Tensor>& images,
                   const std::vector<Tensor>& audios, const std::vector<Tensor>& labels);
    const MoEModalityStats& stats() const { return stats_; }
private:
    AllModalityMoE* model_ = nullptr;
    Tokenizer* tok_ = nullptr;
    MoEModalityStats stats_;
    int64_t t0_us_ = 0;
    static int64_t now_us();
    static float mse(const Tensor& a, const Tensor& b);
};

} // namespace multimodal_moe
} // namespace quant
