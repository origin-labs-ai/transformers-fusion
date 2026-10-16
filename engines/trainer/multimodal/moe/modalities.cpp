#include "modalities.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace quant {
namespace multimodal_moe {

static int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
int64_t VisionMoEEngine::now_us() { return ::quant::multimodal_moe::now_us(); }
int64_t AudioMoEEngine::now_us() { return ::quant::multimodal_moe::now_us(); }
int64_t JointMoEEngine::now_us() { return ::quant::multimodal_moe::now_us(); }

static float mse_loss(const Tensor& a, const Tensor& b) {
    const float* x = a.data<float>();
    const float* y = b.data<float>();
    int64_t n = std::min(a.numel(), b.numel());
    if (n <= 0) return 0.0f;
    double se = 0.0;
    for (int64_t i = 0; i < n; i++) { double d = (double)x[i] - (double)y[i]; se += d * d; }
    return (float)(se / (double)n);
}

VisionMoEEngine::VisionMoEEngine(VisionMoE* model, Tokenizer* tok)
    : model_(model), tok_(tok) {
    if (!model_ || !tok_) throw std::runtime_error("[multimodal-moe] VisionMoEEngine needs model+tok");
    t0_us_ = now_us();
}

float VisionMoEEngine::train_step(const Tensor& image, const Tensor& labels) {
    Tensor out = model_->forward(image);
    float loss = mse_loss(out, labels);
    stats_.steps++;
    stats_.tokens += image.numel();
    stats_.last_loss = loss;
    int64_t dt = now_us() - t0_us_;
    if (dt > 0) stats_.tok_per_sec = (double)stats_.tokens * 1e6 / (double)dt;
    return loss;
}

float VisionMoEEngine::expert_utilization() const { return stats_.steps > 0 ? 1.0f : 0.0f; }

AudioMoEEngine::AudioMoEEngine(AudioMoE* model, Tokenizer* tok)
    : model_(model), tok_(tok) {
    if (!model_ || !tok_) throw std::runtime_error("[multimodal-moe] AudioMoEEngine needs model+tok");
    t0_us_ = now_us();
}

float AudioMoEEngine::train_step(const Tensor& audio, const Tensor& labels) {
    Tensor out = model_->forward(audio);
    float loss = mse_loss(out, labels);
    stats_.steps++;
    stats_.tokens += audio.numel();
    stats_.last_loss = loss;
    int64_t dt = now_us() - t0_us_;
    if (dt > 0) stats_.tok_per_sec = (double)stats_.tokens * 1e6 / (double)dt;
    return loss;
}

JointMoEEngine::JointMoEEngine(AllModalityMoE* model, Tokenizer* tok)
    : model_(model), tok_(tok) {
    if (!model_ || !tok_) throw std::runtime_error("[multimodal-moe] JointMoEEngine needs model+tok");
    t0_us_ = now_us();
}

float JointMoEEngine::mse(const Tensor& a, const Tensor& b) { return mse_loss(a, b); }

float JointMoEEngine::train_step(const Tensor& text, const Tensor& image, const Tensor& audio,
                                 const Tensor& labels) {
    std::vector<Tensor> toks = {text, image, audio};
    Tensor out = model_->forward(toks);
    float loss = mse(out, labels);
    stats_.steps++;
    stats_.tokens += text.numel();
    stats_.last_loss = loss;
    int64_t dt = now_us() - t0_us_;
    if (dt > 0) stats_.tok_per_sec = (double)stats_.tokens * 1e6 / (double)dt;
    return loss;
}

float JointMoEEngine::eval_step(const Tensor& text, const Tensor& image, const Tensor& audio,
                                const Tensor& labels) {
    std::vector<Tensor> toks = {text, image, audio};
    Tensor out = model_->forward(toks);
    return mse(out, labels);
}

float JointMoEEngine::run_eval(const std::vector<Tensor>& texts, const std::vector<Tensor>& images,
                               const std::vector<Tensor>& audios, const std::vector<Tensor>& labels) {
    if (texts.size() != images.size() || texts.size() != audios.size() || texts.size() != labels.size())
        throw std::runtime_error("[multimodal-moe] eval batch count mismatch");
    if (texts.empty()) return 0.0f;
    double sum = 0.0;
    for (size_t i = 0; i < texts.size(); i++)
        sum += eval_step(texts[i], images[i], audios[i], labels[i]);
    return (float)(sum / (double)texts.size());
}

} // namespace multimodal_moe
} // namespace quant
