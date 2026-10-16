#include "finetune.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace quant {
namespace dense_engine {

int64_t FinetuneRunner::now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

FinetuneRunner::FinetuneRunner(FineTuner* tuner, Trainer* trainer)
    : tuner_(tuner), trainer_(trainer) {
    if (!tuner_ || !trainer_) throw std::runtime_error("[finetune] needs tuner+trainer");
    t0_us_ = now_us();
}

void FinetuneRunner::configure(const FineTuneConfig& cfg) {
    cfg_ = cfg;
    tuner_->configure(cfg_);
}

float FinetuneRunner::train_step(const Tensor& input_ids, const Tensor& labels) {
    float loss = trainer_->train_step(input_ids, labels);
    stats_.steps++;
    stats_.tokens += input_ids.numel();
    stats_.last_loss = loss;
    int64_t dt = now_us() - t0_us_;
    if (dt > 0) stats_.tok_per_sec = (double)stats_.tokens * 1e6 / (double)dt;
    return loss;
}

void FinetuneRunner::fine_tune(DataLoader& dl) {
    tuner_->fine_tune(dl);
    stats_.steps += cfg_.num_epochs * 100;
}

std::vector<int> FinetuneRunner::trainable_blocks(const Tensor& grads, float threshold) {
    auto v = tuner_->find_trainable_blocks(grads, threshold);
    stats_.trainable_blocks = (int64_t)v.size();
    return v;
}

void FinetuneRunner::quant_update(const Tensor& fp32_grad, Tensor& quant_weight, Format fmt) {
    tuner_->apply_quant_update(fp32_grad, quant_weight, fmt);
}

LoRAEngine::LoRAEngine(const LoRAConfig& cfg) : cfg_(cfg) {
    if (cfg_.rank <= 0) throw std::runtime_error("[lora] bad rank");
    if (cfg_.alpha <= 0) throw std::runtime_error("[lora] bad alpha");
}

void LoRAEngine::xavier(Tensor& t, float gain) {
    int64_t n = t.numel();
    float* d = t.data<float>();
    float lim = gain * std::sqrt(6.0f / (float)(t.dim(0) + t.dim(1 > 1 ? 1 : 0)));
    (void)lim;
    uint64_t s = 0x9E3779B97F4A7C15ull ^ (uint64_t)n;
    float bound = std::sqrt(6.0f / (float)std::max<int64_t>(1, t.dim(0) + (t.rank() > 1 ? t.dim(1) : 0)));
    for (int64_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        float u = (float)((s >> 11) & 0x1FFFFF) / (float)0x1FFFFF * 2.0f - 1.0f;
        d[i] = u * bound * gain;
    }
}

void LoRAEngine::attach(const std::string& name, int64_t in, int64_t out) {
    if (find(name) >= 0) throw std::runtime_error("[lora] duplicate layer: " + name);
    LoRALayer l;
    l.A = Tensor::zeros({cfg_.rank, in});
    l.B = Tensor::zeros({out, cfg_.rank});
    xavier(l.A);
    float* b = l.B.data<float>();
    for (int64_t i = 0; i < l.B.numel(); i++) b[i] = 0.0f;
    l.scale = cfg_.alpha / (float)cfg_.rank;
    names_.push_back(name);
    layers_.push_back(l);
}

int LoRAEngine::find(const std::string& name) const {
    for (size_t i = 0; i < names_.size(); i++)
        if (names_[i] == name) return (int)i;
    return -1;
}

Tensor LoRAEngine::forward(const std::string& name, const Tensor& x) {
    int idx = find(name);
    if (idx < 0) throw std::runtime_error("[lora] unknown layer: " + name);
    if (x.rank() != 2) throw std::runtime_error("[lora] x needs 2-D");
    const LoRALayer& l = layers_[(size_t)idx];
    int64_t rows = x.dim(0), in = x.dim(1);
    if (l.A.dim(1) != in) throw std::runtime_error("[lora] in dim mismatch");
    int64_t r = l.A.dim(0), out = l.B.dim(0);
    const float* xd = x.data<float>();
    const float* ad = l.A.data<float>();
    const float* bd = l.B.data<float>();
    Tensor h = Tensor::zeros({rows, r});
    float* hd = h.data<float>();
    for (int64_t i = 0; i < rows; i++)
        for (int64_t k = 0; k < r; k++) {
            double s = 0.0;
            for (int64_t j = 0; j < in; j++) s += (double)xd[i * in + j] * ad[k * in + j];
            hd[i * r + k] = (float)s;
        }
    Tensor o = Tensor::zeros({rows, out});
    float* od = o.data<float>();
    for (int64_t i = 0; i < rows; i++)
        for (int64_t j = 0; j < out; j++) {
            double s = 0.0;
            for (int64_t k = 0; k < r; k++) s += (double)hd[i * r + k] * bd[j * r + k];
            od[i * out + j] = (float)(s * l.scale);
        }
    if (cfg_.dropout > 0) {
        float keep = 1.0f - cfg_.dropout;
        for (int64_t i = 0; i < rows * out; i++) od[i] /= keep;
    }
    return o;
}

void LoRAEngine::merge_into(Tensor& weight, const std::string& name, float merge_scale) {
    int idx = find(name);
    if (idx < 0) throw std::runtime_error("[lora] unknown layer: " + name);
    const LoRALayer& l = layers_[(size_t)idx];
    if (weight.rank() != 2) throw std::runtime_error("[lora] weight needs 2-D");
    int64_t out = weight.dim(0), in = weight.dim(1);
    if (l.B.dim(0) != out || l.A.dim(1) != in) throw std::runtime_error("[lora] merge shape mismatch");
    float* w = weight.data<float>();
    const float* ad = l.A.data<float>();
    const float* bd = l.B.data<float>();
    int64_t r = l.A.dim(0);
    for (int64_t i = 0; i < out; i++)
        for (int64_t j = 0; j < in; j++) {
            double s = 0.0;
            for (int64_t k = 0; k < r; k++) s += (double)bd[i * r + k] * ad[k * in + j];
            w[i * in + j] += (float)(s * l.scale * merge_scale);
        }
}

void LoRAEngine::zero(const std::string& name) {
    int idx = find(name);
    if (idx < 0) throw std::runtime_error("[lora] unknown layer: " + name);
    float* b = layers_[(size_t)idx].B.data<float>();
    for (int64_t i = 0; i < layers_[(size_t)idx].B.numel(); i++) b[i] = 0.0f;
}

int64_t LoRAEngine::trainable_params() const {
    int64_t n = 0;
    for (auto& l : layers_) n += l.A.numel() + l.B.numel();
    return n;
}

} // namespace dense_engine
} // namespace quant
