#include "trainer.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace quant {
namespace dense_engine {

int64_t DenseTrainer::now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

DenseTrainer::DenseTrainer(Model* model, Tokenizer* tok)
    : model_(model), tok_(tok), trainer_(model, tok) {
    if (!model_ || !tok_) throw std::runtime_error("[dense-engine] needs Model*+Tokenizer*");
    t0_us_ = now_us();
}

void DenseTrainer::compile(const TrainConfig& cfg) { cfg_ = cfg; trainer_.compile(cfg_); }
void DenseTrainer::compile(AdamW* opt, const TrainConfig& cfg) { cfg_ = cfg; trainer_.compile(opt, cfg_); }
void DenseTrainer::compile(Adafactor* opt, const TrainConfig& cfg) { cfg_ = cfg; trainer_.compile(opt, cfg_); }

float DenseTrainer::train_step(const Tensor& input_ids, const Tensor& labels) {
    float loss = trainer_.train_step(input_ids, labels);
    stats_.steps++;
    stats_.tokens += input_ids.numel();
    stats_.last_loss = loss;
    stats_.last_grad_norm = trainer_.metrics().grad_norm;
    stats_.last_lr = trainer_.metrics().learning_rate;
    int64_t dt = now_us() - t0_us_;
    if (dt > 0) stats_.tok_per_sec = (double)stats_.tokens * 1e6 / (double)dt;
    return loss;
}

float DenseTrainer::micro_step(const Tensor& input_ids, const Tensor& labels, float loss_scale) {
    return trainer_.micro_step(input_ids, labels, loss_scale);
}

void DenseTrainer::fit(DataLoader& dl, const TrainConfig& cfg, DataLoader* val_dl) {
    cfg_ = cfg;
    trainer_.fit(dl, cfg_, val_dl);
}

float DenseTrainer::eval_loss(DataLoader& val_dl, int64_t max_batches) {
    return trainer_.eval_loss(val_dl, max_batches);
}

float DenseTrainer::clip_gradients(float max_norm) { return trainer_.clip_gradients(max_norm); }
void DenseTrainer::save_checkpoint(const std::string& path) { trainer_.save_checkpoint(path); }
void DenseTrainer::load_checkpoint(const std::string& path) { trainer_.load_checkpoint(path); }
void DenseTrainer::set_log_callback(Trainer::LogCallback cb) { trainer_.set_log_callback(cb); }
void DenseTrainer::set_step_callback(Trainer::StepCallback cb) { trainer_.set_step_callback(cb); }
const TrainMetrics& DenseTrainer::metrics() const { return trainer_.metrics(); }
void DenseTrainer::enable_qat(int bits, bool symmetric) { trainer_.enable_qat(bits, symmetric); }
void DenseTrainer::disable_qat() { trainer_.disable_qat(); }
void DenseTrainer::enable_continual(const ContinualEngineConfig& cfg) { trainer_.enable_continual(cfg); }
void DenseTrainer::disable_continual() { trainer_.disable_continual(); }
void DenseTrainer::ema_init(float decay) { trainer_.ema_init(decay); }
void DenseTrainer::ema_step() { trainer_.ema_step(); }
void DenseTrainer::inject_gradient_noise(int step) { trainer_.inject_gradient_noise(step); }

float LRSchedule::lr_at(int step) const {
    if (total_steps <= 0) return base_lr;
    int s = std::max(0, std::min(step, total_steps));
    switch (kind) {
        case AdamW::Schedule::CONSTANT: return base_lr;
        case AdamW::Schedule::LINEAR:
            return base_lr * (1.0f - (float)s / (float)total_steps);
        case AdamW::Schedule::COSINE: {
            float t = (float)s / (float)total_steps;
            float c = 0.5f * (1.0f + std::cos(3.14159265f * t));
            return base_lr * (min_lr_ratio + (1.0f - min_lr_ratio) * c);
        }
        case AdamW::Schedule::WARMUP_COSINE: {
            if (s < warmup_steps && warmup_steps > 0)
                return base_lr * (float)s / (float)warmup_steps;
            float t = (float)(s - warmup_steps) / (float)std::max(1, total_steps - warmup_steps);
            float c = 0.5f * (1.0f + std::cos(3.14159265f * t));
            return base_lr * (min_lr_ratio + (1.0f - min_lr_ratio) * c);
        }
        case AdamW::Schedule::COSINE_RESTART: {
            int cycle = total_steps / 2 > 0 ? total_steps / 2 : total_steps;
            int cs = s % (cycle > 0 ? cycle : 1);
            float t = (float)cs / (float)(cycle > 0 ? cycle : 1);
            float c = 0.5f * (1.0f + std::cos(3.14159265f * t));
            return base_lr * (min_lr_ratio + (1.0f - min_lr_ratio) * c);
        }
    }
    return base_lr;
}

EvalOut eval_loop(Trainer& trainer, DataLoader& dl, int64_t max_batches) {
    EvalOut out;
    dl.reset();
    Tensor in, lb;
    double sum = 0.0;
    while (out.batches < max_batches && dl.next_batch(in, lb)) {
        float l = trainer.micro_step(in, lb, 1.0f);
        sum += l;
        out.batches++;
        out.tokens += in.numel();
    }
    if (out.batches > 0) {
        out.loss = (float)(sum / (double)out.batches);
        out.perplexity = std::exp(std::min(out.loss, 20.0f));
    }
    return out;
}

float global_grad_norm(const std::vector<Tensor*>& params) {
    double sum = 0.0;
    for (auto* p : params) {
        if (!p || !p->has_grad()) continue;
        const float* g = p->grad().data<float>();
        int64_t n = p->numel();
        for (int64_t i = 0; i < n; i++) sum += (double)g[i] * (double)g[i];
    }
    return (float)std::sqrt(sum);
}

void add_grad_noise(std::vector<Tensor*>& params, float eta, float gamma, int step) {
    if (eta <= 0.0f) return;
    float var = eta / std::pow(1.0f + (float)step, gamma);
    float stdv = std::sqrt(var);
    uint64_t seed = 0x9E3779B97F4A7C15ull ^ (uint64_t)(step * 2 + 1);
    for (auto* p : params) {
        if (!p || !p->has_grad()) continue;
        float* g = p->grad().data<float>();
        int64_t n = p->numel();
        for (int64_t i = 0; i < n; i++) {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            float u = (float)((seed >> 11) & 0x1FFFFF) / (float)0x1FFFFF - 0.5f;
            g[i] += stdv * u * 3.4641016f;
        }
    }
}

} // namespace dense_engine
} // namespace quant
