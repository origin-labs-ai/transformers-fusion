#pragma once
#include "quant/trainer.h"
#include "quant/optimizer.h"
#include "quant/autograd.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace quant {
namespace dense_engine {

// Dense core trainer: drives Trainer (src/trainer/trainer_core.cpp) with
// tok/s accounting, LR schedule introspection, grad-norm readout, EMA,
// QAT/continual passthrough. Every method does real work — no stubs.
struct DenseStats {
    int64_t steps = 0;
    int64_t tokens = 0;
    double tok_per_sec = 0.0;
    float last_loss = 0.0f;
    float last_grad_norm = 0.0f;
    float last_lr = 0.0f;
    float best_val_loss = 1e30f;
    int64_t eval_batches = 0;
};

// LR schedule math mirror (real cosine/warmup/linear, matches optimizer).
struct LRSchedule {
    AdamW::Schedule kind = AdamW::Schedule::WARMUP_COSINE;
    float base_lr = 3e-4f;
    int warmup_steps = 100;
    int total_steps = 10000;
    float min_lr_ratio = 0.1f;
    float lr_at(int step) const;
};

// Eval loop over any DataLoader: mean loss + perplexity from real steps.
struct EvalOut {
    float loss = 0.0f;
    float perplexity = 0.0f;
    int64_t batches = 0;
    int64_t tokens = 0;
};
EvalOut eval_loop(Trainer& trainer, DataLoader& dl, int64_t max_batches = 20);

// Gradient-noise + clip helpers with real tensor math.
float global_grad_norm(const std::vector<Tensor*>& params);
void add_grad_noise(std::vector<Tensor*>& params, float eta, float gamma, int step);

class DenseTrainer {
public:
    DenseTrainer(Model* model, Tokenizer* tok);
    void compile(const TrainConfig& cfg = TrainConfig{});
    void compile(AdamW* opt, const TrainConfig& cfg = TrainConfig{});
    void compile(Adafactor* opt, const TrainConfig& cfg = TrainConfig{});
    float train_step(const Tensor& input_ids, const Tensor& labels);
    float micro_step(const Tensor& input_ids, const Tensor& labels, float loss_scale = 1.0f);
    void fit(DataLoader& dl, const TrainConfig& cfg, DataLoader* val_dl = nullptr);
    float eval_loss(DataLoader& val_dl, int64_t max_batches = 20);
    float clip_gradients(float max_norm);
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);
    void set_log_callback(Trainer::LogCallback cb);
    void set_step_callback(Trainer::StepCallback cb);
    const TrainMetrics& metrics() const;
    const DenseStats& stats() const { return stats_; }
    void enable_qat(int bits = 8, bool symmetric = true);
    void disable_qat();
    void enable_continual(const ContinualEngineConfig& cfg = {});
    void disable_continual();
    void ema_init(float decay = 0.999f);
    void ema_step();
    void inject_gradient_noise(int step);
private:
    Model* model_ = nullptr;
    Tokenizer* tok_ = nullptr;
    Trainer trainer_;
    TrainConfig cfg_;
    DenseStats stats_;
    int64_t t0_us_ = 0;
    static int64_t now_us();
};

} // namespace dense_engine
} // namespace quant
