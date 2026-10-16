#pragma once
#include "quant/trainer.h"
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace quant {
namespace dense_engine {

// Unified entry: dispatches Dense/FineTune/MoE/Native kinds to real paths.
// Wraps UnifiedTrainer (trainer_core) + UnifiedTrainArgs parsing.
struct UnifiedStats {
    int64_t steps = 0;
    int64_t tokens = 0;
    double tok_per_sec = 0.0;
    float last_loss = 0.0f;
};

class UnifiedEntry {
public:
    UnifiedEntry(Model* model, Tokenizer* tok);
    bool configure(const UnifiedTrainArgs& args, std::string* err = nullptr);
    bool run(std::string* err = nullptr);
    bool train_step_once(const Tensor& input_ids, const Tensor& labels,
                         float* loss_out, std::string* err = nullptr);
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);
    const TrainMetrics& metrics() const;
    const UnifiedStats& stats() const { return stats_; }
    static const char* kind_name(UnifiedTrainerKind k);
    static bool parse_kind(const std::string& s, UnifiedTrainerKind* out);
private:
    Model* model_ = nullptr;
    Tokenizer* tok_ = nullptr;
    UnifiedTrainer inner_;
    UnifiedStats stats_;
    int64_t t0_us_ = 0;
    static int64_t now_us();
};

// Reward-weighted trainer math: advantage normalize + policy-gradient
// loss with real tensor ops (binds Trainer::train_step rewards overload).
struct RewardStats {
    float mean = 0.0f;
    float std = 1.0f;
};
RewardStats normalize_rewards(const Tensor& rewards);
float policy_grad_loss(const Tensor& logits, const Tensor& labels,
                       const Tensor& advantages, Tensor& dlogits,
                       int64_t ignore_index = -100);

// GRPO group math: per-group baseline + clipped ratio, real ops.
struct GroupOut {
    float loss = 0.0f;
    float mean_reward = 0.0f;
};
GroupOut grpo_step(const Tensor& logits, const Tensor& labels,
                   const Tensor& rewards, int64_t group,
                   float clip_eps = 0.2f, float kl_coef = 0.01f);

} // namespace dense_engine
} // namespace quant
