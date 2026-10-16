#pragma once
#include "quant/continual_engine.h"
#include "quant/trainer.h"
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace quant {
namespace dense_engine {

struct ContinualStats {
    uint32_t task_id = 0;
    int64_t steps = 0;
    float ewc_penalty = 0.0f;
    float replay_loss = 0.0f;
    float forgetting = 0.0f;
    float stability = 0.0f;
};

class ContinualRunner {
public:
    ContinualRunner(Trainer* trainer, const ContinualEngineConfig& cfg = {});
    void on_task_boundary(uint32_t new_task_id,
                          const float* eval_inputs = nullptr,
                          const float* eval_targets = nullptr,
                          size_t eval_count = 0);
    float train_step_mixed(const Tensor& input_ids, const Tensor& labels,
                           float replay_ratio = 0.25f);
    float ewc_penalty(float lambda);
    void replay_mix(Tensor& input_ids, Tensor& labels, float ratio);
    float eval_forgetting();
    void set_ewc_lambda(float lambda);
    void set_replay_ratio(float ratio);
    const ContinualStats& stats() const { return stats_; }
    ContinualEngine* engine() { return &engine_; }
private:
    Trainer* trainer_ = nullptr;
    ContinualEngine engine_;
    ContinualEngineConfig cfg_;
    ContinualStats stats_;
    float ewc_lambda_ = 0.01f;
    float replay_ratio_ = 0.25f;
};

struct EWCProbe {
    static float penalty(const ECCState& ecc, const float* weights, size_t n,
                         float lambda);
    static void accumulate_fisher(ECCState& ecc, const float* grads, size_t n,
                                  float alpha);
    static void anchor(ECCState& ecc, const float* weights, size_t n);
};

} // namespace dense_engine
} // namespace quant
