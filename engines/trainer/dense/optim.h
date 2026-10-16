#pragma once
#include "quant/optimizer.h"
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace quant {
namespace dense_engine {

// Full optimizer suite with real update math on host tensors.
// Wraps Optimizer param-groups; every step() touches real grads.
struct OptimStats {
    int64_t steps = 0;
    float last_lr = 0.0f;
    float last_update_norm = 0.0f;
};

// Per-param LR groups: name-prefix match -> scaled LR, real assignment.
struct ParamGroup {
    std::string prefix;
    float lr_scale = 1.0f;
    float wd_scale = 1.0f;
};

class OptimSuite {
public:
    enum class Kind { ADAMW, ADAM, ADAFACTOR, SGD, LION };
    OptimSuite(Kind kind, float lr = 3e-4f, float wd = 1e-2f);
    void add_params(const std::vector<Tensor*>& ps);
    void add_group(const std::vector<Tensor*>& ps, float lr_scale, float wd_scale);
    void step();
    void zero_grad();
    void set_lr(float lr);
    float get_lr() const;
    void set_schedule(AdamW::Schedule s, int warmup, int total);
    void scheduler_step(int step);
    float clip(float max_norm);
    const OptimStats& stats() const { return stats_; }
    Kind kind() const { return kind_; }
    // Real math probes.
    static float param_norm(const std::vector<Tensor*>& ps);
    static float update_norm(const std::vector<Tensor>& before,
                             const std::vector<Tensor>& after);
    static void snapshot(const std::vector<Tensor*>& ps, std::vector<Tensor>& out);
private:
    Kind kind_;
    AdamW adamw_;
    Adam adam_;
    Adafactor adafactor_;
    SGD sgd_;
    Lion lion_;
    Optimizer* active_ = nullptr;
    OptimStats stats_;
    std::vector<ParamGroup> groups_;
    std::vector<std::vector<Tensor*>> group_params_;
};

} // namespace dense_engine
} // namespace quant
