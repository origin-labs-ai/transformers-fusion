#pragma once
#include "quant/finetune.h"
#include "quant/trainer.h"
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace quant {
namespace dense_engine {

struct FinetuneStats {
    int64_t steps = 0;
    int64_t tokens = 0;
    double tok_per_sec = 0.0;
    float last_loss = 0.0f;
    int64_t trainable_blocks = 0;
};

class FinetuneRunner {
public:
    FinetuneRunner(FineTuner* tuner, Trainer* trainer);
    void configure(const FineTuneConfig& cfg);
    float train_step(const Tensor& input_ids, const Tensor& labels);
    void fine_tune(DataLoader& dl);
    std::vector<int> trainable_blocks(const Tensor& grads, float threshold);
    void quant_update(const Tensor& fp32_grad, Tensor& quant_weight, Format fmt);
    const FinetuneStats& stats() const { return stats_; }
private:
    FineTuner* tuner_ = nullptr;
    Trainer* trainer_ = nullptr;
    FineTuneConfig cfg_;
    FinetuneStats stats_;
    int64_t t0_us_ = 0;
    static int64_t now_us();
};

struct LoRAConfig {
    int64_t rank = 8;
    float alpha = 16.0f;
    float dropout = 0.0f;
    std::vector<std::string> targets;
};

struct LoRALayer {
    Tensor A;
    Tensor B;
    float scale = 1.0f;
};

class LoRAEngine {
public:
    explicit LoRAEngine(const LoRAConfig& cfg = {});
    void attach(const std::string& name, int64_t in, int64_t out);
    Tensor forward(const std::string& name, const Tensor& x);
    void merge_into(Tensor& weight, const std::string& name, float merge_scale = 1.0f);
    void zero(const std::string& name);
    int64_t trainable_params() const;
    const LoRAConfig& config() const { return cfg_; }
private:
    LoRAConfig cfg_;
    std::vector<std::string> names_;
    std::vector<LoRALayer> layers_;
    int find(const std::string& name) const;
    static void xavier(Tensor& t, float gain = 1.0f);
};

} // namespace dense_engine
} // namespace quant
