#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include "quant/types.h"
#include "quant/model.h"
#include "quant/trainer.h"
#include "quant/native_weight.h"

namespace quant {
namespace native {

struct NativeTrainConfig {
    // Architecture
    size_t block_size = 128;     // B=128 (default)
    size_t warmup_steps = 100;   // FP32 steps before transitioning to native QUANT
    
    // CID allocation fractions
    float frac_quant8 = 0.01f;     // top 1% by sensitivity
    float frac_quant = 0.95f;  // middle 95%
    // remainder (4%) → Q1
    
    // Learning rates
    float lr_scale = 1e-4f;      // learning rate for continuous scale updates
    float lr_weight = 1e-4f;     // learning rate for virtual weight (index updates)
    float lr_decay = 0.0f;       // LR decay per step (0 = no decay)
    float weight_decay = 1e-2f;  // weight decay applied to scales
    
    // Dead zone (Theorem 5d.3)
    float dead_zone_scale = 1.0f; // multiplier on theoretical dead zone radius
    
    // Optimizer
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    
    // Training
    size_t batch_size = 8;
    size_t seq_length = 512;
    size_t max_steps = 10000;
    int log_interval = 10;
    int eval_interval = 100;
    int save_interval = 1000;
    std::string output_dir = "./checkpoints";
    
    // Sensitivity estimation
    size_t sens_window = 10;     // moving average window for gradient norms
};

// Training metrics specific to native QUANT
struct NativeTrainMetrics {
    double loss = 0.0;
    double scale_lr = 0.0;
    double weight_lr = 0.0;
    double grad_norm = 0.0;
    double frozen_fraction = 0.0;  // fraction of indices in dead zone
    size_t quant8_count = 0;
    size_t quant1_count = 0;
    size_t quant_count = 0;
    double avg_scale = 1.0;
};

// Native QUANT Trainer — no FP32 master weights
// Implements two-timescale SGD from Theorem 5d.3
class NativeQUANTTrainer {
public:
    NativeQUANTTrainer(DenseModel* model, const NativeTrainConfig& cfg = NativeTrainConfig());
    ~NativeQUANTTrainer() = default;
    
    // Main training entry point
    void train(const std::vector<std::vector<float>>& train_data,
               const std::vector<std::vector<float>>& eval_data = {});
    
    // Run a single training step
    // input: [batch_size, seq_length] token IDs
    // target: [batch_size, seq_length] token IDs
    NativeTrainMetrics train_step(const float* input, const float* target,
                                   size_t batch_size, size_t seq_len);
    
    // Evaluation (no QUANT updates)
    double evaluate(const std::vector<std::vector<float>>& eval_data);
    
    // Save native QUANT checkpoint
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);

    // Phase 0: FP32 warmup + sensitivity collection (exposed for testing)
    void warmup_phase(const std::vector<std::vector<float>>& data);

    // Access
    NativeQWeightStore& weight_store() { return *weight_store_; }
    const NativeTrainConfig& config() const { return cfg_; }
    
private:
    // Phase 1: CID format allocation
    void allocate_formats();
    
    // Copy dequantized weights to model's FP32 tensors
    void push_weights_to_model();
    
    // Pull gradients from model's FP32 tensors to gradient buffer
    void pull_gradients_from_model();
    
    // Apply two-timescale SGD (Theorem 5d.3)
    void apply_two_timescale_sgd();
    
    // Update sensitivity moving average from current gradients
    void update_sensitivity();
    
    DenseModel* model_;
    NativeTrainConfig cfg_;
    std::unique_ptr<NativeQWeightStore> weight_store_;
    
    // Gradient buffer (accumulated from model's FP32 gradients)
    std::unique_ptr<float[]> grad_buffer_;
    
    // Sensitivity buffer (moving average of |grad| per weight)
    std::unique_ptr<float[]> sensitivity_;
    size_t sens_step_;
    
    // Parameter management
    std::vector<Tensor*> model_params_;
    size_t total_params_;
    
    // Optimizer state for scale updates (Adam-style)
    std::unique_ptr<float[]> scale_m_;
    std::unique_ptr<float[]> scale_v_;
    size_t step_;
    
    // Temporary FP32 buffer for dequantized weights
    std::unique_ptr<float[]> temp_weight_buffer_;
    
    // Eval loss tracking
    double best_eval_loss_;
};

} // namespace native
} // namespace quant
