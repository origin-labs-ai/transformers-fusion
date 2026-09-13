#pragma once
#include "quant/moe_model.h"
#include "quant/trainer.h"
#include "quant/distributed.h"
#include "quant/moe_variants.h"
#include <vector>
#include <string>
#include <functional>
#include <memory>

namespace quant {

struct MoETrainConfig {
    int64_t batch_size = 8;
    int64_t seq_length = 512;
    int num_epochs = 3;
    int train_steps = 10000;
    float learning_rate = 3e-4f;
    float weight_decay = 1e-2f;
    int warmup_steps = 100;
    int log_interval = 10;
    int save_interval = 1000;
    int val_interval = 500;
    std::string output_path = "moe_model.quant";
    float max_grad_norm = 1.0f;
    int gradient_accumulation_steps = 1;
    AdamW::Schedule schedule = AdamW::Schedule::WARMUP_COSINE;
    bool mixed_precision = false;
    float loss_scale = 128.0f;
    float load_balance_coef = 0.01f;
    float z_loss_coef = 0.001f;
    float aux_loss_coef = 0.01f;
    float expert_dropout = 0.0f;
    int capacity_factor = 2;
    bool use_expert_parallel = false;
    int num_expert_parallel_ranks = 1;
    int expert_parallel_rank = 0;
};

struct MoEMetrics {
    float loss = 0.0f;
    float load_balance_loss = 0.0f;
    float z_loss = 0.0f;
    float aux_loss = 0.0f;
    float total_loss = 0.0f;
    float perplexity = 0.0f;
    float grad_norm = 0.0f;
    float learning_rate = 0.0f;
    float expert_utilization = 0.0f;
    float tokens_per_sec = 0.0f;
    int64_t tokens_processed = 0;
    // L057: router overflow stats — total dropped tokens across layers in the
    // last micro_step (capacity/overflow accounting, replaces the previous
    // misuse of tokens_per_sec as a drop counter).
    int64_t tokens_dropped_total = 0;
    int num_experts_used = 0;
    int step = 0;
    int epoch = 0;
    int64_t active_params = 0;
    int64_t total_params = 0;
};

class MoETrainer {
public:
    MoETrainer(MoEModel* model, Tokenizer* tokenizer,
               ExpertParallel* expert_parallel = nullptr);

    void compile(Optimizer* optimizer, const MoETrainConfig& cfg = MoETrainConfig{});
    void compile(AdamW* optimizer, const MoETrainConfig& cfg = MoETrainConfig{});
    void compile(Adafactor* optimizer, const MoETrainConfig& cfg = MoETrainConfig{});
    void compile(const MoETrainConfig& cfg = MoETrainConfig{});
    void fit(DataLoader& train_dl, const MoETrainConfig& cfg,
             DataLoader* val_dl = nullptr);

    float train_step(const Tensor& input_ids, const Tensor& labels);
    float train_step(DataLoader& loader, const Tensor& first_input, const Tensor& first_labels);
    float micro_step(const Tensor& input_ids, const Tensor& labels,
                     float loss_scale = 1.0f);
    float eval_loss(DataLoader& val_dl, int64_t max_batches = 20);
    float clip_gradients(float max_norm);
    void unscale_gradients(float scale);
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);

    using LogCallback = std::function<void(const MoEMetrics&)>;
    using EpochCallback = std::function<void(int epoch, const MoEMetrics&)>;
    using StepCallback = std::function<void(int step, const MoEMetrics&)>;
    void set_log_callback(LogCallback cb);
    void set_epoch_callback(EpochCallback cb);
    void set_step_callback(StepCallback cb);

    const MoEMetrics& metrics() const { return metrics_; }
    const std::vector<Tensor*>& get_model_params() const;
    float compute_expert_utilization() const;
    float compute_aux_loss(const Tensor& router_logits,
                           const Tensor& expert_indices) const;

    void ema_init(float decay = 0.999f);
    void ema_step();
    void ema_apply();
    void ema_swap();

    void init_mixed_precision();
    void mp_quantize_forward();
    void mp_restore_master();
    bool mixed_precision_active() const { return mp_active_; }

    void set_gradient_accumulation_steps(int steps) { grad_accum_steps_ = steps; }
    int gradient_accumulation_steps() const { return grad_accum_steps_; }

    void set_verbose(bool v) { verbose_ = v; }

private:
    MoEModel* model_;
    Tokenizer* tokenizer_;
    ExpertParallel* expert_parallel_;
    Optimizer* optimizer_ = nullptr;
    std::unique_ptr<Optimizer> default_opt_;
    MoETrainConfig config_;
    MoEMetrics metrics_;
    LogCallback log_cb_;
    EpochCallback epoch_cb_;
    StepCallback step_cb_;

    int step_ = 0;
    float loss_scale_ = 1.0f;
    int grad_accum_steps_ = 1;
    bool owns_expert_parallel_ = false;
    bool verbose_ = true;

    std::vector<Tensor*> model_params_;
    std::vector<Tensor> ema_params_;
    float ema_decay_ = 0.999f;
    bool ema_enabled_ = false;

    std::vector<Tensor> mp_master_;
    bool mp_active_ = false;

    void collect_params();
    Tensor compute_loss(const Tensor& logits, const Tensor& labels);
    void log_metrics();
    void accumulate_expert_stats(const moe::MoEOutput& moe_out);
    void reset_expert_stats();
};

class MoETrainingPipeline {
public:
    MoETrainingPipeline(MoEModel* model, Tokenizer* tokenizer,
                        const std::string& train_path,
                        const std::string& val_path = "",
                        ExpertParallel* ep = nullptr);

    void configure(const MoETrainConfig& cfg);
    void run(int epochs = -1);
    void set_log_file(const std::string& path);
    void set_checkpoint_dir(const std::string& dir);
    void set_distributed(int world_size, int rank);

    MoEModel* model() { return model_; }
    MoETrainer* trainer() { return trainer_.get(); }
    const MoEMetrics& last_metrics() const;
    std::string status_report() const;

private:
    MoEModel* model_;
    Tokenizer* tokenizer_;
    std::string train_path_;
    std::string val_path_;
    MoETrainConfig config_;
    std::unique_ptr<MoETrainer> trainer_;
    std::unique_ptr<Optimizer> optimizer_;
    std::unique_ptr<DataLoader> train_dl_;
    std::unique_ptr<DataLoader> val_dl_;
    std::unique_ptr<ExpertParallel> ep_;
    std::string log_path_;
    std::string checkpoint_dir_;
    std::ofstream log_stream_;
    int world_size_ = 1;
    int rank_ = 0;

    void init_data();
    void run_epoch(int epoch);
    void save_checkpoint(int epoch, float loss);
    bool should_stop_early(const MoEMetrics& m, int patience);
    int early_stop_patience_ = 3;
    float best_val_loss_ = 1e10f;
    int epochs_no_improve_ = 0;
};

} // namespace quant
