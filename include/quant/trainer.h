#pragma once
#include "quant/types.h"
#include "quant/tensor.h"
#include "quant/model.h"
#include "quant/optimizer.h"
#include "quant/tokenizer.h"
#include "quant/autograd.h"
#include "quant/codebook.h"
#include "quant/random.h"
#include "quant/reward.h"
#include "quant/continual_engine.h"
#include <vector>
#include <string>
#include <functional>
#include <fstream>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <memory>

namespace quant {

struct TrainConfig {
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
    std::string output_path = "model.quant";
    bool use_ste = true;
    float max_grad_norm = 1.0f;
    int gradient_accumulation_steps = 1;
    AdamW::Schedule schedule = AdamW::Schedule::WARMUP_COSINE;
    bool mixed_precision = false;
    float loss_scale = 128.0f;
    int loss_scale_interval = 2000;
    // C17: Label smoothing
    float label_smoothing = 0.0f;
    // C19: R-Drop consistency
    bool use_rdrop = false;
    float rdrop_alpha = 1.0f;
    // C8: Data augmentation
    bool data_augmentation = false;
    float aug_noise_std = 0.01f;
    float aug_mask_prob = 0.0f;
    // C9: Curriculum learning
    bool curriculum = false;
    int curriculum_epochs = 3;
    // Gradient noise injection (Neelakantan et al. ICLR 2016)
    float grad_noise_eta = 0.0f;
    float grad_noise_gamma = 0.55f;
    // QAT (Quantization-Aware Training)
    bool use_qat = false;
    int qat_bits = 8;
    bool qat_symmetric = true;
    bool qat_use_lsq = false;
    float qat_init_scale = 0.1f;
    // MoE Load-Balance auxiliary loss (wired from MoEModel::total_load_balance_loss)
    float moe_load_balance_coef = 0.01f;
    float moe_z_loss_coef = 0.001f;
    // MTP auxiliary loss weight
    float mtp_loss_weight = 0.0f;
};

struct AugmentConfig {
    bool enabled = false;
    float mask_prob = 0.0f;
    float noise_std = 0.01f;
    float replace_prob = 0.0f;
};

class DataLoader {
public:
    DataLoader(Tokenizer* tokenizer, const std::string& data_path,
               int64_t batch_size, int64_t seq_length,
               bool stream_from_disk = false);
    DataLoader(Tokenizer* tokenizer, const std::string& data_path,
               int64_t batch_size, int64_t seq_length,
               bool stream_from_disk, int num_workers,
               int64_t prefetch_capacity, bool use_mmap);
    ~DataLoader();
    
    bool next_batch(Tensor& input_ids, Tensor& labels);
    void shuffle(int epoch = 0);
    void reset();
    int64_t num_batches() const;
    int64_t batch_size() const { return batch_size_; }
    int64_t seq_length() const { return seq_length_; }
    
    // C8: Data augmentation
    void set_augmentation(const AugmentConfig& cfg) { aug_cfg_ = cfg; }
    void apply_augmentation(Tensor& input_ids, Tensor& labels);
    
    // C9: Curriculum learning
    struct CurriculumState {
        int current_epoch = 0;
        int total_epochs = 3;
        float seq_len_multiplier = 0.5f;
        int64_t effective_seq_length() const {
            float t = (float)current_epoch / (float)std::max(total_epochs, 1);
            int64_t min_len = 64;
            return std::max(min_len, (int64_t)(512 * (min_len / 512.0f + t * (1.0f - min_len / 512.0f))));
        }
    };
    void set_curriculum(int total_epochs) { cur_state_.total_epochs = total_epochs; }
    void curriculum_step(int epoch) { cur_state_.current_epoch = epoch; }
    
private:
    Tokenizer* tokenizer_;
    std::vector<int> tokenized_data_;
    int64_t batch_size_;
    int64_t seq_length_;
    int64_t current_pos_ = 0;
    int64_t num_batches_ = 0;
    bool streaming_ = false;
    std::string data_path_;
    std::ifstream file_stream_;
    std::vector<int> stream_chunk_;
    int64_t stream_file_offset_ = 0;
    static constexpr int64_t STREAM_CHUNK_TOKENS = 1024 * 1024;
    
    void tokenize_chunk();
    
    // C3: Memory-mapped I/O
    bool use_mmap_ = false;
    void* mmap_ptr_ = nullptr;
    size_t mmap_size_ = 0;
#ifdef _WIN32
    void* mmap_handle_ = nullptr;
    void* file_handle_ = nullptr;
#else
    int mmap_fd_ = -1;
#endif
    void open_mmap(const std::string& path);
    void close_mmap();
    
    // C2/C16: Multi-worker prefetch
    int num_workers_ = 0;
    int64_t prefetch_capacity_ = 0;
    std::thread prefetch_thread_;
    std::mutex prefetch_mutex_;
    std::queue<std::pair<Tensor, Tensor>> prefetch_queue_;
    std::atomic<bool> prefetch_running_{false};
    void prefetch_worker();
    void start_prefetch();
    void stop_prefetch();
    
    // C8: Data augmentation
    AugmentConfig aug_cfg_;
    
    // C9: Curriculum learning
    CurriculumState cur_state_;
};

// Standalone Exponential Moving Average of model parameters (Task 122 / EMA R-Drop).
class ModelEMA {
public:
    ModelEMA(const std::vector<Tensor*>& params, float decay = 0.999f)
        : params_(params), decay_(decay) {
        for (auto* p : params_) shadow_.push_back(p->clone());
    }

    void update() {
        for (size_t i = 0; i < params_.size(); i++) {
            float* e = shadow_[i].data<float>();
            const float* p = params_[i]->data<float>();
            int64_t n = params_[i]->numel();
            for (int64_t j = 0; j < n; j++)
                e[j] = decay_ * e[j] + (1.0f - decay_) * p[j];
        }
    }

    void apply() const {
        for (size_t i = 0; i < params_.size(); i++)
            std::memcpy(params_[i]->data<float>(), shadow_[i].data<float>(),
                        params_[i]->numel() * sizeof(float));
    }

    void swap() {
        for (size_t i = 0; i < params_.size(); i++) {
            float* p = params_[i]->data<float>();
            float* e = shadow_[i].data<float>();
            int64_t n = params_[i]->numel();
            for (int64_t j = 0; j < n; j++) std::swap(p[j], e[j]);
        }
    }

    float decay() const { return decay_; }
    void set_decay(float d) { decay_ = d; }
    const std::vector<Tensor>& shadow() const { return shadow_; }

private:
    std::vector<Tensor*> params_;
    std::vector<Tensor> shadow_;
    float decay_;
};

// Streaming DataLoader (Task 118): yields batches from a file by tokenizing
// fixed-size chunks on demand, never loading the whole file into memory.
class StreamingDataLoader {
public:
    StreamingDataLoader(Tokenizer* tokenizer, const std::string& data_path,
                       int64_t batch_size, int64_t seq_length);
    bool next_batch(Tensor& input_ids, Tensor& labels);
    void reset();
    int64_t num_batches() const { return num_batches_; }
    int64_t batch_size() const { return batch_size_; }
    int64_t seq_length() const { return seq_length_; }

private:
    Tokenizer* tokenizer_;
    int64_t batch_size_;
    int64_t seq_length_;
    std::ifstream file_;
    std::vector<int> buffer_;      // rolling token buffer
    int64_t num_batches_ = 0;
    int64_t chunk_bytes_ = 1 << 20;  // 1 MiB read per chunk
    bool eof_ = false;
    void fill_buffer();
};

struct TrainMetrics {
    float loss = 0;
    float perplexity = 0;
    float val_loss = 0;
    float val_perplexity = 0;
    float learning_rate = 0;
    float grad_norm = 0;
    int tokens_per_sec = 0;
    int step = 0;
    int epoch = 0;
};

class Trainer {
public:
    using LogCallback = std::function<void(const TrainMetrics&)>;
    using EpochCallback = std::function<void(int epoch, const TrainMetrics&)>;
    using StepCallback = std::function<void(int step, const TrainMetrics&)>;

    Trainer(Model* model, Tokenizer* tokenizer);

    void compile(AdamW* optimizer, const TrainConfig& cfg = TrainConfig{});
    void compile(Adafactor* optimizer, const TrainConfig& cfg = TrainConfig{});
    void compile(const TrainConfig& cfg = TrainConfig{});

    void init_mixed_precision();
    float eval_loss(DataLoader& val_dl, int64_t max_batches);
    void unscale_gradients(float scale);
    void fit(DataLoader& dl, const TrainConfig& cfg, DataLoader* val_dl = nullptr);
    float micro_step(const Tensor& input_ids, const Tensor& labels, float loss_scale);
    float train_step(const Tensor& input_ids, const Tensor& labels);
    float clip_gradients(float max_norm);
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);

    void set_log_callback(LogCallback cb);
    void set_epoch_callback(EpochCallback cb);
    void set_step_callback(StepCallback cb);
    const TrainMetrics& metrics() const;

    float rdrop_loss(const Tensor& input_ids, const Tensor& labels, float alpha);
    Tensor label_smoothing_loss(const Tensor& logits, const Tensor& labels, float smoothing);
    float mtp_loss(const std::vector<Tensor>& mtp_logits, const Tensor& labels);

    void ema_init(float decay);
    void ema_step();
    void ema_apply();
    void ema_swap();
    bool ema_enabled() const { return ema_enabled_; }

    void dynamic_loss_scale(float grad_norm, float max_grad_norm);
    void inject_gradient_noise(int step);

    const std::vector<Tensor*>& get_model_params() const { return model_params_; }

    // QAT hooks: optionally fake-quantize weights in forward graph
    void enable_qat(int bits = 8, bool symmetric = true, bool use_lsq = false, float init_scale = 0.1f);
    void disable_qat();
    bool qat_enabled() const;
    // apply fake-quant to a weight tensor (STE-aware, called inside micro_step if enabled)
    Tensor qat_fake_quantize(const Tensor& weight);
    Tensor qat_fake_quantize_lsq(const Tensor& weight, Tensor& scale_param);

    // Continual learning: EWC + Compressed replay
    void enable_continual(const ContinualEngineConfig& cfg = {});
    void disable_continual();
    bool continual_enabled() const { return continual_enabled_; }
    ContinualEngine* continual_engine() { return continual_engine_.get(); }
    void on_task_boundary(uint32_t new_task_id,
                          const float* eval_inputs = nullptr,
                          const float* eval_targets = nullptr,
                          size_t eval_count = 0);
    void set_ewc_lambda(float lambda);
    void set_replay_ratio(float ratio);

private:
    Model* model_ = nullptr;
    Tokenizer* tokenizer_ = nullptr;
    Optimizer* optimizer_ = nullptr;
    std::unique_ptr<Optimizer> default_opt_;
    std::vector<Tensor*> model_params_;
    std::vector<Codebook*> codebooks_;
    std::vector<Tensor> ema_params_;
    TrainMetrics metrics_;
    LogCallback log_cb_;
    StepCallback step_cb_;
    EpochCallback epoch_cb_;
    RNG grad_noise_rng_;
    int step_ = 0;
    float loss_scale_ = 1.0f;
    int loss_scale_interval_ = 2000;
    int steps_since_scale_update_ = 0;
    float grad_noise_eta_ = 0.0f;
    float grad_noise_gamma_ = 0.55f;
    float label_smoothing_ = 0.0f;
    float ema_decay_ = 0.999f;
    bool ema_enabled_ = false;
    // L051: MTP auxiliary loss weight (from TrainConfig::mtp_loss_weight).
    // >0 with a DenseModel carrying mtp_heads wires the MTP term into
    // micro_step()'s graph. 0 = legacy single-head behavior.
    float mtp_loss_weight_ = 0.0f;
    // QAT state
    bool qat_enabled_ = false;
    int qat_bits_ = 8;
    bool qat_symmetric_ = true;
    bool qat_use_lsq_ = false;
    float qat_init_scale_ = 0.1f;
    std::vector<Tensor> qat_scales_; // per-param scale (LSQ trainable)
    // Continual state
    bool continual_enabled_ = false;
    std::unique_ptr<ContinualEngine> continual_engine_;
    ContinualEngineConfig continual_cfg_;
    float ewc_lambda_ = 1.0f;
    float replay_ratio_ = 0.25f;
    uint32_t current_task_id_ = 0;
};

// ===========================================================================
// PPO Trainer — Proximal Policy Optimization (native C++ implementation)
// ===========================================================================
class PPOTrainer {
public:
    PPOTrainer(Model* policy, Model* ref_model,
               float clip_epsilon = 0.2f,
               float value_coef = 0.5f,
               float entropy_coef = 0.01f,
               float gamma = 0.99f,
               float gae_lambda = 0.95f,
               float kl_target = 0.02f);

    Tensor critic_forward(const Tensor& hidden);
    Tensor compute_log_probs(const Tensor& logits, const Tensor& ids);
    Tensor compute_gae(const Tensor& rewards, const Tensor& values,
                       const Tensor& dones);
    float compute_kl_divergence(const Tensor& logits_a, const Tensor& logits_b);

    void train_step(const Tensor& states, const Tensor& actions,
                    const Tensor& old_logprobs, const Tensor& advantages,
                    const Tensor& returns);

    std::vector<Tensor*> critic_parameters();
    void set_log_callback(std::function<void(float pl, float vl, float ent, float kl)> cb) {
        log_cb_ = cb;
    }
    void set_optimizer(Optimizer* opt) { optimizer_ = opt; }
    Optimizer* optimizer() const { return optimizer_; }
    float kl_alpha() const { return kl_alpha_; }
    float last_kl() const { return last_kl_; }

private:
    Model* policy_;
    Model* ref_model_;
    Optimizer* optimizer_ = nullptr;
    float clip_epsilon_;
    float value_coef_;
    float entropy_coef_;
    float gamma_;
    float gae_lambda_;
    float kl_target_;
    float kl_alpha_;
    float last_kl_ = 0.0f;

    Tensor v_fc1_weight_, v_fc1_bias_;
    Tensor v_fc2_weight_, v_fc2_bias_;

    std::function<void(float, float, float, float)> log_cb_;
};

// ===========================================================================
// DPO Trainer — Direct Preference Optimization
// ===========================================================================
class DPOTrainer {
public:
    DPOTrainer(Model* policy, Model* ref_model, float beta = 0.1f,
               Optimizer* optimizer = nullptr);

    float train_step(const Tensor& chosen_logits, const Tensor& rejected_logits,
                     const Tensor& chosen_ids, const Tensor& rejected_ids);

    void set_log_callback(std::function<void(float loss, float kl)> cb) {
        log_cb_ = cb;
    }
    float last_loss() const { return last_loss_; }
    float beta() const { return beta_; }
    void set_beta(float b) { beta_ = b; }

private:
    Model* policy_;
    Model* ref_model_;
    float beta_;
    Optimizer* optimizer_;
    float last_loss_ = 0.0f;

    float compute_log_probs(const Tensor& logits, const Tensor& ids, Tensor* out_probs = nullptr);
    std::function<void(float, float)> log_cb_;
};

// ===========================================================================
// GRPO Trainer — Group Relative Policy Optimization
// ===========================================================================
class GRPOTrainer {
public:
    GRPOTrainer(Model* policy, Tokenizer* tok, int group_size = 8, float beta = 0.04f);
    void set_optimizer(Optimizer* opt) { optimizer_ = opt; }
    void set_beta(float b) { beta_ = b; }
    float train_step(const std::string& prompt);
    // tensor-batch variant used by RLHFPipeline
    float train_step(const Tensor& input_ids, const Tensor& labels, const Tensor& rewards);
    float last_loss() const { return last_loss_; }
    int group_size() const { return group_size_; }
private:
    Model* model_;
    Tokenizer* tok_;
    int group_size_;
    float beta_;
    Optimizer* optimizer_ = nullptr;
    float last_loss_ = 0.0f;
    Tensor compute_log_probs(const Tensor& logits, const Tensor& ids);
};

// ===========================================================================
// RLVR Trainer — Reinforcement Learning with Verifiable Rewards
// ===========================================================================
class RLVRTrainer {
public:
    RLVRTrainer(Model* model, Tokenizer* tok, Optimizer* optimizer = nullptr);
    void set_optimizer(Optimizer* opt) { optimizer_ = opt; }
    float train_step(const std::string& prompt, const std::string& verifiable_answer);
    float train_step(const Tensor& input_ids, const Tensor& labels, float reward);
    float last_reward() const { return last_reward_; }
private:
    Model* model_;
    Tokenizer* tok_;
    Optimizer* optimizer_ = nullptr;
    float last_reward_ = 0.0f;
    bool verify(const std::string& output, const std::string& answer);
};

// ===========================================================================
// RLHF Pipeline — Complete preference-based alignment loop
// ===========================================================================
class RLHFPipeline {
public:
    RLHFPipeline(Model* model, Model* ref_model, Tokenizer* tokenizer,
                 RewardModel* reward_model, Trainer* trainer,
                 Optimizer* policy_opt, Optimizer* rm_opt);

    void generate_comparisons(const std::vector<std::string>& prompts, int max_new_tokens = 64);
    void train_reward_model(const std::vector<Comparison>& data, int epochs = 3, int batch_size = 8);
    void ppo_finetune(const std::vector<std::string>& prompts, int n_ppo_steps = 100,
                      int ppo_batch_size = 8, int max_new_tokens = 64);
    void dpo_finetune(const std::vector<Comparison>& comparisons, int n_steps = 10,
                      int batch_size = 4);
    void grpo_finetune(const std::vector<std::string>& prompts, int n_steps = 20,
                       int max_new_tokens = 32);
    void rlvf_finetune(const std::vector<std::pair<std::string,std::string>>& qa_pairs,
                       int n_steps = 20, int max_new_tokens = 32);

    void run(int n_rounds = 3, int n_prompts = 50, int n_ppo_steps = 200);

    const std::vector<Comparison>& comparisons() const { return comparison_buffer_; }
    const RLHFMetrics& metrics() const { return metrics_; }

    void set_log_callback(std::function<void(const RLHFMetrics&)> cb) { log_cb_ = cb; }
    void set_verbose(bool v) { verbose_ = v; }

    // GAE computation helpers
    Tensor compute_gae(const Tensor& rewards, const Tensor& values,
                       float gamma = 0.99f, float lam = 0.95f);
    Tensor compute_returns_discounted(const Tensor& rewards, float gamma = 0.99f);
    float compute_kl_between_policies(const Tensor& logits_pi, const Tensor& logits_ref,
                                       const Tensor& actions);
    float compute_reward_accuracy(const std::vector<Comparison>& data);
    float train_reward_model_epoch_with_accuracy(
        const std::vector<Comparison>& train_data,
        const std::vector<Comparison>& val_data,
        int epochs = 3, int batch_size = 8,
        const char* split_name = nullptr);

private:
    Model* model_;
    Model* ref_model_;
    Tokenizer* tokenizer_;
    RewardModel* reward_model_;
    Trainer* trainer_;
    Optimizer* policy_opt_;
    Optimizer* rm_opt_;
    std::vector<Comparison> comparison_buffer_;
    RLHFMetrics metrics_;
    bool verbose_ = true;
    std::function<void(const RLHFMetrics&)> log_cb_;

    Tensor extract_hidden(Tensor& logits, int64_t hidden_size);
    Tensor get_reward_for_sequence(Model* model, const std::vector<int>& ids);
};

} // namespace quant

namespace quant {

void collect_dense_params(DenseModel* dm, std::vector<Tensor*>& params);

// ── L073 Unified trainer entry (Phase 16 Wave 7) ───────────────────────────
// Single entry point for all training CLI paths. quant::Trainer remains the
// implementation; engines/trainer/dense::DenseTrainer and FineTuner configure
// through this facade via from_*() translation (no circular link: this header
// does not include engines/* or finetune.h; translation is field-wise).
// New code must use UnifiedTrainer; direct Trainer use is legacy-compat.
enum class UnifiedTrainerKind : uint8_t {
    Dense = 0,
    FineTune = 1,
    MoE = 2,
    Native = 3
};

struct UnifiedTrainArgs {
    UnifiedTrainerKind kind = UnifiedTrainerKind::Dense;
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
    std::string output_path = "model.quant";
    std::string data_path;
    std::string optimizer_name = "adafactor"; // "adafactor" | "adamw"
    bool use_qat = false;
    int qat_bits = 8;
    float moe_load_balance_coef = 0.01f;
    float moe_z_loss_coef = 0.001f;
};

const char* unified_trainer_kind_name(UnifiedTrainerKind k);
bool unified_trainer_parse_kind(const std::string& s, UnifiedTrainerKind* out);
TrainConfig unified_to_train_config(const UnifiedTrainArgs& a);

class UnifiedTrainer {
public:
    UnifiedTrainer(Model* model, Tokenizer* tokenizer);
    ~UnifiedTrainer();

    bool configure(const UnifiedTrainArgs& args, std::string* err_out);
    bool run(std::string* err_out);
    bool train_step_once(const Tensor& input_ids, const Tensor& labels,
                         float* loss_out, std::string* err_out);
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);
    const TrainMetrics& metrics() const;
    const UnifiedTrainArgs& args() const { return args_; }
    Trainer* inner() { return trainer_.get(); }

    using LogCallback = std::function<void(const TrainMetrics&)>;
    void set_log_callback(LogCallback cb);

private:
    Model* model_ = nullptr;
    Tokenizer* tokenizer_ = nullptr;
    UnifiedTrainArgs args_;
    TrainConfig cfg_;
    std::unique_ptr<Trainer> trainer_;
    std::unique_ptr<Optimizer> owned_opt_;
    Optimizer* opt_ = nullptr;
    LogCallback log_cb_;
    bool configured_ = false;
};

} // namespace quant
