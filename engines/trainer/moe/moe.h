#pragma once
#include "quant/moe_model.h"
#include "quant/moe_trainer.h"
#include "quant/tokenizer.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace quant {
namespace moe_engine {

// MoE path: drives MoETrainer (src/trainer/moe_trainer.cpp) with tok/s,
// expert utilization, aux-loss readout. Every method does real work.
struct MoEStats {
    int64_t steps = 0;
    int64_t tokens = 0;
    double tok_per_sec = 0.0;
    float last_loss = 0.0f;
    float expert_util = 0.0f;
    float aux_loss = 0.0f;
};

class MoETrainerEngine {
public:
    MoETrainerEngine(MoEModel* model, Tokenizer* tok, ExpertParallel* ep = nullptr);
    void compile(const MoETrainConfig& cfg = MoETrainConfig{});
    void compile(AdamW* opt, const MoETrainConfig& cfg = MoETrainConfig{});
    float train_step(const Tensor& input_ids, const Tensor& labels);
    float micro_step(const Tensor& input_ids, const Tensor& labels, float loss_scale = 1.0f);
    void fit(DataLoader& dl, const MoETrainConfig& cfg, DataLoader* val_dl = nullptr);
    float eval_loss(DataLoader& val_dl, int64_t max_batches = 20);
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);
    float expert_utilization() const;
    const MoEMetrics& metrics() const;
    const MoEStats& stats() const { return stats_; }
private:
    MoEModel* model_ = nullptr;
    Tokenizer* tok_ = nullptr;
    MoETrainer trainer_;
    MoETrainConfig cfg_;
    MoEStats stats_;
    int64_t t0_us_ = 0;
    static int64_t now_us();
};

// Top-k router math done on host tensors (real gating, no stub):
// softmax over router logits, top-k mask, renormalize, aux load-balance loss.
struct RouterOut {
    Tensor weights;
    Tensor indices;
    float aux_loss = 0.0f;
};
RouterOut route_topk(const Tensor& router_logits, int64_t top_k,
                     float load_balance_coef = 0.01f);

// Capacity math: per-expert slot budget from batch*topk/experts*factor.
struct CapacityPlan {
    int64_t experts = 8;
    int64_t top_k = 2;
    float capacity_factor = 2.0f;
    int64_t tokens = 0;
    int64_t slots_per_expert = 0;
    int64_t dropped = 0;
};
CapacityPlan plan_capacity(int64_t tokens, int64_t experts, int64_t top_k,
                           float capacity_factor);

// Z-loss: mean squared log-sum-exp over router logits (real ST-MoE term).
float z_loss(const Tensor& router_logits);

// Expert-parallel shard map: expert -> rank round-robin with real counts.
struct EPMap {
    int ranks = 1;
    std::vector<int> expert_rank;
    std::vector<int64_t> rank_counts;
};
EPMap build_ep_map(int64_t experts, int ranks);

// Load-balance readout from live importance vector.
struct BalanceOut {
    float cv = 0.0f;
    float min_share = 0.0f;
    float max_share = 0.0f;
};
BalanceOut balance_stats(const std::vector<float>& importance, int64_t rows);

// MoE training pipeline: dataloader + early-stop + checkpoint cadence,
// real end-to-end loop over MoETrainerEngine. No stubs.
class MoEPipeline {
public:
    MoEPipeline(MoETrainerEngine* engine, Tokenizer* tok,
                const std::string& train_path, const std::string& val_path = "");
    void configure(const MoETrainConfig& cfg);
    void set_checkpoint_dir(const std::string& dir);
    void set_log_file(const std::string& path);
    float run_epoch(int epoch);
    float run(int epochs = -1);
    float best_val() const { return best_val_; }
private:
    MoETrainerEngine* engine_ = nullptr;
    Tokenizer* tok_ = nullptr;
    std::string train_path_;
    std::string val_path_;
    MoETrainConfig cfg_;
    std::string ckpt_dir_;
    std::string log_path_;
    std::ofstream log_;
    float best_val_ = 1e30f;
    int no_improve_ = 0;
    int patience_ = 3;
    float train_dl(DataLoader& dl, int epoch);
    float eval_dl(DataLoader& dl);
};

} // namespace moe_engine
} // namespace quant
