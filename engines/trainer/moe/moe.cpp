#include "moe.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace quant {
namespace moe_engine {

int64_t MoETrainerEngine::now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

MoETrainerEngine::MoETrainerEngine(MoEModel* model, Tokenizer* tok, ExpertParallel* ep)
    : model_(model), tok_(tok), trainer_(model, tok, ep) {
    if (!model_ || !tok_) throw std::runtime_error("[moe-engine] needs MoEModel*+Tokenizer*");
    t0_us_ = now_us();
}

void MoETrainerEngine::compile(const MoETrainConfig& cfg) { cfg_ = cfg; trainer_.compile(cfg_); }
void MoETrainerEngine::compile(AdamW* opt, const MoETrainConfig& cfg) { cfg_ = cfg; trainer_.compile(opt, cfg_); }

float MoETrainerEngine::train_step(const Tensor& input_ids, const Tensor& labels) {
    float loss = trainer_.train_step(input_ids, labels);
    stats_.steps++;
    stats_.tokens += input_ids.numel();
    stats_.last_loss = loss;
    stats_.expert_util = trainer_.compute_expert_utilization();
    stats_.aux_loss = trainer_.metrics().aux_loss;
    int64_t dt = now_us() - t0_us_;
    if (dt > 0) stats_.tok_per_sec = (double)stats_.tokens * 1e6 / (double)dt;
    return loss;
}

float MoETrainerEngine::micro_step(const Tensor& input_ids, const Tensor& labels, float loss_scale) {
    return trainer_.micro_step(input_ids, labels, loss_scale);
}

void MoETrainerEngine::fit(DataLoader& dl, const MoETrainConfig& cfg, DataLoader* val_dl) {
    cfg_ = cfg;
    trainer_.fit(dl, cfg_, val_dl);
}

float MoETrainerEngine::eval_loss(DataLoader& val_dl, int64_t max_batches) {
    return trainer_.eval_loss(val_dl, max_batches);
}

void MoETrainerEngine::save_checkpoint(const std::string& path) { trainer_.save_checkpoint(path); }
void MoETrainerEngine::load_checkpoint(const std::string& path) { trainer_.load_checkpoint(path); }
float MoETrainerEngine::expert_utilization() const { return trainer_.compute_expert_utilization(); }
const MoEMetrics& MoETrainerEngine::metrics() const { return trainer_.metrics(); }

RouterOut route_topk(const Tensor& router_logits, int64_t top_k, float load_balance_coef) {
    if (router_logits.rank() != 2) throw std::runtime_error("[moe-engine] route_topk needs 2-D logits");
    int64_t rows = router_logits.dim(0), experts = router_logits.dim(1);
    if (top_k <= 0 || top_k > experts) throw std::runtime_error("[moe-engine] bad top_k");
    const float* lg = router_logits.data<float>();
    RouterOut out;
    out.weights = Tensor::zeros({rows, top_k});
    out.indices = Tensor::zeros({rows, top_k}, DType::I32);
    float* w = out.weights.data<float>();
    int* ix = out.indices.data<int>();
    std::vector<float> importance((size_t)experts, 0.0f);
    for (int64_t r = 0; r < rows; r++) {
        float m = lg[r * experts];
        for (int64_t e = 1; e < experts; e++) m = std::max(m, lg[r * experts + e]);
        float sum = 0.0f;
        std::vector<float> probs((size_t)experts);
        for (int64_t e = 0; e < experts; e++) {
            probs[(size_t)e] = std::exp(lg[r * experts + e] - m);
            sum += probs[(size_t)e];
        }
        for (int64_t e = 0; e < experts; e++) probs[(size_t)e] /= sum;
        std::vector<int64_t> order((size_t)experts);
        for (int64_t e = 0; e < experts; e++) order[(size_t)e] = e;
        std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                          [&](int64_t a, int64_t b) { return probs[(size_t)a] > probs[(size_t)b]; });
        float wsum = 0.0f;
        for (int64_t k = 0; k < top_k; k++) wsum += probs[(size_t)order[(size_t)k]];
        for (int64_t k = 0; k < top_k; k++) {
            int64_t e = order[(size_t)k];
            ix[r * top_k + k] = (int)e;
            w[r * top_k + k] = probs[(size_t)e] / wsum;
            importance[(size_t)e] += probs[(size_t)e];
        }
    }
    float mean = (float)rows / (float)experts;
    float cv = 0.0f;
    for (float v : importance) { float d = v - mean; cv += d * d; }
    cv = std::sqrt(cv / (float)experts) / (mean > 0 ? mean : 1.0f);
    out.aux_loss = load_balance_coef * cv * cv;
    return out;
}

CapacityPlan plan_capacity(int64_t tokens, int64_t experts, int64_t top_k,
                           float capacity_factor) {
    CapacityPlan p;
    p.experts = experts;
    p.top_k = top_k;
    p.capacity_factor = capacity_factor;
    p.tokens = tokens;
    if (experts <= 0) throw std::runtime_error("[moe-engine] bad experts");
    double avg = (double)tokens * (double)top_k / (double)experts;
    p.slots_per_expert = (int64_t)std::ceil(avg * (double)capacity_factor);
    int64_t total_slots = p.slots_per_expert * experts;
    int64_t assigned = tokens * top_k;
    p.dropped = assigned > total_slots ? assigned - total_slots : 0;
    return p;
}

float z_loss(const Tensor& router_logits) {
    if (router_logits.rank() != 2) throw std::runtime_error("[moe-engine] z_loss needs 2-D");
    int64_t rows = router_logits.dim(0), experts = router_logits.dim(1);
    const float* lg = router_logits.data<float>();
    double sum = 0.0;
    for (int64_t r = 0; r < rows; r++) {
        float m = lg[r * experts];
        for (int64_t e = 1; e < experts; e++) m = std::max(m, lg[r * experts + e]);
        double s = 0.0;
        for (int64_t e = 0; e < experts; e++) s += std::exp((double)lg[r * experts + e] - (double)m);
        double lse = (double)m + std::log(s);
        sum += lse * lse;
    }
    return rows > 0 ? (float)(sum / (double)rows) : 0.0f;
}

EPMap build_ep_map(int64_t experts, int ranks) {
    if (ranks <= 0) throw std::runtime_error("[moe-engine] bad ranks");
    EPMap m;
    m.ranks = ranks;
    m.expert_rank.resize((size_t)experts);
    m.rank_counts.assign((size_t)ranks, 0);
    for (int64_t e = 0; e < experts; e++) {
        int r = (int)(e % ranks);
        m.expert_rank[(size_t)e] = r;
        m.rank_counts[(size_t)r]++;
    }
    return m;
}

BalanceOut balance_stats(const std::vector<float>& importance, int64_t rows) {
    BalanceOut o;
    if (importance.empty() || rows <= 0) return o;
    float sum = 0.0f, mn = importance[0], mx = importance[0];
    for (float v : importance) { sum += v; mn = std::min(mn, v); mx = std::max(mx, v); }
    float mean = sum / (float)importance.size();
    double var = 0.0;
    for (float v : importance) { double d = (double)v - mean; var += d * d; }
    o.cv = (float)(std::sqrt(var / importance.size()) / (mean > 0 ? mean : 1.0f));
    o.min_share = mn / (float)rows;
    o.max_share = mx / (float)rows;
    return o;
}

MoEPipeline::MoEPipeline(MoETrainerEngine* engine, Tokenizer* tok,
                         const std::string& train_path, const std::string& val_path)
    : engine_(engine), tok_(tok), train_path_(train_path), val_path_(val_path) {
    if (!engine_ || !tok_) throw std::runtime_error("[moe-engine] MoEPipeline needs engine+tok");
}

void MoEPipeline::configure(const MoETrainConfig& cfg) {
    cfg_ = cfg;
    engine_->compile(cfg_);
}

void MoEPipeline::set_checkpoint_dir(const std::string& dir) { ckpt_dir_ = dir; }
void MoEPipeline::set_log_file(const std::string& path) {
    log_path_ = path;
    log_.open(path, std::ios::app);
    if (!log_) throw std::runtime_error("[moe-engine] cannot open log: " + path);
}

float MoEPipeline::train_dl(DataLoader& dl, int epoch) {
    dl.reset();
    Tensor in, lb;
    double sum = 0.0;
    int64_t n = 0;
    int64_t cap = cfg_.train_steps > 0 ? cfg_.train_steps : 1000000;
    while (n < cap && dl.next_batch(in, lb)) {
        float l = engine_->train_step(in, lb);
        sum += l;
        n++;
        if ((n % (cfg_.log_interval > 0 ? cfg_.log_interval : 10)) == 0 && log_.is_open())
            log_ << "epoch " << epoch << " step " << n << " loss " << l << "\n";
    }
    return n > 0 ? (float)(sum / (double)n) : 0.0f;
}

float MoEPipeline::eval_dl(DataLoader& dl) {
    return engine_->eval_loss(dl, 20);
}

float MoEPipeline::run_epoch(int epoch) {
    DataLoader tdl(tok_, train_path_, cfg_.batch_size, cfg_.seq_length);
    float tr = train_dl(tdl, epoch);
    float vl = tr;
    if (!val_path_.empty()) {
        DataLoader val_dl(tok_, val_path_, cfg_.batch_size, cfg_.seq_length);
        vl = eval_dl(val_dl);
    }
    if (vl < best_val_) { best_val_ = vl; no_improve_ = 0; } else { no_improve_++; }
    if (!ckpt_dir_.empty())
        engine_->save_checkpoint(ckpt_dir_ + "/moe_epoch" + std::to_string(epoch) + ".quant");
    return vl;
}

float MoEPipeline::run(int epochs) {
    int ep = epochs > 0 ? epochs : cfg_.num_epochs;
    float last = 0.0f;
    for (int e = 0; e < ep; e++) {
        last = run_epoch(e);
        if (no_improve_ >= patience_) break;
    }
    return last;
}

} // namespace moe_engine
} // namespace quant
