#include "unified.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace quant {
namespace dense_engine {

int64_t UnifiedEntry::now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

UnifiedEntry::UnifiedEntry(Model* model, Tokenizer* tok)
    : model_(model), tok_(tok), inner_(model, tok) {
    if (!model_ || !tok_) throw std::runtime_error("[unified] needs Model*+Tokenizer*");
    t0_us_ = now_us();
}

bool UnifiedEntry::configure(const UnifiedTrainArgs& args, std::string* err) {
    bool ok = inner_.configure(args, err);
    return ok;
}

bool UnifiedEntry::run(std::string* err) { return inner_.run(err); }

bool UnifiedEntry::train_step_once(const Tensor& input_ids, const Tensor& labels,
                                   float* loss_out, std::string* err) {
    bool ok = inner_.train_step_once(input_ids, labels, loss_out, err);
    if (ok) {
        stats_.steps++;
        stats_.tokens += input_ids.numel();
        if (loss_out) stats_.last_loss = *loss_out;
        int64_t dt = now_us() - t0_us_;
        if (dt > 0) stats_.tok_per_sec = (double)stats_.tokens * 1e6 / (double)dt;
    }
    return ok;
}

void UnifiedEntry::save_checkpoint(const std::string& path) { inner_.save_checkpoint(path); }
void UnifiedEntry::load_checkpoint(const std::string& path) { inner_.load_checkpoint(path); }
const TrainMetrics& UnifiedEntry::metrics() const { return inner_.metrics(); }

const char* UnifiedEntry::kind_name(UnifiedTrainerKind k) {
    return unified_trainer_kind_name(k);
}

bool UnifiedEntry::parse_kind(const std::string& s, UnifiedTrainerKind* out) {
    return unified_trainer_parse_kind(s, out);
}

RewardStats normalize_rewards(const Tensor& rewards) {
    int64_t n = rewards.numel();
    if (n <= 0) throw std::runtime_error("[reward] empty rewards");
    const float* r = rewards.data<float>();
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) sum += r[i];
    double mean = sum / (double)n;
    double var = 0.0;
    for (int64_t i = 0; i < n; i++) { double d = r[i] - mean; var += d * d; }
    RewardStats o;
    o.mean = (float)mean;
    o.std = (float)std::sqrt(var / (double)n + 1e-8);
    return o;
}

float policy_grad_loss(const Tensor& logits, const Tensor& labels,
                       const Tensor& advantages, Tensor& dlogits,
                       int64_t ignore_index) {
    if (logits.rank() != 2) throw std::runtime_error("[reward] logits need 2-D");
    int64_t rows = logits.dim(0), V = logits.dim(1);
    if (labels.numel() != rows || advantages.numel() != rows)
        throw std::runtime_error("[reward] labels/advantages rows mismatch");
    const float* lg = logits.data<float>();
    const int* lb = labels.data<int>();
    const float* adv = advantages.data<float>();
    dlogits = Tensor::zeros({rows, V});
    float* dl = dlogits.data<float>();
    double total = 0.0;
    int64_t n = 0;
    for (int64_t r = 0; r < rows; r++) {
        int t = lb[r];
        if (t == ignore_index || t < 0 || t >= V) continue;
        float m = lg[r * V];
        for (int64_t v = 1; v < V; v++) m = std::max(m, lg[r * V + v]);
        double s = 0.0;
        for (int64_t v = 0; v < V; v++) s += std::exp((double)lg[r * V + v] - m);
        double logp = ((double)lg[r * V + t] - m) - std::log(s);
        total += -logp * adv[r];
        for (int64_t v = 0; v < V; v++) {
            double p = std::exp((double)lg[r * V + v] - m) / s;
            dl[r * V + v] = (float)(p * adv[r]);
        }
        dl[r * V + t] -= adv[r];
        n++;
    }
    if (n > 0) for (int64_t i = 0; i < rows * V; i++) dl[i] /= (float)n;
    return n > 0 ? (float)(total / (double)n) : 0.0f;
}

GroupOut grpo_step(const Tensor& logits, const Tensor& labels,
                   const Tensor& rewards, int64_t group,
                   float clip_eps, float kl_coef) {
    if (logits.rank() != 2) throw std::runtime_error("[grpo] logits need 2-D");
    int64_t rows = logits.dim(0);
    if (group <= 0 || rows % group != 0) throw std::runtime_error("[grpo] rows must split by group");
    const float* r = rewards.data<float>();
    Tensor adv = Tensor::zeros({rows});
    float* ad = adv.data<float>();
    double rsum = 0.0;
    for (int64_t i = 0; i < rows; i++) rsum += r[i];
    double rmean = rsum / (double)rows;
    int64_t ng = rows / group;
    for (int64_t g = 0; g < ng; g++) {
        double gs = 0.0;
        for (int64_t k = 0; k < group; k++) gs += r[g * group + k];
        double gb = gs / (double)group;
        for (int64_t k = 0; k < group; k++) ad[g * group + k] = (float)(r[g * group + k] - gb);
    }
    Tensor dl;
    float pg = policy_grad_loss(logits, labels, adv, dl);
    double kl = 0.0;
    for (int64_t i = 0; i < rows; i++) {
        double d = (double)r[i] - rmean;
        kl += d * d;
    }
    kl = kl / (double)rows;
    GroupOut o;
    o.loss = pg + kl_coef * (float)kl;
    o.mean_reward = (float)rmean;
    (void)clip_eps;
    return o;
}

} // namespace dense_engine
} // namespace quant
