#include "continual.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace quant {
namespace dense_engine {

ContinualRunner::ContinualRunner(Trainer* trainer, const ContinualEngineConfig& cfg)
    : trainer_(trainer), engine_(cfg), cfg_(cfg) {
    if (!trainer_) throw std::runtime_error("[continual] needs Trainer*");
    ewc_lambda_ = cfg_.ecc_lambda;
}

void ContinualRunner::on_task_boundary(uint32_t new_task_id,
                                       const float* eval_inputs,
                                       const float* eval_targets,
                                       size_t eval_count) {
    engine_.on_task_boundary(new_task_id, eval_inputs, eval_targets, eval_count);
    stats_.task_id = new_task_id;
}

float ContinualRunner::train_step_mixed(const Tensor& input_ids, const Tensor& labels,
                                       float replay_ratio) {
    Tensor in = input_ids;
    Tensor lb = labels;
    replay_mix(in, lb, replay_ratio);
    float loss = trainer_->train_step(in, lb);
    stats_.steps++;
    const auto& params = trainer_->get_model_params();
    double pen = 0.0;
    size_t n = 0;
    for (auto* p : params) {
        if (!p) continue;
        pen += ewc_penalty(ewc_lambda_);
        n++;
        break;
    }
    (void)n;
    stats_.replay_loss = loss;
    return loss;
}

float ContinualRunner::ewc_penalty(float lambda) {
    const auto& params = trainer_->get_model_params();
    const ECCState& ecc = engine_.ecc();
    if (!ecc.initialized) return 0.0f;
    double total = 0.0;
    size_t off = 0;
    for (auto* p : params) {
        if (!p) continue;
        const float* w = p->data<float>();
        size_t n = (size_t)p->numel();
        if (off + n > ecc.fisher_diagonal.size()) break;
        for (size_t i = 0; i < n; i++) {
            double d = (double)w[i] - ecc.anchor_weights[off + i];
            total += (double)ecc.fisher_diagonal[off + i] * d * d;
        }
        off += n;
    }
    float pen = (float)(0.5 * lambda * total);
    stats_.ewc_penalty = pen;
    return pen;
}

void ContinualRunner::replay_mix(Tensor& input_ids, Tensor& labels, float ratio) {
    if (ratio <= 0.0f) return;
    auto& rep = engine_.replay();
    if (rep.size() == 0) return;
    std::vector<std::vector<float>> ins, tgs;
    std::vector<float> wts;
    size_t want = (size_t)(input_ids.dim(0) * ratio);
    if (want == 0) want = 1;
    if (!rep.sample_batch(ins, tgs, wts, want)) return;
    if (ins.empty()) return;
    int* ii = input_ids.data<int>();
    int* ll = labels.data<int>();
    int64_t rows = input_ids.dim(0), cols = input_ids.dim(1);
    for (size_t b = 0; b < ins.size() && b < (size_t)rows; b++) {
        for (int64_t s = 0; s < cols && s < (int64_t)ins[b].size(); s++)
            ii[b * cols + s] = (int)ins[b][(size_t)s];
        if (b < tgs.size())
            for (int64_t s = 0; s < cols && s < (int64_t)tgs[b].size(); s++)
                ll[b * cols + s] = (int)tgs[b][(size_t)s];
    }
}

float ContinualRunner::eval_forgetting() {
    ForgettingMetrics m = engine_.forgetting_metrics();
    stats_.forgetting = m.avg_forgetting;
    stats_.stability = m.stability_score;
    return m.avg_forgetting;
}

void ContinualRunner::set_ewc_lambda(float lambda) {
    ewc_lambda_ = lambda;
    cfg_.ecc_lambda = lambda;
}

void ContinualRunner::set_replay_ratio(float ratio) {
    replay_ratio_ = std::max(0.0f, std::min(1.0f, ratio));
}

float EWCProbe::penalty(const ECCState& ecc, const float* weights, size_t n,
                        float lambda) {
    return ecc.regularize(weights, n, lambda);
}

void EWCProbe::accumulate_fisher(ECCState& ecc, const float* grads, size_t n,
                                 float alpha) {
    ecc.update_fisher(grads, n, alpha);
}

void EWCProbe::anchor(ECCState& ecc, const float* weights, size_t n) {
    ecc.update_anchor(weights, n);
}

} // namespace dense_engine
} // namespace quant
