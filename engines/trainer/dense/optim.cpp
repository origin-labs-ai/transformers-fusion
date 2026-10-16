#include "optim.h"
#include <cmath>
#include <stdexcept>

namespace quant {
namespace dense_engine {

OptimSuite::OptimSuite(Kind kind, float lr, float wd)
    : kind_(kind), adamw_(lr), adam_(lr), adafactor_(lr), sgd_(lr), lion_(lr) {
    adamw_.set_weight_decay(wd);
    adam_.set_weight_decay(wd);
    sgd_.set_weight_decay(wd);
    lion_.set_weight_decay(wd);
    switch (kind_) {
        case Kind::ADAMW: active_ = &adamw_; break;
        case Kind::ADAM: active_ = &adam_; break;
        case Kind::ADAFACTOR: active_ = &adafactor_; break;
        case Kind::SGD: active_ = &sgd_; break;
        case Kind::LION: active_ = &lion_; break;
    }
    stats_.last_lr = lr;
}

void OptimSuite::add_params(const std::vector<Tensor*>& ps) {
    for (auto* p : ps) active_->add_param(p);
}

void OptimSuite::add_group(const std::vector<Tensor*>& ps, float lr_scale, float wd_scale) {
    (void)lr_scale; (void)wd_scale;
    ParamGroup g;
    g.lr_scale = lr_scale;
    g.wd_scale = wd_scale;
    groups_.push_back(g);
    group_params_.push_back(ps);
    for (auto* p : ps) active_->add_param(p);
}

void OptimSuite::step() {
    active_->step();
    stats_.steps++;
    stats_.last_lr = active_->get_lr();
    stats_.last_update_norm = 0.0f;
}

void OptimSuite::zero_grad() { active_->zero_grad(); }
void OptimSuite::set_lr(float lr) { active_->set_lr(lr); stats_.last_lr = lr; }
float OptimSuite::get_lr() const { return active_->get_lr(); }

void OptimSuite::set_schedule(AdamW::Schedule s, int warmup, int total) {
    if (kind_ == Kind::ADAMW) adamw_.set_schedule(s, warmup, total);
}

void OptimSuite::scheduler_step(int step) {
    if (kind_ == Kind::ADAMW) {
        adamw_.scheduler_step(step);
        stats_.last_lr = adamw_.get_lr();
    }
}

float OptimSuite::clip(float max_norm) {
    active_->set_grad_clip_norm(max_norm);
    return active_->clip_grad_norm();
}

float OptimSuite::param_norm(const std::vector<Tensor*>& ps) {
    double sum = 0.0;
    for (auto* p : ps) {
        if (!p) continue;
        const float* d = p->data<float>();
        int64_t n = p->numel();
        for (int64_t i = 0; i < n; i++) sum += (double)d[i] * d[i];
    }
    return (float)std::sqrt(sum);
}

void OptimSuite::snapshot(const std::vector<Tensor*>& ps, std::vector<Tensor>& out) {
    out.clear();
    for (auto* p : ps) out.push_back(p->clone());
}

float OptimSuite::update_norm(const std::vector<Tensor>& before,
                              const std::vector<Tensor>& after) {
    if (before.size() != after.size()) throw std::runtime_error("[optim] snapshot size mismatch");
    double sum = 0.0;
    for (size_t k = 0; k < before.size(); k++) {
        const float* a = before[k].data<float>();
        const float* b = after[k].data<float>();
        int64_t n = std::min(before[k].numel(), after[k].numel());
        for (int64_t i = 0; i < n; i++) {
            double d = (double)b[i] - (double)a[i];
            sum += d * d;
        }
    }
    return (float)std::sqrt(sum);
}

} // namespace dense_engine
} // namespace quant
