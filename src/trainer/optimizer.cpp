#include "quant/optimizer.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace quant {

// ============================================================
// Optimizer base
// ============================================================
void Optimizer::add_param(Tensor* param) { parameters_.push_back(param); }
void Optimizer::add_param_group(std::vector<Tensor*> params) {
    for (auto* p : params) add_param(p);
}
void Optimizer::zero_grad() {
    for (auto* p : parameters_) {
        if (p->requires_grad() && p->has_grad()) p->grad().zero_();
    }
}
float Optimizer::clip_grad_norm() {
    if (grad_clip_norm_ <= 0.0f || parameters_.empty()) return 0.0f;
    double total_norm = 0.0;
    for (auto* p : parameters_) {
        if (!p->has_grad()) continue;
        const float* g = p->grad().data<float>();
        int64_t n = p->numel();
        double p_norm = 0.0;
        for (int64_t i = 0; i < n; i++) p_norm += (double)g[i] * g[i];
        total_norm += p_norm;
    }
    total_norm = std::sqrt(total_norm);
    if (total_norm == 0.0) return 0.0f;
    float scale = (float)(grad_clip_norm_ / total_norm);
    if (scale >= 1.0f) return (float)total_norm;
    for (auto* p : parameters_) {
        if (!p->has_grad()) continue;
        float* g = p->grad().data<float>();
        int64_t n = p->numel();
        for (int64_t i = 0; i < n; i++) g[i] *= scale;
    }
    return (float)total_norm;
}

// ============================================================
// AdamW
// ============================================================
AdamW::AdamW(float lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(lr, weight_decay), beta1_(beta1), beta2_(beta2), eps_(eps), current_lr_(lr) {}
void AdamW::zero_grad() { Optimizer::zero_grad(); }
void AdamW::set_schedule(Schedule s, int warmup_steps, int total_steps, int restart_period) {
    schedule_ = s; warmup_steps_ = warmup_steps;
    total_steps_ = total_steps; restart_period_ = restart_period;
}
void AdamW::scheduler_step(int step) {
    current_lr_ = adjusted_lr(step);
}
float AdamW::adjusted_lr(int step) {
    if (total_steps_ <= 0) return lr_;
    float ratio = (float)step / total_steps_;
    float lr = lr_;
    switch (schedule_) {
        case Schedule::CONSTANT: lr = lr_; break;
        case Schedule::COSINE:
            lr = lr_ * 0.5f * (1.0f + std::cos(3.14159265f * ratio));
            break;
        case Schedule::LINEAR:
            lr = lr_ * (1.0f - ratio);
            break;
        case Schedule::WARMUP_COSINE:
            if (step < warmup_steps_)
                lr = lr_ * (float)step / warmup_steps_;
            else {
                float cos_ratio = (float)(step - warmup_steps_) /
                                  std::max(1, total_steps_ - warmup_steps_);
                lr = lr_ * 0.5f * (1.0f + std::cos(3.14159265f * cos_ratio));
            }
            break;
        case Schedule::COSINE_RESTART: {
            int p = restart_period_ > 0 ? restart_period_ : total_steps_ / 4;
            int cycles = step / p;
            int step_in_cycle = step % p;
            float cycle_ratio = (float)step_in_cycle / p;
            float decay = std::pow(0.5f, (float)cycles);
            lr = lr_ * decay * 0.5f * (1.0f + std::cos(3.14159265f * cycle_ratio));
            break;
        }
    }
    if (lr < 1e-8f) lr = 1e-8f;
    return lr;
}
void AdamW::step() {
    if (parameters_.empty()) return;
    t_++;
    clip_grad_norm();
    float lr = adjusted_lr(t_);
    if (lr <= 0) lr = lr_;
    float bias_corr1 = 1.0f - static_cast<float>(std::pow(beta1_, t_));
    float bias_corr2 = 1.0f - static_cast<float>(std::pow(beta2_, t_));
    float lr_adj = lr * std::sqrt(bias_corr2) / bias_corr1;
    for (auto* param : parameters_) {
        if (!param->requires_grad() || !param->has_grad() || param->grad().numel() == 0) continue;
        auto& state = state_[param];
        if (!state.m.buffer()) {
            state.m = Tensor::zeros(param->shape());
            state.v = Tensor::zeros(param->shape());
        }
        float* m = state.m.data<float>();
        float* v = state.v.data<float>();
        const float* g = param->grad().data<float>();
        float* p = param->data<float>();
        int64_t n = param->numel();
        for (int64_t i = 0; i < n; i++) {
            m[i] = beta1_ * m[i] + (1.0f - beta1_) * g[i];
            v[i] = beta2_ * v[i] + (1.0f - beta2_) * g[i] * g[i];
            p[i] -= lr_adj * m[i] / (std::sqrt(v[i] / bias_corr2) + eps_);
            p[i] -= lr * weight_decay_ * p[i];
        }
    }
}

// ============================================================
// SGD
// ============================================================
SGD::SGD(float lr, float momentum, float weight_decay, bool nesterov)
    : Optimizer(lr, weight_decay), momentum_(momentum), nesterov_(nesterov) {}
void SGD::zero_grad() { Optimizer::zero_grad(); }
void SGD::step() {
    if (parameters_.empty()) return;
    t_++;
    clip_grad_norm();
    for (auto* param : parameters_) {
        if (!param->requires_grad() || !param->has_grad() || param->grad().numel() == 0) continue;
        auto& state = state_[param];
        if (!state.m.buffer())
            state.m = Tensor::zeros(param->shape());
        float* m = state.m.data<float>();
        const float* g = param->grad().data<float>();
        float* p = param->data<float>();
        int64_t n = param->numel();
        if (momentum_ > 0.0f) {
            for (int64_t i = 0; i < n; i++) {
                float g_reg = g[i] + weight_decay_ * p[i];
                float prev_m = m[i];
                m[i] = momentum_ * m[i] + lr_ * g_reg;
                if (nesterov_)
                    p[i] -= momentum_ * m[i] + (1.0f + momentum_) * lr_ * g_reg - momentum_ * prev_m;
                else
                    p[i] -= m[i];
            }
        } else {
            for (int64_t i = 0; i < n; i++)
                p[i] -= lr_ * (g[i] + weight_decay_ * p[i]);
        }
    }
}

// ============================================================
// Adam
// ============================================================
Adam::Adam(float lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(lr, weight_decay), beta1_(beta1), beta2_(beta2), eps_(eps) {}
void Adam::zero_grad() { Optimizer::zero_grad(); }
void Adam::step() {
    if (parameters_.empty()) return;
    t_++; clip_grad_norm();
    float bias_corr1 = 1.0f - static_cast<float>(std::pow(beta1_, t_));
    float bias_corr2 = 1.0f - static_cast<float>(std::pow(beta2_, t_));
    for (auto* param : parameters_) {
        if (!param->requires_grad() || !param->has_grad() || param->grad().numel() == 0) continue;
        auto& state = state_[param];
        if (!state.m.buffer()) {
            state.m = Tensor::zeros(param->shape());
            state.v = Tensor::zeros(param->shape());
        }
        float* m = state.m.data<float>();
        float* v = state.v.data<float>();
        const float* g = param->grad().data<float>();
        float* p = param->data<float>();
        int64_t n = param->numel();
        for (int64_t i = 0; i < n; i++) {
            m[i] = beta1_ * m[i] + (1.0f - beta1_) * g[i];
            v[i] = beta2_ * v[i] + (1.0f - beta2_) * g[i] * g[i];
            float m_hat = m[i] / bias_corr1;
            float v_hat = v[i] / bias_corr2;
            p[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
            p[i] -= lr_ * weight_decay_ * p[i];
        }
    }
}

// ============================================================
// Adamax
// ============================================================
Adamax::Adamax(float lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(lr, weight_decay), beta1_(beta1), beta2_(beta2), eps_(eps) {}
void Adamax::zero_grad() { Optimizer::zero_grad(); }
void Adamax::step() {
    if (parameters_.empty()) return;
    t_++; clip_grad_norm();
    float bias_corr1 = 1.0f - static_cast<float>(std::pow(beta1_, t_));
    for (auto* param : parameters_) {
        if (!param->requires_grad() || !param->has_grad() || param->grad().numel() == 0) continue;
        auto& state = state_[param];
        if (!state.m.buffer()) {
            state.m = Tensor::zeros(param->shape());
            state.v = Tensor::zeros(param->shape());
        }
        float* m = state.m.data<float>();
        float* u = state.v.data<float>();
        const float* g = param->grad().data<float>();
        float* p = param->data<float>();
        int64_t n = param->numel();
        for (int64_t i = 0; i < n; i++) {
            m[i] = beta1_ * m[i] + (1.0f - beta1_) * g[i];
            u[i] = std::max(beta2_ * u[i], std::abs(g[i]) + eps_);
            float m_hat = m[i] / bias_corr1;
            p[i] -= (lr_ / (1.0f - bias_corr1)) * m_hat / u[i];
            p[i] -= lr_ * weight_decay_ * p[i];
        }
    }
}

// ============================================================
// NAdam
// ============================================================
NAdam::NAdam(float lr, float beta1, float beta2, float eps, float weight_decay,
             float momentum_decay)
    : Optimizer(lr, weight_decay), beta1_(beta1), beta2_(beta2), eps_(eps),
      momentum_decay_(momentum_decay) {}
void NAdam::zero_grad() { Optimizer::zero_grad(); }
void NAdam::step() {
    if (parameters_.empty()) return;
    t_++; clip_grad_norm();
    float bias_corr2 = 1.0f - static_cast<float>(std::pow(beta2_, t_));
    float beta1_t = beta1_ * (1.0f - 0.5f * static_cast<float>(std::pow(1.0f - momentum_decay_, (float)t_ * momentum_decay_)));
    for (auto* param : parameters_) {
        if (!param->requires_grad() || !param->has_grad() || param->grad().numel() == 0) continue;
        auto& state = state_[param];
        if (!state.m.buffer()) {
            state.m = Tensor::zeros(param->shape());
            state.v = Tensor::zeros(param->shape());
        }
        float* m = state.m.data<float>();
        float* v = state.v.data<float>();
        const float* g = param->grad().data<float>();
        float* p = param->data<float>();
        int64_t n = param->numel();
        for (int64_t i = 0; i < n; i++) {
            m[i] = beta1_ * m[i] + (1.0f - beta1_) * g[i];
            v[i] = beta2_ * v[i] + (1.0f - beta2_) * g[i] * g[i];
            float m_hat = m[i] / (1.0f - static_cast<float>(std::pow(beta1_, t_)));
            float m_nesterov = beta1_t * m_hat + (1.0f - beta1_t) * g[i] / (1.0f - static_cast<float>(std::pow(beta1_, t_)));
            float v_hat = v[i] / bias_corr2;
            p[i] -= lr_ * m_nesterov / (std::sqrt(v_hat) + eps_);
            p[i] -= lr_ * weight_decay_ * p[i];
        }
    }
}

// ============================================================
// RAdam
// ============================================================
RAdam::RAdam(float lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(lr, weight_decay), beta1_(beta1), beta2_(beta2), eps_(eps) {}
void RAdam::zero_grad() { Optimizer::zero_grad(); }
void RAdam::step() {
    if (parameters_.empty()) return;
    t_++; clip_grad_norm();
    float rho_inf = 2.0f / (1.0f - beta2_) - 1.0f;
    float bias_corr1 = 1.0f - static_cast<float>(std::pow(beta1_, t_));
    float bias_corr2 = 1.0f - static_cast<float>(std::pow(beta2_, t_));
    float rho = rho_inf - 2.0f * (float)t_ * beta2_ / (1.0f - beta2_);
    for (auto* param : parameters_) {
        if (!param->requires_grad() || !param->has_grad() || param->grad().numel() == 0) continue;
        auto& state = state_[param];
        if (!state.m.buffer()) {
            state.m = Tensor::zeros(param->shape());
            state.v = Tensor::zeros(param->shape());
        }
        float* m = state.m.data<float>();
        float* v = state.v.data<float>();
        const float* g = param->grad().data<float>();
        float* p = param->data<float>();
        int64_t n = param->numel();
        for (int64_t i = 0; i < n; i++) {
            m[i] = beta1_ * m[i] + (1.0f - beta1_) * g[i];
            v[i] = beta2_ * v[i] + (1.0f - beta2_) * g[i] * g[i];
            float m_hat = m[i] / bias_corr1;
            float v_hat = v[i] / bias_corr2;
            if (rho > 4.0f) {
                float r_approx = (rho - 4.0f) * (rho - 2.0f) * rho_inf /
                                 (rho_inf - 4.0f) / (rho_inf - 2.0f) / rho;
                p[i] -= lr_ * m_hat * r_approx / (std::sqrt(v_hat) + eps_);
            } else {
                p[i] -= lr_ * m_hat;
            }
            p[i] -= lr_ * weight_decay_ * p[i];
        }
    }
}

// ============================================================
// Lion
// ============================================================
Lion::Lion(float lr, float beta1, float beta2, float weight_decay)
    : Optimizer(lr, weight_decay), beta1_(beta1), beta2_(beta2) {}
void Lion::zero_grad() { Optimizer::zero_grad(); }
void Lion::step() {
    if (parameters_.empty()) return;
    t_++; clip_grad_norm();
    for (auto* param : parameters_) {
        if (!param->requires_grad() || !param->has_grad() || param->grad().numel() == 0) continue;
        auto& state = state_[param];
        if (!state.m.buffer())
            state.m = Tensor::zeros(param->shape());
        float* m = state.m.data<float>();
        const float* g = param->grad().data<float>();
        float* p = param->data<float>();
        int64_t n = param->numel();
        for (int64_t i = 0; i < n; i++) {
            float update = m[i] * beta1_ + g[i] * (1.0f - beta1_);
            float sign = (update > 0.0f) ? 1.0f : ((update < 0.0f) ? -1.0f : 0.0f);
            p[i] -= lr_ * sign;
            p[i] -= lr_ * weight_decay_ * p[i];
            m[i] = m[i] * beta2_ + g[i] * (1.0f - beta2_);
        }
    }
}

// ============================================================
// Adafactor
// ============================================================
Adafactor::Adafactor(float lr, float beta2, float eps, float weight_decay,
                     float clip_threshold, float decay_rate)
    : Optimizer(lr, weight_decay), beta2_(beta2), eps_(eps),
      clip_threshold_(clip_threshold), decay_rate_(decay_rate) {}
void Adafactor::zero_grad() { Optimizer::zero_grad(); }
void Adafactor::step() {
    if (parameters_.empty()) return;
    t_++; clip_grad_norm();
    for (auto* param : parameters_) {
        if (!param->requires_grad() || !param->has_grad() || param->grad().numel() == 0) continue;
        // Adafactor factorizes the second moment into a row vector and a column
        // vector, which only means anything for a 2-D parameter. Everything else
        // is treated as a single column (d1 == 1): then r[i/d1] == r[i] and c[0],
        // i.e. a plain running average of g^2 — the standard non-factorized
        // fallback for biases and for conv weights (rank >= 3).
        //
        // BUG FIXED 2026-09-12: this used `param->dim(1)` unconditionally. Shape
        // zero-fills dims beyond rank, so a 1-D bias gave d1 == 0 and the update
        // loop evaluated `i / d1` -> integer division by zero (SIGFPE on x86, UB
        // per the standard). Any model with a bias under Adafactor crashed.
        const int64_t n = param->numel();
        const bool is_2d = (param->rank() == 2);
        const int64_t d0 = is_2d ? param->dim(0) : n;
        const int64_t d1 = is_2d ? param->dim(1) : 1;

        auto& fs = factor_state_[param];
        if (!fs.m.buffer()) {
            fs.m = Tensor::zeros(param->shape());
            fs.r = Tensor::zeros({d0, 1});
            fs.c = Tensor::zeros({1, d1});
        }
        const float* g = param->grad().data<float>();
        float* p = param->data<float>();
        float* r = fs.r.data<float>();
        float* c = fs.c.data<float>();
        float* m = fs.m.data<float>();
        float g_mean = 0.0f;
        for (int64_t i = 0; i < n; i++) g_mean += g[i] * g[i];
        g_mean = std::sqrt(g_mean / (float)n) + eps_;
        for (int64_t i = 0; i < d0; i++) {
            float row_sum = 0;
            for (int64_t j = 0; j < d1; j++)
                row_sum += g[i * d1 + j] * g[i * d1 + j];
            r[i] = decay_rate_ * r[i] + (1.0f - decay_rate_) * row_sum / (float)d1;
        }
        for (int64_t j = 0; j < d1; j++) {
            float col_sum = 0;
            for (int64_t i = 0; i < d0; i++)
                col_sum += g[i * d1 + j] * g[i * d1 + j];
            c[j] = decay_rate_ * c[j] + (1.0f - decay_rate_) * col_sum / (float)d0;
        }
        float r_mean = 0;
        for (int64_t i = 0; i < d0; i++) r_mean += r[i];
        r_mean /= (float)d0;
        // A parameter whose factorized second moment is zero (or non-finite)
        // received no usable gradient this step — e.g. a fully masked-out
        // block in selective fine-tuning or a zero-initialized adapter factor
        // whose own gradient is identically zero. Updating it would evaluate
        // v_hat = r*c/r_mean as 0/0 = NaN and poison the model, so only the
        // weight-decay part is applied.
        if (!(r_mean > 0.0f) || !std::isfinite(r_mean)) {
            if (weight_decay_ != 0.0f && lr_ != 0.0f) {
                float p_factor = 1.0f - weight_decay_ * lr_;
                for (int64_t i = 0; i < n; i++) p[i] *= p_factor;
            }
            continue;
        }
        float step = lr_ / (std::sqrt(r_mean) + eps_);
        float p_factor = 1.0f - weight_decay_ * lr_;
        for (int64_t i = 0; i < n; i++) {
            float v_hat = r[i / d1] * c[i % d1] / r_mean;
            float update = g[i] / (std::sqrt(v_hat) + eps_);
            update = update * clip_threshold_ / std::max(1.0f, std::abs(update) / g_mean);
            m[i] = beta2_ * m[i] + (1.0f - beta2_) * update;
            p[i] *= p_factor;
            p[i] -= step * m[i];
        }
    }
}

// ============================================================
// RMSProp
// ============================================================
RMSProp::RMSProp(float lr, float rho, float eps, float weight_decay, float momentum)
    : Optimizer(lr, weight_decay), rho_(rho), eps_(eps), momentum_(momentum) {}
void RMSProp::zero_grad() { Optimizer::zero_grad(); }
void RMSProp::step() {
    if (parameters_.empty()) return;
    t_++; clip_grad_norm();
    for (auto* param : parameters_) {
        if (!param->requires_grad() || !param->has_grad() || param->grad().numel() == 0) continue;
        auto& state = state_[param];
        if (!state.v.buffer()) {
            state.v = Tensor::zeros(param->shape());
            if (momentum_ > 0) state.m = Tensor::zeros(param->shape());
        }
        float* v = state.v.data<float>();
        float* buf = momentum_ > 0 ? state.m.data<float>() : nullptr;
        const float* g = param->grad().data<float>();
        float* p = param->data<float>();
        int64_t n = param->numel();
        for (int64_t i = 0; i < n; i++) {
            v[i] = rho_ * v[i] + (1.0f - rho_) * g[i] * g[i];
            float update = lr_ * g[i] / (std::sqrt(v[i]) + eps_);
            if (momentum_ > 0.0f) {
                buf[i] = momentum_ * buf[i] + update;
                p[i] -= buf[i];
            } else {
                p[i] -= update;
            }
            p[i] -= lr_ * weight_decay_ * p[i];
        }
    }
}

// ============================================================
// Ranger
// ============================================================
Ranger::Ranger(float lr, float beta1, float beta2, float eps, float weight_decay,
               int lookahead_k, float lookahead_alpha)
    : Optimizer(lr, weight_decay), lookahead_k_(std::max(1, lookahead_k)),
      lookahead_alpha_(lookahead_alpha), slow_step_(0) {
    radam_ = std::make_unique<RAdam>(lr, beta1, beta2, eps, weight_decay);
}

void Ranger::zero_grad() { Optimizer::zero_grad(); radam_->zero_grad(); }

void Ranger::reset() {
    state_.clear();
    slow_weights_.clear();
    slow_step_ = 0;
}

void Ranger::sync_slow_weights() {
    if (slow_weights_.empty()) {
        slow_weights_.clear();
        slow_weights_.reserve(parameters_.size());
        for (auto* p : parameters_)
            slow_weights_.push_back(p->clone());
    }
}

void Ranger::lookahead_step() {
    for (size_t i = 0; i < parameters_.size() && i < slow_weights_.size(); i++) {
        float* slow = slow_weights_[i].data<float>();
        const float* fast = parameters_[i]->data<float>();
        int64_t n = parameters_[i]->numel();
        for (int64_t j = 0; j < n; j++)
            slow[j] += lookahead_alpha_ * (fast[j] - slow[j]);
    }
}

void Ranger::step() {
    if (parameters_.empty()) return;
    sync_slow_weights();
    // Forward hyperparams to inner RAdam
    radam_->set_lr(lr_);
    radam_->set_weight_decay(weight_decay_);
    radam_->set_grad_clip_norm(grad_clip_norm_);
    // Ensure RAdam has same params registered
    if (radam_->all_state().empty()) {
        for (auto* p : parameters_) {
            radam_->add_param(p);
            auto& rs = radam_->get_state(p);
            auto& os = state_[p];
            rs.m = os.m;
            rs.v = os.v;
        }
    }
    radam_->step();
    // Copy state back
    for (auto* p : parameters_) {
        auto& rs = radam_->get_state(p);
        auto& os = state_[p];
        os.m = rs.m;
        os.v = rs.v;
    }
    t_ = radam_->step_count();
    slow_step_++;
    if (slow_step_ % lookahead_k_ == 0)
        lookahead_step();
}

// ============================================================
// Adabound
// ============================================================
Adabound::Adabound(float lr, float beta1, float beta2, float eps, float weight_decay,
                   float final_lr, float gamma)
    : Optimizer(lr, weight_decay), beta1_(beta1), beta2_(beta2), eps_(eps),
      final_lr_(final_lr), gamma_(gamma) {}
void Adabound::zero_grad() { Optimizer::zero_grad(); }
void Adabound::reset() { state_.clear(); }
void Adabound::step() {
    if (parameters_.empty()) return;
    t_++; clip_grad_norm();
    float bias_corr1 = 1.0f - static_cast<float>(std::pow(beta1_, t_));
    float bias_corr2 = 1.0f - static_cast<float>(std::pow(beta2_, t_));
    // Paper (Luo et al. 2019): bounds converge to final_lr, and they clip
    // the per-element RATE alpha/sqrt(v_hat) — never the signed step.
    // The old code clipped the signed Adam step into [lower, upper] with
    // lower >= 0, forcing every negative-direction update positive and
    // diverging on a plain quadratic (8.0 -> 8.15 over 30 steps).
    float lower = final_lr_ * (1.0f - 1.0f / (gamma_ * (float)t_ + 1.0f));
    float upper = final_lr_ * (1.0f + 1.0f / (gamma_ * (float)t_ + 1.0f));
    if (lower < 0.0f) lower = 0.0f;
    for (auto* param : parameters_) {
        if (!param->requires_grad() || !param->has_grad() || param->grad().numel() == 0) continue;
        auto& state = state_[param];
        if (!state.m.buffer()) {
            state.m = Tensor::zeros(param->shape());
            state.v = Tensor::zeros(param->shape());
        }
        float* m = state.m.data<float>();
        float* v = state.v.data<float>();
        const float* g = param->grad().data<float>();
        float* p = param->data<float>();
        int64_t n = param->numel();
        for (int64_t i = 0; i < n; i++) {
            m[i] = beta1_ * m[i] + (1.0f - beta1_) * g[i];
            v[i] = beta2_ * v[i] + (1.0f - beta2_) * g[i] * g[i];
            float m_hat = m[i] / bias_corr1;
            float v_hat = v[i] / bias_corr2;
            float eta = lr_ / (std::sqrt(v_hat) + eps_);
            eta = std::max(std::min(eta, upper), lower);
            p[i] -= eta * m_hat;
            p[i] -= lr_ * weight_decay_ * p[i];
        }
    }
}

// ============================================================
// Lamb
// ============================================================
Lamb::Lamb(float lr, float beta1, float beta2, float eps, float weight_decay,
           float max_trust_ratio)
    : Optimizer(lr, weight_decay), beta1_(beta1), beta2_(beta2), eps_(eps),
      max_trust_ratio_(max_trust_ratio) {}
void Lamb::zero_grad() { Optimizer::zero_grad(); }
void Lamb::reset() { state_.clear(); }
void Lamb::step() {
    if (parameters_.empty()) return;
    t_++; clip_grad_norm();
    float bias_corr1 = 1.0f - static_cast<float>(std::pow(beta1_, t_));
    float bias_corr2 = 1.0f - static_cast<float>(std::pow(beta2_, t_));
    for (auto* param : parameters_) {
        if (!param->requires_grad() || !param->has_grad() || param->grad().numel() == 0) continue;
        auto& state = state_[param];
        if (!state.m.buffer()) {
            state.m = Tensor::zeros(param->shape());
            state.v = Tensor::zeros(param->shape());
        }
        float* m = state.m.data<float>();
        float* v = state.v.data<float>();
        const float* g = param->grad().data<float>();
        float* p = param->data<float>();
        int64_t n = param->numel();
        double w_norm = 0.0, g_norm = 0.0;
        for (int64_t i = 0; i < n; i++) {
            m[i] = beta1_ * m[i] + (1.0f - beta1_) * g[i];
            v[i] = beta2_ * v[i] + (1.0f - beta2_) * g[i] * g[i];
            float m_hat = m[i] / bias_corr1;
            float v_hat = v[i] / bias_corr2;
            float update = m_hat / (std::sqrt(v_hat) + eps_);
            w_norm += (double)p[i] * p[i];
            g_norm += (double)update * update;
        }
        g_norm = std::sqrt(g_norm);
        w_norm = std::sqrt(w_norm);
        float trust_ratio = (w_norm > 0.0 && g_norm > 0.0)
            ? (float)(w_norm / (g_norm + 1e-8f)) : 1.0f;
        if (trust_ratio > max_trust_ratio_) trust_ratio = max_trust_ratio_;
        for (int64_t i = 0; i < n; i++) {
            p[i] -= lr_ * weight_decay_ * p[i];
            float m_hat = m[i] / bias_corr1;
            float v_hat = v[i] / bias_corr2;
            p[i] -= lr_ * trust_ratio * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}

// ============================================================
// LARS
// ============================================================
LARS::LARS(float lr, float momentum, float weight_decay,
           float trust_coefficient, float eps)
    : Optimizer(lr, weight_decay), momentum_(momentum),
      trust_coefficient_(trust_coefficient), eps_(eps) {}
void LARS::zero_grad() { Optimizer::zero_grad(); }
void LARS::reset() { state_.clear(); }
void LARS::step() {
    if (parameters_.empty()) return;
    t_++; clip_grad_norm();
    for (auto* param : parameters_) {
        if (!param->requires_grad() || !param->has_grad() || param->grad().numel() == 0) continue;
        auto& state = state_[param];
        if (!state.m.buffer())
            state.m = Tensor::zeros(param->shape());
        float* m = state.m.data<float>();
        const float* g = param->grad().data<float>();
        float* p = param->data<float>();
        int64_t n = param->numel();
        double w_norm = 0.0, g_norm = 0.0;
        for (int64_t i = 0; i < n; i++) {
            float g_reg = g[i] + weight_decay_ * p[i];
            g_norm += (double)g_reg * g_reg;
            w_norm += (double)p[i] * p[i];
        }
        g_norm = std::sqrt(g_norm);
        w_norm = std::sqrt(w_norm);
        float local_lr = lr_;
        if (g_norm > 0.0 && w_norm > 0.0)
            local_lr *= trust_coefficient_ * (float)(w_norm / (g_norm + eps_));
        for (int64_t i = 0; i < n; i++) {
            float g_reg = g[i] + weight_decay_ * p[i];
            m[i] = momentum_ * m[i] + local_lr * g_reg;
            p[i] -= m[i];
        }
    }
}

// ============================================================
// NovoGrad
// ============================================================
NovoGrad::NovoGrad(float lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(lr, weight_decay), beta1_(beta1), beta2_(beta2), eps_(eps) {}
void NovoGrad::zero_grad() { Optimizer::zero_grad(); }
void NovoGrad::reset() { state_.clear(); }
void NovoGrad::step() {
    if (parameters_.empty()) return;
    t_++; clip_grad_norm();
    float bias_corr1 = 1.0f - static_cast<float>(std::pow(beta1_, t_));
    for (auto* param : parameters_) {
        if (!param->requires_grad() || !param->has_grad() || param->grad().numel() == 0) continue;
        auto& state = state_[param];
        if (!state.m.buffer()) {
            state.m = Tensor::zeros(param->shape());
            state.v = Tensor::zeros(Shape{1});
        }
        float* m = state.m.data<float>();
        float* v = state.v.data<float>();
        const float* g = param->grad().data<float>();
        float* p = param->data<float>();
        int64_t n = param->numel();
        double pgn = 0.0;
        for (int64_t i = 0; i < n; i++) pgn += (double)g[i] * g[i];
        float param_grad_norm = std::sqrt((float)pgn);
        v[0] = beta2_ * v[0] + (1.0f - beta2_) * param_grad_norm * param_grad_norm;
        float v_hat = v[0];
        for (int64_t i = 0; i < n; i++) {
            float g_normed = g[i] / (param_grad_norm + eps_);
            m[i] = beta1_ * m[i] + (1.0f - beta1_) * g_normed;
            float m_hat = m[i] / bias_corr1;
            p[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
            p[i] -= lr_ * weight_decay_ * p[i];
        }
    }
}

} // namespace quant