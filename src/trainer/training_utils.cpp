#define NOMINMAX
#include "quant/training_utils.h"
#include "quant/math.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstdio>
#include <numeric>

namespace quant {

// ============================================================
// compute_load_balance_loss
// ============================================================
Tensor compute_load_balance_loss(const Tensor& expert_mask) {
    auto sh = expert_mask.shape();
    int64_t B = sh[0], S = sh[1], E = sh[2];
    const float* mask = expert_mask.data<float>();

    std::vector<float> counts(E, 0.0f);
    int64_t total = B * S;
    for (int64_t i = 0; i < total; i++) {
        for (int64_t e = 0; e < E; e++) {
            counts[e] += mask[i * E + e];
        }
    }

    float sum_counts = 0.0f;
    float sum_sq = 0.0f;
    for (int64_t e = 0; e < E; e++) {
        sum_counts += counts[e];
        sum_sq += counts[e] * counts[e];
    }

    float loss = 0.0f;
    if (sum_counts > std::numeric_limits<float>::min()) {
        loss = (float)E * sum_sq / (sum_counts * sum_counts);
    }

    Tensor result = Tensor::zeros(Shape{1});
    result.data<float>()[0] = loss;
    return result;
}

// ============================================================
// compute_z_loss
// ============================================================
Tensor compute_z_loss(const Tensor& router_logits) {
    int64_t n = router_logits.numel();
    const float* data = router_logits.data<float>();

    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        float x = std::abs(data[i]);
        if (x < 1e-10f) x = 1e-10f;
        float lx = std::log(x);
        sum += (double)(lx * lx);
    }

    float loss = (n > 0) ? (float)(sum / (double)n) : 0.0f;
    Tensor result = Tensor::zeros(Shape{1});
    result.data<float>()[0] = loss;
    return result;
}

// ============================================================
// compute_importance_loss
// ============================================================
Tensor compute_importance_loss(const Tensor& expert_gates) {
    auto sh = expert_gates.shape();
    int64_t B = sh[0], S = sh[1], E = sh[2];
    const float* gates = expert_gates.data<float>();

    std::vector<float> importance(E, 0.0f);
    int64_t total = B * S;
    for (int64_t i = 0; i < total; i++) {
        for (int64_t e = 0; e < E; e++) {
            importance[e] += gates[i * E + e];
        }
    }

    float mean_imp = 0.0f;
    for (int64_t e = 0; e < E; e++) mean_imp += importance[e];
    mean_imp /= (float)E;

    float var_imp = 0.0f;
    for (int64_t e = 0; e < E; e++) {
        float diff = importance[e] - mean_imp;
        var_imp += diff * diff;
    }
    var_imp /= (float)E;

    float denom = mean_imp * mean_imp + 1e-8f;
    float loss = var_imp / denom;

    Tensor result = Tensor::zeros(Shape{1});
    result.data<float>()[0] = loss;
    return result;
}

// ============================================================
// compute_moe_aux_losses
// ============================================================
MoEAuxLosses compute_moe_aux_losses(
    const Tensor& router_logits,
    const Tensor& expert_gates,
    const Tensor& expert_mask,
    const MoEAuxLosses& cfg)
{
    MoEAuxLosses result;
    result.load_balance_weight = cfg.load_balance_weight;
    result.z_loss_weight = cfg.z_loss_weight;
    result.importance_weight = cfg.importance_weight;

    Tensor lb = compute_load_balance_loss(expert_mask);
    Tensor zl = compute_z_loss(router_logits);
    Tensor il = compute_importance_loss(expert_gates);

    float* lbd = lb.data<float>();
    float* zld = zl.data<float>();
    float* ild = il.data<float>();

    lbd[0] *= cfg.load_balance_weight;
    zld[0] *= cfg.z_loss_weight;
    ild[0] *= cfg.importance_weight;

    result.load_balance_loss = std::move(lb);
    result.z_loss = std::move(zl);
    result.importance_loss = std::move(il);
    return result;
}

// ============================================================
// MixedPrecisionScaler
// ============================================================
MixedPrecisionScaler::MixedPrecisionScaler(float init_scale,
                                            float growth_factor,
                                            float backoff_factor,
                                            int growth_interval)
    : scale_(init_scale)
    , growth_factor_(growth_factor)
    , backoff_factor_(backoff_factor)
    , growth_interval_(growth_interval)
    , good_steps_(0)
{}

Tensor MixedPrecisionScaler::scale_loss(const Tensor& loss) {
    auto sh = loss.shape();
    Tensor scaled = Tensor::zeros(sh);
    const float* src = loss.data<float>();
    float* dst = scaled.data<float>();
    int64_t n = loss.numel();
    for (int64_t i = 0; i < n; i++) {
        dst[i] = src[i] * scale_;
    }
    return scaled;
}

bool MixedPrecisionScaler::check_gradients(const std::vector<Tensor*>& params_grad) {
    for (auto* g : params_grad) {
        if (!g) continue;
        const float* data = g->data<float>();
        int64_t n = g->numel();
        for (int64_t i = 0; i < n; i++) {
            float v = data[i];
            if (std::isinf(v) || std::isnan(v)) {
                return true;
            }
        }
    }
    return false;
}

void MixedPrecisionScaler::update_scale(bool has_overflow) {
    if (has_overflow) {
        scale_ *= backoff_factor_;
        good_steps_ = 0;
    } else {
        good_steps_++;
        if (good_steps_ >= growth_interval_) {
            scale_ *= growth_factor_;
            good_steps_ = 0;
        }
    }
}

void MixedPrecisionScaler::reset() {
    scale_ = 65536.0f;
    good_steps_ = 0;
}

// ============================================================
// MixedPrecisionManager
// ============================================================

void MixedPrecisionManager::fp32_to_fp16_bulk(const float* src, uint16_t* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        float f = src[i];
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        uint16_t sign = (uint16_t)((bits >> 16) & 0x8000);
        int32_t exp = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = bits & 0x007FFFFF;
        if (exp <= 0) {
            if (exp < -10) {
                dst[i] = sign;
            } else {
                mant = (mant | 0x00800000) >> (1 - exp);
                dst[i] = (uint16_t)(sign | (mant >> 13));
            }
        } else if (exp >= 31) {
            if (exp > 31) {
                dst[i] = (uint16_t)(sign | 0x7C00 | (mant != 0 ? 0x0200 : 0));
            } else {
                dst[i] = (uint16_t)(sign | 0x7C00);
            }
        } else {
            dst[i] = (uint16_t)(sign | ((uint16_t)exp << 10) | (mant >> 13));
        }
    }
}

void MixedPrecisionManager::fp16_to_fp32_bulk(const uint16_t* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        uint16_t h = src[i];
        uint32_t sign = (uint32_t)(h >> 15) << 31;
        int32_t exp = (int32_t)((h >> 10) & 0x1F);
        uint32_t mant = h & 0x03FF;
        if (exp == 0) {
            if (mant == 0) {
                dst[i] = 0.0f;
            } else {
                float m = (float)mant / 1024.0f;
                dst[i] = (sign ? -1.0f : 1.0f) * m * 1.5258789e-5f;
            }
        } else if (exp == 31) {
            uint32_t f32 = sign | 0x7F800000 | (mant << 13);
            std::memcpy(&dst[i], &f32, sizeof(float));
        } else {
            uint32_t f32 = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
            std::memcpy(&dst[i], &f32, sizeof(float));
        }
    }
}

MixedPrecisionManager::MixedPrecisionManager(bool enabled,
                                              float init_loss_scale,
                                              float growth_factor,
                                              float backoff_factor,
                                              int growth_interval)
    : enabled_(enabled)
    , loss_scale_(init_loss_scale)
    , growth_factor_(growth_factor)
    , backoff_factor_(backoff_factor)
    , growth_interval_(growth_interval)
    , good_steps_(0)
{}

Tensor MixedPrecisionManager::to_fp16(const Tensor& t) {
    Tensor result(t.shape(), DType::F16);
    const float* src = t.data<float>();
    uint16_t* dst = result.data<uint16_t>();
    fp32_to_fp16_bulk(src, dst, t.numel());
    return result;
}

Tensor MixedPrecisionManager::to_fp32(const Tensor& t) {
    Tensor result(t.shape(), DType::F32);
    const uint16_t* src = t.data<uint16_t>();
    float* dst = result.data<float>();
    fp16_to_fp32_bulk(src, dst, t.numel());
    return result;
}

void MixedPrecisionManager::create_master_weights(const std::vector<Tensor*>& model_params) {
    if (!enabled_) return;
    master_weights_.clear();
    master_weights_.reserve(model_params.size());
    for (auto* p : model_params) {
        master_weights_.push_back(p->clone());
    }
}

void MixedPrecisionManager::sync_weights_to_model(std::vector<Tensor*>& model_params) {
    if (!enabled_ || master_weights_.empty()) return;
    for (size_t i = 0; i < model_params.size() && i < master_weights_.size(); i++) {
        const float* src = master_weights_[i].data<float>();
        float* dst = model_params[i]->data<float>();
        int64_t n = model_params[i]->numel();
        std::memcpy(dst, src, (size_t)n * sizeof(float));
    }
}

void MixedPrecisionManager::sync_weights_from_model(const std::vector<Tensor*>& model_params) {
    if (!enabled_ || master_weights_.empty()) return;
    for (size_t i = 0; i < model_params.size() && i < master_weights_.size(); i++) {
        const float* src = model_params[i]->data<float>();
        float* dst = master_weights_[i].data<float>();
        int64_t n = model_params[i]->numel();
        std::memcpy(dst, src, (size_t)n * sizeof(float));
    }
}

Tensor MixedPrecisionManager::scale_loss(const Tensor& loss) {
    if (!enabled_) return loss;
    auto sh = loss.shape();
    Tensor scaled(sh, DType::F32);
    const float* src = loss.data<float>();
    float* dst = scaled.data<float>();
    int64_t n = loss.numel();
    for (int64_t i = 0; i < n; i++) {
        dst[i] = src[i] * loss_scale_;
    }
    return scaled;
}

bool MixedPrecisionManager::check_gradients(const std::vector<Tensor*>& grads) {
    if (!enabled_) return false;
    for (auto* g : grads) {
        if (!g) continue;
        const float* data = g->data<float>();
        int64_t n = g->numel();
        for (int64_t i = 0; i < n; i++) {
            float v = data[i];
            if (std::isinf(v) || std::isnan(v)) {
                return true;
            }
        }
    }
    return false;
}

void MixedPrecisionManager::unscale_gradients(const std::vector<Tensor*>& params) {
    if (!enabled_ || loss_scale_ == 1.0f) return;
    float inv_scale = 1.0f / loss_scale_;
    for (auto* p : params) {
        if (!p->has_grad()) continue;
        float* g = p->grad().data<float>();
        int64_t n = p->grad().numel();
        for (int64_t i = 0; i < n; i++) {
            g[i] *= inv_scale;
        }
    }
}

void MixedPrecisionManager::update_loss_scale(bool has_overflow) {
    if (!enabled_) return;
    if (has_overflow) {
        loss_scale_ *= backoff_factor_;
        good_steps_ = 0;
    } else {
        good_steps_++;
        if (good_steps_ >= growth_interval_) {
            loss_scale_ *= growth_factor_;
            good_steps_ = 0;
        }
    }
}

void MixedPrecisionManager::cast_gradients_to_fp32(const std::vector<Tensor*>& params) {
    if (!enabled_) return;
    for (auto* p : params) {
        if (!p->has_grad()) continue;
        if (p->grad().dtype() == DType::F16) {
            Tensor grad_f32 = to_fp32(p->grad());
            p->set_grad(grad_f32);
        }
    }
}

void MixedPrecisionManager::reset() {
    loss_scale_ = 65536.0f;
    good_steps_ = 0;
    master_weights_.clear();
}

// ============================================================
// EMAWeightAveraging
// ============================================================
EMAWeightAveraging::EMAWeightAveraging(float decay, bool apply_ema)
    : decay_(decay)
    , apply_ema_(apply_ema)
    , step_(0)
{}

void EMAWeightAveraging::init(const std::vector<Tensor*>& params) {
    ema_weights_.clear();
    ema_weights_.reserve(params.size());
    for (auto* p : params) {
        ema_weights_.push_back(p->clone());
    }
    step_ = 0;
}

void EMAWeightAveraging::update(const std::vector<Tensor*>& params) {
    if (!apply_ema_) return;
    step_++;

    float bias_correction = 1.0f - std::pow(decay_, (float)step_);
    float ema_decay = decay_ / bias_correction;
    float online_decay = (1.0f - decay_) / bias_correction;

    for (size_t i = 0; i < params.size() && i < ema_weights_.size(); i++) {
        const float* p_data = params[i]->data<float>();
        float* ema_data = ema_weights_[i].data<float>();
        int64_t n = params[i]->numel();
        for (int64_t j = 0; j < n; j++) {
            ema_data[j] = ema_decay * ema_data[j] + online_decay * p_data[j];
        }
    }
}

void EMAWeightAveraging::apply_to_model(std::vector<Tensor*>& model_params) {
    if (!apply_ema_ || ema_weights_.empty()) return;
    for (size_t i = 0; i < model_params.size() && i < ema_weights_.size(); i++) {
        const float* ema_data = ema_weights_[i].data<float>();
        float* p_data = model_params[i]->data<float>();
        int64_t n = model_params[i]->numel();
        for (int64_t j = 0; j < n; j++) {
            p_data[j] = ema_data[j];
        }
    }
}

void EMAWeightAveraging::copy_to_model(std::vector<Tensor*>& model_params) {
    apply_to_model(model_params);
}

// ============================================================
// ProgressTracker
// ============================================================
ProgressTracker::ProgressTracker(int total_steps, int log_interval)
    : total_steps_(std::max(1, total_steps)), current_step_(0),
      log_interval_(std::max(1, log_interval)), start_time_(0.0) {}

void ProgressTracker::step() {
    if (current_step_ == 0)
        start_time_ = (double)std::clock() / CLOCKS_PER_SEC;
    current_step_++;
}

float ProgressTracker::progress() const {
    return (float)current_step_ / (float)total_steps_;
}

double ProgressTracker::elapsed_sec() const {
    return (double)std::clock() / CLOCKS_PER_SEC - start_time_;
}

double ProgressTracker::eta_sec() const {
    if (current_step_ <= 0) return 0.0;
    double rate = elapsed_sec() / (double)current_step_;
    return rate * (double)(total_steps_ - current_step_);
}

void ProgressTracker::reset() {
    current_step_ = 0;
    start_time_ = 0.0;
}

// ============================================================
// MetricTracker
// ============================================================
MetricTracker::MetricTracker(float alpha, bool minimize)
    : alpha_(alpha), running_avg_(0.0f),
      best_(minimize ? 1e10f : -1e10f),
      steps_(0), minimize_(minimize) {}

void MetricTracker::update(float value) {
    if (steps_ == 0) {
        running_avg_ = value;
    } else {
        running_avg_ = alpha_ * running_avg_ + (1.0f - alpha_) * value;
    }
    bool is_better = minimize_ ? (value < best_) : (value > best_);
    if (is_better || steps_ == 0) best_ = value;
    steps_++;
}

bool MetricTracker::improved() const {
    if (steps_ <= 1) return true;
    return minimize_ ? (running_avg_ < best_) : (running_avg_ > best_);
}

void MetricTracker::reset() {
    running_avg_ = 0.0f;
    best_ = minimize_ ? 1e10f : -1e10f;
    steps_ = 0;
}

// ============================================================
// ModelSnapshot
// ============================================================
ModelSnapshot::ModelSnapshot(const std::string& path) : path_(path) {}

void ModelSnapshot::save(Optimizer* opt, int step, float current_lr,
                         const std::vector<Tensor*>& params) {
    std::string snap_path = path_ + ".snap";
    FILE* fp = std::fopen(snap_path.c_str(), "wb");
    if (!fp) return;
    int32_t step_i = (int32_t)step;
    fwrite(&step_i, sizeof(step_i), 1, fp);
    fwrite(&current_lr, sizeof(current_lr), 1, fp);
    int32_t num_params = (int32_t)params.size();
    fwrite(&num_params, sizeof(num_params), 1, fp);
    for (auto* p : params) {
        if (!opt) {
            int64_t zero = 0;
            fwrite(&zero, sizeof(zero), 1, fp);
            fwrite(&zero, sizeof(zero), 1, fp);
            continue;
        }
        auto& st = opt->get_state(p);
        int64_t sz_m = st.m.buffer() ? st.m.numel() : 0;
        int64_t sz_v = st.v.buffer() ? st.v.numel() : 0;
        fwrite(&sz_m, sizeof(sz_m), 1, fp);
        if (sz_m > 0) {
            fwrite(st.m.data<float>(), (size_t)sz_m * sizeof(float), 1, fp);
        }
        fwrite(&sz_v, sizeof(sz_v), 1, fp);
        if (sz_v > 0) {
            fwrite(st.v.data<float>(), (size_t)sz_v * sizeof(float), 1, fp);
        }
    }
    fclose(fp);
}

bool ModelSnapshot::load(Optimizer* opt, int& step, float& current_lr,
                         const std::vector<Tensor*>& params) {
    std::string snap_path = path_ + ".snap";
    FILE* fp = std::fopen(snap_path.c_str(), "rb");
    if (!fp) return false;
    int32_t step_i = 0;
    if (fread(&step_i, sizeof(step_i), 1, fp) != 1) { fclose(fp); return false; }
    step = (int)step_i;
    if (fread(&current_lr, sizeof(current_lr), 1, fp) != 1) { fclose(fp); return false; }
    int32_t num_params = 0;
    if (fread(&num_params, sizeof(num_params), 1, fp) != 1) { fclose(fp); return false; }
    int32_t max_load = (std::min)(num_params, (int32_t)params.size());
    for (int32_t i = 0; i < max_load; i++) {
        auto* p = params[(size_t)i];
        int64_t sz_m = 0, sz_v = 0;
        if (fread(&sz_m, sizeof(sz_m), 1, fp) != 1) break;
        if (opt && sz_m > 0 && sz_m <= p->numel()) {
            auto& st = opt->get_state(p);
            st.m = Tensor::zeros(p->shape());
            if (fread(st.m.data<float>(), (size_t)sz_m * sizeof(float), 1, fp) != 1) break;
        } else if (sz_m > 0) {
            std::vector<float> tmp((size_t)sz_m);
            if (fread(tmp.data(), (size_t)sz_m * sizeof(float), 1, fp) != 1) break;
        }
        if (fread(&sz_v, sizeof(sz_v), 1, fp) != 1) break;
        if (opt && sz_v > 0 && sz_v <= p->numel()) {
            auto& st = opt->get_state(p);
            st.v = Tensor::zeros(p->shape());
            if (fread(st.v.data<float>(), (size_t)sz_v * sizeof(float), 1, fp) != 1) break;
        } else if (sz_v > 0) {
            std::vector<float> tmp((size_t)sz_v);
            if (fread(tmp.data(), (size_t)sz_v * sizeof(float), 1, fp) != 1) break;
        }
    }
    fclose(fp);
    return true;
}

// ============================================================
// EarlyStopping
// ============================================================
EarlyStopping::EarlyStopping(int patience, float min_delta, bool minimize)
    : patience_(patience), min_delta_(min_delta), minimize_(minimize),
      best_metric_(minimize ? 1e10f : -1e10f),
      best_epoch_(0), bad_epochs_(0), stopped_(false) {}

bool EarlyStopping::step(float metric) {
    if (stopped_) return true;
    bool improved = minimize_
        ? (metric < best_metric_ - min_delta_)
        : (metric > best_metric_ + min_delta_);
    if (improved) {
        best_metric_ = metric;
        best_epoch_ = bad_epochs_ + 1;
        bad_epochs_ = 0;
    } else {
        bad_epochs_++;
        if (bad_epochs_ >= patience_) {
            stopped_ = true;
            return true;
        }
    }
    return false;
}

void EarlyStopping::reset() {
    best_metric_ = minimize_ ? 1e10f : -1e10f;
    bad_epochs_ = 0;
    best_epoch_ = 0;
    stopped_ = false;
}

// ============================================================
// LearningRateWarmup
// ============================================================
LearningRateWarmup::LearningRateWarmup(float target_lr, int warmup_steps, Mode mode)
    : target_lr_(target_lr), warmup_steps_(warmup_steps), mode_(mode) {}

float LearningRateWarmup::get_lr(int step) const {
    if (step >= warmup_steps_ || warmup_steps_ <= 0) return target_lr_;
    float p = (float)step / (float)warmup_steps_;
    if (mode_ == Mode::LINEAR)
        return target_lr_ * p;
    else
        return target_lr_ * (1.0f - std::exp(-5.0f * p)) / (1.0f - std::exp(-5.0f));
}

// ============================================================
// GradientAccumulator
// ============================================================
GradientAccumulator::GradientAccumulator(int accumulation_steps)
    : accumulation_steps_(std::max(1, accumulation_steps)), current_step_(0) {}

void GradientAccumulator::accumulate(const std::vector<Tensor*>& params) {
    if (!initialized_ || grad_buffer_.size() != params.size()) {
        grad_buffer_.clear();
        grad_buffer_.reserve(params.size());
        for (auto* p : params)
            grad_buffer_.push_back(Tensor::zeros(p->shape()));
        initialized_ = true;
    }
    for (size_t i = 0; i < params.size(); i++) {
        if (!params[i]->has_grad()) continue;
        const float* src = params[i]->grad().data<float>();
        float* dst = grad_buffer_[i].data<float>();
        int64_t n = params[i]->grad().numel();
        for (int64_t j = 0; j < n; j++)
            dst[j] += src[j];
    }
    current_step_++;
}

void GradientAccumulator::apply_grads(const std::vector<Tensor*>& params,
                                       Optimizer* opt) {
    if (current_step_ <= 0) return;
    float scale = 1.0f / (float)current_step_;
    for (size_t i = 0; i < params.size() && i < grad_buffer_.size(); i++) {
        if (!params[i]->has_grad()) continue;
        float* g = params[i]->grad().data<float>();
        const float* buf = grad_buffer_[i].data<float>();
        int64_t n = params[i]->grad().numel();
        for (int64_t j = 0; j < n; j++)
            g[j] = buf[j] * scale;
    }
    if (opt) opt->step();
    reset();
}

void GradientAccumulator::reset() {
    grad_buffer_.clear();
    current_step_ = 0;
    initialized_ = false;
}

} // namespace quant
