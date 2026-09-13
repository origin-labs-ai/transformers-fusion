#include "quant/scheduler.h"
#include <algorithm>

namespace quant {

float LinearDecayScheduler::get_lr(int step) {
    float ratio = std::min(1.0f, (float)step / total_steps_);
    last_lr_ = start_lr_ + (end_lr_ - start_lr_) * ratio;
    return last_lr_;
}

float CosineDecayScheduler::get_lr(int step) {
    float ratio = std::min(1.0f, (float)step / total_steps_);
    last_lr_ = end_lr_ + (start_lr_ - end_lr_) * 0.5f *
               (1.0f + std::cos(3.14159265f * ratio));
    return last_lr_;
}

float ExponentialDecayScheduler::get_lr(int step) {
    int cycles = step / decay_steps_;
    last_lr_ = initial_lr_ * std::pow(decay_rate_, (float)cycles);
    return last_lr_;
}

float StepDecayScheduler::get_lr(int step) {
    int cycles = step / step_size_;
    last_lr_ = initial_lr_ * std::pow(gamma_, (float)cycles);
    return last_lr_;
}

float ReduceLROnPlateauScheduler::get_lr(int) {
    last_lr_ = base_lr_;
    return base_lr_;
}

void ReduceLROnPlateauScheduler::step_metric(float metric) {
    if (metric < best_metric_ - threshold_) {
        best_metric_ = metric;
        bad_epochs_ = 0;
    } else {
        bad_epochs_++;
        if (bad_epochs_ >= patience_) {
            base_lr_ = std::max(base_lr_ * factor_, min_lr_);
            bad_epochs_ = 0;
        }
    }
    steps_++;
}

float OneCycleScheduler::get_lr(int step) {
    float p = (float)step / total_steps_;
    if (p <= pct_start_) {
        float cycle_p = p / pct_start_;
        last_lr_ = init_lr_ + (max_lr_ - init_lr_) * cycle_p;
    } else {
        float cycle_p = (p - pct_start_) / (1.0f - pct_start_);
        last_lr_ = max_lr_ - (max_lr_ - final_lr_) * cycle_p;
    }
    return last_lr_;
}

float WarmupScheduler::get_lr(int step) {
    if (step < warmup_steps_) {
        float p = (float)step / warmup_steps_;
        last_lr_ = warmup_start_lr_ + (wrapped_->get_lr(0) - warmup_start_lr_) * p;
    } else {
        last_lr_ = wrapped_->get_lr(step - warmup_steps_);
    }
    return last_lr_;
}

float SequentialScheduler::get_lr(int step) {
    int cumulative = 0;
    for (auto& seg : segments_) {
        if (step < cumulative + seg.steps) {
            last_lr_ = seg.scheduler->get_lr(step - cumulative);
            return last_lr_;
        }
        cumulative += seg.steps;
    }
    if (!segments_.empty())
        last_lr_ = segments_.back().scheduler->get_lr(0);
    return last_lr_;
}

// ============================================================
// ChainedScheduler
// ============================================================
ChainedScheduler::ChainedScheduler(LRScheduler* primary, LRScheduler* secondary)
    : primary_(primary), secondary_(secondary) {}

float ChainedScheduler::get_lr(int step) {
    last_lr_ = primary_->get_lr(step) * secondary_->get_lr(step);
    return last_lr_;
}

float ChainedScheduler::get_last_lr() const {
    return last_lr_;
}

// ============================================================
// CosineWarmupRestart
// ============================================================
CosineWarmupRestart::CosineWarmupRestart(float base_lr, int total_steps,
                                         int warmup_steps, float min_lr)
    : base_lr_(base_lr), total_steps_(total_steps),
      warmup_steps_(warmup_steps), min_lr_(min_lr) {}

float CosineWarmupRestart::get_lr(int step) {
    if (warmup_steps_ > 0 && step < warmup_steps_) {
        last_lr_ = base_lr_ * (float)step / (float)warmup_steps_;
    } else {
        int adj_step = std::max(0, step - warmup_steps_);
        int adj_total = std::max(1, total_steps_ - warmup_steps_);
        float p = (float)(adj_step % adj_total) / (float)adj_total;
        int restart = adj_step / adj_total;
        float decay = std::pow(0.5f, (float)restart);
        last_lr_ = min_lr_ + (base_lr_ - min_lr_) * decay * 0.5f *
                   (1.0f + std::cos(3.14159265f * p));
    }
    return last_lr_;
}

// ============================================================
// PolyScheduler
// ============================================================
PolyScheduler::PolyScheduler(float base_lr, float end_lr, int total_steps,
                             float power)
    : base_lr_(base_lr), end_lr_(end_lr), total_steps_(total_steps),
      power_(power) {}

float PolyScheduler::get_lr(int step) {
    float ratio = std::min(1.0f, (float)step / (float)std::max(total_steps_, 1));
    last_lr_ = end_lr_ + (base_lr_ - end_lr_) * std::pow(1.0f - ratio, power_);
    return last_lr_;
}

// ============================================================
// MultiStepDecay
// ============================================================
MultiStepDecay::MultiStepDecay(float base_lr, std::vector<int> milestones,
                               float gamma)
    : base_lr_(base_lr), milestones_(std::move(milestones)), gamma_(gamma) {}

float MultiStepDecay::get_lr(int step) {
    float lr = base_lr_;
    for (int m : milestones_)
        if (step >= m) lr *= gamma_;
    last_lr_ = lr;
    return last_lr_;
}

// ============================================================
// LinearWarmupDecay
// ============================================================
LinearWarmupDecay::LinearWarmupDecay(float start_lr, float peak_lr, float end_lr,
                                     int warmup_steps, int total_steps,
                                     float decay_power)
    : start_lr_(start_lr), peak_lr_(peak_lr), end_lr_(end_lr),
      warmup_steps_(warmup_steps), total_steps_(total_steps),
      decay_power_(decay_power) {}

float LinearWarmupDecay::get_lr(int step) {
    if (step < warmup_steps_) {
        float p = (float)step / std::max(warmup_steps_, 1);
        last_lr_ = start_lr_ + (peak_lr_ - start_lr_) * p;
    } else {
        float p = (float)(step - warmup_steps_) /
                  std::max(total_steps_ - warmup_steps_, 1);
        p = std::min(1.0f, p);
        last_lr_ = peak_lr_ + (end_lr_ - peak_lr_) * std::pow(p, decay_power_);
        if (last_lr_ < end_lr_) last_lr_ = end_lr_;
    }
    return last_lr_;
}

// ============================================================
// CyclicLR
// ============================================================
CyclicLR::CyclicLR(float base_lr, float max_lr, int step_size_up,
                   int step_size_down, float decay, CycleMode mode)
    : base_lr_(base_lr), max_lr_(max_lr), decay_(decay),
      step_size_up_(step_size_up),
      step_size_down_(step_size_down > 0 ? step_size_down : step_size_up),
      mode_(mode) {
    total_size_ = step_size_up_ + step_size_down_;
}

float CyclicLR::get_lr(int step) {
    if (total_size_ <= 0) { last_lr_ = base_lr_; return base_lr_; }
    int cycle = step / total_size_;
    int step_in_cycle = step % total_size_;
    float cycle_amp = max_lr_ - base_lr_;
    if (mode_ == CycleMode::TRIANGULAR2) {
        cycle_amp *= std::pow(0.5f, (float)cycle);
    } else if (mode_ == CycleMode::EXP_RANGE) {
        cycle_amp *= std::pow(decay_, (float)cycle);
    } else if (decay_ > 0.0f) {
        cycle_amp *= std::pow(1.0f - decay_, (float)cycle);
    }
    if (cycle_amp < 0.0f) cycle_amp = 0.0f;
    if (step_in_cycle < step_size_up_) {
        float p = (float)step_in_cycle / step_size_up_;
        last_lr_ = base_lr_ + cycle_amp * p;
    } else {
        float p = (float)(step_in_cycle - step_size_up_) / step_size_down_;
        last_lr_ = base_lr_ + cycle_amp * (1.0f - p);
    }
    return last_lr_;
}

} // namespace quant