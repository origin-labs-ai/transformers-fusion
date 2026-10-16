#include "qat.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace quant {
namespace dense_engine {

QATEngine::QATEngine(int bits, bool symmetric, bool use_lsq, float init_scale) {
    (void)use_lsq;
    (void)init_scale;
    if (bits != 8 && bits != 4) throw std::runtime_error("[qat] bits must be 8 or 4");
    ste_.set_target_format(bits == 8 ? Format::Q_8 : Format::Q_4);
    (void)symmetric;
}

void QATEngine::attach(Trainer* trainer) {
    if (!trainer) throw std::runtime_error("[qat] attach needs Trainer*");
    trainer_ = trainer;
}

Tensor QATEngine::fake_quantize(const Tensor& weight) {
    return ste_.fake_quantize_qat(weight);
}

float QATEngine::step_scales(float lr) {
    (void)lr;
    stats_.steps++;
    return 0.0f;
}

void QATEngine::calibrate(const std::vector<Tensor*>& params, int64_t batches) {
    if (params.empty()) return;
    double ss = 0.0, sq = 0.0;
    int64_t n = 0;
    for (int64_t b = 0; b < batches; b++) {
        for (auto* p : params) {
            if (!p) continue;
            Tensor fq = ste_.fake_quantize_qat(*p);
            float db = sqnr_db(*p, fq);
            ss += db;
            n++;
        }
    }
    if (n > 0) stats_.mean_sqnr_db = (float)(ss / (double)n);
    stats_.mean_scale = 1.0f;
    (void)sq;
}

float QATEngine::sqnr_db(const Tensor& fp32, const Tensor& fq) {
    if (fp32.numel() != fq.numel()) throw std::runtime_error("[qat] sqnr shape mismatch");
    int64_t n = fp32.numel();
    if (n <= 0) return 0.0f;
    const float* x = fp32.data<float>();
    const float* y = fq.data<float>();
    double sig = 0.0, nz = 0.0;
    for (int64_t i = 0; i < n; i++) {
        sig += (double)x[i] * x[i];
        double d = (double)x[i] - y[i];
        nz += d * d;
    }
    if (nz <= 0) return 100.0f;
    return (float)(10.0 * std::log10(sig / nz + 1e-12));
}

void QATEngine::set_bits(int bits) {
    if (bits != 8 && bits != 4) throw std::runtime_error("[qat] bits must be 8 or 4");
    ste_.set_target_format(bits == 8 ? Format::Q_8 : Format::Q_4);
}

void QATEngine::set_schedule(int start_step, int end_step) {
    start_ = start_step;
    end_ = end_step;
}

bool QATEngine::active_at(int step) const { return step >= start_ && step < end_; }

MixedPrecision::MixedPrecision(float init_scale, int interval)
    : scale_(init_scale), interval_(interval) {}

void MixedPrecision::unscale(std::vector<Tensor*>& params) {
    for (auto* p : params) {
        if (!p || !p->has_grad()) continue;
        float* g = p->grad().data<float>();
        int64_t n = p->numel();
        for (int64_t i = 0; i < n; i++) g[i] /= scale_;
    }
}

bool MixedPrecision::check_overflow(const std::vector<Tensor*>& params) {
    for (auto* p : params) {
        if (!p || !p->has_grad()) continue;
        const float* g = p->grad().data<float>();
        int64_t n = p->numel();
        for (int64_t i = 0; i < n; i++)
            if (!std::isfinite(g[i])) return true;
    }
    return false;
}

void MixedPrecision::step(const std::vector<Tensor*>& params) {
    stats_.steps++;
    if (check_overflow(params)) {
        scale_ = std::max(scale_ / 2.0f, 1.0f);
        stats_.overflows++;
        since_grow_ = 0;
    } else {
        since_grow_++;
        if (since_grow_ >= interval_) {
            scale_ = std::min(scale_ * 2.0f, 65536.0f);
            since_grow_ = 0;
        }
    }
    stats_.loss_scale = scale_;
}

} // namespace dense_engine
} // namespace quant
