#include "quant/qat.h"
#include "quant/math.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

namespace quant {
namespace qat {

// ============================================================================
// FakeQuantizeFunction
// ============================================================================
std::vector<Tensor> FakeQuantizeFunction::forward(const std::vector<Tensor>& inputs) {
    const Tensor& x = inputs[0];
    saved = inputs;
    Tensor out(x.shape(), DType::F32);
    const float* xd = x.data<float>();
    float* od = out.data<float>();
    int64_t n = x.numel();
    float inv_scale = 1.0f / std::max(scale_, 1e-8f);
    for (int64_t i = 0; i < n; ++i) {
        float v = xd[i] * inv_scale;
        // round to nearest
        float r = std::round(v);
        if (r < (float)qmin_) r = (float)qmin_;
        if (r > (float)qmax_) r = (float)qmax_;
        od[i] = r * scale_;
    }
    return {out};
}

std::vector<Tensor> FakeQuantizeFunction::backward(const std::vector<Tensor>& grad_output) {
    // STE: gradient passes unchanged (identity)
    // grad w.r.t x is grad_output directly
    return {grad_output[0]};
}

// ============================================================================
// LSQFunction
// ============================================================================
std::vector<Tensor> LSQFunction::forward(const std::vector<Tensor>& inputs) {
    // inputs[0] = x, inputs[1] = scale (scalar tensor)
    const Tensor& x = inputs[0];
    const Tensor& s_t = inputs[1];
    float scale = s_t.data<float>()[0];
    scale = std::max(scale, 1e-8f);
    saved = inputs;
    Tensor out(x.shape(), DType::F32);
    const float* xd = x.data<float>();
    float* od = out.data<float>();
    int64_t n = x.numel();
    float inv = 1.0f / scale;
    for (int64_t i = 0; i < n; ++i) {
        float v = xd[i] * inv;
        float r = std::round(v);
        if (r < (float)qmin_) r = (float)qmin_;
        if (r > (float)qmax_) r = (float)qmax_;
        od[i] = r * scale;
    }
    return {out};
}

std::vector<Tensor> LSQFunction::backward(const std::vector<Tensor>& grad_output) {
    const Tensor& x = saved[0];
    const Tensor& s_t = saved[1];
    float scale = s_t.data<float>()[0];
    scale = std::max(scale, 1e-8f);
    const Tensor& gout = grad_output[0];
    int64_t n = x.numel();
    const float* xd = x.data<float>();
    const float* gd = gout.data<float>();

    // grad w.r.t x : STE with clipping mask
    Tensor gx(x.shape(), DType::F32);
    float* gxd = gx.data<float>();
    float inv = 1.0f / scale;
    float grad_scale_acc = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        float xs = xd[i] * inv;
        float rounded = std::round(xs);
        // clamp rounded already but need xs range check
        float clamped = rounded;
        if (clamped < (float)qmin_) clamped = (float)qmin_;
        if (clamped > (float)qmax_) clamped = (float)qmax_;

        // grad_x: STE inside range, 0 outside
        bool inside = (xs >= (float)qmin_ && xs <= (float)qmax_);
        gxd[i] = inside ? gd[i] : 0.0f;

        // grad_scale per LSQ paper
        float g_s_elem;
        if (xs < (float)qmin_) g_s_elem = (float)qmin_;
        else if (xs > (float)qmax_) g_s_elem = (float)qmax_;
        else g_s_elem = -xs + rounded;
        grad_scale_acc += gd[i] * g_s_elem;
    }

    Tensor gs(s_t.shape(), DType::F32);
    gs.zero_();
    // s_t is scalar shape [1] or []
    if (gs.numel() >= 1) gs.data<float>()[0] = grad_scale_acc;
    else {
        // fallback
        Tensor tmp(Shape{1}, DType::F32);
        tmp.data<float>()[0] = grad_scale_acc;
        return {gx, tmp};
    }
    return {gx, gs};
}

// ============================================================================
// LSQQuantizer
// ============================================================================
LSQQuantizer::LSQQuantizer(const QATConfig& cfg) : cfg_(cfg), scale_(cfg.init_scale) {
    init_qrange();
    if (scale_ <= 1e-8f) scale_ = 1e-6f;
}

void LSQQuantizer::init_qrange() {
    get_qrange(cfg_, qmin_, qmax_);
}

QuantParams LSQQuantizer::quant_params() const {
    QuantParams p;
    p.scale = scale_;
    p.qmin = qmin_;
    p.qmax = qmax_;
    p.zero_point = 0;
    return p;
}

Tensor LSQQuantizer::scale_param_tensor() {
    Tensor t(Shape{1}, DType::F32);
    t.data<float>()[0] = scale_;
    t.requires_grad(true);
    return t;
}

float LSQQuantizer::compute_scale_grad(const Tensor& x, const Tensor& grad_output) const {
    float s = std::max(scale_, 1e-8f);
    int64_t n = x.numel();
    const float* xd = x.data<float>();
    const float* gd = grad_output.data<float>();
    float acc = 0.0f;
    float inv = 1.0f / s;
    for (int64_t i = 0; i < n; ++i) {
        float xs = xd[i] * inv;
        float r = std::round(xs);
        float g;
        if (xs < (float)qmin_) g = (float)qmin_;
        else if (xs > (float)qmax_) g = (float)qmax_;
        else g = -xs + r;
        acc += gd[i] * g;
    }
    // LSQ grad scaling heuristic
    float grad_scale = cfg_.lsq_grad_scale;
    if (grad_scale != 1.0f) acc *= grad_scale;
    return acc;
}

Tensor LSQQuantizer::forward(const Tensor& x) {
    // non-autograd path uses internal scale_
    if (!AutogradEngine::enabled()) {
        Tensor out(x.shape(), DType::F32);
        const float* xd = x.data<float>();
        float* od = out.data<float>();
        int64_t n = x.numel();
        float s = std::max(scale_, 1e-8f);
        float inv = 1.0f / s;
        for (int64_t i = 0; i < n; ++i) {
            float r = std::round(xd[i] * inv);
            if (r < qmin_) r = (float)qmin_;
            if (r > qmax_) r = (float)qmax_;
            od[i] = r * s;
        }
        return out;
    }
    // autograd path: create scale param tensor and use LSQFunction
    Tensor scale_t(Shape{1}, DType::F32);
    scale_t.data<float>()[0] = scale_;
    auto fn = std::make_shared<LSQFunction>(qmin_, qmax_);
    auto outs = fn->forward({x, scale_t});
    Tensor& out = outs[0];
    out.requires_grad(true);
    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {x, scale_t};
    node->outputs = {out};
    AutogradEngine::instance().register_node(node);
    return out;
}

Tensor LSQQuantizer::forward(const Tensor& x, Tensor& scale_param) {
    // allow external trainable scale
    if (scale_param.numel() == 0) {
        scale_param = Tensor(Shape{1}, DType::F32);
        scale_param.data<float>()[0] = scale_;
    }
    scale_param.requires_grad(true);
    if (!AutogradEngine::enabled()) {
        float s = std::max(scale_param.data<float>()[0], 1e-8f);
        Tensor out(x.shape(), DType::F32);
        const float* xd = x.data<float>();
        float* od = out.data<float>();
        int64_t n = x.numel();
        float inv = 1.0f / s;
        for (int64_t i = 0; i < n; ++i) {
            float r = std::round(xd[i] * inv);
            if (r < qmin_) r = (float)qmin_;
            if (r > qmax_) r = (float)qmax_;
            od[i] = r * s;
        }
        return out;
    }
    auto fn = std::make_shared<LSQFunction>(qmin_, qmax_);
    auto outs = fn->forward({x, scale_param});
    Tensor& out = outs[0];
    out.requires_grad(true);
    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {x, scale_param};
    node->outputs = {out};
    AutogradEngine::instance().register_node(node);
    return out;
}

// ============================================================================
// Functional helpers
// ============================================================================
Tensor fake_quantize(const Tensor& x, float scale, int qmin, int qmax) {
    if (!AutogradEngine::enabled()) {
        Tensor out(x.shape(), DType::F32);
        const float* xd = x.data<float>();
        float* od = out.data<float>();
        int64_t n = x.numel();
        float inv = 1.0f / std::max(scale, 1e-8f);
        for (int64_t i = 0; i < n; ++i) {
            float r = std::round(xd[i] * inv);
            if (r < qmin) r = (float)qmin;
            if (r > qmax) r = (float)qmax;
            od[i] = r * scale;
        }
        return out;
    }
    auto fn = std::make_shared<FakeQuantizeFunction>(scale, qmin, qmax);
    auto outs = fn->forward({x});
    Tensor& out = outs[0];
    out.requires_grad(true);
    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {x};
    node->outputs = {out};
    AutogradEngine::instance().register_node(node);
    return out;
}

Tensor fake_quantize_per_tensor(const Tensor& x, float scale, int bits, bool symmetric) {
    int qmin, qmax;
    get_qrange_bits(bits, symmetric, false, qmin, qmax);
    return fake_quantize(x, scale, qmin, qmax);
}

Tensor ste_fake_quantize(const Tensor& x, float scale, int qmin, int qmax) {
    return fake_quantize(x, scale, qmin, qmax);
}

Tensor lsq_fake_quantize(const Tensor& x, Tensor& scale_param, int qmin, int qmax) {
    if (!AutogradEngine::enabled()) {
        float s = std::max(scale_param.data<float>()[0], 1e-8f);
        Tensor out(x.shape(), DType::F32);
        const float* xd = x.data<float>();
        float* od = out.data<float>();
        int64_t n = x.numel();
        float inv = 1.0f / s;
        for (int64_t i = 0; i < n; ++i) {
            float r = std::round(xd[i] * inv);
            if (r < qmin) r = (float)qmin;
            if (r > qmax) r = (float)qmax;
            od[i] = r * s;
        }
        return out;
    }
    auto fn = std::make_shared<LSQFunction>(qmin, qmax);
    auto outs = fn->forward({x, scale_param});
    Tensor& out = outs[0];
    out.requires_grad(true);
    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {x, scale_param};
    node->outputs = {out};
    AutogradEngine::instance().register_node(node);
    return out;
}

Tensor fake_quantize_with_observer(const Tensor& x, Observer& obs, int bits, bool symmetric) {
    obs.observe(x);
    QuantParams p = obs.calc_qparams(bits, symmetric);
    return fake_quantize(x, p.scale, p.qmin, p.qmax);
}

// ============================================================================
// Observer implementations
// ============================================================================
void MinMaxObserver::observe(const Tensor& data) {
    int64_t n = data.numel();
    const float* d = data.data<float>();
    if (n == 0) return;
    float mn = d[0], mx = d[0];
    for (int64_t i = 1; i < n; ++i) {
        if (d[i] < mn) mn = d[i];
        if (d[i] > mx) mx = d[i];
    }
    if (!has_data_) { min_val_ = mn; max_val_ = mx; has_data_ = true; }
    else { min_val_ = std::min(min_val_, mn); max_val_ = std::max(max_val_, mx); }
}

QuantParams MinMaxObserver::calc_qparams(int bits, bool symmetric) const {
    QuantParams p;
    get_qrange_bits(bits, symmetric, false, p.qmin, p.qmax);
    if (!has_data_) { p.scale = 1.0f; p.zero_point = 0; return p; }
    if (symmetric) {
        float abs_max = std::max(std::fabs(min_val_), std::fabs(max_val_));
        if (abs_max < 1e-8f) abs_max = 1e-8f;
        p.scale = abs_max / (float)p.qmax;
        p.zero_point = 0;
    } else {
        float range = max_val_ - min_val_;
        if (range < 1e-8f) range = 1e-8f;
        p.scale = range / (float)(p.qmax - p.qmin);
        p.zero_point = (int32_t)std::round(-min_val_ / p.scale);
        if (p.zero_point < p.qmin) p.zero_point = p.qmin;
        if (p.zero_point > p.qmax) p.zero_point = p.qmax;
    }
    if (p.scale < 1e-8f) p.scale = 1e-8f;
    return p;
}

void MovingAverageMinMaxObserver::observe(const Tensor& data) {
    int64_t n = data.numel();
    const float* d = data.data<float>();
    if (n == 0) return;
    float mn = d[0], mx = d[0];
    for (int64_t i = 1; i < n; ++i) {
        if (d[i] < mn) mn = d[i];
        if (d[i] > mx) mx = d[i];
    }
    if (!has_data_) { min_val_ = mn; max_val_ = mx; has_data_ = true; }
    else {
        // EMA: new = (1 - momentum)*old + momentum*batch
        min_val_ = (1.0f - momentum_) * min_val_ + momentum_ * mn;
        max_val_ = (1.0f - momentum_) * max_val_ + momentum_ * mx;
    }
}

QuantParams MovingAverageMinMaxObserver::calc_qparams(int bits, bool symmetric) const {
    QuantParams p;
    get_qrange_bits(bits, symmetric, false, p.qmin, p.qmax);
    if (!has_data_) { p.scale = 1.0f; p.zero_point = 0; return p; }
    if (symmetric) {
        float abs_max = std::max(std::fabs(min_val_), std::fabs(max_val_));
        if (abs_max < 1e-8f) abs_max = 1e-8f;
        p.scale = abs_max / (float)p.qmax;
        p.zero_point = 0;
    } else {
        float range = max_val_ - min_val_;
        if (range < 1e-8f) range = 1e-8f;
        p.scale = range / (float)(p.qmax - p.qmin);
        p.zero_point = (int32_t)std::round(-min_val_ / p.scale);
        if (p.zero_point < p.qmin) p.zero_point = p.qmin;
        if (p.zero_point > p.qmax) p.zero_point = p.qmax;
    }
    if (p.scale < 1e-8f) p.scale = 1e-8f;
    return p;
}

void KLDivergenceObserver::observe(const Tensor& data) {
    int64_t n = data.numel();
    const float* d = data.data<float>();
    if (n == 0) return;
    // update global min/max for histogram range
    float mn = d[0], mx = d[0];
    for (int64_t i = 1; i < n; ++i) {
        if (d[i] < mn) mn = d[i];
        if (d[i] > mx) mx = d[i];
    }
    if (!has_data_) {
        hist_min_ = mn;
        hist_max_ = mx;
        has_data_ = true;
    } else {
        hist_min_ = std::min(hist_min_, mn);
        hist_max_ = std::max(hist_max_, mx);
    }
    if (hist_max_ - hist_min_ < 1e-8f) {
        hist_max_ = hist_min_ + 1e-6f;
    }
    float bin_width = (hist_max_ - hist_min_) / (float)bins_;
    // rebuild histogram from current batch + existing counts
    // For simplicity accumulate; reset if range expanded significantly we re-bin old data is lost
    // Accumulate counts for this batch
    for (int64_t i = 0; i < n; ++i) {
        float v = d[i];
        int b = (int)((v - hist_min_) / bin_width);
        if (b < 0) b = 0;
        if (b >= bins_) b = bins_ - 1;
        histogram_[(size_t)b]++;
    }
}

float KLDivergenceObserver::compute_kl_threshold(int bits, bool symmetric) const {
    // TensorRT-style KL calibration: find threshold minimizing KL(P||Q)
    // P = original histogram, Q = quantized histogram expanded back
    // Simplified: iterate candidate cutoffs
    if (!has_data_) return 1.0f;
    int64_t total = 0;
    for (auto c : histogram_) total += c;
    if (total == 0) return 1.0f;

    int qlevels = 1 << bits;
    if (symmetric) {
        // symmetric range uses max abs; search over abs threshold
        float abs_max = std::max(std::fabs(hist_min_), std::fabs(hist_max_));
        return abs_max;
    }
    // asymmetric KL search over right cutoff index
    // try each candidate where tail is merged into last bin
    float best_kl = std::numeric_limits<float>::max();
    int best_idx = bins_ - 1;
    int start = quantized_bins_;
    if (start >= bins_) start = bins_ - 1;

    std::vector<float> p(bins_, 0.0f), q(bins_, 0.0f);
    for (int c = start; c < bins_; ++c) {
        // build P: normalized histogram
        for (int i = 0; i < bins_; ++i) p[i] = (float)histogram_[(size_t)i] / (float)total;
        // merge tail beyond c into last kept bin
        float tail = 0.0f;
        for (int i = c + 1; i < bins_; ++i) tail += p[i];
        // build Q: quantize p[0..c] into qlevels bins then expand
        int qbins = std::min(qlevels, c + 1);
        float bin_per_q = (float)(c + 1) / (float)qbins;
        std::vector<float> q_quant(qbins, 0.0f);
        for (int i = 0; i <= c; ++i) {
            int qi = (int)(i / bin_per_q);
            if (qi >= qbins) qi = qbins - 1;
            q_quant[(size_t)qi] += p[i];
        }
        // expand back
        for (int i = 0; i <= c; ++i) {
            int qi = (int)(i / bin_per_q);
            if (qi >= qbins) qi = qbins - 1;
            q[i] = q_quant[(size_t)qi] / bin_per_q;
        }
        // put tail into last kept bin
        q[c] += tail;
        for (int i = c + 1; i < bins_; ++i) q[i] = 0.0f;
        // zero handling
        for (int i = 0; i < bins_; ++i) {
            if (p[i] < 1e-12f) p[i] = 1e-12f;
            if (q[i] < 1e-12f) q[i] = 1e-12f;
        }
        float kl = 0.0f;
        for (int i = 0; i < bins_; ++i) kl += p[i] * std::log(p[i] / q[i]);
        if (kl < best_kl) { best_kl = kl; best_idx = c; }
    }
    float bin_width = (hist_max_ - hist_min_) / (float)bins_;
    float thresh = hist_min_ + (best_idx + 1) * bin_width;
    return thresh;
}

QuantParams KLDivergenceObserver::calc_qparams(int bits, bool symmetric) const {
    QuantParams p;
    get_qrange_bits(bits, symmetric, false, p.qmin, p.qmax);
    if (!has_data_) { p.scale = 1.0f; p.zero_point = 0; return p; }
    if (symmetric) {
        float abs_max = std::max(std::fabs(hist_min_), std::fabs(hist_max_));
        // try to refine via KL if asymmetric would help but symmetric just uses abs_max
        // use KL threshold search for asymmetric case only; symmetric fallback to minmax
        if (abs_max < 1e-8f) abs_max = 1e-8f;
        p.scale = abs_max / (float)p.qmax;
        p.zero_point = 0;
    } else {
        float thresh = compute_kl_threshold(bits, symmetric);
        // derive scale from observed range up to thresh
        // if thresh < hist_max_, we clip tail
        float mn = hist_min_;
        float mx = thresh;
        if (mx <= mn) { mx = mn + 1e-6f; }
        float range = mx - mn;
        if (range < 1e-8f) range = 1e-8f;
        p.scale = range / (float)(p.qmax - p.qmin);
        p.zero_point = (int32_t)std::round(-mn / p.scale);
        if (p.zero_point < p.qmin) p.zero_point = p.qmin;
        if (p.zero_point > p.qmax) p.zero_point = p.qmax;
    }
    if (p.scale < 1e-8f) p.scale = 1e-8f;
    return p;
}

// ============================================================================
// QATContext
// ============================================================================
Tensor QATContext::maybe_quantize(const Tensor& x) {
    if (!enabled_) return x;
    int qmin, qmax;
    get_qrange(cfg_, qmin, qmax);
    float scale = cfg_.init_scale;
    if (scale <= 1e-8f) scale = 1e-6f;
    // simple per-tensor scale from max abs if not provided
    // estimate scale from tensor if init_scale is 1.0 and tensor has larger range
    // keep init_scale as is for determinism; caller can calibrate via observer
    return fake_quantize(x, scale, qmin, qmax);
}

Tensor QATContext::maybe_quantize(const Tensor& x, Observer& obs) {
    if (!enabled_) return x;
    return fake_quantize_with_observer(x, obs, cfg_.bits, cfg_.symmetric);
}

Tensor QATContext::maybe_lsq_quantize(const Tensor& x, Tensor& scale_param) {
    if (!enabled_) return x;
    int qmin, qmax;
    get_qrange(cfg_, qmin, qmax);
    return lsq_fake_quantize(x, scale_param, qmin, qmax);
}

} // namespace qat
} // namespace quant
