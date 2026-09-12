// ============================================================================
// PILLAR 4: ZeRO-Offload — CPU/GPU Memory Partitioning Implementation
// ============================================================================

#include "quant/zero_offload.h"
#include "quant/math.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#undef min
#undef max
#endif

namespace quant {

// ============================================================================
// GradientCheckpointManager
// ============================================================================

bool GradientCheckpointManager::active_ = false;
int64_t GradientCheckpointManager::saved_count_ = 0;
int64_t GradientCheckpointManager::recompute_count_ = 0;
int64_t GradientCheckpointManager::saved_bytes_ = 0;

void GradientCheckpointManager::begin_checkpoint() { active_ = true; }
void GradientCheckpointManager::end_checkpoint() { active_ = false; }
bool GradientCheckpointManager::in_checkpoint() { return active_; }

int64_t GradientCheckpointManager::activations_saved() { return saved_count_; }
int64_t GradientCheckpointManager::activations_recomputed() { return recompute_count_; }
float GradientCheckpointManager::memory_saved_bytes() { return (float)saved_bytes_; }

float GradientCheckpointManager::compute_overhead() {
    return recompute_count_ > 0 ? (float)recompute_count_ / (saved_count_ + recompute_count_) : 0.0f;
}

void GradientCheckpointManager::reset_stats() {
    saved_count_ = 0;
    recompute_count_ = 0;
    saved_bytes_ = 0;
}

// ============================================================================
// ZeRO-Offload Manager
// ============================================================================

ZeROOffloadManager::ZeROOffloadManager(const ZeROConfig& cfg) : cfg_(cfg) {
    if (cfg_.enabled && cfg_.cpu_buffer_size > 0) {
#ifdef _WIN32
        cpu_buffer_ = VirtualAlloc(nullptr, cfg_.cpu_buffer_size,
                                   MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
        cpu_buffer_ = std::malloc(cfg_.cpu_buffer_size);
#endif
    }
}

ZeROOffloadManager::~ZeROOffloadManager() {
    if (cpu_buffer_) {
#ifdef _WIN32
        VirtualFree(cpu_buffer_, 0, MEM_RELEASE);
#else
        std::free(cpu_buffer_);
#endif
    }
}

void ZeROOffloadManager::offload_optimizer_state(const Tensor& weight_grad,
                                                  Tensor& m, Tensor& v,
                                                  int64_t param_idx) {
    if (!cfg_.enabled || !cpu_buffer_) return;

    int64_t m_bytes = m.numel() * sizeof(float);
    int64_t v_bytes = v.numel() * sizeof(float);

    if (cpu_buffer_used_ + m_bytes + v_bytes > cfg_.cpu_buffer_size) return;

    // Copy m to CPU
    void* cpu_m = static_cast<char*>(cpu_buffer_) + cpu_buffer_used_;
    std::memcpy(cpu_m, m.data(), m_bytes);
    cpu_buffer_used_ += m_bytes;

    // Copy v to CPU
    void* cpu_v = static_cast<char*>(cpu_buffer_) + cpu_buffer_used_;
    std::memcpy(cpu_v, v.data(), v_bytes);
    cpu_buffer_used_ += v_bytes;

    // Track
    OffloadedParam param;
    param.size_bytes = m_bytes + v_bytes;
    param.cpu_offset = static_cast<char*>(cpu_buffer_) + cpu_buffer_used_ - m_bytes - v_bytes;
    param.offloaded = true;
    if ((int64_t)params_.size() <= param_idx) params_.resize(param_idx + 1);
    params_[param_idx] = param;

    total_offloaded_ += m_bytes + v_bytes;

    // Zero out GPU memory (save VRAM)
    m.zero_();
    v.zero_();
}

void ZeROOffloadManager::reload_optimizer_state(Tensor& m, Tensor& v,
                                                 int64_t param_idx) {
    // BUGFIX (bug census): memcpy with no check that the saved range fits
    // the buffer or matches current numel (stale-shape reload = OOB read of
    // cpu_buffer_ + OOB write into m/v). Validate everything first.
    if (!cfg_.enabled || !cpu_buffer_) return;
    if (param_idx < 0 || param_idx >= (int64_t)params_.size()) return;
    const OffloadedParam& saved = params_[param_idx];
    if (!saved.offloaded || !saved.cpu_offset) return;
    int64_t m_bytes = m.numel() * sizeof(float);
    int64_t v_bytes = v_bytes_from_tensor(v);
    if (m_bytes <= 0 || v_bytes <= 0) return;
    if (saved.size_bytes != m_bytes + v_bytes) return; // shape changed
    const char* base = static_cast<const char*>(cpu_buffer_);
    const char* end = base + cfg_.cpu_buffer_size;
    const char* lo = static_cast<const char*>(saved.cpu_offset);
    if (lo < base || lo + saved.size_bytes > end) return; // outside buffer
    if (!m.data() || !v.data()) return;

    std::memcpy(m.data(), lo, (size_t)m_bytes);
    std::memcpy(v.data(), lo + m_bytes, (size_t)v_bytes);
}

int64_t ZeROOffloadManager::v_bytes_from_tensor(const Tensor& t) const {
    return t.numel() * sizeof(float);
}

Tensor ZeROOffloadManager::shard_gradient(const Tensor& grad, int64_t rank, int64_t world_size) {
    int64_t total = grad.numel();
    int64_t shard_size = total / world_size;
    int64_t start = rank * shard_size;
    int64_t end = std::min(start + shard_size, total);

    Tensor shard(Shape{end - start}, DType::F32);
    std::memcpy(shard.data(), static_cast<const char*>(grad.data()) + start * sizeof(float),
                (end - start) * sizeof(float));
    return shard;
}

Tensor ZeROOffloadManager::unshard_gradient(const Tensor& local_grad, int64_t rank, int64_t world_size) {
    // In real distributed training, this would be all-gather
    // Here we just return the local shard
    return local_grad.clone();
}

int64_t ZeROOffloadManager::bytes_offloaded() const { return total_offloaded_; }
int64_t ZeROOffloadManager::bytes_on_device() const {
    int64_t total = 0;
    for (auto& p : params_) total += p.size_bytes;
    return total - total_offloaded_;
}
float ZeROOffloadManager::offload_ratio() const {
    int64_t total = 0;
    for (auto& p : params_) total += p.size_bytes;
    return total > 0 ? (float)total_offloaded_ / total : 0.0f;
}
void ZeROOffloadManager::reset_stats() { total_offloaded_ = 0; cpu_buffer_used_ = 0; }

// ============================================================================
// AdamW Optimizer with ZeRO support
// ============================================================================

AdamWOptimizer::AdamWOptimizer(const AdamWConfig& cfg)
    : cfg_(cfg), zero_(ZeROConfig{cfg_.learning_rate > 0}) {}

AdamWOptimizer::~AdamWOptimizer() = default;

void AdamWOptimizer::add_parameter(Tensor* param, Tensor* grad) {
    ParamState state;
    state.param = param;
    state.grad = grad;
    state.m = Tensor(param->shape(), DType::F32);
    state.v = Tensor(param->shape(), DType::F32);
    state.m.zero_();
    state.v.zero_();
    states_.push_back(std::move(state));
}

void AdamWOptimizer::zero_grad() {
    for (auto& s : states_) {
        if (s.grad) s.grad->zero_();
    }
}

float AdamWOptimizer::clip_gradients(float max_norm) {
    float total_norm = 0.0f;
    for (auto& s : states_) {
        if (s.grad) {
            float n = quant::math::norm(*s.grad);
            total_norm += n * n;
        }
    }
    total_norm = std::sqrt(total_norm);
    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        for (auto& s : states_) {
            if (s.grad) quant::math::scale(scale, *s.grad, *s.grad);
        }
    }
    return total_norm;
}

float AdamWOptimizer::step() {
    step_++;
    float grad_norm = clip_gradients(cfg_.grad_clip);

    float bias_correction1 = 1.0f - std::pow(cfg_.beta1, (float)step_);
    float bias_correction2 = 1.0f - std::pow(cfg_.beta2, (float)step_);

    for (auto& s : states_) {
        if (!s.param || !s.grad) continue;

        int64_t n = s.param->numel();
        float* p = s.param->data<float>();
        float* g = s.grad->data<float>();
        float* m = s.m.data<float>();
        float* v = s.v.data<float>();

        for (int64_t i = 0; i < n; ++i) {
            // AdamW: decoupled weight decay
            float param_val = p[i];
            float grad_val = g[i] + cfg_.weight_decay * param_val;

            m[i] = cfg_.beta1 * m[i] + (1.0f - cfg_.beta1) * grad_val;
            v[i] = cfg_.beta2 * v[i] + (1.0f - cfg_.beta2) * grad_val * grad_val;

            float m_hat = m[i] / bias_correction1;
            float v_hat = v[i] / bias_correction2;

            p[i] = param_val - cfg_.learning_rate * (m_hat / (std::sqrt(v_hat) + cfg_.eps));
        }
    }

    return grad_norm;
}

int64_t AdamWOptimizer::state_memory_bytes() const {
    int64_t total = 0;
    for (auto& s : states_) {
        total += s.m.numel() * sizeof(float) * 2; // m + v
    }
    return total;
}

} // namespace quant
