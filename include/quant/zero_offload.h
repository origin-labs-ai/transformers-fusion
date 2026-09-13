#pragma once
// ============================================================================
// PILLAR 4: ZeRO-Offload — CPU/GPU memory partitioning for massive models
// ============================================================================
// WHY: Kaggle T4 has only 30GB VRAM. A14B model with AdamW optimizer needs
// ~120GB (weights + gradients + optimizer states). ZeRO-Offload shards
// optimizer states and gradients to CPU RAM, keeping only forward activations
// on GPU. For CPU-only training (Ryzen 5600GT), this manages the 14GB RAM.
//
// STRATEGY:
//   ZeRO Stage 1: Shard optimizer states (AdamW m, v) across devices
//   ZeRO Stage 2: Shard gradients across devices
//   Offload: Move optimizer states to CPU when VRAM is full
//
// GRADIENT CHECKPOINTING: Discard activations during forward pass.
// Recompute them during backward pass. Trades ~30% more compute for ~60%
// memory savings. Essential for 1M context window training.
// ============================================================================

#include "quant/tensor.h"
#include "quant/autograd.h"
#include <vector>
#include <deque>
#include <cstdint>
#include <atomic>
#include <functional>
#include <memory>

namespace quant {

// ============================================================================
// Gradient Checkpoint — Discard + Recompute activations
// ============================================================================

class GradientCheckpointManager {
public:
    // Mark a region as checkpointed. Activations will be discarded after
    // forward pass and recomputed during backward pass.
    static void begin_checkpoint();
    static void end_checkpoint();

    // Check if currently inside a checkpoint region
    static bool in_checkpoint();

    // Stats
    static int64_t activations_saved();
    static int64_t activations_recomputed();
    static float memory_saved_bytes();
    static float compute_overhead(); // Ratio of extra forward passes

    static void reset_stats();

private:
    // BUGFIX (bug census): plain statics updated without locks (race/lost
    // stats). Atomics now.
    static std::atomic<bool> active_;
    static std::atomic<int64_t> saved_count_;
    static std::atomic<int64_t> recompute_count_;
    static std::atomic<int64_t> saved_bytes_;
};

// ============================================================================
// ZeRO-Offload Manager — Shard optimizer states to CPU
// ============================================================================

struct ZeROConfig {
    bool enabled = false;
    int64_t offload_threshold_bytes = 0; // Offload when VRAM usage exceeds this
    bool shard_gradients = false;        // ZeRO Stage 2
    bool shard_optimizer_states = true;   // ZeRO Stage 1
    bool pin_cpu_memory = true;          // Pin CPU memory for faster transfers
    int64_t cpu_buffer_size = 1024 * 1024 * 1024; // 1GB CPU buffer for offloading
};

class ZeROOffloadManager {
public:
    explicit ZeROOffloadManager(const ZeROConfig& cfg = ZeROConfig{});
    ~ZeROOffloadManager();

    // Offload optimizer state (m, v tensors) to CPU
    void offload_optimizer_state(const Tensor& weight_grad,
                                 Tensor& m, Tensor& v,
                                 int64_t param_idx);

    // Reload optimizer state from CPU to GPU
    void reload_optimizer_state(Tensor& m, Tensor& v, int64_t param_idx);

    // Shard gradient across ranks (for distributed training)
    Tensor shard_gradient(const Tensor& grad, int64_t rank, int64_t world_size);

    // Unshard gradient (all-gather)
    Tensor unshard_gradient(const Tensor& local_grad, int64_t rank, int64_t world_size);

    // Stats
    int64_t bytes_offloaded() const;
    int64_t bytes_on_device() const;
    float offload_ratio() const;

    void reset_stats();

private:
    int64_t v_bytes_from_tensor(const Tensor& t) const;

    ZeROConfig cfg_;

    // CPU pinned memory buffer for offloaded states
    void* cpu_buffer_ = nullptr;
    int64_t cpu_buffer_used_ = 0;

    // Per-parameter offload tracking
    struct OffloadedParam {
        int64_t size_bytes;
        void* cpu_offset; // Offset into cpu_buffer_
        bool offloaded;
    };
    std::vector<OffloadedParam> params_;

    int64_t total_offloaded_ = 0;
};

// ============================================================================
// AdamW Optimizer with ZeRO support
// ============================================================================
// Standard AdamW with optional CPU offloading of m (first moment) and
// v (second moment) tensors.
// ============================================================================

struct AdamWConfig {
    float learning_rate = 3e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.01f;
    float grad_clip = 1.0f;
    bool amsgrad = false;
};

class AdamWOptimizer {
public:
    explicit AdamWOptimizer(const AdamWConfig& cfg = AdamWConfig{});
    ~AdamWOptimizer();

    // Step: update all parameters
    // Returns total gradient norm before clipping
    float step();

    // Add parameter to optimizer
    void add_parameter(Tensor* param, Tensor* grad);

    // Zero all gradients
    void zero_grad();

    // Gradient clipping
    float clip_gradients(float max_norm);

    // Get optimizer state memory usage
    int64_t state_memory_bytes() const;

    // Access ZeRO offload manager
    ZeROOffloadManager& zero_manager() { return zero_; }
    const ZeROOffloadManager& zero_manager() const { return zero_; }

    // Stats
    int64_t step_count() const { return step_; }
    float learning_rate() const { return cfg_.learning_rate; }

private:
    AdamWConfig cfg_;
    ZeROOffloadManager zero_;

    struct ParamState {
        Tensor* param;
        Tensor* grad;
        Tensor m;  // First moment (velocity)
        Tensor v;  // Second moment (squared gradients)
        int64_t offload_idx = -1;
    };
    std::vector<ParamState> states_;
    int64_t step_ = 0;
};

} // namespace quant
