#pragma once

#include "quant/tensor.h"
#include "quant/model.h"
#include "quant/optimizer.h"
#include "quant/distributed.h"
#include "quant/autograd.h"
#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <memory>
#include <unordered_map>

namespace quant {

// ============================================================================
// ZeroOptimizer — ZeRO Stage 3: Sharded optimizer state across ranks
// ============================================================================
// Each rank owns optimizer states for a partition of parameters.
// Step() updates only the owned partition; all_gather_params() broadcasts
// updated params to all ranks; reduce_scatter_grads() collects gradients
// to the owning rank.
// ============================================================================
class ZeroOptimizer {
public:
    struct ShardInfo {
        int param_idx;
        int64_t offset;
        int64_t size;
        bool owns;
    };

    // Hook types for forward/backward automatic grad management
    using GradHook = std::function<void(int param_idx)>;
    using ForwardPreHook = std::function<void(int param_idx)>;

    ZeroOptimizer(Model* model, int rank, int world_size,
                  float lr = 3e-4f, float beta1 = 0.9f, float beta2 = 0.999f,
                  float eps = 1e-8f, float weight_decay = 1e-2f);
    ~ZeroOptimizer();

    // Register model parameters into the optimizer
    void register_model_params();

    // Zero-3 core operations
    void step();
    void all_gather_params();
    void reduce_scatter_grads();

    // Zero all gradients across all parameters
    void zero_grad();

    // Gradient clipping (global norm)
    void set_grad_clip_norm(float max_norm) { grad_clip_norm_ = max_norm; }
    float clip_grad_norm();

    // Scheduler: adjust learning rate
    void set_lr(float lr) { lr_ = lr; }
    float get_lr() const { return lr_; }

    // Accessors
    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    int step_count() const { return t_; }
    int64_t owned_params() const { return (int64_t)owned_params_.size(); }
    int64_t total_params() const { return (int64_t)params_.size(); }

    // Per-parameter forward/backward hooks (attached via register_param)
    void register_grad_hook(GradHook hook) { grad_hook_ = std::move(hook); }
    void register_forward_hook(ForwardPreHook hook) { forward_hook_ = std::move(hook); }

    // Get shard info for a parameter
    ShardInfo get_shard(int param_idx) const;

    // Memory savings report
    int64_t state_bytes_owned() const;
    int64_t state_bytes_full() const;
    float memory_savings_ratio() const;

    // Access optimizer state tensors for a given owned param index
    struct OptimizerState {
        Tensor* param = nullptr;
        Tensor* grad = nullptr;
        Tensor m;
        Tensor v;
        bool is_owned = false;
        int global_idx = -1;
        int64_t numel = 0;
    };
    OptimizerState get_owned_state(size_t idx) const;
    size_t num_owned() const { return owned_params_.size(); }

private:
    Model* model_;
    int rank_;
    int world_size_;
    float lr_, beta1_, beta2_, eps_, weight_decay_;
    int t_ = 0;
    float grad_clip_norm_ = 0.0f;

    // All parameters of the model (full set)
    std::vector<Tensor*> params_;

    // Optimizer states for owned parameters (1/world_size of total)
    struct OwnedParam {
        Tensor m;             // first moment
        Tensor v;             // second moment
        int global_idx;       // index into params_
        bool owns;            // always true here
    };
    std::vector<OwnedParam> owned_params_;

    // Full optimizer states (only non-owned are non-resident — we track metadata)
    struct RemoteState {
        int64_t numel;
        bool owned_locally;
    };
    std::vector<RemoteState> param_meta_;

    // Hooks
    GradHook grad_hook_;
    ForwardPreHook forward_hook_;

    // Shared-memory communication helpers (reuse DistributedContext pattern).
    // BUGFIX (bug census): the zero_barrier rendezvous state is shared here
    // (was static thread_local = never rendezvous). One instance per process
    // is the supported shape; concurrent instances share the statics, matching
    // the pre-existing comm_buffer_ design.
    // NOTE: comm_* are public so the TU-local zero_barrier() helper in
    // zero_optimizer_core.cpp can rendezvous on them.
public:
    static std::mutex comm_mutex_;
    static std::vector<float> comm_buffer_;
    struct ZeroBarrierState {
        std::mutex mtx;
        std::condition_variable cv;
        int count = 0;
        int gen = 0;
    };
    static ZeroBarrierState comm_barrier_;
private:

    // Partition logic
    bool is_param_owned(int global_idx) const;
    int owner_rank(int global_idx) const;

    // Internal all-gather helper (flat buffer, size per rank = param_bytes)
    void internal_all_gather(float* local_data, int64_t local_bytes);
    void internal_reduce_scatter(float* full_data, float* local_out,
                                  int64_t local_bytes);

    // Build partition metadata
    void build_partition();
};

// ============================================================================
// CPUOffloadEngine — Asymmetric CPU offload with double buffering + prefetch
// ============================================================================
// Offloads optimizer states (m, v) and optionally gradients from device to CPU
// memory. Uses double buffering to overlap transfer with computation.
// Prefetch loads parameters back to device before they are needed.
// ============================================================================
class CPUOffloadEngine {
public:
    enum class TransferDir { CPU_TO_DEVICE, DEVICE_TO_CPU };
    enum class Priority { LOW, NORMAL, HIGH };

    struct TransferRequest {
        float* dst;
        const float* src;
        int64_t size_bytes;
        TransferDir dir;
        Priority priority;
        int param_id;
        bool completed = false;
    };

    struct BufferPair {
        std::vector<float> buf_a;
        std::vector<float> buf_b;
        int active = 0; // 0 = buf_a, 1 = buf_b
    };

    CPUOffloadEngine(int64_t cpu_buffer_size = 8LL * 1024 * 1024 * 1024,  // 8GB
                     int num_streams = 2);
    ~CPUOffloadEngine();

    // Offload parameters to CPU
    void offload_params(const std::vector<float*>& params,
                        const std::vector<int64_t>& sizes);
    void offload_single(int param_id, float* data, int64_t size_bytes);

    // Prefetch parameters from CPU before compute
    void prefetch_params(const std::vector<int>& param_ids);
    void prefetch_all();

    // Double-buffered transfer: overlap compute + transfer
    // While compute uses one buffer, the other transfers to/from CPU
    void double_buffer_transfer(float* compute_buf, float* transfer_buf,
                                 int64_t buf_size, TransferDir dir);

    // Async transfer (non-blocking, returns immediately)
    void async_transfer(float* dst, const float* src, int64_t size_bytes,
                        TransferDir dir, int param_id = -1);

    // Wait for all pending transfers to complete
    void synchronize();

    // Pin memory for faster transfers
    void pin_memory(void* ptr, int64_t size_bytes);
    void unpin_memory(void* ptr);

    // Stats
    int64_t bytes_offloaded() const { return bytes_offloaded_; }
    int64_t bytes_prefetched() const { return bytes_prefetched_; }
    int pending_transfers() const;
    bool is_busy() const;

    // Management
    void clear();
    void reset_stats();

private:
    int64_t cpu_buffer_size_;
    int num_streams_;
    int64_t bytes_offloaded_ = 0;
    int64_t bytes_prefetched_ = 0;

    // CPU pinned memory pool
    void* cpu_pool_ = nullptr;
    int64_t cpu_pool_used_ = 0;
    // BUGFIX (bug census): allocate_cpu falls back to malloc when
    // VirtualAlloc/posix_memalign fails; deallocate_cpu must free with the
    // matching call (VirtualFree on malloc'd memory = heap corruption).
    bool cpu_malloc_fallback_ = false;

    // Async worker thread
    std::thread worker_thread_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<TransferRequest> transfer_queue_;
    std::atomic<bool> running_{false};
    std::atomic<int> pending_count_{0};

    void worker_loop();

    // Track offloaded regions for each param
    struct ParamRegion {
        void* cpu_ptr;
        int64_t size_bytes;
        bool offloaded;
        bool prefetched;
    };
    std::unordered_map<int, ParamRegion> param_regions_;

    // Double-buffer tracking
    std::unordered_map<int, BufferPair> buffers_;

    void* allocate_cpu(int64_t size_bytes);
    void deallocate_cpu(void* ptr);
};

// ============================================================================
// MemoryEfficientAttention — FlashAttention with O(L) memory for 4M context
// ============================================================================
// Implements online softmax with tiling, block-sparse attention for long
// sequences, and gradient checkpointing for the attention computation.
// Supports 4M+ context with O(L) memory, NOT O(L^2).
// ============================================================================
class MemoryEfficientAttention {
public:
    struct MEAConfig {
        int64_t block_size_q = 64;
        int64_t block_size_kv = 64;
        bool causal = true;
        float softmax_scale = 0.0f;  // 0 = auto-compute as 1/sqrt(d)
        float dropout_p = 0.0f;
        bool use_block_sparse = false;
        int64_t block_sparse_window = 1024; // local window size for sparse
        bool gradient_checkpoint = false;
        // 4M context support
        int64_t max_context_tokens = 4 * 1024 * 1024;
    };

    explicit MemoryEfficientAttention(const MEAConfig& cfg);

    // Forward pass: O = softmax(Q*K^T/sqrt(d)) * V with online softmax
    Tensor forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                   const Tensor& mask = Tensor{});

    // Backward pass with gradient checkpointing (recompute forward)
    std::vector<Tensor> backward(const Tensor& grad_output,
                                  const Tensor& Q, const Tensor& K, const Tensor& V,
                                  const Tensor& mask);

    // Block-sparse forward (only attends to local window + global tokens)
    Tensor forward_block_sparse(const Tensor& Q, const Tensor& K, const Tensor& V,
                                 const Tensor& block_indices);

    // Online softmax statistics (for debug/monitoring)
    struct SoftmaxStats {
        float max_val;
        float sum_val;
        int64_t num_blocks;
    };
    SoftmaxStats last_softmax_stats() const { return last_stats_; }

    // Memory usage estimation
    int64_t forward_memory_bytes(int64_t B, int64_t H, int64_t N, int64_t D) const;
    int64_t backward_memory_bytes(int64_t B, int64_t H, int64_t N, int64_t D) const;

    const MEAConfig& config() const { return cfg_; }
    void set_config(const MEAConfig& cfg) { cfg_ = cfg; }

    // Statics: raw pointer interface for maximum performance
    static void online_softmax_tile(
        const float* qk_scores, float* row_max, float* row_sum,
        float* output, const float* v_data,
        int64_t num_rows, int64_t kv_block_size, int64_t head_dim,
        int64_t output_offset);

private:
    MEAConfig cfg_;
    SoftmaxStats last_stats_;
    int64_t forward_seen_ = 0;

    // Saved intermediates for backward (when gradient_checkpoint is false)
    Tensor saved_q_, saved_k_, saved_v_;
    Tensor saved_mask_;
    std::vector<float> saved_row_max_;
    std::vector<float> saved_row_sum_;
    bool saved_ = false;

    // Online softmax helper
    void online_softmax_block(const float* qk_block, int64_t block_size,
                               float* row_max, float* row_sum,
                               float* output, const float* v_block,
                               int64_t head_dim);

    // Gradient checkpoint helper: recompute forward during backward
    std::vector<float> recompute_forward_stats(const Tensor& Q, const Tensor& K,
                                                const Tensor& V, const Tensor& mask);
};

// ============================================================================
// ActivationCheckpoint — Selective activation recomputation system
// ============================================================================
// Marks regions for checkpointing (discard activations, recompute in backward).
// Supports full checkpointing (discard all) and partial (discard non-essential).
// Auto-placement based on memory budget, with usage estimation.
// ============================================================================
class ActivationCheckpoint {
public:
    enum class CheckpointMode {
        NONE,        // no checkpointing
        FULL,        // discard all activations within region
        PARTIAL,     // keep essential activations, discard non-essential
        BALANCED     // auto-balance between memory and compute
    };

    struct MemoryBudget {
        int64_t total_bytes = 0;           // 0 = unlimited
        int64_t activation_bytes = 0;      // bytes allocated for activations
        int64_t param_bytes = 0;           // bytes for parameters
        int64_t grad_bytes = 0;            // bytes for gradients
        int64_t opt_state_bytes = 0;       // bytes for optimizer states

        float usage_ratio() const {
            int64_t total = total_bytes > 0 ? total_bytes : 1;
            return (float)(activation_bytes + param_bytes + grad_bytes + opt_state_bytes) / total;
        }
    };

    struct RegionInfo {
        std::string name;
        int64_t forward_activations_bytes;
        int64_t recompute_cost_flops;
        bool is_checkpointed;
        CheckpointMode mode;
    };

    ActivationCheckpoint();
    explicit ActivationCheckpoint(const MemoryBudget& budget);

    // Mark the start of a checkpoint region
    void begin_region(const std::string& name,
                      CheckpointMode mode = CheckpointMode::FULL);

    // Mark the end of the current checkpoint region
    void end_region();

    // Is the current region checkpointed?
    bool in_checkpoint_region() const { return region_depth_ > 0 && current_region_.is_checkpointed; }

    // Memory budget management
    void set_budget(const MemoryBudget& budget) { budget_ = budget; }
    const MemoryBudget& budget() const { return budget_; }
    MemoryBudget& mutable_budget() { return budget_; }

    // Automatic checkpoint placement based on memory budget
    // Returns regions that should be checkpointed to stay within budget
    std::vector<std::string> auto_place_checkpoints();

    // Estimate memory savings from checkpointing a region
    int64_t estimated_savings(const RegionInfo& region) const;
    int64_t estimated_savings_all() const;

    // Compute overhead: ratio of extra FLOPs from recomputation
    float compute_overhead() const;

    // Stats
    int64_t total_activations_saved() const { return total_saved_; }
    int64_t total_activations_recomputed() const { return total_recomputed_; }
    const std::vector<RegionInfo>& regions() const { return regions_; }

    // Reset all stats and regions
    void reset();

private:
    MemoryBudget budget_;
    std::vector<RegionInfo> regions_;
    RegionInfo current_region_;
    int region_depth_ = 0;
    bool in_region_ = false;
    int64_t total_saved_ = 0;
    int64_t total_recomputed_ = 0;

    // Estimate activation memory for a typical transformer layer
    // (batch_size, seq_len, hidden_dim, num_heads)
    int64_t estimate_layer_activations(int64_t B, int64_t N, int64_t H, int64_t D) const;

    // Decide whether to checkpoint based on budget
    bool should_checkpoint(const RegionInfo& region) const;
};

} // namespace quant
