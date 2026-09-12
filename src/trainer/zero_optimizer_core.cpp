#define NOMINMAX
#include "quant/zero_optimizer.h"
#include "quant/math.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cfloat>
#ifdef _WIN32
#include <windows.h>
#undef min
#undef max
#endif

#if defined(QUANT_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace quant {

// ============================================================================
// Shared communication state
// ============================================================================
std::mutex ZeroOptimizer::comm_mutex_;
std::vector<float> ZeroOptimizer::comm_buffer_;
ZeroOptimizer::ZeroBarrierState ZeroOptimizer::comm_barrier_;

// ============================================================================
// ZeroOptimizer — ZeRO Stage 3
// ============================================================================

ZeroOptimizer::ZeroOptimizer(Model* model, int rank, int world_size,
                              float lr, float beta1, float beta2,
                              float eps, float weight_decay)
    : model_(model), rank_(rank), world_size_(world_size),
      lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay) {}

ZeroOptimizer::~ZeroOptimizer() = default;

bool ZeroOptimizer::is_param_owned(int global_idx) const {
    return (global_idx % world_size_) == rank_;
}

int ZeroOptimizer::owner_rank(int global_idx) const {
    return global_idx % world_size_;
}

void ZeroOptimizer::register_model_params() {
    params_.clear();
    owned_params_.clear();
    param_meta_.clear();

    DenseModel* dm = dynamic_cast<DenseModel*>(model_);
    if (!dm) return;

    // Collect all trainable parameters from the DenseModel
    auto add_param = [this](Tensor* p) {
        if (p && p->numel() > 0) {
            params_.push_back(p);
        }
    };

    add_param(&dm->tok_embeddings->weight);
    for (auto& layer : dm->layers) {
        add_param(&layer->attention_norm.weight);
        add_param(&layer->attention.q_proj.weight);
        add_param(&layer->attention.q_proj.bias);
        add_param(&layer->attention.k_proj.weight);
        add_param(&layer->attention.k_proj.bias);
        add_param(&layer->attention.v_proj.weight);
        add_param(&layer->attention.v_proj.bias);
        add_param(&layer->attention.o_proj.weight);
        add_param(&layer->attention.o_proj.bias);
        add_param(&layer->ffn_norm.weight);
        add_param(&layer->ffn.gate_proj.weight);
        add_param(&layer->ffn.gate_proj.bias);
        add_param(&layer->ffn.up_proj.weight);
        add_param(&layer->ffn.up_proj.bias);
        add_param(&layer->ffn.down_proj.weight);
        add_param(&layer->ffn.down_proj.bias);
    }
    add_param(&dm->norm->weight);
    add_param(&dm->lm_head->weight);
    add_param(&dm->lm_head->bias);

    build_partition();
}

void ZeroOptimizer::build_partition() {
    param_meta_.clear();
    param_meta_.reserve(params_.size());
    for (size_t i = 0; i < params_.size(); i++) {
        RemoteState rs;
        rs.numel = params_[i]->numel();
        rs.owned_locally = is_param_owned((int)i);
        param_meta_.push_back(rs);
    }

    owned_params_.clear();
    for (size_t i = 0; i < params_.size(); i++) {
        if (is_param_owned((int)i)) {
            OwnedParam op;
            op.global_idx = (int)i;
            op.m = Tensor::zeros(params_[i]->shape());
            op.v = Tensor::zeros(params_[i]->shape());
            owned_params_.push_back(op);
        }
    }
}

ZeroOptimizer::ShardInfo ZeroOptimizer::get_shard(int param_idx) const {
    ShardInfo info;
    info.param_idx = param_idx;
    info.offset = 0;
    info.size = (param_idx >= 0 && (size_t)param_idx < param_meta_.size())
                ? param_meta_[param_idx].numel : 0;
    info.owns = is_param_owned(param_idx);
    return info;
}

void ZeroOptimizer::zero_grad() {
    for (auto* p : params_) {
        if (p->requires_grad() && p->has_grad()) {
            p->grad().zero_();
        }
    }
}

float ZeroOptimizer::clip_grad_norm() {
    if (grad_clip_norm_ <= 0.0f || params_.empty()) return 0.0f;
    double total_norm = 0.0;
    for (auto* p : params_) {
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
    for (auto* p : params_) {
        if (!p->has_grad()) continue;
        float* g = p->grad().data<float>();
        int64_t n = p->numel();
        for (int64_t i = 0; i < n; i++) g[i] *= scale;
    }
    return (float)total_norm;
}

// Shared barrier helper used by reduce_scatter and internal_all_gather.
// BUGFIX (bug census): was `static thread_local BarrierState` — each thread
// waited on its OWN counter and never rendezvoused (hang for world_size > 1,
// plus the same reset-counter flaw as DistributedContext). Rendezvous state
// is ZeroOptimizer::comm_barrier_ (shared, generation-counted).
static void zero_barrier(int world_size) {
    auto& bar = ZeroOptimizer::comm_barrier_;
    std::unique_lock<std::mutex> lock(bar.mtx);
    int gen = bar.gen;
    bar.count++;
    if (bar.count < world_size) {
        bar.cv.wait(lock, [&] { return bar.gen != gen; });
    } else {
        bar.count = 0;
        bar.gen++;
        bar.cv.notify_all();
    }
}

void ZeroOptimizer::reduce_scatter_grads() {
    if (world_size_ <= 1) return;

    // Trigger gradient hooks first
    if (grad_hook_) {
        for (size_t i = 0; i < params_.size(); i++) {
            if (params_[i]->has_grad())
                grad_hook_((int)i);
        }
    }

    // For each parameter, reduce-scatter the gradient so only the owning
    // rank keeps its chunk.
    for (size_t global_idx = 0; global_idx < params_.size(); global_idx++) {
        Tensor* p = params_[global_idx];
        if (!p->has_grad() || p->grad().numel() == 0) continue;

        int64_t n = p->numel();
        int owner = owner_rank((int)global_idx);

        // All-reduce the gradient
        {
            std::lock_guard<std::mutex> lock(comm_mutex_);
            if (comm_buffer_.size() < (size_t)n)
                comm_buffer_.resize(n, 0.0f);
            const float* g = p->grad().data<float>();
            for (int64_t i = 0; i < n; i++)
                comm_buffer_[i] += g[i];
        }

        zero_barrier(world_size_);

        // Copy reduced gradient; only owning rank keeps it
        {
            std::lock_guard<std::mutex> lock(comm_mutex_);
            std::memcpy(p->grad().data<float>(), comm_buffer_.data(), n * sizeof(float));
            if (rank_ != owner)
                p->grad().zero_();
        }
    }
}

void ZeroOptimizer::step() {
    if (owned_params_.empty()) return;
    t_++;
    clip_grad_norm();

    float bias_corr1 = 1.0f - static_cast<float>(std::pow(beta1_, t_));
    float bias_corr2 = 1.0f - static_cast<float>(std::pow(beta2_, t_));
    float lr_adj = lr_ * std::sqrt(bias_corr2) / bias_corr1;

    // Update only owned parameters (AdamW)
    for (auto& op : owned_params_) {
        Tensor* param = params_[op.global_idx];
        if (!param->requires_grad() || !param->has_grad() || param->grad().numel() == 0)
            continue;

        float* p = param->data<float>();
        const float* g = param->grad().data<float>();
        float* m = op.m.data<float>();
        float* v = op.v.data<float>();
        int64_t n = param->numel();

        for (int64_t i = 0; i < n; i++) {
            // Decoupled weight decay
            p[i] -= lr_ * weight_decay_ * p[i];
            // Biased first moment estimate
            m[i] = beta1_ * m[i] + (1.0f - beta1_) * g[i];
            // Biased second raw moment estimate
            v[i] = beta2_ * v[i] + (1.0f - beta2_) * g[i] * g[i];
            // Update with bias correction
            p[i] -= lr_adj * m[i] / (std::sqrt(v[i] / bias_corr2) + eps_);
        }

        // Zero the gradient on owned params after update
        param->grad().zero_();
    }
}

void ZeroOptimizer::internal_all_gather(float* local_data, int64_t local_bytes) {
    int64_t local_floats = local_bytes / sizeof(float);
    int64_t total_floats = local_floats * world_size_;

    {
        std::lock_guard<std::mutex> lock(comm_mutex_);
        if (comm_buffer_.size() < (size_t)total_floats)
            comm_buffer_.resize(total_floats);
        std::memcpy(&comm_buffer_[rank_ * local_floats],
                    local_data, local_bytes);
    }

    zero_barrier(world_size_);

    // Each rank copies the full gathered buffer
    for (int r = 0; r < world_size_; r++) {
        std::lock_guard<std::mutex> lock(comm_mutex_);
        std::memcpy(local_data + r * local_floats,
                    &comm_buffer_[r * local_floats], local_bytes);
    }
}

void ZeroOptimizer::internal_reduce_scatter(float* full_data, float* local_out,
                                              int64_t local_bytes) {
    int64_t local_floats = local_bytes / sizeof(float);
    int64_t total_floats = local_floats * world_size_;

    // Accumulate full data into shared buffer
    {
        std::lock_guard<std::mutex> lock(comm_mutex_);
        if (comm_buffer_.size() < (size_t)total_floats)
            comm_buffer_.resize(total_floats, 0.0f);
        for (int64_t i = 0; i < (int64_t)total_floats && i < (int64_t)comm_buffer_.size(); i++)
            comm_buffer_[i] += full_data[i];
    }

    zero_barrier(world_size_);

    // Each rank copies its chunk
    {
        std::lock_guard<std::mutex> lock(comm_mutex_);
        std::memcpy(local_out, &comm_buffer_[rank_ * local_floats], local_bytes);
        std::fill(comm_buffer_.begin(), comm_buffer_.begin() + total_floats, 0.0f);
    }
}

void ZeroOptimizer::all_gather_params() {
    if (world_size_ <= 1) return;

    // For each owned parameter, all-gather the updated values to all ranks
    for (auto& op : owned_params_) {
        Tensor* param = params_[op.global_idx];
        int64_t n = param->numel();
        int64_t byte_count = n * sizeof(float);

        // Buffer layout: each rank has its partition of the full param space
        // For simplicity: each rank has {local_data, local_data, ...} for non-owned
        // and the actual data for owned. We all-gather the owned data.
        float* local_data = param->data<float>();
        internal_all_gather(local_data, byte_count);

        // Invoke forward hook after gather
        if (forward_hook_)
            forward_hook_(op.global_idx);
    }
}

int64_t ZeroOptimizer::state_bytes_owned() const {
    int64_t total = 0;
    for (auto& op : owned_params_) {
        total += op.m.numel() * sizeof(float) * 2; // m + v
    }
    return total;
}

int64_t ZeroOptimizer::state_bytes_full() const {
    int64_t total = 0;
    for (auto& meta : param_meta_) {
        total += meta.numel * sizeof(float) * 2; // m + v
    }
    return total;
}

float ZeroOptimizer::memory_savings_ratio() const {
    int64_t full = state_bytes_full();
    int64_t owned = state_bytes_owned();
    return full > 0 ? 1.0f - (float)owned / full : 0.0f;
}

const ZeroOptimizer::OptimizerState& ZeroOptimizer::get_owned_state(size_t idx) const {
    static OptimizerState empty;
    if (idx >= owned_params_.size()) return empty;
    static thread_local OptimizerState os;
    auto& op = owned_params_[idx];
    os.param = params_[op.global_idx];
    os.m = op.m;
    os.v = op.v;
    os.is_owned = true;
    os.global_idx = op.global_idx;
    os.numel = op.m.numel();
    if (os.param && os.param->has_grad())
        os.grad = &os.param->grad();
    return os;
}

// ============================================================================
// CPUOffloadEngine
// ============================================================================

CPUOffloadEngine::CPUOffloadEngine(int64_t cpu_buffer_size, int num_streams)
    : cpu_buffer_size_(cpu_buffer_size), num_streams_(std::max(1, num_streams)) {
    cpu_pool_ = allocate_cpu(cpu_buffer_size_);
}

CPUOffloadEngine::~CPUOffloadEngine() {
    if (running_.exchange(false)) {
        queue_cv_.notify_all();
        if (worker_thread_.joinable())
            worker_thread_.join();
    }
    deallocate_cpu(cpu_pool_);
}

void* CPUOffloadEngine::allocate_cpu(int64_t size_bytes) {
    if (size_bytes <= 0) return nullptr;
#ifdef _WIN32
    void* ptr = VirtualAlloc(nullptr, size_bytes,
                              MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!ptr) ptr = std::malloc(size_bytes);
#else
    void* ptr = nullptr;
    int ret = posix_memalign(&ptr, 64, size_bytes);
    if (ret != 0) ptr = std::malloc(size_bytes);
#endif
    if (ptr) std::memset(ptr, 0, size_bytes);
    return ptr;
}

void CPUOffloadEngine::deallocate_cpu(void* ptr) {
    if (!ptr) return;
#ifdef _WIN32
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    std::free(ptr);
#endif
}

void CPUOffloadEngine::pin_memory(void* ptr, int64_t size_bytes) {
    (void)ptr; (void)size_bytes;
    // On systems with real CUDA, this would call cudaHostRegister
}

void CPUOffloadEngine::unpin_memory(void* ptr) {
    (void)ptr;
}

void CPUOffloadEngine::offload_params(const std::vector<float*>& params,
                                       const std::vector<int64_t>& sizes) {
    for (size_t i = 0; i < params.size() && i < sizes.size(); i++) {
        offload_single((int)i, params[i], sizes[i] * sizeof(float));
    }
}

void CPUOffloadEngine::offload_single(int param_id, float* data, int64_t size_bytes) {
    if (size_bytes <= 0) return;

    auto& region = param_regions_[param_id];
    if (!region.cpu_ptr) {
        if (cpu_pool_used_ + size_bytes > cpu_buffer_size_) {
            // Buffer full; wait for transfers to complete and reuse
            synchronize();
            cpu_pool_used_ = 0;
        }
        region.cpu_ptr = static_cast<char*>(cpu_pool_) + cpu_pool_used_;
        region.size_bytes = size_bytes;
        cpu_pool_used_ += size_bytes;
    }

    // Enqueue transfer: device -> CPU
    async_transfer(static_cast<float*>(region.cpu_ptr), data, size_bytes,
                   TransferDir::DEVICE_TO_CPU, param_id);
    region.offloaded = true;
    region.prefetched = false;
    bytes_offloaded_ += size_bytes;
}

void CPUOffloadEngine::prefetch_params(const std::vector<int>& param_ids) {
    for (int pid : param_ids) {
        auto it = param_regions_.find(pid);
        if (it == param_regions_.end() || !it->second.offloaded) continue;

        // Find the corresponding parameter data
        // (caller must have set up the mapping)
        async_transfer(nullptr, static_cast<const float*>(it->second.cpu_ptr),
                       it->second.size_bytes, TransferDir::CPU_TO_DEVICE, pid);
        it->second.prefetched = true;
        bytes_prefetched_ += it->second.size_bytes;
    }
}

void CPUOffloadEngine::prefetch_all() {
    for (auto& kv : param_regions_) {
        if (kv.second.offloaded && !kv.second.prefetched) {
            async_transfer(nullptr, static_cast<const float*>(kv.second.cpu_ptr),
                           kv.second.size_bytes, TransferDir::CPU_TO_DEVICE, kv.first);
            kv.second.prefetched = true;
            bytes_prefetched_ += kv.second.size_bytes;
        }
    }
}

void CPUOffloadEngine::async_transfer(float* dst, const float* src,
                                       int64_t size_bytes, TransferDir dir,
                                       int param_id) {
    TransferRequest req;
    req.dst = dst;
    req.src = src;
    req.size_bytes = size_bytes;
    req.dir = dir;
    req.param_id = param_id;
    req.priority = Priority::NORMAL;

    // For double-buffered mode, use the buffer pair
    auto buf_it = buffers_.find(param_id);
    if (buf_it != buffers_.end()) {
        auto& bp = buf_it->second;
        int next_buf = 1 - bp.active;
        if (dir == TransferDir::DEVICE_TO_CPU) {
            req.src = (next_buf == 0) ? bp.buf_a.data() : bp.buf_b.data();
        } else {
            req.dst = (next_buf == 0) ? bp.buf_a.data() : bp.buf_b.data();
        }
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (!running_.load()) {
            running_ = true;
            worker_thread_ = std::thread(&CPUOffloadEngine::worker_loop, this);
        }
        transfer_queue_.push_back(std::move(req));
        pending_count_++;
    }
    queue_cv_.notify_one();
}

void CPUOffloadEngine::double_buffer_transfer(float* compute_buf, float* transfer_buf,
                                               int64_t buf_size, TransferDir dir) {
    // Double-buffering: compute uses compute_buf while transfer_buf transfers
    int64_t half = buf_size / 2;

    // First half: transfer while compute uses second half
    async_transfer(transfer_buf, compute_buf + half / sizeof(float),
                   half, dir, -1);

    // Second half: transfer while compute uses first half
    async_transfer(transfer_buf + half / sizeof(float), compute_buf,
                   half, dir, -1);

    synchronize();
}

void CPUOffloadEngine::synchronize() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.wait(lock, [this] { return pending_count_ == 0; });
}

int CPUOffloadEngine::pending_transfers() const {
    return pending_count_.load();
}

bool CPUOffloadEngine::is_busy() const {
    return pending_count_.load() > 0;
}

void CPUOffloadEngine::clear() {
    synchronize();
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        transfer_queue_.clear();
    }
    param_regions_.clear();
    buffers_.clear();
}

void CPUOffloadEngine::reset_stats() {
    bytes_offloaded_ = 0;
    bytes_prefetched_ = 0;
}

void CPUOffloadEngine::worker_loop() {
    while (running_.load()) {
        TransferRequest req;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !transfer_queue_.empty() || !running_.load();
            });
            if (!running_.load()) return;
            req = std::move(transfer_queue_.front());
            transfer_queue_.pop_front();
        }

        // Perform the transfer
        if (req.src && req.dst && req.size_bytes > 0) {
            std::memcpy(req.dst, req.src, req.size_bytes);
        }

        req.completed = true;

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            pending_count_--;
        }
        queue_cv_.notify_all();
    }
}

} // namespace quant
