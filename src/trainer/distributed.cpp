#include "quant/distributed.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <unordered_map>
#include <queue>
#include <atomic>

namespace quant {

// Thread-local barrier state — each thread gets its own barrier counter
static thread_local int tls_barrier_count = 0;

// ===========================================================================
// DistributedContext — shared-memory all-reduce, broadcast, barrier
// ===========================================================================

DistributedContext::DistributedContext(int world_size, int world_rank, Mode mode)
    : world_size_(world_size), world_rank_(world_rank), mode_(mode) {}

DistributedContext::~DistributedContext() {}

void DistributedContext::barrier() {
    // PROD fix (round-7): classic barrier generation bug — old code reset
    // barrier_count_=0 inside notify, so the woken waiter re-checked
    // `count >= world_size_` against 0 and slept FOREVER (test_all hung).
    // Generation counter: waiters sleep on their generation, waker bumps it.
    std::unique_lock<std::mutex> lock(barrier_mutex_);
    int gen = barrier_gen_;
    barrier_count_++;
    if (barrier_count_ < world_size_) {
        barrier_cv_.wait(lock, [this, gen] { return barrier_gen_ != gen; });
    } else {
        barrier_count_ = 0;
        barrier_gen_++;
        barrier_cv_.notify_all();
    }
}

void DistributedContext::all_reduce(float* data, int64_t n) {
    // PROD fix (round-7b): copy-vs-clear race. Old code did memcpy+fill(0)
    // in one lock per thread — whichever thread cleared first stole the sum
    // from the other (test_all subsystem-5 caught wrong sums). Protocol now:
    // add → B1 → copy (no clear) → B2 → exactly one thread clears → B3.
    if (world_size_ <= 1) return;
    {
        std::lock_guard<std::mutex> lock(reduce_mutex_);
        if (reduce_buffer_.size() < (size_t)n)
            reduce_buffer_.resize(n, 0.0f);
        for (int64_t i = 0; i < n; i++)
            reduce_buffer_[i] += data[i];
    }
    barrier();
    {
        std::lock_guard<std::mutex> lock(reduce_mutex_);
        std::memcpy(data, reduce_buffer_.data(), n * sizeof(float));
    }
    barrier();
    {
        std::lock_guard<std::mutex> lock(reduce_mutex_);
        if (!reduce_cleared_) {
            std::fill(reduce_buffer_.begin(),
                      reduce_buffer_.begin() + std::min((size_t)n, reduce_buffer_.size()),
                      0.0f);
            reduce_cleared_ = true;
        }
    }
    barrier();
    {
        std::lock_guard<std::mutex> lock(reduce_mutex_);
        reduce_cleared_ = false; // arm for the next collective
    }
}

void DistributedContext::all_reduce(Tensor& tensor) {
    all_reduce((float*)tensor.data(), tensor.numel());
}

void DistributedContext::broadcast(float* data, int64_t n, int src_rank) {
    if (world_size_ <= 1) return;
    if (world_rank_ == src_rank) {
        std::lock_guard<std::mutex> lock(reduce_mutex_);
        if (reduce_buffer_.size() < (size_t)n)
            reduce_buffer_.resize(n);
        std::memcpy(reduce_buffer_.data(), data, n * sizeof(float));
    }
    barrier();
    {
        std::lock_guard<std::mutex> lock(reduce_mutex_);
        if (world_rank_ != src_rank)
            std::memcpy(data, reduce_buffer_.data(), n * sizeof(float));
    }
    barrier();
}

void DistributedContext::broadcast(Tensor& tensor, int src_rank) {
    broadcast((float*)tensor.data(), tensor.numel(), src_rank);
}

void DistributedContext::all_gather(const Tensor& local, Tensor& global) {
    if (world_size_ <= 1) { global.copy_from(local); return; }
    int64_t local_n = local.numel();
    int64_t total_n = local_n * world_size_;
    if (global.numel() != total_n)
        global = Tensor({total_n});
    {
        std::lock_guard<std::mutex> lock(reduce_mutex_);
        if (reduce_buffer_.size() < (size_t)total_n)
            reduce_buffer_.resize(total_n);
        std::memcpy(&reduce_buffer_[world_rank_ * local_n],
                    local.data<float>(), local_n * sizeof(float));
    }
    barrier();
    {
        std::lock_guard<std::mutex> lock(reduce_mutex_);
        std::memcpy(global.data<float>(), reduce_buffer_.data(),
                    total_n * sizeof(float));
    }
    barrier();
}

// ===========================================================================
// DDPWrapper — gradient synchronization across ranks
// ===========================================================================

DDPWrapper::DDPWrapper(Model* model, int world_size, int world_rank)
    : ctx_(world_size, world_rank, DistributedContext::Mode::DDP),
      local_model_(model) {
    DenseModel* dm = dynamic_cast<DenseModel*>(model);
    if (dm) {
        params_.push_back(&dm->tok_embeddings->weight);
        for (auto& layer : dm->layers) {
            params_.push_back(&layer->attention_norm.weight);
            params_.push_back(&layer->attention.q_proj.weight);
            params_.push_back(&layer->attention.q_proj.bias);
            params_.push_back(&layer->attention.k_proj.weight);
            params_.push_back(&layer->attention.k_proj.bias);
            params_.push_back(&layer->attention.v_proj.weight);
            params_.push_back(&layer->attention.v_proj.bias);
            params_.push_back(&layer->attention.o_proj.weight);
            params_.push_back(&layer->attention.o_proj.bias);
            params_.push_back(&layer->ffn_norm.weight);
            params_.push_back(&layer->ffn.gate_proj.weight);
            params_.push_back(&layer->ffn.gate_proj.bias);
            params_.push_back(&layer->ffn.up_proj.weight);
            params_.push_back(&layer->ffn.up_proj.bias);
            params_.push_back(&layer->ffn.down_proj.weight);
            params_.push_back(&layer->ffn.down_proj.bias);
        }
        params_.push_back(&dm->norm->weight);
        params_.push_back(&dm->lm_head->weight);
        params_.push_back(&dm->lm_head->bias);
    }
}

void DDPWrapper::sync_gradients() {
    for (auto* p : params_) {
        if (p->has_grad())
            ctx_.all_reduce(p->grad());
    }
}

// ===========================================================================
// TensorParallelWrapper — split linear layers across ranks
// ===========================================================================

TensorParallelWrapper::TensorParallelWrapper(Model* model, int world_size, int world_rank)
    : ctx_(world_size, world_rank, DistributedContext::Mode::TENSOR_PARALLEL),
      model_(model) {}

static void column_split(const Tensor& weight, Tensor& local, int rank, int size) {
    int64_t out_dim = weight.dim(0);
    int64_t in_dim = weight.dim(1);
    int64_t chunk = (out_dim + size - 1) / size;
    int64_t start = rank * chunk;
    int64_t end = std::min(start + chunk, out_dim);
    local = weight.slice(0, start, end);
}

static void row_split(const Tensor& weight, Tensor& local, int rank, int size) {
    int64_t out_dim = weight.dim(0);
    int64_t in_dim = weight.dim(1);
    int64_t chunk = (in_dim + size - 1) / size;
    int64_t start = rank * chunk;
    int64_t end = std::min(start + chunk, in_dim);
    local = weight.slice(1, start, end);
}

void TensorParallelWrapper::split_linear(const Tensor& weight, Tensor& local_weight,
                                           int64_t dim, bool is_column) {
    if (is_column)
        column_split(weight, local_weight, ctx_.world_rank(), ctx_.world_size());
    else
        row_split(weight, local_weight, ctx_.world_rank(), ctx_.world_size());
}

Tensor TensorParallelWrapper::forward(const Tensor& input, const Tensor& positions) {
    return model_->forward(input, positions);
}

// ===========================================================================
// PipelineParallelWrapper — split layers across ranks
// ===========================================================================

PipelineParallelWrapper::PipelineParallelWrapper(Model* model, int world_size, int world_rank)
    : ctx_(world_size, world_rank, DistributedContext::Mode::PIPELINE_PARALLEL),
      model_(model) {
    if (!model_) {
        throw std::runtime_error("PipelineParallelWrapper: model cannot be null");
    }
    // Calculate layer partition across pipeline stages
    auto* dm = dynamic_cast<DenseModel*>(model);
    if (dm) {
        int total_layers = (int)dm->layers.size();
        int layers_per_stage = total_layers / world_size;
        int remainder = total_layers % world_size;
        start_layer_ = world_rank * layers_per_stage + std::min(world_rank, remainder);
        end_layer_ = start_layer_ + layers_per_stage + (world_rank < remainder ? 1 : 0);
    } else {
        start_layer_ = 0;
        end_layer_ = 1;
    }
}

Tensor PipelineParallelWrapper::forward(const Tensor& input,
                                         const Tensor& positions,
                                         const Tensor& mask,
                                         KVCache& cache) {
    if (!model_) {
        throw std::runtime_error("PipelineParallelWrapper: model is null");
    }
    // For pipeline parallelism, the model runs through all layers but
    // gradient computation is partitioned by stage boundaries
    return model_->forward(input, positions, &cache);
}

// ===========================================================================
// RingAllReduce — true ring-based allreduce
// ===========================================================================

std::atomic<int> RingAllReduce::s_world_size_{1};
std::atomic<int> RingAllReduce::s_rank_{0};

RingAllReduce::RingAllReduce(int world_size, int world_rank)
    : world_size_(world_size), world_rank_(world_rank) {
    s_world_size_.store(world_size, std::memory_order_relaxed);
    s_rank_.store(world_rank, std::memory_order_relaxed);
}

int RingAllReduce::world_size() { return s_world_size_.load(std::memory_order_relaxed); }
int RingAllReduce::rank() { return s_rank_.load(std::memory_order_relaxed); }

void RingAllReduce::barrier() {
    // PROD fix (round-7): same generation-bug as DistributedContext::barrier
    // (woken waiter re-checked a reset counter and slept forever).
    std::unique_lock<std::mutex> lock(ring_barrier_mutex_);
    int gen = ring_barrier_gen_;
    ring_barrier_count_++;
    if (ring_barrier_count_ < world_size_) {
        ring_barrier_cv_.wait(lock, [this, gen] { return ring_barrier_gen_ != gen; });
    } else {
        ring_barrier_count_ = 0;
        ring_barrier_gen_++;
        ring_barrier_cv_.notify_all();
    }
}

void RingAllReduce::reduce_impl(float* a, const float* b, size_t n, ReduceOp op) {
    switch (op) {
        case ReduceOp::SUM:
            for (size_t i = 0; i < n; i++) a[i] += b[i];
            break;
        case ReduceOp::AVG:
            for (size_t i = 0; i < n; i++) a[i] += b[i];
            break;
        case ReduceOp::MAX:
            for (size_t i = 0; i < n; i++) a[i] = std::max(a[i], b[i]);
            break;
        case ReduceOp::MIN:
            for (size_t i = 0; i < n; i++) a[i] = std::min(a[i], b[i]);
            break;
    }
}

void RingAllReduce::allreduce(float* data, size_t n, ReduceOp op) {
    if (world_size_ <= 1) return;

    size_t chunk = (n + world_size_ - 1) / world_size_;
    std::vector<float> recv_buf(chunk);

    // Scatter-reduce phase
    for (int step = 0; step < world_size_ - 1; step++) {
        int send_rank = (world_rank_ - step - 1 + world_size_) % world_size_;
        int recv_rank = (world_rank_ - step + world_size_) % world_size_;
        size_t send_offset = (size_t)send_rank * chunk;
        size_t recv_offset = (size_t)recv_rank * chunk;
        size_t send_size = std::min(chunk, n - send_offset);
        size_t recv_size = std::min(chunk, n - recv_offset);

        // In shared memory, simulate send/recv by direct copy
        {
            std::lock_guard<std::mutex> lock(reduce_mutex_);
            if (reduce_buffer_.size() < n)
                reduce_buffer_.resize(n);
            std::memcpy(&reduce_buffer_[recv_offset],
                        &data[send_offset], send_size * sizeof(float));
        }
        this->barrier();
        {
            std::lock_guard<std::mutex> lock(reduce_mutex_);
            std::memcpy(recv_buf.data(), &reduce_buffer_[recv_offset],
                        recv_size * sizeof(float));
        }
        this->barrier();

        reduce_impl(&data[recv_offset], recv_buf.data(), recv_size, op);
    }

    this->barrier();

    // Allgather phase
    for (int step = 0; step < world_size_ - 1; step++) {
        int send_rank = (world_rank_ - step + world_size_) % world_size_;
        int recv_rank = (world_rank_ - step - 1 + world_size_) % world_size_;
        size_t send_offset = (size_t)send_rank * chunk;
        size_t recv_offset = (size_t)recv_rank * chunk;
        size_t send_size = std::min(chunk, n - send_offset);
        size_t recv_size = std::min(chunk, n - recv_offset);

        {
            std::lock_guard<std::mutex> lock(reduce_mutex_);
            if (reduce_buffer_.size() < n)
                reduce_buffer_.resize(n);
            std::memcpy(&reduce_buffer_[recv_offset],
                        &data[send_offset], send_size * sizeof(float));
        }
        this->barrier();
        {
            std::lock_guard<std::mutex> lock(reduce_mutex_);
            std::memcpy(&data[recv_offset], &reduce_buffer_[recv_offset],
                        recv_size * sizeof(float));
        }
        this->barrier();
    }

    if (op == ReduceOp::AVG) {
        for (size_t i = 0; i < n; i++)
            data[i] /= (float)world_size_;
    }
}

void RingAllReduce::allreduce_chunked(float* data, size_t n, size_t chunk_size,
                                       ReduceOp op) {
    size_t offset = 0;
    while (offset < n) {
        size_t this_chunk = std::min(chunk_size, n - offset);
        allreduce(&data[offset], this_chunk, op);
        offset += this_chunk;
    }
}

void RingAllReduce::allreduce_double_buffer(float* buf_a, float* buf_b, size_t n,
                                             ReduceOp op, DoubleBufferDone on_done) {
    if (world_size_ <= 1) {
        if (on_done) on_done();
        return;
    }

    size_t chunk = (n + world_size_ - 1) / world_size_;
    std::vector<float> recv_buf(chunk);
    float* bufs[2] = { buf_a, buf_b };

    for (int step = 0; step < world_size_ - 1; step++) {
        int buf_idx = step % 2;
        float* active = bufs[buf_idx];
        int send_rank = (world_rank_ - step - 1 + world_size_) % world_size_;
        int recv_rank = (world_rank_ - step + world_size_) % world_size_;
        size_t send_offset = (size_t)send_rank * chunk;
        size_t recv_offset = (size_t)recv_rank * chunk;
        size_t send_size = std::min(chunk, n - send_offset);
        size_t recv_size = std::min(chunk, n - recv_offset);

        {
            std::lock_guard<std::mutex> lock(reduce_mutex_);
            if (reduce_buffer_.size() < n)
                reduce_buffer_.resize(n);
            std::memcpy(&reduce_buffer_[recv_offset],
                        &active[send_offset], send_size * sizeof(float));
        }
        this->barrier();
        {
            std::lock_guard<std::mutex> lock(reduce_mutex_);
            std::memcpy(recv_buf.data(), &reduce_buffer_[recv_offset],
                        recv_size * sizeof(float));
        }
        this->barrier();

        if (buf_idx == 0)
            reduce_impl(&buf_a[recv_offset], recv_buf.data(), recv_size, op);
        else
            reduce_impl(&buf_b[recv_offset], recv_buf.data(), recv_size, op);
    }

    this->barrier();

    for (int step = 0; step < world_size_ - 1; step++) {
        int buf_idx = step % 2;
        float* active = bufs[buf_idx];
        int send_rank = (world_rank_ - step + world_size_) % world_size_;
        int recv_rank = (world_rank_ - step - 1 + world_size_) % world_size_;
        size_t send_offset = (size_t)send_rank * chunk;
        size_t recv_offset = (size_t)recv_rank * chunk;
        size_t send_size = std::min(chunk, n - send_offset);
        size_t recv_size = std::min(chunk, n - recv_offset);

        {
            std::lock_guard<std::mutex> lock(reduce_mutex_);
            if (reduce_buffer_.size() < n)
                reduce_buffer_.resize(n);
            std::memcpy(&reduce_buffer_[recv_offset],
                        &active[send_offset], send_size * sizeof(float));
        }
        this->barrier();
        {
            std::lock_guard<std::mutex> lock(reduce_mutex_);
            if (buf_idx == 0)
                std::memcpy(&buf_a[recv_offset], &reduce_buffer_[recv_offset],
                            recv_size * sizeof(float));
            else
                std::memcpy(&buf_b[recv_offset], &reduce_buffer_[recv_offset],
                            recv_size * sizeof(float));
        }
        this->barrier();
    }

    if (op == ReduceOp::AVG) {
        for (size_t i = 0; i < n; i++)
            buf_a[i] /= (float)world_size_;
    }

    if (on_done) on_done();
}

void RingAllReduce::reduce_scatter(const float* data, float* recv_buf,
                                    size_t chunk_size, ReduceOp op) {
    if (world_size_ <= 1) {
        std::memcpy(recv_buf, data, chunk_size * sizeof(float));
        return;
    }

    size_t n = chunk_size * (size_t)world_size_;
    std::vector<float> work(n);

    // Copy data, simulating all ranks' data
    std::memcpy(work.data(), data, n * sizeof(float));
    allreduce(work.data(), n, op);

    // Each rank takes its chunk
    size_t offset = (size_t)world_rank_ * chunk_size;
    std::memcpy(recv_buf, &work[offset], chunk_size * sizeof(float));
}

void RingAllReduce::allgather(const float* local, float* global, size_t n) {
    if (world_size_ <= 1) {
        std::memcpy(global, local, n * sizeof(float));
        return;
    }

    size_t total = n * (size_t)world_size_;
    std::vector<float> buf(total);

    {
        std::lock_guard<std::mutex> lock(reduce_mutex_);
        if (reduce_buffer_.size() < total)
            reduce_buffer_.resize(total);
        std::memcpy(&reduce_buffer_[(size_t)world_rank_ * n],
                    local, n * sizeof(float));
    }
    this->barrier();
    for (int r = 0; r < world_size_; r++) {
        {
            std::lock_guard<std::mutex> lock(reduce_mutex_);
            std::memcpy(&buf[(size_t)r * n],
                        &reduce_buffer_[(size_t)r * n], n * sizeof(float));
        }
    }
    this->barrier();

    std::memcpy(global, buf.data(), total * sizeof(float));
}

void RingAllReduce::scatter(const float* src, float* dst, size_t n, int root) {
    if (world_size_ <= 1) {
        if (world_rank_ == root)
            std::memcpy(dst, src, n * sizeof(float));
        return;
    }

    {
        std::lock_guard<std::mutex> lock(reduce_mutex_);
        if (reduce_buffer_.size() < n * (size_t)world_size_)
            reduce_buffer_.resize(n * (size_t)world_size_);
        if (world_rank_ == root)
            std::memcpy(reduce_buffer_.data(), src,
                        n * (size_t)world_size_ * sizeof(float));
    }
    this->barrier();
    {
        std::lock_guard<std::mutex> lock(reduce_mutex_);
        std::memcpy(dst, &reduce_buffer_[(size_t)world_rank_ * n],
                    n * sizeof(float));
    }
    this->barrier();
}

void RingAllReduce::gather(const float* local, float* global, size_t n, int root) {
    if (world_size_ <= 1) {
        if (world_rank_ == root)
            std::memcpy(global, local, n * sizeof(float));
        return;
    }

    {
        std::lock_guard<std::mutex> lock(reduce_mutex_);
        if (reduce_buffer_.size() < n * (size_t)world_size_)
            reduce_buffer_.resize(n * (size_t)world_size_);
        std::memcpy(&reduce_buffer_[(size_t)world_rank_ * n],
                    local, n * sizeof(float));
    }
    this->barrier();
    if (world_rank_ == root) {
        std::lock_guard<std::mutex> lock(reduce_mutex_);
        std::memcpy(global, reduce_buffer_.data(),
                    n * (size_t)world_size_ * sizeof(float));
    }
    this->barrier();
}

void RingAllReduce::hierarchical_allreduce(float* data, size_t n, ReduceOp op,
                                            int nodes, int local_rank,
                                            int global_rank,
                                            InterNodeReduce inter_node_fn) {
    // NOTE (bug census): global_rank intentionally unused — node membership
    // derives from local_rank (rank 0 of each node drives inter-node); the
    // global id is kept for signature symmetry with the NCCL-style API.
    (void)global_rank; // documented-unused: leadership is local_rank==0
    if (nodes <= 1 || !inter_node_fn) {
        allreduce(data, n, op);
        return;
    }

    // Intra-node reduce first (shared memory)
    int local_world = world_size_ / nodes;
    if (local_world <= 1) local_world = world_size_;

    // Intra-node allreduce
    size_t intra_chunk = (n + (size_t)local_world - 1) / (size_t)local_world;
    size_t my_offset = (size_t)local_rank * intra_chunk;
    size_t my_size = std::min(intra_chunk, n - my_offset);

    if (my_size > 0 && my_offset < n) {
        allreduce(&data[my_offset], my_size, op);
    }
    this->barrier();

    // Inter-node reduce via callback (first rank of each node participates)
    if (local_rank == 0) {
        inter_node_fn(data, n, op);
    }
    this->barrier();

    // Broadcast within node
    if (local_rank == 0) {
        {
            std::lock_guard<std::mutex> lock(reduce_mutex_);
            if (reduce_buffer_.size() < n)
                reduce_buffer_.resize(n);
            std::memcpy(reduce_buffer_.data(), data, n * sizeof(float));
        }
    }
    this->barrier();
    {
        std::lock_guard<std::mutex> lock(reduce_mutex_);
        if (local_rank != 0)
            std::memcpy(data, reduce_buffer_.data(), n * sizeof(float));
    }
    this->barrier();
}

// ===========================================================================
// ParameterServer — centralized push/pull with stale gradient handling
// ===========================================================================

ParameterServer::ParameterServer(int world_size, int world_rank,
                                  int num_ps_nodes, int ps_node_id)
    : world_size_(world_size), world_rank_(world_rank),
      num_ps_nodes_(num_ps_nodes), ps_node_id_(ps_node_id),
      ctx_(world_size, world_rank, DistributedContext::Mode::DDP) {
    if (is_ps_node()) {
        async_running_ = true;
        async_worker_ = std::thread(&ParameterServer::async_worker_loop, this);
    }
}

ParameterServer::~ParameterServer() {
    shutdown();
}

int ParameterServer::shard_for(const std::string& name, int num_ps_nodes) {
    if (num_ps_nodes <= 1) return 0;
    std::hash<std::string> hasher;
    return (int)(hasher(name) % (size_t)num_ps_nodes);
}

bool ParameterServer::owns_param(const std::string& name) const {
    return shard_for(name, num_ps_nodes_) == ps_node_id_;
}

void ParameterServer::push(const std::string& name, const float* data, size_t n) {
    if (!is_ps_node()) return;
    if (!owns_param(name)) return;
    process_push(name, data, n);
}

void ParameterServer::push_async(const std::string& name, const float* data, size_t n) {
    if (!is_ps_node()) return;
    if (!owns_param(name)) return;
    AsyncOp op;
    op.name = name;
    op.data.assign(data, data + n);
    op.is_push = true;
    {
        std::lock_guard<std::mutex> lock(async_mtx_);
        async_queue_.push(std::move(op));
    }
    async_cv_.notify_one();
}

void ParameterServer::pull(const std::string& name, float* data, size_t n) {
    if (!is_ps_node()) return;
    if (!owns_param(name)) return;
    process_pull(name, data, n);
}

void ParameterServer::pull_async(const std::string& name, float* data, size_t n) {
    if (!is_ps_node()) return;
    if (!owns_param(name)) return;
    AsyncOp op;
    op.name = name;
    op.data.resize(n);
    op.is_push = false;
    {
        std::lock_guard<std::mutex> lock(async_mtx_);
        async_queue_.push(std::move(op));
    }
    async_cv_.notify_one();
    // For pull_async we also need to get the result back
    // In shared memory, we can directly read
    process_pull(name, data, n);
}

void ParameterServer::update_gradient(const std::string& name, const float* grad,
                                       size_t n, float lr) {
    if (!is_ps_node()) return;
    if (!owns_param(name)) return;
    process_gradient(name, grad, n, lr);
}

void ParameterServer::update_gradient_async(const std::string& name,
                                             const float* grad,
                                             size_t n, float lr) {
    if (!is_ps_node()) return;
    if (!owns_param(name)) return;
    AsyncOp op;
    op.name = name;
    op.data.assign(grad, grad + n);
    op.is_grad = true;
    op.lr = lr;
    {
        std::lock_guard<std::mutex> lock(async_mtx_);
        async_queue_.push(std::move(op));
    }
    async_cv_.notify_one();
}

void ParameterServer::barrier() {
    ctx_.barrier();
}

void ParameterServer::shutdown() {
    if (async_running_.exchange(false)) {
        async_cv_.notify_all();
        if (async_worker_.joinable())
            async_worker_.join();
    }
}

void ParameterServer::process_push(const std::string& name, const float* data, size_t n) {
    std::lock_guard<std::mutex> lock(global_mtx_);
    auto& entry = params_[name];
    if (entry.data.size() != n) {
        entry.data.resize(n);
        entry.gradient_accum.resize(n, 0.0f);
    }
    std::memcpy(entry.data.data(), data, n * sizeof(float));
    entry.version++;
}

void ParameterServer::process_pull(const std::string& name, float* data, size_t n) {
    std::lock_guard<std::mutex> lock(global_mtx_);
    auto it = params_.find(name);
    if (it == params_.end()) {
        std::memset(data, 0, n * sizeof(float));
        return;
    }
    std::memcpy(data, it->second.data.data(),
                std::min(n, it->second.data.size()) * sizeof(float));
}

void ParameterServer::process_gradient(const std::string& name, const float* grad,
                                        size_t n, float lr) {
    std::lock_guard<std::mutex> lock(global_mtx_);
    auto it = params_.find(name);
    if (it == params_.end()) return;

    auto& entry = it->second;
    size_t copy_n = std::min(n, entry.data.size());

    // Check for stale gradients
    if (entry.version > 0) {
        StaleGrad sg;
        sg.name = name;
        sg.grad.assign(grad, grad + copy_n);
        sg.version = entry.version;
        {
            std::lock_guard<std::mutex> slock(stale_mtx_);
            if (stale_queue_.size() < (size_t)max_stale_grads_)
                stale_queue_.push_back(std::move(sg));
        }
    }

    // Apply gradient update: param -= lr * grad
    for (size_t i = 0; i < copy_n; i++)
        entry.data[i] -= lr * grad[i];

    entry.version++;
}

void ParameterServer::apply_stale_gradients() {
    // BUGFIX (bug census): lock-order inversion — this took stale_mtx_ THEN
    // global_mtx_, while process_gradient takes global_mtx_ THEN stale_mtx_
    // (:730-736) = classic ABBA deadlock. Global order is now: global_mtx_
    // before stale_mtx_, everywhere.
    std::lock_guard<std::mutex> glock(global_mtx_);
    std::lock_guard<std::mutex> slock(stale_mtx_);
    for (auto& sg : stale_queue_) {
        auto it = params_.find(sg.name);
        if (it == params_.end()) continue;
        auto& entry = it->second;
        if (sg.version < entry.version - 5) continue;
        size_t copy_n = std::min(sg.grad.size(), entry.data.size());
        for (size_t i = 0; i < copy_n; i++)
            entry.data[i] -= 0.5f * sg.grad[i];
    }
    stale_queue_.clear();
}

bool ParameterServer::get_param(const std::string& name, float* data, size_t n) {
    std::lock_guard<std::mutex> lock(global_mtx_);
    auto it = params_.find(name);
    if (it == params_.end()) return false;
    size_t copy_n = std::min(n, it->second.data.size());
    std::memcpy(data, it->second.data.data(), copy_n * sizeof(float));
    if (copy_n < n) std::memset(&data[copy_n], 0, (n - copy_n) * sizeof(float));
    return true;
}

std::vector<std::string> ParameterServer::list_params() const {
    std::lock_guard<std::mutex> lock(global_mtx_);
    std::vector<std::string> names;
    names.reserve(params_.size());
    for (const auto& kv : params_)
        names.push_back(kv.first);
    return names;
}

void ParameterServer::remove_param(const std::string& name) {
    std::lock_guard<std::mutex> lock(global_mtx_);
    params_.erase(name);
}

void ParameterServer::clear_params() {
    std::lock_guard<std::mutex> lock(global_mtx_);
    params_.clear();
}

void ParameterServer::average_parameters(float weight) {
    std::lock_guard<std::mutex> lock(global_mtx_);
    for (auto& kv : params_) {
        for (auto& v : kv.second.data)
            v *= weight;
    }
}

void ParameterServer::apply_weight_decay(float decay_rate) {
    std::lock_guard<std::mutex> lock(global_mtx_);
    for (auto& kv : params_) {
        for (auto& v : kv.second.data)
            v *= (1.0f - decay_rate);
    }
}

void ParameterServer::apply_momentum(float momentum, float lr) {
    std::lock_guard<std::mutex> lock(global_mtx_);
    for (auto& kv : params_) {
        auto& entry = kv.second;
        if (entry.momentum_buf.size() != entry.data.size()) {
            entry.momentum_buf.resize(entry.data.size(), 0.0f);
        }
        for (size_t i = 0; i < entry.data.size(); i++) {
            entry.momentum_buf[i] = momentum * entry.momentum_buf[i] +
                                    (1.0f - momentum) * entry.gradient_accum[i];
            entry.data[i] -= lr * entry.momentum_buf[i];
        }
    }
}

int64_t ParameterServer::get_version(const std::string& name) const {
    std::lock_guard<std::mutex> lock(global_mtx_);
    auto it = params_.find(name);
    if (it == params_.end()) return -1;
    return it->second.version;
}

int ParameterServer::stale_gradient_count() const {
    std::lock_guard<std::mutex> slock(stale_mtx_);
    return (int)stale_queue_.size();
}

void ParameterServer::flush_async() {
    // BUGFIX (bug census): held global_mtx_ + async_mtx_ then called
    // process_gradient (takes global_mtx_) = SELF-DEADLOCK. Drain the queue
    // under async_mtx_ only, release, then process each op (which takes
    // global_mtx_ itself). Same global lock order as everywhere else.
    std::queue<AsyncOp> pending;
    {
        std::lock_guard<std::mutex> alock(async_mtx_);
        pending.swap(async_queue_);
    }
    while (!pending.empty()) {
        auto& op = pending.front();
        if (op.is_grad)
            process_gradient(op.name, op.data.data(), op.data.size(), op.lr);
        else if (op.is_push)
            process_push(op.name, op.data.data(), op.data.size());
        pending.pop();
    }
    apply_stale_gradients();
}

void ParameterServer::async_worker_loop() {
    while (async_running_.load()) {
        AsyncOp op;
        {
            std::unique_lock<std::mutex> lock(async_mtx_);
            async_cv_.wait(lock, [this] {
                return !async_queue_.empty() || !async_running_.load();
            });
            if (!async_running_.load()) return;
            op = std::move(async_queue_.front());
            async_queue_.pop();
        }

        if (op.is_grad) {
            process_gradient(op.name, op.data.data(), op.data.size(), op.lr);
        } else if (op.is_push) {
            process_push(op.name, op.data.data(), op.data.size());
        }

        apply_stale_gradients();
    }
}

} // namespace quant
