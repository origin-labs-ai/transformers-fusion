#pragma once
#include "quant/tensor.h"
#include "quant/types.h"
#include "quant/model.h"
#include "quant/transformer.h"
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <functional>
#include <condition_variable>
#include <queue>
#include <atomic>

namespace quant {

// C23: Distributed Data Parallel (DDP) context
// Uses shared-memory thread-based parallelism (simulates multi-GPU)
class DistributedContext {
public:
    enum class Mode { DDP, TENSOR_PARALLEL, PIPELINE_PARALLEL };

    DistributedContext(int world_size, int world_rank, Mode mode = Mode::DDP);
    ~DistributedContext();

    int world_size() const { return world_size_; }
    int world_rank() const { return world_rank_; }
    Mode mode() const { return mode_; }
    bool is_main() const { return world_rank_ == 0; }

    // All-reduce across all ranks (sum)
    void all_reduce(Tensor& tensor);
    void all_reduce(float* data, int64_t n);
    void barrier();

    // Broadcast from main rank
    void broadcast(Tensor& tensor, int src_rank = 0);
    void broadcast(float* data, int64_t n, int src_rank = 0);

    // All-gather
    void all_gather(const Tensor& local, Tensor& global);

private:
    int world_size_;
    int world_rank_;
    Mode mode_;
    std::mutex barrier_mutex_;
    std::condition_variable barrier_cv_;
    int barrier_count_ = 0;
    int barrier_gen_ = 0; // PROD round-7: barrier generation (reset-counter deadlock fix)
    std::mutex reduce_mutex_;
    std::vector<float> reduce_buffer_;
    bool reduce_cleared_ = false; // PROD round-7b: one-clearer flag for all_reduce
};

// C23: DDP wrapper — replicates model across ranks, synchronizes gradients
class DDPWrapper {
public:
    DDPWrapper(Model* model, int world_size, int world_rank);
    void sync_gradients();
    DistributedContext& context() { return ctx_; }
    Model* model() { return local_model_; }
private:
    DistributedContext ctx_;
    Model* local_model_;
    std::vector<Tensor*> params_;
};

// C24: Tensor Parallel — splits linear layers across ranks
class TensorParallelWrapper {
public:
    TensorParallelWrapper(Model* model, int world_size, int world_rank);
    Tensor forward(const Tensor& input, const Tensor& positions);
    DistributedContext& context() { return ctx_; }
private:
    DistributedContext ctx_;
    Model* model_;
    void split_linear(const Tensor& weight, Tensor& local_weight,
                      int64_t dim, bool is_column);
};

// C25: Pipeline Parallel — splits layers across ranks
class PipelineParallelWrapper {
public:
    PipelineParallelWrapper(Model* model, int world_size, int world_rank);
    Tensor forward(const Tensor& input, const Tensor& positions,
                   const Tensor& mask, KVCache& cache);
    DistributedContext& context() { return ctx_; }
private:
    DistributedContext ctx_;
    Model* model_;
    int64_t micro_batches_ = 4;
    int start_layer_ = 0;
    int end_layer_ = 1;
};

// ===========================================================================
// RingAllReduce — true ring-based allreduce (scatter-reduce + allgather)
// ===========================================================================
enum class ReduceOp { SUM, AVG, MAX, MIN };

class RingAllReduce {
public:
    RingAllReduce(int world_size, int world_rank);

    void allreduce(float* data, size_t n, ReduceOp op = ReduceOp::SUM);

    static int world_size();
    static int rank();

    // Shared-memory barrier
    void barrier();

    // Chunked communication for large tensors (avoids large per-rank buffers)
    void allreduce_chunked(float* data, size_t n, size_t chunk_size,
                           ReduceOp op = ReduceOp::SUM);

    // Double-buffered allreduce for computation/communication overlap
    using DoubleBufferDone = std::function<void()>;
    void allreduce_double_buffer(float* buf_a, float* buf_b, size_t n,
                                 ReduceOp op, DoubleBufferDone on_done);

    // Reduce-scatter: each rank ends up with a chunk of the reduced result
    void reduce_scatter(const float* data, float* recv_buf, size_t chunk_size,
                        ReduceOp op = ReduceOp::SUM);

    // Allgather: gather all data from all ranks (independent of allreduce)
    void allgather(const float* local, float* global, size_t n);

    // Scatter one rank's data to all others
    void scatter(const float* src, float* dst, size_t n, int root = 0);

    // Gather data from all ranks to one
    void gather(const float* local, float* global, size_t n, int root = 0);

    // Hierarchical allreduce: intra-node reduce via shared memory,
    // inter-node reduce via specified callback (future network support)
    using InterNodeReduce = std::function<void(float*, size_t, ReduceOp)>;
    void hierarchical_allreduce(float* data, size_t n, ReduceOp op,
                                int nodes, int local_rank, int global_rank,
                                InterNodeReduce inter_node_fn = nullptr);

private:
    int world_size_;
    int world_rank_;
    static std::atomic<int> s_world_size_;
    static std::atomic<int> s_rank_;
    mutable std::mutex reduce_mutex_;
    std::vector<float> reduce_buffer_;
    std::mutex ring_barrier_mutex_;
    std::condition_variable ring_barrier_cv_;
    int ring_barrier_count_ = 0;
    int ring_barrier_gen_ = 0; // PROD round-7: same generation fix

    void scatter_reduce(float* data, size_t chunk, ReduceOp op);
    void all_gather(float* data, size_t chunk);
    void reduce_impl(float* a, const float* b, size_t n, ReduceOp op);
};

// ===========================================================================
// ParameterServer — centralized parameter push/pull with async stale handling
// ===========================================================================
class ParameterServer {
public:
    ParameterServer(int world_size, int world_rank,
                    int num_ps_nodes = 1, int ps_node_id = 0);
    ~ParameterServer();

    void push(const std::string& name, const float* data, size_t n);
    void push_async(const std::string& name, const float* data, size_t n);
    void pull(const std::string& name, float* data, size_t n);
    void pull_async(const std::string& name, float* data, size_t n);

    void update_gradient(const std::string& name, const float* grad, size_t n,
                         float lr);
    void update_gradient_async(const std::string& name, const float* grad,
                               size_t n, float lr);

    void barrier();
    void shutdown();

    int world_size() const { return world_size_; }
    int world_rank() const { return world_rank_; }
    int num_ps_nodes() const { return num_ps_nodes_; }
    int ps_node_id() const { return ps_node_id_; }

    // Sharding: returns which PS node owns a given parameter
    static int shard_for(const std::string& name, int num_ps_nodes);

    bool is_ps_node() const { return world_rank_ < num_ps_nodes_; }

private:
    struct ParamEntry {
        std::vector<float> data;
        std::vector<float> gradient_accum;
        std::vector<float> momentum_buf;
        int64_t version = 0;
        std::mutex mtx;
    };

    int world_size_;
    int world_rank_;
    int num_ps_nodes_;
    int ps_node_id_;
    DistributedContext ctx_;

    std::unordered_map<std::string, ParamEntry> params_;
    mutable std::mutex global_mtx_;

    // Async queue
    struct AsyncOp {
        std::string name;
        std::vector<float> data;
        bool is_push;
        float lr = 0.0f;
        bool is_grad = false;
    };
    std::queue<AsyncOp> async_queue_;
    std::thread async_worker_;
    std::mutex async_mtx_;
    std::condition_variable async_cv_;
    std::atomic<bool> async_running_{false};
    void async_worker_loop();

    // Stale gradient handling
    struct StaleGrad {
        std::string name;
        std::vector<float> grad;
        int64_t version;
    };
    std::vector<StaleGrad> stale_queue_;
    mutable std::mutex stale_mtx_;
    int max_stale_grads_ = 64;

    void apply_stale_gradients();
    void process_push(const std::string& name, const float* data, size_t n);
    void process_pull(const std::string& name, float* data, size_t n);
    void process_gradient(const std::string& name, const float* grad,
                          size_t n, float lr);
    bool owns_param(const std::string& name) const;

    // Get parameter data directly (copy from server)
    bool get_param(const std::string& name, float* data, size_t n);

    // List all parameter names on this PS node
    std::vector<std::string> list_params() const;

    // Remove a parameter from the server
    void remove_param(const std::string& name);

    // Clear all parameters
    void clear_params();

    // Average parameters across all PS nodes (model averaging)
    void average_parameters(float weight = 1.0f);

    // Apply weight decay to all parameters
    void apply_weight_decay(float decay_rate);

    // Apply momentum to gradient updates (SGD-style)
    void apply_momentum(float momentum, float lr);

    // Get parameter version (for staleness tracking)
    int64_t get_version(const std::string& name) const;

    // Set max stale gradients to keep
    void set_max_stale_gradients(int max_stale) { max_stale_grads_ = max_stale; }

    // Get current number of stale gradients in queue
    int stale_gradient_count() const;

    // Flush async queue (block until all pending ops are processed)
    void flush_async();

};

} // namespace quant
