#include "quant/tensor_parallelism.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace quant {

// ===========================================================================
// TensorParallelManager
// ===========================================================================

TensorParallelManager::TensorParallelManager(Model* model, int world_size,
                                               int world_rank)
    : model_(model), world_size_(world_size), world_rank_(world_rank),
      ctx_(world_size, world_rank, DistributedContext::Mode::TENSOR_PARALLEL),
      strategy_(PartitionStrategy::AUTO) {}

TensorParallelManager::~TensorParallelManager() {}

void TensorParallelManager::column_parallel_split(const Tensor& weight,
                                                    Tensor& local_weight) {
    column_parallel_split(weight, local_weight, world_rank_, world_size_);
}

void TensorParallelManager::column_parallel_split(const Tensor& weight,
                                                    Tensor& local_weight,
                                                    int64_t rank, int64_t size) {
    int64_t out_dim = weight.dim(0);
    int64_t in_dim = weight.dim(1);
    int64_t chunk = (out_dim + size - 1) / size;
    int64_t start = rank * chunk;
    int64_t end = std::min(start + chunk, out_dim);
    local_weight = weight.slice(0, start, end);
}

void TensorParallelManager::row_parallel_split(const Tensor& weight,
                                                 Tensor& local_weight) {
    row_parallel_split(weight, local_weight, world_rank_, world_size_);
}

void TensorParallelManager::row_parallel_split(const Tensor& weight,
                                                 Tensor& local_weight,
                                                 int64_t rank, int64_t size) {
    int64_t out_dim = weight.dim(0);
    int64_t in_dim = weight.dim(1);
    int64_t chunk = (in_dim + size - 1) / size;
    int64_t start = rank * chunk;
    int64_t end = std::min(start + chunk, in_dim);
    local_weight = weight.slice(1, start, end);
}

Tensor TensorParallelManager::column_parallel_linear(const Tensor& input,
                                                       const Tensor& local_weight) {
    // Y_local = X @ W_local (each rank has a column shard)
    // local_weight shape: [out_shard, in_features]
    // input shape: [batch, seq, in_features]
    // output shape: [batch, seq, out_shard]
    int64_t batch = 1;
    int64_t seq = 1;
    if (input.rank() >= 2) {
        batch = input.dim(0);
        seq = input.dim(1);
    } else if (input.rank() == 1) {
        seq = input.dim(0);
    }
    int64_t in_features = local_weight.dim(1);
    int64_t out_shard = local_weight.dim(0);

    // Reshape input to 2D: [batch*seq, in_features]
    Tensor input_2d = input.reshape({batch * seq, in_features});

    // Matmul: [batch*seq, in_features] @ [out_shard, in_features]^T
    // = [batch*seq, out_shard]
    Tensor output({batch * seq, out_shard});
    output.zero_();

    const float* in_ptr = input_2d.data<float>();
    const float* w_ptr = local_weight.data<float>();
    float* out_ptr = output.data<float>();

    for (int64_t b = 0; b < batch * seq; b++) {
        for (int64_t o = 0; o < out_shard; o++) {
            float sum = 0.0f;
            for (int64_t i = 0; i < in_features; i++) {
                sum += in_ptr[b * in_features + i] * w_ptr[o * in_features + i];
            }
            out_ptr[b * out_shard + o] = sum;
        }
    }

    // AllReduce to combine partial outputs from all ranks
    if (world_size_ > 1) {
        ctx_.all_reduce(output);
    }

    return output.reshape({batch, seq, out_shard});
}

Tensor TensorParallelManager::row_parallel_linear(const Tensor& input,
                                                    const Tensor& local_weight) {
    // Y = AllReduce(X_local @ W_local)
    // local_weight shape: [out_features, in_shard]
    // input shape: [batch, seq, in_shard] (already scattered)
    // output: [batch, seq, out_features] after AllReduce
    int64_t batch = 1;
    int64_t seq = 1;
    if (input.rank() >= 2) {
        batch = input.dim(0);
        seq = input.dim(1);
    } else if (input.rank() == 1) {
        seq = input.dim(0);
    }
    int64_t in_shard = local_weight.dim(1);
    int64_t out_features = local_weight.dim(0);

    Tensor input_2d = input.reshape({batch * seq, in_shard});

    Tensor output({batch * seq, out_features});
    output.zero_();

    const float* in_ptr = input_2d.data<float>();
    const float* w_ptr = local_weight.data<float>();
    float* out_ptr = output.data<float>();

    for (int64_t b = 0; b < batch * seq; b++) {
        for (int64_t o = 0; o < out_features; o++) {
            float sum = 0.0f;
            for (int64_t i = 0; i < in_shard; i++) {
                sum += in_ptr[b * in_shard + i] * w_ptr[o * in_shard + i];
            }
            out_ptr[b * out_features + o] = sum;
        }
    }

    // AllReduce to combine partial results
    if (world_size_ > 1) {
        ctx_.all_reduce(output);
    }

    return output.reshape({batch, seq, out_features});
}

void TensorParallelManager::partition_heads(int64_t total_heads,
                                              int64_t& heads_per_rank,
                                              int64_t& head_start) {
    heads_per_rank = (total_heads + world_size_ - 1) / world_size_;
    head_start = world_rank_ * heads_per_rank;
    if (head_start >= total_heads) {
        head_start = 0;
        heads_per_rank = 0;
    }
    if (head_start + heads_per_rank > total_heads) {
        heads_per_rank = total_heads - head_start;
    }
}

Tensor TensorParallelManager::sequence_parallel_scatter(const Tensor& x,
                                                           int64_t seq_dim) {
    if (world_size_ <= 1) return x;

    int64_t total_seq = x.dim(static_cast<int>(seq_dim));
    int64_t chunk = (total_seq + world_size_ - 1) / world_size_;
    int64_t start = world_rank_ * chunk;
    int64_t end = std::min(start + chunk, total_seq);

    // Build slice indices
    std::vector<int64_t> starts(x.rank(), 0);
    std::vector<int64_t> ends;
    for (int64_t d = 0; d < x.rank(); d++) {
        ends.push_back(x.dim(static_cast<int>(d)));
    }
    starts[seq_dim] = start;
    ends[seq_dim] = end;

    return x.slice(static_cast<int>(seq_dim), start, end);
}

Tensor TensorParallelManager::sequence_parallel_gather(const Tensor& x,
                                                         int64_t seq_dim) {
    if (world_size_ <= 1) return x;

    Tensor gathered;
    ctx_.all_gather(x, gathered);
    return gathered;
}

void TensorParallelManager::allreduce_tensor(Tensor& tensor) {
    ctx_.all_reduce(tensor);
}

void TensorParallelManager::allgather_tensor(const Tensor& local,
                                               Tensor& global) {
    ctx_.all_gather(local, global);
}

void TensorParallelManager::reduce_scatter_tensor(const Tensor& input,
                                                    Tensor& output) {
    int64_t total = input.numel();
    int64_t per_rank = (total + world_size_ - 1) / world_size_;
    output = Tensor({per_rank});
    // Use allreduce then take local shard as approximation
    Tensor work = input;
    ctx_.all_reduce(work);
    int64_t start = world_rank_ * per_rank;
    int64_t end = std::min(start + per_rank, total);
    int64_t local_n = end - start;
    output = Tensor({local_n});
    std::memcpy(output.data<float>(), work.data<float>() + start,
                local_n * sizeof(float));
}

TensorParallelManager::PartitionStrategy
TensorParallelManager::select_strategy(const TransformerConfig& cfg) const {
    // AUTO strategy: column-parallel for most layers
    // For models with large hidden sizes, use column-then-row
    int64_t hidden = cfg.hidden_size;
    int64_t ffn = cfg.ffn_hidden_size;

    if (world_size_ <= 1) return PartitionStrategy::COLUMN_ONLY;

    if (hidden * ffn > 1024 * 4096) {
        return PartitionStrategy::COLUMN_THEN_ROW;
    }
    return PartitionStrategy::COLUMN_ONLY;
}

void TensorParallelManager::parallelize_block(TransformerBlock& block) {
    if (world_size_ <= 1) return;

    TransformerConfig tmp;
    tmp.hidden_size = block.attention_norm.weight.numel();
    tmp.ffn_hidden_size = block.ffn.gate_proj.weight.numel() / tmp.hidden_size;
    strategy_ = select_strategy(tmp);

    if (strategy_ == PartitionStrategy::COLUMN_THEN_ROW ||
        strategy_ == PartitionStrategy::COLUMN_ONLY) {
        apply_column_parallel_attention(block.attention);
        apply_column_parallel_ffn(block.ffn);
    }
    if (strategy_ == PartitionStrategy::COLUMN_THEN_ROW) {
        apply_row_parallel_attention(block.attention);
        apply_row_parallel_ffn(block.ffn);
    }
}

void TensorParallelManager::parallelize_model(DenseModel& model) {
    if (world_size_ <= 1) return;

    for (auto& layer : model.layers) {
        parallelize_block(*layer);
    }

    // Split embedding table across ranks (column-parallel)
    if (model.tok_embeddings) {
        Tensor local_emb;
        column_parallel_split(model.tok_embeddings->weight, local_emb);
        model.tok_embeddings->weight = local_emb;
    }

    // Split lm_head (column-parallel)
    if (model.lm_head) {
        Tensor local_head;
        column_parallel_split(model.lm_head->weight, local_head);
        model.lm_head->weight = local_head;
    }
}

Tensor TensorParallelManager::reconstruct_weight(const Tensor& local_weight,
                                                   int64_t full_dim,
                                                   bool is_column) {
    Tensor global;
    allgather_tensor(local_weight, global);
    return global;
}

int64_t TensorParallelManager::communication_volume(const TransformerConfig& cfg) const {
    // Communication volume per forward pass:
    // Column-parallel: AllReduce of [batch, seq, hidden/TP] per layer = O(batch*seq*hidden)
    // Row-parallel: AllReduce of [batch, seq, hidden] per layer
    // Total per layer: ~2 * batch * seq * hidden_size (forward + backward)
    int64_t per_layer = 2 * cfg.hidden_size;
    int64_t total = per_layer * cfg.num_layers;
    return total;
}

void TensorParallelManager::apply_column_parallel_attention(Attention& attn) {
    // Q, K, V projections are column-parallel (split output dim)
    Tensor q_local, k_local, v_local;
    column_parallel_split(attn.q_proj.weight, q_local);
    column_parallel_split(attn.k_proj.weight, k_local);
    column_parallel_split(attn.v_proj.weight, v_local);
    attn.q_proj.weight = q_local;
    attn.k_proj.weight = k_local;
    attn.v_proj.weight = v_local;
    attn.q_proj.bias = Tensor({q_local.dim(0)});
    attn.k_proj.bias = Tensor({k_local.dim(0)});
    attn.v_proj.bias = Tensor({v_local.dim(0)});
    // Adjust num_heads for this rank
    attn.num_heads = (attn.num_heads + world_size_ - 1) / world_size_;
}

void TensorParallelManager::apply_row_parallel_attention(Attention& attn) {
    // O projection is row-parallel (split input dim)
    Tensor o_local;
    row_parallel_split(attn.o_proj.weight, o_local);
    attn.o_proj.weight = o_local;
    attn.o_proj.bias = Tensor({attn.o_proj.weight.dim(0)});
}

void TensorParallelManager::apply_column_parallel_ffn(FFN& ffn) {
    // gate_proj and up_proj are column-parallel
    Tensor gate_local, up_local;
    column_parallel_split(ffn.gate_proj.weight, gate_local);
    column_parallel_split(ffn.up_proj.weight, up_local);
    ffn.gate_proj.weight = gate_local;
    ffn.up_proj.weight = up_local;
    ffn.gate_proj.bias = Tensor({gate_local.dim(0)});
    ffn.up_proj.bias = Tensor({up_local.dim(0)});
}

void TensorParallelManager::apply_row_parallel_ffn(FFN& ffn) {
    // down_proj is row-parallel
    Tensor down_local;
    row_parallel_split(ffn.down_proj.weight, down_local);
    ffn.down_proj.weight = down_local;
    ffn.down_proj.bias = Tensor({ffn.down_proj.weight.dim(0)});
}

// ===========================================================================
// ColumnParallelLinear
// ===========================================================================

ColumnParallelLinear::ColumnParallelLinear(int64_t in_features,
                                             int64_t out_features,
                                             int world_size, int world_rank)
    : in_features_(in_features), out_features_(out_features),
      world_size_(world_size), world_rank_(world_rank), has_bias_(true) {
    int64_t local_out = (out_features + world_size - 1) / world_size;
    int64_t start = world_rank * local_out;
    int64_t actual_out = std::min(local_out, out_features - start);

    local_weight_ = Tensor({actual_out, in_features});
    local_bias_ = Tensor({actual_out});
    local_weight_.zero_();
    local_bias_.zero_();
}

Tensor ColumnParallelLinear::forward(const Tensor& input) {
    // input: [batch*seq, in_features]
    // local_weight_: [local_out, in_features]
    // output: [batch*seq, local_out] then AllReduce
    int64_t batch_seq = input.dim(0);
    int64_t local_out = local_weight_.dim(0);

    Tensor output({batch_seq, local_out});
    output.zero_();

    const float* in_ptr = input.data<float>();
    const float* w_ptr = local_weight_.data<float>();
    float* out_ptr = output.data<float>();

    for (int64_t b = 0; b < batch_seq; b++) {
        for (int64_t o = 0; o < local_out; o++) {
            float sum = 0.0f;
            for (int64_t i = 0; i < in_features_; i++) {
                sum += in_ptr[b * in_features_ + i] * w_ptr[o * in_features_ + i];
            }
            out_ptr[b * local_out + o] = sum + local_bias_.data<float>()[o];
        }
    }

    if (world_size_ > 1) {
        DistributedContext ctx(world_size_, world_rank_,
                               DistributedContext::Mode::TENSOR_PARALLEL);
        ctx.all_reduce(output);
    }

    return output;
}

int64_t ColumnParallelLinear::local_out_features() const {
    return local_weight_.dim(0);
}

// ===========================================================================
// RowParallelLinear
// ===========================================================================

RowParallelLinear::RowParallelLinear(int64_t in_features, int64_t out_features,
                                       int world_size, int world_rank)
    : in_features_(in_features), out_features_(out_features),
      world_size_(world_size), world_rank_(world_rank), has_bias_(true) {
    int64_t local_in = (in_features + world_size - 1) / world_size;
    int64_t start = world_rank * local_in;
    int64_t actual_in = std::min(local_in, in_features - start);

    local_weight_ = Tensor({out_features, actual_in});
    local_bias_ = Tensor({out_features});
    local_weight_.zero_();
    local_bias_.zero_();
}

Tensor RowParallelLinear::forward(const Tensor& input) {
    // input: [batch*seq, local_in]
    // local_weight_: [out_features, local_in]
    // output: [batch*seq, out_features] after AllReduce
    int64_t batch_seq = input.dim(0);
    int64_t local_in = local_weight_.dim(1);
    int64_t out_feat = local_weight_.dim(0);

    Tensor output({batch_seq, out_feat});
    output.zero_();

    const float* in_ptr = input.data<float>();
    const float* w_ptr = local_weight_.data<float>();
    float* out_ptr = output.data<float>();

    for (int64_t b = 0; b < batch_seq; b++) {
        for (int64_t o = 0; o < out_feat; o++) {
            float sum = 0.0f;
            for (int64_t i = 0; i < local_in; i++) {
                sum += in_ptr[b * local_in + i] * w_ptr[o * local_in + i];
            }
            out_ptr[b * out_feat + o] = sum;
        }
    }

    if (world_size_ > 1) {
        DistributedContext ctx(world_size_, world_rank_,
                               DistributedContext::Mode::TENSOR_PARALLEL);
        ctx.all_reduce(output);
    }

    // Add bias after AllReduce
    if (has_bias_) {
        float* out_ptr2 = output.data<float>();
        const float* bias = local_bias_.data<float>();
        for (int64_t b = 0; b < batch_seq; b++) {
            for (int64_t o = 0; o < out_feat; o++) {
                out_ptr2[b * out_feat + o] += bias[o];
            }
        }
    }

    return output;
}

int64_t RowParallelLinear::local_in_features() const {
    return local_weight_.dim(1);
}

// ===========================================================================
// HeadPartitioner
// ===========================================================================

HeadPartitioner::HeadPartitioner(int64_t num_heads, int64_t head_dim,
                                   int world_size, int world_rank)
    : num_heads_(num_heads), head_dim_(head_dim),
      world_size_(world_size), world_rank_(world_rank) {
    heads_per_rank_ = (num_heads + world_size - 1) / world_size;
    head_start_ = world_rank * heads_per_rank_;
    if (head_start_ >= num_heads) {
        head_start_ = 0;
        heads_per_rank_ = 0;
    }
    if (head_start_ + heads_per_rank_ > num_heads) {
        heads_per_rank_ = num_heads - head_start_;
    }
}

Tensor HeadPartitioner::partition_heads(const Tensor& qkv,
                                          int64_t seq_len) const {
    if (heads_per_rank_ == 0) {
        return Tensor({0});
    }
    // qkv shape: [batch*seq, num_heads * head_dim]
    // Extract heads [head_start, head_start + heads_per_rank)
    int64_t batch_seq = qkv.dim(0);
    int64_t offset = head_start_ * head_dim_;
    int64_t length = heads_per_rank_ * head_dim_;

    Tensor result({batch_seq, length});
    const float* src = qkv.data<float>();
    float* dst = result.data<float>();
    for (int64_t b = 0; b < batch_seq; b++) {
        std::memcpy(&dst[b * length], &src[b * num_heads_ * head_dim_ + offset],
                    length * sizeof(float));
    }
    return result;
}

Tensor HeadPartitioner::merge_heads(const Tensor& attn_out,
                                      int64_t seq_len) const {
    if (heads_per_rank_ == 0) {
        return Tensor({seq_len, 0});
    }
    // attn_out: [batch*seq, heads_per_rank * head_dim]
    // This is local output; after AllReduce we get full output
    return attn_out;
}

void HeadPartitioner::sync_output(Tensor& output) const {
    if (world_size_ <= 1) return;
    DistributedContext ctx(world_size_, world_rank_,
                           DistributedContext::Mode::TENSOR_PARALLEL);
    ctx.all_reduce(output);
}

// ===========================================================================
// SequenceParallel
// ===========================================================================

SequenceParallel::SequenceParallel(int world_size, int world_rank)
    : world_size_(world_size), world_rank_(world_rank) {}

Tensor SequenceParallel::scatter(const Tensor& x, int64_t seq_dim) const {
    if (world_size_ <= 1) return x;

    int64_t total = x.dim(static_cast<int>(seq_dim));
    int64_t chunk = (total + world_size_ - 1) / world_size_;
    int64_t start = world_rank_ * chunk;
    int64_t end = std::min(start + chunk, total);

    return x.slice(static_cast<int>(seq_dim), start, end);
}

Tensor SequenceParallel::gather(const Tensor& x, int64_t seq_dim) const {
    if (world_size_ <= 1) return x;

    // Gather all local chunks into full tensor
    int64_t local_size = x.dim(static_cast<int>(seq_dim));
    int64_t total_size = local_size * world_size_;
    Tensor result = x;

    DistributedContext ctx(world_size_, world_rank_,
                           DistributedContext::Mode::TENSOR_PARALLEL);
    ctx.all_gather(x, result);
    return result;
}

void SequenceParallel::allreduce(Tensor& x) const {
    if (world_size_ <= 1) return;
    DistributedContext ctx(world_size_, world_rank_,
                           DistributedContext::Mode::TENSOR_PARALLEL);
    ctx.all_reduce(x);
}

} // namespace quant
