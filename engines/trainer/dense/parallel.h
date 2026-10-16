#pragma once
#include "quant/distributed.h"
#include <cstdint>
#include <string>
#include <vector>

namespace quant {
namespace dense_engine {

// Distributed training math on host: shard maps, ring allreduce,
// grad averaging, TP split/gather. Real tensor math, single-host
// simulation of multi-rank semantics (ranks run as slices).
struct RankSlice {
    int rank = 0;
    int world = 1;
    int64_t start = 0;
    int64_t count = 0;
};
RankSlice slice_for(int64_t n, int rank, int world);

// Ring allreduce SUM over per-rank float buffers (real scatter-reduce).
void ring_allreduce_sum(std::vector<std::vector<float>>& bufs);
// Broadcast rank-src buffer to all (real copy).
void broadcast_bufs(std::vector<std::vector<float>>& bufs, int src = 0);
// Average grads across rank slices (real divide).
void average_grad_slices(std::vector<Tensor>& grads, int world);

// Tensor-parallel column split + row gather (real slicing).
struct TPSplit {
    std::vector<Tensor> shards;
    int64_t dim = 0;
};
TPSplit tp_split_cols(const Tensor& w, int world);
Tensor tp_gather_cols(const std::vector<Tensor>& shards);
TPSplit tp_split_rows(const Tensor& w, int world);
Tensor tp_gather_rows(const std::vector<Tensor>& shards);

// FSDP-style shard map: param -> rank with real byte accounting.
struct ShardPlan {
    struct Entry { std::string name; int rank = 0; int64_t bytes = 0; };
    std::vector<Entry> entries;
    std::vector<int64_t> rank_bytes;
    int64_t total_bytes = 0;
    double imbalance() const;
};
ShardPlan plan_shards(const std::vector<std::string>& names,
                      const std::vector<int64_t>& bytes, int world);

// Pipeline stage map: layer -> stage with real load balance readout.
struct PipePlan {
    std::vector<int> layer_stage;
    std::vector<int> stage_layers;
    int stages = 1;
};
PipePlan plan_pipeline(int64_t layers, int stages);

} // namespace dense_engine
} // namespace quant
