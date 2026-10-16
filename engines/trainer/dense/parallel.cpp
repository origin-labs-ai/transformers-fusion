#include "parallel.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace quant {
namespace dense_engine {

RankSlice slice_for(int64_t n, int rank, int world) {
    if (world <= 0 || rank < 0 || rank >= world) throw std::runtime_error("[parallel] bad rank/world");
    RankSlice s;
    s.rank = rank;
    s.world = world;
    int64_t base = n / world, rem = n % world;
    s.start = rank * base + std::min<int64_t>(rank, rem);
    s.count = base + (rank < rem ? 1 : 0);
    return s;
}

void ring_allreduce_sum(std::vector<std::vector<float>>& bufs) {
    if (bufs.empty()) return;
    size_t n = bufs[0].size();
    for (auto& b : bufs)
        if (b.size() != n) throw std::runtime_error("[parallel] allreduce size mismatch");
    int world = (int)bufs.size();
    for (int step = 0; step < world - 1; step++) {
        std::vector<std::vector<float>> nxt = bufs;
        for (int r = 0; r < world; r++) {
            int src = (r - step - 1 + world * 2) % world;
            RankSlice sl = slice_for((int64_t)n, (r - step - 1 + world * 4) % world, world);
            (void)sl;
            int chunk = (r - step + world) % world;
            RankSlice c = slice_for((int64_t)n, chunk, world);
            for (int64_t i = c.start; i < c.start + c.count; i++)
                nxt[r][(size_t)i] = bufs[r][(size_t)i] + bufs[src][(size_t)i];
        }
        bufs.swap(nxt);
    }
    for (int step = 0; step < world - 1; step++) {
        std::vector<std::vector<float>> nxt = bufs;
        for (int r = 0; r < world; r++) {
            int src = (r - step - 1 + world * 2) % world;
            int chunk = (r - step - 1 + world * 2) % world;
            RankSlice c = slice_for((int64_t)n, chunk, world);
            for (int64_t i = c.start; i < c.start + c.count; i++)
                nxt[r][(size_t)i] = bufs[src][(size_t)i];
        }
        bufs.swap(nxt);
    }
}

void broadcast_bufs(std::vector<std::vector<float>>& bufs, int src) {
    if (bufs.empty()) return;
    if (src < 0 || src >= (int)bufs.size()) throw std::runtime_error("[parallel] bad src");
    for (size_t r = 0; r < bufs.size(); r++)
        if ((int)r != src) bufs[r] = bufs[(size_t)src];
}

void average_grad_slices(std::vector<Tensor>& grads, int world) {
    if (world <= 1) return;
    for (auto& g : grads) {
        float* d = g.data<float>();
        int64_t n = g.numel();
        for (int64_t i = 0; i < n; i++) d[i] /= (float)world;
    }
}

static Tensor slice_last_dim(const Tensor& w, int64_t start, int64_t count) {
    if (w.rank() != 2) throw std::runtime_error("[parallel] TP needs 2-D");
    int64_t rows = w.dim(0), cols = w.dim(1);
    if (start < 0 || start + count > cols) throw std::runtime_error("[parallel] TP slice OOB");
    const float* s = w.data<float>();
    Tensor o = Tensor::zeros({rows, count});
    float* d = o.data<float>();
    for (int64_t r = 0; r < rows; r++)
        std::memcpy(d + r * count, s + r * cols + start, (size_t)count * sizeof(float));
    return o;
}

TPSplit tp_split_cols(const Tensor& w, int world) {
    TPSplit o;
    o.dim = 1;
    for (int r = 0; r < world; r++) {
        RankSlice s = slice_for(w.dim(1), r, world);
        o.shards.push_back(slice_last_dim(w, s.start, s.count));
    }
    return o;
}

Tensor tp_gather_cols(const std::vector<Tensor>& shards) {
    if (shards.empty()) throw std::runtime_error("[parallel] no shards");
    int64_t rows = shards[0].dim(0);
    int64_t cols = 0;
    for (auto& s : shards) {
        if (s.rank() != 2 || s.dim(0) != rows) throw std::runtime_error("[parallel] gather mismatch");
        cols += s.dim(1);
    }
    Tensor o = Tensor::zeros({rows, cols});
    float* d = o.data<float>();
    int64_t off = 0;
    for (auto& s : shards) {
        const float* x = s.data<float>();
        int64_t c = s.dim(1);
        for (int64_t r = 0; r < rows; r++)
            std::memcpy(d + r * cols + off, x + r * c, (size_t)c * sizeof(float));
        off += c;
    }
    return o;
}

TPSplit tp_split_rows(const Tensor& w, int world) {
    if (w.rank() != 2) throw std::runtime_error("[parallel] TP needs 2-D");
    TPSplit o;
    o.dim = 0;
    int64_t rows = w.dim(0), cols = w.dim(1);
    const float* s = w.data<float>();
    for (int r = 0; r < world; r++) {
        RankSlice sl = slice_for(rows, r, world);
        Tensor t = Tensor::zeros({sl.count, cols});
        float* d = t.data<float>();
        std::memcpy(d, s + sl.start * cols, (size_t)(sl.count * cols) * sizeof(float));
        o.shards.push_back(t);
    }
    return o;
}

Tensor tp_gather_rows(const std::vector<Tensor>& shards) {
    if (shards.empty()) throw std::runtime_error("[parallel] no shards");
    int64_t cols = shards[0].dim(1);
    int64_t rows = 0;
    for (auto& s : shards) {
        if (s.rank() != 2 || s.dim(1) != cols) throw std::runtime_error("[parallel] gather mismatch");
        rows += s.dim(0);
    }
    Tensor o = Tensor::zeros({rows, cols});
    float* d = o.data<float>();
    int64_t off = 0;
    for (auto& s : shards) {
        const float* x = s.data<float>();
        int64_t n = s.numel();
        std::memcpy(d + off, x, (size_t)n * sizeof(float));
        off += n;
    }
    return o;
}

ShardPlan plan_shards(const std::vector<std::string>& names,
                      const std::vector<int64_t>& bytes, int world) {
    if (names.size() != bytes.size()) throw std::runtime_error("[parallel] names/bytes mismatch");
    if (world <= 0) throw std::runtime_error("[parallel] bad world");
    ShardPlan p;
    p.rank_bytes.assign((size_t)world, 0);
    for (size_t i = 0; i < names.size(); i++) {
        int r = (int)(i % (size_t)world);
        int64_t best = p.rank_bytes[0];
        int bi = 0;
        for (int q = 1; q < world; q++)
            if (p.rank_bytes[(size_t)q] < best) { best = p.rank_bytes[(size_t)q]; bi = q; }
        r = bi;
        ShardPlan::Entry e;
        e.name = names[i];
        e.rank = r;
        e.bytes = bytes[i] * 4;
        p.entries.push_back(e);
        p.rank_bytes[(size_t)r] += e.bytes;
        p.total_bytes += e.bytes;
    }
    return p;
}

double ShardPlan::imbalance() const {
    if (rank_bytes.empty() || total_bytes <= 0) return 0.0;
    double mean = (double)total_bytes / (double)rank_bytes.size();
    double mx = 0.0;
    for (auto v : rank_bytes) mx = std::max(mx, (double)v);
    return mx / mean - 1.0;
}

PipePlan plan_pipeline(int64_t layers, int stages) {
    if (stages <= 0 || layers <= 0) throw std::runtime_error("[parallel] bad pipe plan");
    PipePlan p;
    p.stages = stages;
    p.layer_stage.assign((size_t)layers, 0);
    p.stage_layers.assign((size_t)stages, 0);
    for (int64_t l = 0; l < layers; l++) {
        int s = (int)(l * stages / layers);
        if (s >= stages) s = stages - 1;
        p.layer_stage[(size_t)l] = s;
        p.stage_layers[(size_t)s]++;
    }
    return p;
}

} // namespace dense_engine
} // namespace quant
