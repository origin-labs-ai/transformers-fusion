// Phase 5-F6 split of src/inference/inference_opt.cpp (verbatim move, no behavior change).
// This file: PagedAttention (D1) + ContinuousBatching (D3) + RequestScheduler (D12)
// + InferenceMemoryPool (D13). Declarations stay in include/quant/inference_opt.h.
#include "quant/inference_opt.h"
#include "quant/math.h"
#include "quant/int8_quant.h"
#include "quant/random.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <random>
#include <sstream>
#include <cstdio>
namespace quant {

// H3: unified epsilon shared with inference_opt.cpp / inference_speculative.cpp
// (same name, same value, file-local so no header/ODR touch).
namespace { constexpr float kInferenceEps = 1e-10f; }

// ===========================================================================
// D1: Paged attention — vLLM-style block-level KV cache management
// ===========================================================================
PagedAttention::PagedAttention(int64_t head_dim, int64_t n_heads, int64_t block_size,
                               int64_t max_blocks)
    : head_dim_(head_dim), n_heads_(n_heads), block_size_(block_size),
      max_blocks_(max_blocks), next_id_(0) {
    int64_t reserve_count = std::min(max_blocks_, (int64_t)256);
    for (int64_t i = 0; i < reserve_count; i++) {
        Block b;
        b.id = i;
        b.k = Tensor::zeros(Shape{1, n_heads, block_size, head_dim});
        b.v = Tensor::zeros(Shape{1, n_heads, block_size, head_dim});
        b.active = false;
        blocks_.push_back(b);
        free_ids_.push(i);
        next_id_ = i + 1;
    }
}

PagedAttention::Block PagedAttention::alloc_block() {
    if (free_ids_.empty()) {
        if (next_id_ >= max_blocks_) return Block{-1, Tensor(), Tensor(), false};
        int64_t id = next_id_++;
        Block b;
        b.id = id;
        b.k = Tensor::zeros(Shape{1, n_heads_, block_size_, head_dim_});
        b.v = Tensor::zeros(Shape{1, n_heads_, block_size_, head_dim_});
        b.active = true;
        blocks_.push_back(b);
        return blocks_.back();
    }
    int64_t id = free_ids_.front();
    free_ids_.pop();
    blocks_[(size_t)id].active = true;
    blocks_[(size_t)id].k.zero_();
    blocks_[(size_t)id].v.zero_();
    return blocks_[(size_t)id];
}

void PagedAttention::free_block(int64_t id) {
    if (id >= 0 && id < (int64_t)blocks_.size() && blocks_[(size_t)id].active) {
        blocks_[(size_t)id].active = false;
        free_ids_.push(id);
    }
}

Tensor PagedAttention::forward(const Tensor& Q, int64_t* block_table,
                               Block* blocks, int64_t num_blocks) {
    if (Q.rank() < 4) return Tensor::zeros(Shape{1, 1, 1, (int64_t)head_dim_});
    int64_t B = Q.dim(0), H = Q.dim(1), S = Q.dim(2), D = Q.dim(3);
    if (num_blocks <= 0) return Tensor::zeros(Shape{B, H, S, D});
    float scale = 1.0f / std::sqrt((float)D);
    Tensor out({B, H, S, D});
    out.zero_();
    float* od = out.data<float>();
    const float* qd = Q.data<float>();

    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t s = 0; s < S; s++) {
                float row_max = -INFINITY;
                float row_sum = 0;
                float* o_ptr = od + ((b * H + h) * S + s) * D;
                const float* q_ptr = qd + ((b * H + h) * S + s) * D;

                for (int64_t blk = 0; blk < num_blocks; blk++) {
                    int64_t blk_id = block_table[blk];
                    if (blk_id < 0 || blk_id >= (int64_t)blocks_.size()) continue;
                    const float* kblk = blocks[blk_id].k.data<float>() + (h * block_size_ * D);
                    const float* vblk = blocks[blk_id].v.data<float>() + (h * block_size_ * D);

                    for (int64_t p = 0; p < block_size_; p++) {
                        float dot = 0;
                        for (int64_t d = 0; d < D; d++)
                            dot += q_ptr[d] * kblk[p * D + d];
                        float score = dot * scale;
                        float new_max = std::max(row_max, score);
                        float exp_diff = (row_max != -INFINITY) ? std::exp(row_max - new_max) : 0.0f;
                        row_sum *= exp_diff;
                        for (int64_t d = 0; d < D; d++)
                            o_ptr[d] *= exp_diff;
                        float e = std::exp(score - new_max);
                        row_sum += e;
                        for (int64_t d = 0; d < D; d++)
                            o_ptr[d] += e * vblk[p * D + d];
                        row_max = new_max;
                    }
                }
                float inv = 1.0f / (row_sum + kInferenceEps);
                for (int64_t d = 0; d < D; d++)
                    o_ptr[d] *= inv;
            }
        }
    }
    return out;
}

// ===========================================================================
// D3: Continuous batching — dynamic request scheduling with masking
// ===========================================================================
ContinuousBatching::ContinuousBatching(Model* model, int max_batch)
    : model_(model), max_batch_(max_batch) {}

void ContinuousBatching::add_request(const BatchRequest& req) {
    queue_.push(req);
}

Tensor ContinuousBatching::build_attention_mask(int64_t B, int64_t S,
                                                 const std::vector<int>& seq_lens) const {
    Tensor mask(Shape{B, 1, S, S});
    mask.fill(-INFINITY);
    for (int64_t b = 0; b < B; b++) {
        int sl = seq_lens[(size_t)b];
        for (int64_t i = 0; i < S; i++) {
            for (int64_t j = 0; j < S; j++) {
                if (j <= i && j < sl && i < sl)
                    mask.data<float>()[b * S * S + i * S + j] = 0.0f;
            }
        }
    }
    return mask;
}

BatchResponse ContinuousBatching::step() {
    while (!queue_.empty() && (int)active_.size() < max_batch_) {
        active_.push_back(queue_.front());
        outputs_.push_back({});
        if (model_) {
            kv_caches_.emplace_back((int)model_->config.num_layers, model_->config.max_seq_len,
                                    model_->config.num_heads, model_->config.head_dim);
        } else {
            kv_caches_.emplace_back(12, 2048, 12, 64);
        }
        queue_.pop();
    }

    if (active_.empty()) return BatchResponse{};

    int64_t B = (int64_t)active_.size();

    // D6 W3: per-request decode — no batched Model::forward call, no mask.
    // Each active request advances via its own single-token input/position
    // and its own KV cache entry:
    //   model_->forward(single_input, single_pos, &kv_caches_[b])
    // using the unchanged Model::forward(input_ids, positions, cache*)
    // signature (model.h) — per-request loop only, no API change.
    // S==1 decode makes any causal mask vacuous (one query position; prefix
    // history lives in the per-request KV cache), so no mask is built or
    // passed here. build_attention_mask() is retained for compatibility but
    // intentionally unused on this path.
    static bool step_warned_once = false;
    if (!step_warned_once) {
        step_warned_once = true;
        std::fprintf(stderr,
            "[inference_opt][W3] ContinuousBatching::step: per-request "
            "forward(single_input, single_pos, &kv_caches_[b]); S==1 mask "
            "vacuous, no mask built/passed (Model::forward takes no mask).\n");
    }

    Tensor batch_logits;
    if (model_) {
        std::vector<Tensor> per_req;
        per_req.reserve((size_t)B);
        int64_t V = 0;
        for (int64_t b = 0; b < B; b++) {
            Tensor single_input(Shape{1, 1});
            Tensor single_pos(Shape{1, 1});
            single_input.data<float>()[0] = (float)active_[(size_t)b].tokens.back();
            single_pos.data<float>()[0] = (float)((int)active_[(size_t)b].tokens.size() - 1);
            Tensor out = model_->forward(single_input, single_pos, &kv_caches_[(size_t)b]);
            if (b == 0) V = out.dim(2);
            per_req.push_back(out);
        }
        batch_logits = Tensor(Shape{B, 1, V});
        batch_logits.zero_();
        float* dst = batch_logits.data<float>();
        for (int64_t b = 0; b < B; b++) {
            const float* src = per_req[(size_t)b].data<float>();
            std::memcpy(dst + b * V, src, (size_t)V * sizeof(float));
        }
    } else {
        batch_logits = Tensor(Shape{B, 1, 32000});
        batch_logits.zero_();
    }

    int64_t V = batch_logits.dim(2);
    for (int64_t b = 0; b < B; b++) {
        const float* row = batch_logits.data<float>() + b * V;
        int best = 0;
        for (int v = 1; v < V; v++)
            if (row[v] > row[best]) best = v;
        active_[(size_t)b].tokens.push_back(best);
        outputs_[(size_t)b].push_back(best);
    }

    BatchResponse resp;
    resp.id = active_.empty() ? -1 : active_[0].id;
    for (size_t i = 0; i < active_.size(); i++) {
        if ((int)active_[i].tokens.size() >= active_[i].max_tokens) {
            resp.id = active_[i].id;
            std::ostringstream oss;
            for (int t : outputs_[i]) oss << t << " ";
            resp.text = oss.str();
            active_.erase(active_.begin() + (int64_t)i);
            outputs_.erase(outputs_.begin() + (int64_t)i);
            kv_caches_.erase(kv_caches_.begin() + (int64_t)i);
            return resp;
        }
    }

    return resp;
}

bool ContinuousBatching::has_pending() const {
    return !queue_.empty() || !active_.empty();
}

// ===========================================================================
// D12: Request scheduling — priority queue with deadlines
// ===========================================================================
void RequestScheduler::add(const Request& req) { queue_.push(req); }

bool RequestScheduler::Compare::operator()(const Request& a, const Request& b) {
    if (a.priority != b.priority)
        return a.priority < b.priority;
    return a.deadline > b.deadline;
}

Request RequestScheduler::next() {
    Request r = queue_.top();
    queue_.pop();
    return r;
}

bool RequestScheduler::has_next() const { return !queue_.empty(); }

// ===========================================================================
// D13: Memory pool — thread-safe pool allocator
// ===========================================================================
InferenceMemoryPool::InferenceMemoryPool(size_t block_size, int64_t num_blocks)
    : block_size_(block_size), capacity_(num_blocks),
      pool_(block_size * num_blocks), free_list_(num_blocks, true) {}

void* InferenceMemoryPool::alloc() {
    std::lock_guard<std::mutex> lock(mtx_);
    for (int64_t i = 0; i < capacity_; i++) {
        if (free_list_[(size_t)i]) {
            free_list_[(size_t)i] = false;
            used_++;
            return &pool_[(size_t)i * block_size_];
        }
    }
    return nullptr;
}

void InferenceMemoryPool::free(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mtx_);
    int64_t idx = ((char*)ptr - &pool_[0]) / (int64_t)block_size_;
    if (idx >= 0 && idx < capacity_ && !free_list_[(size_t)idx]) {
        free_list_[(size_t)idx] = true;
        used_--;
    }
}

} // namespace quant
