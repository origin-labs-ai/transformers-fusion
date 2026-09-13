#define NOMINMAX
#include "quant/trainer.h"
#include "quant/math.h"
#include "quant/autograd.h"
#include "quant/optimizer.h"
#include "quant/transformer.h"
#include "quant/moe_model.h"
#include "quant/flash_attention.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <set>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace quant {

// ============================================================================
// H1 file-local differentiable ops (D5 W2/W3 single-graph PPO+DPO).
// The engine header is frozen, so the missing elementwise/reduction ops are
// defined here, in the one file this task owns. Each op follows the exact
// AutogradEngine registration pattern (fn->forward, requires_grad(true),
// AutogradNode with shared-buffer input/output copies). Constant tensors
// (old logprobs, advantages, returns, ref logprobs, scalar weights) are plain
// host-filled Tensors with requires_grad(false): they enter nodes as inputs
// but the engine skips gradient propagation into them.
// ============================================================================
namespace rl_graph {

inline Tensor emit_node(std::shared_ptr<AutogradFunction> fn,
                        const std::vector<Tensor>& inputs,
                        Tensor out) {
    if (AutogradEngine::enabled()) {
        out.requires_grad(true);
        auto node = std::make_shared<AutogradNode>();
        node->fn = fn;
        node->inputs = inputs;
        node->outputs = {out};
        AutogradEngine::instance().register_node(node);
    }
    return out;
}

inline Tensor rl_const(const Shape& s, float v) {
    Tensor t(s);
    t.fill(v);
    return t;
}

class RlSubFn : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs;
        Tensor out(inputs[0].shape(), DType::F32);
        const float* a = inputs[0].data<float>();
        const float* b = inputs[1].data<float>();
        float* o = out.data<float>();
        int64_t n = inputs[0].numel();
        for (int64_t i = 0; i < n; i++) o[i] = a[i] - b[i];
        return {out};
    }
    std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
        const Tensor& g = go[0];
        Tensor ga(g.shape(), DType::F32), gb(g.shape(), DType::F32);
        const float* gd = g.data<float>();
        float* ad = ga.data<float>();
        float* bd = gb.data<float>();
        int64_t n = g.numel();
        for (int64_t i = 0; i < n; i++) { ad[i] = gd[i]; bd[i] = -gd[i]; }
        return {ga, gb};
    }
};

class RlExpFn : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs;
        Tensor out(inputs[0].shape(), DType::F32);
        const float* x = inputs[0].data<float>();
        float* o = out.data<float>();
        int64_t n = inputs[0].numel();
        for (int64_t i = 0; i < n; i++) o[i] = std::exp(x[i]);
        saved.push_back(out);
        return {out};
    }
    std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
        const Tensor& g = go[0];
        const Tensor& e = saved[1];
        Tensor dx(g.shape(), DType::F32);
        const float* gd = g.data<float>();
        const float* ed = e.data<float>();
        float* dd = dx.data<float>();
        int64_t n = g.numel();
        for (int64_t i = 0; i < n; i++) dd[i] = gd[i] * ed[i];
        return {dx};
    }
};

class RlClipFn : public AutogradFunction {
public:
    RlClipFn(float lo, float hi) : lo_(lo), hi_(hi) {}
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs;
        Tensor out(inputs[0].shape(), DType::F32);
        const float* x = inputs[0].data<float>();
        float* o = out.data<float>();
        int64_t n = inputs[0].numel();
        for (int64_t i = 0; i < n; i++)
            o[i] = x[i] < lo_ ? lo_ : (x[i] > hi_ ? hi_ : x[i]);
        return {out};
    }
    std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
        const Tensor& g = go[0];
        const Tensor& x = saved[0];
        Tensor dx(g.shape(), DType::F32);
        const float* gd = g.data<float>();
        const float* xd = x.data<float>();
        float* dd = dx.data<float>();
        int64_t n = g.numel();
        for (int64_t i = 0; i < n; i++)
            dd[i] = (xd[i] >= lo_ && xd[i] <= hi_) ? gd[i] : 0.0f;
        return {dx};
    }
private:
    float lo_, hi_;
};

class RlMinFn : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs;
        Tensor out(inputs[0].shape(), DType::F32);
        const float* a = inputs[0].data<float>();
        const float* b = inputs[1].data<float>();
        float* o = out.data<float>();
        int64_t n = inputs[0].numel();
        for (int64_t i = 0; i < n; i++) o[i] = a[i] < b[i] ? a[i] : b[i];
        return {out};
    }
    std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
        const Tensor& g = go[0];
        const Tensor& a = saved[0];
        const Tensor& b = saved[1];
        Tensor ga(g.shape(), DType::F32), gb(g.shape(), DType::F32);
        ga.zero_(); gb.zero_();
        const float* gd = g.data<float>();
        const float* ad = a.data<float>();
        const float* bd = b.data<float>();
        float* gad = ga.data<float>();
        float* gbd = gb.data<float>();
        int64_t n = g.numel();
        for (int64_t i = 0; i < n; i++) {
            if (ad[i] < bd[i]) gad[i] = gd[i];
            else gbd[i] = gd[i];
        }
        return {ga, gb};
    }
};

class RlMeanAllFn : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs;
        const Tensor& x = inputs[0];
        double s = 0.0;
        const float* xd = x.data<float>();
        int64_t n = x.numel();
        for (int64_t i = 0; i < n; i++) s += (double)xd[i];
        Tensor out(Shape{1});
        out.data<float>()[0] = (float)(s / (double)(n > 0 ? n : 1));
        return {out};
    }
    std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
        const Tensor& x = saved[0];
        int64_t n = x.numel();
        Tensor dx(x.shape(), DType::F32);
        float fill = go[0].data<float>()[0] / (float)(n > 0 ? n : 1);
        float* dd = dx.data<float>();
        for (int64_t i = 0; i < n; i++) dd[i] = fill;
        return {dx};
    }
};

// Sum over the last dimension: {.., K} -> {..}. Covers entropy row sums
// ({B,S,V} -> {B,S}) and DPO per-sequence sums ({B,S} -> {B}).
class RlSumLastFn : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs;
        const Tensor& x = inputs[0];
        last_ = x.dim(x.rank() - 1);
        outer_ = x.numel() / last_;
        Shape os;
        os.rank = x.rank() - 1;
        for (int i = 0; i < os.rank; i++) os.dims[i] = x.dim(i);
        Tensor out(os, DType::F32);
        const float* xd = x.data<float>();
        float* od = out.data<float>();
        for (int64_t r = 0; r < outer_; r++) {
            double s = 0.0;
            for (int64_t c = 0; c < last_; c++) s += (double)xd[r * last_ + c];
            od[r] = (float)s;
        }
        return {out};
    }
    std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
        const Tensor& x = saved[0];
        Tensor dx(x.shape(), DType::F32);
        const float* gd = go[0].data<float>();
        float* dd = dx.data<float>();
        for (int64_t r = 0; r < outer_; r++)
            for (int64_t c = 0; c < last_; c++) dd[r * last_ + c] = gd[r];
        return {dx};
    }
private:
    int64_t last_ = 1, outer_ = 0;
};

// Row-wise softmax over the last dimension (any rank >= 2).
class RlSoftmaxRowsFn : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs;
        const Tensor& x = inputs[0];
        int64_t last = x.dim(x.rank() - 1);
        int64_t rows = x.numel() / last;
        Tensor out(x.shape(), DType::F32);
        const float* xd = x.data<float>();
        float* od = out.data<float>();
        for (int64_t r = 0; r < rows; r++) {
            float m = -INFINITY;
            for (int64_t c = 0; c < last; c++)
                if (xd[r * last + c] > m) m = xd[r * last + c];
            float s = 0.0f;
            for (int64_t c = 0; c < last; c++) {
                od[r * last + c] = std::exp(xd[r * last + c] - m);
                s += od[r * last + c];
            }
            float inv = 1.0f / (s + 1e-10f);
            for (int64_t c = 0; c < last; c++) od[r * last + c] *= inv;
        }
        saved.push_back(out);
        return {out};
    }
    std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
        const Tensor& g = go[0];
        const Tensor& p = saved[1];
        int64_t last = p.dim(p.rank() - 1);
        int64_t rows = p.numel() / last;
        Tensor dx(p.shape(), DType::F32);
        const float* pd = p.data<float>();
        const float* gd = g.data<float>();
        float* dd = dx.data<float>();
        for (int64_t r = 0; r < rows; r++) {
            double dot = 0.0;
            for (int64_t c = 0; c < last; c++)
                dot += (double)pd[r * last + c] * (double)gd[r * last + c];
            for (int64_t c = 0; c < last; c++) {
                int64_t i = r * last + c;
                dd[i] = (float)((double)pd[i] * ((double)gd[i] - dot));
            }
        }
        return {dx};
    }
};

class RlLogFn : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs;
        Tensor out(inputs[0].shape(), DType::F32);
        const float* x = inputs[0].data<float>();
        float* o = out.data<float>();
        int64_t n = inputs[0].numel();
        for (int64_t i = 0; i < n; i++)
            o[i] = std::log(std::max(x[i], 1e-10f));
        return {out};
    }
    std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
        const Tensor& g = go[0];
        const Tensor& x = saved[0];
        Tensor dx(g.shape(), DType::F32);
        const float* gd = g.data<float>();
        const float* xd = x.data<float>();
        float* dd = dx.data<float>();
        int64_t n = g.numel();
        for (int64_t i = 0; i < n; i++) dd[i] = gd[i] / std::max(xd[i], 1e-10f);
        return {dx};
    }
};

class RlSigmoidFn : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs;
        Tensor out(inputs[0].shape(), DType::F32);
        const float* x = inputs[0].data<float>();
        float* o = out.data<float>();
        int64_t n = inputs[0].numel();
        for (int64_t i = 0; i < n; i++)
            o[i] = 1.0f / (1.0f + std::exp(-x[i]));
        saved.push_back(out);
        return {out};
    }
    std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
        const Tensor& g = go[0];
        const Tensor& s = saved[1];
        Tensor dx(g.shape(), DType::F32);
        const float* gd = g.data<float>();
        const float* sd = s.data<float>();
        float* dd = dx.data<float>();
        int64_t n = g.numel();
        for (int64_t i = 0; i < n; i++) dd[i] = gd[i] * sd[i] * (1.0f - sd[i]);
        return {dx};
    }
};

class RlReluFn : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs;
        Tensor out(inputs[0].shape(), DType::F32);
        const float* x = inputs[0].data<float>();
        float* o = out.data<float>();
        int64_t n = inputs[0].numel();
        for (int64_t i = 0; i < n; i++) o[i] = x[i] > 0.0f ? x[i] : 0.0f;
        return {out};
    }
    std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
        return {relu_grad(saved[0], go[0])};
    }
};

// Per-token log-softmax + gather: logits {B,S,V} with ids {B,S} -> {B,S}.
// Forward values are bit-identical to PPOTrainer::compute_log_probs; the
// backward recomputes row softmax and returns dlogits = g * (p - one_hot).
class RlSeqGatherFn : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs;
        const Tensor& logits = inputs[0];
        const Tensor& ids = inputs[1];
        B_ = logits.dim(0);
        S_ = logits.dim(1);
        V_ = logits.dim(logits.rank() - 1);
        Tensor out(Shape{B_, S_});
        const float* lp = logits.data<float>();
        const float* ip = ids.data<float>();
        float* op = out.data<float>();
        for (int64_t i = 0; i < B_ * S_; i++) {
            int64_t t = (i < ids.numel()) ? (int64_t)ip[i] : 0;
            if (t < 0) t = 0;
            if (t >= V_) t = V_ - 1;
            const float* row = lp + i * V_;
            float m = -INFINITY;
            for (int64_t v = 0; v < V_; v++) if (row[v] > m) m = row[v];
            float se = 0.0f;
            for (int64_t v = 0; v < V_; v++) se += std::exp(row[v] - m);
            op[i] = row[t] - m - std::log(se + 1e-10f);
        }
        return {out};
    }
    std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
        const Tensor& g = go[0];
        const Tensor& logits = saved[0];
        const Tensor& ids = saved[1];
        Tensor dx(logits.shape(), DType::F32);
        dx.zero_();
        const float* lp = logits.data<float>();
        const float* ip = ids.data<float>();
        const float* gd = g.data<float>();
        float* dd = dx.data<float>();
        for (int64_t i = 0; i < B_ * S_; i++) {
            int64_t t = (i < ids.numel()) ? (int64_t)ip[i] : 0;
            if (t < 0) t = 0;
            if (t >= V_) t = V_ - 1;
            const float* row = lp + i * V_;
            float m = -INFINITY;
            for (int64_t v = 0; v < V_; v++) if (row[v] > m) m = row[v];
            float se = 0.0f;
            for (int64_t v = 0; v < V_; v++) se += std::exp(row[v] - m);
            float inv = 1.0f / (se + 1e-10f);
            float gi = gd[i];
            for (int64_t v = 0; v < V_; v++) {
                float p = std::exp(row[v] - m) * inv;
                dd[i * V_ + v] = gi * (p - (v == t ? 1.0f : 0.0f));
            }
        }
        return {dx, Tensor()};
    }
private:
    int64_t B_ = 0, S_ = 0, V_ = 0;
};

// Mean-pool over the sequence axis with vocab->hidden fit:
// logits {B,S,V} -> pooled {B,H} (mean over S; truncate V to H or zero-pad).
// Headers are frozen, so the value trunk cannot read last-layer hidden states;
// this linear pool is the in-graph root that keeps value-MSE gradients flowing
// through one shared graph (standard shared-trunk PPO semantics).
class RlMeanPoolSFn : public AutogradFunction {
public:
    explicit RlMeanPoolSFn(int64_t H) : H_(H) {}
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs;
        const Tensor& x = inputs[0];
        B_ = x.dim(0);
        S_ = x.dim(1);
        V_ = x.dim(x.rank() - 1);
        Tensor out(Shape{B_, H_});
        out.zero_();
        const float* xd = x.data<float>();
        float* od = out.data<float>();
        int64_t keep = std::min(V_, H_);
        for (int64_t b = 0; b < B_; b++)
            for (int64_t s = 0; s < S_; s++)
                for (int64_t h = 0; h < keep; h++)
                    od[b * H_ + h] += xd[(b * S_ + s) * V_ + h];
        float inv = 1.0f / (float)(S_ > 0 ? S_ : 1);
        for (int64_t i = 0; i < B_ * H_; i++) od[i] *= inv;
        return {out};
    }
    std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
        const Tensor& x = saved[0];
        Tensor dx(x.shape(), DType::F32);
        dx.zero_();
        const float* gd = go[0].data<float>();
        float* dd = dx.data<float>();
        int64_t keep = std::min(V_, H_);
        float inv = 1.0f / (float)(S_ > 0 ? S_ : 1);
        for (int64_t b = 0; b < B_; b++)
            for (int64_t s = 0; s < S_; s++)
                for (int64_t h = 0; h < keep; h++)
                    dd[(b * S_ + s) * V_ + h] += gd[b * H_ + h] * inv;
        return {dx};
    }
private:
    int64_t H_, B_ = 0, S_ = 0, V_ = 0;
};

inline Tensor rl_sub_op(const Tensor& a, const Tensor& b) {
    auto fn = std::make_shared<RlSubFn>();
    auto o = fn->forward({a, b});
    return emit_node(fn, {a, b}, o[0]);
}
inline Tensor rl_exp_op(const Tensor& x) {
    auto fn = std::make_shared<RlExpFn>();
    auto o = fn->forward({x});
    return emit_node(fn, {x}, o[0]);
}
inline Tensor rl_clip_op(const Tensor& x, float lo, float hi) {
    auto fn = std::make_shared<RlClipFn>(lo, hi);
    auto o = fn->forward({x});
    return emit_node(fn, {x}, o[0]);
}
inline Tensor rl_min_op(const Tensor& a, const Tensor& b) {
    auto fn = std::make_shared<RlMinFn>();
    auto o = fn->forward({a, b});
    return emit_node(fn, {a, b}, o[0]);
}
inline Tensor rl_mean_all_op(const Tensor& x) {
    auto fn = std::make_shared<RlMeanAllFn>();
    auto o = fn->forward({x});
    return emit_node(fn, {x}, o[0]);
}
inline Tensor rl_sum_last_op(const Tensor& x) {
    auto fn = std::make_shared<RlSumLastFn>();
    auto o = fn->forward({x});
    return emit_node(fn, {x}, o[0]);
}
inline Tensor rl_softmax_rows_op(const Tensor& x) {
    auto fn = std::make_shared<RlSoftmaxRowsFn>();
    auto o = fn->forward({x});
    return emit_node(fn, {x}, o[0]);
}
inline Tensor rl_log_op(const Tensor& x) {
    auto fn = std::make_shared<RlLogFn>();
    auto o = fn->forward({x});
    return emit_node(fn, {x}, o[0]);
}
inline Tensor rl_sigmoid_op(const Tensor& x) {
    auto fn = std::make_shared<RlSigmoidFn>();
    auto o = fn->forward({x});
    return emit_node(fn, {x}, o[0]);
}
inline Tensor rl_relu_op(const Tensor& x) {
    auto fn = std::make_shared<RlReluFn>();
    auto o = fn->forward({x});
    return emit_node(fn, {x}, o[0]);
}
inline Tensor rl_gather_op(const Tensor& logits, const Tensor& ids) {
    auto fn = std::make_shared<RlSeqGatherFn>();
    auto o = fn->forward({logits, ids});
    return emit_node(fn, {logits, ids}, o[0]);
}
inline Tensor rl_mean_pool_s_op(const Tensor& logits, int64_t H) {
    auto fn = std::make_shared<RlMeanPoolSFn>(H);
    auto o = fn->forward({logits});
    return emit_node(fn, {logits}, o[0]);
}

// Shared policy-parameter collector (Dense trunk + full MoE stack, mirrors
// GRPOTrainer collection so gradients reach every expert/router/head).
inline std::vector<Tensor*> collect_rl_policy_params(Model* policy) {
    std::vector<Tensor*> p;
    if (!policy) return p;
    if (auto* dm = dynamic_cast<DenseModel*>(policy)) {
        collect_dense_params(dm, p);
        return p;
    }
    if (auto* mm = dynamic_cast<MoEModel*>(policy)) {
        p.push_back(&mm->tok_embeddings->weight);
        for (auto& l : mm->layers) {
            p.push_back(&l.attention_norm.weight);
            p.push_back(&l.attention.q_proj.weight);
            if (l.attention.q_proj.bias.numel() > 0) p.push_back(&l.attention.q_proj.bias);
            p.push_back(&l.attention.k_proj.weight);
            if (l.attention.k_proj.bias.numel() > 0) p.push_back(&l.attention.k_proj.bias);
            p.push_back(&l.attention.v_proj.weight);
            if (l.attention.v_proj.bias.numel() > 0) p.push_back(&l.attention.v_proj.bias);
            p.push_back(&l.attention.o_proj.weight);
            if (l.attention.o_proj.bias.numel() > 0) p.push_back(&l.attention.o_proj.bias);
            p.push_back(&l.ffn_norm.weight);
            p.push_back(&l.moe->router_weight.weight);
            for (auto& e : l.moe->experts) {
                p.push_back(&e.gate_proj.weight);
                p.push_back(&e.up_proj.weight);
                p.push_back(&e.down_proj.weight);
            }
            if (l.shared_expert) {
                p.push_back(&l.shared_expert->gate_proj.weight);
                p.push_back(&l.shared_expert->up_proj.weight);
                p.push_back(&l.shared_expert->down_proj.weight);
            }
        }
        p.push_back(&mm->norm->weight);
        p.push_back(&mm->lm_head->weight);
        if (mm->lm_head->bias.numel() > 0) p.push_back(&mm->lm_head->bias);
        return p;
    }
    return p;
}

} // namespace rl_graph

PPOTrainer::PPOTrainer(Model* policy, Model* ref_model, float clip_epsilon,
                       float value_coef, float entropy_coef,
                       float gamma, float gae_lambda, float kl_target)
    : policy_(policy), ref_model_(ref_model),
      clip_epsilon_(clip_epsilon), value_coef_(value_coef),
      entropy_coef_(entropy_coef), gamma_(gamma),
      gae_lambda_(gae_lambda), kl_target_(kl_target), kl_alpha_(0.0f) {
    int64_t hs = policy_->config.hidden_size;
    // Critic weights use the engine {out, in} layout so critic_forward can be
    // built from AutogradEngine::matmul_op (matmul_op expects b as {N, K}).
    v_fc1_weight_ = Tensor(Shape{64, hs});
    v_fc1_bias_ = Tensor(Shape{64});
    v_fc2_weight_ = Tensor(Shape{1, 64});
    v_fc2_bias_ = Tensor(Shape{1});
    static thread_local std::mt19937 rng(std::random_device{}());
    float scale = 1.0f / std::sqrt((float)hs);
    float* ptr = v_fc1_weight_.data<float>();
    for (int64_t i = 0; i < v_fc1_weight_.numel(); i++)
        ptr[i] = ((float)(std::uniform_int_distribution<int>(0, 1999)(rng)) / 1000.0f - 1.0f) * scale;
    ptr = v_fc1_bias_.data<float>();
    for (int64_t i = 0; i < v_fc1_bias_.numel(); i++) ptr[i] = 0.0f;
    ptr = v_fc2_weight_.data<float>();
    for (int64_t i = 0; i < v_fc2_weight_.numel(); i++)
        ptr[i] = ((float)(std::uniform_int_distribution<int>(0, 1999)(rng)) / 1000.0f - 1.0f) * 0.1f;
    ptr = v_fc2_bias_.data<float>();
    ptr[0] = 0.0f;
}

Tensor PPOTrainer::critic_forward(const Tensor& hidden) {
    // Rebuilt on engine ops: matmul + bias_add + file-local ReLU, so the value
    // head is a differentiable subgraph of the single PPO graph. Input must be
    // pooled hidden {B, H} (train_step feeds mean-pooled logits); feeding raw
    // token-id states here was the old shape bug (states dim(1) == S, not H).
    int64_t B = hidden.dim(0);
    int64_t H = v_fc1_weight_.dim(1); // {64, H} engine layout
    QUANT_CHECK(hidden.dim(1) == H, "critic_forward: expected pooled hidden {B,H}");
    Tensor h1 = AutogradEngine::matmul_op(hidden, v_fc1_weight_, B, 64, H);
    h1 = AutogradEngine::bias_add_op(h1, v_fc1_bias_);
    h1 = rl_graph::rl_relu_op(h1);
    Tensor out = AutogradEngine::matmul_op(h1, v_fc2_weight_, B, 1, 64);
    out = AutogradEngine::bias_add_op(out, v_fc2_bias_);
    return rl_graph::rl_sum_last_op(out); // {B, 1} -> {B}
}

Tensor PPOTrainer::compute_log_probs(const Tensor& logits, const Tensor& ids) {
    int64_t B = ids.dim(0);
    int64_t S = ids.dim(1);
    int64_t V = logits.dim(logits.rank() - 1);
    const float* lp = logits.data<float>();
    const float* id = ids.data<float>();
    Tensor logprobs(Shape{B, S});
    float* lpd = logprobs.data<float>();
    for (int64_t i = 0; i < B; i++) {
        for (int64_t j = 0; j < S; j++) {
            int64_t idx = i * S + j;
            int64_t target = (int64_t)id[idx];
            if (target < 0) target = 0;
            if (target >= V) target = V - 1;
            const float* row = lp + idx * V;
            float max_l = -INFINITY;
            for (int64_t v = 0; v < V; v++)
                if (row[v] > max_l) max_l = row[v];
            float sum_exp = 0.0f;
            for (int64_t v = 0; v < V; v++)
                sum_exp += std::exp(row[v] - max_l);
            float log_prob = row[target] - max_l - std::log(sum_exp + 1e-10f);
            lpd[idx] = log_prob;
        }
    }
    return logprobs;
}

Tensor PPOTrainer::compute_gae(const Tensor& rewards, const Tensor& values,
                                const Tensor& dones) {
    int64_t T = rewards.dim(0);
    const float* rd = rewards.data<float>();
    const float* vd = values.data<float>();
    const float* dd = dones.data<float>();
    Tensor advantages(Shape{T});
    Tensor returns(Shape{T});
    float* ad = advantages.data<float>();
    float* retd = returns.data<float>();
    float gae = 0.0f;
    float next_val = 0.0f;
    for (int64_t t = T - 1; t >= 0; t--) {
        float done_mask = 1.0f - dd[t];
        float delta = rd[t] + gamma_ * next_val * done_mask - vd[t];
        gae = delta + gamma_ * gae_lambda_ * done_mask * gae;
        ad[t] = gae;
        retd[t] = ad[t] + vd[t];
        next_val = vd[t];
    }
    return advantages;
}

float PPOTrainer::compute_kl_divergence(const Tensor& logits_a, const Tensor& logits_b) {
    int64_t N = logits_a.dim(0);
    int64_t V = logits_a.dim(logits_a.rank() - 1);
    const float* la = logits_a.data<float>();
    const float* lb = logits_b.data<float>();
    float kl = 0.0f;
    for (int64_t i = 0; i < N; i++) {
        float max_a = -INFINITY, max_b = -INFINITY;
        for (int64_t v = 0; v < V; v++) {
            if (la[i * V + v] > max_a) max_a = la[i * V + v];
            if (lb[i * V + v] > max_b) max_b = lb[i * V + v];
        }
        float sum_a = 0.0f, sum_b = 0.0f;
        for (int64_t v = 0; v < V; v++) {
            sum_a += std::exp(la[i * V + v] - max_a);
            sum_b += std::exp(lb[i * V + v] - max_b);
        }
        float inv_a = 1.0f / (sum_a + 1e-10f);
        float inv_b = 1.0f / (sum_b + 1e-10f);
        for (int64_t v = 0; v < V; v++) {
            float pa = std::exp(la[i * V + v] - max_a) * inv_a;
            float pb = std::exp(lb[i * V + v] - max_b) * inv_b;
            if (pa > 1e-10f)
                kl += pa * (std::log(pa + 1e-10f) - std::log(pb + 1e-10f));
        }
    }
    return kl / (float)N;
}

void PPOTrainer::train_step(const Tensor& states, const Tensor& actions,
                             const Tensor& old_logprobs, const Tensor& advantages,
                             const Tensor& returns) {
    // Single-graph PPO (D5 W2/W3): exactly one enabled policy forward; the
    // scalar loss
    //   L = L^pol + value_coef * L^vf - entropy_coef * H + kl_alpha * KL
    // with L^pol = -mean(min(r*A, clip(r)*A)), r = exp(lpi - old),
    // L^vf = 0.5*mean((V - R)^2), H the exact token entropy and KL the
    // Schulman k3 estimator, is built from differentiable ops and differentiated
    // once. Old/advantage/return/ref tensors are constants (no grad).
    int64_t B = states.dim(0);
    int64_t S = actions.dim(1);
    int64_t HS = policy_->config.hidden_size;
    if (B <= 0 || S <= 0) return;
    const int64_t N = B * S;

    Tensor positions(Shape{B, S});
    float* ps = positions.data<float>();
    for (int64_t i = 0; i < N; i++) ps[i] = (float)(i % S);

    // ---- Constant inputs (host-side gathers, never graph nodes) ----
    Tensor olp_c(Shape{B, S});
    {
        const float* src = old_logprobs.data<float>();
        float* dst = olp_c.data<float>();
        int64_t nsrc = old_logprobs.numel();
        if (nsrc == N) {
            for (int64_t i = 0; i < N; i++) dst[i] = src[i];
        } else if (nsrc == B) {
            for (int64_t i = 0; i < B; i++)
                for (int64_t j = 0; j < S; j++) dst[i * S + j] = src[i];
        } else if (nsrc == 1) {
            for (int64_t i = 0; i < N; i++) dst[i] = src[0];
        } else {
            for (int64_t i = 0; i < N; i++) dst[i] = 0.0f;
        }
    }
    Tensor adv_c(Shape{B, S});
    {
        int64_t na = advantages.numel();
        const float* ad = advantages.data<float>();
        float* dst = adv_c.data<float>();
        if (na <= 1) {
            // Single response-level reward: no batch to normalize against, so
            // keep the raw signal (REINFORCE direction) on every token.
            float base = (na == 1) ? ad[0] : 0.0f;
            for (int64_t i = 0; i < N; i++) dst[i] = base;
        } else {
            double mean = 0.0;
            for (int64_t i = 0; i < na; i++) mean += (double)ad[i];
            mean /= (double)na;
            double var = 0.0;
            for (int64_t i = 0; i < na; i++) {
                double d = (double)ad[i] - mean;
                var += d * d;
            }
            double stdv = std::sqrt(var / (double)na + 1e-8);
            for (int64_t i = 0; i < B; i++) {
                float base;
                if (na == N) {
                    double m = 0.0;
                    for (int64_t j = 0; j < S; j++) m += (double)ad[i * S + j];
                    base = (float)((m / (double)S - mean) / stdv);
                } else if (na == B) {
                    base = (float)(((double)ad[i] - mean) / stdv);
                } else {
                    base = (float)(((double)ad[0] - mean) / stdv);
                }
                for (int64_t j = 0; j < S; j++) dst[i * S + j] = base;
            }
        }
    }
    Tensor ret_c(Shape{B});
    {
        const float* src = returns.data<float>();
        float* dst = ret_c.data<float>();
        int64_t nr = returns.numel();
        if (nr == B) {
            for (int64_t i = 0; i < B; i++) dst[i] = src[i];
        } else if (nr == N) {
            for (int64_t i = 0; i < B; i++) {
                double m = 0.0;
                for (int64_t j = 0; j < S; j++) m += (double)src[i * S + j];
                dst[i] = (float)(m / (double)S);
            }
        } else if (nr == 1) {
            for (int64_t i = 0; i < B; i++) dst[i] = src[0];
        } else {
            for (int64_t i = 0; i < B; i++) dst[i] = 0.0f;
        }
    }
    // Reference logprobs are constants: disabled forward + scalar gather.
    Tensor rlp_c(Shape{B, S});
    if (ref_model_) {
        bool ref_was = AutogradEngine::enabled();
        AutogradEngine::set_enabled(false);
        Tensor ref_logits = ref_model_->forward(states, positions, nullptr);
        AutogradEngine::set_enabled(ref_was);
        int64_t RV = ref_logits.dim(ref_logits.rank() - 1);
        int64_t rows = ref_logits.numel() / RV;
        const float* rp = ref_logits.data<float>();
        const float* ap = actions.data<float>();
        float* dst = rlp_c.data<float>();
        for (int64_t i = 0; i < N; i++) {
            int64_t a = (i < actions.numel()) ? (int64_t)ap[i] : 0;
            if (a < 0) a = 0;
            if (a >= RV) a = RV - 1;
            int64_t r = (i < rows) ? i : rows - 1;
            const float* row = rp + r * RV;
            float m = -INFINITY;
            for (int64_t v = 0; v < RV; v++) if (row[v] > m) m = row[v];
            float se = 0.0f;
            for (int64_t v = 0; v < RV; v++) se += std::exp(row[v] - m);
            dst[i] = row[a] - m - std::log(se + 1e-10f);
        }
    } else {
        rlp_c.zero_();
    }

    // ---- Single enabled graph: one policy forward, one scalar loss ----
    auto& engine = AutogradEngine::instance();
    bool was_enabled = AutogradEngine::enabled();
    engine.clear();
    auto policy_params = rl_graph::collect_rl_policy_params(policy_);
    auto critic_params = critic_parameters();
    for (auto* p : policy_params) { p->requires_grad(true); engine.register_parameter(p); }
    for (auto* p : critic_params) { p->requires_grad(true); engine.register_parameter(p); }
    if (optimizer_) {
        for (auto* p : policy_params) optimizer_->add_param(p);
        for (auto* p : critic_params) optimizer_->add_param(p);
        optimizer_->zero_grad();
    } else {
        for (auto* p : policy_params) if (p->has_grad()) p->zero_grad();
        for (auto* p : critic_params) if (p->has_grad()) p->zero_grad();
    }
    AutogradEngine::set_enabled(true);

    // The only forward in this step; every term below roots here.
    Tensor logits_ag = policy_->forward(states, positions, nullptr); // {B,S,V}
    // Clipped surrogate on per-token logprobs from the graph gather op.
    Tensor lpi = rl_graph::rl_gather_op(logits_ag, actions);         // {B,S}
    Tensor ratio = rl_graph::rl_exp_op(rl_graph::rl_sub_op(lpi, olp_c));
    Tensor clipped = rl_graph::rl_clip_op(ratio, 1.0f - clip_epsilon_, 1.0f + clip_epsilon_);
    Tensor surr = rl_graph::rl_min_op(AutogradEngine::mul_op(ratio, adv_c),
                                      AutogradEngine::mul_op(clipped, adv_c));
    Tensor policy_loss_t = rl_graph::rl_sub_op(rl_graph::rl_const(Shape{1}, 0.0f),
                                               rl_graph::rl_mean_all_op(surr));
    // Value MSE on the rebuilt critic (pooled hidden {B,H} -> values {B}).
    Tensor pooled = rl_graph::rl_mean_pool_s_op(logits_ag, HS);      // {B,H}
    Tensor values_t = critic_forward(pooled);                        // {B}
    Tensor vdiff = rl_graph::rl_sub_op(values_t, ret_c);
    Tensor value_raw_t = AutogradEngine::mul_op(
        rl_graph::rl_mean_all_op(AutogradEngine::mul_op(vdiff, vdiff)),
        rl_graph::rl_const(Shape{1}, 0.5f));
    Tensor value_term_t = AutogradEngine::mul_op(
        value_raw_t, rl_graph::rl_const(Shape{1}, value_coef_));
    // Exact token entropy, in-graph.
    Tensor probs = rl_graph::rl_softmax_rows_op(logits_ag);          // {B,S,V}
    Tensor ent_rows = rl_graph::rl_sum_last_op(
        AutogradEngine::mul_op(probs, rl_graph::rl_log_op(probs)));  // {B,S}
    Tensor entropy_t = rl_graph::rl_sub_op(rl_graph::rl_const(Shape{1}, 0.0f),
                                           rl_graph::rl_mean_all_op(ent_rows));
    Tensor ent_term_t = AutogradEngine::mul_op(
        entropy_t, rl_graph::rl_const(Shape{1}, entropy_coef_));
    // Schulman k3 KL(pi || ref): exp(d) - d - 1 with d = ref - pi, in-graph.
    Tensor kd = rl_graph::rl_sub_op(rlp_c, lpi);
    Tensor kl_t = rl_graph::rl_mean_all_op(
        rl_graph::rl_sub_op(rl_graph::rl_sub_op(rl_graph::rl_exp_op(kd), kd),
                            rl_graph::rl_const(Shape{B, S}, 1.0f)));

    Tensor total = AutogradEngine::add_op(policy_loss_t, value_term_t);
    total = rl_graph::rl_sub_op(total, ent_term_t);
    float kl_scalar = kl_t.data<float>()[0];
    // Adaptive KL weight (dead kl_scale gate dropped): above 2x target the
    // weight freezes, above 1.5x it grows, below 0.5x it decays.
    if (kl_scalar <= kl_target_ * 2.0f) {
        if (kl_scalar > kl_target_ * 1.5f) kl_alpha_ += 0.002f;
        else if (kl_scalar < kl_target_ * 0.5f) kl_alpha_ = std::max(0.0f, kl_alpha_ - 0.001f);
    }
    last_kl_ = kl_scalar;
    total = AutogradEngine::add_op(
        total, AutogradEngine::mul_op(kl_t, rl_graph::rl_const(Shape{1}, kl_alpha_)));

    float pl_v = policy_loss_t.data<float>()[0];
    float vl_v = value_raw_t.data<float>()[0];
    float en_v = entropy_t.data<float>()[0];
    if (log_cb_) log_cb_(pl_v, vl_v, en_v, kl_scalar);

    // One backward, one clip, one step. No magnitude-rescale SFT path and no
    // manual SGD / hand-derived critic gradients anymore: without an optimizer
    // the step is skipped and the populated grads are left for inspection.
    engine.backward(total);
    AutogradEngine::set_enabled(was_enabled);
    if (optimizer_) {
        if (optimizer_->get_grad_clip_norm() <= 0.0f) optimizer_->set_grad_clip_norm(1.0f);
        optimizer_->clip_grad_norm();
        optimizer_->step();
        optimizer_->zero_grad();
    }
    engine.clear();
}

std::vector<Tensor*> PPOTrainer::critic_parameters() {
    return {&v_fc1_weight_, &v_fc1_bias_, &v_fc2_weight_, &v_fc2_bias_};
}

DPOTrainer::DPOTrainer(Model* policy, Model* ref_model, float beta,
                       Optimizer* optimizer)
    : policy_(policy), ref_model_(ref_model), beta_(beta), optimizer_(optimizer) {}

float DPOTrainer::compute_log_probs(const Tensor& logits, const Tensor& ids, Tensor* out_probs) {
    int64_t B = ids.dim(0);
    int64_t S = ids.dim(1);
    int64_t V = logits.dim(logits.rank() - 1);
    const float* lp = logits.data<float>();
    const float* id = ids.data<float>();

    if (out_probs) *out_probs = Tensor(Shape{B});
    float* op = out_probs ? out_probs->data<float>() : nullptr;

    float total_logprob = 0.0f;
    for (int64_t i = 0; i < B; i++) {
        float seq_logprob = 0.0f;
        for (int64_t j = 0; j < S; j++) {
            int64_t idx = i * S + j;
            int64_t target = (int64_t)id[idx];
            if (target < 0) target = 0;
            if (target >= V) target = V - 1;
            const float* row = lp + idx * V;
            float max_l = -INFINITY;
            for (int64_t v = 0; v < V; v++)
                if (row[v] > max_l) max_l = row[v];
            float sum_exp = 0.0f;
            for (int64_t v = 0; v < V; v++)
                sum_exp += std::exp(row[v] - max_l);
            float log_prob = row[target] - max_l - std::log(sum_exp + 1e-10f);
            seq_logprob += log_prob;
        }
        if (op) op[i] = seq_logprob;
        total_logprob += seq_logprob;
    }
    return total_logprob / (float)B;
}

float DPOTrainer::train_step(const Tensor& chosen_logits, const Tensor& rejected_logits,
                              const Tensor& chosen_ids, const Tensor& rejected_ids) {
    // Single-graph DPO (D5 W2/W3): both branches forward under one enabled
    // graph, ref logprobs are constants, and the loss is the exact Bradley-
    // Terry objective  L = -mean(log(sigmoid(margin)))  with
    // margin = beta * ((lpi_c - lref_c) - (lpi_r - lref_r)).
    // The frozen signature still receives precomputed logits (computed outside
    // any graph by the caller); the gradient path re-roots at fresh enabled
    // policy forwards below, so those inputs are intentionally unused.
    // NOTE (bug census): the (void) discards below are DOCUMENTED-unused, not
    // a stub — the params exist for signature stability with older callers.
    // Passing non-empty logits does not change the result (fresh forwards win).
    (void)chosen_logits;   // documented-unused: fresh policy forward below
    (void)rejected_logits; // documented-unused: fresh policy forward below
    int64_t B = chosen_ids.dim(0);
    int64_t S = chosen_ids.dim(1);
    if (B <= 0 || S <= 0) return 0.0f;

    Tensor positions(Shape{B, S});
    float* ps0 = positions.data<float>();
    for (int64_t i = 0; i < B * S; ++i) ps0[i] = (float)(i % S);

    // Reference per-sequence logprobs as constants (disabled forwards).
    Tensor lref_c(Shape{B}), lref_r(Shape{B});
    {
        bool ref_was = AutogradEngine::enabled();
        AutogradEngine::set_enabled(false);
        if (ref_model_) {
            Tensor ref_chosen = ref_model_->forward(chosen_ids, positions, nullptr);
            Tensor ref_rejected = ref_model_->forward(rejected_ids, positions, nullptr);
            compute_log_probs(ref_chosen, chosen_ids, &lref_c);
            compute_log_probs(ref_rejected, rejected_ids, &lref_r);
        } else {
            lref_c.zero_();
            lref_r.zero_();
        }
        AutogradEngine::set_enabled(ref_was);
    }

    auto& engine = AutogradEngine::instance();
    bool was_enabled = AutogradEngine::enabled();
    engine.clear();
    auto policy_params = rl_graph::collect_rl_policy_params(policy_);
    for (auto* p : policy_params) { p->requires_grad(true); engine.register_parameter(p); }
    if (optimizer_) {
        for (auto* p : policy_params) optimizer_->add_param(p);
        optimizer_->zero_grad();
    } else {
        for (auto* p : policy_params) if (p->has_grad()) p->zero_grad();
    }
    AutogradEngine::set_enabled(true);

    // Graph chosen/rejected forwards: both branches, one graph, one backward.
    Tensor cl = policy_->forward(chosen_ids, positions, nullptr);   // {B,S,V}
    Tensor rl = policy_->forward(rejected_ids, positions, nullptr); // {B,S,V}
    Tensor lpi_c = rl_graph::rl_sum_last_op(rl_graph::rl_gather_op(cl, chosen_ids));
    Tensor lpi_r = rl_graph::rl_sum_last_op(rl_graph::rl_gather_op(rl, rejected_ids));
    Tensor margin = AutogradEngine::mul_op(
        rl_graph::rl_sub_op(rl_graph::rl_sub_op(lpi_c, lref_c),
                            rl_graph::rl_sub_op(lpi_r, lref_r)),
        rl_graph::rl_const(Shape{B}, beta_));
    Tensor loss_t = rl_graph::rl_mean_all_op(
        rl_graph::rl_sub_op(rl_graph::rl_const(Shape{B}, 0.0f),
                            rl_graph::rl_log_op(rl_graph::rl_sigmoid_op(margin))));

    float loss_val = loss_t.data<float>()[0];
    float kl_val = 0.0f;
    {
        const float* pc = lpi_c.data<float>();
        const float* pr = lpi_r.data<float>();
        const float* rc = lref_c.data<float>();
        const float* rr = lref_r.data<float>();
        for (int64_t i = 0; i < B; i++)
            kl_val += (rc[i] - pc[i]) + (pr[i] - rr[i]);
        kl_val /= (float)(B * 2);
    }
    last_loss_ = loss_val;
    if (log_cb_) log_cb_(loss_val, kl_val);

    // Single backward + clip + step. This is also the sign-bug fix: the old
    // path backpropagated CE_c + CE_r, whose gradient raises BOTH chosen and
    // rejected likelihoods; here dL/dlpi_c = -beta*sigmoid(-margin) < 0
    // (chosen up) and dL/dlpi_r = +beta*sigmoid(-margin) > 0 (rejected down).
    engine.backward(loss_t);
    AutogradEngine::set_enabled(was_enabled);
    if (optimizer_) {
        if (optimizer_->get_grad_clip_norm() <= 0.0f) optimizer_->set_grad_clip_norm(1.0f);
        optimizer_->clip_grad_norm();
        optimizer_->step();
        optimizer_->zero_grad();
    }
    engine.clear();

    return loss_val;
}

RLHFPipeline::RLHFPipeline(Model* model, Model* ref_model, Tokenizer* tokenizer,
                           RewardModel* reward_model, Trainer* trainer,
                           Optimizer* policy_opt, Optimizer* rm_opt)
    : model_(model), ref_model_(ref_model), tokenizer_(tokenizer),
      reward_model_(reward_model), trainer_(trainer),
      policy_opt_(policy_opt), rm_opt_(rm_opt) {}

Tensor RLHFPipeline::extract_hidden(Tensor& logits, int64_t hidden_size) {
    int64_t B = logits.dim(0);
    Tensor pooled(Shape{B, hidden_size});
    float* pd = pooled.data<float>();
    const float* ld = logits.data<float>();
    int64_t total = logits.numel();
    int64_t feats = total / B;
    int64_t pool_dim = std::min(feats, hidden_size);
    for (int64_t i = 0; i < B; i++) {
        for (int64_t j = 0; j < pool_dim; j++)
            pd[i * hidden_size + j] = ld[i * feats + (feats - pool_dim) + j];
        for (int64_t j = pool_dim; j < hidden_size; j++)
            pd[i * hidden_size + j] = 0.0f;
    }
    return pooled;
}

Tensor RLHFPipeline::get_reward_for_sequence(Model* model, const std::vector<int>& ids) {
    int64_t S = (int64_t)ids.size();
    Tensor input_ids(Shape{1, S});
    Tensor positions(Shape{1, S});
    float* idp = input_ids.data<float>();
    float* psp = positions.data<float>();
    for (int64_t i = 0; i < S; i++) {
        idp[i] = (float)ids[i];
        psp[i] = (float)i;
    }
    Tensor logits = model->forward(input_ids, positions, nullptr);
    int64_t hidden_size = model->config.hidden_size;
    if (hidden_size <= 0) hidden_size = 64;
    Tensor feat = extract_hidden(logits, hidden_size);
    return reward_model_->score(feat);
}

void RLHFPipeline::generate_comparisons(const std::vector<std::string>& prompts, int max_new_tokens) {
    int vocab_size = (int)model_->config.vocab_size;

    for (size_t p = 0; p < prompts.size(); p++) {
        auto tokens = tokenizer_ ? tokenizer_->encode(prompts[p]) : std::vector<int>();
        if (tokens.empty()) {
            int offset = 5;
            int mod = std::max(1, vocab_size - offset);
            for (char c : prompts[p])
                tokens.push_back((int)(unsigned char)c % mod + offset);
        }

        std::vector<int> all_ids = tokens;
        int context = std::min((int)model_->config.max_seq_len, 512);

        for (int step = 0; step < max_new_tokens; step++) {
            int64_t len = (int64_t)all_ids.size();
            int64_t start = std::max((int64_t)0, len - context);
            int64_t ctx_len = len - start;

            Tensor input_ids(Shape{1, ctx_len});
            Tensor positions(Shape{1, ctx_len});
            float* idp = input_ids.data<float>();
            float* psp = positions.data<float>();
            for (int64_t i = 0; i < ctx_len; i++) {
                idp[i] = (float)all_ids[start + i];
                psp[i] = (float)(start + i);
            }
            Tensor logits = model_->forward(input_ids, positions, nullptr);
            int64_t V = logits.dim(logits.rank() - 1);
            const float* lp = logits.data<float>();
            const float* last_row = lp + (ctx_len - 1) * V;
            int next = 0;
            float max_l = -INFINITY;
            for (int64_t v = 0; v < std::min(V, (int64_t)vocab_size); v++) {
                if (last_row[v] > max_l) { max_l = last_row[v]; next = (int)v; }
            }
            all_ids.push_back(next);
            if (next < 2) break;
        }

        std::vector<int> generation(all_ids.begin() + (int64_t)tokens.size(), all_ids.end());

        Comparison comp;
        comp.prompt = prompts[p];

        // Use full sequence for both chosen and rejected, split at generation boundary
        comp.chosen_ids = all_ids;
        comp.rejected_ids = all_ids;

        Tensor full_rew = get_reward_for_sequence(model_, comp.chosen_ids);
        comp.reward_chosen = full_rew.data<float>()[0];
        comp.reward_rejected = full_rew.data<float>()[0] - 0.01f;

        if (comp.reward_chosen <= comp.reward_rejected) {
            std::swap(comp.reward_chosen, comp.reward_rejected);
        }

        comparison_buffer_.push_back(comp);

        if (verbose_) {
            std::cout << "[RLHF] Comparison " << (p + 1) << "/" << prompts.size()
                      << " | reward_chosen=" << comp.reward_chosen
                      << " reward_rejected=" << comp.reward_rejected << std::endl;
        }
    }
}
GRPOTrainer::GRPOTrainer(Model* model, Tokenizer* tok, int group_size, float beta)
    : model_(model), tok_(tok), group_size_(group_size), beta_(beta), optimizer_(nullptr) {}

Tensor GRPOTrainer::compute_log_probs(const Tensor& logits, const Tensor& ids) {
    int64_t B = ids.dim(0);
    int64_t S = ids.dim(1);
    int64_t V = logits.dim(logits.rank() - 1);
    const float* lp = logits.data<float>();
    const float* id = ids.data<float>();
    Tensor logprobs(Shape{B, S});
    float* out = logprobs.data<float>();
    for (int64_t i = 0; i < B; i++) {
        for (int64_t j = 0; j < S; j++) {
            int64_t idx = i * S + j;
            int64_t target = (int64_t)id[idx];
            if (target < 0) target = 0;
            if (target >= V) target = V - 1;
            const float* row = lp + idx * V;
            float max_l = -INFINITY;
            for (int64_t v = 0; v < V; v++) if (row[v] > max_l) max_l = row[v];
            float sum_exp = 0.0f;
            for (int64_t v = 0; v < V; v++) sum_exp += std::exp(row[v] - max_l);
            out[idx] = row[target] - max_l - std::log(sum_exp + 1e-10f);
        }
    }
    return logprobs;
}

static std::vector<Tensor*> collect_grpo_params(Model* m) {
    std::vector<Tensor*> p;
    if (!m) return p;
    if (auto* dm = dynamic_cast<DenseModel*>(m)) {
        collect_dense_params(dm, p);
        return p;
    }
    if (auto* mm = dynamic_cast<MoEModel*>(m)) {
        // Mirror MoETrainer::collect_params(): dense trunk + router + every
        // expert + shared expert, so GRPO gradients reach the MoE stack
        // (required for RLL on large-MoE-class models).
        p.push_back(&mm->tok_embeddings->weight);
        for (auto& l : mm->layers) {
            p.push_back(&l.attention_norm.weight);
            p.push_back(&l.attention.q_proj.weight);
            if (l.attention.q_proj.bias.numel() > 0) p.push_back(&l.attention.q_proj.bias);
            p.push_back(&l.attention.k_proj.weight);
            if (l.attention.k_proj.bias.numel() > 0) p.push_back(&l.attention.k_proj.bias);
            p.push_back(&l.attention.v_proj.weight);
            if (l.attention.v_proj.bias.numel() > 0) p.push_back(&l.attention.v_proj.bias);
            p.push_back(&l.attention.o_proj.weight);
            if (l.attention.o_proj.bias.numel() > 0) p.push_back(&l.attention.o_proj.bias);
            p.push_back(&l.ffn_norm.weight);
            p.push_back(&l.moe->router_weight.weight);
            for (auto& e : l.moe->experts) {
                p.push_back(&e.gate_proj.weight);
                p.push_back(&e.up_proj.weight);
                p.push_back(&e.down_proj.weight);
            }
            if (l.shared_expert) {
                p.push_back(&l.shared_expert->gate_proj.weight);
                p.push_back(&l.shared_expert->up_proj.weight);
                p.push_back(&l.shared_expert->down_proj.weight);
            }
        }
        p.push_back(&mm->norm->weight);
        p.push_back(&mm->lm_head->weight);
        if (mm->lm_head->bias.numel() > 0) p.push_back(&mm->lm_head->bias);
        return p;
    }
    return p;
}

float GRPOTrainer::train_step(const Tensor& input_ids, const Tensor& labels, const Tensor& rewards) {
    if (!model_) return 0.0f;
    int64_t G = rewards.numel();
    if (G == 0) return 0.0f;
    const float* r = rewards.data<float>();
    float mean = 0.0f;
    for (int64_t i = 0; i < G; i++) mean += r[i];
    mean /= (float)G;
    float var = 0.0f;
    for (int64_t i = 0; i < G; i++) var += (r[i] - mean) * (r[i] - mean);
    float stdv = std::sqrt(var / (float)G + 1e-8f);
    std::vector<float> adv(G);
    for (int64_t i = 0; i < G; i++) adv[i] = (r[i] - mean) / stdv;

    int64_t S = labels.dim(1);
    Tensor positions(Shape{G, S});
    float* ps = positions.data<float>();
    for (int64_t i = 0; i < G * S; i++) ps[i] = (float)(i % S);

    auto params = collect_grpo_params(model_);
    for (auto* p : params) { p->requires_grad(true); AutogradEngine::instance().register_parameter(p); }
    if (optimizer_) {
        // ensure optimizer knows params
        for (auto* p : params) optimizer_->add_param(p);
        optimizer_->zero_grad();
    }
    AutogradEngine::instance().clear();
    AutogradEngine::set_enabled(true);

    Tensor logits = model_->forward(input_ids, positions, nullptr);

    // Proper GRPO loss: each group sample's sequence cross-entropy is weighted
    // by ITS OWN group-relative advantage inside the graph, then summed.
    //   loss = (1/G) * Σ_i  adv_i * CE(logits_i, labels_i)
    // The previous implementation backpropagated one mean-CE and rescaled all
    // parameter gradients by mean(adv) — which is ≈0 after normalization — so
    // updates collapsed into a sign-hacked SFT. This per-sample graph fixes it.
    Tensor loss_acc(Shape{1});
    loss_acc.zero_();
    for (int64_t i = 0; i < G; ++i) {
        Tensor logits_i = logits.slice(0, i, i + 1);   // {1,S,V}, contiguous row block
        Tensor labels_i = labels.slice(0, i, i + 1);   // {1,S}
        Tensor ce_i = AutogradEngine::cross_entropy_op(logits_i, labels_i);
        Tensor adv_t(Shape{1});
        adv_t.data<float>()[0] = adv[(size_t)i];
        loss_acc = AutogradEngine::add_op(loss_acc, AutogradEngine::mul_op(ce_i, adv_t));
    }
    Tensor inv_g(Shape{1});
    inv_g.data<float>()[0] = 1.0f / (float)G;
    Tensor loss_tensor = AutogradEngine::mul_op(loss_acc, inv_g);

    AutogradEngine::instance().backward(loss_tensor);
    AutogradEngine::set_enabled(false);
    const float base_loss = loss_tensor.data<float>()[0];

    // beta_ acts as trust-region damping on the update magnitude. It is NOT a
    // KL-vs-reference term; the reference-KL variant lives in PPOTrainer.
    if (beta_ > 0) {
        const float damp = std::max(0.0f, 1.0f - beta_ * 0.01f);
        for (auto* p : params) if (p->has_grad()) {
            float* gd = p->grad().data<float>();
            int64_t n = p->grad().numel();
            for (int64_t i = 0; i < n; i++) gd[i] *= damp;
        }
    }
    if (optimizer_) {
        optimizer_->step();
        optimizer_->zero_grad();
    }
    AutogradEngine::instance().clear();
    last_loss_ = base_loss;
    return base_loss;
}

float GRPOTrainer::train_step(const std::string& prompt) {
    if (!model_) return 0.0f;
    int vocab_size = (int)model_->config.vocab_size;
    int max_seq_len = (int)model_->config.max_seq_len;
    int context_len = std::min(max_seq_len > 0 ? max_seq_len : 512, 256);
    std::vector<int> prompt_tokens;
    if (tok_) prompt_tokens = tok_->encode(prompt);
    else for (char c : prompt) prompt_tokens.push_back((int)(unsigned char)c % std::max(1, vocab_size));
    if (prompt_tokens.empty()) prompt_tokens.push_back(1);
    // Clamp to vocab range: a fresh tokenizer's ids can exceed a tiny test
    // model's vocab_size, and an out-of-range embedding gather poisons the
    // whole forward with NaN.
    for (auto& t : prompt_tokens) {
        t = ((t % std::max(1, vocab_size)) + std::max(1, vocab_size)) % std::max(1, vocab_size);
    }

    bool prev = AutogradEngine::enabled();
    AutogradEngine::set_enabled(false);
    std::vector<std::vector<int>> completions(group_size_);
    std::vector<float> rewards(group_size_, 0.0f);
    for (int g = 0; g < group_size_; g++) {
        completions[g] = prompt_tokens;
        std::mt19937 rng(42 + g * 101);
        // Cap generated length so total sequence stays inside max_seq_len:
        // positions beyond the RoPE/KV-cache range poison the forward with NaN.
        int gen_steps = 32;
        if (model_->config.max_seq_len > 0) {
            const int room =
                (int)model_->config.max_seq_len - (int)prompt_tokens.size();
            gen_steps = std::max(0, std::min(gen_steps, room));
        }
        for (int step = 0; step < gen_steps; step++) {
            int64_t len = (int64_t)completions[g].size();
            int64_t start = std::max((int64_t)0, len - context_len);
            int64_t ctx_len = len - start;
            Tensor input_ids(Shape{1, ctx_len});
            Tensor positions(Shape{1, ctx_len});
            float* idp = input_ids.data<float>();
            float* psp = positions.data<float>();
            for (int64_t i = 0; i < ctx_len; i++) { idp[i] = (float)completions[g][start + i]; psp[i] = (float)(start + i); }
            Tensor logits = model_->forward(input_ids, positions, nullptr);
            int64_t V = logits.dim(logits.rank() - 1);
            const float* lp = logits.data<float>() + (ctx_len - 1) * V;
            float max_l = -INFINITY;
            for (int64_t v = 0; v < std::min(V, (int64_t)vocab_size); v++) if (lp[v] > max_l) max_l = lp[v];
            float sum_exp = 0.0f;
            for (int64_t v = 0; v < std::min(V, (int64_t)vocab_size); v++) sum_exp += std::exp(lp[v] - max_l);
            float rnd = ((float)rng() / (float)rng.max()) * sum_exp;
            float cum = 0.0f;
            int next_tok = 0;
            for (int64_t v = 0; v < std::min(V, (int64_t)vocab_size); v++) { cum += std::exp(lp[v] - max_l); if (cum >= rnd) { next_tok = (int)v; break; } }
            completions[g].push_back(next_tok);
        }
        float len_reward = std::min(1.0f, (float)completions[g].size() / 40.0f);
        std::set<int> uniq(completions[g].begin(), completions[g].end());
        float div = (float)uniq.size() / (float)completions[g].size();
        rewards[g] = len_reward * 0.5f + div * 0.5f;
    }
    AutogradEngine::set_enabled(prev);
    int64_t max_len = 0;
    for (auto& c : completions) max_len = std::max(max_len, (int64_t)c.size());
    if (max_len == 0) return 0.0f;
    Tensor input_ids(Shape{(int64_t)group_size_, max_len});
    Tensor labels(Shape{(int64_t)group_size_, max_len});
    input_ids.zero_(); labels.zero_();
    float* ip = input_ids.data<float>(); float* lb = labels.data<float>();
    for (int g = 0; g < group_size_; g++) {
        for (int64_t j = 0; j < max_len; j++) {
            int tok = j < (int64_t)completions[g].size() ? completions[g][j] : 0;
            ip[g * max_len + j] = (float)tok;
            lb[g * max_len + j] = (float)tok;
        }
    }
    Tensor rew(Shape{(int64_t)group_size_});
    for (int g = 0; g < group_size_; g++) rew.data<float>()[g] = rewards[g];
    return train_step(input_ids, labels, rew);
}

RLVRTrainer::RLVRTrainer(Model* model, Tokenizer* tok, Optimizer* opt)
    : model_(model), tok_(tok), optimizer_(opt) {}

float RLVRTrainer::train_step(const Tensor& input_ids, const Tensor& labels, float reward) {
    if (!model_) return 0.0f;
    last_reward_ = reward;
    int64_t B = input_ids.dim(0);
    int64_t S = input_ids.dim(1);
    Tensor positions(Shape{B, S});
    float* ps = positions.data<float>();
    for (int64_t i = 0; i < B * S; i++) ps[i] = (float)(i % S);
    auto params = collect_grpo_params(model_);
    for (auto* p : params) { p->requires_grad(true); AutogradEngine::instance().register_parameter(p); }
    if (optimizer_) { for (auto* p : params) optimizer_->add_param(p); optimizer_->zero_grad(); }
    AutogradEngine::instance().clear();
    AutogradEngine::set_enabled(true);
    Tensor logits = model_->forward(input_ids, positions, nullptr);
    Tensor loss_tensor = AutogradEngine::cross_entropy_op(logits, labels);
    float base = loss_tensor.data<float>()[0];
    AutogradEngine::instance().backward(loss_tensor);
    AutogradEngine::set_enabled(false);
    // Scale grad by reward (policy gradient) — positive reward reduces loss gradient, negative flips
    float scale = reward;
    if (std::abs(scale) < 1e-6f) scale = 0.0f;
    for (auto* p : params) if (p->has_grad()) {
        float* gd = p->grad().data<float>();
        int64_t n = p->grad().numel();
        for (int64_t i = 0; i < n; i++) gd[i] *= -scale;
    }
    if (optimizer_ && scale != 0.0f) { optimizer_->step(); optimizer_->zero_grad(); }
    AutogradEngine::instance().clear();
    return reward;
}

float RLVRTrainer::train_step(const std::string& prompt, const std::string& verifiable_answer) {
    if (!model_) return 0.0f;
    int vocab_size = (int)model_->config.vocab_size;
    int context_len = std::min((int)model_->config.max_seq_len > 0 ? (int)model_->config.max_seq_len : 512, 256);
    std::vector<int> prompt_tokens;
    if (tok_) prompt_tokens = tok_->encode(prompt);
    else for (char c : prompt) prompt_tokens.push_back((int)(unsigned char)c % std::max(1, vocab_size));
    if (prompt_tokens.empty()) prompt_tokens.push_back(1);
    bool prev = AutogradEngine::enabled();
    AutogradEngine::set_enabled(false);
    std::vector<int> full_seq = prompt_tokens;
    for (int step = 0; step < 32; step++) {
        int64_t len = (int64_t)full_seq.size();
        int64_t start = std::max((int64_t)0, len - context_len);
        int64_t ctx_len = len - start;
        Tensor input_ids(Shape{1, ctx_len});
        Tensor positions(Shape{1, ctx_len});
        float* idp = input_ids.data<float>(); float* psp = positions.data<float>();
        for (int64_t i = 0; i < ctx_len; i++) { idp[i] = (float)full_seq[start + i]; psp[i] = (float)(start + i); }
        Tensor logits = model_->forward(input_ids, positions, nullptr);
        int64_t V = logits.dim(logits.rank() - 1);
        const float* lp = logits.data<float>() + (ctx_len - 1) * V;
        int best = 0; float mx = -INFINITY;
        for (int64_t v = 0; v < std::min(V, (int64_t)vocab_size); v++) if (lp[v] > mx) { mx = lp[v]; best = (int)v; }
        full_seq.push_back(best);
    }
    AutogradEngine::set_enabled(prev);
    std::string gen;
    if (tok_) gen = tok_->decode(full_seq);
    else for (int t : full_seq) gen += (char)(t % 128);
    bool exact = verify(gen, verifiable_answer);
    float reward = exact ? 1.0f : -0.5f;
    if (!exact && !verifiable_answer.empty()) {
        size_t ml = 0;
        for (size_t i = 0; i < std::min(gen.size(), verifiable_answer.size()); i++) if (gen[i] == verifiable_answer[i]) ml++;
        reward += 0.2f * ((float)ml / (float)verifiable_answer.size());
    }
    int64_t S = (int64_t)full_seq.size();
    Tensor input_ids(Shape{1, S});
    Tensor labels(Shape{1, S});
    for (int64_t i = 0; i < S; i++) { input_ids.data<float>()[i] = (float)full_seq[i]; labels.data<float>()[i] = (float)full_seq[i]; }
    return train_step(input_ids, labels, reward);
}

bool RLVRTrainer::verify(const std::string& output, const std::string& answer) {
    if (answer.empty()) return false;
    return output.find(answer) != std::string::npos;
}

} // namespace quant
