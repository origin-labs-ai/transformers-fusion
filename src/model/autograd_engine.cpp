#include "quant/autograd.h"
#include "quant/autograd_functions.h"
#include "quant/math.h"
#include "quant/kernel.h"
#include <unordered_set>
#include <atomic>

namespace quant {

std::atomic<bool> AutogradEngine::enabled_{false};
std::atomic<bool> AutogradEngine::registry_alive_{true};

AutogradEngine::~AutogradEngine() {
    // Order matters: drop the hook and flip the flag BEFORE members are
    // destroyed so any ~Tensor running later (static teardown) early-returns
    // instead of touching the dying param_map_.
    detail::autograd_unregister_hook = nullptr;
    registry_alive_.store(false, std::memory_order_relaxed);
}

void AutogradEngine::unregister_parameter(Tensor* p) {
    if (!p || !registry_alive()) return;
    try {
        std::lock_guard<std::mutex> lock(param_mutex_);
        auto it = param_map_.find(p->data());
        if (it != param_map_.end() && it->second == p) param_map_.erase(it);
    } catch (...) {
        // noexcept-safe: never let registry cleanup throw from ~Tensor.
    }
}

bool AutogradEngine::registry_alive() {
    return registry_alive_.load(std::memory_order_relaxed);
}

// ========================================================================
// AutogradEngine operation helpers
// ========================================================================

Tensor AutogradEngine::matmul_op(const Tensor& a, const Tensor& b, int64_t M, int64_t N, int64_t K) {
    if (!enabled_) {
        Tensor out({M, N}, DType::F32);
        kernel::scalar_gemm_bt(
            a.data<float>(), b.data<float>(),
            out.data<float>(), (int)M, (int)N, (int)K
        );
        return out;
    }
    auto fn = std::make_shared<MatMulFunction>();
    auto outputs = fn->forward({a, b});
    Tensor& out = outputs[0];
    out.requires_grad(true);

    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {a, b};
    node->outputs = {out};
    instance().register_node(node);

    return out;
}

Tensor AutogradEngine::add_op(const Tensor& a, const Tensor& b) {
    if (!enabled_) {
        Tensor out(a.shape(), DType::F32);
        math::add(a, b, out);
        return out;
    }
    auto fn = std::make_shared<AddFunction>();
    auto outputs = fn->forward({a, b});
    Tensor& out = outputs[0];
    out.requires_grad(true);

    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {a, b};
    node->outputs = {out};
    instance().register_node(node);

    return out;
}

Tensor AutogradEngine::silu_op(const Tensor& x) {
    if (!enabled_) {
        Tensor out(x.shape(), DType::F32);
        math::silu(x, out);
        return out;
    }
    auto fn = std::make_shared<SiLUFunction>();
    auto outputs = fn->forward({x});
    Tensor& out = outputs[0];
    out.requires_grad(true);

    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {x};
    node->outputs = {out};
    instance().register_node(node);

    return out;
}

Tensor AutogradEngine::mul_op(const Tensor& a, const Tensor& b) {
    if (!enabled_) {
        Tensor out(a.shape(), DType::F32);
        math::mul(a, b, out);
        return out;
    }
    auto fn = std::make_shared<MulFunction>();
    auto outputs = fn->forward({a, b});
    Tensor& out = outputs[0];
    out.requires_grad(true);

    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {a, b};
    node->outputs = {out};
    instance().register_node(node);

    return out;
}

Tensor AutogradEngine::cross_entropy_op(const Tensor& logits, const Tensor& labels) {
    auto fn = std::make_shared<CrossEntropyFunction>();
    auto outputs = fn->forward({logits, labels});
    Tensor& loss = outputs[0];

    if (enabled_) {
        auto node = std::make_shared<AutogradNode>();
        node->fn = fn;
        node->inputs = {logits, labels};
        node->outputs = {loss};
        loss.requires_grad(true);
        instance().register_node(node);
    }

    return loss;
}

Tensor AutogradEngine::rms_norm_op(const Tensor& x, const Tensor& gamma, float eps) {
    if (!enabled_) {
        Tensor out(x.shape(), DType::F32);
        math::rms_norm(x, gamma, eps, out);
        return out;
    }
    auto fn = std::make_shared<RMSNormFunction>();
    fn->saved_eps = eps;
    auto outputs = fn->forward({x, gamma});
    Tensor& out = outputs[0];
    out.requires_grad(true);

    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {x, gamma};
    node->outputs = {out};
    instance().register_node(node);

    return out;
}

Tensor AutogradEngine::rotary_op(const Tensor& x, const Tensor& cos_cached,
                                  const Tensor& sin_cached,
                                  int64_t seq_start, int64_t seq_len) {
    if (!enabled_) {
        int64_t B = x.dim(0), H = x.dim(1), S = x.dim(2), D = x.dim(3);
        int64_t half = D / 2;
        Tensor out(x.shape(), DType::F32);
        const float* xd = x.data<float>();
        float* od = out.data<float>();
        const float* cos_d = cos_cached.data<float>();
        const float* sin_d = sin_cached.data<float>();
        for (int64_t b = 0; b < B; b++)
            for (int64_t h = 0; h < H; h++)
                for (int64_t s = 0; s < S; s++) {
                    int64_t base = ((b * H + h) * S + s) * D;
                    int64_t pos = seq_start + s;
                    for (int64_t d = 0; d < half; d++) {
                        float x1 = xd[base + d], x2 = xd[base + d + half];
                        float cos_v = cos_d[pos * half + d];
                        float sin_v = sin_d[pos * half + d];
                        od[base + d] = x1 * cos_v - x2 * sin_v;
                        od[base + d + half] = x1 * sin_v + x2 * cos_v;
                    }
                }
        return out;
    }
    auto fn = std::make_shared<RotaryFunction>(x.dim(3), cos_cached, sin_cached, seq_start, seq_len);
    auto outputs = fn->forward({x});
    Tensor& out = outputs[0];
    out.requires_grad(true);
    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {x};
    node->outputs = {out};
    instance().register_node(node);
    return out;
}

Tensor AutogradEngine::attention_op(const Tensor& q, const Tensor& k, const Tensor& v,
                                     int64_t num_heads, int64_t num_kv_heads, int64_t head_dim) {
    if (!enabled_) {
        ScaledDotProductAttentionFunction fn(num_heads, num_kv_heads, head_dim);
        auto outputs = fn.forward({q, k, v});
        return outputs[0];
    }
    auto fn = std::make_shared<ScaledDotProductAttentionFunction>(num_heads, num_kv_heads, head_dim);
    auto outputs = fn->forward({q, k, v});
    Tensor& out = outputs[0];
    out.requires_grad(true);
    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {q, k, v};
    node->outputs = {out};
    instance().register_node(node);
    return out;
}

Tensor AutogradEngine::bias_add_op(const Tensor& x, const Tensor& bias) {
    if (!enabled_) {
        int64_t M = x.dim(0), N = x.dim(1);
        Tensor out(x.shape(), DType::F32);
        const float* xd = x.data<float>();
        const float* bd = bias.data<float>();
        float* od = out.data<float>();
        for (int64_t i = 0; i < M; i++)
            for (int64_t j = 0; j < N; j++)
                od[i * N + j] = xd[i * N + j] + bd[j];
        return out;
    }
    auto fn = std::make_shared<BiasAddFunction>();
    auto outputs = fn->forward({x, bias});
    Tensor& out = outputs[0];
    out.requires_grad(true);
    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {x, bias};
    node->outputs = {out};
    instance().register_node(node);
    return out;
}

Tensor AutogradEngine::flatten_attention_op(const Tensor& x, int64_t B, int64_t H, int64_t S, int64_t D) {
    if (!enabled_) {
        Tensor out(Shape{B * S, H * D}, DType::F32);
        const float* xd = x.data<float>();
        float* od = out.data<float>();
        for (int64_t b = 0; b < B; b++)
            for (int64_t h = 0; h < H; h++)
                for (int64_t s = 0; s < S; s++)
                    for (int64_t d = 0; d < D; d++)
                        od[(b * S + s) * (H * D) + h * D + d] =
                            xd[((b * H + h) * S + s) * D + d];
        return out;
    }
    auto fn = std::make_shared<FlattenAttentionFunction>(B, H, S, D);
    auto outputs = fn->forward({x});
    Tensor& out = outputs[0];
    out.requires_grad(true);
    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {x};
    node->outputs = {out};
    instance().register_node(node);
    return out;
}

Tensor AutogradEngine::transpose_op(const Tensor& x, int dim1, int dim2) {
    if (!enabled_) {
        return x.transpose(dim1, dim2);
    }
    auto fn = std::make_shared<TransposeFunction>(dim1, dim2);
    auto outputs = fn->forward({x});
    Tensor& out = outputs[0];
    out.requires_grad(true);
    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {x};
    node->outputs = {out};
    instance().register_node(node);
    return out;
}

Tensor AutogradEngine::embedding_op(const Tensor& input_ids, const Tensor& weight) {
    if (!enabled_) {
        int64_t N = input_ids.numel();
        int64_t D = weight.dim(1);
        int64_t V = weight.dim(0);
        Tensor out({N, D}, DType::F32);
        const float* id_d = input_ids.data<float>();
        const float* wd = weight.data<float>();
        float* od = out.data<float>();
        for (int64_t i = 0; i < N; i++) {
            int64_t token = (int64_t)id_d[i];
            if (token < 0 || token >= V) token = 0;
            std::memcpy(od + i * D, wd + token * D, D * sizeof(float));
        }
        return out;
    }
    auto fn = std::make_shared<EmbeddingFunction>();
    auto outputs = fn->forward({input_ids, weight});
    Tensor& out = outputs[0];
    out.requires_grad(true);
    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {input_ids, weight};
    node->outputs = {out};
    instance().register_node(node);
    return out;
}

// ========================================================================
// AutogradEngine
// ========================================================================

// Caller note: the registry is now SELF-CLEANING — ~Tensor unregisters any
// tensor flagged via autograd_registered_, so entries cannot outlive the
// owning model even when heap addresses are reused by new allocations.
void AutogradEngine::register_parameter(Tensor* p) {
    if (!p || !p->data()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    param_map_[p->data()] = p;
    p->autograd_registered_ = true;
}

void AutogradEngine::register_node(const std::shared_ptr<AutogradNode>& node) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (next_is_checkpoint_) {
        node->checkpoint = true;
        node->fn->saved.clear();
        next_is_checkpoint_ = false;
        last_checkpoint_ = node;
    }
    nodes_.push_back(node);
    for (auto& out : node->outputs) {
        output_to_node_[out.data()] = node;
    }
}

void AutogradEngine::set_checkpoint() {
    std::lock_guard<std::mutex> lock(instance().mutex_);
    instance().next_is_checkpoint_ = true;
}

bool AutogradEngine::is_checkpoint() {
    std::lock_guard<std::mutex> lock(instance().mutex_);
    return instance().next_is_checkpoint_;
}

long long g_guard_hit = 0;
long long g_guard_pass = 0;
long long g_guard_shape_mismatch = 0;

void AutogradEngine::backward(Tensor& loss) {
    if (!loss.has_grad() || loss.grad().numel() == 0) {
        Tensor g(loss.shape());
        g.fill(1.0f);
        loss.set_grad(g);
    }

    std::lock_guard<std::mutex> lock(mutex_);

    std::unordered_map<const void*, int> consumer_count;
    for (auto& node : nodes_) {
        std::unordered_set<const void*> seen;
        for (auto& inp : node->inputs) {
            if (inp.requires_grad() && seen.insert(inp.data()).second)
                consumer_count[inp.data()]++;
        }
    }

    consumer_count[loss.data()]++;

    std::unordered_map<const void*, int> pending(consumer_count.begin(), consumer_count.end());

    std::unordered_map<const void*, Tensor> accum;
    accum[loss.data()] = loss.grad();

    std::vector<const void*> stack;
    stack.push_back(loss.data());

    while (!stack.empty()) {
        const void* t = stack.back();
        auto pit = pending.find(t);
        if (pit == pending.end()) { stack.pop_back(); continue; }
        pit->second--;
        if (pit->second > 0) { stack.pop_back(); continue; }

        auto nit = output_to_node_.find(t);
        if (nit == output_to_node_.end() || nit->second.expired()) { stack.pop_back(); continue; }
        auto node = nit->second.lock();

        if (node->checkpoint && node->fn->saved.empty()) {
            node->fn->forward(node->inputs);
        }

        auto ait = accum.find(t);
        if (ait == accum.end()) { stack.pop_back(); continue; }
        auto grad_outputs = node->fn->backward({ ait->second });

        stack.pop_back();

        std::unordered_set<const void*> pushed;
        for (size_t i = 0; i < node->inputs.size() && i < grad_outputs.size(); ++i) {
            Tensor& inp = node->inputs[i];
            if (!inp.requires_grad()) continue;

            auto pmit = param_map_.find(inp.data());
            // Stale-entry guard: entries are keyed by the tensor's data()
            // address and survive model destruction (preserved across
            // clear()). If the address was reused by a different tensor, the
            // stored Tensor* aliases it; verify the pointer still refers to
            // its own data before treating it as the parameter.
            if (pmit != param_map_.end()) {
                g_guard_hit++;
                if (pmit->second->data() == inp.data()) {
                    g_guard_pass++;
                }
            }
            if (pmit != param_map_.end() && pmit->second->data() == inp.data()) {
                Tensor* orig = pmit->second;
                bool same_shape = orig->shape() == grad_outputs[i].shape();
                if (!same_shape) g_guard_shape_mismatch++;
                if (!orig->has_grad()) {
                    orig->set_grad(grad_outputs[i]);
                } else if (same_shape) {
                    Tensor acc(orig->shape());
                    math::add(orig->grad(), grad_outputs[i], acc);
                    orig->set_grad(acc);
                } else {
                    orig->set_grad(grad_outputs[i]);
                }
            }

            auto aait = accum.find(inp.data());
            if (aait == accum.end()) {
                accum[inp.data()] = grad_outputs[i];
            } else if (aait->second.numel() == grad_outputs[i].numel()) {
                Tensor tmp(grad_outputs[i].shape());
                math::add(aait->second, grad_outputs[i], tmp);
                aait->second = tmp;
            } else {
                accum[inp.data()] = grad_outputs[i];
            }

            if (pushed.insert(inp.data()).second) {
                stack.push_back(inp.data());
            }
        }
    }
}

void AutogradEngine::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    nodes_.clear();
    output_to_node_.clear();
    next_is_checkpoint_ = false;
    // param_map_ is intentionally PRESERVED: some callers (e.g.
    // NativeQUANTTrainer) register parameters once in the constructor and
    // rely on the mapping surviving backward()+clear() per step. Stale
    // entries are validated at lookup time (data() == key) in backward().
}

void AutogradEngine::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::lock_guard<std::mutex> plock(param_mutex_);
    nodes_.clear();
    output_to_node_.clear();
    param_map_.clear();
    next_is_checkpoint_ = false;
    last_checkpoint_.reset();
    set_enabled(false);
}

AutogradEngine& AutogradEngine::instance() {
    static AutogradEngine engine;
    // Install the layering-safe ~Tensor cleanup hook once the engine exists.
    // NOTE (bug census): the (void)hook_installed was fine (forces the
    // lambda to run), but [[maybe_unused]] states intent without a fake use.
    [[maybe_unused]] static const bool hook_installed = [] {
        detail::autograd_unregister_hook = +[](Tensor* t) {
            instance().unregister_parameter(t);
        };
        return true;
    }();
    return engine;
}

} // namespace quant
