#pragma once

#include "quant/tensor.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>
#include <mutex>
#include <atomic>

namespace quant {

class AutogradFunction {
public:
    virtual ~AutogradFunction() = default;
    virtual std::vector<Tensor> forward(const std::vector<Tensor>& inputs) = 0;
    virtual std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) = 0;
    std::vector<Tensor> saved;
};

// Gradient checkpointing wrapper (Task 116).
// Saves only the inputs and discards intermediate activations after forward.
// During backward, the forward function is recomputed to regenerate the
// intermediate values that the user-supplied backward callback needs.
class CheckpointFn : public AutogradFunction {
public:
    using ForwardFn = std::function<std::vector<Tensor>(const std::vector<Tensor>&)>;
    using BackwardFn = std::function<std::vector<Tensor>(const std::vector<Tensor>&,
                                                         const std::vector<Tensor>&)>;
    CheckpointFn(ForwardFn fwd, BackwardFn bwd)
        : fwd_(std::move(fwd)), bwd_(std::move(bwd)) {}

    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs;                 // keep only inputs; drop intermediates
        return fwd_(inputs);
    }

    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override {
        auto recomputed = fwd_(saved);  // recompute forward to recover intermediates
        return bwd_(grad_output, recomputed);
    }

private:
    ForwardFn fwd_;
    BackwardFn bwd_;
};

struct AutogradNode {
    std::shared_ptr<AutogradFunction> fn;
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
    bool is_leaf = false;
    bool checkpoint = false;
    std::shared_ptr<AutogradNode> checkpoint_next;
};

class AutogradEngine {
public:
    static AutogradEngine& instance();
    void backward(Tensor& loss);
    void register_node(const std::shared_ptr<AutogradNode>& node);
    void clear();

    // Full teardown: clears the recorded graph AND the parameter registry.
    // Must be called between independent training sessions that destroy and
    // recreate models — the parameter registry keeps raw Tensor* entries, and
    // a destroyed model's memory can be reused by a later allocation, leaving
    // stale (dangling) entries behind. Calling reset() before a new session
    // guarantees every registry entry points at a live parameter.
    void reset();

    static bool enabled() { return enabled_; }
    static void set_enabled(bool e) { enabled_ = e; }

    // Gradient checkpointing: mark next op as checkpoint boundary
    static void set_checkpoint();
    static bool is_checkpoint();

    // Register a parameter tensor so gradient is set on the original, not autograd copies
    void register_parameter(Tensor* p);

    // Autograd-aware operation helpers (create node if enabled)
    static Tensor matmul_op(const Tensor& a, const Tensor& b, int64_t M, int64_t N, int64_t K);
    static Tensor add_op(const Tensor& a, const Tensor& b);
    static Tensor silu_op(const Tensor& x);
    static Tensor mul_op(const Tensor& a, const Tensor& b);
    static Tensor rms_norm_op(const Tensor& x, const Tensor& gamma, float eps);
    static Tensor cross_entropy_op(const Tensor& logits, const Tensor& labels);
    static Tensor rotary_op(const Tensor& x, const Tensor& cos_cached, const Tensor& sin_cached,
                            int64_t seq_start, int64_t seq_len);
    static Tensor attention_op(const Tensor& q, const Tensor& k, const Tensor& v,
                               int64_t num_heads, int64_t num_kv_heads, int64_t head_dim);
    static Tensor bias_add_op(const Tensor& x, const Tensor& bias);
    static Tensor embedding_op(const Tensor& input_ids, const Tensor& weight);
    static Tensor flatten_attention_op(const Tensor& x, int64_t B, int64_t H, int64_t S, int64_t D);
    static Tensor transpose_op(const Tensor& x, int dim1, int dim2);

private:
    AutogradEngine() = default;
    ~AutogradEngine();
    mutable std::mutex mutex_;
    std::vector<std::shared_ptr<AutogradNode>> nodes_;
    mutable std::mutex param_mutex_;
    std::unordered_map<const void*, std::weak_ptr<AutogradNode>> output_to_node_;
    std::unordered_map<const void*, Tensor*> param_map_;
    static std::atomic<bool> enabled_;
    // Registry lifetime flag: flipped false as the singleton begins teardown
    // so late-destroying Tensors never touch the dying param_map_.
    static std::atomic<bool> registry_alive_;
    bool next_is_checkpoint_ = false;
    std::shared_ptr<AutogradNode> last_checkpoint_;

public:
    // Self-cleaning parameter registry: called from ~Tensor for any tensor
    // that was ever registered, so stale address-keyed entries cannot be
    // misused after the owning model dies and heap addresses get reused.
    void unregister_parameter(Tensor* p);
    static bool registry_alive();
};

Tensor matmul_grad(const Tensor& a, const Tensor& b, const Tensor& grad_output);
Tensor matmul_grad_wrt_a(const Tensor& grad_output, const Tensor& b);
Tensor matmul_grad_wrt_b(const Tensor& grad_output, const Tensor& a);
Tensor weight_grad(const Tensor& grad_output, const Tensor& a, const Tensor& weight);
Tensor relu_grad(const Tensor& x, const Tensor& grad);
Tensor silu_grad(const Tensor& x, const Tensor& grad);
Tensor gelu_grad(const Tensor& x, const Tensor& grad);
Tensor softmax_grad(const Tensor& output, const Tensor& grad);
Tensor layer_norm_grad(const Tensor& x, const Tensor& gamma, const Tensor& grad, int N, Tensor* dgamma = nullptr);
Tensor rms_norm_grad(const Tensor& x, const Tensor& gamma, const Tensor& grad, int N, Tensor* dgamma = nullptr);

Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets);
Tensor cross_entropy_grad(const Tensor& logits, const Tensor& targets);

} // namespace quant
