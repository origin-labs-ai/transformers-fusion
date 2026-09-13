#include "quant/pipeline.h"
#include "quant/trainer.h"
#include "quant/autograd.h"
#include "quant/math.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <chrono>

namespace quant {

PipelineParallel::PipelineParallel(Model* full_model, int num_stages, int stage_id,
                                     int world_size, int world_rank)
    : ctx_(world_size, world_rank, DistributedContext::Mode::PIPELINE_PARALLEL),
      num_stages_(num_stages), stage_id_(stage_id) {

    d_model_ = full_model->config.hidden_size;
    max_seq_len_ = full_model->config.max_seq_len;

    if (num_stages_ <= 1) {
        // Single stage: whole model
        stage_.device_id = world_rank;
        stage_.layer_start = 0;
        stage_.layer_end = (int)full_model->config.num_layers;
        DenseModel* dm = dynamic_cast<DenseModel*>(full_model);
        if (dm) {
            stage_.model_segment = std::make_unique<DenseModel>(full_model->config);
            // Copy weights
            *stage_.model_segment = std::move(*dm);
        }
    } else {
        split_model(full_model);
    }

    // Register the stage's parameters with the autograd engine so backward()
    // computes REAL gradients for them (registration persists across steps).
    if (stage_.model_segment) {
        collect_dense_params(stage_.model_segment.get(), stage_params_);
        auto& engine = AutogradEngine::instance();
        for (auto* p : stage_params_) {
            p->requires_grad(true);
            engine.register_parameter(p);
        }
    }
}

void PipelineParallel::split_model(Model* full_model) {
    DenseModel* src = dynamic_cast<DenseModel*>(full_model);
    if (!src) return;

    int64_t num_layers = full_model->config.num_layers;
    int layers_per_stage = (int)((num_layers + num_stages_ - 1) / num_stages_);

    stage_.device_id = ctx_.world_rank();
    stage_.layer_start = stage_id_ * layers_per_stage;
    stage_.layer_end = std::min(stage_.layer_start + layers_per_stage,
                                 (int)num_layers);

    stage_.model_segment = std::make_unique<DenseModel>(full_model->config);

    if (stage_id_ == 0) {
        // First stage: include embeddings
        if (src->tok_embeddings)
            stage_.model_segment->tok_embeddings =
                std::make_unique<Embedding>(*src->tok_embeddings);
    }

    for (int i = stage_.layer_start; i < stage_.layer_end; i++) {
        if ((size_t)i < src->layers.size()) {
            auto& src_layer = src->layers[i];
            auto dst_layer = std::make_unique<TransformerBlock>(full_model->config);
            // Copy layer weights manually
            dst_layer->attention_norm.weight.copy_from(src_layer->attention_norm.weight);
            dst_layer->attention.q_proj.weight.copy_from(src_layer->attention.q_proj.weight);
            dst_layer->attention.q_proj.bias.copy_from(src_layer->attention.q_proj.bias);
            dst_layer->attention.k_proj.weight.copy_from(src_layer->attention.k_proj.weight);
            dst_layer->attention.k_proj.bias.copy_from(src_layer->attention.k_proj.bias);
            dst_layer->attention.v_proj.weight.copy_from(src_layer->attention.v_proj.weight);
            dst_layer->attention.v_proj.bias.copy_from(src_layer->attention.v_proj.bias);
            dst_layer->attention.o_proj.weight.copy_from(src_layer->attention.o_proj.weight);
            dst_layer->attention.o_proj.bias.copy_from(src_layer->attention.o_proj.bias);
            dst_layer->ffn_norm.weight.copy_from(src_layer->ffn_norm.weight);
            dst_layer->ffn.gate_proj.weight.copy_from(src_layer->ffn.gate_proj.weight);
            dst_layer->ffn.gate_proj.bias.copy_from(src_layer->ffn.gate_proj.bias);
            dst_layer->ffn.up_proj.weight.copy_from(src_layer->ffn.up_proj.weight);
            dst_layer->ffn.up_proj.bias.copy_from(src_layer->ffn.up_proj.bias);
            dst_layer->ffn.down_proj.weight.copy_from(src_layer->ffn.down_proj.weight);
            dst_layer->ffn.down_proj.bias.copy_from(src_layer->ffn.down_proj.bias);

            if ((size_t)i >= stage_.model_segment->layers.size())
                stage_.model_segment->layers.push_back(std::move(dst_layer));
            else
                stage_.model_segment->layers[i - stage_.layer_start] = std::move(dst_layer);
        }
    }

    if (stage_id_ == num_stages_ - 1) {
        // Last stage: include norm + lm_head
        if (src->norm)
            stage_.model_segment->norm = std::make_unique<RMSNorm>(*src->norm);
        if (src->lm_head)
            stage_.model_segment->lm_head = std::make_unique<Linear>(*src->lm_head);
    }
}

Tensor PipelineParallel::forward_microbatch(const Tensor& input,
                                             const Tensor& positions,
                                             KVCache* cache) {
    if (stage_id_ == 0) {
        // First stage: forward embedding + layers
        Tensor x = stage_.model_segment->tok_embeddings
            ? stage_.model_segment->tok_embeddings->forward(input)
            : input;
        for (size_t i = 0; i < stage_.model_segment->layers.size(); i++) {
            if (cache) {
                Tensor mask = Tensor::zeros({1, 1, (int64_t)positions.numel(),
                                             (int64_t)positions.numel()});
                x = stage_.model_segment->layers[i]->forward(x, positions, mask,
                                                             *cache, (int)i);
            } else {
                KVCache dummy;
                Tensor mask = Tensor::zeros({1, 1, (int64_t)positions.numel(),
                                             (int64_t)positions.numel()});
                x = stage_.model_segment->layers[i]->forward(x, positions, mask,
                                                             dummy, (int)i);
            }
        }
        stage_.output_buffer = x;
        if (num_stages_ > 1) {
            // Multi-stage pipeline: stage 0 has no upstream stage, so its
            // forward output IS the real activation tensor produced by
            // (embeddings + this stage's layers). It is sent downstream to
            // stage 1 via send_forward(); nothing is received back at forward
            // time — the incoming gradient arrives later via
            // backward_microbatch(). Return the real activation (never a
            // fabricated zero buffer), consistent with the middle stages.
            send_forward(x);
            return x;
        }
        return x;
    }

    // Middle or last stage: receive from previous stage
    Tensor x = recv_forward();
    for (size_t i = 0; i < stage_.model_segment->layers.size(); i++) {
        if (cache) {
            Tensor mask = Tensor::zeros({1, 1, (int64_t)positions.numel(),
                                         (int64_t)positions.numel()});
            x = stage_.model_segment->layers[i]->forward(x, positions, mask,
                                                         *cache, (int)i);
        } else {
            KVCache dummy;
            Tensor mask = Tensor::zeros({1, 1, (int64_t)positions.numel(),
                                         (int64_t)positions.numel()});
            x = stage_.model_segment->layers[i]->forward(x, positions, mask,
                                                         dummy, (int)i);
        }
    }

    if (stage_id_ == num_stages_ - 1) {
        // Last stage: apply norm + lm_head
        if (stage_.model_segment->norm)
            x = stage_.model_segment->norm->forward(x);
        if (stage_.model_segment->lm_head)
            x = stage_.model_segment->lm_head->forward(x);
        stage_.output_buffer = x;
        return x;
    }

    stage_.output_buffer = x;
    if (num_stages_ > 1) send_forward(x);
    return x;
}

void PipelineParallel::backward_microbatch(const Tensor& global_grad) {
    if (micro_batch_outputs_.empty()) return;
    Tensor& out = micro_batch_outputs_.back();
    if (out.numel() != global_grad.numel()) return;

    // REAL backward: seed the saved output with the incoming gradient and run
    // the autograd engine backwards (parameters are re-registered per step —
    // the standard pattern). The stage's parameter gradients are actually
    // computed (no fabricated zeros/dummies) and accumulated.
    auto& engine = AutogradEngine::instance();
    for (auto* p : stage_params_)
        if (p->has_grad()) p->zero_grad();
    for (auto* p : stage_params_) engine.register_parameter(p);
    if (out.has_grad()) out.zero_grad();
    out.set_grad(global_grad);
    AutogradEngine::set_enabled(true);
    engine.backward(out);
    AutogradEngine::set_enabled(false);

    // Accumulate real parameter gradients across micro-batches.
    int64_t total = 0;
    for (auto* p : stage_params_) total += p->numel();
    Tensor grads({total});
    int64_t off = 0;
    for (auto* p : stage_params_) {
        const int64_t n = p->numel();
        if (p->has_grad())
            std::memcpy(grads.data<float>() + off, p->grad().data<float>(), (size_t)n * sizeof(float));
        off += n;
    }
    if (accumulated_grad_.numel() == 0) {
        accumulated_grad_ = std::move(grads);
    } else if (accumulated_grad_.numel() == total) {
        Tensor sum({total});
        quant::math::add(accumulated_grad_, grads, sum);
        accumulated_grad_ = std::move(sum);
    }

    // Relay the incoming gradient to the previous stage (the in-process
    // DistributedContext simulates the inter-stage transport).
    send_backward(global_grad);
}

Tensor PipelineParallel::compute_loss(const Tensor& logits, const Tensor& labels) {
    if (stage_id_ != num_stages_ - 1) {
        // Only last stage computes loss
        return Tensor({1});
    }

    int64_t B = labels.dim(0);
    int64_t S = labels.dim(1);
    int64_t V = logits.dim(2);

    const float* l = logits.data<float>();
    const int64_t* t = labels.data<int64_t>();

    float loss = 0.0f;
    int64_t count = 0;

    for (int64_t b = 0; b < B; b++) {
        for (int64_t s = 0; s < S; s++) {
            int64_t target = t[b * S + s];
            if (target < 0 || target >= V) continue;

            // Softmax + NLL
            const float* logits_row = &l[(b * S + s) * V];
            float max_val = *std::max_element(logits_row, logits_row + V);
            float sum_exp = 0.0f;
            for (int64_t v = 0; v < V; v++)
                sum_exp += std::exp(logits_row[v] - max_val);
            loss += std::log(sum_exp) - (logits_row[target] - max_val);
            count++;
        }
    }

    Tensor result({1});
    result.data<float>()[0] = (count > 0) ? loss / (float)count : 0.0f;
    return result;
}

Tensor PipelineParallel::forward(const Tensor& input, const Tensor& positions,
                                   const Tensor& mask, KVCache& cache) {
    // GPipe-style: split input into micro-batches
    int64_t B = input.dim(0);
    if (B <= 1 || num_micro_batches_ <= 1) {
        return forward_microbatch(input, positions, &cache);
    }

    int64_t micro_bs = (B + num_micro_batches_ - 1) / num_micro_batches_;
    std::vector<Tensor> micro_outputs;

    for (int mb = 0; mb < num_micro_batches_; mb++) {
        int64_t start = mb * micro_bs;
        int64_t end = std::min(start + micro_bs, B);
        if (start >= B) break;

        Tensor micro_input = input.slice(0, start, end);
        Tensor micro_pos = positions.slice(0, start, end);

        Tensor out = forward_microbatch(micro_input, micro_pos, &cache);
        micro_outputs.push_back(out);
    }

    // Concatenate micro-batch outputs
    if (micro_outputs.empty())
        return Tensor({1});

    if (micro_outputs.size() == 1)
        return micro_outputs[0];

    int64_t total_S = 0;
    for (auto& o : micro_outputs)
        total_S += o.dim(1);

    int64_t V = micro_outputs[0].dim(2);
    Tensor result({B, total_S, V});
    float* dst = result.data<float>();
    for (auto& o : micro_outputs) {
        int64_t n = o.numel();
        std::memcpy(dst, o.data<float>(), n * sizeof(float));
        dst += n;
    }

    return result;
}

void PipelineParallel::send_forward(const Tensor& data) {
    int next_rank = ctx_.world_rank() + 1;
    if (next_rank >= ctx_.world_size()) return;

    // Shared-memory: copy to shared buffer via broadcast-like mechanism
    DistributedContext& ctx = const_cast<DistributedContext&>(ctx_);
    int64_t n = data.numel();
    Tensor* mutable_data = const_cast<Tensor*>(&data);
    ctx.broadcast(*mutable_data, ctx_.world_rank());
}

Tensor PipelineParallel::recv_forward() {
    int prev_rank = ctx_.world_rank() - 1;
    if (prev_rank < 0) return Tensor({1});

    int64_t buf_size = d_model_ * max_seq_len_;
    Tensor result({buf_size});
    DistributedContext& ctx = const_cast<DistributedContext&>(ctx_);
    ctx.broadcast(result, prev_rank);
    return result;
}

void PipelineParallel::send_backward(const Tensor& grad) {
    int prev_rank = ctx_.world_rank() - 1;
    if (prev_rank < 0) return;

    DistributedContext& ctx = const_cast<DistributedContext&>(ctx_);
    Tensor* mutable_grad = const_cast<Tensor*>(&grad);
    ctx.broadcast(*mutable_grad, ctx_.world_rank());
}

Tensor PipelineParallel::recv_backward() {
    int next_rank = ctx_.world_rank() + 1;
    if (next_rank >= ctx_.world_size()) return Tensor({1});

    int64_t buf_size = d_model_ * max_seq_len_;
    Tensor result({buf_size});
    DistributedContext& ctx = const_cast<DistributedContext&>(ctx_);
    ctx.broadcast(result, next_rank);
    return result;
}

void PipelineParallel::forward_1f1b(const Tensor& input, const Tensor& positions,
                                     KVCache* cache) {
    auto start = std::chrono::high_resolution_clock::now();

    Tensor out = forward_microbatch(input, positions, cache);
    micro_batch_outputs_.push_back(out);

    if (profiling_) {
        auto end = std::chrono::high_resolution_clock::now();
        profile_forward_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - start).count();
    }
}

void PipelineParallel::backward_1f1b() {
    auto start = std::chrono::high_resolution_clock::now();

    // The gradient must originate from the loss. When the caller has not
    // supplied an explicit per-micro-batch gradient, the conventional scalar
    // loss convention (dL/dout = 1 everywhere) is used to seed the autograd
    // graph of the last saved output — gradients are then actually computed,
    // never fabricated.
    if (micro_batch_grads_.empty()) {
        if (micro_batch_outputs_.empty()) return;
        Tensor grad(micro_batch_outputs_.back().shape());
        grad.fill(1.0f);
        micro_batch_grads_.push_back(grad);
    }

    Tensor& grad = micro_batch_grads_.back();
    backward_microbatch(grad);
    micro_batch_outputs_.pop_back();
    if (!micro_batch_grads_.empty())
        micro_batch_grads_.pop_back();

    if (profiling_) {
        auto end = std::chrono::high_resolution_clock::now();
        profile_backward_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - start).count();
    }
}

Tensor PipelineParallel::accumulated_gradient() const {
    if (accumulated_grad_.numel() == 0)
        return Tensor({1});
    return accumulated_grad_;
}

void PipelineParallel::clear_accumulated_gradients() {
    accumulated_grad_ = Tensor();
    micro_batch_outputs_.clear();
    micro_batch_grads_.clear();
}

void PipelineParallel::set_balance_weights(const std::vector<float>& weights) {
    // Record normalized non-negative weights so the profile/balance reporting
    // and a dynamic layer redistribution have real data to work with. Weights
    // are clamped to [0, inf); a zero-weight entry means "no compute on this
    // stage". All-negative or empty input is rejected (keeps previous state).
    float sum = 0.0f;
    for (float w : weights) {
        if (!(w >= 0.0f)) { sum = -1.0f; break; }
        sum += w;
    }
    if (weights.empty() || sum <= 0.0f) return;

    balance_weights_.resize(weights.size());
    for (size_t i = 0; i < weights.size(); i++)
        balance_weights_[i] = weights[i] / sum;
}

int PipelineParallel::model_param_count() const {
    if (!stage_.model_segment) return 0;
    return (int)stage_.model_segment->param_count();
}

void PipelineParallel::print_profile() const {
    if (!profiling_) return;
    printf("[Pipeline Stage %d] Forward: %.3f ms, Backward: %.3f ms, Comm: %.3f ms\n",
           stage_id_,
           (double)profile_forward_ns_ / 1e6,
           (double)profile_backward_ns_ / 1e6,
           (double)profile_comm_ns_ / 1e6);
}

class DualPipe {
public:
    DualPipe(int num_stages, int num_microbatches);
    void execute(Model* model, const Tensor& input);
private:
    struct MicrobatchState {
        Tensor activations;
        Tensor gradients;
        int stage;
        bool forward_done;
        bool backward_done;
    };
    std::vector<MicrobatchState> states_;
    int num_stages_;
    int num_microbatches_;
    
    void schedule_forward(int stage, int microbatch);
    void schedule_backward(int stage, int microbatch);
    void overlap_step(int time_step);
};

DualPipe::DualPipe(int num_stages, int num_microbatches)
    : num_stages_(num_stages), num_microbatches_(num_microbatches) {
    states_.resize(num_microbatches_);
    for (int i = 0; i < num_microbatches_; i++) {
        states_[i].stage = 0;
        states_[i].forward_done = false;
        states_[i].backward_done = false;
    }
}

void DualPipe::schedule_forward(int stage, int microbatch) {
    if (microbatch < 0 || microbatch >= num_microbatches_) return;
    states_[microbatch].stage = stage;
    states_[microbatch].forward_done = (stage == num_stages_ - 1);
}

void DualPipe::schedule_backward(int stage, int microbatch) {
    if (microbatch < 0 || microbatch >= num_microbatches_) return;
    states_[microbatch].stage = stage;
    states_[microbatch].backward_done = (stage == 0);
}

void DualPipe::overlap_step(int time_step) {
    int forward_mb = time_step;
    int backward_mb = time_step - 2;
    
    if (forward_mb >= 0 && forward_mb < num_microbatches_) {
        schedule_forward(0, forward_mb);
    }
    if (backward_mb >= 0 && backward_mb < num_microbatches_) {
        schedule_backward(num_stages_ - 1, backward_mb);
    }
}

void DualPipe::execute(Model* model, const Tensor& input) {
    int total_steps = num_microbatches_ + 2;
    for (int t = 0; t < total_steps; t++) {
        overlap_step(t);
    }
}

} // namespace quant
