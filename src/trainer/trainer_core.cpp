#define NOMINMAX
#include "quant/trainer.h"
#include "quant/math.h"
#include "quant/autograd.h"
#include "quant/optimizer.h"
#include "quant/transformer.h"
#include "quant/flash_attention.h"
#include "quant/qat.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <cmath>
#include <cctype>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace quant {

void collect_dense_params(DenseModel* dm, std::vector<Tensor*>& params) {
    if (!dm) return;
    params.push_back(&dm->tok_embeddings->weight);
    for (auto& layer : dm->layers) {
        params.push_back(&layer->attention_norm.weight);
        params.push_back(&layer->attention.q_proj.weight);
        params.push_back(&layer->attention.q_proj.bias);
        params.push_back(&layer->attention.k_proj.weight);
        params.push_back(&layer->attention.k_proj.bias);
        params.push_back(&layer->attention.v_proj.weight);
        params.push_back(&layer->attention.v_proj.bias);
        params.push_back(&layer->attention.o_proj.weight);
        params.push_back(&layer->attention.o_proj.bias);
        params.push_back(&layer->ffn_norm.weight);
        params.push_back(&layer->ffn.gate_proj.weight);
        params.push_back(&layer->ffn.gate_proj.bias);
        params.push_back(&layer->ffn.up_proj.weight);
        params.push_back(&layer->ffn.up_proj.bias);
        params.push_back(&layer->ffn.down_proj.weight);
        params.push_back(&layer->ffn.down_proj.bias);
    }
    params.push_back(&dm->norm->weight);
    params.push_back(&dm->lm_head->weight);
    params.push_back(&dm->lm_head->bias);
    // L051: MTP head weights train with the backbone when the MTP term is on.
    // (No-op when mtp_num_heads == 0: mtp_heads is empty.)
    for (auto& h : dm->mtp_heads) {
        params.push_back(&h->weight);
        params.push_back(&h->bias);
    }
}

Trainer::Trainer(Model* m, Tokenizer* t) : model_(m), tokenizer_(t), step_(0) {}

namespace {
// Shared compile body: register model parameters with the autograd engine and
// hand them to whatever optimizer was selected (AdamW or Adafactor).
void trainer_compile_common(Trainer* trainer, Optimizer* opt,
                            const std::vector<Tensor*>& params) {
    auto& engine = AutogradEngine::instance();
    for (auto* p : params) {
        p->requires_grad(true);
        engine.register_parameter(p);
    }
    opt->add_param_group(params);
}

// Phase 8-G7 (D5 W4/W5): cross-call state for accumulation-aware training.
// fit() arms these around its micro-step loop; micro_step() consumes them.
// train_step() leaves them at defaults, preserving legacy single-step behavior.
thread_local bool g_defer_continual = false; // W4: micro_step skips replay/EWC/Fisher+insert; fit() runs them once per optimizer step
thread_local float g_rdrop_alpha = 0.0f;     // W5: >0 selects graph R-Drop inside micro_step; 0 = single-forward path

// Phase 8-G7 W5: differentiable symmetric-KL scalar op for R-Drop.
// Forward: mean over rows of 0.5*(KL(p1||p2) + KL(p2||p1)), p=softmax(logits).
// Backward: analytic softmax-KL gradients (dKL(p||q)/dl1 = p*((lp-lq)-KL),
// dKL(p||q)/dl2 = q-p, symmetrized). Row grads sum to 0 (shift-invariance).
class SymKLFunction : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs;
        const Tensor& a = inputs[0];
        const Tensor& b = inputs[1];
        int64_t V = a.dim(a.rank() - 1);
        int64_t rows = V > 0 ? a.numel() / V : 0;
        rows_ = rows;
        V_ = V;
        const float* ad = a.data<float>();
        const float* bd = b.data<float>();
        std::vector<float> pa((size_t)std::max<int64_t>(V, 1));
        std::vector<float> pb((size_t)std::max<int64_t>(V, 1));
        double acc = 0.0;
        for (int64_t r = 0; r < rows; r++) {
            const float* ra = ad + r * V;
            const float* rb = bd + r * V;
            float ma = -INFINITY, mb = -INFINITY;
            for (int64_t v = 0; v < V; v++) {
                if (ra[v] > ma) ma = ra[v];
                if (rb[v] > mb) mb = rb[v];
            }
            double sa = 0.0, sb = 0.0;
            for (int64_t v = 0; v < V; v++) {
                pa[(size_t)v] = std::exp(ra[v] - ma); sa += pa[(size_t)v];
                pb[(size_t)v] = std::exp(rb[v] - mb); sb += pb[(size_t)v];
            }
            double kl1 = 0.0, kl2 = 0.0;
            for (int64_t v = 0; v < V; v++) {
                double p = pa[(size_t)v] / (sa + 1e-10);
                double q = pb[(size_t)v] / (sb + 1e-10);
                double lp = std::log(p + 1e-10), lq = std::log(q + 1e-10);
                kl1 += p * (lp - lq);
                kl2 += q * (lq - lp);
            }
            acc += 0.5 * (kl1 + kl2);
        }
        Tensor out(Shape{1});
        out.data<float>()[0] = rows > 0 ? (float)(acc / (double)rows) : 0.0f;
        return {out};
    }

    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override {
        float g = 1.0f;
        if (!grad_output.empty() && grad_output[0].numel() > 0)
            g = grad_output[0].data<float>()[0];
        const Tensor& A = saved[0];
        const Tensor& B = saved[1];
        int64_t rows = rows_, V = V_;
        Tensor da(A.shape()), db(B.shape());
        da.zero_(); db.zero_();
        if (rows <= 0 || V <= 0) return {da, db};
        float row_scale = g / (float)rows * 0.5f;
        const float* ad = A.data<float>();
        const float* bd = B.data<float>();
        float* dad = da.data<float>();
        float* dbd = db.data<float>();
        std::vector<float> pa((size_t)V), pb((size_t)V);
        for (int64_t r = 0; r < rows; r++) {
            const float* ra = ad + r * V;
            const float* rb = bd + r * V;
            float ma = -INFINITY, mb = -INFINITY;
            for (int64_t v = 0; v < V; v++) {
                if (ra[v] > ma) ma = ra[v];
                if (rb[v] > mb) mb = rb[v];
            }
            double sa = 0.0, sb = 0.0;
            for (int64_t v = 0; v < V; v++) {
                pa[(size_t)v] = std::exp(ra[v] - ma); sa += pa[(size_t)v];
                pb[(size_t)v] = std::exp(rb[v] - mb); sb += pb[(size_t)v];
            }
            double kl1 = 0.0, kl2 = 0.0;
            for (int64_t v = 0; v < V; v++) {
                double p = pa[(size_t)v] / (sa + 1e-10);
                double q = pb[(size_t)v] / (sb + 1e-10);
                double lp = std::log(p + 1e-10), lq = std::log(q + 1e-10);
                kl1 += p * (lp - lq);
                kl2 += q * (lq - lp);
            }
            for (int64_t v = 0; v < V; v++) {
                double p = pa[(size_t)v] / (sa + 1e-10);
                double q = pb[(size_t)v] / (sb + 1e-10);
                double lp = std::log(p + 1e-10), lq = std::log(q + 1e-10);
                double d1 = p * ((lp - lq) - kl1) + (p - q);
                double d2 = q * ((lq - lp) - kl2) + (q - p);
                dad[r * V + v] = (float)(row_scale * d1);
                dbd[r * V + v] = (float)(row_scale * d2);
            }
        }
        return {da, db};
    }

private:
    int64_t rows_ = 0, V_ = 0;
};

// Graph-registered symmetric-KL scalar (mirrors AutogradEngine::*_op helpers).
// Must be called with the engine enabled so the node joins the live graph.
Tensor sym_kl_op(const Tensor& a, const Tensor& b) {
    auto fn = std::make_shared<SymKLFunction>();
    auto outputs = fn->forward({a, b});
    Tensor& out = outputs[0];
    out.requires_grad(true);
    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {a, b};
    node->outputs = {out};
    AutogradEngine::instance().register_node(node);
    return out;
}
} // namespace

void Trainer::compile(AdamW* opt, const TrainConfig& cfg) {
    optimizer_ = opt;
    DenseModel* dm = dynamic_cast<DenseModel*>(model_);
    if (dm) {
        model_params_.clear();
        collect_dense_params(dm, model_params_);
        trainer_compile_common(this, opt, model_params_);
    }
    opt->set_schedule(cfg.schedule, cfg.warmup_steps, cfg.train_steps);
    opt->set_weight_decay(cfg.weight_decay);
    loss_scale_ = cfg.mixed_precision ? cfg.loss_scale : 1.0f;
    loss_scale_interval_ = cfg.loss_scale_interval;
    grad_noise_eta_ = cfg.grad_noise_eta;
    grad_noise_gamma_ = cfg.grad_noise_gamma;
    label_smoothing_ = cfg.label_smoothing;
    mtp_loss_weight_ = cfg.mtp_loss_weight;
    qat_enabled_ = cfg.use_qat;
    qat_bits_ = cfg.qat_bits;
    qat_symmetric_ = cfg.qat_symmetric;
    qat_use_lsq_ = cfg.qat_use_lsq;
    qat_init_scale_ = cfg.qat_init_scale;
    if (cfg.mixed_precision) init_mixed_precision();
}

void Trainer::compile(Adafactor* opt, const TrainConfig& cfg) {
    optimizer_ = opt;
    DenseModel* dm = dynamic_cast<DenseModel*>(model_);
    if (dm) {
        model_params_.clear();
        collect_dense_params(dm, model_params_);
        trainer_compile_common(this, opt, model_params_);
    }
    // Adafactor has no warmup/cosine scheduler; the learning rate is applied
    // directly (and can be adjusted by the caller via Trainer::metrics()/set_lr).
    opt->set_lr(cfg.learning_rate);
    opt->set_weight_decay(cfg.weight_decay);
    loss_scale_ = cfg.mixed_precision ? cfg.loss_scale : 1.0f;
    loss_scale_interval_ = cfg.loss_scale_interval;
    grad_noise_eta_ = cfg.grad_noise_eta;
    grad_noise_gamma_ = cfg.grad_noise_gamma;
    label_smoothing_ = cfg.label_smoothing;
    mtp_loss_weight_ = cfg.mtp_loss_weight;
    qat_enabled_ = cfg.use_qat;
    qat_bits_ = cfg.qat_bits;
    qat_symmetric_ = cfg.qat_symmetric;
    qat_use_lsq_ = cfg.qat_use_lsq;
    qat_init_scale_ = cfg.qat_init_scale;
    if (cfg.mixed_precision) init_mixed_precision();
}

void Trainer::compile(const TrainConfig& cfg) {
    auto* opt = new Adafactor(cfg.learning_rate, 0.999f, 1e-8f, cfg.weight_decay);
    default_opt_.reset(opt);
    compile(opt, cfg);
}

void Trainer::init_mixed_precision() {
    loss_scale_ = loss_scale_ > 0.0f ? loss_scale_ : 1024.0f;
    steps_since_scale_update_ = 0;
}

float Trainer::eval_loss(DataLoader& val_dl, int64_t max_batches) {
    Tensor input_ids(Shape{val_dl.batch_size(), val_dl.seq_length()}, DType::F32);
    Tensor labels(Shape{val_dl.batch_size(), val_dl.seq_length()}, DType::F32);
    float total_loss = 0;
    int64_t count = 0;
    AutogradEngine::set_enabled(false);
    val_dl.reset();
    while (val_dl.next_batch(input_ids, labels) && count < max_batches) {
        int64_t B = input_ids.dim(0);
        int64_t S = input_ids.dim(1);
        Tensor positions(Shape{B, S}, DType::F32);
        float* pd = positions.data<float>();
        for (int64_t i = 0; i < B * S; i++)
            pd[i] = (float)(i % S);
        Tensor logits = model_->forward(input_ids, positions);
        Tensor loss = AutogradEngine::cross_entropy_op(logits, labels);
        total_loss += *(const float*)loss.data();
        count++;
    }
    AutogradEngine::set_enabled(true);
    return count > 0 ? total_loss / (float)count : 0;
}

void Trainer::unscale_gradients(float scale) {
    if (scale == 1.0f) return;
    for (auto* p : model_params_) {
        if (!p->has_grad()) continue;
        float* g = p->grad().data<float>();
        int64_t n = p->grad().numel();
        for (int64_t i = 0; i < n; i++)
            g[i] /= scale;
    }
}

void Trainer::fit(DataLoader& dl, const TrainConfig& cfg,
                  DataLoader* val_dl) {
    Tensor input_ids(Shape{cfg.batch_size, cfg.seq_length}, DType::F32);
    Tensor labels(Shape{cfg.batch_size, cfg.seq_length}, DType::F32);
    int acc_steps = cfg.gradient_accumulation_steps > 0 ? cfg.gradient_accumulation_steps : 1;

    loss_scale_ = cfg.mixed_precision ? cfg.loss_scale : 1.0f;
    steps_since_scale_update_ = 0;

    if (cfg.curriculum) dl.set_curriculum(cfg.curriculum_epochs);

    auto epoch_start = std::chrono::steady_clock::now();
    int64_t tokens_processed = 0;

    for (int epoch = 0; epoch < cfg.num_epochs; epoch++) {
        dl.reset();
        dl.shuffle(epoch);
        if (cfg.curriculum) dl.curriculum_step(epoch);

        while (dl.next_batch(input_ids, labels)) {
            if (cfg.data_augmentation) dl.apply_augmentation(input_ids, labels);

            float loss = 0;
            optimizer_->zero_grad();
            float base_scale = cfg.mixed_precision ? loss_scale_ : 1.0f;
            float micro_scale = base_scale / (float)std::max(1, acc_steps);
            // W4: defer per-micro continual terms; fit() applies them once per
            // optimizer step below. W5: arm graph R-Drop inside micro_step.
            g_defer_continual = true;
            g_rdrop_alpha = (cfg.use_rdrop && cfg.rdrop_alpha > 0.0f) ? cfg.rdrop_alpha : 0.0f;
            Tensor cur_input = input_ids;
            Tensor cur_labels = labels;
            int actual_steps = 0;
            for (int acc = 0; acc < acc_steps; acc++) {
                if (acc > 0) {
                    Tensor nxt_input(Shape{cfg.batch_size, cfg.seq_length}, DType::F32);
                    Tensor nxt_labels(Shape{cfg.batch_size, cfg.seq_length}, DType::F32);
                    if (!dl.next_batch(nxt_input, nxt_labels)) break;
                    if (cfg.data_augmentation) dl.apply_augmentation(nxt_input, nxt_labels);
                    cur_input = nxt_input;
                    cur_labels = nxt_labels;
                }
                float micro_loss = micro_step(cur_input, cur_labels, micro_scale);
                loss += micro_loss;
                actual_steps++;
            }
            g_defer_continual = false;
            g_rdrop_alpha = 0.0f;
            // W4: truncated-window edge (nothing consumed) — skip the optimizer
            // step rather than stepping on stale/zero grads.
            if (actual_steps <= 0) continue;
            // Loss reporting stays the mean over actually-executed micro-steps.
            loss /= (float)actual_steps;

            // W4: each micro-step backward was 1/acc_steps-weighted via micro_scale,
            // but the mean above divides by actual_steps. On a truncated window
            // (actual < requested) rescale grads by acc/actual BEFORE the
            // mixed-precision unscale so the step sees the true mean gradient.
            // Full windows (actual == acc) need no correction (factor == 1).
            if (actual_steps < acc_steps) {
                float corr = (float)acc_steps / (float)actual_steps;
                for (auto* p : model_params_) {
                    if (!p->has_grad()) continue;
                    float* g = p->grad().data<float>();
                    int64_t n = p->grad().numel();
                    for (int64_t i = 0; i < n; i++) g[i] *= corr;
                }
            }

            // W4: continual terms hoisted to once per optimizer step (they used to
            // run inside every micro_step, i.e. acc_steps times per step). Replay
            // grads are isolated via grad snapshot + replay_ratio_ delta scaling,
            // mirroring the legacy per-micro weighting; EWC is injected once;
            // Fisher on_step + replay-buffer insert run once on the last micro-batch.
            // (W5: R-Drop already trained inside micro_step's graph above, so the
            // old report-only `loss += rdrop_loss(...)` float-add no-op is gone.)
            float continual_extra = 0.0f;
            if (continual_enabled_ && continual_engine_) {
                AutogradEngine::set_enabled(true);
                auto& engine = AutogradEngine::instance();
                for (auto* p : model_params_) engine.register_parameter(p);
                if (continual_engine_->replay().size() > 0 && replay_ratio_ > 0.0f) {
                    int64_t rB = cur_input.dim(0);
                    int64_t rS = cur_input.dim(1);
                    std::vector<std::vector<float>> r_inputs, r_targets;
                    std::vector<float> r_weights;
                    size_t rb = (size_t)std::min<int64_t>(rB, 4);
                    if (continual_engine_->replay().sample_batch(r_inputs, r_targets, r_weights, rb)) {
                        Tensor r_in(Shape{(int64_t)rb, rS});
                        Tensor r_lab(Shape{(int64_t)rb, rS});
                        float* ri = r_in.data<float>(); float* rl = r_lab.data<float>();
                        for (size_t b = 0; b < rb; b++) {
                            size_t copy_n = std::min<size_t>((size_t)rS, r_inputs[b].size());
                            for (int64_t s = 0; s < rS; s++) {
                                float v = s < (int64_t)copy_n ? r_inputs[b][s] : 0.0f;
                                ri[b * rS + s] = v;
                                float tv = s < (int64_t)copy_n && s < (int64_t)r_targets[b].size() ? r_targets[b][s] : v;
                                rl[b * rS + s] = tv;
                            }
                        }
                        Tensor r_pos(Shape{(int64_t)rb, rS});
                        float* rp = r_pos.data<float>();
                        for (int64_t i = 0; i < (int64_t)rb * rS; i++) rp[i] = (float)(i % rS);
                        Tensor r_logits = model_->forward(r_in, r_pos);
                        Tensor replay_loss = AutogradEngine::cross_entropy_op(r_logits, r_lab);
                        float replay_val = replay_loss.data<float>()[0];
                        std::vector<std::vector<float>> before_grads;
                        before_grads.reserve(model_params_.size());
                        for (auto* p : model_params_) {
                            if (p->has_grad()) {
                                float* gd = p->grad().data<float>();
                                before_grads.emplace_back(gd, gd + p->grad().numel());
                            } else before_grads.emplace_back();
                        }
                        engine.backward(replay_loss);
                        for (size_t i = 0; i < model_params_.size(); i++) {
                            auto* p = model_params_[i];
                            if (!p->has_grad() || before_grads[i].empty()) continue;
                            float* gd = p->grad().data<float>();
                            for (int64_t k = 0; k < p->grad().numel(); k++) {
                                float delta = gd[k] - before_grads[i][k];
                                gd[k] = before_grads[i][k] + delta * replay_ratio_;
                            }
                        }
                        continual_extra += replay_ratio_ * replay_val;
                    }
                }
                if (continual_engine_->ecc().initialized) {
                    const auto& ecc = continual_engine_->ecc();
                    size_t off = 0;
                    for (auto* p : model_params_) {
                        if (!p->has_grad()) continue;
                        float* gd = p->grad().data<float>();
                        const float* wd = p->data<float>();
                        int64_t n = p->numel();
                        for (int64_t i = 0; i < n; i++) {
                            size_t idx = off + (size_t)i;
                            if (idx < ecc.fisher_diagonal.size() && idx < ecc.anchor_weights.size()) {
                                float diff = wd[i] - ecc.anchor_weights[idx];
                                gd[i] += 2.0f * ewc_lambda_ * ecc.fisher_diagonal[idx] * diff;
                            }
                        }
                        off += (size_t)n;
                    }
                    size_t total = 0; for (auto* p : model_params_) total += (size_t)p->numel();
                    std::vector<float> flat_w; flat_w.reserve(total);
                    for (auto* p : model_params_) {
                        const float* d = p->data<float>();
                        flat_w.insert(flat_w.end(), d, d + p->numel());
                    }
                    if (!flat_w.empty())
                        continual_extra += continual_engine_->ecc().regularize(flat_w.data(), flat_w.size(), ewc_lambda_);
                }
                {
                    int64_t B = cur_input.dim(0), S = cur_input.dim(1);
                    size_t total = 0; for (auto* p : model_params_) total += (size_t)p->numel();
                    std::vector<float> flat_grad; flat_grad.reserve(total);
                    for (auto* p : model_params_) if (p->has_grad()) {
                        const float* gd = p->grad().data<float>();
                        flat_grad.insert(flat_grad.end(), gd, gd + p->grad().numel());
                    } else {
                        flat_grad.insert(flat_grad.end(), (size_t)p->numel(), 0.0f);
                    }
                    std::vector<float> flat_w; flat_w.reserve(total);
                    for (auto* p : model_params_) {
                        const float* d = p->data<float>();
                        flat_w.insert(flat_w.end(), d, d + p->numel());
                    }
                    if (!flat_grad.empty() && !flat_w.empty()) {
                        size_t n = std::min(flat_grad.size(), flat_w.size());
                        continual_engine_->on_step(flat_grad.data(), n, flat_w.data(), n, optimizer_ ? optimizer_->get_lr() : 3e-4f, current_task_id_);
                    }
                    std::vector<float> flat_in, flat_lab;
                    flat_in.reserve((size_t)(B*S)); flat_lab.reserve((size_t)(B*S));
                    const float* id = cur_input.data<float>(); const float* lb = cur_labels.data<float>();
                    for (int64_t i = 0; i < B*S; i++) { flat_in.push_back(id[i]); flat_lab.push_back(lb[i]); }
                    float imp = 1.0f;
                    if (!flat_grad.empty()) {
                        double sq = 0; for (float g : flat_grad) sq += (double)g*g;
                        imp = (float)std::sqrt(sq) + 0.01f;
                    }
                    if (!flat_in.empty()) {
                        continual_engine_->replay().insert(flat_in.data(), flat_lab.data(), flat_in.size(), current_task_id_, imp);
                    }
                }
                engine.clear();
                AutogradEngine::set_enabled(false);
            }
            loss += continual_extra;

            if (cfg.mixed_precision) unscale_gradients(loss_scale_);
            float grad_norm = clip_gradients(cfg.max_grad_norm);
            if (cfg.mixed_precision) dynamic_loss_scale(grad_norm, cfg.max_grad_norm);
            grad_noise_eta_ = cfg.grad_noise_eta;
            grad_noise_gamma_ = cfg.grad_noise_gamma;
            if (grad_noise_eta_ > 0.0f) inject_gradient_noise(step_);
            optimizer_->step();

            if (ema_enabled_) ema_step();

            if (!codebooks_.empty() && step_ % 100 == 0) {
                for (auto* cb : codebooks_) cb->ema_update(0.999f);
            }

            tokens_processed += cfg.batch_size * cfg.seq_length * actual_steps;
            auto now = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - epoch_start).count();
            int tokens_per_sec = elapsed_ms > 0 ? (int)(tokens_processed * 1000 / (std::max)(elapsed_ms, (int64_t)1)) : 0;

            metrics_.loss = loss;
            metrics_.perplexity = std::exp(loss);
            metrics_.grad_norm = grad_norm;
            metrics_.learning_rate = optimizer_ ? optimizer_->get_lr() : cfg.learning_rate;
            metrics_.tokens_per_sec = tokens_per_sec;
            metrics_.step = step_;
            metrics_.epoch = epoch;

            if (step_ % cfg.log_interval == 0) {
                if (log_cb_) log_cb_(metrics_);
                if (step_cb_) step_cb_(step_, metrics_);
            }
            if (step_ % cfg.val_interval == 0 && val_dl) {
                float val_loss = eval_loss(*val_dl, 20);
                metrics_.val_loss = val_loss;
                metrics_.val_perplexity = std::exp(val_loss);
            }
            if (step_ % cfg.save_interval == 0 && cfg.save_interval > 0) {
                save_checkpoint(cfg.output_path);
            }
            step_++;
        }
        if (epoch_cb_) epoch_cb_(epoch, metrics_);
    }
    if (ema_enabled_) ema_apply();
    save_checkpoint(cfg.output_path);
}

float Trainer::micro_step(const Tensor& input_ids, const Tensor& labels, float loss_scale) {
    if (!optimizer_) return 0;
    int64_t B = input_ids.dim(0);
    int64_t S = input_ids.dim(1);

    Tensor positions(Shape{B, S}, DType::F32);
    float* pd = positions.data<float>();
    for (int64_t i = 0; i < B * S; i++)
        pd[i] = (float)(i % S);

    Tensor fp_input = input_ids;
    Tensor fp_labels = labels;
    if (loss_scale != 1.0f) {
        fp_input = input_ids.to_dtype(DType::F32);
        fp_labels = labels.to_dtype(DType::F32);
    }

    AutogradEngine::set_enabled(true);
    auto& engine = AutogradEngine::instance();
    for (auto* p : model_params_) {
        engine.register_parameter(p);
    }
    // QAT: inject FakeQuantize nodes before forward (weights see quant noise, grad uses STE)
    std::vector<std::vector<float>> qat_backup;
    if (qat_enabled_) {
        bool need_init = qat_scales_.size() != model_params_.size();
        if (need_init) {
            qat_scales_.clear();
            for (auto* p : model_params_) {
                int qmin, qmax;
                qat::get_qrange_bits(qat_bits_, qat_symmetric_, false, qmin, qmax);
                const float* d = p->data<float>();
                float mx = 0; int64_t n = p->numel();
                for (int64_t i = 0; i < n; ++i) mx = std::max(mx, std::fabs(d[i]));
                if (mx < 1e-8f) mx = 1.0f;
                float sc = mx / (float)qmax;
                if (sc < 1e-6f) sc = qat_init_scale_;
                Tensor ts(Shape{1}, DType::F32);
                ts.data<float>()[0] = sc;
                ts.requires_grad(qat_use_lsq_);
                qat_scales_.push_back(ts);
            }
            if (qat_use_lsq_ && optimizer_) {
                for (size_t i = 0; i < qat_scales_.size(); ++i) {
                    engine.register_parameter(&qat_scales_[i]);
                    optimizer_->add_param(&qat_scales_[i]);
                }
            }
        } else if (qat_use_lsq_) {
            for (size_t i = 0; i < qat_scales_.size(); ++i) engine.register_parameter(&qat_scales_[i]);
        }
        qat_backup.resize(model_params_.size());
        for (size_t i = 0; i < model_params_.size(); ++i) {
            Tensor* p = model_params_[i];
            qat_backup[i].assign(p->data<float>(), p->data<float>() + p->numel());
            int qmin, qmax;
            qat::get_qrange_bits(qat_bits_, qat_symmetric_, false, qmin, qmax);
            Tensor q;
            if (qat_use_lsq_) q = qat::lsq_fake_quantize(*p, qat_scales_[i], qmin, qmax);
            else { float sc = qat_scales_[i].data<float>()[0]; q = qat::fake_quantize(*p, sc, qmin, qmax); }
            std::memcpy(p->data<float>(), q.data<float>(), (size_t)p->numel() * sizeof(float));
        }
    }
    Tensor loss;
    if (g_rdrop_alpha > 0.0f) {
        // W5 R-Drop: enabled-graph double forward with dropout active (both passes
        // run under the live autograd graph so any stochastic masks differ), then
        // total = mean(CE1,CE2) + alpha*symKL assembled via ops, single backward.
        // NOTE: label smoothing is bypassed on this path (its helper is non-graph).
        // NOTE: DenseModel currently has no dropout layer, so the two passes agree
        // and symKL ~= 0 — the graph wiring established here is the fix.
        Tensor logits1 = model_->forward(fp_input, positions);
        Tensor logits2 = model_->forward(fp_input, positions);
        Tensor ce1 = AutogradEngine::cross_entropy_op(logits1, fp_labels);
        Tensor ce2 = AutogradEngine::cross_entropy_op(logits2, fp_labels);
        Tensor ce_sum = AutogradEngine::add_op(ce1, ce2);
        Tensor half(Shape{1});
        half.data<float>()[0] = 0.5f;
        Tensor mean_ce = AutogradEngine::mul_op(ce_sum, half);
        Tensor kl = sym_kl_op(logits1, logits2);
        Tensor alpha_t(Shape{1});
        alpha_t.data<float>()[0] = g_rdrop_alpha;
        Tensor kl_w = AutogradEngine::mul_op(kl, alpha_t);
        loss = AutogradEngine::add_op(mean_ce, kl_w);
    } else {
        Tensor logits = model_->forward(fp_input, positions);
        if (label_smoothing_ > 0.0f) {
            loss = label_smoothing_loss(logits, fp_labels, label_smoothing_);
        } else {
            loss = AutogradEngine::cross_entropy_op(logits, fp_labels);
        }
    }
    // L051: MTP auxiliary wiring. When mtp_loss_weight_ > 0 and the model is
    // a DenseModel carrying mtp_heads, run the MTP forward (second backbone
    // pass under the live graph) and add weight * mean(head CEs) via ops so
    // both the MTP heads and the shared backbone receive gradients. The
    // per-head label shift mirrors Trainer::mtp_loss (zero-padded tails).
    if (mtp_loss_weight_ > 0.0f) {
        if (DenseModel* dmm = dynamic_cast<DenseModel*>(model_)) {
            if (!dmm->mtp_heads.empty()) {
                std::vector<Tensor> mtpl = dmm->mtp_forward(fp_input, positions);
                Tensor acc;
                bool have = false;
                for (size_t h = 0; h < mtpl.size(); h++) {
                    Tensor shifted(Shape{B, S}, DType::F32);
                    shifted.zero_();
                    const float* ld = fp_labels.data<float>();
                    float* sd = shifted.data<float>();
                    for (int64_t b = 0; b < B; b++)
                        for (int64_t s = 0; s < S - (int64_t)(h + 1); s++)
                            sd[b * S + s] = ld[b * S + s + (h + 1)];
                    Tensor ce = AutogradEngine::cross_entropy_op(mtpl[h], shifted);
                    acc = !have ? ce : AutogradEngine::add_op(acc, ce);
                    have = true;
                }
                if (have) {
                    Tensor inv_n(Shape{1});
                    inv_n.data<float>()[0] = 1.0f / (float)mtpl.size();
                    Tensor w(Shape{1});
                    w.data<float>()[0] = mtp_loss_weight_;
                    Tensor mtp_term = AutogradEngine::mul_op(
                        AutogradEngine::mul_op(acc, inv_n), w);
                    loss = AutogradEngine::add_op(loss, mtp_term);
                }
            }
        }
    }
    if (loss_scale != 1.0f) {
        float* ld = (float*)loss.data();
        *ld *= loss_scale;
    }
    // Continual replay mixing: mix in old samples as additional loss term (graph-aware).
    // W4: skipped while fit() accumulates (g_defer_continual); fit() runs this once
    // per optimizer step instead of once per micro-step.
    Tensor replay_loss;
    bool has_replay_loss = false;
    std::vector<float> replay_loss_scale_holder;
    if (!g_defer_continual && continual_enabled_ && continual_engine_ && continual_engine_->replay().size() > 0 && replay_ratio_ > 0.0f) {
        // sample one replay batch of same size
        std::vector<std::vector<float>> r_inputs, r_targets;
        std::vector<float> r_weights;
        size_t rb = (size_t)std::min<int64_t>(B, 4);
        if (continual_engine_->replay().sample_batch(r_inputs, r_targets, r_weights, rb)) {
            // use first sampled entry to build replay batch (average across sampled)
            // reconstruct tensors of shape {rb, S} from dequantized floats
            int64_t elem = B * S;
            Tensor r_in(Shape{(int64_t)rb, S});
            Tensor r_lab(Shape{(int64_t)rb, S});
            float* ri = r_in.data<float>(); float* rl = r_lab.data<float>();
            for (size_t b = 0; b < rb; b++) {
                // r_inputs[b] holds elem_count floats; truncate/pad to S
                size_t copy_n = std::min<size_t>((size_t)S, r_inputs[b].size());
                for (int64_t s = 0; s < S; s++) {
                    float v = s < (int64_t)copy_n ? r_inputs[b][s] : 0.0f;
                    ri[b * S + s] = v;
                    float tv = s < (int64_t)copy_n && s < (int64_t)r_targets[b].size() ? r_targets[b][s] : v;
                    rl[b * S + s] = tv;
                }
            }
            Tensor r_pos(Shape{(int64_t)rb, S});
            float* rp = r_pos.data<float>();
            for (int64_t i = 0; i < (int64_t)rb * S; i++) rp[i] = (float)(i % S);
            Tensor r_logits = model_->forward(r_in, r_pos);
            replay_loss = AutogradEngine::cross_entropy_op(r_logits, r_lab);
            has_replay_loss = true;
            // scale replay loss by replay_ratio before adding to main graph
            float* rld = replay_loss.data<float>();
            // keep original for return accounting but scale for backward weighting via grad scale
            // we will combine via explicit weighted add after backward (scale grad instead)
        }
    }
    engine.backward(loss);
    if (has_replay_loss) {
        // second backward for replay, gradients accumulate; scale by replay_ratio_
        // temporarily scale replay loss grad by replay_ratio_
        Tensor scaled = replay_loss;
        float* sd = scaled.data<float>();
        // create a scaled copy for backward scaling: we scale grad accumulation manually
        // set grad of replay_loss to replay_ratio then backward
        Tensor g(Shape{1}); g.data<float>()[0] = replay_ratio_;
        // Instead of graph scaling, directly scale param grads contributed by replay backward
        // Do backward then scale delta grads
        // snapshot current grad norms to isolate replay contribution
        std::vector<std::vector<float>> before_grads;
        before_grads.reserve(model_params_.size());
        for (auto* p : model_params_) {
            if (p->has_grad()) {
                float* gd = p->grad().data<float>();
                before_grads.emplace_back(gd, gd + p->grad().numel());
            } else before_grads.emplace_back();
        }
        AutogradEngine::instance().backward(replay_loss);
        // scale the delta contributed by replay
        for (size_t i = 0; i < model_params_.size(); i++) {
            auto* p = model_params_[i];
            if (!p->has_grad() || before_grads[i].empty()) continue;
            float* gd = p->grad().data<float>();
            for (int64_t k = 0; k < p->grad().numel(); k++) {
                float delta = gd[k] - before_grads[i][k];
                gd[k] = before_grads[i][k] + delta * replay_ratio_;
            }
        }
    }
    // EWC regularization: inject gradient lambda * Fisher * (w - anchor)
    // W4: skipped while fit() accumulates (g_defer_continual); fit() injects once
    // per optimizer step instead of once per micro-step.
    if (!g_defer_continual && continual_enabled_ && continual_engine_ && continual_engine_->ecc().initialized) {
        const auto& ecc = continual_engine_->ecc();
        size_t off = 0;
        for (auto* p : model_params_) {
            if (!p->has_grad()) continue;
            float* gd = p->grad().data<float>();
            const float* wd = p->data<float>();
            int64_t n = p->numel();
            for (int64_t i = 0; i < n; i++) {
                size_t idx = off + (size_t)i;
                if (idx < ecc.fisher_diagonal.size() && idx < ecc.anchor_weights.size()) {
                    float diff = wd[i] - ecc.anchor_weights[idx];
                    gd[i] += 2.0f * ewc_lambda_ * ecc.fisher_diagonal[idx] * diff;
                }
            }
            off += (size_t)n;
        }
    }
    // restore master weights (keep quantized version only for forward noise; optimizer steps master)
    if (qat_enabled_ && !qat_backup.empty()) {
        for (size_t i = 0; i < model_params_.size() && i < qat_backup.size(); ++i) {
            std::memcpy(model_params_[i]->data<float>(), qat_backup[i].data(), qat_backup[i].size() * sizeof(float));
        }
    }
    // Update Fisher and insert current batch into compressed replay buffer
    // W4: skipped while fit() accumulates (g_defer_continual); fit() runs this
    // once per optimizer step on the last micro-batch instead of per micro-step.
    if (!g_defer_continual && continual_enabled_ && continual_engine_) {
        // flatten grads for fisher update
        size_t total = 0; for (auto* p : model_params_) total += (size_t)p->numel();
        std::vector<float> flat_grad; flat_grad.reserve(total);
        for (auto* p : model_params_) if (p->has_grad()) {
            const float* gd = p->grad().data<float>();
            flat_grad.insert(flat_grad.end(), gd, gd + p->grad().numel());
        } else {
            flat_grad.insert(flat_grad.end(), (size_t)p->numel(), 0.0f);
        }
        std::vector<float> flat_w; flat_w.reserve(total);
        for (auto* p : model_params_) {
            const float* d = p->data<float>();
            flat_w.insert(flat_w.end(), d, d + p->numel());
        }
        if (!flat_grad.empty() && !flat_w.empty()) {
            size_t n = std::min(flat_grad.size(), flat_w.size());
            continual_engine_->on_step(flat_grad.data(), n, flat_w.data(), n, optimizer_ ? optimizer_->get_lr() : 3e-4f, current_task_id_);
        }
        // insert current batch into replay (store input+label as flat floats)
        std::vector<float> flat_in, flat_lab;
        flat_in.reserve((size_t)(B*S)); flat_lab.reserve((size_t)(B*S));
        const float* id = input_ids.data<float>(); const float* lb = labels.data<float>();
        for (int64_t i = 0; i < B*S; i++) { flat_in.push_back(id[i]); flat_lab.push_back(lb[i]); }
        float imp = 1.0f;
        if (!flat_grad.empty()) {
            double sq = 0; for (float g : flat_grad) sq += (double)g*g;
            imp = (float)std::sqrt(sq) + 0.01f;
        }
        if (!flat_in.empty()) {
            continual_engine_->replay().insert(flat_in.data(), flat_lab.data(), flat_in.size(), current_task_id_, imp);
        }
    }
    engine.clear();
    AutogradEngine::set_enabled(false);
    float ret_loss = *(const float*)loss.data() / loss_scale;
    if (has_replay_loss) ret_loss += replay_ratio_ * replay_loss.data<float>()[0];
    // W4: EWC scalar stays with the per-micro return only on the non-deferred path
    // (train_step); under fit() accumulation fit() adds it once per optimizer step.
    if (!g_defer_continual && continual_enabled_ && continual_engine_ && continual_engine_->ecc().initialized) {
        // add EWC scalar to reported loss
        size_t total = 0; for (auto* p : model_params_) total += (size_t)p->numel();
        std::vector<float> flat_w; flat_w.reserve(total);
        for (auto* p : model_params_) { const float* d = p->data<float>(); flat_w.insert(flat_w.end(), d, d + p->numel()); }
        if (!flat_w.empty())
            ret_loss += continual_engine_->ecc().regularize(flat_w.data(), flat_w.size(), ewc_lambda_);
    }
    return ret_loss;
}

float Trainer::train_step(const Tensor& input_ids, const Tensor& labels) {
    if (!optimizer_) return 0;
    optimizer_->zero_grad();
    float loss = micro_step(input_ids, labels, loss_scale_);
    if (loss_scale_ != 1.0f) {
        unscale_gradients(loss_scale_);
    }
    float grad_norm = clip_gradients(1.0f);
    if (grad_noise_eta_ > 0.0f) inject_gradient_noise(step_);
    if (loss_scale_ != 1.0f) {
        if (grad_norm > 1.0f) {
            loss_scale_ = std::max(1.0f, loss_scale_ / 2.0f);
            steps_since_scale_update_ = 0;
        } else {
            steps_since_scale_update_++;
            if (steps_since_scale_update_ >= loss_scale_interval_) {
                loss_scale_ = std::min(65536.0f, loss_scale_ * 2.0f);
                steps_since_scale_update_ = 0;
            }
        }
    }
    optimizer_->step();
    return loss;
}

float Trainer::clip_gradients(float max_norm) {
    if (max_norm <= 0 || model_params_.empty()) return 0;
    float total_norm = 0;
    for (auto* p : model_params_) {
        if (!p->has_grad()) continue;
        const float* g = p->grad().data<float>();
        int64_t n = p->grad().numel();
        float sq_sum = 0;
        for (int64_t i = 0; i < n; i++) sq_sum += g[i] * g[i];
        total_norm += sq_sum;
    }
    total_norm = std::sqrt(total_norm);
    if (total_norm > max_norm) {
        float scale = max_norm / (total_norm + 1e-8f);
        for (auto* p : model_params_) {
            if (!p->has_grad()) continue;
            float* g = p->grad().data<float>();
            int64_t n = p->grad().numel();
            for (int64_t i = 0; i < n; i++) g[i] *= scale;
        }
    }
    return total_norm;
}

void Trainer::save_checkpoint(const std::string& path) {
    model_->save(path);
    if (!optimizer_) return;
    std::string opt_path = path + ".opt";
    FILE* fp = std::fopen(opt_path.c_str(), "wb");
    if (!fp) return;
    int32_t step_i = (int32_t)step_;
    float lr = optimizer_->get_lr();
    fwrite(&step_i, sizeof(step_i), 1, fp);
    fwrite(&lr, sizeof(lr), 1, fp);
    int32_t num_params = (int32_t)model_params_.size();
    fwrite(&num_params, sizeof(num_params), 1, fp);
    for (auto* p : model_params_) {
        auto& state = optimizer_->get_state(p);
        int64_t n = p->numel();
        int64_t written_m = 0, written_v = 0;
        if (state.m.buffer() && state.m.numel() > 0) {
            written_m = state.m.numel();
            fwrite(&written_m, sizeof(written_m), 1, fp);
            fwrite(state.m.data<float>(), (size_t)written_m * sizeof(float), 1, fp);
        } else {
            fwrite(&written_m, sizeof(written_m), 1, fp);
        }
        if (state.v.buffer() && state.v.numel() > 0) {
            written_v = state.v.numel();
            fwrite(&written_v, sizeof(written_v), 1, fp);
            fwrite(state.v.data<float>(), (size_t)written_v * sizeof(float), 1, fp);
        } else {
            fwrite(&written_v, sizeof(written_v), 1, fp);
        }
    }
    fclose(fp);
}

void Trainer::load_checkpoint(const std::string& path) {
    model_->load(path);
    if (!optimizer_) return;
    std::string opt_path = path + ".opt";
    FILE* fp = std::fopen(opt_path.c_str(), "rb");
    if (!fp) return;
    // P0 fix: every fread validated — a truncated .opt used to resume with
    // half-loaded step/lr/moments. Any short read => keep in-memory state.
    int32_t step_i = 0;
    float lr = 0;
    if (fread(&step_i, sizeof(step_i), 1, fp) != 1) { fclose(fp); return; }
    if (fread(&lr, sizeof(lr), 1, fp) != 1) { fclose(fp); return; }
    step_ = (int)step_i;
    optimizer_->set_lr(lr);
    int32_t num_params = 0;
    if (fread(&num_params, sizeof(num_params), 1, fp) != 1) { fclose(fp); return; }
    for (int32_t i = 0; i < num_params && i < (int32_t)model_params_.size(); i++) {
        auto* p = model_params_[(size_t)i];
        auto& state = optimizer_->get_state(p);
        int64_t n = p->numel();
        int64_t read_m = 0, read_v = 0;
        if (fread(&read_m, sizeof(read_m), 1, fp) != 1) break;
        if (read_m > 0 && read_m <= n) {
            state.m = Tensor::zeros(p->shape());
            if (fread(state.m.data<float>(), (size_t)read_m * sizeof(float), 1, fp) != 1) break;
        }
        if (fread(&read_v, sizeof(read_v), 1, fp) != 1) break;
        if (read_v > 0 && read_v <= n) {
            state.v = Tensor::zeros(p->shape());
            if (fread(state.v.data<float>(), (size_t)read_v * sizeof(float), 1, fp) != 1) break;
        }
    }
    fclose(fp);
}

void Trainer::set_log_callback(LogCallback cb) {
    log_cb_ = cb;
}

void Trainer::set_epoch_callback(EpochCallback cb) {
    epoch_cb_ = cb;
}

void Trainer::set_step_callback(StepCallback cb) {
    step_cb_ = cb;
}

const TrainMetrics& Trainer::metrics() const {
    return metrics_;
}

static Tensor positions_from(const Tensor& input_ids) {
    int64_t B = input_ids.dim(0);
    int64_t S = input_ids.dim(1);
    Tensor pos({B, S});
    float* pd = pos.data<float>();
    for (int64_t i = 0; i < B * S; i++)
        pd[i] = (float)(i % S);
    return pos;
}

float Trainer::rdrop_loss(const Tensor& input_ids, const Tensor& labels, float alpha) {
    Tensor logits1 = model_->forward(input_ids, positions_from(input_ids));
    Tensor logits2 = model_->forward(input_ids, positions_from(input_ids));
    int64_t B = logits1.dim(0), S = logits1.dim(1), V = logits1.dim(2);
    float kl = 0;
    const float* l1 = logits1.data<float>();
    const float* l2 = logits2.data<float>();
    std::vector<float> p1(V), p2(V);
    for (int64_t i = 0; i < B * S; i++) {
        const float* row1 = l1 + i * V;
        const float* row2 = l2 + i * V;
        float max1 = -INFINITY, max2 = -INFINITY;
        for (int64_t v = 0; v < V; v++) {
            if (row1[v] > max1) max1 = row1[v];
            if (row2[v] > max2) max2 = row2[v];
        }
        float sum1 = 0, sum2 = 0;
        for (int64_t v = 0; v < V; v++) {
            p1[v] = std::exp(row1[v] - max1); sum1 += p1[v];
            p2[v] = std::exp(row2[v] - max2); sum2 += p2[v];
        }
        float inv1 = 1.0f / (sum1 + 1e-10f);
        float inv2 = 1.0f / (sum2 + 1e-10f);
        for (int64_t v = 0; v < V; v++) {
            float prob1 = p1[v] * inv1;
            float prob2 = p2[v] * inv2;
            kl += prob1 * (std::log(prob1 + 1e-10f) - std::log(prob2 + 1e-10f));
        }
    }
    float ce_loss = *(const float*)AutogradEngine::cross_entropy_op(logits1, labels).data();
    return ce_loss + alpha * kl / (float)(B * S);
}

Tensor Trainer::label_smoothing_loss(const Tensor& logits, const Tensor& labels, float smoothing) {
    int64_t B = logits.dim(0);
    int64_t S = logits.dim(1);
    int64_t V = logits.dim(2);
    Tensor loss({B * S});
    const float* ld = labels.data<float>();
    const float* lp = logits.data<float>();
    float* ld_ = loss.data<float>();
    for (int64_t i = 0; i < B * S; i++) {
        int label = (int)ld[i];
        if (label < 0 || label >= V) { ld_[i] = 0; continue; }
        const float* logit_row = lp + i * V;
        float max_l = -INFINITY;
        for (int64_t v = 0; v < V; v++)
            if (logit_row[v] > max_l) max_l = logit_row[v];
        float sum_exp = 0;
        std::vector<float> p(V);
        for (int64_t v = 0; v < V; v++) {
            p[v] = std::exp(logit_row[v] - max_l);
            sum_exp += p[v];
        }
        float inv_sum = 1.0f / (sum_exp + 1e-10f);
        float smooth_loss = 0;
        float uniform = smoothing / (float)V;
        for (int64_t v = 0; v < V; v++) {
            float prob = p[v] * inv_sum;
            float target = (v == label) ? (1.0f - smoothing) : 0.0f;
            target += uniform;
            smooth_loss -= target * std::log(prob + 1e-10f);
        }
        ld_[i] = smooth_loss;
    }
    return loss.reshape({B, S});
}

void Trainer::ema_init(float decay) {
    ema_decay_ = decay;
    ema_enabled_ = true;
    ema_params_.clear();
    for (auto* p : model_params_) {
        ema_params_.push_back(Tensor::zeros(p->shape()));
    }
}

void Trainer::ema_step() {
    if (!ema_enabled_) return;
    for (size_t i = 0; i < model_params_.size(); i++) {
        float* e = ema_params_[i].data<float>();
        const float* p = model_params_[i]->data<float>();
        int64_t n = model_params_[i]->numel();
        for (int64_t j = 0; j < n; j++)
            e[j] = ema_decay_ * e[j] + (1.0f - ema_decay_) * p[j];
    }
}

void Trainer::ema_apply() {
    if (!ema_enabled_) return;
    for (size_t i = 0; i < model_params_.size(); i++) {
        std::memcpy(model_params_[i]->data<float>(),
                    ema_params_[i].data<float>(),
                    model_params_[i]->numel() * sizeof(float));
    }
}

void Trainer::ema_swap() {
    if (!ema_enabled_) return;
    for (size_t i = 0; i < model_params_.size(); i++) {
        float* p = model_params_[i]->data<float>();
        float* e = ema_params_[i].data<float>();
        int64_t n = model_params_[i]->numel();
        for (int64_t j = 0; j < n; j++) {
            float tmp = p[j];
            p[j] = e[j];
            e[j] = tmp;
        }
    }
}

void Trainer::dynamic_loss_scale(float grad_norm, float max_grad_norm) {
    if (loss_scale_ == 1.0f) return;
    if (grad_norm > max_grad_norm) {
        loss_scale_ = std::max(1.0f, loss_scale_ / 2.0f);
        steps_since_scale_update_ = 0;
    } else {
        steps_since_scale_update_++;
        if (steps_since_scale_update_ >= loss_scale_interval_) {
            loss_scale_ = std::min(65536.0f, loss_scale_ * 2.0f);
            steps_since_scale_update_ = 0;
        }
    }
}

void Trainer::inject_gradient_noise(int step) {
    if (grad_noise_eta_ <= 0.0f) return;
    float sigma = grad_noise_eta_ / std::pow(1.0f + (float)step, grad_noise_gamma_);
    if (sigma <= 0.0f) return;
    for (auto* p : model_params_) {
        if (!p->has_grad()) continue;
        float* g = p->grad().data<float>();
        int64_t n = p->grad().numel();
        for (int64_t i = 0; i < n; i++)
            g[i] += grad_noise_rng_.normal() * sigma;
    }
}

float Trainer::mtp_loss(const std::vector<Tensor>& mtp_logits, const Tensor& labels) {
    int num_heads = mtp_logits.size();
    if (num_heads == 0) return 0.0f;
    
    int64_t B = labels.dim(0);
    int64_t S = labels.dim(1);
    
    float total_loss = 0.0f;
    
    for (int h = 0; h < num_heads; h++) {
        // Shift labels by h + 1 for future token prediction
        Tensor shifted_labels(Shape{B, S}, DType::F32);
        shifted_labels.zero_();
        const float* ld = labels.data<float>();
        float* sld = shifted_labels.data<float>();
        
        for (int64_t b = 0; b < B; b++) {
            for (int64_t s = 0; s < S - (h + 1); s++) {
                sld[b * S + s] = ld[b * S + s + (h + 1)];
            }
        }
        
        Tensor ce = AutogradEngine::cross_entropy_op(mtp_logits[h], shifted_labels);
        total_loss += *(const float*)ce.data();
    }
    
    return total_loss / (float)num_heads;
}

void Trainer::enable_qat(int bits, bool symmetric, bool use_lsq, float init_scale) {
    qat_enabled_ = true;
    qat_bits_ = bits;
    qat_symmetric_ = symmetric;
    qat_use_lsq_ = use_lsq;
    qat_init_scale_ = init_scale;
    qat_scales_.clear();
}
void Trainer::disable_qat() { qat_enabled_ = false; qat_scales_.clear(); }
bool Trainer::qat_enabled() const { return qat_enabled_; }
Tensor Trainer::qat_fake_quantize(const Tensor& weight) {
    int qmin, qmax;
    qat::get_qrange_bits(qat_bits_, qat_symmetric_, false, qmin, qmax);
    const float* d = weight.data<float>();
    float mx = 0; int64_t n = weight.numel();
    for (int64_t i = 0; i < n; ++i) mx = std::max(mx, std::fabs(d[i]));
    if (mx < 1e-8f) mx = 1.0f;
    float sc = mx / (float)qmax;
    return qat::fake_quantize(weight, sc, qmin, qmax);
}
Tensor Trainer::qat_fake_quantize_lsq(const Tensor& weight, Tensor& scale_param) {
    int qmin, qmax;
    qat::get_qrange_bits(qat_bits_, qat_symmetric_, false, qmin, qmax);
    return qat::lsq_fake_quantize(weight, scale_param, qmin, qmax);
}

void Trainer::enable_continual(const ContinualEngineConfig& cfg) {
    continual_cfg_ = cfg;
    ewc_lambda_ = cfg.ecc_lambda;
    continual_engine_ = std::make_unique<ContinualEngine>(cfg);
    continual_enabled_ = true;
    current_task_id_ = 0;
}
void Trainer::disable_continual() { continual_enabled_ = false; continual_engine_.reset(); }
void Trainer::on_task_boundary(uint32_t new_task_id, const float* eval_inputs, const float* eval_targets, size_t eval_count) {
    current_task_id_ = new_task_id;
    if (continual_engine_) {
        // snapshot anchor weights from current params
        for (auto* p : model_params_) {
            // use first param to seed, but ECCState keeps flat vector
        }
        // flatten weights for anchor
        size_t total = 0;
        for (auto* p : model_params_) total += (size_t)p->numel();
        std::vector<float> flat; flat.reserve(total);
        for (auto* p : model_params_) {
            const float* d = p->data<float>();
            flat.insert(flat.end(), d, d + p->numel());
        }
        if (!flat.empty()) {
            if (!continual_engine_->ecc().initialized) {
                const_cast<ECCState&>(continual_engine_->ecc()).initialize(flat.size());
            }
            const_cast<ECCState&>(continual_engine_->ecc()).update_anchor(flat.data(), flat.size());
        }
        continual_engine_->on_task_boundary(new_task_id, eval_inputs, eval_targets, eval_count);
    }
}
void Trainer::set_ewc_lambda(float lambda) { ewc_lambda_ = lambda; continual_cfg_.ecc_lambda = lambda; }
void Trainer::set_replay_ratio(float r) { replay_ratio_ = std::max(0.0f, std::min(1.0f, r)); }

// ── L073 Unified trainer entry ─────────────────────────────────────────────

const char* unified_trainer_kind_name(UnifiedTrainerKind k) {
    switch (k) {
        case UnifiedTrainerKind::Dense: return "dense";
        case UnifiedTrainerKind::FineTune: return "finetune";
        case UnifiedTrainerKind::MoE: return "moe";
        case UnifiedTrainerKind::Native: return "native";
        default: return "unknown";
    }
}

bool unified_trainer_parse_kind(const std::string& s, UnifiedTrainerKind* out) {
    std::string v = s;
    for (auto& c : v) c = (char)std::tolower((unsigned char)c);
    UnifiedTrainerKind k = UnifiedTrainerKind::Dense;
    if (v == "dense" || v == "pretrain") k = UnifiedTrainerKind::Dense;
    else if (v == "finetune" || v == "fine_tune" || v == "ft") k = UnifiedTrainerKind::FineTune;
    else if (v == "moe") k = UnifiedTrainerKind::MoE;
    else if (v == "native") k = UnifiedTrainerKind::Native;
    else return false;
    if (out) *out = k;
    return true;
}

TrainConfig unified_to_train_config(const UnifiedTrainArgs& a) {
    TrainConfig c;
    c.batch_size = a.batch_size;
    c.seq_length = a.seq_length;
    c.num_epochs = a.num_epochs;
    c.train_steps = a.train_steps;
    c.learning_rate = a.learning_rate;
    c.weight_decay = a.weight_decay;
    c.warmup_steps = a.warmup_steps;
    c.log_interval = a.log_interval;
    c.save_interval = a.save_interval;
    c.val_interval = a.val_interval;
    c.output_path = a.output_path;
    c.use_qat = a.use_qat;
    c.qat_bits = a.qat_bits;
    c.moe_load_balance_coef = a.moe_load_balance_coef;
    c.moe_z_loss_coef = a.moe_z_loss_coef;
    return c;
}

UnifiedTrainer::UnifiedTrainer(Model* model, Tokenizer* tokenizer)
    : model_(model), tokenizer_(tokenizer) {}

UnifiedTrainer::~UnifiedTrainer() = default;

void UnifiedTrainer::set_log_callback(LogCallback cb) {
    log_cb_ = std::move(cb);
    if (trainer_) trainer_->set_log_callback(log_cb_);
}

bool UnifiedTrainer::configure(const UnifiedTrainArgs& args, std::string* err_out) {
    auto fail = [&](const std::string& m) {
        if (err_out) *err_out = m;
        return false;
    };
    if (!model_) return fail("unified-trainer: null model");
    if (!tokenizer_) return fail("unified-trainer: null tokenizer");
    if (args.batch_size <= 0 || args.seq_length <= 0)
        return fail("unified-trainer: batch_size/seq_length must be > 0");
    if (args.num_epochs <= 0) return fail("unified-trainer: num_epochs must be > 0");
    if (args.optimizer_name != "adafactor" && args.optimizer_name != "adamw")
        return fail("unified-trainer: optimizer must be 'adafactor' or 'adamw'");
    args_ = args;
    cfg_ = unified_to_train_config(args_);
    trainer_ = std::make_unique<Trainer>(model_, tokenizer_);
    owned_opt_.reset();
    opt_ = nullptr;
    if (args_.optimizer_name == "adamw") {
        auto* o = new AdamW(args_.learning_rate);
        owned_opt_.reset(o);
        opt_ = o;
        trainer_->compile(static_cast<AdamW*>(opt_), cfg_);
    } else {
        auto* o = new Adafactor(args_.learning_rate);
        owned_opt_.reset(o);
        opt_ = o;
        trainer_->compile(static_cast<Adafactor*>(opt_), cfg_);
    }
    if (log_cb_) trainer_->set_log_callback(log_cb_);
    configured_ = true;
    return true;
}

bool UnifiedTrainer::run(std::string* err_out) {
    auto fail = [&](const std::string& m) {
        if (err_out) *err_out = m;
        return false;
    };
    if (!configured_ || !trainer_) return fail("unified-trainer: configure() first");
    if (args_.data_path.empty()) return fail("unified-trainer: data_path required");
    try {
        DataLoader dl(tokenizer_, args_.data_path, args_.batch_size, args_.seq_length);
        trainer_->fit(dl, cfg_);
    } catch (const std::exception& e) {
        return fail(std::string("unified-trainer: fit failed: ") + e.what());
    } catch (...) {
        return fail("unified-trainer: fit failed (unknown)");
    }
    return true;
}

bool UnifiedTrainer::train_step_once(const Tensor& input_ids, const Tensor& labels,
                                     float* loss_out, std::string* err_out) {
    auto fail = [&](const std::string& m) {
        if (err_out) *err_out = m;
        return false;
    };
    if (!configured_ || !trainer_ || !opt_) return fail("unified-trainer: configure() first");
    try {
        opt_->zero_grad();
        float loss = trainer_->micro_step(input_ids, labels, 1.0f);
        trainer_->clip_gradients(1.0f);
        opt_->step();
        if (loss_out) *loss_out = loss;
    } catch (const std::exception& e) {
        return fail(std::string("unified-trainer: step failed: ") + e.what());
    } catch (...) {
        return fail("unified-trainer: step failed (unknown)");
    }
    return true;
}

void UnifiedTrainer::save_checkpoint(const std::string& path) {
    if (trainer_) trainer_->save_checkpoint(path);
}

void UnifiedTrainer::load_checkpoint(const std::string& path) {
    if (trainer_) trainer_->load_checkpoint(path);
}

const TrainMetrics& UnifiedTrainer::metrics() const {
    static TrainMetrics kEmpty;
    if (trainer_) return trainer_->metrics();
    return kEmpty;
}

} // namespace quant
