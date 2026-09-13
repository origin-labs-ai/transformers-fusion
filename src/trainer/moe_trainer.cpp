#define NOMINMAX
#include "quant/moe_trainer.h"
#include "quant/math.h"
#include "quant/autograd.h"
#include "quant/optimizer.h"
#include "quant/transformer.h"
#include "quant/generator.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <iomanip>
#include <sstream>
#ifdef _WIN32
#include <windows.h>
#endif

namespace quant {

// Triage fix: empty Tensor() positions leave rank-0 / null storage downstream
// (any positions.data<float>() / dim() use is UB). Canonical pattern
// (trainer_core::positions_from, finetune, generator) is real (B,S) float
// positions with pos[i] = i % S.
static Tensor moe_positions_from(const Tensor& input_ids) {
    if (input_ids.rank() < 2) return Tensor();
    int64_t B = input_ids.dim(0);
    int64_t S = input_ids.dim(1);
    if (B <= 0 || S <= 0) return Tensor();
    Tensor pos({B, S});
    float* pd = pos.data<float>();
    for (int64_t i = 0; i < B * S; ++i)
        pd[i] = (float)(i % S);
    return pos;
}

// ===========================================================================
// MoETrainer
// ===========================================================================

MoETrainer::MoETrainer(MoEModel* model, Tokenizer* tokenizer,
                       ExpertParallel* expert_parallel)
    : model_(model), tokenizer_(tokenizer), expert_parallel_(expert_parallel) {}

void MoETrainer::compile(Optimizer* optimizer, const MoETrainConfig& cfg) {
    optimizer_ = optimizer;
    config_ = cfg;
    step_ = 0;
    loss_scale_ = cfg.loss_scale;
    grad_accum_steps_ = cfg.gradient_accumulation_steps;
    collect_params();

    if (optimizer_ && !model_params_.empty()) {
        auto& engine = AutogradEngine::instance();
        for (auto* p : model_params_) {
            p->requires_grad(true);
            engine.register_parameter(p);
        }
        optimizer_->add_param_group(model_params_);
    }

    if (expert_parallel_ && cfg.use_expert_parallel) {
        if (!owns_expert_parallel_) {
            owns_expert_parallel_ = false;
        }
    }

    if (cfg.mixed_precision) {
        init_mixed_precision();
    }
}

void MoETrainer::compile(AdamW* optimizer, const MoETrainConfig& cfg) {
    compile(static_cast<Optimizer*>(optimizer), cfg);
    if (optimizer) {
        optimizer->set_schedule(cfg.schedule, cfg.warmup_steps, cfg.train_steps);
        optimizer->set_weight_decay(cfg.weight_decay);
    }
}

void MoETrainer::compile(Adafactor* optimizer, const MoETrainConfig& cfg) {
    compile(static_cast<Optimizer*>(optimizer), cfg);
    if (optimizer) {
        optimizer->set_lr(cfg.learning_rate);
        optimizer->set_weight_decay(cfg.weight_decay);
    }
}

void MoETrainer::compile(const MoETrainConfig& cfg) {
    auto* opt = new Adafactor(cfg.learning_rate, 0.999f, 1e-8f, cfg.weight_decay);
    default_opt_.reset(opt);
    compile(opt, cfg);
}

void MoETrainer::collect_params() {
    model_params_.clear();
    model_params_.push_back(&model_->tok_embeddings->weight);
    for (auto& l : model_->layers) {
        model_params_.push_back(&l.attention_norm.weight);
        model_params_.push_back(&l.attention.q_proj.weight);
        if (l.attention.q_proj.bias.numel() > 0)
            model_params_.push_back(&l.attention.q_proj.bias);
        model_params_.push_back(&l.attention.k_proj.weight);
        if (l.attention.k_proj.bias.numel() > 0)
            model_params_.push_back(&l.attention.k_proj.bias);
        model_params_.push_back(&l.attention.v_proj.weight);
        if (l.attention.v_proj.bias.numel() > 0)
            model_params_.push_back(&l.attention.v_proj.bias);
        model_params_.push_back(&l.attention.o_proj.weight);
        if (l.attention.o_proj.bias.numel() > 0)
            model_params_.push_back(&l.attention.o_proj.bias);
        model_params_.push_back(&l.ffn_norm.weight);
        model_params_.push_back(&l.moe->router_weight.weight);
        for (auto& e : l.moe->experts) {
            model_params_.push_back(&e.gate_proj.weight);
            model_params_.push_back(&e.up_proj.weight);
            model_params_.push_back(&e.down_proj.weight);
        }
        if (l.shared_expert) {
            model_params_.push_back(&l.shared_expert->gate_proj.weight);
            model_params_.push_back(&l.shared_expert->up_proj.weight);
            model_params_.push_back(&l.shared_expert->down_proj.weight);
        }
    }
    model_params_.push_back(&model_->norm->weight);
    model_params_.push_back(&model_->lm_head->weight);
    if (model_->lm_head->bias.numel() > 0)
        model_params_.push_back(&model_->lm_head->bias);

    // Registered parameters are trainable: ensure the autograd engine accumulates
    // gradients for them even if the model never set requires_grad explicitly.
    for (auto* p : model_params_) {
        p->requires_grad(true);
    }
}

void MoETrainer::fit(DataLoader& train_dl, const MoETrainConfig& cfg,
                     DataLoader* val_dl) {
    config_ = cfg;
    grad_accum_steps_ = cfg.gradient_accumulation_steps;
    loss_scale_ = cfg.loss_scale;

    int64_t total_tokens = 0;
    auto start_time = std::chrono::steady_clock::now();
    MoEMetrics last_log;

    for (int epoch = 0; epoch < cfg.num_epochs; epoch++) {
        if (verbose_) {
            std::cout << "\n=== Epoch " << (epoch + 1) << "/" << cfg.num_epochs << " ===" << std::endl;
        }
        train_dl.shuffle(epoch);
        int64_t batches_in_epoch = train_dl.num_batches();
        int64_t steps_in_epoch = std::min((int64_t)cfg.train_steps, batches_in_epoch);

        for (int64_t batch_idx = 0; batch_idx < steps_in_epoch; batch_idx++) {
            Tensor input_ids, labels;
            if (!train_dl.next_batch(input_ids, labels)) break;

            // W18 fix: grad accumulation uses fresh micro-batch per step via DataLoader.
            // Keep loss-scale correct: each micro_step receives loss_scale / acc_steps.
            float step_loss = 0.0f;
            if (grad_accum_steps_ > 1) {
                step_loss = train_step(train_dl, input_ids, labels);
                // train_step with loader already counts grad_accum_steps batches
                total_tokens += input_ids.numel() * grad_accum_steps_;
            } else {
                step_loss = train_step(input_ids, labels);
                total_tokens += input_ids.numel();
            }
            metrics_.tokens_processed = total_tokens;
            metrics_.loss = step_loss;
            metrics_.total_params = model_->stored_param_count();
            metrics_.active_params = model_->param_count();
            metrics_.step = step_;
            metrics_.epoch = epoch + 1;

            if (step_ % 100 == 0 && expert_parallel_) {
                metrics_.expert_utilization = compute_expert_utilization();
            }

            if (log_cb_ && (step_ % cfg.log_interval == 0)) {
                auto now = std::chrono::steady_clock::now();
                float elapsed = std::chrono::duration<float>(now - start_time).count();
                metrics_.tokens_per_sec = (elapsed > 0) ? (float)total_tokens / elapsed : 0;
                metrics_.grad_norm = 0;
                for (auto* p : model_params_) {
                    if (p->has_grad()) {
                        const float* gd = p->grad().data<float>();
                        int64_t n = p->numel();
                        float norm = 0;
                        for (int64_t i = 0; i < n; i++) norm += gd[i] * gd[i];
                        metrics_.grad_norm += norm;
                    }
                }
                metrics_.grad_norm = std::sqrt(metrics_.grad_norm);
                int64_t tokens_this_interval = metrics_.tokens_processed - last_log.tokens_processed;
                if (tokens_this_interval > 0) {
                    float interval_time = elapsed - (total_tokens > 0 ? elapsed * last_log.tokens_processed / total_tokens : 0);
                    metrics_.tokens_per_sec = interval_time > 0 ? tokens_this_interval / interval_time : 0;
                }
                log_cb_(metrics_);
                last_log = metrics_;
            }

            if (val_dl && (step_ % cfg.val_interval == 0) && step_ > 0) {
                float val_loss = eval_loss(*val_dl);
                metrics_.perplexity = std::exp(val_loss);
                if (verbose_) {
                    std::cout << "  Validation loss: " << val_loss
                              << ", perplexity: " << metrics_.perplexity << std::endl;
                }
            }

            if (step_ % cfg.save_interval == 0 && step_ > 0) {
                save_checkpoint(cfg.output_path);
            }
        }
    }

    if (epoch_cb_) {
        epoch_cb_(cfg.num_epochs, metrics_);
    }

    save_checkpoint(cfg.output_path);
}

float MoETrainer::train_step(const Tensor& input_ids, const Tensor& labels) {
    float total_loss = 0;

    optimizer_->zero_grad();
    reset_expert_stats();

    for (int micro = 0; micro < grad_accum_steps_; micro++) {
        float ls = loss_scale_ / (float)std::max(1, grad_accum_steps_);
        float micro_loss = micro_step(input_ids, labels, ls);
        total_loss += micro_loss;
    }

    if (config_.mixed_precision) {
        unscale_gradients(loss_scale_);
    }

    float grad_norm = clip_gradients(config_.max_grad_norm);
    metrics_.grad_norm = grad_norm;

    if (config_.mixed_precision && grad_norm > config_.max_grad_norm * 2.0f) {
        loss_scale_ = std::max(1.0f, loss_scale_ / 2.0f);
    } else if (config_.mixed_precision) {
        loss_scale_ = std::min(1e8f, loss_scale_ * 1.01f);
    }

    if (expert_parallel_ && config_.use_expert_parallel) {
        expert_parallel_->sync_gradients(model_);
        for (auto& l : model_->layers) {
            for (auto& e : l.moe->experts) {
                if (e.gate_proj.weight.has_grad()) {
                    int64_t n = e.gate_proj.weight.numel();
                    float* gd = e.gate_proj.weight.grad().data<float>();
                    for (int64_t i = 0; i < n; i++) gd[i] /= (float)config_.num_expert_parallel_ranks;
                }
            }
        }
    }

    optimizer_->step();
    metrics_.learning_rate = config_.learning_rate;

    if (ema_enabled_) {
        ema_step();
    }

    // A direct train_step() call counts as one completed training step; fit()
    // relies on this counter for logging/checkpoint cadence.
    step_++;
    metrics_.step = step_;

    return total_loss;
}

float MoETrainer::train_step(DataLoader& loader, const Tensor& first_input, const Tensor& first_labels) {
    float total_loss = 0;
    optimizer_->zero_grad();
    reset_expert_stats();
    Tensor cur_input = first_input;
    Tensor cur_labels = first_labels;
    for (int micro = 0; micro < grad_accum_steps_; micro++) {
        if (micro > 0) {
            // next_batch writes into caller-owned storage (dense-trainer
            // convention, see trainer_data.cpp prefetch_worker) — a default
            // Tensor has a null buffer and segfaults. Allocate like first.
            Tensor nxt_input(first_input.shape(), first_input.dtype());
            Tensor nxt_labels(first_labels.shape(), first_labels.dtype());
            if (!loader.next_batch(nxt_input, nxt_labels)) break;
            cur_input = nxt_input;
            cur_labels = nxt_labels;
        }
        float ls = loss_scale_ / (float)std::max(1, grad_accum_steps_);
        float micro_loss = micro_step(cur_input, cur_labels, ls);
        total_loss += micro_loss;
    }
    if (config_.mixed_precision) unscale_gradients(loss_scale_);
    float grad_norm = clip_gradients(config_.max_grad_norm);
    metrics_.grad_norm = grad_norm;
    if (config_.mixed_precision && grad_norm > config_.max_grad_norm * 2.0f) {
        loss_scale_ = std::max(1.0f, loss_scale_ / 2.0f);
    } else if (config_.mixed_precision) {
        loss_scale_ = std::min(1e8f, loss_scale_ * 1.01f);
    }
    if (expert_parallel_ && config_.use_expert_parallel) {
        expert_parallel_->sync_gradients(model_);
        for (auto& l : model_->layers) {
            for (auto& e : l.moe->experts) {
                if (e.gate_proj.weight.has_grad()) {
                    int64_t n = e.gate_proj.weight.numel();
                    float* gd = e.gate_proj.weight.grad().data<float>();
                    for (int64_t i = 0; i < n; i++) gd[i] /= (float)config_.num_expert_parallel_ranks;
                }
            }
        }
    }
    optimizer_->step();
    metrics_.learning_rate = config_.learning_rate;
    if (ema_enabled_) ema_step();
    step_++;
    metrics_.step = step_;
    return total_loss;
}

float MoETrainer::micro_step(const Tensor& input_ids, const Tensor& labels,
                             float loss_scale) {
    AutogradEngine::set_enabled(true);
    auto& engine = AutogradEngine::instance();
    for (auto* p : model_params_) {
        engine.register_parameter(p);
    }

    Tensor logits = model_->forward(input_ids, moe_positions_from(input_ids), nullptr);

    float load_balance = 0;
    float zloss = 0;
    int64_t dropped = 0;
    for (auto& l : model_->layers) {
        load_balance += l.load_balance_loss;
        zloss += l.z_loss;
        dropped += l.tokens_dropped;
    }
    metrics_.load_balance_loss = model_->layers.empty() ? 0.0f : load_balance / (float)model_->layers.size();
    metrics_.z_loss = model_->layers.empty() ? 0.0f : zloss / (float)model_->layers.size();
    metrics_.tokens_dropped_total = dropped;

    Tensor loss = AutogradEngine::cross_entropy_op(logits, labels);
    float ce_loss = *(const float*)loss.data();

    float aux_loss = 0;
    if (config_.aux_loss_coef > 0 && !model_->layers.empty()) {
        double sum_aux = 0.0;
        int cnt = 0;
        for (auto& layer : model_->layers) {
            if (layer.last_router_logits.numel() > 0 && layer.last_expert_indices.numel() > 0) {
                sum_aux += compute_aux_loss(layer.last_router_logits, layer.last_expert_indices);
                cnt++;
            }
        }
        if (cnt > 0) aux_loss = (float)(sum_aux / (double)cnt);
    }
    metrics_.aux_loss = aux_loss;

    float total_loss_val = ce_loss
        + config_.load_balance_coef * metrics_.load_balance_loss
        + config_.z_loss_coef * metrics_.z_loss
        + config_.aux_loss_coef * aux_loss;
    metrics_.total_loss = total_loss_val;
    metrics_.loss = ce_loss;
    metrics_.perplexity = std::exp(std::min(ce_loss, 20.0f));

    // Apply loss scaling via autograd mul, not scalar overwrite, to preserve graph
    Tensor scaled_loss = loss;
    if (loss_scale != 1.0f || total_loss_val != ce_loss) {
        float scale_factor = loss_scale;
        if (ce_loss > 1e-8f && total_loss_val != ce_loss) {
            scale_factor *= (total_loss_val / ce_loss);
        } else if (total_loss_val != ce_loss) {
            // ce near zero, add small epsilon factor
            scale_factor *= 1.0f;
        }
        if (scale_factor != 1.0f) {
            Tensor scale({1});
            scale.data<float>()[0] = scale_factor;
            scaled_loss = AutogradEngine::mul_op(loss, scale);
        }
    }
    engine.backward(scaled_loss);
    engine.clear();
    AutogradEngine::set_enabled(false);

    return total_loss_val;
}

float MoETrainer::eval_loss(DataLoader& val_dl, int64_t max_batches) {
    float total_loss = 0;
    int64_t count = 0;

    for (int64_t i = 0; i < max_batches; i++) {
        Tensor input_ids, labels;
        if (!val_dl.next_batch(input_ids, labels)) break;

        Tensor logits = model_->forward(input_ids, moe_positions_from(input_ids), nullptr);
        float batch_loss = 0;
        const float* ld = logits.data<float>();
        int64_t B = input_ids.dim(0);
        int64_t S = input_ids.dim(1);
        int64_t V = model_->config.vocab_size;

        for (int64_t b = 0; b < B; b++) {
            for (int64_t s = 0; s < S - 1; s++) {
                int label = (int)labels.data<float>()[b * S + s + 1];
                if (label < 0 || label >= V) continue;
                float max_logit = -1e10f;
                for (int64_t v = 0; v < V; v++)
                    max_logit = std::max(max_logit, ld[b * S * V + s * V + v]);
                float sum_exp = 0;
                for (int64_t v = 0; v < V; v++)
                    sum_exp += std::exp(ld[b * S * V + s * V + v] - max_logit);
                float log_sum = max_logit + std::log(sum_exp);
                batch_loss += log_sum - ld[b * S * V + s * V + label];
                count++;
            }
        }
        total_loss += batch_loss;
    }

    return (count > 0) ? total_loss / (float)count : 0;
}

float MoETrainer::clip_gradients(float max_norm) {
    if (max_norm <= 0) return 0;

    float total_norm = 0;
    for (auto* p : model_params_) {
        if (p->has_grad()) {
            const float* gd = p->grad().data<float>();
            int64_t n = p->numel();
            float norm = 0;
            for (int64_t i = 0; i < n; i++) norm += gd[i] * gd[i];
            total_norm += norm;
        }
    }
    total_norm = std::sqrt(total_norm);

    if (total_norm > max_norm) {
        float scale = max_norm / (total_norm + 1e-8f);
        for (auto* p : model_params_) {
            if (p->has_grad()) {
                float* gd = p->grad().data<float>();
                int64_t n = p->numel();
                for (int64_t i = 0; i < n; i++) gd[i] *= scale;
            }
        }
    }

    return total_norm;
}

void MoETrainer::unscale_gradients(float scale) {
    if (scale <= 0 || scale == 1.0f) return;
    float inv_scale = 1.0f / scale;
    for (auto* p : model_params_) {
        if (p->has_grad()) {
            float* gd = p->grad().data<float>();
            int64_t n = p->numel();
            for (int64_t i = 0; i < n; i++) gd[i] *= inv_scale;
        }
    }
}

void MoETrainer::save_checkpoint(const std::string& path) {
    model_->save(path);
    if (verbose_) {
        std::cout << "  Checkpoint saved: " << path
                  << " (step " << step_ << ")" << std::endl;
    }
}

void MoETrainer::load_checkpoint(const std::string& path) {
    model_->load(path);
    if (verbose_) {
        std::cout << "  Checkpoint loaded: " << path << std::endl;
    }
}

Tensor MoETrainer::compute_loss(const Tensor& logits, const Tensor& labels) {
    int64_t B = logits.dim(0);
    int64_t S = logits.dim(1);
    int64_t V = logits.dim(2);
    Tensor loss({B, S - 1});
    float* ld = loss.data<float>();
    const float* lp = logits.data<float>();
    const float* lb = labels.data<float>();

    for (int64_t b = 0; b < B; b++) {
        for (int64_t s = 0; s < S - 1; s++) {
            int label = (int)lb[b * S + s + 1];
            if (label < 0 || label >= (int)V) {
                ld[b * (S - 1) + s] = 0;
                continue;
            }
            float max_logit = -1e10f;
            for (int64_t v = 0; v < V; v++)
                max_logit = std::max(max_logit, lp[b * S * V + s * V + v]);
            float sum_exp = 0;
            for (int64_t v = 0; v < V; v++)
                sum_exp += std::exp(lp[b * S * V + s * V + v] - max_logit);
            ld[b * (S - 1) + s] = -(lp[b * S * V + s * V + label] - max_logit - std::log(sum_exp));
        }
    }
    return loss;
}

float MoETrainer::compute_expert_utilization() const {
    if (model_->layers.empty()) return 0;
    // Fraction of experts with nonzero gate mass, averaged over layers.
    // (Was double-counting: layer_util was derived from the already
    // accumulated total_util and added back, yielding e.g. 5.0 for a
    // 1-layer/4-expert model instead of <= 1.)
    float total_util = 0;
    int counted_layers = 0;
    for (auto& l : model_->layers) {
        if (!l.moe) continue;
        int num_experts = (int)l.moe->experts.size();
        if (num_experts == 0) continue;
        int active = 0;
        for (int e = 0; e < num_experts; e++) {
            float* wd = l.moe->experts[e].gate_proj.weight.data<float>();
            int64_t n = l.moe->experts[e].gate_proj.weight.numel();
            float norm = 0;
            for (int64_t i = 0; i < n; i++) norm += wd[i] * wd[i];
            if (norm > 1e-6f) active++;
        }
        total_util += (float)active / (float)num_experts;
        counted_layers++;
    }
    if (counted_layers == 0) return 0;
    return total_util / (float)counted_layers;
}

float MoETrainer::compute_aux_loss(const Tensor& router_logits,
                                    const Tensor& expert_indices) const {
    // P15: canonical Switch/GShard aux loss ka single source
    // moe::compute_load_balance_loss hai (moe_variants.cpp:141-174) —
    // duplicate softmax/f_i logic yahan nahi rakhte taaki divergence na ho.
    // Caller (micro_step) real last_router_logits / last_expert_indices deta hai;
    // dummy {1,1} path hata hua hai.
    if (router_logits.numel() == 0 || expert_indices.numel() == 0) return 0.0f;

    const int64_t n_tokens = router_logits.dim(0);
    const int64_t n_experts = router_logits.dim(1);
    if (n_experts <= 0 || n_tokens <= 0) return 0.0f;

    if (expert_indices.rank() >= 2) {
        return moe::compute_load_balance_loss(router_logits, expert_indices, n_experts);
    }
    // 1-D indices (flat token list) ko {N,1} view dekar same canonical path use karo.
    // reshape const view hai, data copy nahi hota.
    Tensor idx2 = expert_indices.reshape({expert_indices.numel(), 1});
    return moe::compute_load_balance_loss(router_logits, idx2, n_experts);
}

const std::vector<Tensor*>& MoETrainer::get_model_params() const {
    return model_params_;
}

void MoETrainer::set_log_callback(LogCallback cb) { log_cb_ = cb; }
void MoETrainer::set_epoch_callback(EpochCallback cb) { epoch_cb_ = cb; }
void MoETrainer::set_step_callback(StepCallback cb) { step_cb_ = cb; }

void MoETrainer::reset_expert_stats() {
    metrics_.load_balance_loss = 0;
    metrics_.z_loss = 0;
    metrics_.aux_loss = 0;
    metrics_.tokens_dropped_total = 0;
}

void MoETrainer::accumulate_expert_stats(const moe::MoEOutput& moe_out) {
    metrics_.load_balance_loss += moe_out.load_balance_loss;
    metrics_.z_loss += moe_out.z_loss;
    metrics_.num_experts_used = (int)moe_out.num_activated_experts;
    // L057: overflow accounting lives in tokens_dropped_total (int64); the
    // old tokens_per_sec misuse is removed.
    metrics_.tokens_dropped_total += moe_out.tokens_dropped;
}

void MoETrainer::ema_init(float decay) {
    ema_enabled_ = true;
    ema_decay_ = decay;
    ema_params_.clear();
    for (auto* p : model_params_)
        ema_params_.push_back(p->clone());
}

void MoETrainer::ema_step() {
    if (!ema_enabled_) return;
    for (size_t i = 0; i < model_params_.size(); i++) {
        float* e = ema_params_[i].data<float>();
        const float* p = model_params_[i]->data<float>();
        int64_t n = model_params_[i]->numel();
        for (int64_t j = 0; j < n; j++)
            e[j] = ema_decay_ * e[j] + (1.0f - ema_decay_) * p[j];
    }
}

void MoETrainer::ema_apply() {
    if (!ema_enabled_) return;
    for (size_t i = 0; i < model_params_.size(); i++)
        std::memcpy(model_params_[i]->data<float>(), ema_params_[i].data<float>(),
                    model_params_[i]->numel() * sizeof(float));
}

void MoETrainer::ema_swap() {
    if (!ema_enabled_) return;
    for (size_t i = 0; i < model_params_.size(); i++) {
        float* p = model_params_[i]->data<float>();
        float* e = ema_params_[i].data<float>();
        int64_t n = model_params_[i]->numel();
        for (int64_t j = 0; j < n; j++) std::swap(p[j], e[j]);
    }
}

void MoETrainer::init_mixed_precision() {
    if (mp_active_) return;
    mp_master_.clear();
    for (auto* p : model_params_) {
        Tensor master = p->clone();
        mp_master_.push_back(master);
    }
    mp_active_ = true;
}

void MoETrainer::mp_quantize_forward() {
    if (!mp_active_) return;
    for (size_t i = 0; i < model_params_.size(); i++) {
        const float* src = mp_master_[i].data<float>();
        float* dst = model_params_[i]->data<float>();
        int64_t n = model_params_[i]->numel();
        std::memcpy(dst, src, n * sizeof(float));
    }
}

void MoETrainer::mp_restore_master() {
    if (!mp_active_) return;
    for (size_t i = 0; i < model_params_.size(); i++) {
        const float* src = model_params_[i]->data<float>();
        float* dst = mp_master_[i].data<float>();
        int64_t n = model_params_[i]->numel();
        for (int64_t j = 0; j < n; j++)
            dst[j] = dst[j] * 0.999f + src[j] * 0.001f;
    }
}

// ===========================================================================
// MoETrainingPipeline
// ===========================================================================

MoETrainingPipeline::MoETrainingPipeline(MoEModel* model, Tokenizer* tokenizer,
                                          const std::string& train_path,
                                          const std::string& val_path,
                                          ExpertParallel* ep)
    : model_(model), tokenizer_(tokenizer),
      train_path_(train_path), val_path_(val_path) {
    if (ep) {
        ep_ = std::make_unique<ExpertParallel>(ep->num_experts(), ep->num_ranks(), ep->rank());
    }
    trainer_ = std::make_unique<MoETrainer>(model_, tokenizer_, ep_.get());
    checkpoint_dir_ = "checkpoints";
}

void MoETrainingPipeline::configure(const MoETrainConfig& cfg) {
    config_ = cfg;
    world_size_ = cfg.num_expert_parallel_ranks;
    rank_ = cfg.expert_parallel_rank;

    float lr = cfg.learning_rate;
    float wd = cfg.weight_decay;

    auto adafactor = std::make_unique<Adafactor>(lr, 0.999f, 1e-8f, wd);
    optimizer_ = std::move(adafactor);

    init_data();
    trainer_->compile(optimizer_.get(), cfg);
}

void MoETrainingPipeline::init_data() {
    if (!tokenizer_ || train_path_.empty()) return;

    train_dl_ = std::make_unique<DataLoader>(tokenizer_, train_path_,
                                              config_.batch_size,
                                              config_.seq_length,
                                              true);

    if (!val_path_.empty()) {
        val_dl_ = std::make_unique<DataLoader>(tokenizer_, val_path_,
                                                config_.batch_size,
                                                config_.seq_length,
                                                true);
    }

    if (!log_path_.empty()) {
        log_stream_.open(log_path_, std::ios::app);
    }
}

void MoETrainingPipeline::run(int epochs) {
    if (epochs < 0) epochs = config_.num_epochs;

    trainer_->set_log_callback([this](const MoEMetrics& m) {
        std::stringstream ss;
        ss << "Step " << m.step
           << " | loss: " << std::fixed << std::setprecision(4) << m.loss
           << " | ppl: " << std::setprecision(2) << m.perplexity
           << " | aux_loss: " << std::setprecision(6) << m.aux_loss
           << " | grad_norm: " << std::setprecision(4) << m.grad_norm
           << " | lr: " << std::setprecision(6) << m.learning_rate
           << " | util: " << std::setprecision(2) << (m.expert_utilization * 100) << "%";
        if (m.load_balance_loss > 0)
            ss << " | lb: " << std::setprecision(6) << m.load_balance_loss;
        if (m.z_loss > 0)
            ss << " | z: " << std::setprecision(6) << m.z_loss;

        std::string log_str = ss.str();
        std::cout << "  " << log_str << std::endl;
        if (log_stream_.is_open()) {
            log_stream_ << log_str << std::endl;
        }
    });

    trainer_->set_epoch_callback([this](int epoch, const MoEMetrics& m) {
        std::stringstream ss;
        ss << "Epoch " << epoch << " complete | avg_loss: " << m.loss
           << " | ppl: " << m.perplexity;
        std::string log_str = ss.str();
        std::cout << log_str << std::endl;
        if (log_stream_.is_open()) {
            log_stream_ << log_str << std::endl;
        }
    });

    val_dl_.reset();
    if (!val_path_.empty()) {
        val_dl_ = std::make_unique<DataLoader>(tokenizer_, val_path_,
                                                config_.batch_size,
                                                config_.seq_length,
                                                true);
    }

    MoETrainConfig run_cfg = config_;
    run_cfg.num_epochs = epochs;
    run_cfg.train_steps = config_.train_steps / std::max(1, epochs);

    if (val_dl_) {
        trainer_->fit(*train_dl_, run_cfg, val_dl_.get());
    } else {
        trainer_->fit(*train_dl_, run_cfg);
    }

    save_checkpoint(epochs, trainer_->metrics().loss);

    if (log_stream_.is_open()) {
        log_stream_.close();
    }
}

void MoETrainingPipeline::run_epoch(int epoch) {
    if (!trainer_) return;

    int64_t total_tokens = 0;
    float total_loss = 0.0f;
    int step_count = 0;

    if (train_dl_) {
        Tensor input_ids, labels;
        while (train_dl_->next_batch(input_ids, labels)) {
            float loss = trainer_->train_step(input_ids, labels);
            total_loss += loss;
            step_count++;
            total_tokens += input_ids.numel();
        }
    }

    const auto& m = trainer_->metrics();

    // Log epoch to file if open
    if (log_stream_.is_open()) {
        log_stream_ << "Epoch " << epoch
                    << " | loss=" << m.loss
                    << " | perplexity=" << m.perplexity
                    << " | tokens=" << total_tokens
                    << " | steps=" << step_count << "\n";
        log_stream_.flush();
    }

    // Save checkpoint
    if (!checkpoint_dir_.empty()) {
        save_checkpoint(epoch, m.loss);
    }
}

void MoETrainingPipeline::save_checkpoint(int epoch, float loss) {
    std::string path = checkpoint_dir_ + "/moe_epoch_" + std::to_string(epoch) + ".quant";
    model_->save(path);

    std::string metadata_path = checkpoint_dir_ + "/moe_epoch_" + std::to_string(epoch) + ".json";
    std::ofstream meta(metadata_path);
    if (meta.is_open()) {
        meta << "{\n"
             << "  \"epoch\": " << epoch << ",\n"
             << "  \"loss\": " << loss << ",\n"
             << "  \"step\": " << trainer_->metrics().step << ",\n"
             << "  \"perplexity\": " << trainer_->metrics().perplexity << ",\n"
             << "  \"hidden_size\": " << model_->config.hidden_size << ",\n"
             << "  \"num_layers\": " << model_->config.num_layers << ",\n"
             << "  \"num_experts\": " << model_->moe_config.num_experts << ",\n"
             << "  \"top_k\": " << model_->moe_config.top_k << ",\n"
             << "  \"total_params\": " << model_->stored_param_count() << ",\n"
             << "  \"active_params\": " << model_->param_count() << "\n"
             << "}\n";
    }
}

bool MoETrainingPipeline::should_stop_early(const MoEMetrics& m, int patience) {
    if (m.loss < best_val_loss_) {
        best_val_loss_ = m.loss;
        epochs_no_improve_ = 0;
        return false;
    }
    epochs_no_improve_++;
    return epochs_no_improve_ >= patience;
}

void MoETrainingPipeline::set_log_file(const std::string& path) {
    log_path_ = path;
    if (!log_path_.empty()) {
        log_stream_.open(log_path_, std::ios::app);
    }
}

void MoETrainingPipeline::set_checkpoint_dir(const std::string& dir) {
    checkpoint_dir_ = dir;
}

void MoETrainingPipeline::set_distributed(int world_size, int rank) {
    world_size_ = world_size;
    rank_ = rank;
    config_.num_expert_parallel_ranks = world_size;
    config_.expert_parallel_rank = rank;
}

const MoEMetrics& MoETrainingPipeline::last_metrics() const {
    return trainer_->metrics();
}

std::string MoETrainingPipeline::status_report() const {
    std::stringstream ss;
    const auto& m = trainer_->metrics();
    ss << "MoE Training Report\n"
       << "===================\n"
       << "Model: " << model_->stored_param_count() << " total params, "
       << model_->param_count() << " active per token\n"
       << "Layers: " << model_->config.num_layers << "\n"
       << "Experts: " << model_->moe_config.num_experts
       << " (top-" << model_->moe_config.top_k << ")\n"
       << "Hidden: " << model_->config.hidden_size << "\n"
       << "Steps: " << m.step << "\n"
       << "Loss: " << m.loss << "\n"
       << "Perplexity: " << m.perplexity << "\n"
       << "Grad Norm: " << m.grad_norm << "\n"
       << "LR: " << m.learning_rate << "\n"
       << "Expert Utilization: " << (m.expert_utilization * 100) << "%\n";
    return ss.str();
}

} // namespace quant
