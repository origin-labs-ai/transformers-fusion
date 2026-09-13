#include "quant/native_trainer.h"
#include "quant/tensor.h"
#include "quant/optimizer.h"
#include "quant/autograd.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <memory>

namespace quant {
namespace native {

NativeQUANTTrainer::NativeQUANTTrainer(DenseModel* model, const NativeTrainConfig& cfg)
    : model_(model), cfg_(cfg) {
    quant::collect_dense_params(model_, model_params_);
    total_params_ = 0;
    for (auto* p : model_params_) total_params_ += (size_t)p->numel();

    auto& engine = AutogradEngine::instance();
    for (auto* p : model_params_) {
        p->requires_grad(true);
        engine.register_parameter(p);
    }

    weight_store_ = std::make_unique<NativeQUANTWeightStore>(total_params_, cfg_.block_size);
    grad_buffer_ = std::make_unique<float[]>(total_params_);
    sensitivity_ = std::make_unique<float[]>(total_params_);
    temp_weight_buffer_ = std::make_unique<float[]>(total_params_);
    std::memset(grad_buffer_.get(), 0, total_params_ * sizeof(float));
    std::memset(sensitivity_.get(), 0, total_params_ * sizeof(float));
    std::memset(temp_weight_buffer_.get(), 0, total_params_ * sizeof(float));
    size_t num_blocks = weight_store_->num_blocks();
    scale_m_ = std::make_unique<float[]>(num_blocks);
    scale_v_ = std::make_unique<float[]>(num_blocks);
    std::memset(scale_m_.get(), 0, num_blocks * sizeof(float));
    std::memset(scale_v_.get(), 0, num_blocks * sizeof(float));
    sens_step_ = 0;
    step_ = 0;
    best_eval_loss_ = 1e30;
}

void NativeQUANTTrainer::push_weights_to_model() {
    weight_store_->dequantize(temp_weight_buffer_.get());
    size_t offset = 0;
    for (auto* p : model_params_) {
        size_t n = (size_t)p->numel();
        std::memcpy(p->data<float>(), temp_weight_buffer_.get() + offset, n * sizeof(float));
        offset += n;
    }
}

void NativeQUANTTrainer::pull_gradients_from_model() {
    size_t offset = 0;
    for (auto* p : model_params_) {
        size_t n = (size_t)p->numel();
        if (p->has_grad()) {
            const float* g = p->grad().data<float>();
            for (size_t i = 0; i < n; i++) {
                grad_buffer_[offset + i] += g[i];
            }
        }
        offset += n;
    }
}

void NativeQUANTTrainer::update_sensitivity() {
    float alpha = 2.0f / (float)(cfg_.sens_window + 1);
    for (size_t i = 0; i < total_params_; i++) {
        float g_abs = std::abs(grad_buffer_[i]);
        sensitivity_[i] = (1.0f - alpha) * sensitivity_[i] + alpha * g_abs;
        if (sensitivity_[i] < 1e-12f) sensitivity_[i] = 1e-12f;
    }
    sens_step_++;
}

void NativeQUANTTrainer::allocate_formats() {
    weight_store_->reallocate_by_sensitivity(sensitivity_.get(),
                                              cfg_.frac_quant8,
                                              cfg_.frac_quant);
    weight_store_->dequantize(temp_weight_buffer_.get());
    weight_store_->convert_from_fp32(temp_weight_buffer_.get());
}

void NativeQUANTTrainer::apply_two_timescale_sgd() {
    float lr_s = cfg_.lr_scale;
    float lr_w = cfg_.lr_weight;
    if (cfg_.lr_decay > 0.0f) {
        lr_s *= (1.0f / (1.0f + cfg_.lr_decay * (float)step_));
        lr_w *= (1.0f / (1.0f + cfg_.lr_decay * (float)step_));
    }
    size_t num_blocks = weight_store_->num_blocks();
    // Adam step for each block's scale
    for (size_t b = 0; b < num_blocks; b++) {
        size_t start = b * cfg_.block_size;
        size_t end = std::min(total_params_, start + cfg_.block_size);
        float grad_scale_sum = 0.0f;
        for (size_t i = start; i < end; i++) {
            float cv;
            NativeFormat fmt = weight_store_->get_format(i);
            uint8_t idx = weight_store_->get_index(i);
            switch (fmt) {
                case NativeFormat::QUANT1: cv = quant1_value(idx); break;
                case NativeFormat::QUANT4:    cv = NativeQUANTWeightStore::global_quant4_codebook().centroid(idx); break;
                case NativeFormat::QUANT8:    cv = NativeQUANTWeightStore::global_codebook().centroid(idx); break;
                default: cv = 0.0f;
            }
            grad_scale_sum += grad_buffer_[i] * cv;
        }
        float m = scale_m_[b];
        float v = scale_v_[b];
        m = cfg_.beta1 * m + (1.0f - cfg_.beta1) * grad_scale_sum;
        v = cfg_.beta2 * v + (1.0f - cfg_.beta2) * grad_scale_sum * grad_scale_sum;
        scale_m_[b] = m;
        scale_v_[b] = v;
        float m_hat = m / (1.0f - std::pow(cfg_.beta1, (float)step_ + 1));
        float v_hat = v / (1.0f - std::pow(cfg_.beta2, (float)step_ + 1));
        float lr_adjusted = lr_s / (float)cfg_.block_size;
        float update_s = -lr_adjusted * m_hat / (std::sqrt(v_hat) + cfg_.epsilon);
        update_s -= lr_adjusted * cfg_.weight_decay * weight_store_->get_scale(start);
        float new_scale = weight_store_->get_scale(start) + update_s;
        if (new_scale < 1e-10f) new_scale = 1e-10f;
        weight_store_->block_scales_data()[b] = new_scale;
    }
    // Apply index update only (Theorem 5d.3) — scale already updated via Adam above
    // Pass lr_scale=0 so apply_quant_update skips its own scale update
    weight_store_->apply_quant_update(grad_buffer_.get(), 0.0f, lr_w);
}

NativeTrainMetrics NativeQUANTTrainer::train_step(const float* input, const float* target,
                                                  size_t batch_size, size_t seq_len) {
    NativeTrainMetrics metrics;
    push_weights_to_model();
    AutogradEngine::set_enabled(true);
    Tensor inp(Shape{(int64_t)batch_size, (int64_t)seq_len});
    std::memcpy(inp.data<float>(), input, (size_t)(batch_size * seq_len) * sizeof(float));
    Tensor tgt(Shape{(int64_t)batch_size, (int64_t)seq_len});
    std::memcpy(tgt.data<float>(), target, (size_t)(batch_size * seq_len) * sizeof(float));
    Tensor pos(Shape{(int64_t)batch_size, (int64_t)seq_len});
    for (int64_t i = 0; i < (int64_t)(batch_size * seq_len); i++)
        pos.data<float>()[i] = (float)(i % (int64_t)seq_len);
    Tensor logits = model_->forward(inp, pos);
    Tensor loss_t = AutogradEngine::cross_entropy_op(logits, tgt);
    // Reset parameter gradients so backward() accumulates only this step's
    // gradients (AutogradEngine::clear() preserves registered parameters).
    for (auto* p : model_params_)
        if (p->has_grad()) p->zero_grad();
    auto& engine = AutogradEngine::instance();
    engine.backward(loss_t);
    engine.clear();
    AutogradEngine::set_enabled(false);
    std::memset(grad_buffer_.get(), 0, total_params_ * sizeof(float));
    pull_gradients_from_model();
    update_sensitivity();
    apply_two_timescale_sgd();
    step_++;
    metrics.loss = *(const float*)loss_t.data();
    metrics.scale_lr = cfg_.lr_scale;
    metrics.weight_lr = cfg_.lr_weight;
    float grad_norm = 0.0f;
    for (size_t i = 0; i < total_params_; i++) grad_norm += grad_buffer_[i] * grad_buffer_[i];
    metrics.grad_norm = std::sqrt(grad_norm / (float)total_params_);
    size_t frozen = 0, quant8 = 0, quant = 0, quant1 = 0;
    double avg_s = 0.0;
    for (size_t i = 0; i < total_params_; i++) {
        NativeFormat fmt = weight_store_->get_format(i);
        if (fmt == NativeFormat::QUANT8) quant8++;
        else if (fmt == NativeFormat::QUANT1) quant1++;
        else quant++;
        float dz = dead_zone_radius(fmt, weight_store_->get_scale(i));
        if (std::abs(grad_buffer_[i]) * cfg_.lr_weight < dz) frozen++;
        avg_s += weight_store_->get_scale(i);
    }
    metrics.frozen_fraction = (double)frozen / (double)std::max((size_t)1, total_params_);
    metrics.quant8_count = quant8;
    metrics.quant1_count = quant1;
    metrics.quant_count = quant;
    metrics.avg_scale = avg_s / (double)total_params_;
    return metrics;
}

void NativeQUANTTrainer::warmup_phase(const std::vector<std::vector<float>>& data) {
    if (data.empty()) return;
    std::cout << "[NativeQUANT] Warmup: " << cfg_.warmup_steps << " FP32 steps\n";
    size_t offset = 0;
    for (auto* p : model_params_) {
        size_t n = (size_t)p->numel();
        std::memcpy(temp_weight_buffer_.get() + offset, p->data<float>(), n * sizeof(float));
        offset += n;
    }
    weight_store_->initialize(temp_weight_buffer_.get(), sensitivity_.get(),
                               cfg_.frac_quant8, cfg_.frac_quant);
    for (size_t step = 0; step < cfg_.warmup_steps; step++) {
        size_t idx = step % data.size();
        std::vector<float> seq = data[idx];
        if (seq.empty()) continue;
        push_weights_to_model();
        AutogradEngine::set_enabled(true);
        Tensor inp(Shape{1, (int64_t)seq.size()});
        std::memcpy(inp.data<float>(), seq.data(), seq.size() * sizeof(float));
        Tensor tgt(Shape{1, (int64_t)seq.size()});
        std::memcpy(tgt.data<float>(), seq.data(), seq.size() * sizeof(float));
        Tensor pos(Shape{1, (int64_t)seq.size()});
        for (int64_t i = 0; i < (int64_t)seq.size(); i++) pos.data<float>()[i] = (float)i;
        Tensor logits = model_->forward(inp, pos);
        Tensor loss = AutogradEngine::cross_entropy_op(logits, tgt);
        for (auto* p : model_params_)
            if (p->has_grad()) p->zero_grad();
        auto& engine = AutogradEngine::instance();
        engine.backward(loss);
        engine.clear();
        AutogradEngine::set_enabled(false);
        std::memset(grad_buffer_.get(), 0, total_params_ * sizeof(float));
        pull_gradients_from_model();
        update_sensitivity();
        push_weights_to_model();
        if ((step + 1) % 10 == 0) {
            std::cout << "  Step " << (step + 1) << "/" << cfg_.warmup_steps
                      << " loss=" << *(const float*)loss.data() << "\n";
        }
    }
    allocate_formats();
    std::cout << "[NativeQUANT] CID: QUANT8=" << (cfg_.frac_quant8 * 100.0f)
              << "% T=" << (cfg_.frac_quant * 100.0f)
              << "% B=" << ((1.0f - cfg_.frac_quant8 - cfg_.frac_quant) * 100.0f) << "%\n";
}

void NativeQUANTTrainer::train(const std::vector<std::vector<float>>& train_data,
                              const std::vector<std::vector<float>>& eval_data) {
    if (train_data.empty()) return;
    warmup_phase(train_data);
    std::cout << "\n[NativeQUANT] Training (" << cfg_.max_steps << " steps)\n";
    for (size_t step = 0; step < cfg_.max_steps; step++) {
        size_t idx = step % train_data.size();
        std::vector<float> seq = train_data[idx];
        NativeTrainMetrics m = train_step(seq.data(), seq.data(), 1, seq.size());
        if ((step + 1) % cfg_.log_interval == 0) {
            std::cout << "Step " << (step + 1) << "/" << cfg_.max_steps
                      << " loss=" << m.loss
                      << " |g|=" << m.grad_norm
                      << " frozen=" << (m.frozen_fraction * 100.0f) << "%"
                      << " O8=" << m.quant8_count << " SP=" << m.quant_count
                      << " O1=" << m.quant1_count << " s=" << m.avg_scale << "\n";
        }
        if ((step + 1) % cfg_.eval_interval == 0 && !eval_data.empty()) {
            double eval_loss = evaluate(eval_data);
            std::cout << "  Eval loss=" << eval_loss;
            if (eval_loss < best_eval_loss_) { best_eval_loss_ = eval_loss; std::cout << " (best)"; }
            std::cout << "\n";
        }
        if ((step + 1) % cfg_.save_interval == 0) {
            save_checkpoint(cfg_.output_dir + "/native_quant_step_" + std::to_string(step + 1) + ".quant");
        }
    }
}

double NativeQUANTTrainer::evaluate(const std::vector<std::vector<float>>& eval_data) {
    if (eval_data.empty()) return 0.0;
    push_weights_to_model();
    AutogradEngine::set_enabled(false);
    double total = 0.0;
    for (size_t ei = 0; ei < eval_data.size(); ei++) {
        std::vector<float> seq = eval_data[ei];
        if (seq.empty()) continue;
        Tensor inp(Shape{1, (int64_t)seq.size()});
        std::memcpy(inp.data<float>(), seq.data(), seq.size() * sizeof(float));
        Tensor tgt(Shape{1, (int64_t)seq.size()});
        std::memcpy(tgt.data<float>(), seq.data(), seq.size() * sizeof(float));
        Tensor pos(Shape{1, (int64_t)seq.size()});
        for (int64_t pi = 0; pi < (int64_t)seq.size(); pi++) pos.data<float>()[pi] = (float)pi;
        Tensor logits = model_->forward(inp, pos);
        Tensor loss = AutogradEngine::cross_entropy_op(logits, tgt);
        total += loss.data<float>()[0];
    }
    return total / (double)std::max((size_t)1, eval_data.size());
}

void NativeQUANTTrainer::save_checkpoint(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error("Cannot write " + path); }
    uint32_t magic = 0x4F494C4E;
    uint32_t version = 1;
    f.write((const char*)&magic, 4);
    f.write((const char*)&version, 4);
    f.write((const char*)&total_params_, sizeof(total_params_));
    size_t bs = cfg_.block_size;
    f.write((const char*)&bs, sizeof(bs));
    size_t nb = weight_store_->num_blocks();
    f.write((const char*)&nb, sizeof(nb));
    f.write((const char*)weight_store_->formats_data(), nb);
    f.write((const char*)weight_store_->indices_data(), total_params_);
    f.write((const char*)weight_store_->block_scales_data(), nb * sizeof(float));
    f.write((const char*)scale_m_.get(), nb * sizeof(float));
    f.write((const char*)scale_v_.get(), nb * sizeof(float));
    f.write((const char*)&step_, sizeof(step_));
    if (!f.good()) {
        f.close();
        throw std::runtime_error("Failed to write full checkpoint data to " + path);
    }
    f.close();
}

void NativeQUANTTrainer::load_checkpoint(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error("Cannot read " + path); }
    uint32_t magic, version;
    f.read((char*)&magic, 4); f.read((char*)&version, 4);
    if (magic != 0x4F494C4E) { throw std::runtime_error("Bad checkpoint magic in " + path); }
    size_t saved_params, bs, nb;
    f.read((char*)&saved_params, sizeof(saved_params));
    f.read((char*)&bs, sizeof(bs));
    f.read((char*)&nb, sizeof(nb));
    if (saved_params != total_params_) { throw std::runtime_error("Param count mismatch in checkpoint " + path); }
    if (bs != cfg_.block_size || nb != weight_store_->num_blocks()) {
        throw std::runtime_error("Block size or block count mismatch in checkpoint " + path);
    }
    f.read((char*)weight_store_->formats_data(), nb);
    f.read((char*)weight_store_->indices_data(), total_params_);
    f.read((char*)weight_store_->block_scales_data(), nb * sizeof(float));
    f.read((char*)scale_m_.get(), nb * sizeof(float));
    f.read((char*)scale_v_.get(), nb * sizeof(float));
    f.read((char*)&step_, sizeof(step_));
    if (!f.good()) {
        f.close();
        throw std::runtime_error("Failed to read full checkpoint data from " + path);
    }
    f.close();
}

} // namespace native
} // namespace quant
