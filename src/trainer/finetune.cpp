#include "quant/finetune.h"
#include "quant/ste_quantizer.h"
#include "quant/tokenizer.h"
#include "quant/trainer.h"
#include "quant/transformer.h"
#include "quant/autograd.h"
#include <fstream>
#include <cmath>
#include <cstdio>

namespace quant {

FineTuner::FineTuner(Model* m, Tokenizer* t) : model_(m), tokenizer_(t) {}

void FineTuner::configure(const FineTuneConfig& cfg) {
    cfg_ = cfg;
    if (cfg_.optimizer_name == "adamw") {
        auto opt = std::make_unique<AdamW>(cfg.learning_rate);
        opt->set_weight_decay(0);
        opt->set_schedule(AdamW::Schedule::WARMUP_COSINE, cfg.warmup_steps, cfg.num_epochs * 1000);
        optimizer_ = std::move(opt);
    } else {
        auto opt = std::make_unique<Adafactor>(cfg.learning_rate);
        opt->set_weight_decay(0);
        optimizer_ = std::move(opt);
    }

    DenseModel* dm = dynamic_cast<DenseModel*>(model_);
    if (dm) {
        std::vector<Tensor*> params;
        collect_dense_params(dm, params);
        optimizer_->add_param_group(params);
    }
}

void FineTuner::fine_tune(const std::string& data_path) {
    DataLoader dataloader(tokenizer_, data_path, cfg_.batch_size, cfg_.seq_length);
    fine_tune(dataloader);
}

void FineTuner::fine_tune(DataLoader& dataloader) {
    Tensor input_ids(Shape{cfg_.batch_size, cfg_.seq_length}, DType::F32);
    Tensor labels(Shape{cfg_.batch_size, cfg_.seq_length}, DType::F32);

    const bool prev_ag = AutogradEngine::enabled();
    AutogradEngine::set_enabled(true);
    unfreeze_for_finetune();

    for (int epoch = 0; epoch < cfg_.num_epochs; epoch++) {
        dataloader.reset();
        int batch_idx = 0;
    while (dataloader.next_batch(input_ids, labels)) {
            Tensor positions(Shape{cfg_.batch_size, cfg_.seq_length}, DType::F32);
            float* pd = positions.data<float>();
            for (int64_t i = 0; i < cfg_.batch_size * cfg_.seq_length; i++)
                pd[i] = (float)(i % cfg_.seq_length);

            Tensor logits = model_->forward(input_ids, positions);
            Tensor loss = cross_entropy_loss(logits, labels);

            optimizer_->zero_grad();
            AutogradEngine::instance().backward(loss);

            float grad_norm = 0;
            DenseModel* dm = dynamic_cast<DenseModel*>(model_);
            if (dm) {
                std::vector<Tensor*> params;
                collect_dense_params(dm, params);
                for (auto* p : params) {
                    if (p->requires_grad() && p->has_grad()) {
                        const float* g = (const float*)p->grad().data<float>();
                        for (int64_t i = 0; i < p->numel(); ++i)
                            grad_norm += g[i] * g[i];
                    }
                }
            }
            grad_norm = std::sqrt(grad_norm);

            if (cfg_.max_grad_norm > 0 && grad_norm > cfg_.max_grad_norm) {
                float scale = cfg_.max_grad_norm / (grad_norm + 1e-8f);
                if (dm) {
                    std::vector<Tensor*> params;
                    collect_dense_params(dm, params);
                    for (auto* p : params) {
                        if (p->requires_grad() && p->has_grad()) {
                            float* g = p->grad().data<float>();
                            for (int64_t i = 0; i < p->numel(); ++i)
                                g[i] *= scale;
                        }
                    }
                }
                grad_norm = cfg_.max_grad_norm;
            }

            if (grad_norm < cfg_.grad_threshold) {
                batch_idx++;
                continue;
            }
            optimizer_->step();

            if (batch_idx % cfg_.log_interval == 0 && cfg_.log_interval > 0) {
                float loss_val = loss.numel() > 0 ? loss.data<float>()[0] : 0.0f;
                // BUGFIX (bug census): loss computed then (void)-discarded —
                // training ran blind (no log output at all). Emit progress.
                std::fprintf(stderr, "[finetune] batch %lld loss %.6f\n",
                             (long long)batch_idx, (double)loss_val);
            }
            if (batch_idx % cfg_.save_interval == 0 && cfg_.save_interval > 0) {
                save(cfg_.output_path);
            }
            batch_idx++;
        }
    }

    AutogradEngine::set_enabled(prev_ag);
    save(cfg_.output_path);
}

std::vector<int> FineTuner::find_trainable_blocks(const Tensor& gradients, float threshold) {
    std::vector<int> trainable;
    const float* gd = (const float*)gradients.data();
    int64_t n = gradients.numel();
    int block_size = 256;
    int num_blocks = (int)(n / block_size);
    for (int b = 0; b < num_blocks; b++) {
        float block_norm = 0;
        for (int i = 0; i < block_size && b * block_size + i < n; i++) {
            float g = gd[b * block_size + i];
            block_norm += g * g;
        }
        block_norm = std::sqrt(block_norm / (float)block_size);
        if (block_norm > threshold) {
            trainable.push_back(b);
        }
    }
    return trainable;
}

void FineTuner::apply_quant_update(const Tensor& fp32_grad, Tensor& quant_weight, Format fmt) {
    if (quant_weight.numel() == 0 || fp32_grad.numel() == 0) return;

    STEQuantizer ste(fmt);
    Tensor updated = ste.forward(quant_weight);

    float* wd = (float*)updated.data();
    const float* gd = (const float*)fp32_grad.data();
    int64_t n = updated.numel();
    for (int64_t i = 0; i < n; i++) {
        wd[i] -= cfg_.learning_rate * gd[i];
    }

    if (fmt == Format::Q8) {
        CodebookQUANT8 cb;
        cb.train(wd, (size_t)n);
        Tensor quantized = ste.quantize_with_codebook(updated, cb);
        quantized.copy_to(quant_weight);
    } else if (fmt == Format::Q4) {
        CodebookQUANT4 cb;
        cb.train(wd, (size_t)n);
        Tensor quantized = ste.quantize_with_codebook(updated, cb);
        quantized.copy_to(quant_weight);
        uint8_t* dst = (uint8_t*)quant_weight.data();
        float scale;
        ste.quantize_quant(wd, dst, &scale, n);
    } else if (fmt == Format::Q1) {
        uint8_t* dst = (uint8_t*)quant_weight.data();
        float scale;
        ste.quantize_Q1(wd, dst, &scale, n);
    }
}

void FineTuner::save(const std::string& path) {
    model_->save(path);
}

void FineTuner::freeze_base_model() {
    DenseModel* dm = dynamic_cast<DenseModel*>(model_);
    if (dm) {
        std::vector<Tensor*> params;
        collect_dense_params(dm, params);
        for (auto* p : params) p->requires_grad(false);
    }
}

void FineTuner::unfreeze_for_finetune() {
    DenseModel* dm = dynamic_cast<DenseModel*>(model_);
    if (dm) {
        std::vector<Tensor*> params;
        collect_dense_params(dm, params);
        for (auto* p : params) p->requires_grad(true);
    }
}

} // namespace quant
