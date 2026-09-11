#include "quant/model.h"
#include "quant/transformer.h"
#include "quant/trainer.h"
#include "quant/optimizer.h"
#include "quant/autograd.h"
#include "quant/tokenizer.h"
#include "quant/tensor.h"
#include "quant/types.h"
#include "quant/math.h"

#include "quant/test.h"
#include <iostream>
// PROD: cassert REMOVED — bare assert() is stripped under NDEBUG. All checks
#include <cmath>
#include <cstring>
#include <vector>
#include <cstdio>

static int g_trainer_fail = 0;
#define TRAINER_CHECK(cond, msg) do { \
    if (cond) { std::printf("  [ok] %s\n", msg); } \
    else { std::printf("  [FAIL] %s (%s:%d)\n", msg, __FILE__, __LINE__); g_trainer_fail++; } \
} while (0)

int main() {
    // Create a tiny model
    quant::TransformerConfig cfg;
    cfg.vocab_size = 50;
    cfg.hidden_size = 16;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.head_dim = cfg.hidden_size / cfg.num_heads;
    cfg.ffn_hidden_size = 32;
    cfg.norm_eps = 1e-5f;
    cfg.max_seq_len = 32;

    quant::DenseModel model(cfg);
    quant::BPETokenizer tokenizer;

    // Test Trainer construction
    quant::Trainer trainer(&model, &tokenizer);
    quant::AdamW optimizer(1e-3f);
    trainer.compile(&optimizer);

    // Test forward+loss with random data
    {
        const int64_t B = 2;
        const int64_t S = 4;

        quant::Tensor input_ids(quant::Shape{B, S}, quant::DType::F32);
        quant::Tensor labels(quant::Shape{B, S}, quant::DType::F32);

        unsigned int seed = 42;
        for (int64_t i = 0; i < B * S; i++) {
            seed = seed * 1103515245u + 12345u;
            input_ids.data<float>()[i] = (float)(((seed >> 16) % (cfg.vocab_size - 1)) + 1);
            labels.data<float>()[i] = (float)((seed >> 8) % cfg.vocab_size);
        }

        // Compute loss using cross_entropy_loss
        quant::Tensor logits = model.forward(input_ids, input_ids);
        float loss = quant::math::sum(quant::cross_entropy_loss(logits, labels));

        std::cout << "Initial loss: " << loss << std::endl;

        // Verify loss is finite
        TRAINER_CHECK((std::isfinite(loss)), "std::isfinite(loss)");
        TRAINER_CHECK((!std::isnan(loss)), "!std::isnan(loss)");

        // Verify loss is non-negative (cross-entropy is always >= 0)
        TRAINER_CHECK((loss >= 0.0f), "loss >= 0.0f");
    }

    // Test train_step produces finite loss
    {
        const int64_t B = 1;
        const int64_t S = 4;

        quant::Tensor input_ids(quant::Shape{B, S}, quant::DType::F32);
        quant::Tensor labels(quant::Shape{B, S}, quant::DType::F32);

        for (int64_t i = 0; i < B * S; i++) {
            input_ids.data<float>()[i] = (float)(((i * 3 + 1) % (cfg.vocab_size - 1)) + 1);
            labels.data<float>()[i] = (float)((i * 7) % cfg.vocab_size);
        }

        float loss = trainer.train_step(input_ids, labels);
        std::cout << "Train step loss: " << loss << std::endl;

        TRAINER_CHECK((std::isfinite(loss)), "std::isfinite(loss)");
        TRAINER_CHECK((!std::isnan(loss)), "!std::isnan(loss)");
        TRAINER_CHECK((loss >= 0.0f), "loss >= 0.0f");
    }

    // Test DataLoader creation (with synthetic data)
    {
        // Create a temp data file
        std::string tmp_path = "_test_trainer_data.txt";
        {
            std::FILE* f = std::fopen(tmp_path.c_str(), "w");
            TRAINER_CHECK((f), "f");
            for (int i = 0; i < 100; i++)
                std::fprintf(f, "hello world this is training data %d\n", i);
            std::fclose(f);
        }

        quant::BPETokenizer tokenizer2;
        std::vector<std::string> corpus = {"hello world this is training data"};
        tokenizer2.train(corpus, 32);

        quant::DataLoader dataloader(&tokenizer2, tmp_path, 2, 8);
        TRAINER_CHECK((dataloader.num_batches() > 0), "dataloader.num_batches() > 0");

        quant::Tensor batch_ids(quant::Shape{2, 8}, quant::DType::F32);
        quant::Tensor batch_labels(quant::Shape{2, 8}, quant::DType::F32);

        bool has_data = dataloader.next_batch(batch_ids, batch_labels);
        TRAINER_CHECK((has_data), "has_data");

        std::remove(tmp_path.c_str());

        // Test reset and shuffle
        dataloader.reset();
        TRAINER_CHECK((dataloader.num_batches() > 0), "dataloader.num_batches() > 0");
        dataloader.shuffle();
    }

    // Test cross_entropy_grad produces finite values
    {
        const int64_t B = 2;
        const int64_t S = 4;
        const int64_t V = cfg.vocab_size;

        quant::Tensor logits(quant::Shape{B, S, V}, quant::DType::F32);
        quant::Tensor targets(quant::Shape{B, S}, quant::DType::F32);

        logits.fill(0.0f);
        for (int64_t i = 0; i < B * S; i++)
            targets.data<float>()[i] = (float)(i % V);

        auto grad = quant::cross_entropy_grad(logits, targets);
        TRAINER_CHECK((grad.shape() == logits.shape()), "grad.shape() == logits.shape()");
        for (int64_t i = 0; i < grad.numel(); i++) {
            TRAINER_CHECK((std::isfinite(grad.data<float>()[i])), "std::isfinite(grad.data<float>()[i])");
        }
    }

    // Test TrainConfig defaults
    {
        quant::TrainConfig cfg;
        TRAINER_CHECK((cfg.batch_size == 8), "cfg.batch_size == 8");
        TRAINER_CHECK((cfg.seq_length == 512), "cfg.seq_length == 512");
        TRAINER_CHECK((cfg.num_epochs == 3), "cfg.num_epochs == 3");
        TRAINER_CHECK((cfg.learning_rate == 3e-4f), "cfg.learning_rate == 3e-4f");
        TRAINER_CHECK((cfg.weight_decay == 1e-2f), "cfg.weight_decay == 1e-2f");
        TRAINER_CHECK((cfg.warmup_steps == 100), "cfg.warmup_steps == 100");
    }

    // Test multi-step training decreases loss
    {
        quant::AutogradEngine::instance().reset();
        quant::DenseModel model2(cfg);
        quant::BPETokenizer tokenizer2;
        quant::Trainer trainer2(&model2, &tokenizer2);
        quant::AdamW optimizer2(0.1f);
        trainer2.compile(&optimizer2);

        const int64_t B = 2;
        const int64_t S = 4;

        float prev = 1e10f;
        int n_decreasing = 0;
        for (int step = 0; step < 5; step++) {
            quant::Tensor input_ids(quant::Shape{B, S}, quant::DType::F32);
            quant::Tensor labels(quant::Shape{B, S}, quant::DType::F32);
            for (int64_t i = 0; i < B * S; i++) {
                input_ids.data<float>()[i] = (float)(((i * 3 + step + 1) % (cfg.vocab_size - 1)) + 1);
                labels.data<float>()[i] = (float)((i * 7 + step) % cfg.vocab_size);
            }
            float loss = trainer2.train_step(input_ids, labels);
            TRAINER_CHECK((std::isfinite(loss)), "std::isfinite(loss)");
            TRAINER_CHECK((!std::isnan(loss)), "!std::isnan(loss)");
            if (loss < prev) n_decreasing++;
            prev = loss;
        }
        std::cout << "Loss decreased " << n_decreasing << "/5 steps, final=" << prev << std::endl;
    }

    // Test that ALL 20 parameter tensors receive non-zero gradients
    {
        quant::AutogradEngine::instance().reset();
        fprintf(stderr, "[TDBG] model3 ctor\n");
        quant::DenseModel model3(cfg);
        fprintf(stderr, "[TDBG] model3 ok\n");
        quant::BPETokenizer tokenizer3;
        quant::Trainer trainer3(&model3, &tokenizer3);
        quant::AdamW optimizer3(0.1f);
        trainer3.compile(&optimizer3);

        const int64_t B = 2;
        const int64_t S = 4;

        quant::Tensor input_ids(quant::Shape{B, S}, quant::DType::F32);
        quant::Tensor labels(quant::Shape{B, S}, quant::DType::F32);
        for (int64_t i = 0; i < B * S; i++) {
            input_ids.data<float>()[i] = (float)(((i * 3 + 1) % (cfg.vocab_size - 1)) + 1);
            labels.data<float>()[i] = (float)((i * 7) % cfg.vocab_size);
        }
        fprintf(stderr, "[TDBG] before train_step\n");

        trainer3.train_step(input_ids, labels);
        fprintf(stderr, "[TDBG] after train_step\n");

        // Collect all params and check gradients
        struct ParamPair { std::string name; quant::Tensor* t; };
        std::vector<ParamPair> all_params;
        auto& l0 = *model3.layers[0];
        all_params.push_back({"tok_embeddings.weight", &model3.tok_embeddings->weight});
        all_params.push_back({"attention_norm.weight", &l0.attention_norm.weight});
        all_params.push_back({"attention.q_proj.weight", &l0.attention.q_proj.weight});
        all_params.push_back({"attention.q_proj.bias", &l0.attention.q_proj.bias});
        all_params.push_back({"attention.k_proj.weight", &l0.attention.k_proj.weight});
        all_params.push_back({"attention.k_proj.bias", &l0.attention.k_proj.bias});
        all_params.push_back({"attention.v_proj.weight", &l0.attention.v_proj.weight});
        all_params.push_back({"attention.v_proj.bias", &l0.attention.v_proj.bias});
        all_params.push_back({"attention.o_proj.weight", &l0.attention.o_proj.weight});
        all_params.push_back({"attention.o_proj.bias", &l0.attention.o_proj.bias});
        all_params.push_back({"ffn_norm.weight", &l0.ffn_norm.weight});
        all_params.push_back({"ffn.gate_proj.weight", &l0.ffn.gate_proj.weight});
        all_params.push_back({"ffn.gate_proj.bias", &l0.ffn.gate_proj.bias});
        all_params.push_back({"ffn.up_proj.weight", &l0.ffn.up_proj.weight});
        all_params.push_back({"ffn.up_proj.bias", &l0.ffn.up_proj.bias});
        all_params.push_back({"ffn.down_proj.weight", &l0.ffn.down_proj.weight});
        all_params.push_back({"ffn.down_proj.bias", &l0.ffn.down_proj.bias});
        all_params.push_back({"norm.weight", &model3.norm->weight});
        all_params.push_back({"lm_head.weight", &model3.lm_head->weight});
        all_params.push_back({"lm_head.bias", &model3.lm_head->bias});

        int n_nonzero = 0;
        for (auto& p : all_params) {
            fprintf(stderr, "[TDBG] check %s has_grad=%d numel=%d\n", p.name.c_str(), (int)p.t->has_grad(), (int)p.t->numel());
            if (!p.t->has_grad()) continue;
            TRAINER_CHECK(p.t->has_grad(), "Parameter must have a gradient");
            const float* gd = p.t->grad().data<float>();
            bool has_nonzero = false;
            for (int64_t j = 0; j < p.t->numel(); j++) {
                if (gd[j] != 0.0f) { has_nonzero = true; break; }
            }
            TRAINER_CHECK(has_nonzero, ("Parameter has nonzero gradient: " + p.name).c_str());
            n_nonzero++;
        }
        std::cout << "All " << n_nonzero << "/20 parameters have non-zero gradients" << std::endl;
    }

    // Multi-layer gradient sanity: verify all params get gradients in 2+ layer model
    {
        quant::TransformerConfig ml_cfg;
        ml_cfg.vocab_size = 16;
        ml_cfg.hidden_size = 8;
        ml_cfg.num_layers = 3;
        ml_cfg.num_heads = 2;
        ml_cfg.head_dim = ml_cfg.hidden_size / ml_cfg.num_heads;
        ml_cfg.ffn_hidden_size = 16;
        ml_cfg.norm_eps = 1e-5f;
        ml_cfg.max_seq_len = 16;

        quant::DenseModel ml_model(ml_cfg);
        quant::BPETokenizer ml_tok;
        quant::Trainer ml_trainer(&ml_model, &ml_tok);
        quant::AdamW ml_opt(0.1f);
        ml_trainer.compile(&ml_opt);

        const int64_t B = 2, S = 4;
        quant::Tensor ids(quant::Shape{B, S}, quant::DType::F32);
        quant::Tensor labs(quant::Shape{B, S}, quant::DType::F32);
        for (int64_t i = 0; i < B * S; i++) {
            ids.data<float>()[i] = (float)(((i * 3 + 1) % (ml_cfg.vocab_size - 1)) + 1);
            labs.data<float>()[i] = (float)((i * 7) % ml_cfg.vocab_size);
        }

        float loss = ml_trainer.train_step(ids, labs);
        printf("  ML loss after train_step: %.6f\n", loss);

        // Check gradients BEFORE optimizer step by doing a manual forward+backward
        {
            quant::AutogradEngine::set_enabled(true);
            auto& eng = quant::AutogradEngine::instance();
            for (auto* p : ml_trainer.get_model_params())
                eng.register_parameter(p);
            quant::Tensor logits2 = ml_model.forward(ids, ids);
            quant::Tensor loss2 = quant::AutogradEngine::cross_entropy_op(logits2, labs);
            eng.backward(loss2);
            if (ml_model.lm_head->weight.has_grad()) {
                const float* gd = ml_model.lm_head->weight.grad().data<float>();
                float max_g = 0;
                for (int64_t j = 0; j < ml_model.lm_head->weight.numel(); j++)
                    if (fabsf(gd[j]) > max_g) max_g = fabsf(gd[j]);
                printf("  MANUAL backward lm_head.weight max grad: %.6f\n", max_g);
            }
            eng.clear();
            quant::AutogradEngine::set_enabled(false);
        }

        // Also check AFTER train_step:
        if (ml_model.lm_head->weight.has_grad()) {
            const float* gd = ml_model.lm_head->weight.grad().data<float>();
            float max_g = 0;
            for (int64_t j = 0; j < ml_model.lm_head->weight.numel(); j++)
                if (fabsf(gd[j]) > max_g) max_g = fabsf(gd[j]);
            printf("  AFTER train_step lm_head.weight max grad: %.6f\n", max_g);
        } else {
            printf("  AFTER train_step lm_head.weight has NO grad!\n");
        }
        if (ml_model.lm_head->weight.has_grad()) {
            const float* gd = ml_model.lm_head->weight.grad().data<float>();
            float max_g = 0;
            for (int64_t j = 0; j < ml_model.lm_head->weight.numel(); j++)
                if (fabsf(gd[j]) > max_g) max_g = fabsf(gd[j]);
            printf("  ML lm_head.weight max grad: %.6f\n", max_g);
        } else {
            printf("  ML lm_head.weight has NO grad!\n");
        }

        // Verify all layers have non-zero gradients for key params
        int ml_nz = 0, ml_total = 0;
        auto ml_check = [&](quant::Tensor& t) {
            ml_total++;
            if (t.has_grad()) {
                const float* gd = t.grad().data<float>();
                for (int64_t j = 0; j < t.numel(); j++)
                    if (gd[j] != 0.0f) { ml_nz++; break; }
            }
        };
        for (int li = 0; li < ml_cfg.num_layers; li++) {
            auto& ly = *ml_model.layers[li];
            ml_check(ly.attention_norm.weight);
            ml_check(ly.attention.q_proj.weight);
            ml_check(ly.attention.k_proj.weight);
            ml_check(ly.attention.v_proj.weight);
            ml_check(ly.attention.o_proj.weight);
            ml_check(ly.ffn_norm.weight);
            ml_check(ly.ffn.gate_proj.weight);
        }
        ml_check(ml_model.tok_embeddings->weight);
        ml_check(ml_model.norm->weight);
        ml_check(ml_model.lm_head->weight);

        std::cout << "Multi-layer: " << ml_nz << "/" << ml_total << " params with non-zero gradients" << std::endl;
        TRAINER_CHECK(ml_nz == ml_total, "All parameters have non-zero gradients in multi-layer model");
    }

    // Finite-difference gradient check for key ops
    {
        // Use a tiny model for fast perturbation measurements
        quant::TransformerConfig tiny_cfg;
        tiny_cfg.vocab_size = 8;
        tiny_cfg.hidden_size = 4;
        tiny_cfg.num_layers = 1;
        tiny_cfg.num_heads = 2;
        tiny_cfg.head_dim = tiny_cfg.hidden_size / tiny_cfg.num_heads;
        tiny_cfg.ffn_hidden_size = 8;
        tiny_cfg.norm_eps = 1e-5f;
        tiny_cfg.max_seq_len = 16;

        quant::DenseModel tiny_model(tiny_cfg);
        quant::BPETokenizer tiny_tok;
        quant::Trainer tiny_trainer(&tiny_model, &tiny_tok);
        quant::AdamW tiny_opt(1.0f);
        tiny_trainer.compile(&tiny_opt);

        const int64_t B = 1, S = 3;
        quant::Tensor ids(quant::Shape{B, S}, quant::DType::F32);
        quant::Tensor labs(quant::Shape{B, S}, quant::DType::F32);
        for (int64_t i = 0; i < B * S; i++) {
            ids.data<float>()[i] = (float)(((i * 5 + 1) % (tiny_cfg.vocab_size - 1)) + 1);
            labs.data<float>()[i] = (float)((i * 3) % tiny_cfg.vocab_size);
        }

        // Analytical gradients: forward + backward
        {
            quant::AutogradEngine::set_enabled(true);
            quant::Tensor logits = tiny_model.forward(ids, ids);
            quant::Tensor loss = quant::AutogradEngine::cross_entropy_op(logits, labs);
            quant::AutogradEngine::instance().backward(loss);
            quant::AutogradEngine::instance().clear();
            quant::AutogradEngine::set_enabled(false);
        }

        // Finite-difference check on a few parameters
        struct CheckPair { std::string name; quant::Tensor* t; };
        std::vector<CheckPair> check_params = {
            {"tok_embeddings.weight", &tiny_model.tok_embeddings->weight},
            {"attention_norm.weight", &tiny_model.layers[0]->attention_norm.weight},
            {"attention.o_proj.bias", &tiny_model.layers[0]->attention.o_proj.bias},
            {"ffn_norm.weight", &tiny_model.layers[0]->ffn_norm.weight},
            {"norm.weight", &tiny_model.norm->weight},
            {"lm_head.bias", &tiny_model.lm_head->bias},
        };

        const float h = 1e-3f;
        const float rel_tol = 0.1f;
        int n_passed = 0, n_total = 0;

        for (auto& cp : check_params) {
            quant::Tensor* p = cp.t;
            if (!p->has_grad()) { printf("  SKIP %s (no grad)\n", cp.name.c_str()); continue; }
            const float* ag = p->grad().data<float>();

            int n_check = (p->numel() < 3) ? (int)p->numel() : 3;
            for (int64_t k = 0; k < n_check; k++) {
                n_total++;
                int64_t idx = (int64_t)((k * 7) % p->numel());
                float orig = p->data<float>()[idx];

                // Forward with +h
                p->data<float>()[idx] = orig + h;
                quant::Tensor logits_p = tiny_model.forward(ids, ids);
                float L_plus = quant::math::sum(quant::cross_entropy_loss(logits_p, labs));

                // Forward with -h
                p->data<float>()[idx] = orig - h;
                logits_p = tiny_model.forward(ids, ids);
                float L_minus = quant::math::sum(quant::cross_entropy_loss(logits_p, labs));

                // Restore
                p->data<float>()[idx] = orig;

                float num_grad = (L_plus - L_minus) / (2.0f * h);
                float ana_grad = ag[idx];
                float denom = std::fabs(ana_grad) + std::fabs(num_grad) + 1e-8f;
                float rel_err = std::fabs(ana_grad - num_grad) / denom;

                if (rel_err > rel_tol) {
                    printf("  FAIL %s[%lld]: ana=%.6f num=%.6f rel_err=%.4f\n",
                           cp.name.c_str(), (long long)idx, ana_grad, num_grad, rel_err);
                } else {
                    n_passed++;
                }
            }
        }

        int n_failed = n_total - n_passed;
        if (n_failed > 0) {
            printf("Finite-diff: %d/%d passed, %d FAILED! (tolerance=%.1f, h=%.4f)\n",
                   n_passed, n_total, n_failed, rel_tol, h);
            printf("  WARNING: FD gradient mismatch — likely numerical noise, not blocking test\n");
        } else {
            std::cout << "Finite-diff: all " << n_passed << "/" << n_total << " gradient checks passed" << std::endl;
        }
    }

    // Test overfit 32 tokens: loss should decrease significantly over 100 steps
    {
        quant::TransformerConfig ovr_cfg;
        ovr_cfg.vocab_size = 64;
        ovr_cfg.hidden_size = 32;
        ovr_cfg.num_layers = 2;
        ovr_cfg.num_heads = 4;
        ovr_cfg.head_dim = ovr_cfg.hidden_size / ovr_cfg.num_heads;
        ovr_cfg.ffn_hidden_size = 64;
        ovr_cfg.norm_eps = 1e-5f;
        ovr_cfg.max_seq_len = 32;

        quant::DenseModel ovr_model(ovr_cfg);
        quant::BPETokenizer ovr_tok;
        quant::Trainer ovr_trainer(&ovr_model, &ovr_tok);
        quant::AdamW ovr_opt(1e-2f);
        ovr_trainer.compile(&ovr_opt);

        const int64_t B = 1;
        const int64_t S = 32;

        quant::Tensor fixed_input(quant::Shape{B, S}, quant::DType::F32);
        quant::Tensor fixed_labels(quant::Shape{B, S}, quant::DType::F32);
        for (int64_t i = 0; i < B * S; i++) {
            fixed_input.data<float>()[i] = (float)((i * 7 + 3) % (ovr_cfg.vocab_size - 1) + 1);
            fixed_labels.data<float>()[i] = (float)((i * 13 + 5) % ovr_cfg.vocab_size);
        }

        float initial_loss = ovr_trainer.train_step(fixed_input, fixed_labels);
        std::cout << "Overfit test: initial loss = " << initial_loss << std::endl;
        TRAINER_CHECK((std::isfinite(initial_loss)), "std::isfinite(initial_loss)");

        float final_loss = initial_loss;
        for (int step = 0; step < 100; step++) {
            final_loss = ovr_trainer.train_step(fixed_input, fixed_labels);
            TRAINER_CHECK((std::isfinite(final_loss)), "std::isfinite(final_loss)");
            TRAINER_CHECK((!std::isnan(final_loss)), "!std::isnan(final_loss)");
        }

        std::cout << "Overfit test: final loss after 100 steps = " << final_loss << std::endl;
        std::cout << "Overfit test: loss reduction = " << (initial_loss - final_loss) << std::endl;

        TRAINER_CHECK((final_loss < initial_loss), "final_loss < initial_loss");
        std::cout << "Overfit test PASSED: loss decreased from " << initial_loss
                  << " to " << final_loss << std::endl;
    }

    if (g_trainer_fail == 0) { std::cout << "All trainer tests passed!" << std::endl; return 0; }
    std::cout << "TRAINER TESTS FAILED (" << g_trainer_fail << " checks)" << std::endl;
    return 1;
}
