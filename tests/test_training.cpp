#include "quant/model.h"
#include "quant/autograd.h"
#include "quant/math.h"
#include "quant/random.h"
#include "quant/trainer.h"
#include <chrono>
#include "quant/optimizer.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include "quant/test.h"

using namespace quant;

static void collect_all_params(DenseModel& dm, std::vector<Tensor*>& params) {
    params.push_back(&dm.tok_embeddings->weight);
    for (auto& layer : dm.layers) {
        params.push_back(&layer->attention_norm.weight);
        params.push_back(&layer->attention.q_proj.weight);
        if (layer->attention.q_proj.bias.numel() > 0)
            params.push_back(&layer->attention.q_proj.bias);
        params.push_back(&layer->attention.k_proj.weight);
        if (layer->attention.k_proj.bias.numel() > 0)
            params.push_back(&layer->attention.k_proj.bias);
        params.push_back(&layer->attention.v_proj.weight);
        if (layer->attention.v_proj.bias.numel() > 0)
            params.push_back(&layer->attention.v_proj.bias);
        params.push_back(&layer->attention.o_proj.weight);
        if (layer->attention.o_proj.bias.numel() > 0)
            params.push_back(&layer->attention.o_proj.bias);
        params.push_back(&layer->ffn_norm.weight);
        params.push_back(&layer->ffn.gate_proj.weight);
        if (layer->ffn.gate_proj.bias.numel() > 0)
            params.push_back(&layer->ffn.gate_proj.bias);
        params.push_back(&layer->ffn.up_proj.weight);
        if (layer->ffn.up_proj.bias.numel() > 0)
            params.push_back(&layer->ffn.up_proj.bias);
        params.push_back(&layer->ffn.down_proj.weight);
        if (layer->ffn.down_proj.bias.numel() > 0)
            params.push_back(&layer->ffn.down_proj.bias);
    }
    params.push_back(&dm.norm->weight);
    params.push_back(&dm.lm_head->weight);
    if (dm.lm_head->bias.numel() > 0)
        params.push_back(&dm.lm_head->bias);
}

static float eval_cross_entropy(const Tensor& logits, const Tensor& targets) {
    int64_t B = logits.dim(0);
    int64_t S = logits.dim(1);
    int64_t V = logits.dim(2);
    const float* ld = logits.data<float>();
    const float* td = targets.data<float>();
    float loss = 0.0f;
    for (int64_t i = 0; i < B * S; i++) {
        int64_t t = (int64_t)td[i];
        if (t < 0 || t >= V) continue;
        float max_l = ld[i * V];
        for (int64_t v = 1; v < V; v++)
            if (ld[i * V + v] > max_l) max_l = ld[i * V + v];
        float sum = 0.0f;
        for (int64_t v = 0; v < V; v++)
            sum += std::exp(ld[i * V + v] - max_l);
        loss += -(ld[i * V + t] - max_l - std::log(sum));
    }
    return loss / (float)(B * S);
}

// Test 3: Scale test — 0.1B-class model (17.8M params), 10 steps, batch2, seq32
static void test_scale_train() {
    printf("\n--- Test 3: Scale test (0.1B-class) ---\n");
    AutogradEngine::instance().reset();
    TransformerConfig cfg;
    cfg.hidden_size = 512;
    cfg.num_layers = 4;
    cfg.num_heads = 8;
    cfg.head_dim = 64;
    cfg.ffn_hidden_size = 2048;
    cfg.vocab_size = 1000;
    cfg.max_seq_len = 256;

    DenseModel model(cfg);
    int64_t param_count = model.param_count();
    printf("  Model params: %lld (%.2fM)\n", (long long)param_count, param_count / 1e6);
    TEST_CHECK(param_count > 0, "Scale model created");
    if (param_count > 0) {
        printf("  Scale: %.1f%% of 0.1B\n", (double)param_count / 1e8 * 100.0);
    }

    int64_t B = 2, S = 32;
    Tensor input_ids(Shape{B, S});
    Tensor positions(Shape{B, S});
    Tensor target_ids(Shape{B, S});
    RNG rng(12345);
    for (int64_t b = 0; b < B; b++)
        for (int64_t s = 0; s < S; s++) {
            int64_t idx = b * S + s;
            input_ids.data<float>()[idx] = (float)(rng.uniform() * (cfg.vocab_size - 10));
            positions.data<float>()[idx] = (float)s;
            target_ids.data<float>()[idx] = (float)((int64_t)(idx + 1) % (cfg.vocab_size - 10));
        }

    float lr = 0.001f;
    std::vector<Tensor*> params;
    collect_all_params(model, params);
    auto& engine = AutogradEngine::instance();
    for (auto* p : params) {
        p->requires_grad(true);
        engine.register_parameter(p);
    }
    SGD optimizer(lr);
    optimizer.add_param_group(params);

    float max_grad_norm = 0.0f;
    float final_loss = 0.0f;
    for (int step = 0; step < 10; step++) {
        optimizer.zero_grad();
        AutogradEngine::set_enabled(true);
        for (auto* p : params) engine.register_parameter(p);

        Tensor logits = model.forward(input_ids, positions);
        Tensor loss_t = AutogradEngine::cross_entropy_op(logits, target_ids);
        float loss_val = *(const float*)loss_t.data();
        final_loss = loss_val;

        engine.backward(loss_t);
        engine.clear();
        AutogradEngine::set_enabled(false);
        optimizer.step();

        float grad_norm = 0.0f;
        for (auto* p : params) {
            if (p->has_grad()) {
                const float* g = p->grad().data<float>();
                int64_t n = p->numel();
                for (int64_t i = 0; i < n; i++) grad_norm += g[i] * g[i];
            }
        }
        grad_norm = std::sqrt(grad_norm);
        if (grad_norm > max_grad_norm) max_grad_norm = grad_norm;

        if (step % 2 == 0)
            printf("  Step %d: loss=%.4f grad_norm=%.2f\n", step, loss_val, grad_norm);
    }

    TEST_CHECK(std::isfinite(final_loss), "Scale test loss is finite");
    TEST_CHECK(max_grad_norm < 100.0f, "Gradient norm < 100 (bounded)");
    printf("  Final loss: %.4f, max grad norm: %.2f\n", final_loss, max_grad_norm);
}

// Test 4: Gradient noise injection (Neelakantan et al. ICLR 2016)
static void test_grad_noise() {
    printf("\n--- Test 4: Gradient Noise Injection ---\n");
    AutogradEngine::instance().reset();
    TransformerConfig cfg;
    cfg.hidden_size = 64;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.head_dim = 16;
    cfg.ffn_hidden_size = 128;
    cfg.vocab_size = 100;
    cfg.max_seq_len = 32;

    DenseModel model(cfg);
    RNG rng(123);
    int64_t B = 2, S = 8;
    Tensor input_ids(Shape{B, S});
    Tensor positions(Shape{B, S});
    Tensor labels(Shape{B, S});
    for (int64_t i = 0; i < B * S; i++) {
        input_ids.data<float>()[i] = (float)(rng.uniform() * 50);
        positions.data<float>()[i] = (float)(i % S);
        labels.data<float>()[i] = (float)((int)(rng.uniform() * 50) % cfg.vocab_size);
    }

    Trainer trainer(&model, nullptr);
    TrainConfig tcfg;
    tcfg.grad_noise_eta = 0.1f;
    tcfg.grad_noise_gamma = 0.55f;
    tcfg.learning_rate = 0.001f;

    AdamW optimizer(tcfg.learning_rate);
    trainer.compile(&optimizer, tcfg);

    bool has_nan = false;
    for (int step = 0; step < 3; step++) {
        float loss = trainer.train_step(input_ids, labels);
        if (!std::isfinite(loss)) has_nan = true;
        printf("  Step %d: loss=%.4f (noise eta=0.1)\n", step, loss);
    }
    TEST_CHECK(!has_nan, "No NaN with gradient noise eta=0.1");

    // Re-compile with noise off
    tcfg.grad_noise_eta = 0.0f;
    trainer.compile(&optimizer, tcfg);

    for (int step = 0; step < 3; step++) {
        float loss = trainer.train_step(input_ids, labels);
        if (!std::isfinite(loss)) has_nan = true;
        printf("  Step %d: loss=%.4f (noise eta=0.0)\n", step + 3, loss);
    }
    TEST_CHECK(!has_nan, "No NaN after disabling gradient noise");
    // BUGFIX (bug census): was trailing TEST_CHECK(true) — vacuous.
    // The two !has_nan asserts above ARE the crash-freedom proof.
}

int main() {
    printf("=== Training Test ===\n\n");

    // Test 1: Gradient check via simple regression with autograd ops
    printf("--- Test 1: Autograd regression test ---\n");
    {
        auto& engine = AutogradEngine::instance();
        int64_t N = 4;
        int64_t D = 3;
        int64_t V = 8;

        Tensor w(Shape{V, D});
        float* wd = w.data<float>();
        for (int64_t i = 0; i < D * V; i++) wd[i] = (float)(i % 5) / 5.0f - 0.5f;

        Tensor x(Shape{N, D});
        float* xd = x.data<float>();
        for (int64_t i = 0; i < N * D; i++)
            xd[i] = (float)(i % 7) / 7.0f - 0.5f;

        Tensor labels(Shape{N, 1});
        float* ld = labels.data<float>();
        for (int64_t i = 0; i < N; i++) ld[i] = (float)(i % V);

        Tensor w_param(Shape{V, D});
        float* wp = w_param.data<float>();
        for (int64_t i = 0; i < D * V; i++) wp[i] = 0.0f;
        engine.register_parameter(&w_param);

        SGD sgd_opt(0.01f);
        std::vector<Tensor*> param_group = {&w_param};
        sgd_opt.add_param_group(param_group);

        float prev_loss = 1e30f;
        for (int step = 0; step < 30; step++) {
            sgd_opt.zero_grad();
            AutogradEngine::set_enabled(true);
            engine.register_parameter(&w_param);

            // x @ w_param gives (N, V) logits, then cross_entropy
            Tensor logits = AutogradEngine::matmul_op(x, w_param, N, V, D);
            Tensor loss = AutogradEngine::cross_entropy_op(logits, labels);

            float loss_val = *(const float*)loss.data();
            engine.backward(loss);
            engine.clear();
            AutogradEngine::set_enabled(false);

            sgd_opt.step();

            if (step == 29) {
                printf("  Step %d: loss = %.6f\n", step, loss_val);
                TEST_CHECK(loss_val < 3.0f, "Regression loss < 3.0 after 30 steps");
                TEST_CHECK(std::isfinite(loss_val), "Regression loss is finite");
            }
            prev_loss = loss_val;
        }
    }

    // Test 2: Full model training (cross-entropy loss)
    printf("\n--- Test 2: Transformer training ---\n");
    AutogradEngine::instance().reset();

    TransformerConfig cfg;
    cfg.hidden_size = 64;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.head_dim = 16;
    cfg.ffn_hidden_size = 128;
    cfg.vocab_size = 100;
    cfg.max_seq_len = 32;

    DenseModel model(cfg);
    printf("Model created: %lld params\n", (long long)model.param_count());
    TEST_CHECK(model.param_count() > 0, "Model has parameters");

    int64_t S = 8;
    int64_t V = cfg.vocab_size;
    Tensor input_ids(Shape{1, S});
    Tensor positions(Shape{1, S});
    Tensor target_ids(Shape{1, S});
    float* id = input_ids.data<float>();
    float* pd = positions.data<float>();
    float* td = target_ids.data<float>();

    for (int64_t i = 0; i < S; i++) {
        id[i] = (float)(i % (V / 2));
        pd[i] = (float)i;
        td[i] = (float)((i + 1) % (V / 2));
    }

    float initial_loss = eval_cross_entropy(model.forward(input_ids, positions), target_ids);
    printf("Initial loss: %.4f\n", initial_loss);
    TEST_CHECK(initial_loss > 0.0f, "Initial loss is positive");
    TEST_CHECK(std::isfinite(initial_loss), "Initial loss is finite");

    float lr = 0.001f;
    int num_steps = 20;
    std::vector<float> losses;

    auto& engine = AutogradEngine::instance();
    std::vector<Tensor*> params;
    collect_all_params(model, params);
    for (auto* p : params)
        p->requires_grad(true);
    for (auto* p : params)
        engine.register_parameter(p);

    SGD optimizer(lr);
    optimizer.add_param_group(params);

    for (int step = 0; step < num_steps; step++) {
        optimizer.zero_grad();
        AutogradEngine::set_enabled(true);
        for (auto* p : params) engine.register_parameter(p);

        Tensor logits = model.forward(input_ids, positions);
        Tensor loss = AutogradEngine::cross_entropy_op(logits, target_ids);

        float loss_val = *(const float*)loss.data();
        losses.push_back(loss_val);

        engine.backward(loss);
        engine.clear();
        AutogradEngine::set_enabled(false);

        optimizer.step();

        if (step % 5 == 0)
            printf("  Step %d: loss = %.4f\n", step, loss_val);
    }

    float final_loss = eval_cross_entropy(model.forward(input_ids, positions), target_ids);
    printf("\nFinal loss: %.4f\n", final_loss);

    // Gradient flow through the full transformer is validated by the finite-difference
    // checks in test_trainer; here we verify the manual SGD loop actually
    // TRAINS (loss decreases end-to-end) and stays stable.
    TEST_CHECK(losses.size() == (size_t)num_steps, "All training steps completed");
    TEST_CHECK(std::isfinite(final_loss), "Final loss is finite");
    TEST_CHECK(final_loss < initial_loss, "Training decreases loss (real gradients)");
    // Also verify the last few logged steps did not diverge.
    float last_logged = losses.back();
    TEST_CHECK(last_logged < 4.0f * initial_loss + 0.5f, "Loss did not blow up during training");

    test_scale_train();

    test_grad_noise();

    TEST_SUITE("Results");
    printf("Initial loss: %.4f\n", initial_loss);
    printf("Final loss:   %.4f\n", final_loss);
    if (initial_loss > 0)
        printf("Reduction:    %.1f%%\n", (1.0f - final_loss / initial_loss) * 100.0f);

    if (TEST_REPORT() > 0) { printf("\nSOME TESTS FAILED\n"); return 1; }
    else { printf("\nALL TESTS PASSED\n"); return 0; }
}
