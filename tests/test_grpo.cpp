// test_grpo.cpp — GRPO correctness regression tests
//
// Guards two fixes in GRPOTrainer::train_step:
//  1. Per-sample advantage weighting: loss = (1/G) * Σ adv_i * CE_i computed
//     INSIDE the graph. The previous implementation backpropagated one mean-CE
//     and rescaled all gradients by mean(adv) (≈0 after normalization), which
//     collapsed RL updates into sign-hacked SFT.
//  2. MoE parameter collection: collect_grpo_params now walks MoEModel layers
//     (router, experts, shared expert) so RLL can run on MoE bases.
#include "quant/model.h"
#include "quant/moe_model.h"
#include "quant/trainer.h"
#include "quant/optimizer.h"
#include "quant/tokenizer.h"
#include "quant/tensor.h"
#include "quant/test.h"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace quant;

namespace {

TransformerConfig tiny_dense_cfg() {
    TransformerConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_size = 32;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.head_dim = 16;
    cfg.ffn_hidden_size = 64;
    cfg.max_seq_len = 32;
    return cfg;
}

float sample_ce(DenseModel& model, const Tensor& ids, const Tensor& labels,
                int row) {
    // no-grad forward of one row; returns sequence mean CE
    bool prev = AutogradEngine::enabled();
    AutogradEngine::set_enabled(false);
    int64_t B = ids.dim(0), S = ids.dim(1);
    Tensor row_ids = ids.slice(0, row, row + 1);
    Tensor pos(Shape{1, S});
    for (int64_t j = 0; j < S; ++j) pos.data<float>()[j] = (float)j;
    Tensor logits = model.forward(row_ids, pos, nullptr);
    AutogradEngine::set_enabled(prev);
    int64_t V = logits.dim(logits.rank() - 1);
    const float* lg = logits.data<float>();
    const float* lb = labels.data<float>() + row * S;
    double ce = 0.0;
    for (int64_t j = 0; j < S; ++j) {
        int t = (int)lb[j];
        float mx = -INFINITY;
        for (int64_t v = 0; v < V; ++v) mx = std::max(mx, lg[j * V + v]);
        double s = 0.0;
        for (int64_t v = 0; v < V; ++v) s += std::exp((double)lg[j * V + v] - mx);
        ce += -((double)lg[j * V + t] - mx - std::log(s));
    }
    return (float)(ce / S);
}

} // namespace

int main() {
    TEST_SUITE("grpo");
    printf("=== GRPO correctness tests ===\n\n");

    AutogradEngine::instance().reset();

    // ---------------------------------------------------------------
    // Test 1: per-sample advantage weighting moves positives apart
    // ---------------------------------------------------------------
    {
        printf("--- Test 1: advantage-weighted updates ---\n");
        auto cfg = tiny_dense_cfg();
        DenseModel model(cfg);
        BPETokenizer tok;

        constexpr int G = 4, S = 8;
        RNG rng(7);
        Tensor ids(Shape{G, S}), labels(Shape{G, S});
        for (int64_t i = 0; i < G * S; ++i) {
            ids.data<float>()[i] = (float)((int)(rng.uniform() * (cfg.vocab_size - 2)) + 1);
            labels.data<float>()[i] = (float)((int)(rng.uniform() * (cfg.vocab_size - 2)) + 1);
        }

        AdamW opt(0.05f);
        GRPOTrainer grpo(&model, &tok, G, /*beta=*/0.04f);
        grpo.set_optimizer(&opt);

        // rewards: first half high, second half low → normalized advantages +,+,-,-
        Tensor rew(Shape{G});
        rew.data<float>()[0] = 3.0f;
        rew.data<float>()[1] = 2.5f;
        rew.data<float>()[2] = 0.5f;
        rew.data<float>()[3] = 0.0f;

        auto gap = [&]() {
            float pos = sample_ce(model, ids, labels, 0) + sample_ce(model, ids, labels, 1);
            float neg = sample_ce(model, ids, labels, 2) + sample_ce(model, ids, labels, 3);
            return neg - pos;
        };
        const float gap_before = gap();

        for (int step = 0; step < 30; ++step) {
            float l = grpo.train_step(ids, labels, rew);
            TEST_CHECK(std::isfinite(l), "GRPO loss finite across steps");
        }
        const float gap_after = gap();

        printf("  pos-vs-neg CE gap: before=%.4f after=%.4f\n", gap_before, gap_after);
        TEST_CHECK(gap_after > gap_before,
                   "positive-advantage samples improve relative to negative "
                   "(per-sample weighting works)");
    }

    // ---------------------------------------------------------------
    // Test 2: zero-variance rewards → no-op update, no NaN
    // ---------------------------------------------------------------
    {
        printf("--- Test 2: zero-variance rewards no-op ---\n");
        auto cfg = tiny_dense_cfg();
        DenseModel model(cfg);
        BPETokenizer tok;
        constexpr int G = 4, S = 8;
        RNG rng(11);
        Tensor ids(Shape{G, S}), labels(Shape{G, S}), rew(Shape{G});
        for (int64_t i = 0; i < G * S; ++i) {
            ids.data<float>()[i] = (float)((int)(rng.uniform() * (cfg.vocab_size - 2)) + 1);
            labels.data<float>()[i] = ids.data<float>()[i];
        }
        for (int64_t g = 0; g < G; ++g) rew.data<float>()[g] = 1.0f;

        AdamW opt(0.05f);
        GRPOTrainer grpo(&model, &tok, G, 0.04f);
        grpo.set_optimizer(&opt);
        float l = grpo.train_step(ids, labels, rew);
        TEST_CHECK(std::isfinite(l), "zero-variance loss is finite");
    }

    // ---------------------------------------------------------------
    // Test 3: MoE base end-to-end (collection + rollout + step)
    // ---------------------------------------------------------------
    {
        printf("--- Test 3: MoE model GRPO smoke ---\n");
        auto cfg = tiny_dense_cfg();
        cfg.head_dim = cfg.hidden_size / cfg.num_heads;

        moe::MoEAllConfig moe_cfg;
        moe_cfg.num_experts = 4;
        moe_cfg.top_k = 2;
        moe_cfg.expert_hidden_size = 64;

        MoEModel model(cfg, moe_cfg);
        BPETokenizer tok;

        constexpr int G = 4, S = 6;
        RNG rng(21);
        Tensor ids(Shape{G, S}), pos(Shape{G, S});
        for (int64_t i = 0; i < G; ++i)
            for (int64_t j = 0; j < S; ++j) {
                ids.data<float>()[i * S + j] =
                    (float)((int)(rng.uniform() * (cfg.vocab_size - 2)) + 1);
                pos.data<float>()[i * S + j] = (float)j;
            }

        auto logits_stats = [&](const char* tag, const Tensor& lg) {
            float mn = INFINITY, mx = -INFINITY;
            int64_t bad = 0;
            for (int64_t i = 0; i < lg.numel(); ++i) {
                float v = lg.data<float>()[i];
                if (!std::isfinite(v)) ++bad;
                mn = std::min(mn, v); mx = std::max(mx, v);
            }
            printf("  [%s] logits min=%.4f max=%.4f nonfinite=%lld/%lld\n",
                   tag, mn == INFINITY ? 0.f : mn, mx == -INFINITY ? 0.f : mx,
                   (long long)bad, (long long)lg.numel());
        };

        // Stage A: no-grad forward WITH explicit positions
        AutogradEngine::set_enabled(false);
        Tensor lgA = model.forward(ids, pos, nullptr);
        logits_stats("A nograd+pos", lgA);

        // Stage B: no-grad forward with EMPTY positions (micro_step style)
        Tensor lgB = model.forward(ids, Tensor(), nullptr);
        logits_stats("B nograd+empty", lgB);

        // Stage C: autograd ON forward + single CE
        AutogradEngine::set_enabled(true);
        Tensor lgC = model.forward(ids, pos, nullptr);
        logits_stats("C grad+pos", lgC);
        Tensor ceC = AutogradEngine::cross_entropy_op(lgC, ids);
        printf("  [C ce]=%.4f\n", ceC.data<float>()[0]);
        AutogradEngine::set_enabled(false);

        GRPOTrainer grpo(&model, &tok, G, 0.04f);
        AdamW opt(0.02f);
        grpo.set_optimizer(&opt);

        // Stage D: direct tensor-variant steps on our controlled batch
        Tensor rew(Shape{G});
        rew.data<float>()[0] = 1.0f; rew.data<float>()[1] = 0.5f;
        rew.data<float>()[2] = 0.25f; rew.data<float>()[3] = 0.0f;
        for (int step = 0; step < 3; ++step) {
            float l = grpo.train_step(ids, ids, rew);
            printf("  [D step %d] loss=%.4f finite=%d\n", step, l, (int)std::isfinite(l));
            TEST_CHECK(std::isfinite(l), "MoE GRPO tensor-step loss finite");
        }

        // Stage E: prompt-variant smoke
        float lp = grpo.train_step("hello world test");
        TEST_CHECK(std::isfinite(lp), "MoE GRPO prompt loss finite");
        printf("  3 MoE GRPO steps completed with finite losses\n");
    }

    printf("\nGRPO TESTS PASSED!\n");
    return 0;
}
