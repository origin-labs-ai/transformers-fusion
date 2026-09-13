// test_rll_e2e.cpp — End-to-end RLL proof: GRPO on HybridMoeModel (hybrid)
//
// Validates the full stack: standard attention + SparseMoE FFN +
// GRPO per-sample advantage weighting + rollout generation, all native C++.
#include "quant/hybrid_moe_model.h"
#include "quant/trainer.h"
#include "quant/optimizer.h"
#include "quant/test.h"
#include <cmath>
#include <cstdio>

using namespace quant;

int main() {
    TEST_SUITE("rll_e2e");
    printf("=== RLL e2e: GRPO x HybridMoeModel ===\n\n");

    // --- Build tiny hybrid MoE model ---
    TransformerConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_size = 32;
    cfg.num_heads = 2;
    cfg.head_dim = 16;
    cfg.num_layers = 4;   // standard attention (unified STD schedule)
    cfg.max_seq_len = 64;
    cfg.ffn_hidden_size = 64;

    moe::MoEAllConfig mcfg;
    mcfg.num_experts = 4;
    mcfg.top_k = 2;
    mcfg.expert_hidden_size = 64;

    auto sched = build_hybrid_schedule(cfg.num_layers, 3);
    HybridMoeModel model(cfg, mcfg, sched);
    TEST_CHECK(model.param_count() > 0, "hybrid model has params");
    printf("  model params: %lld (schedule %d layers)\n",
           (long long)model.param_count(), sched.total());

    // --- Wire GRPO ---
    BPETokenizer tok;
    GRPOTrainer grpo(&model, &tok, /*group_size=*/4, /*beta=*/0.04f);
    AdamW opt(0.02f);
    grpo.set_optimizer(&opt);

    // --- E2E: prompt-variant rollouts + advantage-weighted updates ---
    printf("--- E2E: 5 GRPO steps on hybrid MoE ---\n");
    for (int step = 0; step < 5; ++step) {
        float loss = grpo.train_step("rll test");
        printf("  step %d: loss=%.4f\n", step, loss);
        TEST_CHECK(std::isfinite(loss), "loss finite across steps");
    }

    // --- Tensor-batch variant too (direct advantage control) ---
    constexpr int G = 4, S = 8;
    RNG rng(33);
    Tensor ids(Shape{G, S}), labels(Shape{G, S}), rew(Shape{G});
    for (int64_t i = 0; i < G * S; ++i) {
        ids.data<float>()[i] = (float)((int)(rng.uniform() * (cfg.vocab_size - 2)) + 1);
        labels.data<float>()[i] = ids.data<float>()[i];
    }
    rew.data<float>()[0] = 1.0f; rew.data<float>()[1] = 0.6f;
    rew.data<float>()[2] = 0.2f; rew.data<float>()[3] = 0.0f;

    float l = grpo.train_step(ids, labels, rew);
    TEST_CHECK(std::isfinite(l), "tensor-variant loss finite on hybrid MoE");
    printf("  tensor-variant loss=%.4f\n", l);

    // --- Tokenizer encode path (BPE) ---
    auto enc_ids = tok.encode("hello world");
    TEST_CHECK(!enc_ids.empty(), "BPE encodes prompt to ids");

    printf("\nRLL E2E TESTS DONE. (GRPO x HybridMoeModel x BPETokenizer)\n");
    int fails = TEST_REPORT();
    return fails > 0 ? 1 : 0;
}
