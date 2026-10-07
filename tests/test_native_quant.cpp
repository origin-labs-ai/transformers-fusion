// test_native_quant.cpp — NativeQWeightStore + NativeQUANTTrainer
#include "quant/native_trainer.h"
#include "quant/model.h"
#include "quant/transformer.h"
#include "quant/tensor.h"
#include "quant/test.h"
#include <vector>
#include <cmath>
#include <cstdio>

using namespace quant;

int main() {
    TEST_SUITE("native_quant");
    printf("=== Native QUANT training test ===\n\n");

    TransformerConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_size = 16;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.head_dim = cfg.hidden_size / cfg.num_heads;
    cfg.ffn_hidden_size = 32;
    cfg.norm_eps = 1e-5f;
    cfg.max_seq_len = 32;

    DenseModel model(cfg);
    TEST_CHECK(model.param_count() > 0, "model has parameters");

    native::NativeTrainConfig ncfg;
    ncfg.block_size = 64;
    ncfg.warmup_steps = 2;
    ncfg.max_steps = 60;
    ncfg.lr_scale = 0.2f;
    ncfg.lr_weight = 2.0f;
    ncfg.log_interval = 2;

    // Learnable periodic pattern (next-token shifted)
    std::vector<std::vector<float>> train_data, train_targets;
    for (size_t i = 0; i < 4; i++) {
        std::vector<float> seq((size_t)cfg.max_seq_len);
        for (size_t s = 0; s < (size_t)cfg.max_seq_len; s++)
            seq[s] = (float)((s + i) % (size_t)cfg.vocab_size);
        train_data.push_back(seq);
    }
    train_targets = train_data;
    for (auto& t : train_targets)
        for (size_t s = 0; s + 1 < t.size(); s++)
            t[s] = t[s + 1];

    native::NativeQUANTTrainer trainer(&model, ncfg);
    trainer.warmup_phase(train_data);

    double initial_loss = 0.0, final_loss = 0.0;
    for (size_t step = 0; step < ncfg.max_steps; step++) {
        auto& seq = train_data[step % train_data.size()];
        auto& tgt = train_targets[step % train_targets.size()];
        auto m = trainer.train_step(seq.data(), tgt.data(), 1, seq.size());
        if (step == 0) initial_loss = m.loss;
        if (step == ncfg.max_steps - 1) final_loss = m.loss;
    }
    printf("  native training loss: %.4f -> %.4f\n", initial_loss, final_loss);
    TEST_CHECK(std::isfinite((float)final_loss), "native training loss finite");
    TEST_CHECK(final_loss < initial_loss, "native training reduces loss");

    auto& ws = trainer.weight_store();
    TEST_CHECK(ws.size() > 0, "weight store populated");
    TEST_CHECK(trainer.config().block_size == 64, "trainer config roundtrip");

    int failures = TEST_REPORT();
    printf("\nNATIVE QUANT TEST %s\n", failures == 0 ? "PASSED" : "FAILED");
    return failures > 0 ? 1 : 0;
}
