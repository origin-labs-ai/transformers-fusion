#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
#include "quant/generator.h"
#include "quant/moe_model.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_batch_inference() {
    quant::TransformerConfig cfg = quant::MoEModel::config_100B();
    cfg.num_layers = 2; // small for test
    quant::moe::MoEAllConfig moe_cfg = quant::MoEModel::moe_config_100B();
    moe_cfg.num_experts = 4;
    quant::MoEModel model(cfg, moe_cfg);

    int64_t B = 2; // batch size
    int64_t S = 4; // seq len
    quant::Tensor input_ids(quant::Shape{B, S}, quant::DType::I32);
    int32_t* id_data = input_ids.data<int32_t>();
    for (int i = 0; i < B*S; ++i) id_data[i] = i % cfg.vocab_size;

    quant::Tensor positions(quant::Shape{B, S}, quant::DType::I32);
    int32_t* pos_data = positions.data<int32_t>();
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
            pos_data[b*S + s] = s;
        }
    }

    quant::KVCache cache;
    cache.init((int)cfg.num_layers, cfg.max_seq_len, cfg.num_heads, cfg.head_dim);

    quant::Tensor logits = model.forward(input_ids, positions, &cache);
    assert(logits.shape() == quant::Shape({B, S, cfg.vocab_size}));

    // Verify cache blocks
    // Just ensure it ran without crashing and outputs are finite
    const float* l_data = logits.data<float>();
    for (int i = 0; i < B*S*cfg.vocab_size; ++i) {
        if (!std::isfinite(l_data[i])) {
            assert(false && "Logits contain NaN or Inf");
        }
    }
}

int main() {
    std::cout << "[Test] Running Batch Inference test..." << std::endl;
    test_batch_inference();
    std::cout << "Batch Inference Test Passed!" << std::endl;
    return 0;
}
