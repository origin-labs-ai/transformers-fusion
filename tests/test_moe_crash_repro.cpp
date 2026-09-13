#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
#include "quant/moe_model.h"
#include <iostream>
#include <cassert>

int main() {
    quant::moe::MoEAllConfig moe_cfg;
    moe_cfg.num_experts = 4;
    moe_cfg.top_k = 2;
    moe_cfg.expert_hidden_size = 128;

    quant::TransformerConfig cfg;
    cfg.vocab_size = 100;
    cfg.hidden_size = 16;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.head_dim = 8;
    cfg.max_seq_len = 64;

    quant::MoEModel model(cfg, moe_cfg);
    assert(model.param_count() > 0);
    
    // Normal forward pass
    quant::Tensor input_ids(quant::Shape{2, 5}, quant::DType::F32);
    quant::Tensor pos(quant::Shape{2, 5}, quant::DType::F32);
    for (int i = 0; i < 10; i++) {
        input_ids.data<float>()[i] = 10.0f;
        pos.data<float>()[i] = (float)(i % 5);
    }
    quant::Tensor logits = model.forward(input_ids, pos);
    assert(logits.rank() == 3);
    assert(logits.dim(0) == 2);
    assert(logits.dim(1) == 5);
    assert(logits.dim(2) == 100);
    
    // Empty sequence should not crash, might throw or return empty tensor
    try {
        quant::Tensor empty_input(quant::Shape{0, 0}, quant::DType::F32);
        quant::Tensor empty_pos(quant::Shape{0, 0}, quant::DType::F32);
        quant::Tensor empty_logits = model.forward(empty_input, empty_pos);
    } catch (const std::exception&) {
        // Expected or caught gracefully
    }
    
    // Invalid sequence length should not crash (too many tokens)
    try {
        quant::Tensor large_input(quant::Shape{1, 128}, quant::DType::F32);
        quant::Tensor large_pos(quant::Shape{1, 128}, quant::DType::F32);
        quant::Tensor large_logits = model.forward(large_input, large_pos);
    } catch (const std::exception&) {
        // Caught gracefully due to max_seq_len
    }

    std::cout << "MoE Crash Repro test passed!" << std::endl;
    return 0;
}
