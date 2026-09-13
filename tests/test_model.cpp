#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
#include "quant/model.h"
#include <iostream>
#include <cassert>
#include <vector>

void test_dense_model() {
    quant::TransformerConfig cfg;
    cfg.vocab_size = 100;
    cfg.hidden_size = 16;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.head_dim = 8;
    cfg.max_seq_len = 10;

    quant::DenseModel model(cfg);

    quant::Tensor input_ids(quant::Shape{1, 2}, quant::DType::I32);
    int32_t* id_data = input_ids.data<int32_t>();
    id_data[0] = 0; id_data[1] = 1;

    quant::Tensor positions(quant::Shape{1, 2}, quant::DType::I32);
    int32_t* pos_data = positions.data<int32_t>();
    pos_data[0] = 0; pos_data[1] = 1;

    quant::Tensor logits = model.forward(input_ids, positions);
    assert(logits.shape() == quant::Shape({1, 2, 100}));
    
    // Check serialization
    model.save("test_model.quant");
    quant::DenseModel model2(cfg);
    model2.load("test_model.quant");

    // Test parameter iteration
    int param_count = 0;
    for (auto& layer : model.layers) {
        param_count += layer->attention_norm.weight.numel();
        param_count += layer->ffn_norm.weight.numel();
    }
    assert(param_count > 0);
}

int main() {
    std::cout << "[Test] Running Model test..." << std::endl;
    test_dense_model();
    std::cout << "Model test passed!" << std::endl;
    return 0;
}
