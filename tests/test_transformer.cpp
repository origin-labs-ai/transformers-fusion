#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
#include "quant/model.h"
#include "quant/transformer.h"
#include "quant/kv_cache.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

static bool all_finite(const quant::Tensor& t) {
    const float* d = t.data<float>();
    for (int64_t i = 0; i < t.numel(); i++) {
        if (!std::isfinite(d[i])) return false;
    }
    return true;
}

static quant::TransformerConfig tiny_cfg() {
    quant::TransformerConfig cfg;
    cfg.vocab_size = 32;
    cfg.hidden_size = 16;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.head_dim = 8;
    cfg.ffn_hidden_size = 32;
    cfg.max_seq_len = 16;
    return cfg;
}

static void fill_ids_positions(quant::Tensor& ids, quant::Tensor& pos) {
    int32_t* id_data = ids.data<int32_t>();
    int32_t* pos_data = pos.data<int32_t>();
    id_data[0] = 0; id_data[1] = 1;
    pos_data[0] = 0; pos_data[1] = 1;
}

void test_forward_shape_finite() {
    quant::TransformerConfig cfg = tiny_cfg();
    quant::DenseModel model(cfg);

    quant::Tensor input_ids(quant::Shape{1, 2}, quant::DType::I32);
    quant::Tensor positions(quant::Shape{1, 2}, quant::DType::I32);
    fill_ids_positions(input_ids, positions);

    quant::Tensor logits = model.forward(input_ids, positions);
    assert(logits.shape() == quant::Shape({1, 2, 32}));
    assert(all_finite(logits));
}

void test_gqa_forward() {
    quant::TransformerConfig cfg = tiny_cfg();
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 4;  // 4 heads * 4 dim = 16 == hidden_size
    quant::DenseModel model(cfg);
    assert(model.layers[0]->attention.num_heads == 4);
    assert(model.layers[0]->attention.num_kv_heads == 2);

    // GQA cache must be sized by num_kv_heads (see engines/inference/inference.cpp).
    quant::KVCache cache;
    cache.init((int)cfg.num_layers, cfg.max_seq_len, cfg.num_kv_heads, cfg.head_dim);

    quant::Tensor input_ids(quant::Shape{1, 2}, quant::DType::I32);
    quant::Tensor positions(quant::Shape{1, 2}, quant::DType::I32);
    fill_ids_positions(input_ids, positions);

    quant::Tensor logits = model.forward(input_ids, positions, &cache);
    assert(logits.shape() == quant::Shape({1, 2, 32}));
    assert(all_finite(logits));
    assert(cache.context_len(0) == 2);
}

void test_rope_linear_flag() {
    quant::TransformerConfig cfg = tiny_cfg();
    cfg.rope_scaling_mode = quant::RoPEScalingMode::Linear;
    cfg.rope_scaling_factor = 4.0f;
    quant::DenseModel model(cfg);

    const quant::RotaryEmbedding& rope = model.layers[0]->attention.rope;
    assert(rope.scaling_mode == quant::RoPEScalingMode::Linear);
    assert(std::abs(rope.scaling_factor - 4.0f) < 1e-6f);
    assert(std::abs(rope.attn_scale_mult() - 1.0f) < 1e-6f);
    assert(rope.cos_cached.dim(0) == cfg.max_seq_len);
    assert(rope.cos_cached.dim(1) == cfg.head_dim / 2);
    assert(all_finite(rope.cos_cached));
    assert(all_finite(rope.sin_cached));

    quant::Tensor input_ids(quant::Shape{1, 2}, quant::DType::I32);
    quant::Tensor positions(quant::Shape{1, 2}, quant::DType::I32);
    fill_ids_positions(input_ids, positions);
    quant::Tensor logits = model.forward(input_ids, positions);
    assert(all_finite(logits));
}

void test_rope_ntk_flag() {
    quant::TransformerConfig cfg = tiny_cfg();
    cfg.rope_scaling_mode = quant::RoPEScalingMode::NTK;
    cfg.rope_scaling_factor = 2.0f;
    quant::DenseModel model(cfg);

    const quant::RotaryEmbedding& rope = model.layers[0]->attention.rope;
    assert(rope.scaling_mode == quant::RoPEScalingMode::NTK);
    assert(all_finite(rope.cos_cached));
    assert(all_finite(rope.sin_cached));

    quant::Tensor input_ids(quant::Shape{1, 2}, quant::DType::I32);
    quant::Tensor positions(quant::Shape{1, 2}, quant::DType::I32);
    fill_ids_positions(input_ids, positions);
    quant::Tensor logits = model.forward(input_ids, positions);
    assert(all_finite(logits));
}

void test_rope_yarn_flags() {
    quant::TransformerConfig cfg = tiny_cfg();
    cfg.max_seq_len = 32;
    cfg.rope_scaling_mode = quant::RoPEScalingMode::YARN;
    cfg.rope_scaling_factor = 4.0f;
    cfg.rope_original_max_seq_len = 8;
    cfg.yarn_beta_fast = 32.0f;
    cfg.yarn_beta_slow = 1.0f;
    cfg.yarn_attn_factor = 1.5f;
    quant::DenseModel model(cfg);

    const quant::RotaryEmbedding& rope = model.layers[0]->attention.rope;
    assert(rope.scaling_mode == quant::RoPEScalingMode::YARN);
    assert(rope.mscale > 1.0f);
    assert(rope.attn_scale_mult() > 1.0f);
    assert(all_finite(rope.cos_cached));
    assert(all_finite(rope.sin_cached));

    quant::Tensor input_ids(quant::Shape{1, 2}, quant::DType::I32);
    quant::Tensor positions(quant::Shape{1, 2}, quant::DType::I32);
    fill_ids_positions(input_ids, positions);
    quant::Tensor logits = model.forward(input_ids, positions);
    assert(logits.shape() == quant::Shape({1, 2, 32}));
    assert(all_finite(logits));
}

void test_rotary_apply_finite() {
    quant::RotaryEmbedding rope(8, 16, 10000.0f);
    quant::Tensor x(quant::Shape{1, 2, 4, 8}, quant::DType::F32);
    float* d = x.data<float>();
    for (int64_t i = 0; i < x.numel(); i++) d[i] = (float)(i % 7) * 0.25f - 0.5f;
    rope.apply(x, 0, 4);
    assert(all_finite(x));
}

int main() {
    std::cout << "[Test] Running Transformer test..." << std::endl;
    test_forward_shape_finite();
    test_gqa_forward();
    test_rope_linear_flag();
    test_rope_ntk_flag();
    test_rope_yarn_flags();
    test_rotary_apply_finite();
    std::cout << "Transformer test passed!" << std::endl;
    return 0;
}
