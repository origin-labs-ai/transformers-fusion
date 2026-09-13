// Phase 5-F6 split of src/inference/inference_opt.cpp (verbatim move, no behavior change).
// This file: INT8Inference (D8) + FP8Inference (D9) + ModelShard (D10).
// Declarations stay in include/quant/inference_opt.h.
#include "quant/inference_opt.h"
#include "quant/math.h"
#include "quant/int8_quant.h"
#include "quant/random.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <random>
#include <sstream>
#include <cstdio>
namespace quant {

// ===========================================================================
// D8-D9: INT8/FP8 inference
// ===========================================================================
INT8Inference::INT8Inference(Model* model) : model_(model) {}

Tensor INT8Inference::forward(const Tensor& input, const Tensor& positions) {
    int64_t n = input.numel();
    int64_t rows = input.rank() > 1 ? input.dim(0) : 1;
    int64_t cols = n / rows;

    // Quantize input to INT8 per-token
    std::vector<int8_t> q_data((size_t)n);
    auto params = quantize_per_token(input.data<float>(), q_data.data(), rows, cols);

    // Dequantize back (simulating INT8 compute pipeline)
    Tensor quant_input(input.shape());
    float* qd = quant_input.data<float>();
    for (int64_t r = 0; r < rows; r++) {
        dequantize_per_tensor(q_data.data() + r * cols,
                              qd + r * cols, cols, params[(size_t)r].inv_scale);
    }

    is_quantized_ = true;
    if (model_) return model_->forward(quant_input, positions);
    return quant_input;
}

FP8Inference::FP8Inference(Model* model) : model_(model) {}

Tensor FP8Inference::forward(const Tensor& input, const Tensor& positions) {
    int64_t n = input.numel();
    int64_t num_blocks = (n + KVCache::FP8_BLOCK_SIZE - 1) / KVCache::FP8_BLOCK_SIZE;

    std::vector<uint8_t> fp8_data((size_t)n);
    std::vector<float> scales((size_t)num_blocks);
    const float* src = input.data<float>();

    for (int64_t blk = 0; blk < num_blocks; blk++) {
        int64_t blk_start = blk * KVCache::FP8_BLOCK_SIZE;
        int64_t blk_end = std::min(blk_start + KVCache::FP8_BLOCK_SIZE, n);
        KVCache::quantize_fp8_block(src + blk_start, fp8_data.data() + blk_start,
                                     &scales[(size_t)blk], blk_end - blk_start);
    }

    Tensor fp8_input(input.shape());
    float* dst = fp8_input.data<float>();
    for (int64_t blk = 0; blk < num_blocks; blk++) {
        int64_t blk_start = blk * KVCache::FP8_BLOCK_SIZE;
        int64_t blk_end = std::min(blk_start + KVCache::FP8_BLOCK_SIZE, n);
        KVCache::dequantize_fp8_block(fp8_data.data() + blk_start, scales[(size_t)blk],
                                       dst + blk_start, blk_end - blk_start);
    }

    if (model_) return model_->forward(fp8_input, positions);
    return fp8_input;
}

void FP8Inference::fp8_residual_accum(const float* src, const uint8_t* fp8_data,
                                       const float* scales, int64_t num_blocks,
                                       float* out, int64_t n) {
    std::vector<float> residual((size_t)n, 0.0f);
    int64_t blk_size = KVCache::FP8_BLOCK_SIZE;
    float* r = residual.data();
    for (int64_t blk = 0; blk < num_blocks; blk++) {
        int64_t blk_start = blk * blk_size;
        int64_t blk_end = std::min(blk_start + blk_size, n);
        int64_t count = blk_end - blk_start;
        // dequant this block
        float scale = scales[(size_t)blk];
        const uint8_t* fp8_blk = fp8_data + (size_t)blk_start;
        for (int64_t i = 0; i < count; i++) {
            float deq = ((float)(int8_t)fp8_blk[i] - 0.0f) * scale / 127.0f + 0.0f;
            // residual = original - dequantized (rounding error)
            r[(size_t)(blk_start + i)] = src[(size_t)(blk_start + i)] - deq;
            out[(size_t)(blk_start + i)] = deq;
        }
    }
    // Accumulate running residual: add previous block's residual into current
    // This absorbs the FP8 rounding error across blocks
    float running_residual = 0.0f;
    for (int64_t blk = 0; blk < num_blocks; blk++) {
        int64_t blk_start = blk * blk_size;
        int64_t blk_end = std::min(blk_start + blk_size, n);
        for (int64_t i = blk_start; i < blk_end; i++) {
            running_residual += r[(size_t)i];
            out[(size_t)i] += running_residual * 0.5f; // dampened residual correction
            running_residual *= 0.5f; // decay the residual
        }
    }
}

Tensor FP8Inference::forward_two_stage(const Tensor& input, const Tensor& positions) {
    int64_t n = input.numel();
    int64_t num_blocks = (n + KVCache::FP8_BLOCK_SIZE - 1) / KVCache::FP8_BLOCK_SIZE;

    std::vector<uint8_t> fp8_data((size_t)n);
    std::vector<float> scales((size_t)num_blocks);
    const float* src = input.data<float>();

    for (int64_t blk = 0; blk < num_blocks; blk++) {
        int64_t blk_start = blk * KVCache::FP8_BLOCK_SIZE;
        int64_t blk_end = std::min(blk_start + KVCache::FP8_BLOCK_SIZE, n);
        KVCache::quantize_fp8_block(src + blk_start, fp8_data.data() + blk_start,
                                     &scales[(size_t)blk], blk_end - blk_start);
    }

    Tensor result(input.shape());
    float* dst = result.data<float>();
    fp8_residual_accum(src, fp8_data.data(), scales.data(), num_blocks, dst, n);

    if (model_) return model_->forward(result, positions);
    return result;
}

// ===========================================================================
// D10: Model sharding — run only specified layers
// ===========================================================================
ModelShard::ModelShard(Model* model, int64_t start_layer, int64_t end_layer)
    : model_(model), start_(start_layer), end_(end_layer) {}

Tensor ModelShard::forward(const Tensor& input) {
    auto* dm = dynamic_cast<DenseModel*>(model_);
    if (!dm) return input;

    int64_t B = input.dim(0);
    int64_t S = input.rank() > 1 ? input.dim(1) : 1;

    Tensor x = input;

    // Build positions tensor and causal mask
    Tensor positions(Shape{B, S});
    for (int64_t b = 0; b < B; b++)
        for (int64_t s = 0; s < S; s++)
            positions.data<float>()[b * S + s] = (float)s;

    Tensor mask(Shape{1, 1, S, S});
    mask.fill(-INFINITY);
    for (int64_t i = 0; i < S; i++)
        for (int64_t j = 0; j <= i; j++)
            mask.data<float>()[i * S + j] = 0.0f;

    int64_t n_layers = (int64_t)dm->layers.size();
    int64_t lo = std::max((int64_t)0, start_);
    int64_t hi = std::min(end_, n_layers);
    int n_shard_layers = (int)(hi - lo);

    if (n_shard_layers > 0) {
        KVCache cache(n_shard_layers, model_->config.max_seq_len,
                      model_->config.num_heads, model_->config.head_dim);
        int layer_idx = 0;
        for (int64_t l = lo; l < hi; l++) {
            x = dm->layers[(size_t)l]->forward(x, positions, mask, cache, layer_idx++);
        }
    }

    return x;
}

} // namespace quant
