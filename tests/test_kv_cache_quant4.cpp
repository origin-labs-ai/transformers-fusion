#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
#include "quant/kv_cache_quant4.h"
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

static float max_abs_err(const float* a, const float* b, int64_t n) {
    float m = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float e = std::abs(a[i] - b[i]);
        if (e > m) m = e;
    }
    return m;
}

// Append helper: QUANT4KVCache::append reads k.shape().dims[2] as seq_len and
// expects at least seq_len * num_heads * head_dim floats (layout {1,H,S,D}).
static void append_tokens(quant::QUANT4KVCache& cache, int layer,
                          const std::vector<float>& k_all,
                          const std::vector<float>& v_all,
                          int64_t num_heads, int64_t head_dim, int64_t seq_len) {
    quant::Tensor k(quant::Shape{1, num_heads, seq_len, head_dim}, quant::DType::F32);
    quant::Tensor v(quant::Shape{1, num_heads, seq_len, head_dim}, quant::DType::F32);
    float* kd = k.data<float>();
    float* vd = v.data<float>();
    for (int64_t i = 0; i < seq_len * num_heads * head_dim; i++) {
        kd[i] = k_all[(size_t)i];
        vd[i] = v_all[(size_t)i];
    }
    cache.append(layer, k, v);
}

void test_block_roundtrip_full() {
    const int64_t n = 32;
    std::vector<float> src((size_t)n);
    for (int64_t i = 0; i < n; i++) src[i] = (float)(i % 9 - 4) * 0.25f;
    std::vector<uint8_t> idx((size_t)n, 0);
    std::vector<uint16_t> cb(16, 0);
    quant::QUANT4KVCache::quantize_block_quant4(src.data(), idx.data(), cb.data(), n);
    for (int64_t i = 0; i < n; i++) assert((idx[(size_t)i] & 0xF0) == 0);
    std::vector<float> dst((size_t)n, 0.0f);
    quant::QUANT4KVCache::dequantize_block_quant4(idx.data(), cb.data(), dst.data(), n);
    for (int64_t i = 0; i < n; i++) assert(std::isfinite(dst[(size_t)i]));
    assert(max_abs_err(src.data(), dst.data(), n) < 0.6f);
}

void test_block_roundtrip_partial() {
    // Partial-block edges: odd and tiny tails (n not a multiple of 2 / block).
    for (int64_t n : {1, 7, 31}) {
        std::vector<float> src((size_t)n);
        for (int64_t i = 0; i < n; i++) src[i] = (float)(i % 5 - 2) * 0.3f + 0.1f;
        std::vector<uint8_t> idx((size_t)n, 0);
        std::vector<uint16_t> cb(16, 0);
        quant::QUANT4KVCache::quantize_block_quant4(src.data(), idx.data(), cb.data(), n);
        std::vector<float> dst((size_t)n, 0.0f);
        quant::QUANT4KVCache::dequantize_block_quant4(idx.data(), cb.data(), dst.data(), n);
        for (int64_t i = 0; i < n; i++) assert(std::isfinite(dst[(size_t)i]));
        assert(max_abs_err(src.data(), dst.data(), n) < 0.6f);
    }
    // Constant block must not produce NaN (min==max fast path).
    {
        std::vector<float> src(7, 0.5f);
        std::vector<uint8_t> idx(7, 0);
        std::vector<uint16_t> cb(16, 0);
        quant::QUANT4KVCache::quantize_block_quant4(src.data(), idx.data(), cb.data(), 7);
        std::vector<float> dst(7, 0.0f);
        quant::QUANT4KVCache::dequantize_block_quant4(idx.data(), cb.data(), dst.data(), 7);
        for (int i = 0; i < 7; i++) assert(std::isfinite(dst[(size_t)i]));
    }
}

void test_append_singletons_partial() {
    // per_token=12 does not divide block_size=32: each singleton lands at a
    // partial-block offset (lines 263-267: cur_offset != 0 codebook path).
    const int64_t H = 2, D = 6, per_token = H * D;
    quant::QUANT4KVCache cache(1, 16, H, D, 32);
    for (int t = 0; t < 3; t++) {
        std::vector<float> k((size_t)per_token), v((size_t)per_token);
        for (int64_t i = 0; i < per_token; i++) {
            k[(size_t)i] = (float)((t * per_token + i) % 9 - 4) * 0.25f;
            v[(size_t)i] = (float)((t * per_token + i) % 7 - 3) * 0.2f;
        }
        append_tokens(cache, 0, k, v, H, D, 1);
        assert(cache.context_len() == t + 1);
    }
    auto got = cache.get_range(0, 0, 3);
    assert(got.first.numel() == 3 * per_token);
    assert(got.second.numel() == 3 * per_token);
    assert(all_finite(got.first));
    assert(all_finite(got.second));
}

void test_append_straddles_blocks() {
    // 5 tokens * 12 elems = 60 elems straddles the 32-elem block boundary
    // (chunk split across cur_block_idx / cur_block_idx+1).
    const int64_t H = 2, D = 6, S = 5;
    quant::QUANT4KVCache cache(1, 16, H, D, 32);
    std::vector<float> k((size_t)(S * H * D)), v((size_t)(S * H * D));
    for (int64_t i = 0; i < S * H * D; i++) {
        k[(size_t)i] = (float)(i % 9 - 4) * 0.25f;
        v[(size_t)i] = (float)(i % 7 - 3) * 0.2f;
    }
    append_tokens(cache, 0, k, v, H, D, S);
    assert(cache.context_len() == S);
    auto got = cache.get_all(0);
    assert(got.first.numel() == S * H * D);
    assert(all_finite(got.first));
    assert(all_finite(got.second));
    assert(max_abs_err(k.data(), got.first.data<float>(), S * H * D) < 3.0f);
    assert(cache.size_bytes() > 0);
    assert(cache.compression_ratio() > 0.0);
}

void test_odd_per_token_offsets() {
    // Odd per_token=7 hits the cur_offset/2 byte-truncation edge for packed
    // 4-bit indices; append must stay finite without OOB.
    const int64_t H = 1, D = 7, S = 3;
    quant::QUANT4KVCache cache(1, 16, H, D, 32);
    std::vector<float> k((size_t)(S * H * D)), v((size_t)(S * H * D));
    for (int64_t i = 0; i < S * H * D; i++) {
        k[(size_t)i] = (float)(i % 5 - 2) * 0.3f;
        v[(size_t)i] = (float)(i % 5 - 2) * -0.2f;
    }
    append_tokens(cache, 0, k, v, H, D, S);
    assert(cache.context_len() == S);
    auto got = cache.get_range(0, 0, S);
    assert(all_finite(got.first));
    assert(all_finite(got.second));
    auto mid = cache.get_range(0, 1, 3);
    assert(mid.first.numel() == 2 * H * D);
    assert(all_finite(mid.first));
}

void test_tail_partial_and_clear() {
    // Total 5*12=60 elems over 32-elem blocks leaves a partial tail block.
    const int64_t H = 2, D = 6, S = 5;
    quant::QUANT4KVCache cache(1, 5, H, D, 32);
    assert(cache.block_size() == 32);
    std::vector<float> k((size_t)(S * H * D)), v((size_t)(S * H * D));
    for (int64_t i = 0; i < S * H * D; i++) {
        k[(size_t)i] = (float)(i % 9 - 4) * 0.25f;
        v[(size_t)i] = (float)(i % 7 - 3) * 0.2f;
    }
    append_tokens(cache, 0, k, v, H, D, S);
    assert(cache.context_len() == S);
    assert(cache.max_seq_len() == 5);
    auto got = cache.get_all(0);
    assert(all_finite(got.first));
    cache.clear();
    assert(cache.context_len() == 0);
}

int main() {
    std::cout << "[Test] Running QUANT4KVCache test..." << std::endl;
    test_block_roundtrip_full();
    test_block_roundtrip_partial();
    test_append_singletons_partial();
    test_append_straddles_blocks();
    test_odd_per_token_offsets();
    test_tail_partial_and_clear();
    std::cout << "QUANT4KVCache test passed!" << std::endl;
    return 0;
}
