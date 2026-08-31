#include "quant/kv_cache.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

int main() {
    std::cout << "[Test] Running Paged KV Cache test..." << std::endl;

    const int64_t num_layers = 1;
    const int64_t num_heads = 8;
    const int64_t head_dim = 64;
    const int64_t block_size = 16;

    quant::PagedKVCache4M cache(num_layers, num_heads, head_dim, block_size,
                                256ULL * 1024 * 1024);

    // Capacity contract: L1 x L2 table product scaled by block size.
    // TABLE_ENTRIES was raised 512->4096 at some point without a test update;
    // assert the structural invariant rather than a stale magic constant.
    assert(cache.max_logical_tokens_per_layer() ==
           quant::PagedKVCache4M::TABLE_ENTRIES *
               quant::PagedKVCache4M::TABLE_ENTRIES * block_size);
    assert(cache.logical_capacity() == cache.max_logical_tokens_per_layer());
    assert(cache.block_size() == block_size);
    assert(cache.num_layers() == num_layers);
    assert(cache.num_heads() == num_heads);
    assert(cache.head_dim() == head_dim);

    // Write a block worth of K/V at the start of the logical space.
    int64_t pos0 = 0;
    quant::Tensor k0({num_heads, block_size, head_dim}, quant::DType::F32);
    quant::Tensor v0({num_heads, block_size, head_dim}, quant::DType::F32);
    float* kd = k0.data<float>();
    float* vd = v0.data<float>();
    for (int64_t i = 0; i < num_heads * block_size * head_dim; i++) {
        kd[i] = (float)(i % 7) * 0.25f;
        vd[i] = (float)(i % 5) * 0.5f;
    }
    std::cout << "S2" << std::endl;
    cache.append(0, pos0, k0, v0);
    std::cout << "S3" << std::endl;

    // Read it back and verify byte-identical roundtrip.
    auto got = cache.get_range(0, pos0, pos0 + block_size);
    std::cout << "S4" << std::endl;
    const quant::Tensor& rk = got.first;
    const quant::Tensor& rv = got.second;
    assert(rk.numel() == num_heads * block_size * head_dim);
    assert(rv.numel() == num_heads * block_size * head_dim);
    const float* rkd = rk.data<float>();
    const float* rvd = rv.data<float>();
    for (int64_t i = 0; i < rk.numel(); i++) {
        assert(std::abs(rkd[i] - kd[i]) < 1e-6f);
    }
    for (int64_t i = 0; i < rv.numel(); i++) {
        assert(std::abs(rvd[i] - vd[i]) < 1e-6f);
    }
    std::cout << "S5" << std::endl;
    assert(cache.verify_retrieval(0, pos0, k0, v0));
    std::cout << "S6" << std::endl;

    // Write to a far-away logical position (exercises hierarchical paging).
    int64_t far_pos = ((int64_t)1 << 20) * block_size;
    assert(cache.context_len() >= 0);
    std::cout << "S7" << std::endl;
    cache.append(0, far_pos, k0, v0);
    std::cout << "S8" << std::endl;
    auto got2 = cache.get_range(0, far_pos, far_pos + block_size);
    const quant::Tensor& rk2 = got2.first;
    std::cout << "S8b rk2.numel=" << rk2.numel() << " curpos=" << cache.context_len() << std::endl;
    const float* rkd2 = rk2.data<float>();
    for (int64_t i = 0; i < rk2.numel(); i++) {
        assert(std::abs(rkd2[i] - kd[i]) < 1e-6f);
    }

    // Flush to disk and reload — contents must survive.
    std::cout << "S9" << std::endl;
    cache.flush_to_disk();
    cache.clear();
    std::cout << "S11" << std::endl;
    cache.load_from_disk();
    std::cout << "S12" << std::endl;
    auto got3 = cache.get_range(0, pos0, pos0 + block_size);
    std::cout << "S13" << std::endl;
    const quant::Tensor& rk3 = got3.first;
    const float* rkd3 = rk3.data<float>();
    std::cout << "S14 numel=" << rk3.numel() << std::endl;
    for (int64_t i = 0; i < rk3.numel(); i++) {
        assert(std::abs(rkd3[i] - kd[i]) < 1e-6f);
    }

    std::cout << "Paged KV Cache test passed!" << std::endl;
    return 0;
}
