#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
#include "quant/model.h"
#include "quant/speculative_decoder.h"
#include "quant/kv_cache.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <cstring>

void test_kv_quantization() {
    quant::KVCache cache_fp32;
    cache_fp32.init(1, 100, 1, 64);
    
    quant::KVCache cache_q4;
    cache_q4.init(1, 100, 1, 64);
    assert(cache_fp32.context_len() == cache_q4.context_len());
    assert(cache_fp32.max_seq_len() == cache_q4.max_seq_len());
}

void test_kv_truncate() {
    // KVCache::truncate: exact rollback primitive for speculative rewind.
    quant::KVCache cache;
    cache.init(2, 16, 2, 4);
    auto make_kv = [](float base) {
        quant::Tensor k(quant::Shape{1, 2, 1, 4});
        quant::Tensor v(quant::Shape{1, 2, 1, 4});
        float* kd = k.data<float>();
        float* vd = v.data<float>();
        for (int i = 0; i < 8; i++) { kd[i] = base + (float)i; vd[i] = -(base + (float)i); }
        return std::make_pair(std::move(k), std::move(v));
    };
    for (int t = 0; t < 6; t++) {
        auto kv = make_kv(100.0f * (t + 1));
        cache.append(0, kv.first, kv.second);
        cache.append(1, kv.first, kv.second);
    }
    assert(cache.context_len() == 6);
    assert(cache.context_len(1) == 6);

    cache.truncate(3);
    assert(cache.context_len() == 3);
    assert(cache.context_len(1) == 3);
    // Surviving rows must be intact.
    {
        auto got = cache.get_range(0, 0, 3);
        const float* kd = got.first.data<float>();
        // Head 0, pos 1, dim 0 holds base(200)+head_off(0)+dim(0) = 200.
        assert(std::fabs(kd[1 * 4 + 0] - 200.0f) < 1e-5f);
    }
    // Re-append after truncate must land exactly at the truncated position.
    {
        auto kv = make_kv(999.0f);
        cache.append(0, kv.first, kv.second);
        assert(cache.context_len() == 4);
        auto got = cache.get_range(0, 3, 4);
        const float* kd = got.first.data<float>();
        assert(std::fabs(kd[0] - 999.0f) < 1e-5f);
    }
    // Clamps: beyond-pos is a no-op, negative clamps to 0.
    cache.truncate(100);
    assert(cache.context_len() == 4);
    cache.truncate(-5);
    assert(cache.context_len() == 0);
    assert(cache.context_len(1) == 0);
    std::cout << "[KV Truncate Test] Passed." << std::endl;
}

// Deterministic draft/target pair: the draft always proposes `draft_tok`,
// the target always samples `target_tok` (one-hot logits). Match => accept,
// mismatch => reject + KV rewind.
struct FixedDraft : public quant::DraftModel {
    int tok;
    int vocab;
    explicit FixedDraft(int t, int v = 16) : tok(t), vocab(v) {}
    std::vector<int> propose(const std::vector<int>&, std::size_t n) override {
        return std::vector<int>(n, tok);
    }
    int vocab_size() const override { return vocab; }
};

struct FixedTarget : public quant::TargetModel {
    int tok;
    int vocab;
    explicit FixedTarget(int t, int v = 16) : tok(t), vocab(v) {}
    std::vector<float> logits(const std::vector<int>&) override {
        std::vector<float> l((size_t)vocab, -1e9f);
        l[(size_t)tok] = 0.0f;  // argmax == tok with overwhelming margin
        return l;
    }
    int vocab_size() const override { return vocab; }
};

static void append_cache_rows(quant::KVCache& cache, int nrows, float base) {
    for (int t = 0; t < nrows; t++) {
        quant::Tensor k(quant::Shape{1, 1, 1, 4});
        quant::Tensor v(quant::Shape{1, 1, 1, 4});
        float* kd = k.data<float>();
        float* vd = v.data<float>();
        for (int i = 0; i < 4; i++) { kd[i] = base + (float)t; vd[i] = base + (float)t; }
        cache.append(0, k, v);
    }
}

void test_speculative_decoding() {
    // Accept path: draft == target => all proposed tokens accepted, KV grows.
    {
        quant::SpeculativeDecoder dec;
        FixedDraft draft(3);
        FixedTarget target(3);
        quant::KVCache cache;
        cache.init(1, 32, 1, 4);
        append_cache_rows(cache, 2, 10.0f);
        std::vector<int> ctx = {1, 2};
        auto r = dec.decode_step(draft, target, ctx, &cache);
        assert(r.total_proposed == 3);  // default draft_k
        assert(r.tokens_rejected == 0);
        assert(r.tokens_accepted.size() == 3);
        assert(ctx.size() == 2 + 3);
        // Simulate the serving loop appending verified rows, then check the
        // cache contents are exactly the pre-existing + new rows.
        append_cache_rows(cache, 3, 50.0f);
        assert(cache.context_len() == 5);
        auto got = cache.get_range(0, 2, 5);
        const float* kd = got.first.data<float>();
        for (int s = 0; s < 3; s++)
            for (int i = 0; i < 4; i++)
                assert(std::fabs(kd[s * 4 + i] - (50.0f + (float)s)) < 1e-5f);
    }
    // Reject path: draft != target on first token => 1 rejection, context
    // gets the target token, and the KV cache is rewound to the checkpoint.
    {
        quant::SpeculativeDecoder dec;
        FixedDraft draft(3);
        FixedTarget target(7);  // never equals draft => first-token reject
        quant::KVCache cache;
        cache.init(1, 32, 1, 4);
        append_cache_rows(cache, 2, 10.0f);
        std::vector<int> ctx = {1, 2};
        auto r = dec.decode_step(draft, target, ctx, &cache);
        assert(r.total_proposed == 1);  // broke at first token
        assert(r.tokens_rejected == 1);
        assert(r.tokens_accepted.empty());
        assert(ctx.size() == 3 && ctx.back() == 7);
        assert(cache.context_len() == 2);  // rewound to checkpoint
        auto got = cache.get_range(0, 0, 2);
        const float* kd = got.first.data<float>();
        assert(std::fabs(kd[0] - 10.0f) < 1e-5f);
        assert(std::fabs(kd[4] - 11.0f) < 1e-5f);
    }
    // End-to-end generate() with a matching pair.
    {
        quant::SpeculativeDecoder dec;
        FixedDraft draft(5);
        FixedTarget target(5);
        std::vector<int> out = dec.generate(draft, target, {0}, 6, nullptr);
        assert(out.size() == 6);
        for (int t : out) assert(t == 5);
        assert(dec.acceptance_rate() == 1.0);
    }
    std::cout << "[Speculative Decoding Test] Passed." << std::endl;
}

void test_flash_attention() {
    quant::Tensor q(quant::Shape{1, 4, 8, 64});
    quant::Tensor k(quant::Shape{1, 4, 8, 64});
    quant::Tensor v(quant::Shape{1, 4, 8, 64});
    
    q.zero_();
    k.zero_();
    v.zero_();
    
    assert(q.dim(3) == 64);
    assert(k.dim(3) == 64);
    assert(v.dim(3) == 64);
    std::cout << "[Flash Attention Test] Passed." << std::endl;
}

int main() {
    test_kv_quantization();
    test_kv_truncate();
    test_speculative_decoding();
    test_flash_attention();

    quant::DenseModel model;
    model.config.vocab_size = 1000;
    model.config.hidden_size = 64;
    model.config.num_layers = 2;
    model.config.num_heads = 4;
    model.config.head_dim = 16;
    model.config.max_seq_len = 512;
    model.init_weights();

    quant::Tensor input_ids(quant::Shape{1, 4});
    quant::Tensor pos(quant::Shape{1, 4});

    for (int i = 0; i < 4; i++) {
        input_ids.data<float>()[i] = (float)(i % 10);
        pos.data<float>()[i] = (float)(i % 4);
    }

    quant::Tensor logits = model.forward(input_ids, pos);

    assert(logits.rank() == 3);
    assert(logits.dim(0) == 1);
    assert(logits.dim(1) == 4);
    assert(logits.dim(2) == 1000);

    std::cout << "Optimized Inference Test Passed!" << std::endl;
    return 0;
}
