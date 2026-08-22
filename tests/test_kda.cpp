// test_kda.cpp — G-1: Kimi Delta Attention correctness proof tests
//
// Properties proven here:
//   P1 delta-overwrite : same key, new value → old value erased exactly
//                        (orthonormal key, decay=0 ⇒ clean algebra)
//   P2 orthogonality   : query with an orthogonal key reads ≈ 0
//   P3 decay-forget    : old writes shrink geometrically with decay
//   P4 locality        : decay≈0 ⇒ late outputs independent of early tokens
//   P5 shape/finite    : forward_naive contract on batched input
#include "quant/kda_attention.h"
#include "quant/random.h"
#include "quant/test.h"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace quant;

namespace {
constexpr float kTol = 2e-3f;

std::vector<float> readout(const std::vector<float>& state,
                           const std::vector<float>& q, int64_t dk, int64_t dv) {
    std::vector<float> o((size_t)dv, 0.0f);
    for (int64_t i = 0; i < dk; ++i)
        for (int64_t j = 0; j < dv; ++j)
            o[(size_t)j] += q[(size_t)i] * state[(size_t)i * dv + j];
    return o;
}
} // namespace

int main() {
    TEST_SUITE("kda");
    printf("=== KDA (G-1) correctness tests ===\n\n");

    // ---------------------------------------------------------------
    // P1 + P2: delta-overwrite & orthogonality on orthonormal keys
    // ---------------------------------------------------------------
    {
        printf("--- P1/P2: overwrite + orthogonality (decay=0, k=e1) ---\n");
        constexpr int64_t dk = 8, dv = 8;
        std::vector<float> state((size_t)dk * dv, 0.0f);

        std::vector<float> k1(dk, 0.0f);  k1[0] = 1.0f;           // e1
        std::vector<float> k2(dk, 0.0f);  k2[1] = 1.0f;           // e2 ⊥ e1
        std::vector<float> v1(dv), v2(dv);
        for (int64_t j = 0; j < dv; ++j) { v1[(size_t)j] = (float)(j + 1); v2[(size_t)j] = -(float)(j + 3); }

        KDAAttention::delta_step(state.data(), k1.data(), v1.data(),
                                 /*decay=*/0.0f, dk, dv);
        auto o1 = readout(state, k1, dk, dv);
        for (int64_t j = 0; j < dv; ++j)
            TEST_CHECK_CLOSE(o1[(size_t)j], v1[(size_t)j], kTol,
                             "first write is retrievable by its key");

        auto o_ortho = readout(state, k2, dk, dv);
        for (int64_t j = 0; j < dv; ++j)
            TEST_CHECK(std::fabs(o_ortho[(size_t)j]) < kTol,
                       "orthogonal key reads ~zero");

        // Overwrite the SAME slot with a different value:
        KDAAttention::delta_step(state.data(), k1.data(), v2.data(),
                                 /*decay=*/0.0f, dk, dv);
        auto o2 = readout(state, k1, dk, dv);
        for (int64_t j = 0; j < dv; ++j)
            TEST_CHECK_CLOSE(o2[(size_t)j], v2[(size_t)j], kTol,
                             "delta rule ERASES old value on re-write");
        printf("  overwrite + orthogonality verified\n");
    }

    // ---------------------------------------------------------------
    // P3: decay-forget — old write shrinks by decay^steps
    // ---------------------------------------------------------------
    {
        printf("--- P3: geometric forgetting ---\n");
        constexpr int64_t dk = 4, dv = 4;
        const float decay = 0.5f;
        std::vector<float> state((size_t)dk * dv, 0.0f);

        std::vector<float> k(dk, 0.0f); k[0] = 1.0f;
        std::vector<float> v(dv, 1.0f);
        KDAAttention::delta_step(state.data(), k.data(), v.data(), decay, dk, dv);

        // N no-op steps (k=0): state decays but nothing new written
        std::vector<float> zero_k(dk, 0.0f);
        for (int n = 0; n < 3; ++n)
            KDAAttention::delta_step(state.data(), zero_k.data(), v.data(), decay, dk, dv);

        auto o = readout(state, k, dk, dv);
        // Write lands undecayed (fresh state); each of the 3 no-op steps
        // then fades memory by one decay factor → decay^3.
        const float expect = decay * decay * decay; // 0.5^3
        for (int64_t j = 0; j < dv; ++j)
            TEST_CHECK_CLOSE(o[(size_t)j] / v[(size_t)j], expect, 1e-3f,
                             "old memory decays as decay^(post-write steps)");
        printf("  retention after write + 3 steps = %.4f (expected %.4f)\n",
               o[0], expect);
    }

    // ---------------------------------------------------------------
    // P4 + P5: full block — locality under tiny decay + shapes/finite
    // ---------------------------------------------------------------
    {
        printf("--- P4/P5: forward_naive locality + shapes ---\n");
        TransformerConfig cfg;
        cfg.vocab_size = 64;
        cfg.hidden_size = 32;
        cfg.num_layers = 1;
        cfg.num_heads = 2;
        cfg.head_dim = 16;
        cfg.ffn_hidden_size = 64;
        cfg.max_seq_len = 32;

        KDAAttention kda(cfg);
        TEST_CHECK(kda.param_count() > 0, "KDA has parameters");

        RNG rng(99);
        constexpr int B = 2, S = 12;
        Tensor a(Shape{B, S, cfg.hidden_size});
        Tensor b(Shape{B, S, cfg.hidden_size});
        for (int64_t i = 0; i < a.numel(); ++i) {
            a.data<float>()[i] = (rng.uniform() - 0.5f) * 0.5f;
            b.data<float>()[i] = a.data<float>()[i];
        }
        // b differs from a ONLY at position 0 of batch 0
        for (int64_t h = 0; h < cfg.hidden_size; ++h)
            b.data<float>()[h] += 3.7f;

        // near-zero decay → instant forgetting
        for (int64_t h = 0; h < kda.num_heads; ++h)
            kda.log_decay.data<float>()[h] = -20.0f;

        Tensor oa = kda.forward_naive(a);
        Tensor ob = kda.forward_naive(b);

        TEST_CHECK(oa.shape().dims[0] == B && oa.shape().dims[1] == S &&
                   oa.shape().dims[2] == cfg.hidden_size,
                   "output shape {B,S,hidden}");
        bool finite = true;
        for (int64_t i = 0; i < oa.numel(); ++i)
            if (!std::isfinite(oa.data<float>()[i])) finite = false;
        TEST_CHECK(finite, "outputs finite");

        // positions >= 2 must be identical between the two sequences
        float max_diff_late = 0.0f;
        for (int64_t t = 2; t < S; ++t)
            for (int64_t h = 0; h < cfg.hidden_size; ++h)
                max_diff_late = std::max(
                    max_diff_late,
                    std::fabs(oa.data<float>()[(0 * S + t) * cfg.hidden_size + h] -
                              ob.data<float>()[(0 * S + t) * cfg.hidden_size + h]));
        TEST_CHECK(max_diff_late < 1e-3f,
                   "decay≈0: late outputs ignore early tokens (locality)");
        printf("  late-position max diff after pos-0 edit: %.6f\n",
               max_diff_late);
    }

    printf("\nKDA (G-1) TESTS PASSED!\n");
    return 0;
}
