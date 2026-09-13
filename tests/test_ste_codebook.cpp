#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
#include "quant/ste_quantizer.h"
#include "quant/tensor.h"
#include "quant/codebook.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace quant;

int main() {
    // Regression test for static-codebook bug: two tensors with different distributions must produce different codebooks
    // Before fix, static CodebookQUANT8 reused first tensor's codebook for second tensor (order dependent)
    const int N = 1024;
    Tensor t1(Shape{N}, DType::F32);
    Tensor t2(Shape{N}, DType::F32);
    float* d1 = t1.data<float>();
    float* d2 = t2.data<float>();
    // t1: gaussian-like small scale, t2: large scale — codebooks must differ
    // Use deterministic pseudo-random to ensure many distinct values >256
    for (int i = 0; i < N; ++i) {
        // Simple LCG-like deterministic
        float r = (float)((i * 1664525u + 1013904223u) % 100000) / 100000.0f;
        d1[i] = (r - 0.5f) * 2.0f;          // range [-1,1]
        d2[i] = (r - 0.5f) * 20.0f + 5.0f;  // range [-5,15] shifted
    }

    STEQuantizer q8(Format::Q8);
    Tensor r1 = q8.forward(t1);
    Tensor r2 = q8.forward(t2);

    // Verify quantization actually reduces precision (not passthrough)
    float diff1 = 0, diff2 = 0;
    for (int i = 0; i < N; ++i) {
        diff1 += std::fabs(r1.data<float>()[i] - d1[i]);
        diff2 += std::fabs(r2.data<float>()[i] - d2[i]);
    }
    // Both should have some error due to quantization
    assert(diff1 > 1e-6f && "Q8 quantization should introduce error");
    assert(diff2 > 1e-6f && "Q8 quantization should introduce error");

    // Order independence: forward(t1) then forward(t2) vs forward(t2) then forward(t1).
    // Test via second quantizer with reversed order (per-tensor, order-independent).
    STEQuantizer q8_b(Format::Q8);
    Tensor r2_first = q8_b.forward(t2);
    Tensor r1_second = q8_b.forward(t1);
    float diff_order = 0;
    for (int i = 0; i < N; ++i) diff_order += std::fabs(r1.data<float>()[i] - r1_second.data<float>()[i]);
    // With per-tensor codebook, r1 and r1_second should be very similar (both trained on t1)
    // With static bug, r1_second would use t2's codebook and differ significantly
    assert(diff_order < 1e-3f && "Order independence failed: static codebook bug still present");

    // Reverse-order check (r2): r2 (t1-then-t2) vs r2_first (t2-first) must also match.
    // With per-tensor codebooks both are trained on t2; with the static bug r2
    // would reuse t1's codebook and diverge from r2_first.
    float diff_order_r2 = 0;
    for (int i = 0; i < N; ++i) diff_order_r2 += std::fabs(r2.data<float>()[i] - r2_first.data<float>()[i]);
    assert(diff_order_r2 < 1e-3f && "Reverse-order independence failed (r2): static codebook bug still present");

    // Centroid-inequality: codebooks trained on different distributions must differ.
    // Compare permutation-invariant means of two CodebookQUANT8 trained on t1 vs t2
    // (t1 ~ [-1,1] mean ~0, t2 ~ [-5,15] mean ~5).
    double cb_mean_diff = 0;
    {
        CodebookQUANT8 cb1, cb2;
        cb1.train(d1, (size_t)N);
        cb2.train(d2, (size_t)N);
        double m1 = 0, m2 = 0;
        for (int c = 0; c < CodebookQUANT8::SIZE; ++c) { m1 += cb1.centroids[c]; m2 += cb2.centroids[c]; }
        m1 /= CodebookQUANT8::SIZE; m2 /= CodebookQUANT8::SIZE;
        cb_mean_diff = std::fabs(m1 - m2);
        assert(cb_mean_diff > 0.5 && "Centroid inequality failed: distinct distributions must yield distinct codebooks");
    }

    // Also test Q4
    STEQuantizer q4(Format::Q4);
    Tensor rq4_1 = q4.forward(t1);
    Tensor rq4_2 = q4.forward(t2);
    float d4_1 = 0, d4_2 = 0;
    for (int i = 0; i < N; ++i) { d4_1 += std::fabs(rq4_1.data<float>()[i]-d1[i]); d4_2 += std::fabs(rq4_2.data<float>()[i]-d2[i]); }
    assert(d4_1 > 1e-6f && d4_2 > 1e-6f);

    // Extended formats Q3/Q6/Q12 must not be passthrough (D3 Sec.5/Sec.7).
    // STEQuantizer::forward implements Q1/Q4/Q8 natively (Q2 via forward_mixed);
    // Q3/Q6/Q12 are exercised via their direct codebook roundtrip so the
    // non-passthrough invariant holds independent of forward() coverage.
    float d3 = 0, d6 = 0, d12 = 0;
    {
        CodebookQ3 cb3; cb3.train(d1, (size_t)N);
        for (int i = 0; i < N; ++i) d3 += std::fabs(cb3.dequantize(cb3.quantize(d1[i])) - d1[i]);
        assert(d3 > 1e-6f && "Q3 should not be passthrough");
        CodebookQ6 cb6; cb6.train(d1, (size_t)N);
        for (int i = 0; i < N; ++i) d6 += std::fabs(cb6.dequantize(cb6.quantize(d1[i])) - d1[i]);
        assert(d6 > 1e-6f && "Q6 should not be passthrough");
        CodebookQ12 cb12; cb12.train(d1, (size_t)N);
        for (int i = 0; i < N; ++i) d12 += std::fabs(cb12.dequantize(cb12.quantize(d1[i])) - d1[i]);
        assert(d12 > 1e-6f && "Q12 should not be passthrough");
    }

    // Q1-vs-Q2 difference (D3 Sec.5/Sec.7): 1-bit sign vs 2-bit ternary must differ.
    float dq1q2 = 0;
    {
        STEQuantizer q1(Format::Q1);
        Tensor rq1 = q1.forward(t1);
        std::vector<Format> one_q2 = {Format::Q2};
        Tensor rq2 = q1.forward_mixed(t1, one_q2, N);
        for (int i = 0; i < N; ++i) dq1q2 += std::fabs(rq1.data<float>()[i] - rq2.data<float>()[i]);
        assert(dq1q2 > 1e-3f && "Q1 vs Q2 outputs must differ (distinct levels)");
    }

    std::cout << "test_ste_codebook: PASS (diff1=" << diff1 << " diff2=" << diff2 << " order_diff=" << diff_order << " order_r2_diff=" << diff_order_r2 << " cb_mean_diff=" << cb_mean_diff << " q6diff=" << d6 << " q3diff=" << d3 << " q12diff=" << d12 << " q1q2diff=" << dq1q2 << ")\n";
    return 0;
}
