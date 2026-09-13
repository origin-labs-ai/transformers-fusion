#include "quant/quant_engines.h"
#include "quant/tensor.h"
#include "quant/types.h"
#include "quant/kernel.h"
#include "quant/codebook.h"
#include "quant/math.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <cstdint>
#include "quant/test.h"

using namespace quant;
using namespace quant::engines;

static bool approx_equal(float a, float b, float eps = 1e-3f) {
    return std::abs(a - b) < eps;
}

static bool all_close(const float* a, const float* b, int64_t n, float eps = 1e-3f) {
    for (int64_t i = 0; i < n; i++)
        if (!approx_equal(a[i], b[i], eps)) return false;
    return true;
}

// QUANT8 quantize/dequantize roundtrip
static void test_quant8_roundtrip() {
    TEST_SUITE("QUANT8 quantize/dequantize");
    QUANT8Engine quant8;
    std::vector<float> data(256);
    for (int i = 0; i < 256; i++)
        data[i] = (float)(i - 128) / 8.0f;
    quant8.train_codebook(data.data(), 256);

    float val = 5.0f;
    uint8_t idx = quant8.quantize(val);
    float deq = quant8.dequantize(idx);
    TEST_CHECK_CLOSE(val, deq, 1.0f, "QUANT8 scalar roundtrip");

    // Tensor roundtrip
    int64_t N = 100;
    Tensor orig({N});
    for (int64_t i = 0; i < N; i++)
        orig.data<float>()[i] = std::sin((float)i * 0.1f) * 8.0f;
    Tensor q = quant8.quantize_tensor(orig.data<float>(), N);
    Tensor dq = quant8.dequant_tensor(q.data<uint8_t>(), N);
    TEST_CHECK(all_close(orig.data<float>(), dq.data<float>(), N, 4.0f), "QUANT8 tensor roundtrip");

    // Per-channel
    Tensor orig2d({4, 32});
    for (int64_t i = 0; i < 4 * 32; i++)
        orig2d.data<float>()[i] = (float)(i % 20);
    Tensor qc, scales;
    quant8.quantize_per_channel(orig2d, 0, qc, scales);
    Tensor dqc({4, 32});
    quant8.dequantize_per_channel(qc, scales, 0, dqc);
    TEST_CHECK(all_close(orig2d.data<float>(), dqc.data<float>(), 4 * 32, 5.0f), "QUANT8 per-channel roundtrip");

    // quant_error
    float err = quant8.quant_error(orig, dq);
    TEST_CHECK(std::isfinite(err), "QUANT8 quant_error finite");
    TEST_CHECK(err >= 0, "QUANT8 quant_error non-negative");

    // quant_snr
    float snr = quant8.quant_snr(orig, dq);
    TEST_CHECK(std::isfinite(snr), "QUANT8 quant_snr finite");
}

// QUANT4 quantize/dequantize roundtrip
static void test_quant4_roundtrip() {
    TEST_SUITE("QUANT4 quantize/dequantize");
    QUANT4Engine quant4;
    std::vector<float> train_data(64);
    for (int i = 0; i < 64; i++)
        train_data[i] = (float)(i - 32) / 2.0f;
    quant4.train_codebook(train_data.data(), 64);

    float val = 5.0f;
    uint8_t idx = quant4.quantize(val);
    float deq = quant4.dequantize(idx);
    TEST_CHECK_CLOSE(val, deq, 2.0f, "QUANT4 scalar roundtrip");

    int64_t N = 64;
    Tensor orig({N});
    for (int64_t i = 0; i < N; i++)
        orig.data<float>()[i] = (float)(i - 32) / 4.0f;
    Tensor q = quant4.quantize_tensor(orig.data<float>(), N);
    Tensor dq = quant4.dequant_tensor(q.data<uint8_t>(), N);
    TEST_CHECK(all_close(orig.data<float>(), dq.data<float>(), N, 5.0f), "QUANT4 tensor roundtrip");

    // Per-channel
    Tensor orig2d({4, 16});
    for (int64_t i = 0; i < 4 * 16; i++)
        orig2d.data<float>()[i] = (float)(i % 10);
    Tensor qc, scales;
    quant4.quantize_per_channel(orig2d, 0, qc, scales);
    Tensor dqc({4, 16});
    quant4.dequantize_per_channel(qc, scales, 0, dqc);
    TEST_CHECK(all_close(orig2d.data<float>(), dqc.data<float>(), 4 * 16, 5.0f), "QUANT4 per-channel roundtrip");

    float err = quant4.quant_error(orig, dq);
    TEST_CHECK(std::isfinite(err), "QUANT4 quant_error finite");
}

// FP8 E4M3 roundtrip
static void test_fp8_e4m3() {
    TEST_SUITE("FP8 E4M3");
    float vals[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, 4.0f, 8.0f, -8.0f, 3.14f, -3.14f, 0.125f};
    for (int i = 0; i < 12; i++) {
        uint8_t q = fp8_e4m3_quantize(vals[i]);
        float dq = fp8_e4m3_dequantize(q);
        TEST_CHECK_CLOSE(vals[i], dq, 0.5f, "E4M3 scalar roundtrip");
    }

    int64_t N = 50;
    Tensor orig({N});
    for (int64_t i = 0; i < N; i++)
        orig.data<float>()[i] = std::sin((float)i) * 4.0f;
    Tensor q = fp8_e4m3_quantize_tensor(orig.data<float>(), N);
    Tensor dq = fp8_e4m3_dequant_tensor(q.data<uint8_t>(), N);
    TEST_CHECK(all_close(orig.data<float>(), dq.data<float>(), N, 1.0f), "E4M3 tensor roundtrip");

    // Per-channel
    Tensor orig2d({4, 16});
    for (int64_t i = 0; i < 4 * 16; i++)
        orig2d.data<float>()[i] = (float)(i % 8) - 4.0f;
    Tensor qc, sc;
    fp8_e4m3_quantize_per_channel(orig2d, 0, qc, sc);
    Tensor dqc({4, 16});
    fp8_e4m3_dequantize_per_channel(qc, sc, 0, dqc);
    TEST_CHECK(all_close(orig2d.data<float>(), dqc.data<float>(), 4 * 16, 1.0f), "E4M3 per-channel roundtrip");

    float err = fp8_e4m3_quant_error(orig, dq);
    TEST_CHECK(std::isfinite(err), "E4M3 quant_error finite");

    float snr = fp8_e4m3_quant_snr(orig, dq);
    TEST_CHECK(std::isfinite(snr), "E4M3 quant_snr finite");
}

// FP8 E5M2 roundtrip
static void test_fp8_e5m2() {
    TEST_SUITE("FP8 E5M2");
    float vals[] = {0.0f, 1.0f, -1.0f, 0.5f, 2.0f, 4.0f, 8.0f, 16.0f, -16.0f, 32.0f, 64.0f, 128.0f};
    for (int i = 0; i < 12; i++) {
        uint8_t q = fp8_e5m2_quantize(vals[i]);
        float dq = fp8_e5m2_dequantize(q);
        TEST_CHECK_CLOSE(vals[i], dq, 2.0f, "E5M2 scalar roundtrip");
    }

    int64_t N = 50;
    Tensor orig({N});
    for (int64_t i = 0; i < N; i++)
        orig.data<float>()[i] = std::sin((float)i) * 32.0f;
    Tensor q = fp8_e5m2_quantize_tensor(orig.data<float>(), N);
    Tensor dq = fp8_e5m2_dequant_tensor(q.data<uint8_t>(), N);
    TEST_CHECK(all_close(orig.data<float>(), dq.data<float>(), N, 4.0f), "E5M2 tensor roundtrip");

    // Per-channel
    Tensor orig2d({3, 20});
    for (int64_t i = 0; i < 3 * 20; i++)
        orig2d.data<float>()[i] = (float)(i % 10) - 5.0f;
    Tensor qc, sc;
    fp8_e5m2_quantize_per_channel(orig2d, 0, qc, sc);
    Tensor dqc({3, 20});
    fp8_e5m2_dequantize_per_channel(qc, sc, 0, dqc);
    TEST_CHECK(all_close(orig2d.data<float>(), dqc.data<float>(), 3 * 20, 2.0f), "E5M2 per-channel roundtrip");

    float err = fp8_e5m2_quant_error(orig, dq);
    TEST_CHECK(std::isfinite(err), "E5M2 quant_error finite");
}

// QUANT quantize/dequantize
static void test_quant() {
    TEST_SUITE("Quant");
    QuantEngine te(32);
    int64_t N = 64;
    Tensor orig({N});
    // Normalized weights in [-1, 1] — typical for QUANT quantization
    for (int64_t i = 0; i < N; i++)
        orig.data<float>()[i] = (float)(i - 32) / 32.0f;
    Tensor q = te.quantize(orig);
    Tensor dq = te.dequantize(q, Tensor({1}), N);
    TEST_CHECK(all_close(orig.data<float>(), dq.data<float>(), N, 0.7f), "Quant tensor roundtrip");

    // Per-channel — wider range with per-channel scaling
    Tensor orig2d({4, 16});
    for (int64_t i = 0; i < 4 * 16; i++)
        orig2d.data<float>()[i] = (float)(i % 8) - 4.0f;
    Tensor qc, sc;
    te.quantize_per_channel(orig2d, 0, qc, sc);
    Tensor dqc({4, 16});
    te.dequantize_per_channel(qc, sc, 0, dqc);
    TEST_CHECK(all_close(orig2d.data<float>(), dqc.data<float>(), 4 * 16, 3.0f), "Quant per-channel roundtrip");

    float err = te.quant_error(orig, dq);
    TEST_CHECK(std::isfinite(err), "Quant quant_error finite");
}

// QUANT1 quantize/dequantize
static void test_quant1() {
    TEST_SUITE("Quant1");
    Quant1Engine be;
    int64_t N = 64;
    Tensor orig({N});
    // Normalized weights in [-1, 1] — typical for QUANT1 quantization
    for (int64_t i = 0; i < N; i++)
        orig.data<float>()[i] = (float)(i - 32) / 32.0f;
    Tensor q = be.quantize(orig);
    float scale = 1.0f;
    Tensor dq = be.dequantize(q, scale, N);
    TEST_CHECK(all_close(orig.data<float>(), dq.data<float>(), N, 1.1f), "Quant1 tensor roundtrip");

    // Per-channel
    Tensor orig2d({4, 16});
    for (int64_t i = 0; i < 4 * 16; i++)
        orig2d.data<float>()[i] = (float)(i % 8) - 4.0f;
    Tensor qc, sc;
    be.quantize_per_channel(orig2d, 0, qc, sc);
    Tensor dqc({4, 16});
    be.dequantize_per_channel(qc, sc, 0, dqc);
    TEST_CHECK(all_close(orig2d.data<float>(), dqc.data<float>(), 4 * 16, 4.1f), "Quant1 per-channel roundtrip");

    float err = be.quant_error(orig, dq);
    TEST_CHECK(std::isfinite(err), "Quant1 quant_error finite");
}

// quant_gemm fused operation
static void test_quant_gemm() {
    TEST_SUITE("Quant GEMM fused");
    int64_t M = 2, N = 4, K = 8;

    // FP8 E4M3 quant_gemm
    Tensor a({M, K});
    std::vector<uint8_t> b_q(K * N);
    for (int64_t i = 0; i < M * K; i++)
        a.data<float>()[i] = std::sin((float)i) * 2.0f;
    for (int64_t i = 0; i < K * N; i++)
        b_q[i] = fp8_e4m3_quantize(std::cos((float)i) * 2.0f);
    Tensor c_e4m3 = fp8_e4m3_quant_gemm(a, b_q.data(), M, N, K);
    TEST_CHECK(c_e4m3.numel() == M * N, "E4M3 quant_gemm output shape");
    for (int64_t i = 0; i < c_e4m3.numel(); i++)
        TEST_CHECK(std::isfinite(c_e4m3.data<float>()[i]), "E4M3 quant_gemm finite");

    // FP8 E5M2 quant_gemm
    Tensor c_e5m2 = fp8_e5m2_quant_gemm(a, b_q.data(), M, N, K);
    TEST_CHECK(c_e5m2.numel() == M * N, "E5M2 quant_gemm output shape");
    for (int64_t i = 0; i < c_e5m2.numel(); i++)
        TEST_CHECK(std::isfinite(c_e5m2.data<float>()[i]), "E5M2 quant_gemm finite");

    // QUANT8 quant_gemm
    QUANT8Engine quant8;
    std::vector<float> train(256);
    for (int i = 0; i < 256; i++) train[i] = (float)(i - 128) / 16.0f;
    quant8.train_codebook(train.data(), 256);
    std::vector<uint8_t> b_idx(K * N);
    for (int64_t i = 0; i < K * N; i++)
        b_idx[i] = quant8.quantize((float)(i % 10));
    Tensor c_quant8 = quant8.quant_gemm(a, b_idx.data(), M, N, K);
    TEST_CHECK(c_quant8.numel() == M * N, "QUANT8 quant_gemm output shape");
    for (int64_t i = 0; i < c_quant8.numel(); i++)
        TEST_CHECK(std::isfinite(c_quant8.data<float>()[i]), "QUANT8 quant_gemm finite");
}

// quant_error metrics
static void test_quant_error_metrics() {
    TEST_SUITE("Quant Error Metrics");
    int64_t N = 32;
    Tensor orig({N});
    for (int64_t i = 0; i < N; i++)
        orig.data<float>()[i] = (float)(i - 16) / 4.0f;
    Tensor perfect = orig.clone();

    float err = compute_quant_error(orig, perfect);
    TEST_CHECK_CLOSE(err, 0.0f, 1e-6f, "quant_error zero for identical");
    float mse = compute_quant_mse(orig, perfect);
    TEST_CHECK_CLOSE(mse, 0.0f, 1e-6f, "quant_mse zero for identical");

    // Non-zero error
    Tensor bad = orig.clone();
    bad.fill(0.0f);
    float err_bad = compute_quant_error(orig, bad);
    TEST_CHECK(err_bad > 0, "quant_error positive for mismatch");
    float mse_bad = compute_quant_mse(orig, bad);
    TEST_CHECK(mse_bad > 0, "quant_mse positive for mismatch");

    // SNR computation
    float snr = compute_quant_snr(orig, perfect);
    TEST_CHECK(std::isfinite(snr), "quant_snr finite for identical");
}

// Per-block QUANT8 vs uniform 8-bit grid: QUANT codebook beats plain uniform grid
static void test_perblock_quant8_beats_uniform8() {
    TEST_SUITE("Per-block QUANT8 vs uniform 8-bit grid");
    const int64_t N = 100320;
    const int64_t BS = 32;

    std::vector<float> w(N);
    {
        unsigned seed = 42;
        auto rng = [&]() { seed = seed * 1103515245 + 12345; return (float)((seed >> 16) & 0x7FFF) / 32768.0f; };
        for (int64_t i = 0; i < N; i++) {
            float u1 = rng() * 0.999f + 0.001f;
            float u2 = rng();
            float z = std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
            w[i] = z * 0.02f;
        }
    }

    // Uniform 8-bit grid per-block (in-house reference)
    double u8_sq = 0.0;
    for (int64_t b = 0; b < N / BS; b++) {
        float amax = 0.0f;
        for (int64_t i = 0; i < BS; i++)
            amax = (std::max)(amax, std::abs(w[b * BS + i]));
        float scale = amax / 127.0f;
        if (scale < 1e-10f) scale = 1.0f;
        for (int64_t i = 0; i < BS; i++) {
            float q = std::round(w[b * BS + i] / scale);
            q = (std::max)(-128.0f, (std::min)(127.0f, q));
            float dq = q * scale;
            double d = (double)w[b * BS + i] - (double)dq;
            u8_sq += d * d;
        }
    }
    double u8_mse = u8_sq / (double)N;

    // QUANT8 per-block k-means
    QUANT8Engine quant8;
    quant8.train_codebook_per_block(w.data(), N, BS, 30);

    std::vector<uint8_t> indices(N);
    std::vector<float> scales(N / BS + 1);
    quant8.quantize_per_block(w.data(), N, BS, indices.data(), scales.data());

    std::vector<float> dq(N);
    quant8.dequantize_per_block(indices.data(), scales.data(), N, BS, dq.data());

    double quant8_sq = 0.0;
    for (int64_t i = 0; i < N; i++) {
        double d = (double)w[i] - (double)dq[i];
        quant8_sq += d * d;
    }
    double quant8_mse = quant8_sq / (double)N;

    printf("  u8_0  (uniform per-block) MSE: %.6e\n", u8_mse);
    printf("  QUANT8  (k-means per-block) MSE: %.6e\n", quant8_mse);
    if (quant8_mse < u8_mse) {
        printf("  >>> QUANT8 BEATS u8_0 by %.2fx <<<\n", u8_mse / quant8_mse);
    } else {
        printf("  u8_0 wins by %.2fx\n", quant8_mse / u8_mse);
    }
    TEST_CHECK(quant8_mse < u8_mse, "QUANT8 per-block beats u8_0 uniform");
}

// Per-block QUANT4 vs uniform 4-bit grid: QUANT codebook beats plain uniform grid
static void test_perblock_quant4_beats_uniform4() {
    TEST_SUITE("Per-block QUANT4 vs uniform 4-bit grid");
    const int64_t N = 100320;
    const int64_t BS = 32;

    std::vector<float> w(N);
    {
        unsigned seed = 42;
        auto rng = [&]() { seed = seed * 1103515245 + 12345; return (float)((seed >> 16) & 0x7FFF) / 32768.0f; };
        for (int64_t i = 0; i < N; i++) {
            float u1 = rng() * 0.999f + 0.001f;
            float u2 = rng();
            float z = std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
            w[i] = z * 0.02f;
        }
    }

    // Uniform 4-bit grid per-block (in-house reference)
    double u4_sq = 0.0;
    for (int64_t b = 0; b < N / BS; b++) {
        float bmin = w[b * BS], bmax = w[b * BS];
        for (int64_t i = 1; i < BS; i++) {
            bmin = (std::min)(bmin, w[b * BS + i]);
            bmax = (std::max)(bmax, w[b * BS + i]);
        }
        float range = bmax - bmin;
        float scale = range / 15.0f;
        if (scale < 1e-10f) scale = 1.0f;
        for (int64_t i = 0; i < BS; i++) {
            float q = std::round((w[b * BS + i] - bmin) / scale);
            q = (std::max)(0.0f, (std::min)(15.0f, q));
            float dq = q * scale + bmin;
            double d = (double)w[b * BS + i] - (double)dq;
            u4_sq += d * d;
        }
    }
    double u4_mse = u4_sq / (double)N;

    // QUANT4 per-block k-means
    QUANT4Engine quant4;
    quant4.train_codebook_per_block(w.data(), N, BS, 30);

    std::vector<uint8_t> indices(N);
    std::vector<float> scales((N / BS + 1) * 2);
    quant4.quantize_per_block(w.data(), N, BS, indices.data(), scales.data());

    std::vector<float> dq(N);
    quant4.dequantize_per_block(indices.data(), scales.data(), N, BS, dq.data());

    double quant4_sq = 0.0;
    for (int64_t i = 0; i < N; i++) {
        double d = (double)w[i] - (double)dq[i];
        quant4_sq += d * d;
    }
    double quant4_mse = quant4_sq / (double)N;

    printf("  u4_0  (uniform per-block) MSE: %.6e\n", u4_mse);
    printf("  QUANT4  (k-means per-block) MSE: %.6e\n", quant4_mse);
    if (quant4_mse < u4_mse) {
        printf("  >>> QUANT4 BEATS u4_0 by %.2fx <<<\n", u4_mse / quant4_mse);
    } else {
        printf("  u4_0 wins by %.2fx\n", quant4_mse / u4_mse);
    }
    TEST_CHECK(quant4_mse < u4_mse, "QUANT4 per-block beats u4_0 uniform");
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Transcender — Quantization Engine Test Suite\n");
    printf("============================================\n");

    test_quant8_roundtrip();
    test_quant4_roundtrip();
    test_fp8_e4m3();
    test_fp8_e5m2();
    test_quant();
    test_quant1();
    test_quant_gemm();
    test_quant_error_metrics();
    test_perblock_quant8_beats_uniform8();
    test_perblock_quant4_beats_uniform4();

    printf("\n============================================\n");

    return TEST_REPORT() > 0 ? 1 : 0;
}
