// test_simd_math.cpp — simd (dispatched) vs scalar parity
#include "quant/simd_math.h"
#include "quant/test.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace quant;

namespace {

float det_val(int64_t i) {
    return (float)(((i * 37) % 13) - 6) / 6.0f;  // [-1, 1]
}

void fill_det(std::vector<float>& v, float scale = 1.0f) {
    for (size_t i = 0; i < v.size(); i++) v[i] = det_val((int64_t)i) * scale;
}

float max_abs_diff(const float* a, const float* b, int64_t n) {
    float m = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

bool all_finite(const float* v, int64_t n) {
    for (int64_t i = 0; i < n; i++)
        if (!std::isfinite(v[i])) return false;
    return true;
}

} // namespace

static void test_rms_norm_parity() {
    TEST_SUITE("simd rms_norm");
    const int64_t n = 30;  // non-multiple of 8 exercises scalar tails
    std::vector<float> x(n), gamma(n), y_simd(n), y_ref(n);
    fill_det(x);
    for (int64_t i = 0; i < n; i++) gamma[i] = 0.5f + 0.01f * (float)i;
    simd::rms_norm_scalar(x.data(), gamma.data(), y_ref.data(), n, 1e-5f);
    simd::rms_norm(x.data(), gamma.data(), y_simd.data(), n, 1e-5f);
    TEST_CHECK(all_finite(y_simd.data(), n), "rms_norm dispatched output finite");
    float d = max_abs_diff(y_ref.data(), y_simd.data(), n);
    printf("  rms_norm max abs diff: %.3e\n", d);
    TEST_CHECK(d < 1e-5f, "rms_norm simd matches scalar");
}

static void test_swiglu_parity() {
    TEST_SUITE("simd swiglu");
    const int64_t n = 30;
    std::vector<float> gate(n), up(n), y_simd(n), y_ref(n);
    fill_det(gate, 1.0f);
    fill_det(up, 0.5f);
    simd::swiglu_scalar(gate.data(), up.data(), y_ref.data(), n);
    simd::swiglu(gate.data(), up.data(), y_simd.data(), n);
    TEST_CHECK(all_finite(y_simd.data(), n), "swiglu dispatched output finite");
    float d = max_abs_diff(y_ref.data(), y_simd.data(), n);
    printf("  swiglu max abs diff: %.3e\n", d);
    TEST_CHECK(d < 1e-5f, "swiglu simd matches scalar");
}

static void test_geglu_parity() {
    TEST_SUITE("simd geglu");
    const int64_t n = 30;
    std::vector<float> gate(n), up(n), y_simd(n), y_ref(n);
    fill_det(gate, 1.0f);
    fill_det(up, 0.5f);
    simd::geglu_scalar(gate.data(), up.data(), y_ref.data(), n);
    simd::geglu(gate.data(), up.data(), y_simd.data(), n);
    TEST_CHECK(all_finite(y_simd.data(), n), "geglu dispatched output finite");
    float d = max_abs_diff(y_ref.data(), y_simd.data(), n);
    printf("  geglu max abs diff: %.3e\n", d);
    TEST_CHECK(d < 1e-5f, "geglu simd matches scalar");
}

static void test_rope_parity() {
    TEST_SUITE("simd rope");
    const int64_t hd = 16, seq = 3, max_seq = 8;  // half-dim 8: exact SIMD width
    const int64_t half = hd / 2;
    std::vector<float> cos_c(max_seq * half), sin_c(max_seq * half);
    simd::rope_precompute_freqs(cos_c.data(), sin_c.data(), hd, max_seq, 10000.0f);
    std::vector<float> q0(seq * hd), k0(seq * hd);
    fill_det(q0, 0.5f);
    fill_det(k0, 0.5f);

    std::vector<float> qs = q0, ks = k0, qv = q0, kv = k0;
    simd::rope_scalar(qs.data(), ks.data(), cos_c.data(), sin_c.data(), hd, seq);
    simd::rope(qv.data(), kv.data(), cos_c.data(), sin_c.data(), hd, seq);
    TEST_CHECK(all_finite(qv.data(), seq * hd), "rope dispatched q finite");
    TEST_CHECK(all_finite(kv.data(), seq * hd), "rope dispatched k finite");
    float dq = max_abs_diff(qs.data(), qv.data(), seq * hd);
    float dk = max_abs_diff(ks.data(), kv.data(), seq * hd);
    printf("  rope max abs diff: q=%.3e k=%.3e\n", dq, dk);
    TEST_CHECK(dq < 1e-5f, "rope simd q matches scalar");
    TEST_CHECK(dk < 1e-5f, "rope simd k matches scalar");

    // null-k variant (q-only rotation)
    std::vector<float> qns = q0, qnv = q0;
    simd::rope_scalar(qns.data(), nullptr, cos_c.data(), sin_c.data(), hd, seq);
    simd::rope(qnv.data(), nullptr, cos_c.data(), sin_c.data(), hd, seq);
    float dn = max_abs_diff(qns.data(), qnv.data(), seq * hd);
    printf("  rope null-k max abs diff: %.3e\n", dn);
    TEST_CHECK(dn < 1e-5f, "rope null-k simd matches scalar");
}

static void test_softmax_parity() {
    TEST_SUITE("simd softmax");
    const int64_t rows = 2, cols = 10;  // non-multiple of 8 exercises tails
    std::vector<float> x(rows * cols), y_simd(rows * cols), y_ref(rows * cols);
    fill_det(x, 2.0f);
    simd::softmax_scalar(x.data(), y_ref.data(), rows, cols);
    simd::softmax(x.data(), y_simd.data(), rows, cols);
    TEST_CHECK(all_finite(y_simd.data(), rows * cols), "softmax dispatched output finite");
    float d = max_abs_diff(y_ref.data(), y_simd.data(), rows * cols);
    printf("  softmax max abs diff: %.3e\n", d);
    TEST_CHECK(d < 1e-5f, "softmax simd matches scalar");
    for (int64_t r = 0; r < rows; r++) {
        double s = 0.0;
        for (int64_t c = 0; c < cols; c++) s += y_simd[r * cols + c];
        TEST_CHECK(std::fabs(s - 1.0) < 1e-5, "softmax dispatched row sums to 1");
    }
}

static void test_gemv_parity() {
    TEST_SUITE("simd gemv");
    const int64_t M = 7, K = 10;
    std::vector<float> A(M * K), x(K), y0(M);
    fill_det(A, 0.5f);
    fill_det(x, 1.0f);
    fill_det(y0, 0.25f);
    std::vector<float> ys = y0, yv = y0;
    simd::gemv_scalar(A.data(), x.data(), ys.data(), M, K, 0.5f, 0.5f);
    simd::gemv(A.data(), x.data(), yv.data(), M, K, 0.5f, 0.5f);
    TEST_CHECK(all_finite(yv.data(), M), "gemv dispatched output finite");
    float d = max_abs_diff(ys.data(), yv.data(), M);
    printf("  gemv max abs diff: %.3e\n", d);
    TEST_CHECK(d < 1e-4f, "gemv simd matches scalar");
}

static void test_tiled_gemm_parity() {
    TEST_SUITE("simd tiled_gemm");
    const int64_t M = 7, N = 9, K = 10;  // off-tile sizes exercise edge tiles
    std::vector<float> A(M * K), B(K * N);
    std::vector<float> Cs(M * N, 0.0f), Cv(M * N, 0.0f);
    fill_det(A, 0.5f);
    fill_det(B, 0.5f);
    simd::tiled_gemm_scalar(A.data(), B.data(), Cs.data(), M, N, K, 1.0f, 0.0f);
    simd::tiled_gemm(A.data(), B.data(), Cv.data(), M, N, K, 1.0f, 0.0f);
    TEST_CHECK(all_finite(Cv.data(), M * N), "tiled_gemm dispatched output finite");
    float d = max_abs_diff(Cs.data(), Cv.data(), M * N);
    printf("  tiled_gemm max abs diff: %.3e\n", d);
    TEST_CHECK(d < 1e-4f, "tiled_gemm simd matches scalar");
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("Transcender — SIMD Math Test Suite\n");
    printf("==========================================\n");

    test_rms_norm_parity();
    test_swiglu_parity();
    test_geglu_parity();
    test_rope_parity();
    test_softmax_parity();
    test_gemv_parity();
    test_tiled_gemm_parity();

    printf("\n==========================================\n");
    return TEST_REPORT() > 0 ? 1 : 0;
}
