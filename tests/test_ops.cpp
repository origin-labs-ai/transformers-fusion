#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
#include "quant/math.h"
#include "quant/simd_math.h"
#include "quant/tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

void test_rmsnorm() {
    quant::Tensor a({4}, quant::DType::F32);
    quant::Tensor w({4}, quant::DType::F32);
    quant::Tensor c({4}, quant::DType::F32);
    float* ad = a.data<float>();
    float* wd = w.data<float>();
    ad[0] = 1.0f; ad[1] = 2.0f; ad[2] = 3.0f; ad[3] = 4.0f;
    wd[0] = 1.0f; wd[1] = 1.0f; wd[2] = 1.0f; wd[3] = 1.0f;
    
    // (1^2 + 2^2 + 3^2 + 4^2)/4 = (1 + 4 + 9 + 16)/4 = 30/4 = 7.5
    // rms = sqrt(7.5) ~= 2.73861278
    // norm = [1, 2, 3, 4] / 2.73861278
    
    quant::math::rms_norm(a, w, 1e-5f, c);
    
    const float* cd = c.data<float>();
    float expected_rms = std::sqrt(7.5f + 1e-5f);
    for (int i = 0; i < 4; ++i) {
        assert(std::abs(cd[i] - ad[i] / expected_rms) < 1e-4f);
    }
}

void test_softmax() {
    quant::Tensor a({2, 2}, quant::DType::F32);
    quant::Tensor c({2, 2}, quant::DType::F32);
    float* ad = a.data<float>();
    ad[0] = 1.0f; ad[1] = 2.0f;
    ad[2] = -1.0f; ad[3] = 0.0f;
    
    quant::math::softmax(a, c);
    
    const float* cd = c.data<float>();
    float sum1 = std::exp(1.0f) + std::exp(2.0f);
    assert(std::abs(cd[0] - std::exp(1.0f)/sum1) < 1e-4f);
    assert(std::abs(cd[1] - std::exp(2.0f)/sum1) < 1e-4f);
    
    float sum2 = std::exp(-1.0f) + std::exp(0.0f);
    assert(std::abs(cd[2] - std::exp(-1.0f)/sum2) < 1e-4f);
    assert(std::abs(cd[3] - std::exp(0.0f)/sum2) < 1e-4f);
}

void test_activations() {
    quant::Tensor a({2}, quant::DType::F32);
    quant::Tensor c({2}, quant::DType::F32);
    float* ad = a.data<float>();
    ad[0] = 0.0f; ad[1] = 1.0f;
    
    // SiLU: x * sigmoid(x)
    quant::math::silu(a, c);
    const float* cd = c.data<float>();
    assert(std::abs(cd[0] - 0.0f) < 1e-4f);
    float sig1 = 1.0f / (1.0f + std::exp(-1.0f));
    assert(std::abs(cd[1] - 1.0f * sig1) < 1e-4f);
    
    // GELU: x * 0.5 * (1 + erf(x/sqrt(2)))
    quant::math::gelu(a, c);
    assert(std::abs(cd[0] - 0.0f) < 1e-4f);
    float gelu1 = 1.0f * 0.5f * (1.0f + std::erf(1.0f / std::sqrt(2.0f)));
    assert(std::abs(cd[1] - gelu1) < 1e-4f);
}

void test_rope() {
    quant::Tensor q({1, 1, 2}, quant::DType::F32);
    quant::Tensor pos({1}, quant::DType::I32);
    float* qd = q.data<float>();
    int32_t* pd = pos.data<int32_t>();
    qd[0] = 1.0f; qd[1] = 2.0f;
    pd[0] = 1;
    
    // pos = 1, theta = 10000.0
    // d = 2
    // freq = 1.0 / (10000^(0/2)) = 1.0
    // cos = cos(1.0), sin = sin(1.0)
    // q0 = q0*cos - q1*sin
    // q1 = q1*cos + q0*sin
    
    std::vector<float> cos_cache(2), sin_cache(2);
    quant::simd::rope_precompute_freqs(cos_cache.data(), sin_cache.data(), 2, 2, 10000.0f);
    quant::simd::rope(qd, nullptr, cos_cache.data(), sin_cache.data(), 2, 1, 1);
    
    float cos_val = std::cos(1.0f);
    float sin_val = std::sin(1.0f);
    
    float expected_q0 = 1.0f * cos_val - 2.0f * sin_val;
    float expected_q1 = 2.0f * cos_val + 1.0f * sin_val;
    
    assert(std::abs(qd[0] - expected_q0) < 1e-4f);
    assert(std::abs(qd[1] - expected_q1) < 1e-4f);
}

int main() {
    std::cout << "[Test] Running Ops test..." << std::endl;
    quant::Tensor a({2, 2}, quant::DType::F32);
    quant::Tensor b({2, 2}, quant::DType::F32);
    a.fill(2.0f);
    b.fill(3.0f);

    quant::Tensor c({2, 2}, quant::DType::F32);
    quant::math::add(a, b, c);

    const float* cd = c.data<float>();
    for (int i = 0; i < 4; ++i) {
        assert(std::abs(cd[i] - 5.0f) < 1e-5f);
    }
    
    quant::math::mul(a, b, c);
    for (int i = 0; i < 4; ++i) {
        assert(std::abs(cd[i] - 6.0f) < 1e-5f);
    }
    
    quant::Tensor c_matmul({2, 2}, quant::DType::F32);
    quant::math::gemm(1.0f, a, b, 0.0f, c_matmul);
    const float* cmd = c_matmul.data<float>();
    for (int i = 0; i < 4; ++i) {
        assert(std::abs(cmd[i] - 12.0f) < 1e-5f);
    }
    
    test_rmsnorm();
    test_softmax();
    test_activations();
    test_rope();
    
    std::cout << "Ops test passed!" << std::endl;
    return 0;
}
