// test_autograd.cpp — grad-chain: matmul/add/mul backward correctness vs numeric grad
#include "quant/autograd.h"
#include "quant/tensor.h"
#include "quant/test.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

using namespace quant;

namespace {

constexpr float kEps = 1e-3f;
constexpr float kTol = 1e-2f;

float det_val(int64_t i, float scale, float off) {
    return off + scale * (float)((i * 37 % 11) - 5) / 5.0f;
}

void fill_det(Tensor& t, float scale = 0.5f, float off = 0.0f) {
    float* d = t.data<float>();
    for (int64_t i = 0; i < t.numel(); i++) d[i] = det_val(i, scale, off);
}

float rel_err(float numeric, float analytic) {
    float denom = std::max(1.0f, std::fabs(numeric));
    return std::fabs(numeric - analytic) / denom;
}

float sum_loss(const Tensor& t) {
    const float* d = t.data<float>();
    double s = 0.0;
    for (int64_t i = 0; i < t.numel(); i++) s += (double)d[i];
    return (float)s;
}

bool all_finite(const Tensor& t) {
    const float* d = t.data<float>();
    for (int64_t i = 0; i < t.numel(); i++)
        if (!std::isfinite(d[i])) return false;
    return true;
}

// Central-difference check of analytic grads (backward seeds ones, i.e.
// d sum(out)/dp) against numeric grads of loss_fn (autograd disabled).
float check_vs_numeric(const std::vector<Tensor*>& params,
                       const std::function<float()>& loss_fn,
                       const char* what) {
    float max_err = 0.0f;
    for (Tensor* p : params) {
        float* d = p->data<float>();
        const float* g = p->grad().data<float>();
        for (int64_t i = 0; i < p->numel(); i++) {
            float orig = d[i];
            d[i] = orig + kEps;
            float lp = loss_fn();
            d[i] = orig - kEps;
            float lm = loss_fn();
            d[i] = orig;
            float numeric = (lp - lm) / (2.0f * kEps);
            float e = rel_err(numeric, g[i]);
            if (e > max_err) max_err = e;
        }
    }
    printf("  max rel err [%s]: %.6f\n", what, max_err);
    return max_err;
}

} // namespace

static void test_matmul_backward() {
    TEST_SUITE("autograd matmul");
    auto& engine = AutogradEngine::instance();
    const int64_t M = 2, K = 3, N = 2;

    Tensor a(Shape{M, K}, DType::F32);
    Tensor b(Shape{N, K}, DType::F32);
    fill_det(a, 0.5f, 0.0f);
    fill_det(b, 0.5f, 0.1f);
    a.requires_grad(true);
    b.requires_grad(true);

    engine.reset();
    engine.register_parameter(&a);
    engine.register_parameter(&b);
    AutogradEngine::set_enabled(true);
    Tensor out = AutogradEngine::matmul_op(a, b, M, N, K);
    float fwd = sum_loss(out);
    engine.backward(out);
    engine.clear();
    AutogradEngine::set_enabled(false);

    TEST_CHECK(std::isfinite(fwd), "matmul forward finite");
    TEST_CHECK(a.has_grad(), "matmul: a has grad");
    TEST_CHECK(b.has_grad(), "matmul: b has grad");
    TEST_CHECK(all_finite(a.grad()), "matmul: grad(a) finite");
    TEST_CHECK(all_finite(b.grad()), "matmul: grad(b) finite");

    auto loss_fn = [&]() -> float {
        Tensor o = AutogradEngine::matmul_op(a, b, M, N, K);
        return sum_loss(o);
    };
    float max_err = check_vs_numeric({&a, &b}, loss_fn, "matmul");
    TEST_CHECK(max_err < kTol, "matmul backward matches numeric grad");
    engine.reset();
}

static void test_add_backward() {
    TEST_SUITE("autograd add");
    auto& engine = AutogradEngine::instance();

    Tensor a(Shape{2, 3}, DType::F32);
    Tensor b(Shape{2, 3}, DType::F32);
    fill_det(a, 0.5f, 0.0f);
    fill_det(b, 0.5f, 0.2f);
    a.requires_grad(true);
    b.requires_grad(true);

    engine.reset();
    engine.register_parameter(&a);
    engine.register_parameter(&b);
    AutogradEngine::set_enabled(true);
    Tensor out = AutogradEngine::add_op(a, b);
    float fwd = sum_loss(out);
    engine.backward(out);
    engine.clear();
    AutogradEngine::set_enabled(false);

    TEST_CHECK(std::isfinite(fwd), "add forward finite");
    TEST_CHECK(a.has_grad(), "add: a has grad");
    TEST_CHECK(b.has_grad(), "add: b has grad");
    // d sum(a+b)/da = ones
    bool ones = true;
    for (int64_t i = 0; i < a.numel(); i++)
        if (std::fabs(a.grad().data<float>()[i] - 1.0f) > 1e-5f) ones = false;
    TEST_CHECK(ones, "add: grad(a) is ones");
    TEST_CHECK(all_finite(b.grad()), "add: grad(b) finite");

    auto loss_fn = [&]() -> float {
        Tensor o = AutogradEngine::add_op(a, b);
        return sum_loss(o);
    };
    float max_err = check_vs_numeric({&a, &b}, loss_fn, "add");
    TEST_CHECK(max_err < kTol, "add backward matches numeric grad");
    engine.reset();
}

static void test_mul_backward() {
    TEST_SUITE("autograd mul");
    auto& engine = AutogradEngine::instance();

    Tensor a(Shape{2, 3}, DType::F32);
    Tensor b(Shape{2, 3}, DType::F32);
    fill_det(a, 0.5f, 0.1f);
    fill_det(b, 0.5f, -0.2f);
    a.requires_grad(true);
    b.requires_grad(true);

    engine.reset();
    engine.register_parameter(&a);
    engine.register_parameter(&b);
    AutogradEngine::set_enabled(true);
    Tensor out = AutogradEngine::mul_op(a, b);
    float fwd = sum_loss(out);
    engine.backward(out);
    engine.clear();
    AutogradEngine::set_enabled(false);

    TEST_CHECK(std::isfinite(fwd), "mul forward finite");
    TEST_CHECK(a.has_grad(), "mul: a has grad");
    TEST_CHECK(b.has_grad(), "mul: b has grad");
    // d sum(a*b)/da = b
    bool matches = true;
    for (int64_t i = 0; i < a.numel(); i++)
        if (std::fabs(a.grad().data<float>()[i] - b.data<float>()[i]) > 1e-5f) matches = false;
    TEST_CHECK(matches, "mul: grad(a) equals b");
    TEST_CHECK(all_finite(b.grad()), "mul: grad(b) finite");

    auto loss_fn = [&]() -> float {
        Tensor o = AutogradEngine::mul_op(a, b);
        return sum_loss(o);
    };
    float max_err = check_vs_numeric({&a, &b}, loss_fn, "mul");
    TEST_CHECK(max_err < kTol, "mul backward matches numeric grad");
    engine.reset();
}

static void test_grad_chain() {
    TEST_SUITE("autograd grad-chain");
    auto& engine = AutogradEngine::instance();
    const int64_t M = 2, K = 3, N = 2;

    Tensor a(Shape{M, K}, DType::F32);
    Tensor w(Shape{N, K}, DType::F32);
    Tensor bias(Shape{M, N}, DType::F32);
    Tensor scale(Shape{M, N}, DType::F32);
    fill_det(a, 0.5f, 0.0f);
    fill_det(w, 0.5f, 0.1f);
    fill_det(bias, 0.5f, -0.1f);
    fill_det(scale, 0.5f, 0.3f);
    a.requires_grad(true);
    w.requires_grad(true);
    bias.requires_grad(true);
    scale.requires_grad(true);

    engine.reset();
    engine.register_parameter(&a);
    engine.register_parameter(&w);
    engine.register_parameter(&bias);
    engine.register_parameter(&scale);
    AutogradEngine::set_enabled(true);
    Tensor c = AutogradEngine::matmul_op(a, w, M, N, K);
    Tensor d = AutogradEngine::add_op(c, bias);
    Tensor e = AutogradEngine::mul_op(d, scale);
    float fwd = sum_loss(e);
    engine.backward(e);
    engine.clear();
    AutogradEngine::set_enabled(false);

    TEST_CHECK(std::isfinite(fwd), "chain forward finite");
    TEST_CHECK(a.has_grad(), "chain: a has grad");
    TEST_CHECK(w.has_grad(), "chain: w has grad");
    TEST_CHECK(bias.has_grad(), "chain: bias has grad");
    TEST_CHECK(scale.has_grad(), "chain: scale has grad");
    TEST_CHECK(all_finite(a.grad()), "chain: grad(a) finite");
    TEST_CHECK(all_finite(w.grad()), "chain: grad(w) finite");
    TEST_CHECK(all_finite(bias.grad()), "chain: grad(bias) finite");
    TEST_CHECK(all_finite(scale.grad()), "chain: grad(scale) finite");

    auto loss_fn = [&]() -> float {
        Tensor cc = AutogradEngine::matmul_op(a, w, M, N, K);
        Tensor dd = AutogradEngine::add_op(cc, bias);
        Tensor ee = AutogradEngine::mul_op(dd, scale);
        return sum_loss(ee);
    };
    float max_err = check_vs_numeric({&a, &w, &bias, &scale}, loss_fn, "matmul+add+mul chain");
    TEST_CHECK(max_err < kTol, "chain backward matches numeric grad");
    engine.reset();
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("Transcender — Autograd Test Suite\n");
    printf("==========================================\n");

    test_matmul_backward();
    test_add_backward();
    test_mul_backward();
    test_grad_chain();

    printf("\n==========================================\n");
    return TEST_REPORT() > 0 ? 1 : 0;
}
