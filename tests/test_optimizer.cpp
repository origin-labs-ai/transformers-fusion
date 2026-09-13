// test_optimizer.cpp — one converging run per major optimizer on a quadratic bowl
#include "quant/optimizer.h"
#include "quant/tensor.h"
#include "quant/test.h"

#include <cmath>
#include <cstdio>

using namespace quant;

namespace {

constexpr float kTarget = 1.0f;
constexpr int kSteps = 30;

float quad_loss(const Tensor& w) {
    const float* d = w.data<float>();
    double s = 0.0;
    for (int64_t i = 0; i < w.numel(); i++) {
        double v = (double)d[i] - (double)kTarget;
        s += v * v;
    }
    return (float)(0.5 * s);
}

void quad_grad(Tensor& w) {
    Tensor g(w.shape(), DType::F32);
    const float* d = w.data<float>();
    float* gd = g.data<float>();
    for (int64_t i = 0; i < w.numel(); i++) gd[i] = d[i] - kTarget;
    w.set_grad(g);
}

// Runs kSteps exact-gradient steps on f(w) = 0.5*||w - 1||^2 from w = 0.
// A correct optimizer must strictly reduce the loss with finite values.
void expect_converges(const char* name, Optimizer& opt) {
    TEST_SUITE(name);
    Tensor w(Shape{4, 4}, DType::F32);  // 2-D: satisfies Adafactor's row/col stats
    w.fill(0.0f);
    w.requires_grad(true);
    opt.add_param(&w);

    float before = quad_loss(w);
    for (int s = 0; s < kSteps; s++) {
        quad_grad(w);
        opt.step();
    }
    float after = quad_loss(w);
    printf("  %s: loss %.6f -> %.6f\n", name, before, after);
    TEST_CHECK(std::isfinite(after), "final loss finite");
    TEST_CHECK(after < before, "loss decreased after optimizer steps");
    TEST_CHECK(w.data<float>()[0] != 0.0f, "parameters moved");

    opt.zero_grad();
    bool cleared = true;
    if (w.has_grad()) {
        const float* g = w.grad().data<float>();
        for (int64_t i = 0; i < w.numel(); i++)
            if (g[i] != 0.0f) { cleared = false; break; }
    }
    TEST_CHECK(cleared, "zero_grad clears gradient");
}

} // namespace

static void test_adamw() {
    AdamW opt(0.05f, 0.9f, 0.999f, 1e-8f, 0.0f);
    expect_converges("optimizer adamw", opt);
}

static void test_sgd() {
    SGD opt(0.1f, 0.9f, 0.0f, false);
    expect_converges("optimizer sgd", opt);
}

static void test_adam() {
    Adam opt(0.05f, 0.9f, 0.999f, 1e-8f, 0.0f);
    expect_converges("optimizer adam", opt);
}

static void test_adamax() {
    Adamax opt(0.05f, 0.9f, 0.999f, 1e-8f, 0.0f);
    expect_converges("optimizer adamax", opt);
}

static void test_nadam() {
    NAdam opt(0.02f, 0.9f, 0.999f, 1e-8f, 0.0f, 0.004f);
    expect_converges("optimizer nadam", opt);
}

static void test_radam() {
    RAdam opt(0.02f, 0.9f, 0.999f, 1e-8f, 0.0f);
    expect_converges("optimizer radam", opt);
}

static void test_lion() {
    Lion opt(0.02f, 0.9f, 0.99f, 0.0f);
    expect_converges("optimizer lion", opt);
}

static void test_adafactor() {
    Adafactor opt(0.02f, 0.999f, 1e-30f, 0.0f, 1.0f, 0.8f);
    expect_converges("optimizer adafactor", opt);
}

// Same convergence check on a caller-supplied shape.
void expect_converges_rank(const char* name, Optimizer& opt, const Shape& shape) {
    TEST_SUITE(name);
    Tensor w(shape, DType::F32);
    w.fill(0.0f);
    w.requires_grad(true);
    opt.add_param(&w);

    float before = quad_loss(w);
    for (int s = 0; s < kSteps; s++) {
        quad_grad(w);
        opt.step();
    }
    float after = quad_loss(w);
    printf("  %s: loss %.6f -> %.6f (rank %d)\n", name, before, after, shape.rank);
    TEST_CHECK(std::isfinite(after), "final loss finite");
    TEST_CHECK(after < before, "loss decreased on non-2-D parameter");
}

// Regression, 2026-09-12. Adafactor used param->dim(1) unconditionally to size
// its column factor. Shape zero-fills dims beyond rank, so a 1-D parameter (any
// bias) yielded d1 == 0 and the update loop then evaluated `i / d1` — integer
// division by zero, i.e. SIGFPE on x86 and UB per the standard. Every model with
// a bias crashed under Adafactor. The 2-D case below is covered by
// test_adafactor(); these two shapes are the ones that used to die.
static void test_adafactor_rank_safety() {
    Adafactor opt1d(0.02f, 0.999f, 1e-30f, 0.0f, 1.0f, 0.8f);
    expect_converges_rank("optimizer adafactor 1-D bias", opt1d, Shape{8});

    Adafactor opt3d(0.02f, 0.999f, 1e-30f, 0.0f, 1.0f, 0.8f);
    expect_converges_rank("optimizer adafactor 3-D conv", opt3d, Shape{2, 2, 2});
}

static void test_rmsprop() {
    RMSProp opt(0.02f, 0.99f, 1e-8f, 0.0f, 0.0f);
    expect_converges("optimizer rmsprop", opt);
}

static void test_ranger() {
    Ranger opt(0.02f, 0.9f, 0.999f, 1e-8f, 0.0f, 6, 0.5f);
    expect_converges("optimizer ranger", opt);
}

static void test_adabound() {
    Adabound opt(0.02f, 0.9f, 0.999f, 1e-8f, 0.0f, 0.1f, 1e-3f);
    expect_converges("optimizer adabound", opt);
}

static void test_lamb() {
    Lamb opt(0.02f, 0.9f, 0.999f, 1e-8f, 0.0f, 10.0f);
    expect_converges("optimizer lamb", opt);
}

static void test_lars() {
    LARS opt(0.05f, 0.9f, 0.0f, 0.001f, 1e-8f);
    expect_converges("optimizer lars", opt);
}

static void test_novograd() {
    NovoGrad opt(0.05f, 0.9f, 0.999f, 1e-8f, 0.0f);
    expect_converges("optimizer novograd", opt);
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("Transcender — Optimizer Test Suite\n");
    printf("==========================================\n");

    test_adamw();
    test_sgd();
    test_adam();
    test_adamax();
    test_nadam();
    test_radam();
    test_lion();
    test_adafactor();
    test_adafactor_rank_safety();
    test_rmsprop();
    test_ranger();
    test_adabound();
    test_lamb();
    test_lars();
    test_novograd();

    printf("\n==========================================\n");
    return TEST_REPORT() > 0 ? 1 : 0;
}
