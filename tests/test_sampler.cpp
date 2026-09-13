#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
#include "quant/sampler.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

void test_greedy() {
    quant::Sampler s(42);
    float logits[4] = {0.1f, 2.5f, 1.0f, -0.5f};
    assert(s.greedy(logits, 4) == 1);
    float peaked[4] = {-1.0f, -2.0f, 5.0f, 0.0f};
    assert(s.greedy(peaked, 4) == 2);
}

void test_temperature() {
    quant::Sampler s(42);
    float logits[3] = {1.0f, 2.0f, 4.0f};
    s.apply_temperature(logits, 3, 2.0f);
    assert(std::abs(logits[0] - 0.5f) < 1e-6f);
    assert(std::abs(logits[1] - 1.0f) < 1e-6f);
    assert(std::abs(logits[2] - 2.0f) < 1e-6f);
    // Near-zero temp must clamp, not crash or produce non-finite.
    float tiny[2] = {1.0f, -1.0f};
    s.apply_temperature(tiny, 2, 0.0f);
    assert(std::isfinite(tiny[0]));
    assert(std::isfinite(tiny[1]));
}

void test_repetition_penalty() {
    quant::Sampler s(42);
    // penalty <= 1 is a no-op.
    float logits[3] = {1.0f, -2.0f, 0.5f};
    s.apply_repetition_penalty(logits, 3, {0, 1}, 1.0f);
    assert(std::abs(logits[0] - 1.0f) < 1e-6f);
    assert(std::abs(logits[1] + 2.0f) < 1e-6f);
    // Positive logit is divided, negative logit is multiplied.
    float pen[3] = {2.0f, -2.0f, 1.0f};
    s.apply_repetition_penalty(pen, 3, {0, 1}, 2.0f);
    assert(std::abs(pen[0] - 1.0f) < 1e-6f);
    assert(std::abs(pen[1] + 4.0f) < 1e-6f);
    assert(std::abs(pen[2] - 1.0f) < 1e-6f);
    // Out-of-range prev tokens are ignored.
    float oor[2] = {0.5f, 0.5f};
    s.apply_repetition_penalty(oor, 2, {-1, 99}, 2.0f);
    assert(std::abs(oor[0] - 0.5f) < 1e-6f);
    assert(std::abs(oor[1] - 0.5f) < 1e-6f);
}

void test_top_k_basic() {
    quant::Sampler s(42);
    float logits[4] = {0.1f, 2.5f, 1.0f, -0.5f};
    // k=1 always returns the argmax.
    for (int i = 0; i < 5; i++) {
        assert(s.sample_top_k(logits, 4, 1, 1.0f) == 1);
    }
    // Near-zero temp falls back to greedy.
    assert(s.sample_top_k(logits, 4, 2, 0.0f) == 1);
    // k=2 over a peaked distribution only returns the top-2 ids.
    float peaked[8] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f, -1.0f, -2.0f};
    for (int i = 0; i < 20; i++) {
        int id = s.sample_top_k(peaked, 8, 2, 1.0f);
        assert(id == 0 || id == 1);
    }
}

void test_top_p_basic() {
    quant::Sampler s(42);
    float logits[4] = {0.1f, 2.5f, 1.0f, -0.5f};
    // Near-zero temp falls back to greedy.
    assert(s.sample_top_p(logits, 4, 0.9f, 0.0f) == 1);
    // Sharply peaked distribution with small p collapses to the argmax.
    float peaked[4] = {10.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 5; i++) {
        assert(s.sample_top_p(peaked, 4, 0.5f, 1.0f) == 0);
    }
    // General case stays in range.
    for (int i = 0; i < 20; i++) {
        int id = s.sample_top_p(logits, 4, 0.9f, 1.0f);
        assert(id >= 0 && id < 4);
    }
}

void test_sample_dispatch() {
    quant::Sampler s(42);
    float logits[4] = {0.1f, 2.5f, 1.0f, -0.5f};
    // top_p < 1 takes the top-p path; top_p == 1 takes the top-k path.
    quant::SamplerConfig cfg_p;
    cfg_p.temperature = 1.0f;
    cfg_p.top_k = 2;
    cfg_p.top_p = 0.9f;
    for (int i = 0; i < 10; i++) {
        int id = s.sample(logits, 4, cfg_p, {});
        assert(id >= 0 && id < 4);
    }
    quant::SamplerConfig cfg_k;
    cfg_k.temperature = 1.0f;
    cfg_k.top_k = 1;
    cfg_k.top_p = 1.0f;
    for (int i = 0; i < 5; i++) {
        assert(s.sample(logits, 4, cfg_k, {}) == 1);
    }
    // Repetition penalty path stays in range.
    quant::SamplerConfig cfg_r;
    cfg_r.temperature = 1.0f;
    cfg_r.top_k = 4;
    cfg_r.top_p = 1.0f;
    cfg_r.repetition_penalty = 1.5f;
    for (int i = 0; i < 10; i++) {
        int id = s.sample(logits, 4, cfg_r, {1});
        assert(id >= 0 && id < 4);
    }
}

int main() {
    std::cout << "[Test] Running Sampler test..." << std::endl;
    test_greedy();
    test_temperature();
    test_repetition_penalty();
    test_top_k_basic();
    test_top_p_basic();
    test_sample_dispatch();
    std::cout << "Sampler test passed!" << std::endl;
    return 0;
}
