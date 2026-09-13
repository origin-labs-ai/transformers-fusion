#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
// test_continual_anticollapse.cpp — Unit test for Continual Learning Anti-Collapse mechanisms
#include "quant/continual_engine.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << " Continual Learning Anti-Collapse Test  " << std::endl;
    std::cout << "=========================================" << std::endl;

    std::cout << "[Test 1] Testing Orthogonal Projection Constraint..." << std::endl;
    constexpr std::size_t N = 100;
    std::vector<float> base_weight(N, 1.0f); // Base direction [1, 1, ..., 1]
    std::vector<float> update_delta(N);
    for (std::size_t i = 0; i < N; i++) update_delta[i] = (float)i * 0.1f;

    // Apply orthogonal projection: void apply_orthogonal_projection(float* update, const float* weights, std::size_t n)
    quant::apply_orthogonal_projection(update_delta.data(), base_weight.data(), N);

    // Verify dot product between base_weight and updated delta is ~0 (orthogonal)
    double dot = 0.0;
    for (std::size_t i = 0; i < N; i++) dot += (double)base_weight[i] * (double)update_delta[i];
    std::cout << "  -> Dot product after orthogonal projection: " << dot << std::endl;
    assert(std::abs(dot) < 1e-4);
    std::cout << "  -> PASSED: Orthogonal projection constraint verified!" << std::endl;

    std::cout << "[Test 2] Testing Experience Replay Buffer & Generation Tagging..." << std::endl;
    quant::ExperienceReplayBuffer buffer(100, 10);

    // Generation 1 sample (allowed)
    std::vector<float> sample_g1 = {1.0f, 2.0f, 3.0f};
    buffer.add(sample_g1, sample_g1, 1);

    // Generation 3 sample (must be rejected by generation tagging)
    std::vector<float> sample_g3 = {9.0f, 9.0f, 9.0f};
    buffer.add(sample_g3, sample_g3, 3);

    assert(buffer.size() == 1);
    std::cout << "  -> PASSED: Generation tagging filter verified (Gen 3 rejected, Gen 1 stored)!" << std::endl;

    std::cout << "[Test 3] Testing Entropy Floor Injection..." << std::endl;
    std::vector<float> weights(N, 0.5f);
    std::vector<float> importance(N, 1.0f);
    float low_entropy = 1.2f; // Below floor of 2.0

    // void apply_entropy_floor(float output_entropy, float* weights, const float* importance, std::size_t n, float noise_scale = 0.01f)
    quant::apply_entropy_floor(low_entropy, weights.data(), importance.data(), N, 0.01f);
    std::cout << "  -> PASSED: Entropy floor injection executed cleanly!" << std::endl;

    std::cout << "\nCONTINUAL LEARNING ANTI-COLLAPSE TEST PASSED!" << std::endl;
    return 0;
}
