#pragma once

#include "quant/random.h"
#include <cstdint>
#include <vector>

namespace quant {

// L058: reasoning-budget tiers (serving flexibility: speed vs thoroughness).
// The budget caps sampling depth: the effective generation length is
// min(requested max_tokens, budget cap) unless reasoning_max_tokens sets an
// explicit (lower-or-equal) limit.
enum class ReasoningBudget : uint8_t { Low = 0, High = 1, Max = 2 };

inline int reasoning_budget_cap(ReasoningBudget b) {
    switch (b) {
        case ReasoningBudget::Low: return 256;
        case ReasoningBudget::Max: return 8192;
        case ReasoningBudget::High:
        default: return 2048;
    }
}

inline const char* reasoning_budget_name(ReasoningBudget b) {
    switch (b) {
        case ReasoningBudget::Low: return "low";
        case ReasoningBudget::Max: return "max";
        case ReasoningBudget::High:
        default: return "high";
    }
}

struct SamplerConfig {
    float temperature = 1.0f;
    int top_k = 40;
    float top_p = 0.9f;
    float repetition_penalty = 1.0f;
    // Default ceiling equals the Max-tier cap so every tier is reachable by
    // setting reasoning_budget alone (High still caps at 2048, Low at 256).
    // Callers that need a lower request set max_tokens explicitly.
    int max_tokens = 8192;
    // L058: budget tier + optional explicit override (0 = derive from tier).
    ReasoningBudget reasoning_budget = ReasoningBudget::High;
    int reasoning_max_tokens = 0;
    int effective_max_tokens() const {
        int want = reasoning_max_tokens > 0 ? reasoning_max_tokens : max_tokens;
        int cap = reasoning_budget_cap(reasoning_budget);
        return want < cap ? want : cap;
    }
};

class Sampler {
public:
    explicit Sampler(uint64_t seed = 42);

    // BUGFIX: reseed the sampler stream (default ctor seed 42 duplicated
    // sequences across workers; callers can derive per-worker seeds via
    // GlobalSeedManager::next_seed() / make_seed for determinism control).
    void set_seed(uint64_t s);

    int greedy(const float* logits, int vocab_size);
    int sample_top_k(const float* logits, int vocab_size, int k, float temp);
    int sample_top_p(const float* logits, int vocab_size, float p, float temp);
    int sample(const float* logits, int vocab_size, const SamplerConfig& cfg,
               const std::vector<int>& prev_tokens = {});

    void apply_temperature(float* logits, int n, float temp) const;
    void apply_repetition_penalty(float* logits, int n,
                                   const std::vector<int>& prev_tokens,
                                   float penalty) const;

private:
    RNG rng_;
};

} // namespace quant
