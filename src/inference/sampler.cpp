#include "quant/sampler.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <vector>

namespace quant {

Sampler::Sampler(uint64_t seed) : rng_(seed) {}

// BUGFIX: expose RNG reseed (see sampler.h); rng_ is otherwise fixed at
// construction, making every Sampler(42) emit identical streams.
void Sampler::set_seed(uint64_t s) { rng_.seed(s); }

int Sampler::greedy(const float* logits, int vocab_size) {
    // BUGFIX: guard empty/invalid vocab and null input (was UB: read of
    // logits[0..vocab) and meaningless return for vocab<=0).
    if (!logits || vocab_size <= 0) return -1;
    int best = 0;
    float best_val = -INFINITY;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

void Sampler::apply_temperature(float* logits, int n, float temp) const {
    // BUGFIX: guard null/n<=0 (was UB) and non-finite temp (NaN/inf temp
    // propagated NaNs into logits then div-by-zero downstream); clamp tiny temp.
    if (!logits || n <= 0) return;
    if (!std::isfinite(temp) || temp < 1e-5f) temp = 1e-5f;
    for (int i = 0; i < n; i++) {
        logits[i] /= temp;
    }
}

void Sampler::apply_repetition_penalty(float* logits, int n,
                                        const std::vector<int>& prev_tokens,
                                        float penalty) const {
    // BUGFIX: guard null/n<=0 and non-finite penalty (was UB / NaN logits).
    if (!logits || n <= 0) return;
    if (!std::isfinite(penalty)) return;
    if (penalty <= 1.0f) return;
    for (int t : prev_tokens) {
        if (t >= 0 && t < n) {
            logits[t] = (logits[t] < 0) ? logits[t] * penalty : logits[t] / penalty;
        }
    }
}

int Sampler::sample_top_k(const float* logits, int vocab_size, int k, float temp) {
    // BUGFIX: guard null/empty vocab (was UB) and normalize K edge cases:
    // K<=0 means argmax-only (not full vocab); K>vocab clamps to vocab.
    // BUGFIX: non-finite/near-zero temp routes to greedy (was div-by-zero or
    // NaN-scaled logits -> unnormalized/NaN probs).
    if (!logits || vocab_size <= 0) return -1;
    if (k <= 0) k = 1;
    if (k > vocab_size) k = vocab_size;
    if (!std::isfinite(temp) || temp < 1e-5f) return greedy(logits, vocab_size);
    
    std::vector<std::pair<float, int>> scored(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        // BUGFIX: clamp temp-scaled logits to [-50,50] so huge logits/tiny
        // temp can't overflow exp() (inf->NaN probs); exp arg <= 0 post max-sub.
        float v = logits[i] / temp;
        if (!std::isfinite(v)) v = (v > 0) ? 50.0f : -50.0f;
        else if (v > 50.0f) v = 50.0f;
        else if (v < -50.0f) v = -50.0f;
        scored[i] = {v, i};
    }

    // BUGFIX: nth_element with middle==end (k==vocab) is UB; skip partition
    // when the whole vocab is selected.
    if (k < vocab_size) {
        std::nth_element(scored.begin(), scored.begin() + k, scored.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    float max_val = -INFINITY;
    for (int i = 0; i < k; i++)
        if (scored[i].first > max_val) max_val = scored[i].first;
    float sum_exp = 0;
    for (int i = 0; i < k; i++) {
        sum_exp += std::exp(scored[i].first - max_val);
    }
    // BUGFIX: degenerate sum (all -inf/NaN -> sum<=0/non-finite) left probs
    // unnormalized (div-by-zero -> NaN cum); fall back to uniform over top-k.
    if (!std::isfinite(sum_exp) || sum_exp <= 0.0f) {
        int idx = (int)(rng_.uniform() * k);
        if (idx < 0) idx = 0;
        if (idx >= k) idx = k - 1;
        return scored[idx].second;
    }
    
    float r = rng_.uniform();
    float cum = 0;
    for (int i = 0; i < k; i++) {
        cum += std::exp(scored[i].first - max_val) / sum_exp;
        if (r < cum) return scored[i].second;
    }
    return scored[k - 1].second;
}

int Sampler::sample_top_p(const float* logits, int vocab_size, float p, float temp) {
    // BUGFIX: guard null/empty vocab (was UB: scored[0] OOB on empty) and
    // normalize p edges: p<=0 -> argmax-only (not NaN loop), p>=1 -> full
    // vocab distribution; non-finite/near-zero temp -> greedy (div-by-zero).
    if (!logits || vocab_size <= 0) return -1;
    if (!std::isfinite(temp) || temp < 1e-5f) return greedy(logits, vocab_size);
    if (!std::isfinite(p) || p <= 0.0f) return greedy(logits, vocab_size);
    if (p > 1.0f) p = 1.0f;

    std::vector<std::pair<float, int>> scored(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        // BUGFIX: clamp temp-scaled logits (same exp-overflow guard as top_k).
        float v = logits[i] / temp;
        if (!std::isfinite(v)) v = (v > 0) ? 50.0f : -50.0f;
        else if (v > 50.0f) v = 50.0f;
        else if (v < -50.0f) v = -50.0f;
        scored[i] = {v, i};
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    float max_val = scored[0].first;
    std::vector<float> probs(vocab_size);
    float sum_exp = 0;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = std::exp(scored[i].first - max_val);
        sum_exp += probs[i];
    }
    // BUGFIX: degenerate sum_exp (<=0/non-finite) caused div-by-zero ->
    // unnormalized/NaN probs; fall back to argmax.
    if (!std::isfinite(sum_exp) || sum_exp <= 0.0f) return scored[0].second;

    float cum = 0;
    int cutoff = vocab_size;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum_exp;
        // BUGFIX: clamp non-finite prob entries so cum stays normalized.
        if (!std::isfinite(probs[i])) probs[i] = 0.0f;
        cum += probs[i];
        if (cum >= p || i == vocab_size - 1) { cutoff = i + 1; break; }
    }
    
    // Re-normalize over cutoff
    float sub_sum = 0;
    for (int i = 0; i < cutoff; i++) sub_sum += probs[i];
    // BUGFIX: cutoff probs could sum to <=0/non-finite (unnormalized div);
    // fall back to argmax instead of dividing by zero.
    if (!std::isfinite(sub_sum) || sub_sum <= 0.0f) return scored[0].second;
    
    float r = rng_.uniform();
    cum = 0;
    for (int i = 0; i < cutoff; i++) {
        cum += probs[i] / sub_sum;
        if (r < cum) return scored[i].second;
    }
    return scored[cutoff - 1].second;
}

int Sampler::sample(const float* logits, int vocab_size, const SamplerConfig& cfg,
                    const std::vector<int>& prev_tokens) {
    // BUGFIX: guard null/empty vocab (was UB: vector ctor from null range).
    if (!logits || vocab_size <= 0) return -1;
    std::vector<float> adjusted(logits, logits + vocab_size);
    
    if (cfg.repetition_penalty > 1.0f) {
        apply_repetition_penalty(adjusted.data(), vocab_size, prev_tokens, cfg.repetition_penalty);
    }
    
    if (cfg.top_p < 1.0f) {
        return sample_top_p(adjusted.data(), vocab_size, cfg.top_p, cfg.temperature);
    }
    return sample_top_k(adjusted.data(), vocab_size, cfg.top_k, cfg.temperature);
}

} // namespace quant
