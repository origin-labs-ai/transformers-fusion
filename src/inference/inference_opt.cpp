// Phase 5-F6 split: src/inference/inference_opt.cpp is now thin.
// All class method bodies moved verbatim to inference_paging.cpp,
// inference_speculative.cpp, inference_kv.cpp, inference_lowprec.cpp and
// inference_serve.cpp. This TU retains only the shared free helper
// compute_logprobs (D17, unassigned by the split spec) plus the common
// include block. Declarations stay in include/quant/inference_opt.h.
#include "quant/inference_opt.h"
#include "quant/math.h"
#include "quant/int8_quant.h"
#include "quant/random.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <random>
#include <sstream>
#include <cstdio>
namespace quant {

// H3: unified epsilon shared with inference_paging.cpp /
// inference_speculative.cpp (same name, same value, file-local so no
// header/ODR touch).
namespace { constexpr float kInferenceEps = 1e-10f; }

// ===========================================================================
// D17: Logprob computation — log softmax output
// ===========================================================================
std::vector<std::vector<float>> compute_logprobs(
    const Tensor& logits, const std::vector<int>& tokens) {
    // BUGFIX (bug census): V<=0 → div-by-zero downstream (numel()/V) and
    // null data deref; non-finite logits → NaN logprobs. Guarded now.
    if (logits.rank() < 1) return {};
    int64_t V = logits.dim(logits.rank() - 1);
    if (V <= 0) return {};
    const float* base = logits.data<float>();
    if (!base) return {};
    int64_t S = (int64_t)tokens.size();
    if (S == 0) return {};

    int64_t logit_S = logits.numel() / V;
    std::vector<std::vector<float>> result((size_t)S);

    for (int64_t i = 0; i < S && i < logit_S; i++) {
        const float* row = base + i * V;
        float max_l = -INFINITY;
        for (int64_t v = 0; v < V; v++)
            if (std::isfinite(row[v])) max_l = std::max(max_l, row[v]);
        if (!std::isfinite(max_l)) max_l = 0.0f;
        float sum = 0;
        for (int64_t v = 0; v < V; v++) {
            float e = std::isfinite(row[v]) ? std::exp(row[v] - max_l) : 0.0f;
            sum += e;
        }

        std::vector<float> log_probs((size_t)V);
        float log_sum = std::log(sum + kInferenceEps);
        if (!std::isfinite(log_sum)) log_sum = 0.0f;
        for (int64_t v = 0; v < V; v++)
            log_probs[(size_t)v] = (std::isfinite(row[v]) ? (row[v] - max_l) : -50.0f) - log_sum;
        result[(size_t)i] = log_probs;
    }

    return result;
}

} // namespace quant
