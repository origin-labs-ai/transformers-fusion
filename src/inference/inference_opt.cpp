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
    int64_t V = logits.dim(logits.rank() - 1);
    int64_t S = (int64_t)tokens.size();
    if (S == 0) return {};

    int64_t logit_S = logits.numel() / V;
    std::vector<std::vector<float>> result((size_t)S);

    for (int64_t i = 0; i < S && i < logit_S; i++) {
        const float* row = logits.data<float>() + i * V;
        float max_l = -INFINITY;
        for (int64_t v = 0; v < V; v++) max_l = std::max(max_l, row[v]);
        float sum = 0;
        for (int64_t v = 0; v < V; v++) sum += std::exp(row[v] - max_l);

        std::vector<float> log_probs((size_t)V);
        float log_sum = std::log(sum + kInferenceEps);
        for (int64_t v = 0; v < V; v++)
            log_probs[(size_t)v] = (row[v] - max_l) - log_sum;
        result[(size_t)i] = log_probs;
    }

    return result;
}

} // namespace quant
