#pragma once
#include "quant/model.h"
#include "quant/tokenizer.h"
#include "quant/types.h"
#include <vector>
#include <string>
#include <future>

namespace quant {

struct RolloutConfig {
    int group_size = 4;
    int max_new_tokens = 32;
    float temperature = 0.8f;
    int max_seq_len = 256;
};

class RolloutWorker {
public:
    explicit RolloutWorker(Model* model, Tokenizer* tok = nullptr,
                           const RolloutConfig& cfg = RolloutConfig{});
    // Synchronous single-prompt rollout (sampled generation)
    std::vector<int> generate(const std::string& prompt);
    // Batched generation (parallel via std::async if group_size>1)
    std::vector<std::vector<int>> generate_batch(
        const std::vector<std::string>& prompts);

    int param_count() const;

private:
    Model* model_;
    Tokenizer* tok_;
    RolloutConfig cfg_;
};

} // namespace quant
