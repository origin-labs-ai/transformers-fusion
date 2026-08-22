#include "quant/rollout_worker.h"
#include "quant/tensor.h"
#include "quant/random.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace quant {

RolloutWorker::RolloutWorker(Model* model, Tokenizer* tok,
                             const RolloutConfig& cfg)
    : model_(model), tok_(tok), cfg_(cfg) {}

std::vector<int> RolloutWorker::generate(const std::string& prompt) {
    if (!model_) return {};
    int vocab = (int)model_->config.vocab_size;
    if (vocab <= 1) vocab = 64;
    std::vector<int> tokens;
    if (tok_) tokens = tok_->encode(prompt);
    else for (char c : prompt) tokens.push_back((int)(unsigned char)c % vocab);
    if (tokens.empty()) tokens.push_back(1);
    int max_len = cfg_.max_seq_len > 0 ? cfg_.max_seq_len : 256;
    int budget = std::max(0, max_len - (int)tokens.size());
    budget = std::min(budget, cfg_.max_new_tokens);
    RNG rng(42);
    for (int step = 0; step < budget; ++step) {
        int64_t S = (int64_t)tokens.size();
        int64_t ctx = std::min<int64_t>(S, 32);
        int64_t start = S - ctx;
        Tensor ids(Shape{1, ctx});
        Tensor pos(Shape{1, ctx});
        for (int64_t i = 0; i < ctx; ++i) {
            ids.data<float>()[i] = (float)tokens[start + i];
            pos.data<float>()[i] = (float)(start + i);
        }
        Tensor logits = model_->forward(ids, pos, nullptr);
        int64_t V = logits.dim(logits.rank() - 1);
        const float* lp = logits.data<float>() + (ctx - 1) * V;
        float mx = -std::numeric_limits<float>::infinity();
        for (int64_t v = 0; v < std::min<int64_t>(V, vocab); ++v) mx = std::max(mx, lp[v]);
        float se = 0.0f;
        for (int64_t v = 0; v < std::min<int64_t>(V, vocab); ++v) se += std::exp((lp[v]-mx)/ std::max(0.1f, cfg_.temperature));
        float rnd = rng.uniform() * se;
        float cum = 0.0f; int nxt = 0;
        for (int64_t v = 0; v < std::min<int64_t>(V, vocab); ++v) {
            cum += std::exp((lp[v]-mx)/ std::max(0.1f, cfg_.temperature));
            if (cum >= rnd) { nxt = (int)v; break; }
        }
        tokens.push_back(nxt);
    }
    return tokens;
}

std::vector<std::vector<int>> RolloutWorker::generate_batch(
    const std::vector<std::string>& prompts) {
    std::vector<std::future<std::vector<int>>> futs;
    for (auto &p : prompts) futs.push_back(std::async(std::launch::async, [this,p]{ return generate(p); }));
    std::vector<std::vector<int>> out;
    for (auto &f : futs) out.push_back(f.get());
    return out;
}

int RolloutWorker::param_count() const { return model_ ? (int)model_->param_count() : 0; }

} // namespace quant
