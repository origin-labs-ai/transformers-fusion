#include "quant/rollout_worker.h"
#include "quant/tensor.h"
#include "quant/random.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace quant {

// ========================================================================
// L061: AsyncRolloutBuffer — implementation
// ========================================================================

AsyncRolloutBuffer::AsyncRolloutBuffer(size_t capacity)
    : capacity_(capacity > 0 ? capacity : 1) {}

void AsyncRolloutBuffer::set_deterministic_replay(bool v) {
    std::lock_guard<std::mutex> lk(mutex_);
    deterministic_replay_ = v;
}

bool AsyncRolloutBuffer::deterministic_replay() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return deterministic_replay_;
}

bool AsyncRolloutBuffer::push(Trajectory t) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (closed_) return false;
    if (queue_.size() >= capacity_) return false; // backpressure: drop, don't block
    t.seq = next_seq_++;
    queue_.emplace(t.seq, std::move(t));
    cv_.notify_one();
    return true;
}

bool AsyncRolloutBuffer::pop(Trajectory& out) {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return closed_ || !queue_.empty(); });
    if (queue_.empty()) return false; // closed AND empty
    auto it = queue_.begin(); // smallest seq => deterministic FIFO
    out = std::move(it->second);
    queue_.erase(it);
    return true;
}

bool AsyncRolloutBuffer::try_pop(Trajectory& out) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (queue_.empty()) return false;
    auto it = queue_.begin();
    out = std::move(it->second);
    queue_.erase(it);
    return true;
}

void AsyncRolloutBuffer::close() {
    std::lock_guard<std::mutex> lk(mutex_);
    closed_ = true;
    cv_.notify_all();
}

bool AsyncRolloutBuffer::closed() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return closed_;
}

size_t AsyncRolloutBuffer::size() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return queue_.size();
}

size_t AsyncRolloutBuffer::capacity() const { return capacity_; }

std::vector<Trajectory> AsyncRolloutBuffer::replay() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<Trajectory> out;
    out.reserve(queue_.size());
    for (const auto& kv : queue_) out.push_back(kv.second); // seq order
    return out;
}

// ========================================================================
// RolloutWorker
// ========================================================================

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
    // L058: reasoning budget caps generation depth.
    budget = std::min({budget, cfg_.max_new_tokens, cfg_.effective_max_new_tokens()});
    // BUGFIX: hardcoded RNG(42) duplicated sequences across workers/runs
    // (nondeterministic-without-seed in reverse: identical every run). Derive
    // a unique stream; QUANT_SEED controls determinism centrally.
    RNG rng(GlobalSeedManager::next_seed());
    // BUGFIX: clamp temp (was std::max(0.1f,...) -> wrong distribution for
    // temp<0.1 and div-by-zero for NaN); non-finite temp falls back to greedy.
    float temp = cfg_.temperature;
    bool greedy = !std::isfinite(temp) || temp < 1e-5f;
    if (std::isfinite(temp)) {
        if (temp < 1e-5f) temp = 1e-5f;
        else if (temp > 100.0f) temp = 100.0f;
    }
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
        // BUGFIX: empty vocab slice (min(V,vocab)<=0) left mx=-inf and se=0
        // -> rnd=NaN (0*inf path) and nxt default; guard before softmax.
        int64_t n = std::min<int64_t>(V, vocab);
        if (n <= 0) { tokens.push_back(0); continue; }
        float mx = -std::numeric_limits<float>::infinity();
        for (int64_t v = 0; v < n; ++v) mx = std::max(mx, lp[v]);
        // BUGFIX: all -inf/NaN logits made mx non-finite -> (lp-mx) NaN and
        // se=0 -> div-by-zero unnormalized probs; fall back to token 0.
        if (!std::isfinite(mx)) { tokens.push_back(0); continue; }
        if (greedy) {
            int best = 0;
            for (int64_t v = 1; v < n; ++v) if (lp[v] > lp[best]) best = (int)v;
            tokens.push_back(best);
            continue;
        }
        float se = 0.0f;
        for (int64_t v = 0; v < n; ++v) {
            // BUGFIX: clamp scaled logit to [-50,50] so exp() can't overflow
            // to inf (inf/inf -> NaN cumulative, biased sampling).
            float z = (lp[v] - mx) / temp;
            if (!std::isfinite(z)) z = (z > 0) ? 50.0f : -50.0f;
            else if (z > 50.0f) z = 50.0f;
            else if (z < -50.0f) z = -50.0f;
            se += std::exp(z);
        }
        // BUGFIX: degenerate se (<=0/non-finite) made rnd NaN and cum never
        // reach it; fall back to argmax instead of silently picking token 0.
        if (!std::isfinite(se) || se <= 0.0f) {
            int best = 0;
            for (int64_t v = 1; v < n; ++v) if (lp[v] > lp[best]) best = (int)v;
            tokens.push_back(best);
            continue;
        }
        float rnd = rng.uniform() * se;
        float cum = 0.0f; int nxt = 0;
        for (int64_t v = 0; v < n; ++v) {
            float z = (lp[v] - mx) / temp;
            if (!std::isfinite(z)) z = (z > 0) ? 50.0f : -50.0f;
            else if (z > 50.0f) z = 50.0f;
            else if (z < -50.0f) z = -50.0f;
            cum += std::exp(z);
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

Trajectory RolloutWorker::generate_trajectory(const std::string& prompt, float reward) {
    Trajectory t;
    t.tokens = generate(prompt);
    t.reward = reward;
    t.done = true;
    return t;
}

void RolloutWorker::produce_to(AsyncRolloutBuffer& buf,
                               const std::vector<std::string>& prompts) {
    for (const auto& p : prompts) {
        Trajectory t = generate_trajectory(p);
        // Spin on backpressure with a bounded retry so producers never lose
        // trajectories to a momentarily-full buffer (close() breaks the loop).
        for (int retry = 0; retry < 10000; ++retry) {
            if (buf.push(t)) break;
            if (buf.closed()) return;
        }
    }
}

int RolloutWorker::param_count() const { return model_ ? (int)model_->param_count() : 0; }

} // namespace quant
