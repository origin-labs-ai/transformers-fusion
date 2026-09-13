#pragma once
#include "quant/model.h"
#include "quant/tokenizer.h"
#include "quant/types.h"
#include "quant/sampler.h"
#include <vector>
#include <string>
#include <future>
#include <mutex>
#include <condition_variable>
#include <map>
#include <cstdint>

namespace quant {

struct RolloutConfig {
    int group_size = 4;
    int max_new_tokens = 32;
    float temperature = 0.8f;
    int max_seq_len = 256;
    // L058: reasoning budget caps max_new_tokens (same cap table as sampler).
    ReasoningBudget reasoning_budget = ReasoningBudget::High;
    int effective_max_new_tokens() const {
        int cap = reasoning_budget_cap(reasoning_budget);
        return max_new_tokens < cap ? max_new_tokens : cap;
    }
};

// L061: single trajectory produced by a rollout worker.
struct Trajectory {
    uint64_t seq = 0;              // global push order (deterministic replay key)
    std::vector<int> tokens;       // prompt + generated tokens
    float reward = 0.0f;           // filled by the consumer / reward model
    bool done = true;
};

// L061: single-node async rollout buffer (producer/consumer skeleton).
// Producers (rollout workers) push() trajectories while the trainer consumes
// via pop(). Ordering is ALWAYS deterministic (sequence-numbered map, FIFO
// by push order); `deterministic_replay` additionally exposes an ordered
// snapshot via replay() for reproducible re-consumption. close() wakes all
// waiters so shutdown never deadlocks.
class AsyncRolloutBuffer {
public:
    explicit AsyncRolloutBuffer(size_t capacity = 64);

    void set_deterministic_replay(bool v);
    bool deterministic_replay() const;

    // Returns false (drops) when the buffer is at capacity.
    bool push(Trajectory t);
    // Blocking pop; returns false only when closed AND empty.
    bool pop(Trajectory& out);
    bool try_pop(Trajectory& out);
    void close();
    bool closed() const;
    size_t size() const;
    size_t capacity() const;
    // Ordered snapshot of resident trajectories (replay in push order).
    std::vector<Trajectory> replay() const;

private:
    size_t capacity_;
    bool deterministic_replay_ = true;
    bool closed_ = false;
    uint64_t next_seq_ = 0;
    std::map<uint64_t, Trajectory> queue_; // seq-ordered => deterministic
    mutable std::mutex mutex_;
    std::condition_variable cv_;
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
    // L061: trajectory wrapper for the async buffer (reward filled by caller).
    Trajectory generate_trajectory(const std::string& prompt, float reward = 0.0f);
    // L061: produce a batch of trajectories into a shared buffer (producer).
    void produce_to(AsyncRolloutBuffer& buf,
                    const std::vector<std::string>& prompts);

    int param_count() const;

private:
    Model* model_;
    Tokenizer* tok_;
    RolloutConfig cfg_;
};

} // namespace quant
