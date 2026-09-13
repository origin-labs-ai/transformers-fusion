#pragma once
#include "quant/types.h"
#include "quant/tensor.h"
#include "quant/model.h"
#include "quant/sampler.h"
#include "quant/tokenizer.h"
#include "quant/kv_cache.h"
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <atomic>
#include <chrono>

namespace quant {

struct GenerationResult {
    std::vector<int> token_ids;
    std::string text;
    float duration_sec;
    int tokens_per_sec;
    bool truncated;
};

class Generator {
public:
    Generator(Model* model, Tokenizer* tokenizer);
    
    std::vector<int> generate_tokens(const std::vector<int>& input_ids,
                                      const SamplerConfig& cfg);
    std::string generate(const std::string& prompt, const SamplerConfig& cfg);
    void generate_stream(const std::string& prompt, const SamplerConfig& cfg,
                          std::function<void(const std::string&)> on_token);
    GenerationResult generate_full(const std::string& prompt, const SamplerConfig& cfg);
    
private:
    Model* model_;
    Tokenizer* tokenizer_;
    Sampler sampler_;
    KVCache kv_cache_;
};

// ===========================================================================
// StreamingGenerator — token-by-token generation with callback control
// ===========================================================================
enum class TokenEvent {
    START,
    TOKEN,
    COMPLETE,
    ERROR,
    STOP
};

struct TokenEventHandler {
    std::function<void()> on_start;
    std::function<bool(int, float)> on_token;       // token_id, logprob -> continue flag
    std::function<void(const std::string&)> on_complete;
    std::function<void(const std::string&)> on_error;
    std::function<void()> on_stop;
};

struct StreamingConfig {
    int max_tokens = 512;
    int eos_id = 2;
    float temperature = 1.0f;
    int top_k = 40;
    float top_p = 0.9f;
    float repetition_penalty = 1.0f;
    std::vector<int> stop_token_ids;
    std::vector<std::string> stop_strings;
    bool stream_logprobs = false;
    int logprobs_k = 5;
    // L058: reasoning-budget tier + explicit override, same semantics as
    // SamplerConfig (budget caps the requested max_tokens).
    ReasoningBudget reasoning_budget = ReasoningBudget::High;
    int reasoning_max_tokens = 0;
    int effective_max_tokens() const {
        int want = reasoning_max_tokens > 0 ? reasoning_max_tokens : max_tokens;
        int cap = reasoning_budget_cap(reasoning_budget);
        return want < cap ? want : cap;
    }
};

class StreamingGenerator {
public:
    StreamingGenerator(Model* model, Tokenizer* tokenizer, uint64_t seed = 42);
    
    void set_event_handler(const TokenEventHandler& handler);
    
    bool generate(const std::vector<int>& prompt_ids, const StreamingConfig& cfg);
    bool generate(const std::string& prompt, const StreamingConfig& cfg);
    
    std::vector<int> output_ids() const { return output_ids_; }
    std::vector<std::vector<std::pair<int, float>>> token_logprobs() const { return token_logprobs_; }
    
    float tokens_per_sec() const;
    int num_tokens_generated() const { return (int)output_ids_.size() - 1; }
    bool stopped() const { return stopped_; }
    
    void request_stop();
    
private:
    Model* model_;
    Tokenizer* tokenizer_;
    Sampler sampler_;
    KVCache kv_cache_;
    TokenEventHandler handler_;
    
    std::vector<int> output_ids_;
    std::vector<std::vector<std::pair<int, float>>> token_logprobs_;
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> stopped_{false};
    std::chrono::high_resolution_clock::time_point start_time_;
    
    float compute_logprob(const float* logits, int token, int vocab_size);
    std::vector<std::pair<int, float>> top_k_logprobs(const float* logits, int k, int vocab_size);
    bool check_stop_sequences(const std::vector<int>& ids, const StreamingConfig& cfg);
};

} // namespace quant
