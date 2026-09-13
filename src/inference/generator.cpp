#include "quant/generator.h"
#include "quant/tokenizer.h"
#include "quant/model.h"
#include "quant/sampler.h"
#include "quant/random.h"
#include <chrono>
#include <sstream>
#include <cstring>

namespace quant {

Generator::Generator(Model* model, Tokenizer* tokenizer)
    : model_(model), tokenizer_(tokenizer),
      sampler_(make_seed(resolve_base_seed(0), SeedStream::GeneratorSampler, 0)) {}

std::vector<int> Generator::generate_tokens(const std::vector<int>& input_ids,
                                              const SamplerConfig& cfg) {
    int64_t seq_len = input_ids.size();
    int64_t B = 1;
    
    Tensor input_tensor(Shape{B, seq_len}, DType::F32);
    float* id_ptr = input_tensor.data<float>();
    for (size_t i = 0; i < input_ids.size(); i++) {
        id_ptr[i] = static_cast<float>(input_ids[i]);
    }
    
    Tensor positions(Shape{B, seq_len}, DType::F32);
    float* pos_ptr = positions.data<float>();
    for (int64_t i = 0; i < seq_len; i++) {
        float fval = static_cast<float>(i);
        std::memcpy(pos_ptr + i, &fval, sizeof(float));
    }
    
    // Prefill
    kv_cache_.init((int)model_->config.num_layers, model_->config.max_seq_len,
                   model_->config.num_heads, model_->config.head_dim);
    Tensor logits = model_->forward(input_tensor, positions, &kv_cache_);
    int64_t vocab = logits.shape().dims[2];
    float* last_logits = (float*)logits.data() + (seq_len - 1) * vocab;
    // Pass prompt tokens so repetition penalty applies from first sample
    int next_token = sampler_.sample(last_logits, (int)vocab, cfg, input_ids);

    std::vector<int> output_ids(input_ids);
    output_ids.push_back(next_token);

    // L058: reasoning budget caps the decode loop.
    const int budget = cfg.effective_max_tokens();
    // Decode loop
    for (int t = 0; t < budget - 1; t++) {
        Tensor single_input(Shape{B, 1}, DType::F32);
        float fval = static_cast<float>(next_token);
        std::memcpy(single_input.data<float>(), &fval, sizeof(float));
        
        Tensor single_pos(Shape{B, 1}, DType::F32);
        float pval = static_cast<float>(seq_len + t);
        std::memcpy(single_pos.data<float>(), &pval, sizeof(float));
        
        logits = model_->forward(single_input, single_pos, &kv_cache_);
        float* out = (float*)logits.data();
        next_token = sampler_.sample(out, (int)vocab, cfg, output_ids);
        output_ids.push_back(next_token);
        
        if (next_token == tokenizer_->eos_id()) break;
    }
    
    return output_ids;
}

std::string Generator::generate(const std::string& prompt, const SamplerConfig& cfg) {
    auto ids = tokenizer_->encode(prompt);
    auto output_ids = generate_tokens(ids, cfg);
    return tokenizer_->decode(output_ids);
}

void Generator::generate_stream(const std::string& prompt, const SamplerConfig& cfg,
                                 std::function<void(const std::string&)> on_token) {
    // True token-by-token streaming (not fake whole-string dump)
    auto ids = tokenizer_->encode(prompt);
    if (ids.empty()) {
        on_token("");
        return;
    }

    int64_t seq_len = (int64_t)ids.size();
    int64_t B = 1;

    Tensor input_tensor(Shape{B, seq_len}, DType::F32);
    float* id_ptr = input_tensor.data<float>();
    for (size_t i = 0; i < ids.size(); i++) {
        float fval = static_cast<float>(ids[i]);
        std::memcpy(id_ptr + i, &fval, sizeof(float));
    }

    Tensor positions(Shape{B, seq_len}, DType::F32);
    float* pos_ptr = positions.data<float>();
    for (int64_t i = 0; i < seq_len; i++) {
        float fval = static_cast<float>(i);
        std::memcpy(pos_ptr + i, &fval, sizeof(float));
    }

    kv_cache_.init((int)model_->config.num_layers, model_->config.max_seq_len,
                   model_->config.num_heads, model_->config.head_dim);
    Tensor logits = model_->forward(input_tensor, positions, &kv_cache_);
    int64_t vocab = logits.shape().dims[2];
    float* last_logits = (float*)logits.data() + (seq_len - 1) * vocab;
    int next_token = sampler_.sample(last_logits, (int)vocab, cfg, ids);

    std::vector<int> history = ids;
    history.push_back(next_token);
    on_token(tokenizer_->decode({next_token}));

    if (next_token == tokenizer_->eos_id()) return;

    const int stream_budget = cfg.effective_max_tokens();
    for (int t = 0; t < stream_budget - 1; t++) {
        Tensor single_input(Shape{B, 1}, DType::F32);
        float fval = static_cast<float>(next_token);
        std::memcpy(single_input.data<float>(), &fval, sizeof(float));

        Tensor single_pos(Shape{B, 1}, DType::F32);
        float pval = static_cast<float>(seq_len + t);
        std::memcpy(single_pos.data<float>(), &pval, sizeof(float));

        logits = model_->forward(single_input, single_pos, &kv_cache_);
        float* out = (float*)logits.data();
        next_token = sampler_.sample(out, (int)vocab, cfg, history);
        history.push_back(next_token);
        on_token(tokenizer_->decode({next_token}));

        if (next_token == tokenizer_->eos_id()) break;
    }
}

GenerationResult Generator::generate_full(const std::string& prompt,
                                            const SamplerConfig& cfg) {
    auto start = std::chrono::high_resolution_clock::now();
    auto ids = tokenizer_->encode(prompt);
    auto output_ids = generate_tokens(ids, cfg);
    auto end = std::chrono::high_resolution_clock::now();
    
    GenerationResult result;
    result.token_ids = output_ids;
    result.text = tokenizer_->decode(output_ids);
    result.duration_sec = std::chrono::duration<float>(end - start).count();
    int new_tokens = (int)output_ids.size() - (int)ids.size();
    result.tokens_per_sec = result.duration_sec > 0.0f ? (int)(new_tokens / result.duration_sec) : 0;
    result.truncated = new_tokens >= cfg.effective_max_tokens();
    return result;
}

// ===========================================================================
// StreamingGenerator — token-by-token with callback control + logprobs
// ===========================================================================
StreamingGenerator::StreamingGenerator(Model* model, Tokenizer* tokenizer, uint64_t seed)
    : model_(model), tokenizer_(tokenizer), sampler_(seed) {}

void StreamingGenerator::set_event_handler(const TokenEventHandler& handler) {
    handler_ = handler;
}

void StreamingGenerator::request_stop() {
    stop_requested_.store(true);
}

float StreamingGenerator::tokens_per_sec() const {
    auto end = std::chrono::high_resolution_clock::now();
    float sec = std::chrono::duration<float>(end - start_time_).count();
    int n = num_tokens_generated();
    return sec > 0 ? n / sec : 0;
}

float StreamingGenerator::compute_logprob(const float* logits, int token, int vocab_size) {
    float max_val = -INFINITY;
    for (int i = 0; i < vocab_size; i++) max_val = std::max(max_val, logits[i]);
    float sum = 0;
    for (int i = 0; i < vocab_size; i++) sum += std::exp(logits[i] - max_val);
    return (logits[token] - max_val) - std::log(sum + 1e-10f);
}

std::vector<std::pair<int, float>> StreamingGenerator::top_k_logprobs(
    const float* logits, int k, int vocab_size) {
    std::vector<std::pair<int, float>> scored;
    scored.reserve((size_t)vocab_size);
    float max_val = -1e30f;
    for (int i = 0; i < vocab_size; i++) max_val = std::max(max_val, logits[i]);
    float sum = 0;
    for (int i = 0; i < vocab_size; i++) sum += std::exp(logits[i] - max_val);
    float log_sum = std::log(sum + 1e-10f);
    for (int i = 0; i < vocab_size; i++)
        scored.push_back({i, logits[i] - max_val - log_sum});
    std::partial_sort(scored.begin(), scored.begin() + std::min(k, vocab_size),
                      scored.end(),
                      [](auto& a, auto& b) { return a.second > b.second; });
    if (k < vocab_size) scored.resize(k);
    return scored;
}

bool StreamingGenerator::check_stop_sequences(const std::vector<int>& ids,
                                                const StreamingConfig& cfg) {
    for (int stop_id : cfg.stop_token_ids) {
        if (!ids.empty() && ids.back() == stop_id) return true;
    }
    if (!cfg.stop_strings.empty() && tokenizer_) {
        for (auto& stop_str : cfg.stop_strings) {
            if (stop_str.empty()) continue;
            std::string gen = tokenizer_->decode(ids);
            if (gen.size() >= stop_str.size() &&
                gen.substr(gen.size() - stop_str.size()) == stop_str) {
                return true;
            }
        }
    }
    return cfg.eos_id >= 0 && !ids.empty() && ids.back() == cfg.eos_id;
}

bool StreamingGenerator::generate(const std::vector<int>& prompt_ids,
                                   const StreamingConfig& cfg) {
    output_ids_.clear();
    token_logprobs_.clear();
    stopped_ = false;
    stop_requested_.store(false);

    SamplerConfig scfg;
    scfg.temperature = cfg.temperature;
    scfg.top_k = cfg.top_k;
    scfg.top_p = cfg.top_p;
    scfg.repetition_penalty = cfg.repetition_penalty;
    scfg.max_tokens = cfg.max_tokens;
    scfg.reasoning_budget = cfg.reasoning_budget;
    scfg.reasoning_max_tokens = cfg.reasoning_max_tokens;

    start_time_ = std::chrono::high_resolution_clock::now();

    if (handler_.on_start) handler_.on_start();

    output_ids_ = prompt_ids;

    int64_t B = 1;
    int64_t seq_len = (int64_t)prompt_ids.size();

    Tensor input_tensor(Shape{B, seq_len}, DType::F32);
    float* id_ptr = input_tensor.data<float>();
    for (size_t i = 0; i < prompt_ids.size(); i++) {
        float fval = static_cast<float>(prompt_ids[i]);
        std::memcpy(id_ptr + i, &fval, sizeof(float));
    }

    Tensor positions(Shape{B, seq_len}, DType::F32);
    float* pos_ptr = positions.data<float>();
    for (int64_t i = 0; i < seq_len; i++) {
        float fval = static_cast<float>(i);
        std::memcpy(pos_ptr + i, &fval, sizeof(float));
    }

    if (model_) {
        kv_cache_.init((int)model_->config.num_layers, model_->config.max_seq_len,
                       model_->config.num_heads, model_->config.head_dim);
    }

    Tensor logits;
    try {
        logits = model_->forward(input_tensor, positions, &kv_cache_);
    } catch (const std::exception& e) {
        if (handler_.on_error) handler_.on_error(e.what());
        return false;
    }

    int64_t V = logits.dim(logits.rank() - 1);
    float* last_logits = (float*)logits.data() + (seq_len - 1) * V;

    int next_token;
    if (cfg.stream_logprobs) {
        next_token = sampler_.sample(last_logits, (int)V, scfg, prompt_ids);
        float lp = compute_logprob(last_logits, next_token, (int)V);
        token_logprobs_.push_back(top_k_logprobs(last_logits, cfg.logprobs_k, (int)V));
    } else {
        next_token = sampler_.sample(last_logits, (int)V, scfg, prompt_ids);
    }

    output_ids_.push_back(next_token);

    if (handler_.on_token) {
        bool cont = handler_.on_token(next_token,
            cfg.stream_logprobs ? compute_logprob(last_logits, next_token, (int)V) : 0.0f);
        if (!cont || stop_requested_.load()) {
            stopped_ = true;
            if (handler_.on_stop) handler_.on_stop();
            if (handler_.on_complete) handler_.on_complete(tokenizer_->decode(output_ids_));
            return true;
        }
    }

    if (check_stop_sequences(output_ids_, cfg)) {
        stopped_ = true;
        if (handler_.on_stop) handler_.on_stop();
        if (handler_.on_complete) handler_.on_complete(tokenizer_->decode(output_ids_));
        return true;
    }

    for (int t = 1; t < scfg.effective_max_tokens() && !stop_requested_.load(); t++) {
        try {
            Tensor single_input(Shape{B, 1}, DType::F32);
            float fval = static_cast<float>(next_token);
            std::memcpy(single_input.data<float>(), &fval, sizeof(float));

            Tensor single_pos(Shape{B, 1}, DType::F32);
            float pval = static_cast<float>(seq_len + t - 1);
            std::memcpy(single_pos.data<float>(), &pval, sizeof(float));

            logits = model_->forward(single_input, single_pos, &kv_cache_);
            float* out = (float*)logits.data();

            if (cfg.stream_logprobs) {
                next_token = sampler_.sample(out, (int)V, scfg, output_ids_);
                token_logprobs_.push_back(top_k_logprobs(out, cfg.logprobs_k, (int)V));
            } else {
                next_token = sampler_.sample(out, (int)V, scfg, output_ids_);
            }

            output_ids_.push_back(next_token);

            if (handler_.on_token) {
                float lp = cfg.stream_logprobs ?
                    compute_logprob(out, next_token, (int)V) : 0.0f;
                bool cont = handler_.on_token(next_token, lp);
                if (!cont) {
                    stopped_ = true;
                    break;
                }
            }

            if (check_stop_sequences(output_ids_, cfg)) {
                stopped_ = true;
                break;
            }
        } catch (const std::exception& e) {
            if (handler_.on_error) handler_.on_error(e.what());
            stopped_ = true;
            return false;
        }
    }

    if (stop_requested_.load() && !stopped_) {
        stopped_ = true;
        if (handler_.on_stop) handler_.on_stop();
    }

    if (handler_.on_complete) handler_.on_complete(tokenizer_->decode(output_ids_));
    return true;
}

bool StreamingGenerator::generate(const std::string& prompt, const StreamingConfig& cfg) {
    if (!tokenizer_) {
        if (handler_.on_error) handler_.on_error("no tokenizer");
        return false;
    }
    auto ids = tokenizer_->encode(prompt);
    if (ids.empty()) ids.push_back(1);
    return generate(ids, cfg);
}

} // namespace quant
