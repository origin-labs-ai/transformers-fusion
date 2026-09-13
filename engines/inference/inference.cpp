#include "inference.h"
#include "quant/random.h"
#include <chrono>
#include <cstring>
#include <algorithm>
#include <filesystem>

namespace quant {
namespace engine {

InferenceEngine::InferenceEngine() {
    std::memset(&stats_, 0, sizeof(stats_));
}

InferenceEngine::~InferenceEngine() {
    if (owns_model_) delete model_;
    if (owns_tokenizer_) delete tokenizer_;
}

void InferenceEngine::init(Model* model, Tokenizer* tokenizer, const EngineConfig& cfg) {
    model_ = model;
    tokenizer_ = tokenizer;
    config_ = cfg;
    owns_model_ = false;
    owns_tokenizer_ = false;

    if (config_.use_kv_cache) {
        kv_cache_ = std::make_unique<KVCache>();
        int64_t num_kv = model_->config.num_kv_heads > 0
            ? model_->config.num_kv_heads
            : model_->config.num_heads;
        kv_cache_->init(
            static_cast<int>(model_->config.num_layers),
            config_.max_seq_len,
            num_kv,
            model_->config.head_dim,
            false);
    }

    sampler_ = std::make_unique<Sampler>(
        make_seed(resolve_base_seed(0), SeedStream::EngineSampler, 0));
    generator_ = std::make_unique<Generator>(model_, tokenizer_);

    std::memset(&stats_, 0, sizeof(stats_));
}

static Status deserialize_config(const std::vector<uint8_t>& data, TransformerConfig& cfg) {
    if (data.size() < sizeof(int64_t) * 6 + sizeof(float) * 2 + sizeof(int64_t) * 3 + sizeof(int8_t) + sizeof(int64_t) + sizeof(int8_t))
        return Status::error("Config data too small");
    const uint8_t* ptr = data.data();
    std::memcpy(&cfg.vocab_size, ptr, sizeof(int64_t)); ptr += sizeof(int64_t);
    std::memcpy(&cfg.hidden_size, ptr, sizeof(int64_t)); ptr += sizeof(int64_t);
    std::memcpy(&cfg.num_layers, ptr, sizeof(int64_t)); ptr += sizeof(int64_t);
    std::memcpy(&cfg.num_heads, ptr, sizeof(int64_t)); ptr += sizeof(int64_t);
    std::memcpy(&cfg.head_dim, ptr, sizeof(int64_t)); ptr += sizeof(int64_t);
    std::memcpy(&cfg.ffn_hidden_size, ptr, sizeof(int64_t)); ptr += sizeof(int64_t);
    std::memcpy(&cfg.norm_eps, ptr, sizeof(float)); ptr += sizeof(float);
    std::memcpy(&cfg.rope_theta, ptr, sizeof(float)); ptr += sizeof(float);
    std::memcpy(&cfg.max_seq_len, ptr, sizeof(int64_t)); ptr += sizeof(int64_t);
    int8_t act;
    std::memcpy(&act, ptr, sizeof(int8_t)); ptr += sizeof(int8_t);
    cfg.activation = static_cast<Activation>(act);
    if (ptr + sizeof(int64_t) <= data.data() + data.size()) {
        std::memcpy(&cfg.num_kv_heads, ptr, sizeof(int64_t)); ptr += sizeof(int64_t);
    }
    if (ptr + sizeof(int8_t) <= data.data() + data.size()) {
        int8_t res;
        std::memcpy(&res, ptr, sizeof(int8_t));
        cfg.use_parallel_residual = (res != 0);
    }
    return Status::success();
}

Status InferenceEngine::load(const std::string& quant_path) {
    try {
        if (!std::filesystem::exists(quant_path))
            return Status::error("File not found: " + quant_path);

        // Free previously owned resources before loading new ones
        if (owns_model_) { delete model_; model_ = nullptr; owns_model_ = false; }
        if (owns_tokenizer_) { delete tokenizer_; tokenizer_ = nullptr; owns_tokenizer_ = false; }

        QUANTReader reader(quant_path);
        auto config_data = reader.read_config();

        TransformerConfig model_cfg;
        Status s = deserialize_config(config_data, model_cfg);
        if (!s) return s;

        auto model = std::make_unique<DenseModel>(model_cfg);
        model->load(quant_path);

        std::string vocab_path = quant_path;
        size_t dot = vocab_path.rfind('.');
        if (dot != std::string::npos)
            vocab_path = vocab_path.substr(0, dot);
        vocab_path += ".vocab";

        if (!std::filesystem::exists(vocab_path))
            return Status::error("Vocabulary file not found: " + vocab_path);

        auto tokenizer = std::make_unique<BPETokenizer>();
        tokenizer->load(vocab_path);

        EngineConfig cfg;
        cfg.max_seq_len = (std::min)(model_cfg.max_seq_len, (int64_t)2048);
        cfg.use_kv_cache = true;
        cfg.sampler.max_tokens = 2048;
        init(model.release(), tokenizer.release(), cfg);
        owns_model_ = true;
        owns_tokenizer_ = true;

        return Status::success();
    } catch (const std::exception& e) {
        return Status::error(std::string("Load failed: ") + e.what());
    }
}

std::string InferenceEngine::generate(const std::string& prompt) {
    if (!generator_) return {};
    return generator_->generate(prompt, config_.sampler);
}

std::pair<std::string, InferenceStats> InferenceEngine::generate_with_stats(const std::string& prompt) {
    if (!generator_) return { {}, stats_ };

    auto start = std::chrono::high_resolution_clock::now();

    GenerationResult result = generator_->generate_full(prompt, config_.sampler);

    auto end = std::chrono::high_resolution_clock::now();
    float duration = std::chrono::duration<float>(end - start).count();

    int num_tokens = static_cast<int>(result.token_ids.size());
    update_stats(num_tokens, result.duration_sec > 0 ? result.duration_sec : duration);

    return { result.text, stats_ };
}

std::vector<std::string> InferenceEngine::generate_batch(const std::vector<std::string>& prompts) {
    std::vector<std::string> results;
    results.reserve(prompts.size());
    for (const auto& prompt : prompts) {
        reset_context();
        results.push_back(generate(prompt));
    }
    return results;
}

void InferenceEngine::reset_context() {
    if (kv_cache_) {
        kv_cache_->clear();
    }
    std::memset(&stats_, 0, sizeof(stats_));
}

const EngineConfig& InferenceEngine::config() const {
    return config_;
}

InferenceStats InferenceEngine::last_stats() const {
    return stats_;
}

void InferenceEngine::update_stats(int tokens, float duration) {
    stats_.tokens_generated = tokens;
    stats_.duration_sec = duration;
    stats_.tokens_per_sec = (duration > 0.0f) ? static_cast<float>(tokens) / duration : 0.0f;
    stats_.peak_memory_bytes = kv_cache_ ? kv_cache_->size_bytes() : 0;
    stats_.kv_cache_utilization =
        (config_.use_kv_cache && kv_cache_ && kv_cache_->max_seq_len() > 0)
            ? static_cast<float>(kv_cache_->context_len()) / static_cast<float>(kv_cache_->max_seq_len())
            : 0.0f;
}

} // namespace engine
} // namespace quant
