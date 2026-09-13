#pragma once
#include "quant/tensor.h"
#include "quant/model.h"
#include "quant/kv_cache.h"
#include "quant/generator.h"
#include "quant/sampler.h"
#include <queue>
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <mutex>

namespace quant {

// D1: Paged attention — vLLM-style block-level KV cache management
class PagedAttention {
public:
    struct Block { int64_t id; Tensor k; Tensor v; bool active; };
    PagedAttention(int64_t head_dim, int64_t n_heads, int64_t block_size = 32,
                   int64_t max_blocks = 256);
    Tensor forward(const Tensor& Q, int64_t* block_table,
                   Block* blocks, int64_t num_blocks);
    Block alloc_block();
    void free_block(int64_t id);
    int64_t num_blocks() const { return (int64_t)blocks_.size(); }
    int64_t available_blocks() const { return (int64_t)free_ids_.size(); }
private:
    int64_t head_dim_, n_heads_, block_size_;
    std::vector<Block> blocks_;
    std::queue<int64_t> free_ids_;
    int64_t max_blocks_;
    int64_t next_id_;
};

// D2: Speculative decoding — draft model + verify + adaptive gamma
class SpeculativeDecoder {
public:
    SpeculativeDecoder(Model* draft, Model* target, float gamma = 5.0f,
                       float min_gamma = 1.0f, float max_gamma = 10.0f);
    std::vector<int> generate(const std::vector<int>& prompt, int max_tokens);
    float acceptance_rate() const { return acceptance_rate_; }
    int accepted_count() const { return accepted_count_; }
    int total_count() const { return total_count_; }
    float current_gamma() const { return gamma_; }
    // P13: true jab draft_==nullptr uniform fallback use hua ho (legit path,
    // fake nahi). verify_tokens/generate me set hota hai; metric ke liye flag hai.
    bool used_uniform_draft_fallback() const { return draft_absent_fallback_; }
private:
    Model* draft_;
    Model* target_;
    float gamma_, min_gamma_, max_gamma_;
    Sampler sampler_;
    SamplerConfig sampler_cfg_;
    RNG rng_;
    float acceptance_rate_ = 0.0f;
    float acc_ema_ = 0.6f; // EWMA of acceptance rate for adaptive gamma
    int accepted_count_ = 0;
    int total_count_ = 0;
    // P13 metric flag: draft absent me uniform p_draft fallback liya gaya tha.
    bool draft_absent_fallback_ = false;
    int adapt_interval_ = 10; // re-evaluate gamma every N calls
    int calls_since_adapt_ = 0;
    bool verify_tokens(const std::vector<int>& draft_tokens,
                       const Tensor& target_logits, int vocab_size);
    void adapt_gamma();
};

// D3: Continuous batching — dynamic request scheduling
struct BatchRequest { std::vector<int> tokens; int max_tokens; int id; };
struct BatchResponse { std::string text; int id; };
class ContinuousBatching {
public:
    ContinuousBatching(Model* model, int max_batch = 8);
    void add_request(const BatchRequest& req);
    BatchResponse step();
    bool has_pending() const;
private:
    Model* model_;
    int max_batch_;
    std::queue<BatchRequest> queue_;
    std::vector<BatchRequest> active_;
    std::vector<std::vector<int>> outputs_;
    std::vector<KVCache> kv_caches_;
    Tensor build_attention_mask(int64_t B, int64_t S,
                                const std::vector<int>& seq_lens) const;
};

// D4: KV cache compression — QUANT4/QUANT for K/V storage
class CompressedKVCache {
public:
    CompressedKVCache(int64_t max_seq, int64_t n_layers, int64_t head_dim);
    void append(int layer, const Tensor& k, const Tensor& v);
    Tensor get_k(int layer, int64_t pos) const;
    Tensor get_v(int layer, int64_t pos) const;
    int64_t size() const { return seq_len_; }
    void clear();
private:
    int64_t max_seq_, n_layers_, head_dim_;
    int64_t seq_len_ = 0;
    struct CompressedBlock { std::vector<uint8_t> k_data; std::vector<uint8_t> v_data; };
    std::vector<std::vector<CompressedBlock>> k_blocks_, v_blocks_;
};

// D5: Prefix caching — share KV cache across requests with common prefix
class PrefixCache {
public:
    PrefixCache(int64_t layer_count);
    int64_t match_prefix(const std::vector<int>& tokens);
    void store(const std::vector<int>& tokens, int64_t cache_id,
               int64_t max_seq_len = 2048, int64_t num_heads = 12, int64_t head_dim = 64);
    KVCache* get_cache(int64_t id, int64_t layer);
private:
    int64_t layers_;
    struct PrefixEntry { std::vector<int> prefix; std::unique_ptr<KVCache> cache; };
    std::vector<PrefixEntry> entries_;
};

// D6: Token tree decoding — branch-and-verify parallel decoding
class TreeDecoder {
public:
    TreeDecoder(Model* model, int beam_width = 4);
    std::vector<int> decode(const std::vector<int>& prompt, int max_tokens);
private:
    Model* model_;
    int beam_width_;
    struct Node { int token; float score; Node* parent; std::vector<Node*> children; };
    void expand_node(Node* n, int depth, int max_depth);
    void prune_tree(std::vector<Node*>& candidates);
    void delete_subtree(Node* n);
};

// D7: Flash decoding — parallel KV reduction for multi-turn
Tensor flash_decoding(const Tensor& Q, const Tensor& K, const Tensor& V, int64_t block_size = 64);

// D8: INT8 inference — quantized forward pass
class INT8Inference {
public:
    INT8Inference(Model* model);
    Tensor forward(const Tensor& input, const Tensor& positions);
private:
    Model* model_;
    bool is_quantized_ = false;
};

// D9: FP8 inference — FP8 GEMM forward pass with two-stage residual accumulation (FA-3 style)
class FP8Inference {
public:
    FP8Inference(Model* model);
    Tensor forward(const Tensor& input, const Tensor& positions);
    // Two-stage residual accumulation: dequant + residual tracking for long context
    Tensor forward_two_stage(const Tensor& input, const Tensor& positions);
private:
    Model* model_;
    static constexpr int64_t BLOCK = 64;
    void fp8_residual_accum(const float* src, const uint8_t* fp8_data,
                            const float* scales, int64_t num_blocks,
                            float* out, int64_t n);
};

// D10: Model sharding — split across devices and load only partial model
class ModelShard {
public:
    ModelShard(Model* model, int64_t start_layer, int64_t end_layer);
    Tensor forward(const Tensor& input);
private:
    Model* model_;
    int64_t start_, end_;
};

// D11: Dynamic batching — variable batch sizes
class DynamicBatcher {
public:
    DynamicBatcher(Model* model);
    std::vector<std::string> batch_generate(
        const std::vector<std::string>& prompts, int max_tokens);
private:
    Model* model_;
};

// D12: Request scheduling — priority queue with deadlines
struct Request { int id; std::vector<int> tokens; int priority; int64_t deadline; };
class RequestScheduler {
public:
    void add(const Request& req);
    Request next();
    bool has_next() const;
    int pending_count() const { return (int)queue_.size(); }
private:
    struct Compare { bool operator()(const Request& a, const Request& b); };
    std::priority_queue<Request, std::vector<Request>, Compare> queue_;
};

// D13: Memory management — pool allocator for inference
class InferenceMemoryPool {
public:
    InferenceMemoryPool(size_t block_size, int64_t num_blocks);
    void* alloc();
    void free(void* ptr);
    int64_t used() const { return used_; }
    int64_t capacity() const { return capacity_; }
private:
    size_t block_size_;
    int64_t capacity_, used_ = 0;
    std::vector<char> pool_;
    std::vector<bool> free_list_;
    std::mutex mtx_;
};

// D14-D16: Stream, stop, logprob
struct StreamConfig {
    bool stream = false;
    int max_tokens = 512;
    int eos_id = -1;
    bool stop_on_newline = false;
    std::vector<std::string> stop_sequences;
};

struct DecodeResult {
    std::string text;
    std::vector<int> tokens;
    std::vector<std::vector<float>> logprobs; // token × vocab
    float perplexity = 0;
};

// D17: Logprob output
std::vector<std::vector<float>> compute_logprobs(const Tensor& logits, const std::vector<int>& tokens);

// D18: Embedding endpoint
class EmbeddingEndpoint {
public:
    EmbeddingEndpoint(Model* model);
    Tensor embed(const std::string& text);
    Tensor embed_batch(const std::vector<std::string>& texts);
private:
    Model* model_;
};

// D19: Reranking — score query-document pairs
class Reranker {
public:
    Reranker(Model* model);
    float score(const std::string& query, const std::string& document);
    std::vector<float> score_batch(const std::string& query,
                                    const std::vector<std::string>& documents);
private:
    Model* model_;
};

// ===========================================================================
// E1: Speculative decoding v2 — draft KV cache reuse & tree attention
// ===========================================================================
struct RejectionStats {
    int64_t total_draft_tokens = 0;
    int64_t accepted_draft_tokens = 0;
    int64_t target_calls = 0;
    int64_t tokens_saved = 0;
    double wall_time_sec = 0.0;
    double acceptance_rate() const {
        return total_draft_tokens > 0 ? (double)accepted_draft_tokens / total_draft_tokens : 0.0;
    }
    double tokens_saved_per_call() const {
        return target_calls > 0 ? (double)tokens_saved / target_calls : 0.0;
    }
};

// REMOVED: SpeculativeDecoderV2 had zero implementations, see P13
// (P13: class sirf declare thi — grep "SpeculativeDecoderV2::" se zero defs,
// zero callers mile; link-time ghost thi. Poori declaration ab
// docs/P13_SPECULATIVE_V2_REMOVED.md note me rakhi hai. P6 no-delete rule ke
// tahat ye comment chhoda gaya hai, class wapas nahi laani jab tak .cpp me
// real defs + caller + test na ho.)
// RejectionStats header-only hai isliye yahin rakha hai.

// ===========================================================================
// E2: Multi-Query Attention (MQA) — shared KV across heads
// ===========================================================================
class MultiQueryAttention {
public:
    MultiQueryAttention(int64_t d_model, int64_t n_heads, int64_t d_kv = -1);
    Tensor forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                   const Tensor& mask = Tensor(), KVCache* cache = nullptr,
                   int layer = 0);
    int64_t d_model() const { return d_model_; }
    int64_t n_heads() const { return n_heads_; }

private:
    int64_t d_model_, n_heads_, d_kv_;
    Tensor w_q_, w_k_, w_v_, w_o_;
    float scale_;
};

// ===========================================================================
// E3: Grouped-Query Attention (GQA) — inference-optimized
// ===========================================================================
class GroupedQueryAttention {
public:
    GroupedQueryAttention(int64_t d_model, int64_t n_heads, int64_t n_kv_heads);
    Tensor forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                   const Tensor& mask = Tensor(), KVCache* cache = nullptr,
                   int layer = 0);
    int64_t n_kv_heads() const { return n_kv_heads_; }

private:
    int64_t d_model_, n_heads_, n_kv_heads_, head_dim_;
    int64_t n_groups_; // n_heads / n_kv_heads
    Tensor w_q_, w_k_, w_v_, w_o_;
    float scale_;
    void expand_kv_heads(const Tensor& K, const Tensor& V,
                          Tensor& K_exp, Tensor& V_exp) const;
};

// ===========================================================================
// E4: Sliding window attention — fixed-size local window
// ===========================================================================
class SlidingWindowAttention {
public:
    SlidingWindowAttention(int64_t d_model, int64_t n_heads,
                            int64_t window_size = 512);
    Tensor forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                   const Tensor& mask = Tensor(), KVCache* cache = nullptr,
                   int layer = 0);

private:
    int64_t d_model_, n_heads_, head_dim_, window_size_;
    Tensor w_q_, w_k_, w_v_, w_o_;
    float scale_;
};

// ===========================================================================
// E5: ALiBi positional bias — alternative to RoPE
// ===========================================================================
class ALiBi {
public:
    ALiBi(int64_t n_heads);
    Tensor forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                   const Tensor& mask = Tensor(), KVCache* cache = nullptr,
                   int layer = 0);
    Tensor build_alibi_bias(int64_t S_Q, int64_t S_K) const;

private:
    int64_t n_heads_;
    std::vector<float> slopes_;
    float scale_;
    void compute_slopes();
};

// D20: Grammar decoding — enforce JSON/regex constraint
class GrammarDecoder {
public:
    GrammarDecoder(const std::string& grammar_file, int vocab_size = 32000);
    std::vector<int> constrain(const std::vector<float>& logits,
                                const std::vector<int>& prefix);
private:
    std::vector<std::vector<bool>> allowed_tokens_;
    int vocab_size_ = 32000;
    bool matches_prefix(const std::vector<int>& prefix) const;
    void parse_grammar(const std::string& grammar_file, int vocab_size = 32000);
};

} // namespace quant
