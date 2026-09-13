# P13 note — SpeculativeDecoderV2 (header se hataya, delete nahi)

## Kya tha
`include/quant/inference_opt.h` me `SpeculativeDecoderV2` class sirf **declare**
thi (E1 section, pehle lines ~262-298). `RejectionStats` struct header-only hai
aur header me hi raha — sirf `SpeculativeDecoderV2` class hatayi gayi hai.

## Kyun hataya (verified)
- `grep "SpeculativeDecoderV2::"` → **zero definitions** (koi `.cpp` me impl nahi).
- Callers ka grep → **zero callers** (sirf header + research docs me zikr).
- `tests/test_inference_opt.cpp` isko touch hi nahi karta.
- Matlab instantiate karne pe link fail hota — ghost declaration thi.

## Hatayi hui declaration (record ke liye, P6 no-delete rule)

```cpp
class SpeculativeDecoderV2 {
public:
    SpeculativeDecoderV2(Model* draft, Model* target, int vocab_size = 32000,
                         float gamma = 5.0f, int n_tree_candidates = 4);

    std::vector<int> generate(const std::vector<int>& prompt, int max_tokens);
    const RejectionStats& stats() const { return stats_; }
    void reset_stats();

    // Enable/disable tree attention verification
    void set_use_tree_attention(bool use) { use_tree_attn_ = use; }
    bool use_tree_attention() const { return use_tree_attn_; }

private:
    Model* draft_;
    Model* target_;
    int vocab_size_;
    float gamma_;
    int n_tree_candidates_;
    bool use_tree_attn_ = true;

    KVCache draft_kv_cache_;
    KVCache target_kv_cache_;
    Sampler sampler_;
    SamplerConfig sampler_cfg_;
    RejectionStats stats_;

    std::vector<int> generate_draft_tokens(int prev_token, int count, int pos);
    int verify_with_tree(const std::vector<int>& draft_tokens,
                          const std::vector<int>& prefix,
                          std::vector<int>& output);
    int verify_linear(const std::vector<int>& draft_tokens,
                       const std::vector<int>& prefix,
                       std::vector<int>& output);
    float get_target_prob(const float* logits, int token);
    int sample_replacement(const float* logits);
};
```

## Wapas laane ki shart
Class tabhi wapas aaye jab ek saath ho:
1. `src/inference_opt.cpp` (ya naye TU) me **saari methods ki real defs**,
2. kam se kam ek **production caller**,
3. `tests/test_inference_opt.cpp` me **real generate/verify test** (sirf vocab
   equality wala sham assert nahi).
