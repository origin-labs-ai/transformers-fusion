#pragma once
#include "quant/model.h"
#include "quant/transformer.h"
#include "quant/moe_variants.h"
#include "quant/distributed.h"
#include "quant/expert_prefetch.h"
#include <vector>
#include <memory>

namespace quant {

class MoEModel;

// ── MoEBlock — Transformer block with MoE FFN ──
class MoEBlock {
public:
    RMSNorm attention_norm;
    Attention attention;
    RMSNorm ffn_norm;

    std::unique_ptr<moe::SparseMoE> moe;
    std::unique_ptr<moe::ExpertFFN> shared_expert;

    float load_balance_loss = 0.0f;
    float z_loss = 0.0f;
    // L057: router overflow accounting — tokens dropped by capacity limits in
    // the last forward (from moe::MoEOutput::tokens_dropped).
    int64_t tokens_dropped = 0;

    moe::MoEAllConfig moe_config;
    Tensor last_expert_indices;
    Tensor last_router_logits;

    MoEBlock() = default;
    MoEBlock(const TransformerConfig& cfg, const moe::MoEAllConfig& moe_cfg);

    Tensor forward(const Tensor& x, const Tensor& positions,
                   const Tensor& mask, KVCache& cache, int layer_idx,
                   bool training = true);

    int64_t param_count() const;
    int64_t activated_param_count() const;
    int64_t num_stored_experts() const;
};

// ── MoEModel — Full MoE model with 48T+ scaling support ──
class MoEModel : public Model {
public:
    MoEModel() = default;
    MoEModel(const TransformerConfig& cfg, const moe::MoEAllConfig& moe_cfg);

    Tensor forward(const Tensor& input_ids, const Tensor& positions,
                   KVCache* cache = nullptr) override;
    void load(const std::string& quant_path) override;
    void save(const std::string& quant_path) const override;
    int64_t param_count() const override;
    int64_t vocab_size() const override;
    int64_t stored_param_count() const;

    std::unique_ptr<Embedding> tok_embeddings;
    std::vector<MoEBlock> layers;
    std::unique_ptr<RMSNorm> norm;
    std::unique_ptr<Linear> lm_head;
    
    std::unique_ptr<class ExpertPrefetcher> prefetcher;

    // Create and initialize the expert prefetcher. The prefetcher is wired to the
    // model's in-memory expert weights so prefetch loads real data, not zeros.
    void init_prefetcher(int prefetch_ahead = 1, int max_resident = 8);

    // Register (or re-register) the model's current expert weight pointers with
    // the already-initialized prefetcher. Called after load() and after any
    // weight update.
    void wire_prefetcher_sources();

    moe::MoEAllConfig moe_config;

    float total_load_balance_loss = 0.0f;
    float total_z_loss = 0.0f;

    // Scaling config presets
    static TransformerConfig config_48T();
    static TransformerConfig config_1T();
    static TransformerConfig config_100B();

    static moe::MoEAllConfig moe_config_48T();
    static moe::MoEAllConfig moe_config_1T();
    static moe::MoEAllConfig moe_config_100B();

private:
    void build_layers();
};

// ── ExpertParallel — Distribute experts across devices ──
class ExpertParallel {
public:
    ExpertParallel(int num_experts, int num_ranks, int rank,
                   DistributedContext* ctx = nullptr);
    ~ExpertParallel();

    std::vector<int> local_experts() const;
    void alltoall_experts(const Tensor& input, Tensor& output) const;
    Tensor forward_with_parallel(MoEModel* model, const Tensor& input,
                                  const Tensor& positions, KVCache* cache) const;
    void sync_gradients(MoEModel* model) const;

    int num_experts() const { return num_experts_; }
    int num_ranks() const { return num_ranks_; }
    int rank() const { return rank_; }
    int num_local_experts() const { return (int)local_experts_.size(); }

private:
    int num_experts_;
    int num_ranks_;
    int rank_;
    std::vector<int> local_experts_;
    DistributedContext* ctx_;
    bool owns_ctx_;
    void build_expert_distribution();
};

} // namespace quant
