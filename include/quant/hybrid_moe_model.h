#pragma once
#include "quant/model.h"
#include "quant/hybrid_scheduler.h"
#include "quant/moe_variants.h"
#include "quant/transformer.h"
#include "quant/kv_cache.h"

namespace quant {

class HybridMoeBlock {
public:
    RMSNorm attention_norm;
    RMSNorm ffn_norm;
    HybridAttnKind kind = HybridAttnKind::STD;
    Attention attn;
    std::unique_ptr<moe::SparseMoE> moe;
    std::unique_ptr<moe::ExpertFFN> shared_expert;
    moe::MoEAllConfig moe_cfg;

    HybridMoeBlock() = default;
    HybridMoeBlock(const TransformerConfig& cfg, const moe::MoEAllConfig& mcfg,
                   HybridAttnKind k);

    Tensor forward(const Tensor& x, const Tensor& positions,
                   const Tensor& mask, KVCache& cache, int layer_idx) const;

    int64_t param_count() const;
};

class HybridMoeModel : public Model {
public:
    HybridMoeModel() = default;
    HybridMoeModel(const TransformerConfig& cfg, const moe::MoEAllConfig& mcfg,
                   HybridSchedule sched);

    Tensor forward(const Tensor& input_ids, const Tensor& positions,
                   KVCache* cache = nullptr) override;
    // BUGFIX (bug census): load/save were silent vacu-stubs (discarded the
    // path). HybridMoeModel has no .quant loader of its own (block layout
    // differs from DenseModel) — fail loud instead of pretending to save.
    void load(const std::string& p) override {
        throw Error("HybridMoeModel::load not implemented (no .quant mapping for hybrid blocks): " + p);
    }
    void save(const std::string& p) const override {
        throw Error("HybridMoeModel::save not implemented (no .quant mapping for hybrid blocks): " + p);
    }
    int64_t param_count() const override;
    int64_t vocab_size() const override { return config.vocab_size; }
    int64_t stored_param_count() const { return param_count(); }

    HybridSchedule schedule_;
    std::vector<HybridMoeBlock> blocks;
    std::unique_ptr<Embedding> tok_embeddings;
    std::unique_ptr<RMSNorm> norm;
    std::unique_ptr<Linear> lm_head;
    moe::MoEAllConfig moe_config;
};

} // namespace quant
