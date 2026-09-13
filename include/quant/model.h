#pragma once
#include "quant/types.h"
#include "quant/tensor.h"
#include "quant/transformer.h"
#include "quant/kv_cache.h"
#include "quant/quant_format.h"
#include "quant/codebook.h"
#include <vector>
#include <memory>
#include <string>

namespace quant {

class Model {
public:
    virtual ~Model() = default;
    
    virtual Tensor forward(const Tensor& input_ids, const Tensor& positions,
                           KVCache* cache = nullptr) = 0;
    virtual void load(const std::string& quant_path);
    virtual void save(const std::string& quant_path) const;
    virtual int64_t param_count() const = 0;
    virtual int64_t vocab_size() const = 0;
    
    TransformerConfig config;
};

class DenseModel : public Model {
public:
    DenseModel() = default;
    explicit DenseModel(const TransformerConfig& cfg);
    
    Tensor forward(const Tensor& input_ids, const Tensor& positions,
                   KVCache* cache = nullptr) override;
    void load(const std::string& quant_path) override;
    void save(const std::string& quant_path) const override;
    void save_quantized(const std::string& quant_path, Format fmt) const;
    int64_t param_count() const override;
    int64_t vocab_size() const override;

    void init_weights();
    // Seeded variant (BUGFIX bug census round-14): init_weights() used
    // std::random_device, so every call produced different weights — the
    // convergence test compared FP32 vs QUANT8 from DIFFERENT inits and went
    // flaky (delta 0.19 one run, 0.42 the next). seed=0 keeps legacy entropy;
    // tests pass an explicit seed for identical inits.
    void init_weights(uint64_t seed);
    void get_parameters(std::vector<Tensor*>& params);

    // MTP: forward returning per-head logits (head 0 = next-token, head 1+ = future tokens)
    std::vector<Tensor> mtp_forward(const Tensor& input_ids, const Tensor& positions,
                                    KVCache* cache = nullptr);
    float mtp_loss(const std::vector<Tensor>& mtp_logits, const Tensor& labels) const;
    
    std::unique_ptr<Embedding> tok_embeddings;
    std::vector<std::unique_ptr<TransformerBlock>> layers;
    std::unique_ptr<RMSNorm> norm;
    std::unique_ptr<Linear> lm_head;

    // MTP prediction heads (one per future token position)
    std::vector<std::unique_ptr<Linear>> mtp_heads;
    
private:
    void build_layers();
};

} // namespace quant
