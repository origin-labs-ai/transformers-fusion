#pragma once
#include <vector>
#include <cstdint>
#include "quant/tensor.h"
#include "quant/transformer.h"
#include "quant/math.h"
#include "quant/kv_cache.h"

namespace quant {
namespace moe {

extern int prefetch_threshold;
bool should_prefetch_all_experts(int batch_size);

// ========================================================================
// Enums for MoE variant types and routing strategies
// ========================================================================

enum class MoEVariant {
    SPARSE_TOP1 = 0,
    SPARSE_TOP2,
    SPARSE_TOPK,
    SOFT_MIXTURE,
    HIERARCHICAL,
    MOMOE,              // Mixture of Mixture of Experts
    EXPERT_CHOICE,
    HASH_ROUTED,
    CROSS_LAYER,
    MULTIMODAL,          // Our MoMMoE
    MMOE,                // Multi-gate MoE
    DEEPSEEK_MOE,        // DeepSeek-MoE (shared + routed)
    BASE_LAYER,          // BASE Layer MoE
    DENSE_MOE,           // Dense MoE (all experts active)
    SHARED_EXPERT,       // Shared Expert MoE (standalone)
    RESIDUAL_MOE,        // Residual MoE (overflow via residual)
    GATING_DROPOUT,      // Gating Dropout MoE
    DOMAIN_MOE,          // Domain-specialized MoE
    PRODUCT_KEY,         // Product Key MoE (large vocab via product keys)
    ATTENTION_MOE,       // Attention-based MoE routing
    LOWRANK_MOE,         // Low-rank bottleneck MoE
    MAMBA_MOE,           // Mamba (SSM) + MoE hybrid
    QUANTIZED_INT8_MOE,  // INT8 quantized experts
    QUANT_MOE,         // Q1_5 {-1,0,+1} quantized experts
    Q1_MOE,           // QUANT1 {-1,+1} quantized experts
    QUANT8_MOE,            // QUANT8 codebook quantized experts
    QUANT4_MOE             // QUANT4 codebook quantized experts
};

const char* moe_variant_name(MoEVariant v);

enum class CapacityStrategy {
    FIXED,      // Fixed capacity per expert
    TOKEN_DROP, // Overflow tokens dropped
    TOKEN_PASS  // Overflow passed via residual
};

// ========================================================================
// MoE Config (extended for all variants)
// ========================================================================

struct MoEAllConfig {
    // General
    MoEVariant variant = MoEVariant::SPARSE_TOPK;
    int64_t num_experts = 16;
    int64_t top_k = 2;
    int64_t expert_hidden_size = 2048;
    float load_balance_coef = 0.01f;
    float z_loss_coef = 0.001f;

    // Hierarchical / MoMoE
    int64_t num_groups = 4;
    int64_t experts_per_group = 4;
    int64_t top_groups = 2;
    int64_t top_experts_per_group = 1;
    float group_load_balance_coef = 0.005f;

    // Soft MoE
    int64_t num_slots_per_expert = 1;

    // Expert Choice
    float capacity_factor = 2.0f;

    // Hash MoE
    int64_t hash_bucket_size = 2;

    // Cross-layer
    int64_t num_shared_layers = 4;

    // MMoE
    int64_t num_tasks = 1;

    // DeepSeek-MoE
    int64_t num_shared_experts = 1;
    int64_t num_routed_experts = 8;
    bool use_shared_expert = true;

    // Capacity
    CapacityStrategy capacity_strategy = CapacityStrategy::TOKEN_DROP;

    // Modality (for MULTIMODAL variant)
    int64_t text_experts = 4;
    int64_t vision_experts = 3;
    int64_t image_gen_experts = 2;
    int64_t video_gen_experts = 2;
    int64_t audio_experts = 2;
    int64_t ocr_experts = 1;
    int64_t cross_modal_experts = 2;
};

// ========================================================================
// Router output (unified for all variants)
// ========================================================================

struct MoEOutput {
    Tensor output;           // {B, S, hidden}
    Tensor router_logits;    // {T, E} or {T, G} for hierarchical
    Tensor expert_weights;   // {T, K} or variable
    Tensor expert_indices;   // {T, K} or variable
    float load_balance_loss = 0.0f;
    float z_loss = 0.0f;
    int64_t num_activated_experts = 0;
    int64_t tokens_dropped = 0;
};

// ========================================================================
// Expert FFN (shared by all variants)
// ========================================================================

class ExpertFFN {
public:
    Linear gate_proj, up_proj, down_proj;
    Activation activation;

    ExpertFFN();
    explicit ExpertFFN(int64_t hidden_size, int64_t ffn_hidden, Activation act = Activation::SiLU);

    Tensor forward(const Tensor& x) const;
};

// ========================================================================
// 1. SPARSE MoE (Top-1, Top-2, Top-K Token Choice)
// ========================================================================

class SparseMoE {
public:
    SparseMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x, bool training = true);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);

    std::vector<ExpertFFN> experts;
    Linear router_weight;
    MoEAllConfig config;
    int64_t hidden_size;

private:
    MoEOutput route_topk(const Tensor& x, int K);
};

// ========================================================================
// 2. SOFT MoE (Dense Mixture — all experts, learned mixing weights)
// ========================================================================
// From: "From Sparse to Soft Mixtures of Experts" (Puigcerver et al., ICLR 2024)
// Each expert processes a learned weighted combination of ALL input tokens.
// Output = weighted combination of expert outputs.

class SoftMoE {
public:
    SoftMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);

    std::vector<ExpertFFN> experts;
    Linear input_mixing;
    Linear output_mixing;
    MoEAllConfig config;
    int64_t hidden_size;
};

// ========================================================================
// 3. HIERARCHICAL MoE (Tree Gating — Jordan & Jacobs, 1994)
// ========================================================================
// First-level router selects expert GROUPS, second-level selects experts
// within the selected group. Tree structure: root → interior → leaf.

class HierarchicalMoE {
public:
    HierarchicalMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);

    std::vector<std::vector<ExpertFFN>> expert_groups;
    Linear group_router;
    std::vector<Linear> expert_routers;
    MoEAllConfig config;
    int64_t hidden_size;
};

// ========================================================================
// 4. MoMoE — Mixture of Mixture of Experts (2-level hierarchical)
// ========================================================================
// Level 1: routes tokens to top-K expert groups
// Level 2: within each selected group, routes to top-K experts
// Output = weighted combination of (group_weight × expert_weight)

class MoMoE {
public:
    MoMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);

    std::vector<std::vector<ExpertFFN>> groups;
    Linear primary_router;
    std::vector<Linear> secondary_routers;
    MoEAllConfig config;
    int64_t hidden_size;
};

// ========================================================================
// 5. EXPERT CHOICE MoE (Zhou et al., 2022)
// ========================================================================
// Instead of tokens picking experts, EACH EXPERT picks its top-K tokens.
// Perfect load balance by construction.

class ExpertChoiceMoE {
public:
    ExpertChoiceMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);

    std::vector<ExpertFFN> experts;
    Linear router_weight;
    MoEAllConfig config;
    int64_t hidden_size;
};

// ========================================================================
// 6. HASH MoE (Roller et al., 2021 — deterministic routing)
// ========================================================================
// No learned router. Expert assignment = hash(token_id) mod num_experts.
// Simple, no load balancing loss, deterministic.

class HashMoE {
public:
    HashMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x, const Tensor& token_ids);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);

    std::vector<ExpertFFN> experts;
    MoEAllConfig config;
    int64_t hidden_size;
    int64_t num_buckets;

    static int64_t hash_to_expert(int64_t token_id, int64_t num_experts, int64_t bucket_size);
};

// ========================================================================
// 7. CROSS-LAYER MoE (Shared experts across transformer layers)
// ========================================================================
// Same expert pool shared by multiple transformer layers.
// Each layer can route to different experts from the shared pool.

class CrossLayerMoE {
public:
    CrossLayerMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x, int64_t layer_idx);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);

    std::vector<ExpertFFN> shared_experts;
    std::vector<Linear> layer_routers;
    MoEAllConfig config;
    int64_t hidden_size;
};

// ========================================================================
// 8. MULTIMODAL MoE (MoMMoE — Modality-Aware Sparse MoE)
// ========================================================================
// Extension of SparseMoE with modality-aware routing bias.
// Already partially in engines/trainer/moe/moe.h, here as standalone.

class MultiModalMoE {
public:
    MultiModalMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x, const Tensor& modality_hints);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);

    std::vector<ExpertFFN> experts;
    Linear router_weight;
    Linear modality_classifier;
    MoEAllConfig config;
    int64_t hidden_size;
    int64_t num_modalities = 9;

    std::vector<int64_t> expert_modality_map;
};

// ========================================================================
// 9. MMoE — Multi-gate Mixture of Experts (Ma et al., KDD 2018)
// ========================================================================
// N task-specific gates, shared expert pool.
// Each task gets its own routing weights from the same expert pool.
// From: "Modeling Task Relationships in Multi-task Learning 
//        with Multi-gate Mixture-of-Experts"

class MMoE {
public:
    MMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x, int64_t task_id = 0);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);

    std::vector<ExpertFFN> experts;
    std::vector<Linear> task_gates;
    MoEAllConfig config;
    int64_t hidden_size;
};

// ========================================================================
// 10. DeepSeek-MoE (DeepSeek, 2024)
// ========================================================================
// 1 always-active shared expert + K routed experts per token.
// Bias-based routing for load balancing (no auxiliary loss).
// From: "DeepSeekMoE: Towards Ultimate Expert Specialization"

class DeepSeekMoE {
public:
    DeepSeekMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x, bool training = true);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);

    ExpertFFN shared_expert;
    std::vector<ExpertFFN> routed_experts;
    Linear router_weight;
    MoEAllConfig config;
    int64_t hidden_size;
    std::vector<float> expert_biases;
};

// ========================================================================
// 11. BASE Layer MoE — minimal MoE with single router + experts
// ========================================================================

class BaseLayerMoE {
public:
    BaseLayerMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x, bool training = true);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);
    std::vector<ExpertFFN> experts;
    Linear router;
    MoEAllConfig config;
    int64_t hidden_size;
};

// ========================================================================
// 12. Dense MoE — all experts active, weighted sum
// ========================================================================

class DenseMoE {
public:
    DenseMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);
    std::vector<ExpertFFN> experts;
    Linear gate;
    MoEAllConfig config;
    int64_t hidden_size;
};

// ========================================================================
// 13. Shared Expert MoE — standalone shared expert + routed experts
// ========================================================================

class SharedExpertMoE {
public:
    SharedExpertMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x, bool training = true);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);
    ExpertFFN shared_expert;
    std::vector<ExpertFFN> routed_experts;
    Linear router;
    MoEAllConfig config;
    int64_t hidden_size;
};

// ========================================================================
// 14. Residual MoE — overflow tokens pass through residual
// ========================================================================

class ResidualMoE {
public:
    ResidualMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);
    std::vector<ExpertFFN> experts;
    Linear router;
    MoEAllConfig config;
    int64_t hidden_size;
};

// ========================================================================
// 15. Gating Dropout MoE — dropout on gating weights during training
// ========================================================================

class GatingDropoutMoE {
public:
    GatingDropoutMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x, bool training = true);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);
    std::vector<ExpertFFN> experts;
    Linear router;
    MoEAllConfig config;
    int64_t hidden_size;
    float dropout_rate = 0.1f;
};

// ========================================================================
// 16. Domain MoE — domain-specialized expert groups
// ========================================================================

class DomainMoE {
public:
    DomainMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x, int64_t domain_id = -1);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);
    std::vector<std::vector<ExpertFFN>> domain_experts;
    std::vector<Linear> domain_routers;
    Linear domain_classifier;
    MoEAllConfig config;
    int64_t hidden_size;
    int64_t num_domains = 4;
};

// ========================================================================
// 17. Product Key MoE — large effective expert count via product keys
// ========================================================================

class ProductKeyMoE {
public:
    ProductKeyMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);
    std::vector<ExpertFFN> experts_a;
    std::vector<ExpertFFN> experts_b;
    Linear key_router_a;
    Linear key_router_b;
    MoEAllConfig config;
    int64_t hidden_size;
};

// ========================================================================
// 18. Attention MoE — attention-based routing
// ========================================================================

class AttentionMoE {
public:
    AttentionMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);
    std::vector<ExpertFFN> experts;
    Linear q_proj;
    Linear k_proj;
    MoEAllConfig config;
    int64_t hidden_size;
};

// ========================================================================
// 19. Low-rank MoE — low-rank bottleneck MoE
// ========================================================================

class LowRankMoE {
public:
    LowRankMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);
    std::vector<ExpertFFN> experts;
    Linear down_proj;
    Linear up_proj;
    Linear router;
    MoEAllConfig config;
    int64_t hidden_size;
    int64_t latent_dim = 256;
};

// ========================================================================
// 20. Mamba MoE — Mamba SSM + MoE hybrid
// ========================================================================

class MambaMoE {
public:
    MambaMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);
    std::vector<ExpertFFN> experts;
    Linear ssm_proj;
    Linear router;
    MoEAllConfig config;
    int64_t hidden_size;
    int64_t state_dim = 64;
};

// ========================================================================
// 21. Quantized INT8 MoE — INT8 per-expert weight quantization
// ========================================================================

class QuantizedINT8MoE {
public:
    QuantizedINT8MoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);
    std::vector<ExpertFFN> experts;
    std::vector<float> expert_scales;
    Linear router;
    MoEAllConfig config;
    int64_t hidden_size;
};

// ========================================================================
// 22. Quant MoE — QUANT quantized expert weights
// ========================================================================

class QuantMoE {
public:
    QuantMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);
    std::vector<ExpertFFN> experts;
    std::vector<float> quant_scales;
    Linear router;
    MoEAllConfig config;
    int64_t hidden_size;
};

// ========================================================================
// 23. QUANT1 MoE — QUANT1 quantized expert weights
// ========================================================================

class Quant1MoE {
public:
    Quant1MoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);
    std::vector<ExpertFFN> experts;
    std::vector<float> quant1_scales;
    Linear router;
    MoEAllConfig config;
    int64_t hidden_size;
};

// ========================================================================
// 24. QUANT8 MoE — QUANT8 codebook quantized expert weights
// ========================================================================

class QUANT8MoE {
public:
    QUANT8MoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);
    std::vector<ExpertFFN> experts;
    std::vector<std::vector<float>> codebooks;
    Linear router;
    MoEAllConfig config;
    int64_t hidden_size;
};

// ========================================================================
// 25. QUANT4 MoE — QUANT4 codebook quantized expert weights
// ========================================================================

class QUANT4MoE {
public:
    QUANT4MoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x);
    float load_balance_loss(const Tensor& gates) const;
    float z_loss(const Tensor& logits) const;
    int64_t compute_capacity(int64_t T) const;
    std::vector<uint8_t> export_weights() const;
    void import_weights(const std::vector<uint8_t>& data);
    std::vector<ExpertFFN> experts;
    std::vector<std::vector<float>> codebooks;
    Linear router;
    MoEAllConfig config;
    int64_t hidden_size;
};

// ========================================================================
// MoE Factory — maps variant name string to creator
// ========================================================================

std::unique_ptr<void, void(*)(void*)> create_moe_variant(MoEVariant variant, int64_t hidden, const MoEAllConfig& cfg);
int64_t moe_variant_count();
const char* moe_variant_name_by_index(int64_t index);
bool test_moe_variant_routing();
bool test_moe_load_balance_loss();
bool test_moe_variant_serialization();
bool test_moe_capacity();
bool test_moe_routing_statistics();
bool test_moe_z_loss();
bool run_all_moe_tests();

float compute_load_balance_loss(const Tensor& router_logits, const Tensor& expert_indices, int64_t num_experts);
float compute_z_loss(const Tensor& expert_output_norms);
Tensor softmax_with_topk(const Tensor& logits, int64_t k, Tensor& indices_out, Tensor& weights_out);
int64_t hash_token(int64_t token_id, int64_t range);

// ========================================================================
// Batched dispatch: groups tokens by expert, runs one forward per expert
// Replaces token-by-token dispatch in MMoE/DeepSeekMoE/CrossLayer
// ========================================================================

Tensor moe_dispatch_batched(const Tensor& x_flat,
                            const std::vector<ExpertFFN>& experts,
                            const int64_t* indices, const float* weights,
                            int64_t T, int64_t K, int64_t E, int64_t D,
                            float* z_loss_out = nullptr,
                            int64_t* dropped_out = nullptr);

// ========================================================================
// AVX2-optimized MoE kernels
// ========================================================================

namespace avx2 {
    void moe_combine(float* output, const float* expert_outputs,
                     const float* weights, const int64_t* indices,
                     int64_t T, int64_t K, int64_t D);
    void moe_softmax_topk(float* probs, int64_t* indices, float* weights,
                          const float* logits, int64_t T, int64_t E, int64_t K);
    void moe_load_balance(float* f_i, float* P_i,
                          const int64_t* indices, const float* weights,
                          int64_t T, int64_t K, int64_t E);
}

// Expert FFN creation helper (non-static — shared across translation units)
std::vector<ExpertFFN> create_experts(int64_t count, int64_t hidden, int64_t ffn_hidden, Activation act);

} // namespace moe
} // namespace quant
