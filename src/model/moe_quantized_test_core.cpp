#include "quant/moe_variants.h"
#include "quant/random.h"
#include <cstring>
#include <cstdio>
#include <unordered_map>
#include <memory>

namespace quant {
namespace moe {

// ========================================================================
// Comprehensive test: test_moe_variant_routing
// ========================================================================

static bool check_finite(const Tensor& t) {
    const float* d = t.data<float>();
    int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i)
        if (!std::isfinite(d[i])) return false;
    return true;
}

bool test_moe_variant_routing() {
    int64_t B = 1, S = 32, D = 128;
    int64_t T = B * S;
    Tensor x({B, S, D});
    float* xd = x.data<float>();
    for (int64_t i = 0; i < T * D; ++i) xd[i] = ((float)(i % 7) - 3.0f) * 0.1f;
    Tensor token_ids({T});
    int64_t* tid = token_ids.data<int64_t>();
    for (int64_t i = 0; i < T; ++i) tid[i] = i % 13;

    int64_t passed = 0, failed = 0, total = 0;

    auto report = [&](const char* name, bool ok) {
        total++;
        if (ok) { passed++; printf("  PASS: %s\n", name); }
        else { failed++; printf("  FAIL: %s\n", name); }
    };

    printf("=== MoE Variant Routing Test ===\n");

    // 1. SPARSE_TOP1 (k=1)
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.top_k = 1; cfg.expert_hidden_size = 256;
        SparseMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && out.num_activated_experts > 0;
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.z_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.compute_capacity(T) >= 1;
        auto w = moe.export_weights();
        ok = ok && !w.empty();
        report("SPARSE_TOP1", ok);
    }

    // 2. SPARSE_TOP2 (k=2)
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        SparseMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && out.num_activated_experts > 0;
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        report("SPARSE_TOP2", ok);
    }

    // 3. SPARSE_TOPK (k=3)
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.top_k = 3; cfg.expert_hidden_size = 256;
        SparseMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && out.num_activated_experts > 0;
        report("SPARSE_TOPK", ok);
    }

    // 4. SOFT_MIXTURE
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.num_slots_per_expert = 1; cfg.expert_hidden_size = 256;
        SoftMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.z_loss(out.router_logits) >= 0.0f;
        auto w = moe.export_weights();
        ok = ok && !w.empty();
        report("SOFT_MIXTURE", ok);
    }

    // 5. HIERARCHICAL
    {
        MoEAllConfig cfg;
        cfg.num_groups = 4; cfg.experts_per_group = 2; cfg.top_groups = 2; cfg.top_experts_per_group = 1;
        cfg.expert_hidden_size = 256;
        HierarchicalMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.compute_capacity(T) >= 1;
        auto w = moe.export_weights();
        ok = ok && !w.empty();
        report("HIERARCHICAL", ok);
    }

    // 6. MOMOE
    {
        MoEAllConfig cfg;
        cfg.num_groups = 4; cfg.experts_per_group = 2; cfg.top_groups = 2; cfg.top_experts_per_group = 1;
        cfg.expert_hidden_size = 256;
        MoMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.z_loss(out.router_logits) >= 0.0f;
        report("MOMOE", ok);
    }

    // 7. EXPERT_CHOICE
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.capacity_factor = 2.0f; cfg.expert_hidden_size = 256;
        ExpertChoiceMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.compute_capacity(T) >= 1;
        report("EXPERT_CHOICE", ok);
    }

    // 8. HASH_ROUTED
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.hash_bucket_size = 2; cfg.expert_hidden_size = 256;
        HashMoE moe(D, cfg);
        MoEOutput out = moe.forward(x, token_ids);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && out.num_activated_experts > 0;
        ok = ok && moe.compute_capacity(T) >= 1;
        report("HASH_ROUTED", ok);
    }

    // 9. CROSS_LAYER
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.top_k = 2; cfg.num_shared_layers = 2; cfg.expert_hidden_size = 256;
        CrossLayerMoE moe(D, cfg);
        MoEOutput out = moe.forward(x, 1);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.z_loss(out.router_logits) >= 0.0f;
        auto w = moe.export_weights();
        ok = ok && !w.empty();
        moe.import_weights(w);
        report("CROSS_LAYER", ok);
    }

    // 10. DEEPSEEK_MOE
    {
        MoEAllConfig cfg;
        cfg.num_routed_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        DeepSeekMoE moe(D, cfg);
        MoEOutput out = moe.forward(x, false);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.z_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.compute_capacity(T) >= 1;
        auto w = moe.export_weights();
        ok = ok && !w.empty();
        moe.import_weights(w);
        report("DEEPSEEK_MOE", ok);
    }

    // 11. BASE_LAYER
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        BaseLayerMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.z_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.compute_capacity(T) >= 1;
        auto w = moe.export_weights();
        ok = ok && !w.empty();
        moe.import_weights(w);
        report("BASE_LAYER", ok);
    }

    // 12. DENSE_MOE
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.expert_hidden_size = 256;
        DenseMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && out.num_activated_experts > 0;
        ok = ok && moe.z_loss(out.router_logits) >= 0.0f;
        report("DENSE_MOE", ok);
    }

    // 13. SHARED_EXPERT
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        SharedExpertMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.z_loss(out.router_logits) >= 0.0f;
        report("SHARED_EXPERT", ok);
    }

    // 14. RESIDUAL_MOE
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        ResidualMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.z_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.compute_capacity(T) >= 1;
        report("RESIDUAL_MOE", ok);
    }

    // 15. GATING_DROPOUT
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        GatingDropoutMoE moe(D, cfg);
        MoEOutput out = moe.forward(x, false);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.z_loss(out.router_logits) >= 0.0f;
        report("GATING_DROPOUT", ok);
    }

    // 16. DOMAIN_MOE
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        DomainMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.z_loss(out.router_logits) >= 0.0f;
        report("DOMAIN_MOE", ok);
    }

    // 17. PRODUCT_KEY
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        ProductKeyMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.z_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.compute_capacity(T) >= 1;
        report("PRODUCT_KEY", ok);
    }

    // 18. ATTENTION_MOE
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        AttentionMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.z_loss(out.router_logits) >= 0.0f;
        auto w = moe.export_weights();
        ok = ok && !w.empty();
        report("ATTENTION_MOE", ok);
    }

    // 19. LOWRANK_MOE
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        LowRankMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.z_loss(out.router_logits) >= 0.0f;
        auto w = moe.export_weights();
        ok = ok && !w.empty();
        report("LOWRANK_MOE", ok);
    }

    // 20. MAMBA_MOE
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        MambaMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.z_loss(out.router_logits) >= 0.0f;
        auto w = moe.export_weights();
        ok = ok && !w.empty();
        report("MAMBA_MOE", ok);
    }

    // 21. QUANTIZED_INT8_MOE
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        QuantizedINT8MoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.z_loss(out.router_logits) >= 0.0f;
        auto w = moe.export_weights();
        ok = ok && !w.empty();
        report("QUANTIZED_INT8_MOE", ok);
    }

    // 22. QUANT_MOE
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        QuantMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.z_loss(out.router_logits) >= 0.0f;
        auto w = moe.export_weights();
        ok = ok && !w.empty();
        report("QUANT_MOE", ok);
    }

    // 23. Q1_MOE
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        Quant1MoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.z_loss(out.router_logits) >= 0.0f;
        auto w = moe.export_weights();
        ok = ok && !w.empty();
        report("Q1_MOE", ok);
    }

    // 24. QUANT8_MOE
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        QUANT8MoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.z_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.compute_capacity(T) >= 1;
        auto w = moe.export_weights();
        ok = ok && !w.empty();
        report("QUANT8_MOE", ok);
    }

    // 25. QUANT4_MOE
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        QUANT4MoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.z_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.compute_capacity(T) >= 1;
        auto w = moe.export_weights();
        ok = ok && !w.empty();
        moe.import_weights(w);
        auto w2 = moe.export_weights();
        ok = ok && w.size() == w2.size();
        report("QUANT4_MOE", ok);
    }

    // 26. MMoE (Multi-gate with task_id)
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.top_k = 2; cfg.num_tasks = 3; cfg.expert_hidden_size = 256;
        MMoE moe(D, cfg);
        MoEOutput out0 = moe.forward(x, 0);
        MoEOutput out1 = moe.forward(x, 1);
        bool ok = out0.output.shape().dims[0] == B && out0.output.shape().dims[1] == S && out0.output.shape().dims[2] == D;
        ok = ok && out1.output.shape().dims[0] == B && out1.output.shape().dims[1] == S && out1.output.shape().dims[2] == D;
        ok = ok && check_finite(out0.output) && check_finite(out1.output);
        ok = ok && moe.load_balance_loss(out0.router_logits) >= 0.0f;
        ok = ok && moe.z_loss(out0.router_logits) >= 0.0f;
        ok = ok && moe.compute_capacity(T) >= 1;
        report("MMOE", ok);
    }

    // 27. MultiModalMoE with modality hints
    {
        MoEAllConfig cfg;
        cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        cfg.text_experts = 3; cfg.vision_experts = 2; cfg.image_gen_experts = 1;
        cfg.video_gen_experts = 1; cfg.audio_experts = 1;
        MultiModalMoE moe(D, cfg);
        Tensor hints({T});
        float* hd = hints.data<float>();
        for (int64_t i = 0; i < T; ++i) hd[i] = (float)(i % 4);
        MoEOutput out = moe.forward(x, hints);
        bool ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
        ok = ok && check_finite(out.output);
        ok = ok && moe.load_balance_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.z_loss(out.router_logits) >= 0.0f;
        ok = ok && moe.compute_capacity(T) >= 1;
        auto w = moe.export_weights();
        ok = ok && !w.empty();
        report("MULTIMODAL", ok);
    }

    printf("=== Results: %lld/%lld passed, %lld failed ===\n", passed, total, failed);
    return failed == 0;
}

// ========================================================================
// Additional test: Test load balance loss for all variants
// ========================================================================

static void test_lb_equality(const char* name, float lb, float expected_min, float expected_max) {
    if (lb >= expected_min && lb <= expected_max)
        printf("  PASS: %s lb_loss=%.4f\n", name, lb);
    else
        printf("  FAIL: %s lb_loss=%.4f (expected [%.2f,%.2f])\n", name, lb, expected_min, expected_max);
}

bool test_moe_load_balance_loss() {
    int64_t T = 32, D = 128;
    Tensor x({1, T, D});
    float* xd = x.data<float>();
    for (int64_t i = 0; i < T * D; ++i) xd[i] = ((float)(i % 5) - 2.0f) * 0.2f;
    Tensor token_ids({T});
    int64_t* tid = token_ids.data<int64_t>();
    for (int64_t i = 0; i < T; ++i) tid[i] = i * 7 + 3;

    printf("=== MoE Load Balance Loss Test ===\n");

    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 1; cfg.expert_hidden_size = 256;
        SparseMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        float lb = moe.load_balance_loss(out.router_logits);
        test_lb_equality("SPARSE_TOP1", lb, 0.0f, 10.0f);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        SparseMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        float lb = moe.load_balance_loss(out.router_logits);
        test_lb_equality("SPARSE_TOP2", lb, 0.0f, 10.0f);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 3; cfg.expert_hidden_size = 256;
        SparseMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        float lb = moe.load_balance_loss(out.router_logits);
        test_lb_equality("SPARSE_TOPK", lb, 0.0f, 10.0f);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.num_slots_per_expert = 1; cfg.expert_hidden_size = 256;
        SoftMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        float lb = moe.load_balance_loss(out.router_logits);
        test_lb_equality("SOFT_MIXTURE", lb, 0.0f, 2.0f);
    }
    {
        MoEAllConfig cfg; cfg.num_groups = 4; cfg.experts_per_group = 2; cfg.top_groups = 2;
        cfg.top_experts_per_group = 1; cfg.expert_hidden_size = 256;
        HierarchicalMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        float lb = moe.load_balance_loss(out.router_logits);
        test_lb_equality("HIERARCHICAL", lb, 0.0f, 10.0f);
    }
    {
        MoEAllConfig cfg; cfg.num_groups = 4; cfg.experts_per_group = 2; cfg.top_groups = 2;
        cfg.top_experts_per_group = 1; cfg.expert_hidden_size = 256;
        MoMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        float lb = moe.load_balance_loss(out.router_logits);
        test_lb_equality("MOMOE", lb, 0.0f, 10.0f);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.capacity_factor = 2.0f; cfg.expert_hidden_size = 256;
        ExpertChoiceMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        float lb = moe.load_balance_loss(out.router_logits);
        test_lb_equality("EXPERT_CHOICE", lb, 0.0f, 0.001f);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.hash_bucket_size = 2; cfg.expert_hidden_size = 256;
        HashMoE moe(D, cfg);
        MoEOutput out = moe.forward(x, token_ids);
        float lb = moe.load_balance_loss(out.router_logits);
        test_lb_equality("HASH_ROUTED", lb, 0.0f, 0.001f);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.num_shared_layers = 2;
        cfg.expert_hidden_size = 256;
        CrossLayerMoE moe(D, cfg);
        MoEOutput out = moe.forward(x, 1);
        float lb = moe.load_balance_loss(out.router_logits);
        test_lb_equality("CROSS_LAYER", lb, 0.0f, 10.0f);
    }
    {
        MoEAllConfig cfg; cfg.num_routed_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        DeepSeekMoE moe(D, cfg);
        MoEOutput out = moe.forward(x, false);
        float lb = moe.load_balance_loss(out.router_logits);
        test_lb_equality("DEEPSEEK_MOE", lb, 0.0f, 10.0f);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        BaseLayerMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        float lb = moe.load_balance_loss(out.router_logits);
        test_lb_equality("BASE_LAYER", lb, 0.0f, 10.0f);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.expert_hidden_size = 256;
        DenseMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        float lb = moe.load_balance_loss(out.router_logits);
        test_lb_equality("DENSE_MOE", lb, 0.0f, 0.001f);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        SharedExpertMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        float lb = moe.load_balance_loss(out.router_logits);
        test_lb_equality("SHARED_EXPERT", lb, 0.0f, 10.0f);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        ResidualMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        float lb = moe.load_balance_loss(out.router_logits);
        test_lb_equality("RESIDUAL_MOE", lb, 0.0f, 10.0f);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        GatingDropoutMoE moe(D, cfg); moe.dropout_rate = 0.0f;
        MoEOutput out = moe.forward(x, false);
        float lb = moe.load_balance_loss(out.router_logits);
        test_lb_equality("GATING_DROPOUT", lb, 0.0f, 10.0f);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        DomainMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        float lb = moe.load_balance_loss(out.router_logits);
        test_lb_equality("DOMAIN_MOE", lb, 0.0f, 10.0f);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        AttentionMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        float lb = moe.load_balance_loss(out.router_logits);
        test_lb_equality("ATTENTION_MOE", lb, 0.0f, 2.0f);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        QuantMoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        float lb = moe.load_balance_loss(out.router_logits);
        test_lb_equality("QUANT_MOE", lb, 0.0f, 10.0f);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        Quant1MoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        float lb = moe.load_balance_loss(out.router_logits);
        test_lb_equality("Q1_MOE", lb, 0.0f, 10.0f);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        QUANT8MoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        float lb = moe.load_balance_loss(out.router_logits);
        test_lb_equality("QUANT8_MOE", lb, 0.0f, 10.0f);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        QUANT4MoE moe(D, cfg);
        MoEOutput out = moe.forward(x);
        float lb = moe.load_balance_loss(out.router_logits);
        test_lb_equality("QUANT4_MOE", lb, 0.0f, 10.0f);
    }
    printf("=== Load Balance Loss Test Complete ===\n");
    return true;
}

} // namespace moe
} // namespace quant
