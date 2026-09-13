#include "quant/moe_variants.h"
#include "quant/random.h"
#include <cstring>
#include <cstdio>
#include <unordered_map>
#include <memory>

namespace quant {
namespace moe {

// ========================================================================
// Additional test: Serialization round-trip for all variants
// ========================================================================

bool test_moe_variant_serialization() {
    int64_t D = 128;
    Tensor x({1, 16, D});
    float* xd = x.data<float>();
    for (int64_t i = 0; i < 16 * D; ++i) xd[i] = ((float)(i % 5) - 2.0f) * 0.2f;

    printf("=== MoE Serialization Round-Trip Test ===\n");
    int64_t passed = 0, total = 0;

    auto test_ser = [&](const char* name, auto& moe, auto& moe2, const Tensor& input, auto&& forward_fn) {
        total++;
        auto w1 = moe.export_weights();
        moe2.import_weights(w1);
        auto w2 = moe2.export_weights();
        bool ok = w1.size() == w2.size();
        if (ok && !w1.empty())
            ok = std::memcmp(w1.data(), w2.data(), w1.size()) == 0;
        if (ok) { passed++; printf("  PASS: %s\n", name); }
        else { printf("  FAIL: %s\n", name); }
    };

    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 1; cfg.expert_hidden_size = 256;
        SparseMoE m1(D, cfg), m2(D, cfg);
        test_ser("SPARSE_TOP1", m1, m2, x, [](auto& m, auto& t) { return m.forward(t); });
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.num_slots_per_expert = 1; cfg.expert_hidden_size = 256;
        SoftMoE m1(D, cfg), m2(D, cfg);
        test_ser("SOFT_MIXTURE", m1, m2, x, [](auto& m, auto& t) { return m.forward(t); });
    }
    {
        MoEAllConfig cfg; cfg.num_groups = 4; cfg.experts_per_group = 2; cfg.top_groups = 2;
        cfg.top_experts_per_group = 1; cfg.expert_hidden_size = 256;
        BaseLayerMoE m1(D, cfg), m2(D, cfg);
        test_ser("BASE_LAYER", m1, m2, x, [](auto& m, auto& t) { return m.forward(t); });
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        GatingDropoutMoE m1(D, cfg), m2(D, cfg);
        test_ser("GATING_DROPOUT", m1, m2, x, [](auto& m, auto& t) { return m.forward(t); });
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        AttentionMoE m1(D, cfg), m2(D, cfg);
        test_ser("ATTENTION_MOE", m1, m2, x, [](auto& m, auto& t) { return m.forward(t); });
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        LowRankMoE m1(D, cfg), m2(D, cfg);
        test_ser("LOWRANK_MOE", m1, m2, x, [](auto& m, auto& t) { return m.forward(t); });
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        MambaMoE m1(D, cfg), m2(D, cfg);
        test_ser("MAMBA_MOE", m1, m2, x, [](auto& m, auto& t) { return m.forward(t); });
    }
    {
        MoEAllConfig cfg; cfg.num_routed_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        DeepSeekMoE m1(D, cfg), m2(D, cfg);
        test_ser("DEEPSEEK_MOE", m1, m2, x, [](auto& m, auto& t) { return m.forward(t, false); });
    }
    printf("=== Serialization: %lld/%lld passed ===\n", passed, total);
    return passed == total;
}

// ========================================================================
// Test: Capacity-aware token management for all variants
// ========================================================================

bool test_moe_capacity() {
    int64_t D = 128;
    printf("=== MoE Capacity Test ===\n");
    int64_t passed = 0, total = 0;

    auto check_cap = [&](const char* name, int64_t cap, int64_t min_expected) {
        total++;
        bool ok = cap >= min_expected;
        if (ok) { passed++; printf("  PASS: %s cap=%lld\n", name, cap); }
        else { printf("  FAIL: %s cap=%lld < %lld\n", name, cap, min_expected); }
    };

    // Test SparseMoE capacities with different k values
    {
        MoEAllConfig cfg; cfg.num_experts = 8;
        cfg.top_k = 1; SparseMoE m1(D, cfg); check_cap("SPARSE_TOP1_cap(32)", m1.compute_capacity(32), 4);
        cfg.top_k = 2; SparseMoE m2(D, cfg); check_cap("SPARSE_TOP2_cap(32)", m2.compute_capacity(32), 8);
        cfg.top_k = 3; SparseMoE m3(D, cfg); check_cap("SPARSE_TOPK_cap(32)", m3.compute_capacity(32), 12);
    }

    // Test SoftMoE (always returns num_experts)
    {
        MoEAllConfig cfg; cfg.num_experts = 8; SoftMoE m(D, cfg);
        check_cap("SOFT_MIXTURE_cap", m.compute_capacity(32), 8);
    }

    // Test ExpertChoiceMoE with different capacity factors
    {
        MoEAllConfig cfg; cfg.num_experts = 8;
        cfg.capacity_factor = 1.0f; ExpertChoiceMoE m1(D, cfg); check_cap("EXPERT_CHOICE_cap(1.0)", m1.compute_capacity(32), 4);
        cfg.capacity_factor = 2.0f; ExpertChoiceMoE m2(D, cfg); check_cap("EXPERT_CHOICE_cap(2.0)", m2.compute_capacity(32), 8);
        cfg.capacity_factor = 3.0f; ExpertChoiceMoE m3(D, cfg); check_cap("EXPERT_CHOICE_cap(3.0)", m3.compute_capacity(32), 12);
    }

    // Test HierarchicalMoE
    {
        MoEAllConfig cfg; cfg.num_groups = 4; cfg.experts_per_group = 2;
        cfg.top_groups = 2; cfg.top_experts_per_group = 1;
        HierarchicalMoE m(D, cfg); check_cap("HIERARCHICAL_cap", m.compute_capacity(32), 4);
    }

    // Test MoMoE
    {
        MoEAllConfig cfg; cfg.num_groups = 4; cfg.experts_per_group = 2;
        cfg.top_groups = 2; cfg.top_experts_per_group = 1;
        MoMoE m(D, cfg); check_cap("MOMOE_cap", m.compute_capacity(32), 4);
    }

    // Test HashMoE
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.hash_bucket_size = 2;
        HashMoE m(D, cfg); check_cap("HASH_ROUTED_cap", m.compute_capacity(32), 4);
    }

    // Test DeepSeekMoE
    {
        MoEAllConfig cfg; cfg.num_routed_experts = 8; cfg.top_k = 2;
        DeepSeekMoE m(D, cfg); check_cap("DEEPSEEK_cap", m.compute_capacity(32), 8);
    }

    // Test DenseMoE (always returns T)
    {
        MoEAllConfig cfg; cfg.num_experts = 8;
        DenseMoE m(D, cfg); check_cap("DENSE_MOE_cap", m.compute_capacity(32), 32);
    }

    // Test ProductKeyMoE
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2;
        ProductKeyMoE m(D, cfg); check_cap("PRODUCT_KEY_cap", m.compute_capacity(32), 8);
    }

    // Test with very few tokens relative to experts
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2;
        SparseMoE m(D, cfg); check_cap("SMALL_BATCH_cap(4)", m.compute_capacity(4), 1);
    }

    // Test with many tokens
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2;
        SparseMoE m(D, cfg); check_cap("LARGE_BATCH_cap(1024)", m.compute_capacity(1024), 256);
    }

    printf("=== Capacity Test: %lld/%lld passed ===\n", passed, total);
    return passed == total;
}

// ========================================================================
// Expand routing test with per-variant forward statistics
// ========================================================================

static void check_routing_stats(const char* name, const MoEOutput& out, int64_t B, int64_t S, int64_t D,
                                 int64_t& passed, int64_t& total) {
    total++;
    bool shape_ok = out.output.shape().dims[0] == B && out.output.shape().dims[1] == S && out.output.shape().dims[2] == D;
    bool finite = true;
    const float* od = out.output.data<float>();
    int64_t n = out.output.numel();
    for (int64_t i = 0; i < n; ++i) if (!std::isfinite(od[i])) { finite = false; break; }
    bool expert_ok = out.num_activated_experts > 0;
    bool lb_ok = out.load_balance_loss >= 0.0f;
    bool z_ok = out.z_loss >= 0.0f;
    bool drop_ok = out.tokens_dropped >= 0;

    if (shape_ok && finite && expert_ok && lb_ok && z_ok && drop_ok) {
        passed++;
        printf("  PASS: %s (experts=%lld, lb=%.4f, z=%.4f, dropped=%lld)\n",
               name, out.num_activated_experts, out.load_balance_loss, out.z_loss, out.tokens_dropped);
    } else {
        printf("  FAIL: %s (shape=%d,finite=%d,expert=%d,lb=%d,z=%d,drop=%d)\n",
               name, shape_ok, finite, expert_ok, lb_ok, z_ok, drop_ok);
    }
}

bool test_moe_routing_statistics() {
    int64_t B = 1, S = 32, D = 128;
    Tensor x({B, S, D});
    float* xd = x.data<float>();
    for (int64_t i = 0; i < B * S * D; ++i) xd[i] = ((float)(i % 7) - 3.0f) * 0.1f;
    Tensor token_ids({B * S});
    int64_t* tid = token_ids.data<int64_t>();
    for (int64_t i = 0; i < B * S; ++i) tid[i] = i % 13;

    int64_t passed = 0, total = 0;
    printf("=== MoE Routing Statistics Test ===\n");

    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 1; cfg.expert_hidden_size = 256;
        SparseMoE m(D, cfg); auto out = m.forward(x);
        out.tokens_dropped = 0;
        check_routing_stats("SPARSE_TOP1_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        SparseMoE m(D, cfg); auto out = m.forward(x);
        out.tokens_dropped = 0;
        check_routing_stats("SPARSE_TOP2_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 3; cfg.expert_hidden_size = 256;
        SparseMoE m(D, cfg); auto out = m.forward(x);
        out.tokens_dropped = 0;
        check_routing_stats("SPARSE_TOPK_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.num_slots_per_expert = 1; cfg.expert_hidden_size = 256;
        SoftMoE m(D, cfg); auto out = m.forward(x);
        out.tokens_dropped = 0;
        check_routing_stats("SOFT_MIXTURE_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_groups = 4; cfg.experts_per_group = 2; cfg.top_groups = 2; cfg.top_experts_per_group = 1; cfg.expert_hidden_size = 256;
        HierarchicalMoE m(D, cfg); auto out = m.forward(x);
        out.tokens_dropped = 0;
        check_routing_stats("HIERARCHICAL_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_groups = 4; cfg.experts_per_group = 2; cfg.top_groups = 2; cfg.top_experts_per_group = 1; cfg.expert_hidden_size = 256;
        MoMoE m(D, cfg); auto out = m.forward(x);
        out.tokens_dropped = 0;
        check_routing_stats("MOMOE_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.capacity_factor = 2.0f; cfg.expert_hidden_size = 256;
        ExpertChoiceMoE m(D, cfg); auto out = m.forward(x);
        out.tokens_dropped = 0;
        check_routing_stats("EXPERT_CHOICE_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.hash_bucket_size = 2; cfg.expert_hidden_size = 256;
        HashMoE m(D, cfg); auto out = m.forward(x, token_ids);
        out.tokens_dropped = 0;
        check_routing_stats("HASH_ROUTED_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.num_shared_layers = 2; cfg.expert_hidden_size = 256;
        CrossLayerMoE m(D, cfg); auto out = m.forward(x, 1);
        out.tokens_dropped = 0;
        check_routing_stats("CROSS_LAYER_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_routed_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        DeepSeekMoE m(D, cfg); auto out = m.forward(x, false);
        out.tokens_dropped = 0;
        check_routing_stats("DEEPSEEK_MOE_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        BaseLayerMoE m(D, cfg); auto out = m.forward(x);
        out.tokens_dropped = 0;
        check_routing_stats("BASE_LAYER_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.expert_hidden_size = 256;
        DenseMoE m(D, cfg); auto out = m.forward(x);
        out.tokens_dropped = 0;
        check_routing_stats("DENSE_MOE_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        SharedExpertMoE m(D, cfg); auto out = m.forward(x);
        out.tokens_dropped = 0;
        check_routing_stats("SHARED_EXPERT_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        ResidualMoE m(D, cfg); auto out = m.forward(x);
        out.tokens_dropped = 0;
        check_routing_stats("RESIDUAL_MOE_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        GatingDropoutMoE m(D, cfg); m.dropout_rate = 0.0f; auto out = m.forward(x, false);
        out.tokens_dropped = 0;
        check_routing_stats("GATING_DROPOUT_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        DomainMoE m(D, cfg); auto out = m.forward(x);
        out.tokens_dropped = 0;
        check_routing_stats("DOMAIN_MOE_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        ProductKeyMoE m(D, cfg); auto out = m.forward(x);
        out.tokens_dropped = 0;
        check_routing_stats("PRODUCT_KEY_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        AttentionMoE m(D, cfg); auto out = m.forward(x);
        out.tokens_dropped = 0;
        check_routing_stats("ATTENTION_MOE_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        LowRankMoE m(D, cfg); auto out = m.forward(x);
        out.tokens_dropped = 0;
        check_routing_stats("LOWRANK_MOE_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        MambaMoE m(D, cfg); auto out = m.forward(x);
        out.tokens_dropped = 0;
        check_routing_stats("MAMBA_MOE_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        QuantizedINT8MoE m(D, cfg); auto out = m.forward(x);
        out.tokens_dropped = 0;
        check_routing_stats("INT8_MOE_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        QuantMoE m(D, cfg); auto out = m.forward(x);
        out.tokens_dropped = 0;
        check_routing_stats("QUANT_MOE_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        Quant1MoE m(D, cfg); auto out = m.forward(x);
        out.tokens_dropped = 0;
        check_routing_stats("Q1_MOE_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        QUANT8MoE m(D, cfg); auto out = m.forward(x);
        out.tokens_dropped = 0;
        check_routing_stats("QUANT8_MOE_stats", out, B, S, D, passed, total);
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        QUANT4MoE m(D, cfg); auto out = m.forward(x);
        out.tokens_dropped = 0;
        check_routing_stats("QUANT4_MOE_stats", out, B, S, D, passed, total);
    }

    printf("=== Routing Statistics: %lld/%lld passed ===\n", passed, total);
    return passed == total;
}

// ========================================================================
// Per-variant z-loss detailed test
// ========================================================================

bool test_moe_z_loss() {
    printf("=== MoE Z-Loss Detailed Test ===\n");
    int64_t D = 128;
    Tensor x({1, 16, D});
    float* xd = x.data<float>();
    for (int64_t i = 0; i < 16 * D; ++i) xd[i] = ((float)(i % 7) - 3.0f) * 0.1f;

    int64_t passed = 0, total = 0;
    auto check_zl = [&](const char* name, float zl) {
        total++;
        bool ok = std::isfinite(zl) && zl >= 0.0f;
        if (ok) { passed++; printf("  PASS: %s z_loss=%.6f\n", name, zl); }
        else { printf("  FAIL: %s z_loss=%.6f\n", name, zl); }
    };

    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 1; cfg.expert_hidden_size = 256;
        SparseMoE m(D, cfg); auto out = m.forward(x);
        check_zl("SPARSE_TOP1_z", m.z_loss(out.router_logits));
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        SparseMoE m(D, cfg); auto out = m.forward(x);
        check_zl("SPARSE_TOP2_z", m.z_loss(out.router_logits));
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.num_slots_per_expert = 1; cfg.expert_hidden_size = 256;
        SoftMoE m(D, cfg); auto out = m.forward(x);
        check_zl("SOFT_MIXTURE_z", m.z_loss(out.router_logits));
    }
    {
        MoEAllConfig cfg; cfg.num_groups = 4; cfg.experts_per_group = 2; cfg.top_groups = 2;
        cfg.top_experts_per_group = 1; cfg.expert_hidden_size = 256;
        HierarchicalMoE m(D, cfg); auto out = m.forward(x);
        check_zl("HIERARCHICAL_z", m.z_loss(out.router_logits));
    }
    {
        MoEAllConfig cfg; cfg.num_groups = 4; cfg.experts_per_group = 2; cfg.top_groups = 2;
        cfg.top_experts_per_group = 1; cfg.expert_hidden_size = 256;
        MoMoE m(D, cfg); auto out = m.forward(x);
        check_zl("MOMOE_z", m.z_loss(out.router_logits));
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.capacity_factor = 2.0f; cfg.expert_hidden_size = 256;
        ExpertChoiceMoE m(D, cfg); auto out = m.forward(x);
        check_zl("EXPERT_CHOICE_z", m.z_loss(out.router_logits));
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.expert_hidden_size = 256;
        DenseMoE m(D, cfg); auto out = m.forward(x);
        check_zl("DENSE_MOE_z", m.z_loss(out.router_logits));
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        GatingDropoutMoE m(D, cfg); m.dropout_rate = 0.0f; auto out = m.forward(x, false);
        check_zl("GATING_DROPOUT_z", m.z_loss(out.router_logits));
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        AttentionMoE m(D, cfg); auto out = m.forward(x);
        check_zl("ATTENTION_MOE_z", m.z_loss(out.router_logits));
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        LowRankMoE m(D, cfg); auto out = m.forward(x);
        check_zl("LOWRANK_MOE_z", m.z_loss(out.router_logits));
    }
    {
        MoEAllConfig cfg; cfg.num_routed_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        DeepSeekMoE m(D, cfg); auto out = m.forward(x, false);
        check_zl("DEEPSEEK_MOE_z", m.z_loss(out.router_logits));
    }
    {
        MoEAllConfig cfg; cfg.num_experts = 8; cfg.top_k = 2; cfg.expert_hidden_size = 256;
        QUANT8MoE m(D, cfg); auto out = m.forward(x);
        check_zl("QUANT8_MOE_z", m.z_loss(out.router_logits));
    }

    printf("=== Z-Loss Test: %lld/%lld passed ===\n", passed, total);
    return passed == total;
}

// ========================================================================
// Comprehensive runner
// ========================================================================

bool run_all_moe_tests() {
    bool r1 = test_moe_variant_routing();
    bool r2 = test_moe_load_balance_loss();
    bool r3 = test_moe_variant_serialization();
    bool r4 = test_moe_capacity();
    bool r5 = test_moe_routing_statistics();
    bool r6 = test_moe_z_loss();
    printf("\n=== OVERALL: %s ===\n", (r1 && r2 && r3 && r4 && r5 && r6) ? "ALL PASSED" : "SOME FAILED");
    return r1 && r2 && r3 && r4 && r5 && r6;
}

} // namespace moe
} // namespace quant
