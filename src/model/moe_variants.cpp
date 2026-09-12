#include "quant/moe_variants.h"
#include "quant/random.h"
#include "quant/autograd.h"
#include "quant/autograd_functions.h"
#include "quant/simd_math.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_map>
#include <memory>
#include <atomic>

namespace quant {
namespace moe {

namespace {
// ============================================================================
// Phase 8-G6 W1 helpers (file-local; no header change).
//
// Design rules applied to every variant below:
//  - All expert combines go through moe_dispatch_batched (its enabled-path
//    already routes expert_out through AutogradEngine::mul_op/add_op with a
//    constant routing mask) plus explicit AutogradEngine::add_op/mul_op with
//    constant masks for residual/shared/domain/product/quant combines.
//    No manual `od[]` accumulation loops, no memcpy batching, no const_cast.
//  - Router/bias/hint/dropout adjustments use bias_add_op/add_op/mul_op with
//    constant tensors. Graph tensors (Linear outputs) are only ever read via
//    const access for host-side routing decisions, never mutated raw, so the
//    routing weights stay graph tensors.
//  - No math::softmax/silu/mul calls in this file. Full-row distributions
//    needed as dispatch weights are obtained via softmax_with_topk(k=cols)
//    scattered back to natural order (read-only on logits).
//  - Aux scalars are reported as floats (MoEOutput is header-fixed) but the
//    coefficient scaling is performed through AutogradEngine::mul_op so the
//    coef lives in-graph when autograd is enabled. Values are bit-identical
//    in both modes because the ops fall back to plain math when disabled.
// ============================================================================

// Scale a host scalar by coef/denominator through a graph mul_op.
inline float w1_scaled_loss(float raw, float coef, float inv_n) {
    Tensor r({1});
    r.data<float>()[0] = raw;
    Tensor c({1});
    c.data<float>()[0] = coef * inv_n;
    Tensor s = AutogradEngine::mul_op(r, c);
    return s.data<float>()[0];
}

inline float w1_lb_loss(const Tensor& logits, const Tensor& indices,
                        int64_t E, float coef) {
    float raw = compute_load_balance_loss(logits, indices, E);
    return w1_scaled_loss(raw, coef, 1.0f);
}

inline float w1_z_loss(float zl, float coef, float inv_t) {
    return w1_scaled_loss(zl, coef, inv_t);
}

// Full-row softmax distribution without math::softmax: top-k with k=cols
// returns the whole distribution (sorted); scatter back to column order.
// Read-only on logits; result is a host constant used as dispatch weights.
inline Tensor w1_full_probs(const Tensor& logits) {
    int64_t T = logits.dim(0), C = logits.dim(1);
    Tensor idx, w;
    softmax_with_topk(logits, C, idx, w);
    Tensor probs({T, C});
    probs.zero_();
    float* p = probs.data<float>();
    const int64_t* ii = idx.data<int64_t>();
    const float* ww = w.data<float>();
    for (int64_t t = 0; t < T; ++t)
        for (int64_t k = 0; k < C; ++k)
            p[t * C + ii[t * C + k]] = ww[t * C + k];
    return probs;
}

// Broadcast a per-token column into a {T,D} constant mask.
inline Tensor w1_broadcast_col(const float* col, int64_t T, int64_t D) {
    Tensor m({T, D});
    float* md = m.data<float>();
    for (int64_t t = 0; t < T; ++t) {
        float v = col[t];
        for (int64_t d = 0; d < D; ++d)
            md[t * D + d] = v;
    }
    return m;
}

// DeepSeek auxiliary-bias adaptation, kept OUT of the forward routing path:
// host-side only, operates on the bias vector after the output is computed.
// (Adaptation cannot move to a trainer-owned step without a header change;
// this helper isolates it so the forward routing computation is pure.)
inline void w1_deepseek_bias_step(std::vector<float>& biases,
                                  const int64_t* idx, int64_t T, int64_t K) {
    int64_t E = (int64_t)biases.size();
    if (E <= 0 || idx == nullptr) return;
    for (int64_t t = 0; t < T; ++t)
        for (int64_t k = 0; k < K; ++k) {
            int64_t e = idx[t * K + k];
            if (e >= 0 && e < E) biases[(size_t)e] -= 0.001f;
        }
    double mean = 0.0;
    for (float v : biases) mean += (double)v;
    mean /= (double)E;
    for (float& v : biases) v -= (float)mean;
}

// Straight-through quantize: host-side step/codebook mapping expressed as a
// constant residual so the output stays a graph tensor (grad flows through
// as identity) and both autograd modes produce identical values.
inline Tensor w1_ste_residual(const Tensor& graph_out,
                              const float* quant_const) {
    Tensor delta(graph_out.shape());
    const float* od = graph_out.data<float>();
    float* dd = delta.data<float>();
    int64_t n = graph_out.numel();
    for (int64_t i = 0; i < n; ++i)
        dd[i] = quant_const[i] - od[i];
    return AutogradEngine::add_op(graph_out, delta);
}

} // namespace

// ========================================================================
// 1. SPARSE MoE
// ========================================================================

SparseMoE::SparseMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      router_weight(hidden, cfg.num_experts)
{
    int64_t ffn_hidden = cfg.expert_hidden_size;
    experts = create_experts(cfg.num_experts, hidden, ffn_hidden, Activation::SiLU);
}

MoEOutput SparseMoE::forward(const Tensor& x, bool training) {
    // NOTE (bug census): `training` is intentionally unused — sparse top-k routing is
    // deterministic (no expert/gating dropout in this variant; cf. GatingDropoutMoE).
    // The flag stays for the common variant signature. (void)training removed (was hiding intent).
    (void)training; // documented-unused: deterministic routing
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;

    Tensor x_flat = x.reshape({T, D});
    Tensor logits = router_weight.forward(x_flat);

    Tensor indices, weights;
    Tensor probs = softmax_with_topk(logits, K, indices, weights);

    float zl = 0.0f;
    int64_t dropped = 0;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(),
        T, K, E, D, &zl, &dropped);

    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.expert_indices = indices;
    out.expert_weights = weights;
    out.load_balance_loss = w1_lb_loss(logits, indices, E, config.load_balance_coef);
    out.z_loss = w1_z_loss(zl, config.z_loss_coef, 1.0f / (float)T);
    out.num_activated_experts = K;
    out.tokens_dropped = dropped;
    return out;
}

// ========================================================================
// 2. SOFT MoE (Dense Mixture)
// ========================================================================

SoftMoE::SoftMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden)
{
    int64_t E = cfg.num_experts;
    int64_t S = cfg.num_slots_per_expert;
    int64_t ffn_hidden = cfg.expert_hidden_size;
    int64_t total_slots = E * S;
    input_mixing = Linear(hidden, total_slots);
    output_mixing = Linear(hidden, hidden);
    experts = create_experts(E, hidden, ffn_hidden, Activation::SiLU);
}

MoEOutput SoftMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    int64_t E = config.num_experts;
    int64_t slots = config.num_slots_per_expert;
    int64_t total_slots = E * slots;

    Tensor x_flat = x.reshape({T, D});

    // Slot logits stay a graph tensor (unmutated). Full slot distribution
    // via top-k(k=total_slots) scatter — no math::softmax call.
    Tensor slot_logits = input_mixing.forward(x_flat);
    Tensor slot_idx, slot_w;
    softmax_with_topk(slot_logits, total_slots, slot_idx, slot_w);
    const int64_t* si = slot_idx.data<int64_t>();
    const float* sw = slot_w.data<float>();

    // W1: per-token per-expert weight = sum of that expert's slot probs.
    // All experts process tokens via the batched enabled-path dispatch;
    // manual expert_in/expert_out/combine od loops deleted.
    Tensor indices({T, E}, DType::I64);
    Tensor weights({T, E});
    weights.zero_();
    int64_t* oi = indices.data<int64_t>();
    float* ow = weights.data<float>();
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t e = 0; e < E; ++e)
            oi[t * E + e] = e;
        for (int64_t k = 0; k < total_slots; ++k) {
            int64_t slot = si[t * total_slots + k];
            if (slot < 0 || slot >= total_slots) continue;
            ow[t * E + slot / slots] += sw[t * total_slots + k];
        }
    }

    float zl = 0.0f;
    Tensor dispatched = moe_dispatch_batched(x_flat, experts,
        oi, ow, T, E, E, D, &zl, nullptr);

    // Output projection (graph).
    Tensor final = output_mixing.forward(dispatched);

    MoEOutput out;
    out.output = final.reshape({B, S, D});
    out.router_logits = slot_logits;
    out.expert_indices = indices;
    out.expert_weights = weights;
    out.load_balance_loss = 0.0f;
    out.z_loss = w1_z_loss(zl, config.z_loss_coef, 1.0f / (float)T);
    out.num_activated_experts = E;
    return out;
}

// ========================================================================
// 3. HIERARCHICAL MoE
// ========================================================================

HierarchicalMoE::HierarchicalMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      group_router(hidden, cfg.num_groups)
{
    int64_t ffn_hidden = cfg.expert_hidden_size;
    expert_groups.resize((size_t)cfg.num_groups);
    expert_routers.reserve(cfg.num_groups);
    for (int64_t g = 0; g < cfg.num_groups; ++g) {
        expert_groups[(size_t)g] = create_experts(cfg.experts_per_group, hidden, ffn_hidden, Activation::SiLU);
        expert_routers.emplace_back(hidden, cfg.experts_per_group);
    }
}

MoEOutput HierarchicalMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    int64_t G = config.num_groups;
    int64_t TG = config.top_groups;
    int64_t E = config.experts_per_group;
    int64_t K = config.top_experts_per_group;

    Tensor x_flat = x.reshape({T, D});

    // Level 1: group selection (graph logits, never mutated raw).
    Tensor group_logits = group_router.forward(x_flat);
    Tensor g_indices, g_weights;
    Tensor g_probs = softmax_with_topk(group_logits, TG, g_indices, g_weights);
    const int64_t* gi = g_indices.data<int64_t>();
    const float* gw = g_weights.data<float>();

    // Level 2 routing decisions (host, read-only): per (token, group-slot)
    // top-K experts within the selected group.
    std::vector<int64_t> e_idx((size_t)(T * TG * K), -1);
    std::vector<float> e_w((size_t)(T * TG * K), 0.0f);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t gk = 0; gk < TG; ++gk) {
            int64_t g = gi[t * TG + gk];
            if (g < 0 || g >= G) continue;
            Tensor expert_logits = expert_routers[(size_t)g].forward(
                x_flat.slice(0, t, t + 1));
            Tensor e_indices, e_weights;
            Tensor e_probs = softmax_with_topk(expert_logits, K, e_indices, e_weights);
            const int64_t* ei = e_indices.data<int64_t>();
            const float* ew = e_weights.data<float>();
            for (int64_t ek = 0; ek < K; ++ek) {
                e_idx[(size_t)((t * TG + gk) * K + ek)] = ei[ek];
                e_w[(size_t)((t * TG + gk) * K + ek)] = ew[ek];
            }
        }
    }

    // W1: one batched enabled-path dispatch per group with combined
    // group_weight * expert_weight; group outputs combined via add_op.
    // Per-token slice/dispatch/manual od loop deleted.
    Tensor output({T, D});
    output.zero_();
    float total_zl = 0.0f;
    for (int64_t g = 0; g < G; ++g) {
        Tensor sub_idx({T, K}, DType::I64);
        Tensor sub_w({T, K});
        sub_w.zero_();
        int64_t* subi = sub_idx.data<int64_t>();
        float* subw = sub_w.data<float>();
        for (int64_t i = 0; i < T * K; ++i) subi[i] = -1;
        bool any = false;
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t gk = 0; gk < TG; ++gk) {
                if (gi[t * TG + gk] != g) continue;
                float gw_val = gw[t * TG + gk];
                if (gw_val <= 0.0f) continue;
                for (int64_t ek = 0; ek < K; ++ek) {
                    int64_t e = e_idx[(size_t)((t * TG + gk) * K + ek)];
                    float ew_val = e_w[(size_t)((t * TG + gk) * K + ek)];
                    if (e < 0 || e >= E || ew_val <= 0.0f) continue;
                    subi[t * K + ek] = e;
                    subw[t * K + ek] = gw_val * ew_val;
                    any = true;
                }
            }
        }
        if (!any) continue;
        float zl_g = 0.0f;
        Tensor d_out = moe_dispatch_batched(x_flat, expert_groups[(size_t)g],
            subi, subw, T, K, E, D, &zl_g, nullptr);
        total_zl += zl_g;
        output = AutogradEngine::add_op(output, d_out);
    }

    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = group_logits;
    out.load_balance_loss = w1_lb_loss(group_logits, g_indices, G, config.load_balance_coef);
    out.z_loss = w1_z_loss(total_zl, config.z_loss_coef, 1.0f / (float)T);
    out.num_activated_experts = TG * K;
    return out;
}

// ========================================================================
// 4. MoMoE — Mixture of Mixture of Experts
// ========================================================================

MoMoE::MoMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      primary_router(hidden, cfg.num_groups)
{
    int64_t ffn_hidden = cfg.expert_hidden_size;
    groups.resize((size_t)cfg.num_groups);
    secondary_routers.reserve(cfg.num_groups);
    for (int64_t g = 0; g < cfg.num_groups; ++g) {
        groups[(size_t)g] = create_experts(cfg.experts_per_group, hidden, ffn_hidden, Activation::SiLU);
        secondary_routers.emplace_back(hidden, cfg.experts_per_group);
    }
}

MoEOutput MoMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    int64_t G = config.num_groups;
    int64_t TG = config.top_groups;
    int64_t E = config.experts_per_group;
    int64_t TK = config.top_experts_per_group;

    Tensor x_flat = x.reshape({T, D});

    // Primary routing: tokens -> groups (graph logits, never mutated raw).
    Tensor primary_logits = primary_router.forward(x_flat);
    Tensor g_indices, g_weights;
    Tensor g_probs = softmax_with_topk(primary_logits, TG, g_indices, g_weights);
    const int64_t* gi = g_indices.data<int64_t>();
    const float* gw = g_weights.data<float>();

    // Secondary routing decisions (host, read-only).
    std::vector<int64_t> e_idx((size_t)(T * TG * TK), -1);
    std::vector<float> e_w((size_t)(T * TG * TK), 0.0f);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t gk = 0; gk < TG; ++gk) {
            int64_t g = gi[t * TG + gk];
            if (g < 0 || g >= G) continue;
            Tensor sec_logits = secondary_routers[(size_t)g].forward(
                x_flat.slice(0, t, t + 1));
            Tensor e_indices, e_weights;
            Tensor e_probs = softmax_with_topk(sec_logits, TK, e_indices, e_weights);
            const int64_t* ei = e_indices.data<int64_t>();
            const float* ew = e_weights.data<float>();
            for (int64_t ek = 0; ek < TK; ++ek) {
                e_idx[(size_t)((t * TG + gk) * TK + ek)] = ei[ek];
                e_w[(size_t)((t * TG + gk) * TK + ek)] = ew[ek];
            }
        }
    }

    // W1: one batched enabled-path dispatch per group; manual per-token
    // dispatch/combine od loop deleted.
    Tensor output({T, D});
    output.zero_();
    float total_zl = 0.0f;
    for (int64_t g = 0; g < G; ++g) {
        Tensor sub_idx({T, TK}, DType::I64);
        Tensor sub_w({T, TK});
        sub_w.zero_();
        int64_t* subi = sub_idx.data<int64_t>();
        float* subw = sub_w.data<float>();
        for (int64_t i = 0; i < T * TK; ++i) subi[i] = -1;
        bool any = false;
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t gk = 0; gk < TG; ++gk) {
                if (gi[t * TG + gk] != g) continue;
                float gw_val = gw[t * TG + gk];
                if (gw_val <= 0.0f) continue;
                for (int64_t ek = 0; ek < TK; ++ek) {
                    int64_t e = e_idx[(size_t)((t * TG + gk) * TK + ek)];
                    float ew_val = e_w[(size_t)((t * TG + gk) * TK + ek)];
                    if (e < 0 || e >= E || ew_val <= 0.0f) continue;
                    subi[t * TK + ek] = e;
                    subw[t * TK + ek] = gw_val * ew_val;
                    any = true;
                }
            }
        }
        if (!any) continue;
        float zl_g = 0.0f;
        Tensor d_out = moe_dispatch_batched(x_flat, groups[(size_t)g],
            subi, subw, T, TK, E, D, &zl_g, nullptr);
        total_zl += zl_g;
        output = AutogradEngine::add_op(output, d_out);
    }

    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = primary_logits;
    out.load_balance_loss = w1_lb_loss(primary_logits, g_indices, G, config.load_balance_coef);
    out.z_loss = w1_z_loss(total_zl, config.z_loss_coef, 1.0f / (float)T);
    out.num_activated_experts = TG * TK;
    return out;
}

// ========================================================================
// 5. EXPERT CHOICE MoE
// ========================================================================

ExpertChoiceMoE::ExpertChoiceMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      router_weight(hidden, cfg.num_experts)
{
    int64_t ffn_hidden = cfg.expert_hidden_size;
    experts = create_experts(cfg.num_experts, hidden, ffn_hidden, Activation::SiLU);
}

MoEOutput ExpertChoiceMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    int64_t E = config.num_experts;
    int64_t capacity = std::min(T, (int64_t)(config.capacity_factor * T / E));
    if (capacity < 1) capacity = 1;

    Tensor x_flat = x.reshape({T, D});
    Tensor logits = router_weight.forward(x_flat);
    const float* l = logits.data<float>();

    // Each expert picks its top-capacity tokens (host routing decisions,
    // read-only on the graph logits). Remapped to per-token dispatch rows
    // with unit weights; manual batch_input memcpy/forward/combine deleted.
    int64_t Kcap = E;
    Tensor indices({T, Kcap}, DType::I64);
    Tensor weights({T, Kcap});
    weights.zero_();
    int64_t* ii = indices.data<int64_t>();
    float* ww = weights.data<float>();
    for (int64_t i = 0; i < T * Kcap; ++i) ii[i] = -1;
    std::vector<int64_t> used((size_t)T, 0);
    for (int64_t e = 0; e < E; ++e) {
        std::vector<std::pair<float, int64_t>> scored;
        scored.reserve((size_t)T);
        for (int64_t t = 0; t < T; ++t)
            scored.push_back({l[t * E + e], t});
        int64_t actual_cap = std::min(capacity, T);
        std::partial_sort(scored.begin(), scored.begin() + actual_cap, scored.end(),
            [](auto& a, auto& b) { return a.first > b.first; });
        for (int64_t k = 0; k < actual_cap; ++k) {
            if (scored[(size_t)k].first <= 0.0f) continue;
            int64_t t = scored[(size_t)k].second;
            int64_t slot = used[(size_t)t]++;
            if (slot >= Kcap) continue;
            ii[t * Kcap + slot] = e;
            ww[t * Kcap + slot] = 1.0f;
        }
    }

    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        ii, ww, T, Kcap, E, D, &zl, nullptr);

    // Normalize by number of experts that selected each token via constant
    // mask + mul_op (replaces the raw normalize od loop, graph-connected).
    Tensor norm({T, D});
    float* nd = norm.data<float>();
    for (int64_t t = 0; t < T; ++t) {
        float count = 0.0f;
        for (int64_t e = 0; e < E; ++e)
            if (l[t * E + e] > 0.0f) count += 1.0f;
        float inv = (count > 1.0f) ? (1.0f / count) : 1.0f;
        for (int64_t d = 0; d < D; ++d)
            nd[t * D + d] = inv;
    }
    Tensor normed = AutogradEngine::mul_op(output, norm);

    MoEOutput out;
    out.output = normed.reshape({B, S, D});
    out.router_logits = logits;
    out.expert_indices = indices;
    out.expert_weights = weights;
    out.load_balance_loss = 0.0f;
    out.z_loss = w1_z_loss(zl, config.z_loss_coef, 1.0f / (float)T);
    out.num_activated_experts = E;
    return out;
}

// ========================================================================
// 6. HASH MoE
// ========================================================================

HashMoE::HashMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      num_buckets(cfg.num_experts * cfg.hash_bucket_size)
{
    int64_t ffn_hidden = cfg.expert_hidden_size;
    experts = create_experts(cfg.num_experts, hidden, ffn_hidden, Activation::SiLU);
}

int64_t HashMoE::hash_to_expert(int64_t token_id, int64_t num_experts, int64_t bucket_size) {
    uint64_t h = (uint64_t)token_id * 0x9E3779B97F4A7C15ULL;
    h ^= h >> 37;
    h *= 0xBF58476D1CE4E5B9ULL;
    uint64_t bucket = h % (uint64_t)(num_experts * bucket_size);
    return (int64_t)(bucket / (uint64_t)bucket_size);
}

MoEOutput HashMoE::forward(const Tensor& x, const Tensor& token_ids) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    int64_t E = config.num_experts;

    Tensor x_flat = x.reshape({T, D});
    const int64_t* ids = token_ids.data<int64_t>();

    // W1: deterministic hash routing expressed as dispatch rows (K=1, unit
    // weight). Group-by-expert vectors + memcpy batches + od loops deleted.
    Tensor indices({T, 1}, DType::I64);
    Tensor weights({T, 1});
    int64_t* ii = indices.data<int64_t>();
    float* ww = weights.data<float>();
    for (int64_t t = 0; t < T; ++t) {
        int64_t e = hash_to_expert(ids[t], E, config.hash_bucket_size);
        if (e < 0 || e >= E) { ii[t] = -1; ww[t] = 0.0f; continue; }
        ii[t] = e;
        ww[t] = 1.0f;
    }

    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        ii, ww, T, 1, E, D, &zl, nullptr);

    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.expert_indices = indices;
    out.expert_weights = weights;
    out.load_balance_loss = 0.0f;
    out.z_loss = w1_z_loss(zl, config.z_loss_coef, 1.0f / (float)T);
    out.num_activated_experts = E;
    return out;
}

// ========================================================================
// 7. CROSS-LAYER MoE
// ========================================================================

CrossLayerMoE::CrossLayerMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden)
{
    int64_t ffn_hidden = cfg.expert_hidden_size;
    shared_experts = create_experts(cfg.num_experts, hidden, ffn_hidden, Activation::SiLU);
    layer_routers.reserve(cfg.num_shared_layers);
    for (int64_t i = 0; i < cfg.num_shared_layers; ++i)
        layer_routers.emplace_back(hidden, cfg.num_experts);
}

MoEOutput CrossLayerMoE::forward(const Tensor& x, int64_t layer_idx) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    int64_t E = config.num_experts;
    int64_t K = config.top_k;

    size_t ridx = (size_t)(layer_idx % config.num_shared_layers);
    Tensor x_flat = x.reshape({T, D});
    Tensor logits = layer_routers[ridx].forward(x_flat);

    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);

    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(x_flat, shared_experts,
        indices.data<int64_t>(), weights.data<float>(),
        T, K, E, D, &zl, nullptr);

    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = w1_lb_loss(logits, indices, E, config.load_balance_coef);
    out.z_loss = w1_z_loss(zl, config.z_loss_coef, 1.0f / (float)T);
    out.num_activated_experts = K;
    return out;
}

// ========================================================================
// 8. MULTIMODAL MoE (MoMMoE)
// ========================================================================

MultiModalMoE::MultiModalMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      router_weight(hidden, cfg.num_experts),
      modality_classifier(hidden, 9)
{
    int64_t ffn_hidden = cfg.expert_hidden_size;
    experts = create_experts(cfg.num_experts, hidden, ffn_hidden, Activation::SiLU);

    // Build expert-to-modality map
    expert_modality_map.resize((size_t)cfg.num_experts);
    int64_t offset = 0;
    struct { int64_t count; int64_t mod; } mods[] = {
        {cfg.text_experts, 0}, {cfg.vision_experts, 1}, {cfg.image_gen_experts, 2},
        {cfg.video_gen_experts, 3}, {cfg.audio_experts, 4}, {cfg.ocr_experts, 5},
        {cfg.cross_modal_experts, 6}
    };
    for (auto& m : mods) {
        for (int64_t i = 0; i < m.count && offset < cfg.num_experts; ++i)
            expert_modality_map[(size_t)offset++] = m.mod;
    }
    // Fill remaining with text
    while (offset < cfg.num_experts)
        expert_modality_map[(size_t)offset++] = 0;
}

MoEOutput MultiModalMoE::forward(const Tensor& x, const Tensor& modality_hints) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    int64_t E = config.num_experts;
    int64_t K = config.top_k;
    int64_t NM = num_modalities;

    Tensor x_flat = x.reshape({T, D});

    // Classify modality per token (graph). Distribution via full-probs
    // helper — no math::softmax call, no raw mutation of graph tensors.
    Tensor mod_logits = modality_classifier.forward(x_flat);
    Tensor mod_probs = w1_full_probs(mod_logits);
    const float* mp = mod_probs.data<float>();

    // W1: modality bias as a constant + graph add_op (replaces raw
    // `l[t*E+e] += 10*mp[...]` mutation; logits stay a graph tensor).
    Tensor logits_base = router_weight.forward(x_flat);
    Tensor bias({T, E});
    float* bd = bias.data<float>();
    for (int64_t t = 0; t < T; ++t)
        for (int64_t e = 0; e < E; ++e)
            bd[t * E + e] = 10.0f * mp[t * NM + expert_modality_map[(size_t)e]];
    Tensor logits = AutogradEngine::add_op(logits_base, bias);

    // W1: external modality hints as a second constant bias + add_op
    // (replaces the raw hint mutation loop).
    if (modality_hints.numel() > 0) {
        const float* mh = modality_hints.data<float>();
        int64_t mh_T = modality_hints.numel();
        Tensor hint_bias({T, E});
        hint_bias.zero_();
        float* hb = hint_bias.data<float>();
        for (int64_t t = 0; t < T && t < mh_T; ++t) {
            int64_t hint_mod = (int64_t)mh[t];
            if (hint_mod < 0 || hint_mod >= NM) continue;
            for (int64_t e = 0; e < E; ++e)
                if (expert_modality_map[(size_t)e] == hint_mod)
                    hb[t * E + e] += 5.0f;
        }
        logits = AutogradEngine::add_op(logits, hint_bias);
    }

    Tensor indices, weights;
    Tensor probs = softmax_with_topk(logits, K, indices, weights);

    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(),
        T, K, E, D, &zl, nullptr);

    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = w1_lb_loss(logits, indices, E, config.load_balance_coef);
    out.z_loss = w1_z_loss(zl, config.z_loss_coef, 1.0f / (float)T);
    out.num_activated_experts = K;
    return out;
}

// ========================================================================
// 9. MMoE — Multi-gate Mixture of Experts
// ========================================================================

MMoE::MMoE(int64_t hidden_size, const MoEAllConfig& cfg)
    : hidden_size(hidden_size), config(cfg) {
    int64_t num_experts = cfg.num_experts;
    int64_t ffn_hidden = cfg.expert_hidden_size;
    int64_t num_tasks = cfg.num_tasks > 0 ? cfg.num_tasks : 1;

    experts.reserve(num_experts);
    for (int64_t i = 0; i < num_experts; i++)
        experts.emplace_back(hidden_size, ffn_hidden);

    task_gates.reserve(num_tasks);
    for (int64_t i = 0; i < num_tasks; i++)
        task_gates.emplace_back(hidden_size, num_experts);
}

MoEOutput MMoE::forward(const Tensor& x, int64_t task_id) {
    MoEOutput out;
    int64_t T = x.numel() / hidden_size;
    Tensor x_flat = x.reshape({T, hidden_size});
    int64_t E = config.num_experts;
    int64_t K = config.top_k > 0 ? config.top_k : 2;

    int64_t tid = (task_id >= 0 && task_id < (int64_t)task_gates.size()) ? task_id : 0;
    Tensor gate_logits = task_gates[tid].forward(x_flat);

    Tensor indices({T, K}, DType::I64);
    Tensor weights({T, K});
    Tensor probs = softmax_with_topk(gate_logits, K, indices, weights);

    out.router_logits = gate_logits;
    out.expert_indices = indices;
    out.expert_weights = weights;
    out.load_balance_loss = w1_lb_loss(gate_logits, indices, E, config.load_balance_coef);
    out.z_loss = w1_z_loss(compute_z_loss(gate_logits), config.z_loss_coef, 1.0f);

    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(),
        T, K, E, hidden_size);

    out.output = output.reshape(x.shape());
    out.num_activated_experts = E;
    return out;
}

// ========================================================================
// 10. DeepSeek-MoE (shared + routed experts)
// ========================================================================

DeepSeekMoE::DeepSeekMoE(int64_t hidden_size, const MoEAllConfig& cfg)
    : hidden_size(hidden_size), config(cfg),
      shared_expert(hidden_size, cfg.expert_hidden_size) {
    int64_t num_routed = cfg.num_routed_experts > 0 ? cfg.num_routed_experts : 8;
    int64_t ffn_hidden = cfg.expert_hidden_size;

    routed_experts.reserve(num_routed);
    for (int64_t i = 0; i < num_routed; i++)
        routed_experts.emplace_back(hidden_size, ffn_hidden);

    router_weight = Linear(hidden_size, num_routed);
    expert_biases.assign(num_routed, 0.0f);
}

MoEOutput DeepSeekMoE::forward(const Tensor& x, bool training) {
    MoEOutput out;
    int64_t T = x.numel() / hidden_size;
    Tensor x_flat = x.reshape({T, hidden_size});
    int64_t E = (int64_t)routed_experts.size();
    int64_t K = config.top_k > 0 ? config.top_k : 2;

    // Shared expert is always active
    Tensor shared_out = shared_expert.forward(x_flat);

    // W1: load-balance bias applied as a graph bias_add_op with a constant
    // bias vector (replaces the raw `gl[t] += expert_biases[]` mutation;
    // gate logits stay a graph tensor).
    Tensor gate_base = router_weight.forward(x_flat);
    Tensor bias_vec({E});
    float* bv = bias_vec.data<float>();
    for (int64_t e = 0; e < E; ++e) bv[e] = expert_biases[(size_t)e];
    Tensor gate_logits = AutogradEngine::bias_add_op(gate_base, bias_vec);

    Tensor indices({T, K}, DType::I64);
    Tensor weights({T, K});
    Tensor probs = softmax_with_topk(gate_logits, K, indices, weights);

    out.router_logits = gate_logits;
    out.expert_indices = indices;
    out.expert_weights = weights;

    Tensor routed_out = moe_dispatch_batched(x_flat, routed_experts,
        indices.data<int64_t>(), weights.data<float>(),
        T, K, E, hidden_size);

    // W1: shared + routed combine via graph add_op (raw od loop deleted).
    Tensor output = AutogradEngine::add_op(shared_out, routed_out);

    out.output = output.reshape(x.shape());
    out.num_activated_experts = E + 1;

    // W1: bias adaptation moved out of the forward routing computation into
    // the file-local aux helper; host-side only, runs after the output is
    // built so the current step's routing is unaffected.
    if (training)
        w1_deepseek_bias_step(expert_biases, indices.data<int64_t>(), T, K);
    return out;
}

} // namespace moe
} // namespace quant

// ========================================================================
// 11-24: Additional 14 MoE variant implementations
// ========================================================================

namespace quant {
namespace moe {

// 11. BASE Layer MoE
BaseLayerMoE::BaseLayerMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden), router(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
}

MoEOutput BaseLayerMoE::forward(const Tensor& x, bool training) {
    // NOTE (bug census): `training` intentionally unused — deterministic routing (see SparseMoE note).
    (void)training; // documented-unused: deterministic routing
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor logits = router.forward(x_flat);
    Tensor indices, weights;
    Tensor probs = softmax_with_topk(logits, K, indices, weights);
    float zl = 0.0f;
    int64_t dropped = 0;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl, &dropped);
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.expert_indices = indices;
    out.expert_weights = weights;
    out.load_balance_loss = w1_lb_loss(logits, indices, E, config.load_balance_coef);
    out.z_loss = w1_z_loss(zl, config.z_loss_coef, 1.0f / (float)T);
    out.num_activated_experts = K;
    out.tokens_dropped = dropped;
    return out;
}

// 12. Dense MoE — all experts active
DenseMoE::DenseMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden), gate(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
}

MoEOutput DenseMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts;
    Tensor x_flat = x.reshape({T, D});
    Tensor logits = gate.forward(x_flat);
    // W1: dense routing = top-k with K=E (no math::softmax call); single
    // batched enabled-path dispatch replaces the per-expert forward loop
    // and the raw w*eo combine/zl od loop.
    Tensor indices, weights;
    Tensor probs = softmax_with_topk(logits, E, indices, weights);
    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(), T, E, E, D, &zl, nullptr);
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.expert_indices = indices;
    out.expert_weights = weights;
    out.load_balance_loss = 0.0f;
    out.z_loss = w1_z_loss(zl, config.z_loss_coef, 1.0f / (float)T);
    out.num_activated_experts = E;
    return out;
}

// 13. Shared Expert MoE (standalone)
SharedExpertMoE::SharedExpertMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      shared_expert(hidden, cfg.expert_hidden_size),
      router(hidden, cfg.num_experts) {
    routed_experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
}

MoEOutput SharedExpertMoE::forward(const Tensor& x, bool training) {
    // NOTE (bug census): `training` intentionally unused — deterministic routing (see SparseMoE note).
    (void)training; // documented-unused: deterministic routing
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor shared_out = shared_expert.forward(x_flat);
    Tensor logits = router.forward(x_flat);
    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);
    Tensor routed_out = moe_dispatch_batched(x_flat, routed_experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D);
    // W1: shared + routed combine via graph add_op (raw od loop deleted).
    Tensor output = AutogradEngine::add_op(shared_out, routed_out);
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.expert_indices = indices;
    out.expert_weights = weights;
    out.load_balance_loss = w1_lb_loss(logits, indices, E, config.load_balance_coef);
    out.num_activated_experts = E + 1;
    return out;
}

// 14. Residual MoE — overflow via residual
ResidualMoE::ResidualMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden), router(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
}

MoEOutput ResidualMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor logits = router.forward(x_flat);
    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);
    float zl = 0.0f;
    int64_t dropped = 0;
    Tensor moe_out = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl, &dropped);
    // W1: residual combine via graph add_op (raw od loop deleted).
    Tensor output = AutogradEngine::add_op(x_flat, moe_out);
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = w1_lb_loss(logits, indices, E, config.load_balance_coef);
    out.z_loss = w1_z_loss(zl, config.z_loss_coef, 1.0f / (float)T);
    out.num_activated_experts = K;
    out.tokens_dropped = dropped;
    return out;
}

// 15. Gating Dropout MoE
GatingDropoutMoE::GatingDropoutMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden), router(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
}

MoEOutput GatingDropoutMoE::forward(const Tensor& x, bool training) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor logits_base = router.forward(x_flat);
    // W1: gating dropout as a constant {T,E} keep-mask + graph mul_op.
    // Dropped positions multiply by 0 (identical to the old raw zeroing),
    // logits graph is never mutated in place.
    Tensor logits = logits_base;
    if (training && dropout_rate > 0.0f) {
        Tensor mask({T, E});
        mask.fill(1.0f);
        float* md = mask.data<float>();
        RNG rng(42);
        for (int64_t i = 0; i < T * E; ++i) {
            if (rng.uniform() < dropout_rate)
                md[i] = 0.0f;
        }
        logits = AutogradEngine::mul_op(logits_base, mask);
    }
    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);
    float zl = 0.0f;
    int64_t dropped = 0;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl, &dropped);
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = w1_lb_loss(logits, indices, E, config.load_balance_coef);
    out.z_loss = w1_z_loss(zl, config.z_loss_coef, 1.0f / (float)T);
    out.num_activated_experts = K;
    out.tokens_dropped = dropped;
    return out;
}

// 16. Domain MoE
DomainMoE::DomainMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden), domain_classifier(hidden, num_domains) {
    domain_experts.resize(num_domains);
    domain_routers.reserve(num_domains);
    int64_t per_domain = cfg.num_experts / num_domains;
    if (per_domain < 1) per_domain = 1;
    for (int64_t d = 0; d < num_domains; ++d) {
        domain_experts[(size_t)d] = create_experts(per_domain, hidden, cfg.expert_hidden_size, Activation::SiLU);
        domain_routers.emplace_back(hidden, per_domain);
    }
}

MoEOutput DomainMoE::forward(const Tensor& x, int64_t domain_id) {
    (void)domain_id;
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    Tensor x_flat = x.reshape({T, D});
    Tensor dom_logits = domain_classifier.forward(x_flat);
    // W1: domain distribution via full-probs helper (no math::softmax call).
    Tensor dom_probs = w1_full_probs(dom_logits);
    const float* dp_all = dom_probs.data<float>();
    int64_t K = config.top_k;
    // W1: per-domain batched dispatch; domain weighting via constant
    // broadcast mask + mul_op, accumulation via add_op (raw od loop deleted).
    Tensor output({T, D});
    output.zero_();
    float total_lb = 0.0f;
    float total_zl = 0.0f;
    std::vector<float> col((size_t)T);
    for (int64_t d = 0; d < num_domains; ++d) {
        int64_t E = (int64_t)domain_experts[(size_t)d].size();
        Tensor r_logits = domain_routers[(size_t)d].forward(x_flat);
        Tensor indices, weights;
        softmax_with_topk(r_logits, K, indices, weights);
        float zl = 0.0f;
        Tensor d_out = moe_dispatch_batched(x_flat, domain_experts[(size_t)d],
            indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl);
        total_zl += zl;
        for (int64_t t = 0; t < T; ++t)
            col[(size_t)t] = dp_all[t * num_domains + d];
        Tensor mask = w1_broadcast_col(col.data(), T, D);
        Tensor weighted = AutogradEngine::mul_op(d_out, mask);
        output = AutogradEngine::add_op(output, weighted);
        total_lb += compute_load_balance_loss(r_logits, indices, E);
    }
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = dom_logits;
    out.load_balance_loss = w1_scaled_loss(total_lb / (float)num_domains,
                                           config.load_balance_coef, 1.0f);
    out.z_loss = w1_z_loss(total_zl, config.z_loss_coef,
                           1.0f / ((float)T * (float)num_domains));
    out.num_activated_experts = K * num_domains;
    return out;
}

// 17. Product Key MoE
ProductKeyMoE::ProductKeyMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      key_router_a(hidden, cfg.num_experts),
      key_router_b(hidden, cfg.num_experts) {
    int64_t half = cfg.num_experts / 2;
    if (half < 1) half = 1;
    experts_a = create_experts(half, hidden, cfg.expert_hidden_size, Activation::SiLU);
    experts_b = create_experts(cfg.num_experts - half, hidden, cfg.expert_hidden_size, Activation::SiLU);
}

MoEOutput ProductKeyMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor logits_a = key_router_a.forward(x_flat);
    Tensor logits_b = key_router_b.forward(x_flat);
    Tensor indices_a, weights_a, indices_b, weights_b;
    softmax_with_topk(logits_a, K, indices_a, weights_a);
    softmax_with_topk(logits_b, K, indices_b, weights_b);
    int64_t Ea = (int64_t)experts_a.size();
    int64_t Eb = (int64_t)experts_b.size();
    float zl_a = 0, zl_b = 0;
    Tensor out_a = moe_dispatch_batched(x_flat, experts_a,
        indices_a.data<int64_t>(), weights_a.data<float>(), T, K, Ea, D, &zl_a);
    Tensor out_b = moe_dispatch_batched(x_flat, experts_b,
        indices_b.data<int64_t>(), weights_b.data<float>(), T, K, Eb, D, &zl_b);
    // W1: half combine via graph add_op (raw od loop deleted).
    Tensor output = AutogradEngine::add_op(out_a, out_b);
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits_a;
    out.load_balance_loss = w1_lb_loss(logits_a, indices_a, Ea, config.load_balance_coef);
    out.z_loss = w1_z_loss(zl_a + zl_b, config.z_loss_coef, 1.0f / (float)T);
    out.num_activated_experts = 2 * K;
    return out;
}

// 18. Attention MoE — attention-based routing
AttentionMoE::AttentionMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      q_proj(hidden, cfg.num_experts),
      k_proj(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
}

MoEOutput AttentionMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor Q = q_proj.forward(x_flat);
    Tensor K_keys = k_proj.forward(x_flat);
    // W1: routing scores as graph ops — elementwise Q*K scaled by the same
    // D*scale factor the old manual dot loop accumulated (identical values
    // in both modes, grad flows to q_proj/k_proj). Manual qd/kd/ld od loop
    // deleted.
    float scale = 1.0f / std::sqrt((float)D);
    Tensor qk = AutogradEngine::mul_op(Q, K_keys);
    Tensor sconst({T, E});
    sconst.fill((float)D * scale);
    Tensor logits = AutogradEngine::mul_op(qk, sconst);
    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);
    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl);
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = w1_lb_loss(logits, indices, E, config.load_balance_coef);
    out.z_loss = w1_z_loss(zl, config.z_loss_coef, 1.0f / (float)T);
    out.num_activated_experts = K;
    return out;
}

// 19. Low-rank MoE — low-rank bottleneck MoE
LowRankMoE::LowRankMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      down_proj(hidden, latent_dim),
      up_proj(latent_dim, hidden),
      router(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
}

MoEOutput LowRankMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor latent = down_proj.forward(x_flat);
    Tensor recovered = up_proj.forward(latent);
    Tensor logits = router.forward(recovered);
    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);
    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(recovered, experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl);
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = w1_lb_loss(logits, indices, E, config.load_balance_coef);
    out.z_loss = w1_z_loss(zl, config.z_loss_coef, 1.0f / (float)T);
    out.num_activated_experts = K;
    return out;
}

// 20. Mamba MoE — SSM + MoE hybrid
MambaMoE::MambaMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      ssm_proj(hidden, state_dim),
      router(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
}

MoEOutput MambaMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor ssm_out = ssm_proj.forward(x_flat);
    // W1: mod-index gather {T,S}->{T,D} as a graph matmul with a constant
    // one-hot ({D,S}, row d selects s=d%S), then graph add (replaces the raw
    // combined[t*D+d]=xd+sd[...] od loop; grad reaches ssm_proj).
    Tensor gather({D, state_dim});
    gather.zero_();
    float* gd = gather.data<float>();
    for (int64_t d = 0; d < D; ++d)
        gd[d * state_dim + d % state_dim] = 1.0f;
    Tensor expanded = AutogradEngine::matmul_op(ssm_out, gather, T, D, state_dim);
    Tensor combined = AutogradEngine::add_op(x_flat, expanded);
    Tensor logits = router.forward(combined);
    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);
    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(combined, experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl);
    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = w1_lb_loss(logits, indices, E, config.load_balance_coef);
    out.z_loss = w1_z_loss(zl, config.z_loss_coef, 1.0f / (float)T);
    out.num_activated_experts = K;
    return out;
}

// 21. Quantized INT8 MoE
QuantizedINT8MoE::QuantizedINT8MoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden), router(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
    expert_scales.resize(cfg.num_experts, 1.0f);
}

MoEOutput QuantizedINT8MoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor logits = router.forward(x_flat);
    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);
    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl);
    // W1: the old code multiplied the whole output by every expert scale in
    // sequence (= product, plus a const_cast on read-only data). Single
    // constant product-scale + graph mul_op: identical in both modes, no UB.
    double prod = 1.0;
    for (int64_t e = 0; e < E; ++e) prod *= (double)expert_scales[(size_t)e];
    Tensor scaleT({T, D});
    scaleT.fill((float)prod);
    Tensor scaled = AutogradEngine::mul_op(output, scaleT);
    MoEOutput out;
    out.output = scaled.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = w1_lb_loss(logits, indices, E, config.load_balance_coef);
    out.z_loss = w1_z_loss(zl, config.z_loss_coef, 1.0f / (float)T);
    out.num_activated_experts = K;
    return out;
}

// 22. Quant MoE
QuantMoE::QuantMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden), router(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
    quant_scales.resize(cfg.num_experts, 1.0f);
}

MoEOutput QuantMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor logits = router.forward(x_flat);
    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);
    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl);
    // W1: the old E-pass sequential {-s,0,+s} cascade is decided by its last
    // pass; single pass with the last scale + STE residual (graph-connected,
    // identical in both modes). Raw od mutation loops deleted.
    float s = (E > 0) ? quant_scales[(size_t)(E - 1)] : 1.0f;
    const float* od = output.data<float>();
    std::vector<float> qv((size_t)(T * D));
    for (int64_t i = 0; i < T * D; ++i) {
        float v = od[i] * s;
        qv[(size_t)i] = (v > 0.1f) ? s : ((v < -0.1f) ? -s : 0.0f);
    }
    Tensor quantized = w1_ste_residual(output, qv.data());
    MoEOutput out;
    out.output = quantized.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = w1_lb_loss(logits, indices, E, config.load_balance_coef);
    out.z_loss = w1_z_loss(zl, config.z_loss_coef, 1.0f / (float)T);
    out.num_activated_experts = K;
    return out;
}

// 23. QUANT1 MoE
Quant1MoE::Quant1MoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden), router(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
    quant1_scales.resize(cfg.num_experts, 1.0f);
}

MoEOutput Quant1MoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor logits = router.forward(x_flat);
    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);
    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl);
    // W1: E-pass sequential {-s,+s} cascade decided by its last pass; single
    // pass with the last scale + STE residual. Raw od loops deleted.
    float s = (E > 0) ? quant1_scales[(size_t)(E - 1)] : 1.0f;
    const float* od = output.data<float>();
    std::vector<float> qv((size_t)(T * D));
    for (int64_t i = 0; i < T * D; ++i) {
        float v = od[i] * s;
        qv[(size_t)i] = (v >= 0.0f) ? s : -s;
    }
    Tensor quantized = w1_ste_residual(output, qv.data());
    MoEOutput out;
    out.output = quantized.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = w1_lb_loss(logits, indices, E, config.load_balance_coef);
    out.z_loss = w1_z_loss(zl, config.z_loss_coef, 1.0f / (float)T);
    out.num_activated_experts = K;
    return out;
}

// 24. QUANT8 MoE — codebook quantized experts
QUANT8MoE::QUANT8MoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden), router(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
    codebooks.resize(cfg.num_experts);
    for (int64_t e = 0; e < cfg.num_experts; ++e) {
        codebooks[(size_t)e].resize(256, 0.0f);
        for (int i = 0; i < 256; ++i)
            codebooks[(size_t)e][(size_t)i] = (float)(i - 128) * 0.01f;
    }
}

MoEOutput QUANT8MoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor logits = router.forward(x_flat);
    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);
    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl);
    // W1: E-pass sequential codebook cascade decided by its last pass; single
    // pass with the last codebook + STE residual. Raw od loops deleted.
    const std::vector<float>& cb = codebooks[(size_t)(E > 0 ? E - 1 : 0)];
    const float* od = output.data<float>();
    std::vector<float> qv((size_t)(T * D));
    for (int64_t i = 0; i < T * D; ++i) {
        float v = od[i];
        int idx = (int)std::clamp((int)(v * 100.0f + 128), 0, 255);
        qv[(size_t)i] = cb[(size_t)idx];
    }
    Tensor quantized = w1_ste_residual(output, qv.data());
    MoEOutput out;
    out.output = quantized.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = w1_lb_loss(logits, indices, E, config.load_balance_coef);
    out.z_loss = w1_z_loss(zl, config.z_loss_coef, 1.0f / (float)T);
    out.num_activated_experts = K;
    return out;
}

// 25. QUANT4 MoE — 4-bit codebook quantized experts
QUANT4MoE::QUANT4MoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden), router(hidden, cfg.num_experts) {
    experts = create_experts(cfg.num_experts, hidden, cfg.expert_hidden_size, Activation::SiLU);
    codebooks.resize(cfg.num_experts);
    for (int64_t e = 0; e < cfg.num_experts; ++e) {
        codebooks[(size_t)e].resize(16, 0.0f);
        for (int i = 0; i < 16; ++i)
            codebooks[(size_t)e][(size_t)i] = (float)(i - 8) * 0.1f;
    }
}

MoEOutput QUANT4MoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;
    Tensor x_flat = x.reshape({T, D});
    Tensor logits = router.forward(x_flat);
    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);
    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(), T, K, E, D, &zl);
    // W1: E-pass sequential codebook cascade decided by its last pass; single
    // pass with the last codebook + STE residual. Raw od loops deleted.
    const std::vector<float>& cb = codebooks[(size_t)(E > 0 ? E - 1 : 0)];
    const float* od = output.data<float>();
    std::vector<float> qv((size_t)(T * D));
    for (int64_t i = 0; i < T * D; ++i) {
        float v = od[i];
        int idx = (int)std::clamp((int)(v * 10.0f + 8), 0, 15);
        qv[(size_t)i] = cb[(size_t)idx];
    }
    Tensor quantized = w1_ste_residual(output, qv.data());
    MoEOutput out;
    out.output = quantized.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = w1_lb_loss(logits, indices, E, config.load_balance_coef);
    out.z_loss = w1_z_loss(zl, config.z_loss_coef, 1.0f / (float)T);
    out.num_activated_experts = K;
    return out;
}

} // namespace moe
} // namespace quant
