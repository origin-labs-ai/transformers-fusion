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

// ========================================================================
// Hash utility
// ========================================================================

int64_t hash_token(int64_t token_id, int64_t range) {
    uint64_t h = (uint64_t)token_id * 0x9E3779B97F4A7C15ULL;
    h ^= h >> 37;
    h *= 0xBF58476D1CE4E5B9ULL;
    return (int64_t)(h % (uint64_t)range);
}

// ========================================================================
// MoE variant name lookup
// ========================================================================

const char* moe_variant_name(MoEVariant v) {
    switch (v) {
        case MoEVariant::SPARSE_TOP1: return "SPARSE_TOP1";
        case MoEVariant::SPARSE_TOP2: return "SPARSE_TOP2";
        case MoEVariant::SPARSE_TOPK: return "SPARSE_TOPK";
        case MoEVariant::SOFT_MIXTURE: return "SOFT_MIXTURE";
        case MoEVariant::HIERARCHICAL: return "HIERARCHICAL";
        case MoEVariant::MOMOE: return "MOMOE";
        case MoEVariant::EXPERT_CHOICE: return "EXPERT_CHOICE";
        case MoEVariant::HASH_ROUTED: return "HASH_ROUTED";
        case MoEVariant::CROSS_LAYER: return "CROSS_LAYER";
        case MoEVariant::MULTIMODAL: return "MULTIMODAL";
        case MoEVariant::MMOE: return "MMOE";
        case MoEVariant::DEEPSEEK_MOE: return "DEEPSEEK_MOE";
        case MoEVariant::BASE_LAYER: return "BASE_LAYER";
        case MoEVariant::DENSE_MOE: return "DENSE_MOE";
        case MoEVariant::SHARED_EXPERT: return "SHARED_EXPERT";
        case MoEVariant::RESIDUAL_MOE: return "RESIDUAL_MOE";
        case MoEVariant::GATING_DROPOUT: return "GATING_DROPOUT";
        case MoEVariant::DOMAIN_MOE: return "DOMAIN_MOE";
        case MoEVariant::PRODUCT_KEY: return "PRODUCT_KEY";
        case MoEVariant::ATTENTION_MOE: return "ATTENTION_MOE";
        case MoEVariant::LOWRANK_MOE: return "LOWRANK_MOE";
        case MoEVariant::MAMBA_MOE: return "MAMBA_MOE";
        case MoEVariant::QUANTIZED_INT8_MOE: return "QUANTIZED_INT8_MOE";
        case MoEVariant::QUANT_MOE: return "QUANT_MOE";
        case MoEVariant::Q1_MOE: return "Q1_MOE";
        case MoEVariant::QUANT8_MOE: return "QUANT8_MOE";
        case MoEVariant::QUANT4_MOE: return "QUANT4_MOE";
        default: return "UNKNOWN";
    }
}

// ========================================================================
// ExpertFFN
// ========================================================================

ExpertFFN::ExpertFFN() : activation(Activation::SiLU) {}

ExpertFFN::ExpertFFN(int64_t hidden_size, int64_t ffn_hidden, Activation act)
    : gate_proj(hidden_size, ffn_hidden),
      up_proj(hidden_size, ffn_hidden),
      down_proj(ffn_hidden, hidden_size),
      activation(act) {}

Tensor ExpertFFN::forward(const Tensor& x) const {
    Tensor gate = gate_proj.forward(x);
    Tensor up = up_proj.forward(x);
    Tensor act_out({gate.shape()});
    if (activation == Activation::SiLU) {
        math::silu(gate, act_out);
    } else if (activation == Activation::GELU) {
        math::gelu(gate, act_out);
    } else {
        math::relu(gate, act_out);
    }
    Tensor gated({gate.shape()});
    math::mul(act_out, up, gated);
    return down_proj.forward(gated);
}

// ========================================================================
// Softmax with Top-K extraction
// ========================================================================

Tensor softmax_with_topk(const Tensor& logits, int64_t k, Tensor& indices_out, Tensor& weights_out) {
    if (logits.rank() < 2) {
        indices_out = Tensor({1, 1}, DType::I64);
        weights_out = Tensor({1, 1});
        return Tensor({1, 1});
    }
    int64_t T = logits.dim(0);
    int64_t E = logits.dim(1);
    if (T <= 0 || E <= 0) {
        int64_t Ts = T > 0 ? T : 1;
        int64_t Es = E > 0 ? E : 1;
        indices_out = Tensor({Ts, 1}, DType::I64);
        weights_out = Tensor({Ts, 1});
        return Tensor({Ts, Es});
    }
    if (k < 1) k = 1;
    if (k > E) k = E;
    Tensor probs({T, E});
    const float* l = logits.data<float>();
    float* p = probs.data<float>();
    indices_out = Tensor({T, k}, DType::I64);
    weights_out = Tensor({T, k});
    int64_t* idx = indices_out.data<int64_t>();
    float* w = weights_out.data<float>();

    for (int64_t t = 0; t < T; ++t) {
        float maxv = l[t * E];
        for (int64_t e = 1; e < E; ++e)
            if (l[t * E + e] > maxv) maxv = l[t * E + e];
        float sum = 0.0f;
        for (int64_t e = 0; e < E; ++e) {
            float v = std::exp(l[t * E + e] - maxv);
            p[t * E + e] = v;
            sum += v;
        }
        float inv = 1.0f / sum;
        for (int64_t e = 0; e < E; ++e)
            p[t * E + e] *= inv;

        std::vector<std::pair<float, int64_t>> scored;
        scored.reserve(E);
        for (int64_t e = 0; e < E; ++e)
            scored.push_back({p[t * E + e], e});
        std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
            [](auto& a, auto& b) { return a.first > b.first; });
        for (int64_t j = 0; j < k; ++j) {
            idx[t * k + j] = scored[j].second;
            w[t * k + j] = scored[j].first;
        }
    }
    return probs;
}

// ========================================================================
// Load balancing loss calculation
// ========================================================================

float compute_load_balance_loss(const Tensor& router_logits, const Tensor& expert_indices, int64_t num_experts) {
    int64_t T = router_logits.dim(0);
    int64_t K = expert_indices.dim(1);
    const int64_t* idx = expert_indices.data<int64_t>();
    const float* logits = router_logits.data<float>();
    std::vector<double> f_i(num_experts, 0.0);
    std::vector<double> P_i(num_experts, 0.0);
    // f_i: fraction of tokens routed to expert (count / T)
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t k = 0; k < K; ++k) {
            int64_t e = idx[t * K + k];
            if (e >= 0 && e < num_experts) f_i[e] += 1.0;
        }
    }
    // P_i: mean gate probability via softmax over logits
    for (int64_t t = 0; t < T; ++t) {
        const float* row = logits + t * num_experts;
        float maxv = row[0];
        for (int64_t e = 1; e < num_experts; ++e) if (row[e] > maxv) maxv = row[e];
        double sum = 0.0;
        for (int64_t e = 0; e < num_experts; ++e) sum += std::exp((double)row[e] - maxv);
        for (int64_t e = 0; e < num_experts; ++e) {
            double p = std::exp((double)row[e] - maxv) / sum;
            P_i[e] += p;
        }
    }
    double loss = 0.0;
    for (int64_t e = 0; e < num_experts; ++e) {
        f_i[e] /= (double)T;
        P_i[e] /= (double)T;
        loss += f_i[e] * P_i[e];
    }
    return (float)(loss * (double)num_experts);
}

float compute_z_loss(const Tensor& expert_output_norms) {
    const float* d = expert_output_norms.data<float>();
    float sum = 0.0f;
    for (int64_t i = 0; i < expert_output_norms.numel(); ++i)
        sum += d[i] * d[i];
    return sum;
}

// ========================================================================
// ExpertFFN loader
// ========================================================================

std::vector<ExpertFFN> create_experts(int64_t count, int64_t hidden, int64_t ffn_hidden, Activation act) {
    std::vector<ExpertFFN> exps;
    exps.reserve(count);
    for (int64_t i = 0; i < count; ++i)
        exps.emplace_back(hidden, ffn_hidden, act);
    return exps;
}

// ========================================================================
// Batched expert dispatch
// ========================================================================

Tensor moe_dispatch_batched(const Tensor& x_flat,
                            const std::vector<ExpertFFN>& experts,
                            const int64_t* indices, const float* weights,
                            int64_t T, int64_t K, int64_t E, int64_t D,
                            float* z_loss_out, int64_t* dropped_out) {
    if (T <= 0 || K <= 0 || E <= 0 || D <= 0) {
        if (z_loss_out) *z_loss_out = 0.0f;
        if (dropped_out) *dropped_out = 0;
        Tensor out({T > 0 ? T : 1, D > 0 ? D : 1});
        out.zero_();
        return out;
    }
    if (experts.empty() || indices == nullptr || weights == nullptr) {
        if (z_loss_out) *z_loss_out = 0.0f;
        if (dropped_out) *dropped_out = 0;
        Tensor out({T, D});
        out.zero_();
        return out;
    }
    if ((int64_t)experts.size() < E) {
        E = (int64_t)experts.size();
        if (E <= 0) {
            if (z_loss_out) *z_loss_out = 0.0f;
            if (dropped_out) *dropped_out = 0;
            Tensor out({T, D});
            out.zero_();
            return out;
        }
        if (K > E) K = E;
    }
    // Autograd-aware path: when enabled, route through mul/add ops so
    // router and expert parameters receive gradients. Fallback to manual
    // batched dispatch when autograd is off (inference).
    if (AutogradEngine::enabled()) {
        Tensor output({T, D});
        output.zero_();
        // Pre-check which experts have work to avoid empty forwards
        std::vector<char> has_work((size_t)E, 0);
        for (int64_t t = 0; t < T; ++t)
            for (int64_t k = 0; k < K; ++k) {
                int64_t e = indices[t * K + k];
                if (e >= 0 && e < E) has_work[(size_t)e] = 1;
            }
        float zl = 0.0f;
        for (int64_t e = 0; e < E; ++e) {
            if (!has_work[(size_t)e]) continue;
            // Forward all tokens through this expert; Linear matmul is autograd-aware
            // so expert weights stay in the graph.
            Tensor expert_out = experts[(size_t)e].forward(x_flat); // {T,D}
            // Build per-token mask that broadcasts the routing weight to D dims
            Tensor mask({T, D});
            mask.zero_();
            float* md = mask.data<float>();
            bool any = false;
            for (int64_t t = 0; t < T; ++t) {
                float wgt = 0.0f;
                for (int64_t k = 0; k < K; ++k)
                    if (indices[t * K + k] == e) { wgt = weights[t * K + k]; break; }
                if (wgt != 0.0f) {
                    any = true;
                    for (int64_t d = 0; d < D; ++d) md[t * D + d] = wgt;
                }
            }
            if (!any) continue;
            Tensor weighted = AutogradEngine::mul_op(expert_out, mask);
            output = AutogradEngine::add_op(output, weighted);
            // z_loss accumulates squared weighted output for this expert
            const float* wd = weighted.data<float>();
            for (int64_t i = 0; i < T * D; ++i) zl += wd[i] * wd[i];
        }
        if (z_loss_out) *z_loss_out = zl;
        if (dropped_out) *dropped_out = 0;
        return output;
    }

    Tensor output({T, D});
    output.zero_();

    std::vector<int> counts((size_t)E, 0);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t k = 0; k < K; ++k) {
            int64_t e = indices[t * K + k];
            if (e >= 0 && e < E) counts[(size_t)e]++;
        }

    std::vector<std::vector<int64_t>> expert_tokens((size_t)E);
    for (int64_t e = 0; e < E; ++e)
        expert_tokens[(size_t)e].reserve((size_t)counts[(size_t)e]);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t k = 0; k < K; ++k) {
            int64_t e = indices[t * K + k];
            if (e >= 0 && e < E) expert_tokens[(size_t)e].push_back(t);
        }

    const float* xd = x_flat.data<float>();
    float* od = output.data<float>();
    float zl = 0.0f;
    int64_t dropped = 0;

    for (int64_t e = 0; e < E; ++e) {
        int64_t nt = (int64_t)expert_tokens[(size_t)e].size();
        if (nt == 0) continue;

        Tensor batch_input({nt, D});
        float* bi = batch_input.data<float>();
        for (int64_t i = 0; i < nt; ++i) {
            int64_t t = expert_tokens[(size_t)e][(size_t)i];
            std::memcpy(bi + i * D, xd + t * D, (size_t)D * sizeof(float));
        }

        Tensor batch_output = experts[(size_t)e].forward(batch_input);
        const float* bo = batch_output.data<float>();

        for (int64_t i = 0; i < nt; ++i) {
            int64_t t = expert_tokens[(size_t)e][(size_t)i];
            float wgt = 0.0f;
            for (int64_t k = 0; k < K; ++k)
                if (indices[t * K + k] == e) { wgt = weights[t * K + k]; break; }
            if (wgt <= 0.0f) { dropped++; continue; }
            for (int64_t d = 0; d < D; ++d) {
                float v = wgt * bo[i * D + d];
                od[t * D + d] += v;
                zl += v * v;
            }
        }
    }

    if (z_loss_out) *z_loss_out = zl;
    if (dropped_out) *dropped_out = dropped;
    return output;
}

namespace avx2 {

void moe_combine(float* output, const float* expert_outputs,
                 const float* weights, const int64_t* indices,
                 int64_t T, int64_t K, int64_t D) {
    for (int64_t t = 0; t < T; ++t)
        for (int64_t k = 0; k < K; ++k) {
            int64_t e = indices[t * K + k];
            float w = weights[t * K + k];
            if (e < 0 || w <= 0.0f) continue;
            const float* src = expert_outputs + (e * T + t) * D;
            float* dst = output + t * D;
            for (int64_t d = 0; d < D; ++d)
                dst[d] += w * src[d];
        }
}

void moe_softmax_topk(float* probs, int64_t* indices, float* weights,
                      const float* logits, int64_t T, int64_t E, int64_t K) {
    for (int64_t t = 0; t < T; ++t) {
        const float* row = logits + t * E;
        float* p = probs + t * E;
        float maxv = row[0];
        for (int64_t e = 1; e < E; ++e)
            if (row[e] > maxv) maxv = row[e];

        float sum = 0.0f;
        int64_t e = 0;
#if defined(QUANT_AVX2) || defined(__AVX2__)
        {
            __m256 maxv8 = _mm256_set1_ps(maxv);
            __m256 sumv = _mm256_setzero_ps();
            for (; e + 8 <= E; e += 8) {
                __m256 rv = _mm256_loadu_ps(row + e);
                __m256 ev = quant::simd::quant_exp_ps(_mm256_sub_ps(rv, maxv8));
                _mm256_storeu_ps(p + e, ev);
                sumv = _mm256_add_ps(sumv, ev);
            }
            float hsum[8];
            _mm256_storeu_ps(hsum, sumv);
            sum = hsum[0]+hsum[1]+hsum[2]+hsum[3]+hsum[4]+hsum[5]+hsum[6]+hsum[7];
        }
#endif
        for (; e < E; ++e) {
            p[e] = std::exp(row[e] - maxv);
            sum += p[e];
        }

        float inv = 1.0f / sum;
        for (int64_t e2 = 0; e2 < E; ++e2)
            p[e2] *= inv;

        std::vector<std::pair<float, int64_t>> scored;
        scored.reserve(E);
        for (int64_t e2 = 0; e2 < E; ++e2)
            scored.push_back({p[e2], e2});
        std::partial_sort(scored.begin(), scored.begin() + K, scored.end(),
            [](auto& a, auto& b) { return a.first > b.first; });
        for (int64_t k = 0; k < K; ++k) {
            indices[t * K + k] = scored[k].second;
            weights[t * K + k] = scored[k].first;
        }
    }
}

void moe_load_balance(float* f_i, float* P_i,
                      const int64_t* indices, const float* weights,
                      int64_t T, int64_t K, int64_t E) {
    std::memset(f_i, 0, (size_t)E * sizeof(float));
    std::memset(P_i, 0, (size_t)E * sizeof(float));
    for (int64_t t = 0; t < T; ++t)
        for (int64_t k = 0; k < K; ++k) {
            int64_t e = indices[t * K + k];
            if (e >= 0 && e < E) {
                f_i[e] += 1.0f;
                P_i[e] += weights[t * K + k];
            }
        }
    float inv_T = 1.0f / (float)T;
    for (int64_t e = 0; e < E; ++e) {
        f_i[e] *= inv_T;
        P_i[e] *= inv_T;
    }
}

} // namespace avx2

// ========================================================================
// MoE Factory — maps variant enum to variant name/count
// ========================================================================

int64_t moe_variant_count() {
    return 26;
}

const char* moe_variant_name_by_index(int64_t index) {
    if (index < 0 || index >= moe_variant_count()) return "UNKNOWN";
    return moe_variant_name((MoEVariant)index);
}

std::unique_ptr<void, void(*)(void*)> create_moe_variant(MoEVariant variant, int64_t hidden, const MoEAllConfig& cfg) {
    switch (variant) {
        case MoEVariant::SPARSE_TOP1: case MoEVariant::SPARSE_TOP2: case MoEVariant::SPARSE_TOPK: {
            auto* p = new SparseMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<SparseMoE*>(v); }};
        }
        case MoEVariant::SOFT_MIXTURE: {
            auto* p = new SoftMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<SoftMoE*>(v); }};
        }
        case MoEVariant::HIERARCHICAL: {
            auto* p = new HierarchicalMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<HierarchicalMoE*>(v); }};
        }
        case MoEVariant::MOMOE: {
            auto* p = new MoMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<MoMoE*>(v); }};
        }
        case MoEVariant::EXPERT_CHOICE: {
            auto* p = new ExpertChoiceMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<ExpertChoiceMoE*>(v); }};
        }
        case MoEVariant::HASH_ROUTED: {
            auto* p = new HashMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<HashMoE*>(v); }};
        }
        case MoEVariant::CROSS_LAYER: {
            auto* p = new CrossLayerMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<CrossLayerMoE*>(v); }};
        }
        case MoEVariant::MULTIMODAL: {
            auto* p = new MultiModalMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<MultiModalMoE*>(v); }};
        }
        case MoEVariant::MMOE: {
            auto* p = new MMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<MMoE*>(v); }};
        }
        case MoEVariant::DEEPSEEK_MOE: {
            auto* p = new DeepSeekMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<DeepSeekMoE*>(v); }};
        }
        case MoEVariant::BASE_LAYER: {
            auto* p = new BaseLayerMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<BaseLayerMoE*>(v); }};
        }
        case MoEVariant::DENSE_MOE: {
            auto* p = new DenseMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<DenseMoE*>(v); }};
        }
        case MoEVariant::SHARED_EXPERT: {
            auto* p = new SharedExpertMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<SharedExpertMoE*>(v); }};
        }
        case MoEVariant::RESIDUAL_MOE: {
            auto* p = new ResidualMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<ResidualMoE*>(v); }};
        }
        case MoEVariant::GATING_DROPOUT: {
            auto* p = new GatingDropoutMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<GatingDropoutMoE*>(v); }};
        }
        case MoEVariant::DOMAIN_MOE: {
            auto* p = new DomainMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<DomainMoE*>(v); }};
        }
        case MoEVariant::PRODUCT_KEY: {
            auto* p = new ProductKeyMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<ProductKeyMoE*>(v); }};
        }
        case MoEVariant::ATTENTION_MOE: {
            auto* p = new AttentionMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<AttentionMoE*>(v); }};
        }
        case MoEVariant::LOWRANK_MOE: {
            auto* p = new LowRankMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<LowRankMoE*>(v); }};
        }
        case MoEVariant::MAMBA_MOE: {
            auto* p = new MambaMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<MambaMoE*>(v); }};
        }
        case MoEVariant::QUANTIZED_INT8_MOE: {
            auto* p = new QuantizedINT8MoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<QuantizedINT8MoE*>(v); }};
        }
        case MoEVariant::QUANT_MOE: {
            auto* p = new QuantMoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<QuantMoE*>(v); }};
        }
        case MoEVariant::Q1_MOE: {
            auto* p = new Quant1MoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<Quant1MoE*>(v); }};
        }
        case MoEVariant::QUANT8_MOE: {
            auto* p = new QUANT8MoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<QUANT8MoE*>(v); }};
        }
        case MoEVariant::QUANT4_MOE: {
            auto* p = new QUANT4MoE(hidden, cfg);
            return {p, [](void* v) { delete static_cast<QUANT4MoE*>(v); }};
        }
        default:
            return std::unique_ptr<void, void(*)(void*)>(nullptr, [](void*){});
    }
}

} // namespace moe
} // namespace quant
