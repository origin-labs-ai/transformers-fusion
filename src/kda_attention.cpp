// ============================================================================
// KDA — Kimi Delta Attention: gated delta-rule linear attention (reference)
// ============================================================================
// See include/quant/kda_attention.h for the math contract.
// Deterministic init mirrors transformer.cpp conventions (RNG seeded per
// block index so stacked layers differ).

#include "quant/kda_attention.h"
#include "quant/random.h"

#include <cmath>
#include <vector>

namespace quant {

namespace {
// Same seeding discipline as src/transformer.cpp init_uniform.
void kda_init_uniform(Tensor& t, float bound, int block_idx) {
    RNG rng(42 + block_idx * 7919);
    float* d = t.data<float>();
    for (int64_t i = 0; i < t.numel(); ++i)
        d[i] = (rng.uniform() * 2.0f - 1.0f) * bound;
}
constexpr float kEps = 1e-6f;
} // namespace

KDAAttention::KDAAttention(const TransformerConfig& cfg)
    : num_heads(std::max<int64_t>(1, cfg.num_heads)),
      key_dim(cfg.head_dim > 0 ? cfg.head_dim : 32),
      value_dim(key_dim),
      normalize(true) {
    const int64_t hidden = cfg.hidden_size;
    q_proj = Linear(hidden, num_heads * key_dim);
    k_proj = Linear(hidden, num_heads * key_dim);
    v_proj = Linear(hidden, num_heads * value_dim);
    o_proj = Linear(num_heads * value_dim, hidden);

    // Base decay ~0.9 per step (log space, learnable downstream in RLL).
    log_decay = Tensor::zeros(Shape{num_heads});
    for (int64_t h = 0; h < num_heads; ++h)
        log_decay.data<float>()[h] = std::log(0.9f);

    const float bound = 1.0f / std::sqrt((float)hidden);
    kda_init_uniform(q_proj.weight, bound, 1);
    kda_init_uniform(k_proj.weight, bound, 2);
    kda_init_uniform(v_proj.weight, bound, 3);
    kda_init_uniform(o_proj.weight,
                     1.0f / std::sqrt((float)(num_heads * value_dim)), 4);
}

float KDAAttention::delta_step(float* state, const float* k, const float* v,
                               float decay, int64_t dk, int64_t dv) {
    // Gated DeltaNet update:
    //   pred_j = decay · Σ_i state[i][j] · k_i     (retrieval against the
    //                                               DECAYED memory — old
    //                                               associations have faded)
    //   err_j  = v_j − pred_j                      (correction needed)
    //   state  ← decay · state + k ⊗ err           (fade old, write correction)
    //
    // With decay=0 this becomes a HARD overwrite: state ← k ⊗ v exactly,
    // which is what the overwrite unit test asserts.
    float err_norm = 0.0f;
    std::vector<float> err((size_t)dv, 0.0f);
    for (int64_t j = 0; j < dv; ++j) {
        float acc = 0.0f;
        for (int64_t i = 0; i < dk; ++i)
            acc += state[(size_t)i * dv + j] * k[i];
        err[(size_t)j] = v[j] - decay * acc;
        err_norm += err[(size_t)j] * err[(size_t)j];
    }
    for (int64_t i = 0; i < dk; ++i) {
        const float ki = k[i];
        float* row = state + (size_t)i * dv;
        for (int64_t j = 0; j < dv; ++j)
            row[j] = decay * row[j] + ki * err[j];
    }
    return std::sqrt(err_norm);
}

Tensor KDAAttention::forward_naive(const Tensor& x) const {
    QUANT_CHECK(x.rank() == 3, "KDAAttention expects {B,S,hidden}");
    const int64_t B = x.dim(0), S = x.dim(1), HIDDEN = x.dim(2);

    Tensor q = q_proj.forward(x); // {B,S,nh*dk}
    Tensor k = k_proj.forward(x);
    Tensor v = v_proj.forward(x);

    Tensor out(Shape{B, S, num_heads * value_dim});

    for (int64_t b = 0; b < B; ++b) {
        std::vector<float> state((size_t)num_heads * key_dim * value_dim, 0.0f);
        std::vector<float> cum_weight((size_t)num_heads, 1.0f);

        for (int64_t t = 0; t < S; ++t) {
            for (int64_t h = 0; h < num_heads; ++h) {
                const float* kt =
                    k.data<float>() + ((b * S + t) * num_heads + h) * key_dim;
                const float* vt =
                    v.data<float>() + ((b * S + t) * num_heads + h) * value_dim;
                const float* qt =
                    q.data<float>() + ((b * S + t) * num_heads + h) * key_dim;
                float* st = state.data() + (size_t)h * key_dim * value_dim;

                const float decay =
                    std::exp(log_decay.data<float>()[h]);
                (void)KDAAttention::delta_step(st, kt, vt, decay, key_dim,
                                               value_dim);

                cum_weight[h] = cum_weight[h] * decay;

                float* ot =
                    out.data<float>() + ((b * S + t) * num_heads + h) * value_dim;
                for (int64_t j = 0; j < value_dim; ++j) {
                    float acc = 0.0f;
                    for (int64_t i = 0; i < key_dim; ++i)
                        acc += qt[i] * st[(size_t)i * value_dim + j];
                    if (normalize)
                        acc /= std::max(cum_weight[h], kEps);
                    ot[j] = acc;
                }
            }
        }
    }
    return o_proj.forward(out);
}

int64_t KDAAttention::param_count() const {
    return q_proj.param_count() + k_proj.param_count() +
           v_proj.param_count() + o_proj.param_count() + num_heads;
}

} // namespace quant
