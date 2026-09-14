#include "quant/transformer.h"
#include "quant/math.h"
#include "quant/autograd.h"
#include "quant/random.h"
#include "quant/simd_math.h"
#include "quant/flash_attention.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <utility>

#if defined(QUANT_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace quant {

// Initialize weight with uniform random values in [-bound, bound]
static void init_uniform(Tensor& t, float bound, int block_idx = 0) {
    RNG rng(42 + block_idx * 7919);
    float* d = t.data<float>();
    for (int64_t i = 0; i < t.numel(); i++)
        d[i] = (rng.uniform() * 2.0f - 1.0f) * bound;
}

// Embedding
Embedding::Embedding(int64_t vocab_size, int64_t dim)
    : weight(Tensor::zeros(Shape{vocab_size, dim})) {
    // BUGFIX: div-by-zero/sqrt-of-zero guard — dim<=0 would make
    // 1/sqrt(0)=+INF and poison init_uniform; fall back to scale 1.
    float scale = (dim > 0) ? 1.0f / std::sqrt((float)dim) : 1.0f;
    if (!std::isfinite(scale) || scale <= 0.0f) scale = 1.0f;
    init_uniform(weight, scale);
}

Tensor Embedding::forward(const Tensor& input_ids) const {
    if (AutogradEngine::enabled())
        return AutogradEngine::embedding_op(input_ids, weight);
    int64_t batch = input_ids.numel();
    int64_t dim = weight.shape().dims[1];
    int64_t vocab = weight.shape().dims[0];
    Tensor out(Shape{batch, dim}, DType::F32);
    const float* ids = input_ids.data<float>();
    const float* w = weight.data<float>();
    float* od = out.data<float>();
    for (int64_t i = 0; i < batch; i++) {
        int64_t id = (int64_t)ids[i];
        if (id < 0) id = 0;
        if (id >= vocab) id = 0;
        memcpy(od + i * dim, w + id * dim, dim * sizeof(float));
    }
    return out;
}

size_t Embedding::param_count() const { return weight.numel(); }

// Linear
Linear::Linear(int64_t in_features, int64_t out_features)
    : weight(Tensor::zeros(Shape{out_features, in_features})),
      bias(Tensor::zeros(Shape{out_features})) {
    // BUGFIX: same sqrt/div-zero guard as Embedding for in_features<=0.
    float scale = (in_features > 0) ? 1.0f / std::sqrt((float)in_features) : 1.0f;
    if (!std::isfinite(scale) || scale <= 0.0f) scale = 1.0f;
    init_uniform(weight, scale);
}

Tensor Linear::forward(const Tensor& input) const {
    int64_t in_dim = weight.shape().dims[1];
    int64_t out_dim = weight.shape().dims[0];
    int64_t in_rank = input.rank();
    int64_t batch = input.numel() / in_dim;
    
    Tensor inp2d = (in_rank > 2) ? input.reshape(Shape{batch, in_dim}) : input;
    
    Tensor out = AutogradEngine::matmul_op(inp2d, weight, batch, out_dim, in_dim);
    
    if (bias.numel() > 0) {
        if (AutogradEngine::enabled()) {
            out = AutogradEngine::bias_add_op(out, bias);
        } else {
            float* od = (float*)out.data();
            const float* bd = (const float*)bias.data();
            for (int64_t i = 0; i < batch; i++) {
                float* row = od + i * out_dim;
                int64_t j = 0;
#if defined(QUANT_AVX2) || defined(__AVX2__)
                for (; j + 8 <= out_dim; j += 8) {
                    __m256 ov = _mm256_loadu_ps(row + j);
                    __m256 bv = _mm256_loadu_ps(bd + j);
                    _mm256_storeu_ps(row + j, _mm256_add_ps(ov, bv));
                }
#endif
                for (; j < out_dim; j++)
                    row[j] += bd[j];
            }
        }
    }
    
    // Restore leading dims if input was > 2D
    if (in_rank > 2) {
        Shape out_shape = input.shape();
        out_shape.dims[in_rank - 1] = out_dim;
        return out.reshape(out_shape);
    }
    return out;
}

size_t Linear::param_count() const { return weight.numel() + bias.numel(); }

// RMSNorm
RMSNorm::RMSNorm(int64_t size, float eps_val)
    // BUGFIX: sqrt-of-negative/zero-eps guard — a negative eps would poison
    // the rms sqrt (ss+eps could go negative) and eps==0 risks div-by-zero
    // on all-zero rows; clamp to a small positive floor.
    : weight(Tensor::ones(Shape{size})), eps((std::isfinite(eps_val) && eps_val > 0.0f) ? eps_val : 1e-6f) {}

Tensor RMSNorm::forward(const Tensor& input) const {
    return AutogradEngine::rms_norm_op(input, weight, eps);
}

// RotaryEmbedding
RotaryEmbedding::RotaryEmbedding(int64_t hd, int64_t max_seq_len, float t)
    : head_dim(hd), theta(t) {
    cos_cached = Tensor(Shape{max_seq_len, hd / 2}, DType::F32);
    sin_cached = Tensor(Shape{max_seq_len, hd / 2}, DType::F32);
    float* cos_d = (float*)cos_cached.data();
    float* sin_d = (float*)sin_cached.data();
    for (int64_t i = 0; i < max_seq_len; i++) {
        for (int64_t j = 0; j < hd / 2; j++) {
            float inv_freq = 1.0f / std::pow(theta, (float)(2 * j) / hd);
            float val = (float)i * inv_freq;
            cos_d[i * hd / 2 + j] = std::cos(val);
            sin_d[i * hd / 2 + j] = std::sin(val);
        }
    }
}

// YARN / NTK-aware RoPE — long-context extension
RotaryEmbedding::RotaryEmbedding(int64_t hd, int64_t max_seq_len, float t,
                                 RoPEScalingMode mode, float factor,
                                 int64_t original_max_seq_len,
                                 float yarn_beta_fast,
                                 float yarn_beta_slow,
                                 float yarn_attn_factor)
    : head_dim(hd), theta(t), scaling_mode(mode), scaling_factor(factor),
      mscale(1.0f), yarn_attn_factor(yarn_attn_factor) {
    // Effective YARN scale derives from the context-window extension ratio.
    // original_max_seq_len == 0 means "no separate original window" → fall
    // back to max_seq_len (ratio 1), so `factor` alone drives the scale.
    int64_t orig_len = original_max_seq_len > 0 ? original_max_seq_len : max_seq_len;
    float len_ratio = (orig_len > 0) ? (float)max_seq_len / (float)orig_len : 1.0f;
    float effective_scale = factor;
    if (len_ratio > effective_scale) effective_scale = len_ratio;
    if (mode == RoPEScalingMode::YARN && effective_scale > 1.0f) {
        mscale = 0.1f * std::log(effective_scale) + 1.0f;
    }
    cos_cached = Tensor(Shape{max_seq_len, hd / 2}, DType::F32);
    sin_cached = Tensor(Shape{max_seq_len, hd / 2}, DType::F32);
    float* cos_d = (float*)cos_cached.data();
    float* sin_d = (float*)sin_cached.data();

    for (int64_t i = 0; i < max_seq_len; i++) {
        for (int64_t j = 0; j < hd / 2; j++) {
            float inv_freq_base = 1.0f / std::pow(theta, (float)(2 * j) / hd);
            float freq = inv_freq_base;
            float v_pos = (float)i;

            if (mode == RoPEScalingMode::Linear) {
                v_pos /= factor;
            } else if (mode == RoPEScalingMode::NTK) {
                float alpha = factor;
                float base = std::pow(alpha, (float)hd / (hd - 2.0f));
                float ntK_theta = theta * base;
                float ntK_inv_freq = 1.0f / std::pow(ntK_theta, (float)(2 * j) / hd);
                freq = ntK_inv_freq;
            } else if (mode == RoPEScalingMode::YARN) {
                // YARN ramp over normalized dim coordinate w ∈ [0,1] inclusive:
                // j = 0 → 0, j = half-1 → 1. Guarded against degenerate
                // beta config (beta_fast == 0 or == beta_slow → denom 0).
                int64_t half = hd / 2;
                float w = (half <= 1) ? 1.0f : (float)j / (float)(half - 1);
                float ext_f = (factor > 0.0f) ? 1.0f / factor : 1.0f;
                float lo = 0.0f;
                float denom = 1.0f;
                bool ramp_ok = false;
                if (yarn_beta_fast != 0.0f && yarn_beta_fast != yarn_beta_slow) {
                    lo = yarn_beta_slow / yarn_beta_fast;
                    denom = 1.0f - lo;
                    ramp_ok = (denom != 0.0f);
                }
                if (!ramp_ok) {
                    freq = (w < 1.0f) ? inv_freq_base : inv_freq_base * ext_f;
                } else if (w <= lo) {
                    freq = inv_freq_base;
                } else if (w >= 1.0f) {
                    freq = inv_freq_base * ext_f;
                } else {
                    float smooth = (w - lo) / denom;
                    smooth = 0.5f * (1.0f - std::cos(smooth * 3.14159265f));
                    freq = inv_freq_base * ((1.0f - smooth) + smooth * ext_f);
                }
            }

            float val = v_pos * freq;
            cos_d[i * hd / 2 + j] = std::cos(val);
            sin_d[i * hd / 2 + j] = std::sin(val);
        }
    }
}

void RotaryEmbedding::apply(Tensor& x, int64_t seq_start, int64_t seq_len) const {
    int64_t B = x.shape().dims[0];
    int64_t H = x.shape().dims[1];
    int64_t S = x.shape().dims[2];
    int64_t D = x.shape().dims[3];
    int64_t half_D = D / 2;
    float* xd = (float*)x.data();
    const float* cos_d = (const float*)cos_cached.data();
    const float* sin_d = (const float*)sin_cached.data();
    int64_t HS = H * S;
    int64_t HSD = HS * D;
    int64_t cos_stride = cos_cached.shape().dims[1];
    // ASAN fix (2026-09-14, CI GCC-13 ASAN heap-buffer-overflow): callers
    // drive generation past the rope window (generate_new_tokens keeps a
    // 64-token sliding window while cos_cached covers max_seq_len=32 in
    // toy configs; seq_start grows unboundedly). pos >= max positions used
    // to read past cos/sin (512B region + 4B READ at its end). Clamp the
    // position into the cached window — frequencies repeat by construction
    // (cos/sin of angle), so wrapping is the numerically faithful fallback;
    // degenerate (empty cache) → no-op instead of OOB.
    int64_t cache_rows = cos_cached.shape().dims[0];
    if (cache_rows <= 0 || cos_stride <= 0) return;
    auto clamp_pos = [&](int64_t p) -> int64_t {
        if (p < 0) return 0;
        if (p >= cache_rows) return p % cache_rows;
        return p;
    };
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t s = 0; s < S; s++) {
                int64_t pos = clamp_pos(seq_start + s);
                int64_t x_off = b * HSD + h * S * D + s * D;
                int64_t c_off = pos * cos_stride;
                int64_t d = 0;
#if defined(QUANT_AVX2) || defined(__AVX2__)
                for (; d + 8 <= half_D; d += 8) {
                    __m256 x1v = _mm256_loadu_ps(xd + x_off + d);
                    __m256 x2v = _mm256_loadu_ps(xd + x_off + d + half_D);
                    __m256 cosv = _mm256_loadu_ps(cos_d + c_off + d);
                    __m256 sinv = _mm256_loadu_ps(sin_d + c_off + d);
                    _mm256_storeu_ps(xd + x_off + d,
                        _mm256_sub_ps(_mm256_mul_ps(x1v, cosv), _mm256_mul_ps(x2v, sinv)));
                    _mm256_storeu_ps(xd + x_off + d + half_D,
                        _mm256_add_ps(_mm256_mul_ps(x1v, sinv), _mm256_mul_ps(x2v, cosv)));
                }
#endif
                for (; d < half_D; d++) {
                    float x1 = xd[x_off + d];
                    float x2 = xd[x_off + d + half_D];
                    float cv = cos_d[c_off + d];
                    float sv = sin_d[c_off + d];
                    xd[x_off + d] = x1 * cv - x2 * sv;
                    xd[x_off + d + half_D] = x1 * sv + x2 * cv;
                }
            }
        }
    }
}

// Attention
Attention::Attention(const TransformerConfig& cfg)
    : num_heads(cfg.num_heads), 
      num_kv_heads(cfg.num_kv_heads > 0 ? cfg.num_kv_heads : cfg.num_heads),
      head_dim(cfg.head_dim),
      use_fp8(cfg.use_fp8),
      fp8_e4m3(cfg.fp8_use_e4m3)
{
    if (cfg.rope_scaling_mode != RoPEScalingMode::None) {
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta,
                               cfg.rope_scaling_mode, cfg.rope_scaling_factor,
                               cfg.rope_original_max_seq_len,
                               cfg.yarn_beta_fast, cfg.yarn_beta_slow,
                               cfg.yarn_attn_factor);
    } else {
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta);
    }
    q_proj = Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim);
    k_proj = Linear(cfg.hidden_size, num_kv_heads * cfg.head_dim);
    v_proj = Linear(cfg.hidden_size, num_kv_heads * cfg.head_dim);
    o_proj = Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size);
}

Tensor Attention::forward(const Tensor& x, const Tensor& positions,
                           const Tensor& mask, KVCache& cache, int layer_idx) const {
    int64_t B = x.shape().dims[0];
    int64_t S = x.shape().dims[1];

    // L052: FP8 attention path — inference-only (autograd disabled). Training
    // keeps the FP32 graph path below so gradients are exact.
    const bool fp8_path = use_fp8 && !AutogradEngine::enabled();
    Tensor q = fp8_path ? linear_forward_fp8(q_proj, x, fp8_e4m3) : q_proj.forward(x);
    Tensor k = fp8_path ? linear_forward_fp8(k_proj, x, fp8_e4m3) : k_proj.forward(x);
    Tensor v = fp8_path ? linear_forward_fp8(v_proj, x, fp8_e4m3) : v_proj.forward(x);
    
    Tensor q_reshaped = q.reshape(Shape{B, S, num_heads, head_dim});
    Tensor k_reshaped = k.reshape(Shape{B, S, num_kv_heads, head_dim});
    Tensor v_reshaped = v.reshape(Shape{B, S, num_kv_heads, head_dim});
    
    Tensor attn_out;
    if (AutogradEngine::enabled()) {
        Tensor q_t = AutogradEngine::transpose_op(q_reshaped, 1, 2);
        Tensor k_t = AutogradEngine::transpose_op(k_reshaped, 1, 2);
        Tensor v_t = AutogradEngine::transpose_op(v_reshaped, 1, 2);
        Tensor q_rope = AutogradEngine::rotary_op(q_t, rope.cos_cached, rope.sin_cached, 0, S);
        Tensor k_rope = AutogradEngine::rotary_op(k_t, rope.cos_cached, rope.sin_cached, 0, S);
        attn_out = AutogradEngine::attention_op(q_rope, k_rope, v_t, num_heads, num_kv_heads, head_dim);
    } else {
        Tensor q_t = q_reshaped.transpose(1, 2);  // {B, H, S, D}
        Tensor k_t = k_reshaped.transpose(1, 2);  // {B, KV_H, S, D}
        Tensor v_t = v_reshaped.transpose(1, 2);  // {B, KV_H, S, D}
        
        int64_t seq_start = 0;
        int64_t S_full = S;
        Tensor k_used, v_used;
        if (B > 1) {
            // Multi-batch inputs: cache is batch-1; do full-window causal attention
            rope.apply(q_t, 0, S);
            rope.apply(k_t, 0, S);
            k_used = k_t;
            v_used = v_t;
        } else {
            seq_start = cache.context_len(layer_idx);
            rope.apply(q_t, seq_start, S);
            rope.apply(k_t, seq_start, S);
            cache.append(layer_idx, k_t, v_t);
            auto [k_full, v_full] = cache.get_all(layer_idx);
            k_used = k_full;
            v_used = v_full;
            S_full = k_full.shape().dims[2];
        }
        // YARN attention scale: applied EXACTLY ONCE via Q pre-scale, before
        // the S_full dispatch. Both branches then use a base 1/sqrt(D) scale
        // (short path `scale` below; long path inside flash_attention_forward
        // which hardcodes 1/sqrt(D)). Never both `scale *= mult` and `q *= mult`.
        float mult = rope.attn_scale_mult();
        if (mult != 1.0f) {
            // BUGFIX: NaN/INF guard — a corrupt attn mult must not poison Q.
            if (!std::isfinite(mult)) mult = 1.0f;
            float* qd = q_t.data<float>();
            for (int64_t i = 0; i < q_t.numel(); ++i) qd[i] *= mult;
        }
        // BUGFIX: div-by-zero/sqrt guard for head_dim<=0 (1/sqrt(0)=+INF
        // would NaN-poison every score); degenerate configs get scale 0
        // so the masked softmax below yields a safe uniform/zero row.
        float scale = (head_dim > 0) ? 1.0f / std::sqrt((float)head_dim) : 0.0f;
        if (!std::isfinite(scale)) scale = 0.0f;

        // For GQA: expand K/V from {B, KV_H, S_full, D} to {B, H, S, D}
        Tensor k_expanded, v_expanded;
        if (num_kv_heads < num_heads) {
            int64_t group_size = num_heads / num_kv_heads;
            k_expanded = Tensor(Shape{B, num_heads, S_full, head_dim}, DType::F32);
            v_expanded = Tensor(Shape{B, num_heads, S_full, head_dim}, DType::F32);
            for (int64_t b = 0; b < B; b++) {
                for (int64_t h = 0; h < num_heads; h++) {
                    int64_t kh = h / group_size;
                    for (int64_t s = 0; s < S_full; s++) {
                        int64_t src_off = ((b * num_kv_heads + kh) * S_full + s) * head_dim;
                        int64_t dst_off = ((b * num_heads + h) * S_full + s) * head_dim;
                        memcpy(k_expanded.data<float>() + dst_off,
                               k_used.data<float>() + src_off, head_dim * sizeof(float));
                        memcpy(v_expanded.data<float>() + dst_off,
                               v_used.data<float>() + src_off, head_dim * sizeof(float));
                    }
                }
            }
        } else {
            k_expanded = k_used;
            v_expanded = v_used;
        }

        // Use FlashAttention for long sequences (memory O(n) vs O(n²))
        if (S_full > 64) {
            Tensor causal_mask(Shape{1, 1, S, S_full}, DType::F32);
            float* md = causal_mask.data<float>();
            for (int64_t s = 0; s < S; s++) {
                for (int64_t t = 0; t < S_full; t++) {
                    md[s * S_full + t] = (t > s + seq_start) ? -INFINITY : 0.0f;
                }
            }
            attn_out = flash_attention_forward(q_t, k_expanded, v_expanded,
                                               causal_mask, 0.0f, true);
        } else {
            // Short sequence: use standard attention (simpler, no block overhead)
            const float* qd = (const float*)q_t.data();
            const float* kd = (const float*)k_expanded.data();
            const float* vd = (const float*)v_expanded.data();
            
            Tensor score(Shape{B, num_heads, S, S_full}, DType::F32);
            float* sd = (float*)score.data();
            for (int64_t b = 0; b < B; b++) {
                for (int64_t h = 0; h < num_heads; h++) {
                    int64_t q_base = ((b * num_heads + h) * S) * head_dim;
                    int64_t k_base = ((b * num_heads + h)) * S_full * head_dim;
                    int64_t s_base = (b * num_heads + h) * S * S_full;
                    for (int64_t s = 0; s < S; s++) {
                        const float* qptr = qd + q_base + s * head_dim;
                        for (int64_t t = 0; t < S_full; t++) {
                            const float* kptr = kd + k_base + t * head_dim;
                            float sum = 0;
                            for (int64_t d = 0; d < head_dim; d++)
                                sum += qptr[d] * kptr[d];
                            // BUGFIX: NaN propagation guard — NaN q/k would
                            // poison the score row and then the softmax max;
                            // collapse to -INF (zero weight after softmax).
                            if (!std::isfinite(sum)) sum = -INFINITY;
                            sd[s_base + s * S_full + t] = sum * scale;
                            if (t > s + seq_start)
                                sd[s_base + s * S_full + t] = -INFINITY;
                        }
                    }
                }
            }
            
            Tensor attn_weights(score.shape(), DType::F32);
            float* wd = (float*)attn_weights.data();
            for (int64_t b = 0; b < B; b++) {
                for (int64_t h = 0; h < num_heads; h++) {
                    int64_t base = (b * num_heads + h) * S * S_full;
                    for (int64_t s = 0; s < S; s++) {
                        int64_t row = base + s * S_full;
                        // BUGFIX: max-subtracted softmax — max_v over the
                        // row keeps every exp() arg <= 0 so it cannot
                        // overflow; NaN scores are skipped so they cannot
                        // poison the max (else max_v=NaN -> all NaN).
                        float max_v = -1e30f;
                        for (int64_t t = 0; t < S_full; t++) {
                            float v = sd[row + t];
                            if (std::isfinite(v) && v > max_v) max_v = v;
                        }
                        float sum_exp = 0;
                        for (int64_t t = 0; t < S_full; t++) {
                            // BUGFIX: non-finite scores get zero weight;
                            // exp(-INF-max_v) underflows to 0 safely.
                            float e = std::isfinite(sd[row + t])
                                ? std::exp(sd[row + t] - max_v) : 0.0f;
                            if (!std::isfinite(e)) e = 0.0f;
                            wd[row + t] = e;
                            sum_exp += e;
                        }
                        // BUGFIX: div-by-zero guard — fully-masked rows have
                        // sum_exp==0; keep weights at 0 instead of 1/1=uniform.
                        float inv_sum = (sum_exp > 0.0f && std::isfinite(sum_exp))
                            ? 1.0f / sum_exp : 0.0f;
                        for (int64_t t = 0; t < S_full; t++)
                            wd[row + t] *= inv_sum;
                    }
                }
            }
            
            attn_out = Tensor(Shape{B, num_heads, S, head_dim}, DType::F32);
            float* aod = (float*)attn_out.data();
            for (int64_t b = 0; b < B; b++) {
                for (int64_t h = 0; h < num_heads; h++) {
                    int64_t w_base = (b * num_heads + h) * S * S_full;
                    int64_t v_base = (b * num_heads + h) * S_full * head_dim;
                    int64_t o_base = ((b * num_heads + h) * S) * head_dim;
                    for (int64_t s = 0; s < S; s++) {
                        for (int64_t d = 0; d < head_dim; d++) {
                            float sum = 0;
                            const float* wptr = wd + w_base + s * S_full;
                            const float* vptr = vd + v_base + d;
                            for (int64_t t = 0; t < S_full; t++)
                                sum += wptr[t] * vptr[t * head_dim];
                            aod[o_base + s * head_dim + d] = sum;
                        }
                    }
                }
            }
        }
    }
    
    // Output projection: flatten {B,H,S,D} -> {B*S, H*D}
    Tensor attn_flat;
    if (AutogradEngine::enabled()) {
        attn_flat = AutogradEngine::flatten_attention_op(attn_out, B, num_heads, S, head_dim);
    } else {
        // Transpose {B,H,S,D} -> {B,S,H,D} then flatten
        Tensor attn_t = attn_out.transpose(1, 2);
        attn_flat = attn_t.reshape(Shape{B * S, num_heads * head_dim});
    }
    Tensor o_out = fp8_path
        ? linear_forward_fp8(o_proj, attn_flat, fp8_e4m3)
        : o_proj.forward(attn_flat);
    return o_out.reshape(Shape{B, S, o_out.dim(1)});
}

// L052: FP8 linear GEMM slow path (inference-only; no autograd nodes).
Tensor linear_forward_fp8(const Linear& lin, const Tensor& input, bool use_e4m3) {
    int64_t in_dim = lin.weight.shape().dims[1];
    int64_t out_dim = lin.weight.shape().dims[0];
    int64_t in_rank = input.rank();
    int64_t batch = input.numel() / in_dim;
    Tensor inp2d = (in_rank > 2) ? input.reshape(Shape{batch, in_dim}) : input;
    const float* ad = inp2d.data<float>();
    const float* wd = lin.weight.data<float>();
    // math::fp8_gemm wants B as {K, N}; weight is {N, K} so transpose-copy.
    std::vector<float> wth((size_t)in_dim * (size_t)out_dim);
    for (int64_t k = 0; k < in_dim; k++)
        for (int64_t n = 0; n < out_dim; n++)
            wth[(size_t)k * (size_t)out_dim + (size_t)n] =
                wd[(size_t)n * (size_t)in_dim + (size_t)k];
    Tensor out(Shape{batch, out_dim}, DType::F32);
    math::fp8_gemm(ad, wth.data(), out.data<float>(), batch, out_dim, in_dim, use_e4m3);
    if (lin.bias.numel() > 0) {
        float* od = out.data<float>();
        const float* bd = lin.bias.data<float>();
        for (int64_t i = 0; i < batch; i++) {
            float* row = od + i * out_dim;
            for (int64_t j = 0; j < out_dim; j++) row[j] += bd[j];
        }
    }
    if (in_rank > 2) {
        Shape out_shape = input.shape();
        out_shape.dims[in_rank - 1] = out_dim;
        return out.reshape(out_shape);
    }
    return out;
}

// FFN
FFN::FFN(const TransformerConfig& cfg)
    : activation(cfg.activation)
{
    gate_proj = Linear(cfg.hidden_size, cfg.ffn_hidden_size);
    up_proj = Linear(cfg.hidden_size, cfg.ffn_hidden_size);
    down_proj = Linear(cfg.ffn_hidden_size, cfg.hidden_size);
}

Tensor FFN::forward(const Tensor& x) const {
    Tensor gate = gate_proj.forward(x);
    Tensor up = up_proj.forward(x);
    
    Tensor hidden;
    if (activation == Activation::SwiGLU || activation == Activation::SiLU) {
        // SwiGLU: silu(gate) * up  — SIMD dispatch
        if (activation == Activation::SwiGLU) {
            hidden = Tensor(gate.shape(), DType::F32);
            const float* gd = gate.data<float>();
            const float* ud = up.data<float>();
            float* hd = hidden.data<float>();
            int64_t n = gate.numel();
            for (int64_t i = 0; i < n; i++) {
                float g = gd[i];
                float sil = g / (1.0f + std::exp(-g));
                hd[i] = sil * ud[i];
            }
        } else {
            gate = AutogradEngine::silu_op(gate);
            hidden = AutogradEngine::mul_op(gate, up);
        }
    } else if (activation == Activation::GeGLU || activation == Activation::GELU) {
        if (activation == Activation::GeGLU) {
            hidden = Tensor(gate.shape(), DType::F32);
            const float* gd = gate.data<float>();
            const float* ud = up.data<float>();
            float* hd = hidden.data<float>();
            int64_t n = gate.numel();
            const float s = 0.7071067811865475f;
            for (int64_t i = 0; i < n; i++) {
                float g = gd[i];
                float gel = 0.5f * g * (1.0f + std::erf(g * s));
                hd[i] = gel * ud[i];
            }
        } else {
            math::gelu(gate, gate);
            hidden = Tensor(gate.shape(), DType::F32);
            const float* gd = gate.data<float>();
            const float* ud = up.data<float>();
            float* hd = hidden.data<float>();
            for (int64_t i = 0; i < gate.numel(); i++) hd[i] = gd[i] * ud[i];
        }
    } else {
        math::relu(gate, gate);
        hidden = Tensor(gate.shape(), DType::F32);
        const float* gd = gate.data<float>();
        const float* ud = up.data<float>();
        float* hd = hidden.data<float>();
        for (int64_t i = 0; i < gate.numel(); i++) hd[i] = gd[i] * ud[i];
    }
    
    return down_proj.forward(hidden);
}

// TransformerBlock
TransformerBlock::TransformerBlock(const TransformerConfig& cfg)
    : attention_norm(cfg.hidden_size, cfg.norm_eps),
      attention(cfg),
      ffn_norm(cfg.hidden_size, cfg.norm_eps),
      ffn(cfg),
      use_parallel_residual(cfg.use_parallel_residual) {}

Tensor TransformerBlock::forward(const Tensor& x, const Tensor& positions,
                                  const Tensor& mask, KVCache& cache, int layer_idx) const {
    if (use_parallel_residual) {
        // GPT-NeoX parallel residual: both attention and FFN see the same pre-norm input
        // output = x + attn(norm1(x)) + ffn(norm2(x))
        Tensor normed_attn = attention_norm.forward(x);
        Tensor normed_ffn  = ffn_norm.forward(x);
        Tensor attn_out = attention.forward(normed_attn, positions, mask, cache, layer_idx);
        Tensor ffn_out  = ffn.forward(normed_ffn);
        Tensor combined = AutogradEngine::add_op(attn_out, ffn_out);
        return AutogradEngine::add_op(combined, x);
    } else {
        // Standard sequential residual (GPT-2/LLaMA style)
        Tensor attn_input = attention_norm.forward(x);
        Tensor attn_out = attention.forward(attn_input, positions, mask, cache, layer_idx);
        attn_out = AutogradEngine::add_op(attn_out, x);

        Tensor ffn_input = ffn_norm.forward(attn_out);
        Tensor ffn_out = ffn.forward(ffn_input);
        ffn_out = AutogradEngine::add_op(ffn_out, attn_out);
        return ffn_out;
    }
}

class AttentionResidual {
public:
    AttentionResidual(int hidden_size);
    Tensor forward(const Tensor& current_output, const Tensor& earlier_layer_output);
private:
    Linear gate_proj_;
    int hidden_size_;
};

AttentionResidual::AttentionResidual(int hidden_size) 
    : gate_proj_(hidden_size * 2, hidden_size), hidden_size_(hidden_size) {}

Tensor AttentionResidual::forward(const Tensor& current_output, const Tensor& earlier_layer_output) {
    int64_t B = current_output.dim(0);
    int64_t S = current_output.dim(1);
    int64_t D = hidden_size_;
    
    Tensor concat(Shape{B, S, D * 2}, DType::F32);
    const float* cd = current_output.data<float>();
    const float* ed = earlier_layer_output.data<float>();
    float* ccd = concat.data<float>();
    
    for (int64_t b = 0; b < B; b++) {
        for (int64_t s = 0; s < S; s++) {
            for (int64_t d = 0; d < D; d++) {
                ccd[(b * S + s) * (D * 2) + d] = cd[(b * S + s) * D + d];
                ccd[(b * S + s) * (D * 2) + D + d] = ed[(b * S + s) * D + d];
            }
        }
    }
    
    Tensor gate = gate_proj_.forward(concat.reshape(Shape{B * S, D * 2}));
    Tensor out(Shape{B, S, D}, DType::F32);
    const float* gd = gate.data<float>();
    float* od = out.data<float>();
    
    for (int64_t i = 0; i < B * S * D; i++) {
        float g = 1.0f / (1.0f + std::exp(-gd[i]));
        od[i] = cd[i] * (1.0f - g) + ed[i] * g;
    }
    
    return out;
}

// Retired dead code (owner purge 2026-09-07):
// - LatentKVAttention / GatedDeltaAttention / TranscenderDeltaAttention:
//   RETIRED — owner purge 2026-09-07, zero call sites outside their own tests.
// - FP8_E4M3 / FP8_E5M2 / fp8_matmul (local scalar triple-loop copies):
//   RETIRED — superseded by math::fp8_gemm (src/math) + engines::fp8_*
//   (src/codec/quant_engines_fp.cpp), wired via linear_forward_fp8 above.
//   Had zero call sites tree-wide; removal is behavior-preserving.
// - MultiTokenPredictionHead (transformer.cpp-local duplicate): RETIRED —
//   superseded by DenseModel::mtp_heads + mtp_forward/mtp_loss
//   (src/model/model.cpp) wired into Trainer::micro_step (L051).
//   Had zero call sites tree-wide; removal is behavior-preserving.
// (AttentionResidual below is dead but out of Wave-6 scope; see ledger gap.)

} // namespace quant
