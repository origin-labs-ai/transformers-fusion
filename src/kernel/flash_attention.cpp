#include "quant/flash_attention.h"
#include "quant/math.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

#if defined(QUANT_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace quant {

FlashAttention::FlashAttention(const FlashAttentionConfig& cfg) : cfg_(cfg) {}

static inline float dot_product_avx2(const float* a, const float* b, int64_t d) {
#if defined(QUANT_AVX2) || defined(__AVX2__)
    __m256 sumv = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= d; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sumv = _mm256_fmadd_ps(va, vb, sumv);
    }
    float hsum[8];
    _mm256_storeu_ps(hsum, sumv);
    float dot = hsum[0] + hsum[1] + hsum[2] + hsum[3]
              + hsum[4] + hsum[5] + hsum[6] + hsum[7];
    for (; i < d; i++) dot += a[i] * b[i];
    return dot;
#else
    float dot = 0;
    for (int64_t i = 0; i < d; i++) dot += a[i] * b[i];
    return dot;
#endif
}

Tensor flash_attention_forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                               const Tensor& mask, float dropout_p, bool causal) {
    static thread_local std::mt19937 drop_rng(std::random_device{}());
    static thread_local std::uniform_real_distribution<float> drop_dist(0.0f, 1.0f);

    int64_t B = Q.dim(0), H = Q.dim(1), N = Q.dim(2), D = Q.dim(3);
    // BUGFIX: guard head_dim==0 / seq==0 (and non-positive batch/heads):
    // sqrt(0) scale would be +INF and the output/normalize math would
    // divide by zero. Return an empty tensor instead of NaN-poisoning.
    if (B <= 0 || H <= 0 || N <= 0 || D <= 0) return Tensor();
    // BUGFIX: guard sqrt of negative/zero dim — D is checked > 0 above,
    // and the float cast of a huge dim could still overflow to +INF.
    float scale = 1.0f / std::sqrt((float)D);
    if (!std::isfinite(scale) || scale <= 0.0f) return Tensor();

    // BUGFIX: sanitize dropout_p (NaN / out-of-range): NaN compares false
    // everywhere and INF would corrupt drop_keep; clamp to valid [0,1).
    if (!std::isfinite(dropout_p) || dropout_p < 0.0f) dropout_p = 0.0f;
    if (dropout_p >= 1.0f) dropout_p = 0.9999f;

    float drop_keep = 1.0f - dropout_p;
    bool do_drop = (dropout_p > 0.0f);

    Tensor output({B, H, N, D});
    output.zero_();
    float* out = output.data<float>();
    const float* q = Q.data<float>();
    const float* k = K.data<float>();
    const float* v = V.data<float>();
    const float* m = mask.data<float>();

    // Block size chosen so that 2 * block * D * 4 bytes fits in ~32KB L1 cache
    // For D=64: 2*32*64*4 = 16KB — fits in L1 comfortably
    int64_t block = (D <= 64) ? 32 : 16;
    std::vector<float> row_max(B * H * N, -INFINITY);
    std::vector<float> row_sum(B * H * N, 0.0f);

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            int64_t bh_off = (b * H + h) * N * D;
            const float* q_ptr = q + (b * H + h) * N * D;
            const float* k_ptr = k + (b * H + h) * N * D;
            const float* v_ptr = v + (b * H + h) * N * D;

            for (int64_t jb = 0; jb < N; jb += block) {
                int64_t j_end = std::min(jb + block, N);
                int64_t k_block = j_end - jb;

                std::vector<float> k_block_data(k_block * D);
                std::vector<float> v_block_data(k_block * D);
                for (int64_t j = 0; j < k_block; ++j) {
                    std::memcpy(&k_block_data[j * D], k_ptr + (jb + j) * D, D * sizeof(float));
                    std::memcpy(&v_block_data[j * D], v_ptr + (jb + j) * D, D * sizeof(float));
                }

                for (int64_t i = 0; i < N; ++i) {
                    if (causal && i < jb) continue;

                    int64_t idx = b * H * N + h * N + i;
                    float rm = row_max[idx];
                    float rs = row_sum[idx];
                    float* o_row = out + bh_off + i * D;

                    float qk_max = -INFINITY;
                    std::vector<float> scores(k_block);
                    for (int64_t j = 0; j < k_block; ++j) {
                        float dot = dot_product_avx2(q_ptr + i * D, k_block_data.data() + j * D, D);
                        // BUGFIX: NaN propagation guard — a NaN q/k element
                        // (or INF*0) would make `dot` NaN and then poison
                        // qk_max/new_rm/exp; collapse it to -INF (zero weight).
                        if (!std::isfinite(dot)) dot = -INFINITY;
                        float masked = dot * scale;
                        // BUGFIX: NaN mask guard — NaN mask entries would
                        // propagate NaN into scores; treat as -INF (masked).
                        if (m) {
                            float mv = m[i * N + (jb + j)];
                            masked += std::isfinite(mv) ? mv : -INFINITY;
                        }
                        if (causal && (jb + j) > i) masked = -INFINITY;
                        if (do_drop) {
                            // Element-wise attention dropout on the raw
                            // scores: one Bernoulli draw per (i, j) element.
                            // Dropped positions become -INFINITY (zero
                            // attention weight after softmax); kept positions
                            // are rescaled by 1/(1-p) in log space so the
                            // expected attention weights are preserved
                            // (equivalent to dropout applied after softmax).
                            if (drop_dist(drop_rng) < dropout_p) {
                                masked = -INFINITY;
                            } else if (drop_keep > 0.0f) {
                                masked += std::log(1.0f / drop_keep);
                            }
                        }
                        scores[j] = masked;
                        if (masked > qk_max) qk_max = masked;
                    }

                    if (!std::isfinite(qk_max)) {
                        // Fully masked key block (every key is -INFINITY, e.g.
                        // a fully masked/causal- or dropout-masked row): skip
                        // the exp/softmax accumulation for this block so that
                        // exp(-INF - -INF) = exp(NaN) can never poison
                        // row_sum/output. The output row was zero-initialized,
                        // so a fully masked row simply stays zero.
                        continue;
                    }

                    float new_rm = std::max(rm, qk_max);
                    float exp_diff = rm != -INFINITY ? std::exp(rm - new_rm) : 0.0f;
                    rs *= exp_diff;

#if defined(QUANT_AVX2) || defined(__AVX2__)
                    {
                        __m256 edv = _mm256_set1_ps(exp_diff);
                        int64_t d = 0;
                        for (; d + 8 <= D; d += 8) {
                            __m256 ov = _mm256_loadu_ps(o_row + d);
                            _mm256_storeu_ps(o_row + d, _mm256_mul_ps(ov, edv));
                        }
                        for (; d < D; d++) o_row[d] *= exp_diff;
                    }
#else
                    for (int64_t d = 0; d < D; ++d)
                        o_row[d] *= exp_diff;
#endif

                    float sum_exp = 0;
                    for (int64_t j = 0; j < k_block; ++j) {
                        // BUGFIX: max-subtracted softmax — scores[j]-new_rm
                        // is <= 0 by construction (new_rm >= qk_max >= every
                        // finite score), so exp() can never overflow; the
                        // fully-masked (-INF) block is already skipped above.
                        float e = std::exp(scores[j] - new_rm);
                        // BUGFIX: NaN/INF guard on exp result and on V —
                        // a NaN V element must not poison the whole row.
                        if (!std::isfinite(e)) e = 0.0f;
                        sum_exp += e;
#if defined(QUANT_AVX2) || defined(__AVX2__)
                        {
                            __m256 ev = _mm256_set1_ps(e);
                            int64_t d = 0;
                            for (; d + 8 <= D; d += 8) {
                                __m256 ov = _mm256_loadu_ps(o_row + d);
                                __m256 vv = _mm256_loadu_ps(v_block_data.data() + j * D + d);
                                _mm256_storeu_ps(o_row + d, _mm256_fmadd_ps(ev, vv, ov));
                            }
                            for (; d < D; d++)
                                o_row[d] += e * v_block_data[j * D + d];
                        }
#else
                        for (int64_t d = 0; d < D; ++d)
                            o_row[d] += e * v_block_data[j * D + d];
#endif
                    }
                    rs += sum_exp;
                    row_max[idx] = new_rm;
                    row_sum[idx] = rs;
                }
            }

            for (int64_t i = 0; i < N; ++i) {
                float rs_i = row_sum[b * H * N + h * N + i];
                // BUGFIX: div-by-zero/NaN guard on the final normalize:
                // a fully-masked row has rs==0 (1/eps would blow it up to
                // 1e10 * garbage); NaN rs must not propagate either.
                // Such rows were zero-initialized, so leave them as zero.
                if (!(rs_i > 0.0f) || !std::isfinite(rs_i)) continue;
                float inv_sum = 1.0f / rs_i;
                float* o_row = out + bh_off + i * D;
#if defined(QUANT_AVX2) || defined(__AVX2__)
                {
                    __m256 iv = _mm256_set1_ps(inv_sum);
                    int64_t d = 0;
                    for (; d + 8 <= D; d += 8) {
                        __m256 ov = _mm256_loadu_ps(o_row + d);
                        _mm256_storeu_ps(o_row + d, _mm256_mul_ps(ov, iv));
                    }
                    for (; d < D; d++) o_row[d] *= inv_sum;
                }
#else
                for (int64_t d = 0; d < D; ++d)
                    o_row[d] *= inv_sum;
#endif
            }
        }
    }
    return output;
}

Tensor FlashAttention::forward(const Tensor& Q, const Tensor& K,
                                const Tensor& V, const Tensor& mask) {
    return flash_attention_forward(Q, K, V, mask, cfg_.dropout, cfg_.causal);
}

} // namespace quant
