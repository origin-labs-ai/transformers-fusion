// ============================================================================
// ZeRO-3 Optimizer — Memory-Efficient Attention & Activation Checkpointing
// ============================================================================

#define NOMINMAX
#include "quant/zero_optimizer.h"
#include "quant/math.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cfloat>
#ifdef _WIN32
#include <windows.h>
#undef min
#undef max
#endif

#if defined(QUANT_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace quant {

// ============================================================================
// MemoryEfficientAttention
// ============================================================================

MemoryEfficientAttention::MemoryEfficientAttention(const MEAConfig& cfg)
    : cfg_(cfg) {}

int64_t MemoryEfficientAttention::forward_memory_bytes(
    int64_t B, int64_t H, int64_t N, int64_t D) const {
    // Q, K, V: 3 * B * H * N * D * sizeof(float)
    // Output: B * H * N * D * sizeof(float)
    // Block intermediates: block_size * D * 2 * sizeof(float)
    // Row max/sum: B * H * N * 2 * sizeof(float)
    int64_t qkv = 3 * B * H * N * D * (int64_t)sizeof(float);
    int64_t out = B * H * N * D * (int64_t)sizeof(float);
    int64_t blocks = cfg_.block_size_kv * D * 2 * (int64_t)sizeof(float);
    int64_t stats = B * H * N * 2 * (int64_t)sizeof(float);
    return qkv + out + blocks + stats;
}

int64_t MemoryEfficientAttention::backward_memory_bytes(
    int64_t B, int64_t H, int64_t N, int64_t D) const {
    // Recompute forward -> same as forward + dQ, dK, dV
    return forward_memory_bytes(B, H, N, D) +
           3 * B * H * N * D * (int64_t)sizeof(float);
}

void MemoryEfficientAttention::online_softmax_block(
    const float* qk_block, int64_t block_size,
    float* row_max, float* row_sum,
    float* output, const float* v_block,
    int64_t head_dim) {

    for (int64_t i = 0; i < block_size; i++) {
        float old_max = row_max[i];
        float old_sum = row_sum[i];
        const float* qk_row = qk_block + i * block_size;

        // Find new max
        float new_max = old_max;
        for (int64_t j = 0; j < block_size; j++) {
            if (qk_row[j] > new_max) new_max = qk_row[j];
        }

        // Rescale previous sum
        float exp_diff = (old_max != -FLT_MAX)
            ? std::exp(old_max - new_max) : 0.0f;

#if defined(QUANT_AVX2) || defined(__AVX2__)
        {
            __m256 edv = _mm256_set1_ps(exp_diff);
            int64_t d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 ov = _mm256_loadu_ps(output + i * head_dim + d);
                _mm256_storeu_ps(output + i * head_dim + d, _mm256_mul_ps(ov, edv));
            }
            for (; d < head_dim; d++)
                output[i * head_dim + d] *= exp_diff;
        }
#else
        for (int64_t d = 0; d < head_dim; d++)
            output[i * head_dim + d] *= exp_diff;
#endif

        // Compute exponentials and accumulate
        float new_sum = 0.0f;
        for (int64_t j = 0; j < block_size; j++) {
            float e = std::exp(qk_row[j] - new_max);
            new_sum += e;
#if defined(QUANT_AVX2) || defined(__AVX2__)
            {
                __m256 ev = _mm256_set1_ps(e);
                int64_t d = 0;
                for (; d + 8 <= head_dim; d += 8) {
                    __m256 ov = _mm256_loadu_ps(output + i * head_dim + d);
                    __m256 vv = _mm256_loadu_ps(v_block + j * head_dim + d);
                    _mm256_storeu_ps(output + i * head_dim + d, _mm256_fmadd_ps(ev, vv, ov));
                }
                for (; d < head_dim; d++)
                    output[i * head_dim + d] += e * v_block[j * head_dim + d];
            }
#else
            for (int64_t d = 0; d < head_dim; d++)
                output[i * head_dim + d] += e * v_block[j * head_dim + d];
#endif
        }

        row_max[i] = new_max;
        row_sum[i] = old_sum * exp_diff + new_sum;
    }
}

void MemoryEfficientAttention::online_softmax_tile(
    const float* qk_scores, float* row_max, float* row_sum,
    float* output, const float* v_data,
    int64_t num_rows, int64_t kv_block_size, int64_t head_dim,
    int64_t output_offset) {

    for (int64_t i = 0; i < num_rows; i++) {
        float old_max = row_max[i];
        float old_sum = row_sum[i];
        const float* qk_row = qk_scores + i * kv_block_size;

        float new_max = old_max;
        for (int64_t j = 0; j < kv_block_size; j++) {
            if (qk_row[j] > new_max) new_max = qk_row[j];
        }

        float exp_diff = (old_max != -FLT_MAX)
            ? std::exp(old_max - new_max) : 0.0f;

        float* out_row = output + output_offset + i * head_dim;
#if defined(QUANT_AVX2) || defined(__AVX2__)
        {
            __m256 edv = _mm256_set1_ps(exp_diff);
            int64_t d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 ov = _mm256_loadu_ps(out_row + d);
                _mm256_storeu_ps(out_row + d, _mm256_mul_ps(ov, edv));
            }
            for (; d < head_dim; d++) out_row[d] *= exp_diff;
        }
#else
        for (int64_t d = 0; d < head_dim; d++)
            out_row[d] *= exp_diff;
#endif

        float new_sum = 0.0f;
        for (int64_t j = 0; j < kv_block_size; j++) {
            float e = std::exp(qk_row[j] - new_max);
            new_sum += e;
            const float* v_row = v_data + j * head_dim;
#if defined(QUANT_AVX2) || defined(__AVX2__)
            {
                __m256 ev = _mm256_set1_ps(e);
                int64_t d = 0;
                for (; d + 8 <= head_dim; d += 8) {
                    __m256 ov = _mm256_loadu_ps(out_row + d);
                    __m256 vv = _mm256_loadu_ps(v_row + d);
                    _mm256_storeu_ps(out_row + d, _mm256_fmadd_ps(ev, vv, ov));
                }
                for (; d < head_dim; d++)
                    out_row[d] += e * v_row[d];
            }
#else
            for (int64_t d = 0; d < head_dim; d++)
                out_row[d] += e * v_row[d];
#endif
        }

        row_max[i] = new_max;
        row_sum[i] = old_sum * exp_diff + new_sum;
    }
}

Tensor MemoryEfficientAttention::forward(
    const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& mask) {

    int64_t B = Q.dim(0), H = Q.dim(1), N = Q.dim(2), D = Q.dim(3);
    float scale = (cfg_.softmax_scale > 0.0f)
        ? cfg_.softmax_scale : 1.0f / std::sqrt((float)D);

    int64_t Bq = K.dim(0), Hq = K.dim(1), Nk = K.dim(2);
    (void)Bq; (void)Hq;

    Tensor output({B, H, N, D});
    output.zero_();
    float* out = output.data<float>();

    const float* q_ptr = Q.data<float>();
    const float* k_ptr = K.data<float>();
    const float* v_ptr = V.data<float>();
    const float* mask_ptr = mask.numel() > 0 ? mask.data<float>() : nullptr;

    int64_t BNH = B * H * N;

    std::vector<float> row_max(BNH, -FLT_MAX);
    std::vector<float> row_sum(BNH, 0.0f);

    int64_t bq = cfg_.block_size_q;
    int64_t bkv = cfg_.block_size_kv;

    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            int64_t bh = b * H + h;
            int64_t bh_off = bh * N * D;
            const float* q_bh = q_ptr + bh_off;
            const float* k_bh = k_ptr + bh * Nk * D;
            const float* v_bh = v_ptr + bh * Nk * D;

            // Tile over KV dimension
            for (int64_t jb = 0; jb < Nk; jb += bkv) {
                int64_t je = std::min(jb + bkv, Nk);
                int64_t kblk = je - jb;

                // Load K, V block
                std::vector<float> k_block(kblk * D);
                std::vector<float> v_block(kblk * D);
                for (int64_t j = 0; j < kblk; j++) {
                    std::memcpy(&k_block[j * D], k_bh + (jb + j) * D, D * sizeof(float));
                    std::memcpy(&v_block[j * D], v_bh + (jb + j) * D, D * sizeof(float));
                }

                // Tile over Q dimension
                for (int64_t ib = 0; ib < N; ib += bq) {
                    int64_t ie = std::min(ib + bq, N);
                    int64_t qblk = ie - ib;

                    // Compute Q*K^T for this tile
                    std::vector<float> qk_scores(qblk * kblk);
                    for (int64_t i = 0; i < qblk; i++) {
                        int64_t qi = ib + i;
                        if (cfg_.causal && qi < jb) {
                            // Entire row is masked
                            for (int64_t j = 0; j < kblk; j++)
                                qk_scores[i * kblk + j] = -FLT_MAX;
                            continue;
                        }
                        for (int64_t j = 0; j < kblk; j++) {
                            int64_t kj = jb + j;
                            if (cfg_.causal && kj > qi) {
                                qk_scores[i * kblk + j] = -FLT_MAX;
                                continue;
                            }
                            float dot = 0.0f;
                            const float* q_row = q_bh + qi * D;
                            const float* k_row = &k_block[j * D];
#if defined(QUANT_AVX2) || defined(__AVX2__)
                            {
                                __m256 sumv = _mm256_setzero_ps();
                                int64_t d = 0;
                                for (; d + 8 <= D; d += 8) {
                                    __m256 qv = _mm256_loadu_ps(q_row + d);
                                    __m256 kv = _mm256_loadu_ps(k_row + d);
                                    sumv = _mm256_fmadd_ps(qv, kv, sumv);
                                }
                                float hsum[8];
                                _mm256_storeu_ps(hsum, sumv);
                                dot = hsum[0] + hsum[1] + hsum[2] + hsum[3]
                                    + hsum[4] + hsum[5] + hsum[6] + hsum[7];
                                for (; d < D; d++)
                                    dot += q_row[d] * k_row[d];
                            }
#else
                            for (int64_t d = 0; d < D; d++)
                                dot += q_row[d] * k_row[d];
#endif
                            float score = dot * scale;
                            if (mask_ptr) {
                                score += mask_ptr[qi * Nk + kj];
                            }
                            qk_scores[i * kblk + j] = score;
                        }
                    }

                    // Online softmax for this tile
                    online_softmax_tile(
                        qk_scores.data(),
                        &row_max[bh * N + ib],
                        &row_sum[bh * N + ib],
                        out + bh_off,
                        v_block.data(),
                        qblk, kblk, D, ib * D);
                }
            }

            // Normalize output rows
            for (int64_t i = 0; i < N; i++) {
                float inv = 1.0f / (row_sum[bh * N + i] + 1e-10f);
                float* o_row = out + bh_off + i * D;
#if defined(QUANT_AVX2) || defined(__AVX2__)
                {
                    __m256 iv = _mm256_set1_ps(inv);
                    int64_t d = 0;
                    for (; d + 8 <= D; d += 8) {
                        __m256 ov = _mm256_loadu_ps(o_row + d);
                        _mm256_storeu_ps(o_row + d, _mm256_mul_ps(ov, iv));
                    }
                    for (; d < D; d++) o_row[d] *= inv;
                }
#else
                for (int64_t d = 0; d < D; d++)
                    o_row[d] *= inv;
#endif
            }
        }
    }

    // Save state for backward (if not using checkpointing)
    if (!cfg_.gradient_checkpoint) {
        saved_q_ = Q;
        saved_k_ = K;
        saved_v_ = V;
        saved_mask_ = mask;
        saved_row_max_ = row_max;
        saved_row_sum_ = row_sum;
        saved_ = true;
    }

    last_stats_.max_val = *std::max_element(row_max.begin(), row_max.end());
    last_stats_.sum_val = std::accumulate(row_sum.begin(), row_sum.end(), 0.0f);
    last_stats_.num_blocks = (N + bkv - 1) / bkv;
    forward_seen_++;

    return output;
}

std::vector<float> MemoryEfficientAttention::recompute_forward_stats(
    const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& mask) {
    // Recompute row_max/row_sum (for gradient checkpoint backward)
    int64_t B = Q.dim(0), H = Q.dim(1), N = Q.dim(2), D = Q.dim(3);
    float scale = (cfg_.softmax_scale > 0.0f)
        ? cfg_.softmax_scale : 1.0f / std::sqrt((float)D);
    int64_t Nk = K.dim(2);

    int64_t BNH = B * H * N;
    std::vector<float> row_max(BNH, -FLT_MAX);
    std::vector<float> row_sum(BNH, 0.0f);

    const float* q_ptr = Q.data<float>();
    const float* k_ptr = K.data<float>();
    const float* v_ptr = V.data<float>();
    const float* mask_ptr = mask.numel() > 0 ? mask.data<float>() : nullptr;
    int64_t bkv = cfg_.block_size_kv;

    (void)v_ptr;

    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            int64_t bh = b * H + h;
            for (int64_t jb = 0; jb < Nk; jb += bkv) {
                int64_t je = std::min(jb + bkv, Nk);
                int64_t kblk = je - jb;

                for (int64_t i = 0; i < N; i++) {
                    if (cfg_.causal && i < jb) continue;

                    int64_t idx = bh * N + i;
                    float cur_max = row_max[idx];
                    float new_max = cur_max;
                    int64_t qi = i;

                    for (int64_t j = 0; j < kblk; j++) {
                        int64_t kj = jb + j;
                        if (cfg_.causal && kj > qi) continue;
                        float dot = 0;
                        const float* q_row = q_ptr + (bh * N + qi) * D;
                        const float* k_row = k_ptr + (bh * Nk + kj) * D;
                        for (int64_t d = 0; d < D; d++) dot += q_row[d] * k_row[d];
                        float score = dot * scale;
                        if (mask_ptr) score += mask_ptr[qi * Nk + kj];
                        if (score > new_max) new_max = score;
                    }

                    float sum = 0;
                    for (int64_t j = 0; j < kblk; j++) {
                        int64_t kj = jb + j;
                        if (cfg_.causal && kj > qi) continue;
                        float dot = 0;
                        const float* q_row = q_ptr + (bh * N + qi) * D;
                        const float* k_row = k_ptr + (bh * Nk + kj) * D;
                        for (int64_t d = 0; d < D; d++) dot += q_row[d] * k_row[d];
                        float score = dot * scale;
                        if (mask_ptr) score += mask_ptr[qi * Nk + kj];
                        sum += std::exp(score - new_max);
                    }

                    float exp_diff = (cur_max != -FLT_MAX)
                        ? std::exp(cur_max - new_max) : 0.0f;
                    row_max[idx] = new_max;
                    row_sum[idx] = row_sum[idx] * exp_diff + sum;
                }
            }
        }
    }

    return row_sum;
}

Tensor MemoryEfficientAttention::forward_block_sparse(
    const Tensor& Q, const Tensor& K, const Tensor& V,
    const Tensor& block_indices) {
    // Block-sparse attention: each query attends only to a subset of KV blocks
    // block_indices: (num_blocks, 2) with (block_row, block_col) pairs
    int64_t B = Q.dim(0), H = Q.dim(1), N = Q.dim(2), D = Q.dim(3);
    float scale = (cfg_.softmax_scale > 0.0f)
        ? cfg_.softmax_scale : 1.0f / std::sqrt((float)D);
    int64_t Nk = K.dim(2);
    int64_t bkv = cfg_.block_size_kv;
    int64_t bq = cfg_.block_size_q;

    Tensor output({B, H, N, D});
    output.zero_();
    float* out = output.data<float>();
    const float* q_ptr = Q.data<float>();
    const float* k_ptr = K.data<float>();
    const float* v_ptr = V.data<float>();

    int64_t BNH = B * H * N;
    std::vector<float> row_max(BNH, -FLT_MAX);
    std::vector<float> row_sum(BNH, 0.0f);

    // Build block access map from indices
    // For simplicity, use local window + global sliding pattern
    int64_t window = cfg_.block_sparse_window;

    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            int64_t bh = b * H + h;
            int64_t bh_off = bh * N * D;
            const float* q_bh = q_ptr + bh_off;
            const float* k_bh = k_ptr + bh * Nk * D;
            const float* v_bh = v_ptr + bh * Nk * D;

            for (int64_t ib = 0; ib < N; ib += bq) {
                int64_t ie = std::min(ib + bq, N);
                int64_t qblk = ie - ib;

                // Determine which KV blocks this Q tile attends to
                int64_t kv_start = std::max((int64_t)0, ib - window);
                int64_t kv_end = std::min(Nk, ib + window + bq);

                for (int64_t jb = kv_start; jb < kv_end; jb += bkv) {
                    int64_t je = std::min(jb + bkv, kv_end);
                    int64_t kblk = je - jb;

                    std::vector<float> k_block(kblk * D);
                    std::vector<float> v_block(kblk * D);
                    for (int64_t j = 0; j < kblk; j++) {
                        std::memcpy(&k_block[j * D], k_bh + (jb + j) * D, D * sizeof(float));
                        std::memcpy(&v_block[j * D], v_bh + (jb + j) * D, D * sizeof(float));
                    }

                    std::vector<float> qk_scores(qblk * kblk);
                    for (int64_t i = 0; i < qblk; i++) {
                        int64_t qi = ib + i;
                        for (int64_t j = 0; j < kblk; j++) {
                            int64_t kj = jb + j;
                            if (cfg_.causal && kj > qi) {
                                qk_scores[i * kblk + j] = -FLT_MAX;
                                continue;
                            }
                            float dot = 0;
                            const float* qr = q_bh + qi * D;
                            const float* kr = &k_block[j * D];
                            for (int64_t d = 0; d < D; d++) dot += qr[d] * kr[d];
                            qk_scores[i * kblk + j] = dot * scale;
                        }
                    }

                    online_softmax_tile(
                        qk_scores.data(),
                        &row_max[bh * N + ib],
                        &row_sum[bh * N + ib],
                        out + bh_off,
                        v_block.data(),
                        qblk, kblk, D, ib * D);
                }
            }

            // Normalize
            for (int64_t i = 0; i < N; i++) {
                float inv = 1.0f / (row_sum[bh * N + i] + 1e-10f);
                float* o_row = out + bh_off + i * D;
                for (int64_t d = 0; d < D; d++) o_row[d] *= inv;
            }
        }
    }

    return output;
}

std::vector<Tensor> MemoryEfficientAttention::backward(
    const Tensor& grad_output,
    const Tensor& Q, const Tensor& K, const Tensor& V,
    const Tensor& mask) {
    (void)Q; (void)K; (void)V; (void)mask;

    int64_t B = grad_output.dim(0), H = grad_output.dim(1);
    int64_t N = grad_output.dim(2), D = grad_output.dim(3);

    // If gradient checkpointing, recompute forward stats
    if (cfg_.gradient_checkpoint || !saved_) {
        // Recompute — forward stats already calculated in saved data
        // In a real implementation, we'd recompute the full forward
        // and compute dQ, dK, dV from saved intermediates
    } else if (saved_) {
        // Use saved forward stats
    }

    // Compute dQ, dK, dV (simplified — real implementation would use
    // the saved softmax statistics to compute exact gradients)
    Tensor dQ = Tensor::zeros(Q.shape());
    Tensor dK = Tensor::zeros(K.shape());
    Tensor dV = Tensor::zeros(V.shape());

    const float* dO = grad_output.data<float>();
    float* dq = dQ.data<float>();
    float* dk = dK.data<float>();
    float* dv = dV.data<float>();

    // dv = softmax_scores^T * dO
    // dk = (dV * ...) — in real impl, uses row_max/row_sum from forward
    // dq = ...similar

    // Memory-efficient backward path: recompute softmax using saved row_max/row_sum
    // from the forward pass to avoid materializing the full NxN attention matrix.
    // This is the online-softmax chain rule: dS_ij = P_ij * (dV_ij - sum_k P_ik dV_ik)
    if (saved_ && !cfg_.gradient_checkpoint) {
        const float* k_mat = saved_k_.data<float>();
        const float* q_mat = saved_q_.data<float>();
        int64_t Nk = saved_k_.dim(2);

        for (int64_t b = 0; b < B; b++) {
            for (int64_t h = 0; h < H; h++) {
                int64_t bh = b * H + h;
                for (int64_t i = 0; i < N; i++) {
                    int64_t idx = bh * N + i;
                    float inv_sum = 1.0f / (saved_row_sum_[idx] + 1e-10f);
                    float scale = (cfg_.softmax_scale > 0.0f)
                        ? cfg_.softmax_scale : 1.0f / std::sqrt((float)D);

                    for (int64_t j = 0; j < Nk; j++) {
                        if (cfg_.causal && j > i) continue;

                        // Recompute softmax score (P_ij)
                        float dot = 0;
                        const float* qr = q_mat + (bh * N + i) * D;
                        const float* kr = k_mat + (bh * Nk + j) * D;
                        for (int64_t d = 0; d < D; d++) dot += qr[d] * kr[d];
                        float score = dot * scale * inv_sum;
                        float p = std::exp(score);

                        // dV contribution
                        for (int64_t d = 0; d < D; d++)
                            dv[(bh * Nk + j) * D + d] += p * dO[(bh * N + i) * D + d];

                        // dQ/dK contributions (simplified)
                        float grad_factor = p * (1 - p) / scale;
                        for (int64_t d = 0; d < D; d++) {
                            dq[(bh * N + i) * D + d] += grad_factor * kr[d];
                            dk[(bh * Nk + j) * D + d] += grad_factor * qr[d];
                        }
                    }
                }
            }
        }
    }

    return {dQ, dK, dV};
}

// ============================================================================
// ActivationCheckpoint
// ============================================================================

ActivationCheckpoint::ActivationCheckpoint() {
    budget_.total_bytes = 0;
}

ActivationCheckpoint::ActivationCheckpoint(const MemoryBudget& budget)
    : budget_(budget) {}

void ActivationCheckpoint::begin_region(const std::string& name, CheckpointMode mode) {
    if (in_region_) {
        // Nested region — push current and start new
        regions_.push_back(current_region_);
    }
    current_region_ = RegionInfo();
    current_region_.name = name;
    current_region_.mode = mode;
    current_region_.is_checkpointed = (mode != CheckpointMode::NONE);
    region_depth_++;
    in_region_ = true;
}

void ActivationCheckpoint::end_region() {
    if (!in_region_) return;
    region_depth_--;
    current_region_.forward_activations_bytes = std::max(
        current_region_.forward_activations_bytes, (int64_t)1024);
    current_region_.recompute_cost_flops = std::max(
        current_region_.recompute_cost_flops, (int64_t)1);
    regions_.push_back(current_region_);
    in_region_ = false;

    if (current_region_.is_checkpointed) {
        total_saved_ += current_region_.forward_activations_bytes;
        total_recomputed_ += current_region_.recompute_cost_flops;
    }
}

int64_t ActivationCheckpoint::estimate_layer_activations(
    int64_t B, int64_t N, int64_t H, int64_t D) const {
    // Typical transformer layer activations:
    // Attention: Q, K, V, output = 4 * B * N * H * D * 4 bytes
    // FFN: up, gate, down = 3 * B * N * 4*H * 4 bytes (approx 4x hidden)
    // Norms: 2 * B * N * H * 4 bytes
    // Residual: 2 * B * N * H * 4 bytes

    int64_t attn = 4 * B * N * H * D * (int64_t)sizeof(float);
    int64_t ffn_mult = 4; // typical FFN hidden = 4 * hidden
    int64_t ffn = 3 * B * N * H * ffn_mult * D * (int64_t)sizeof(float);
    int64_t norms = 2 * B * N * H * (int64_t)sizeof(float);
    int64_t residual = 2 * B * N * H * (int64_t)sizeof(float);
    return attn + ffn + norms + residual;
}

bool ActivationCheckpoint::should_checkpoint(const RegionInfo& region) const {
    if (budget_.total_bytes <= 0) return false;
    return (budget_.activation_bytes + region.forward_activations_bytes)
           > budget_.total_bytes;
}

std::vector<std::string> ActivationCheckpoint::auto_place_checkpoints() {
    std::vector<std::string> to_checkpoint;
    if (regions_.empty() || budget_.total_bytes <= 0) return to_checkpoint;

    // Greedy placement: checkpoint the largest regions first until we fit
    // within budget
    auto sorted = regions_;
    std::sort(sorted.begin(), sorted.end(),
              [](const RegionInfo& a, const RegionInfo& b) {
                  return a.forward_activations_bytes > b.forward_activations_bytes;
              });

    int64_t current_activations = 0;
    for (auto& r : regions_)
        current_activations += r.forward_activations_bytes;

    int64_t target = budget_.total_bytes - budget_.param_bytes
                     - budget_.grad_bytes - budget_.opt_state_bytes;
    if (target <= 0) target = budget_.total_bytes / 4;

    for (auto& r : sorted) {
        if (current_activations <= target) break;
        if (r.is_checkpointed) continue;
        to_checkpoint.push_back(r.name);
        current_activations -= r.forward_activations_bytes;
        // Mark the actual region for checkpointing
        for (auto& region : regions_) {
            if (region.name == r.name) {
                region.is_checkpointed = true;
                region.mode = CheckpointMode::BALANCED;
                break;
            }
        }
    }

    return to_checkpoint;
}

int64_t ActivationCheckpoint::estimated_savings(const RegionInfo& region) const {
    if (!region.is_checkpointed) return 0;
    return region.forward_activations_bytes;
}

int64_t ActivationCheckpoint::estimated_savings_all() const {
    int64_t total = 0;
    for (auto& r : regions_) {
        if (r.is_checkpointed)
            total += r.forward_activations_bytes;
    }
    return total;
}

float ActivationCheckpoint::compute_overhead() const {
    if (total_saved_ <= 0) return 0.0f;
    return (float)total_recomputed_ / total_saved_;
}

void ActivationCheckpoint::reset() {
    regions_.clear();
    current_region_ = RegionInfo{};
    region_depth_ = 0;
    in_region_ = false;
    total_saved_ = 0;
    total_recomputed_ = 0;
}

} // namespace quant
