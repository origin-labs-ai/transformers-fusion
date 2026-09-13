// Phase 5-F6 split of src/inference/inference_opt.cpp (verbatim move, no behavior change).
// This file: CompressedKVCache (D4) + decode_quant helper + PrefixCache (D5)
// + flash_decoding (D7). Declarations stay in include/quant/inference_opt.h.
#include "quant/inference_opt.h"
#include "quant/math.h"
#include "quant/int8_quant.h"
#include "quant/random.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <random>
#include <sstream>
#include <cstdio>
namespace quant {

// ===========================================================================
// D4: Compressed KV cache — QUANT4 QUANT encoding
// ===========================================================================
CompressedKVCache::CompressedKVCache(int64_t max_seq, int64_t n_layers, int64_t head_dim)
    : max_seq_(max_seq), n_layers_(n_layers), head_dim_(head_dim) {
    k_blocks_.resize(n_layers);
    v_blocks_.resize(n_layers);
    for (int64_t l = 0; l < n_layers; l++) {
        k_blocks_[l].resize(max_seq);
        v_blocks_[l].resize(max_seq);
    }
}

void CompressedKVCache::append(int layer, const Tensor& k, const Tensor& v) {
    if (seq_len_ >= max_seq_) return;
    int64_t n = k.numel();
    k_blocks_[layer][seq_len_].k_data.resize(n);
    v_blocks_[layer][seq_len_].v_data.resize(n);
    const float* kd = k.data<float>();
    const float* vd = v.data<float>();
    for (int64_t i = 0; i < n; i++) {
        // QUANT4 QUANT: 2-bit per element, packed 4 per byte
        // 00 = -1, 01 = 0, 10 = +1, 11 = unused
        // Use threshold of 0.1 * max_abs to determine zero
        int8_t k_ter = (kd[i] > 0.1f) ? 1 : ((kd[i] < -0.1f) ? -1 : 0);
        int8_t v_ter = (vd[i] > 0.1f) ? 1 : ((vd[i] < -0.1f) ? -1 : 0);
        // Pack 4 QUANT values per byte (2 bits each)
        size_t byte_idx = (size_t)i / 4;
        size_t bit_off = ((size_t)i % 4) * 2;
        if (byte_idx >= k_blocks_[layer][seq_len_].k_data.size()) {
            k_blocks_[layer][seq_len_].k_data.resize(byte_idx + 1);
            v_blocks_[layer][seq_len_].v_data.resize(byte_idx + 1);
        }
        uint8_t k_val = (uint8_t)((k_ter + 1) & 0x03); // -1→0, 0→1, +1→2
        uint8_t v_val = (uint8_t)((v_ter + 1) & 0x03);
        if (bit_off == 0) {
            k_blocks_[layer][seq_len_].k_data[byte_idx] = k_val;
            v_blocks_[layer][seq_len_].v_data[byte_idx] = v_val;
        } else {
            k_blocks_[layer][seq_len_].k_data[byte_idx] |= k_val << bit_off;
            v_blocks_[layer][seq_len_].v_data[byte_idx] |= v_val << bit_off;
        }
    }
    // Trim padded bytes
    size_t needed = ((size_t)n + 3) / 4;
    k_blocks_[layer][seq_len_].k_data.resize(needed);
    v_blocks_[layer][seq_len_].v_data.resize(needed);
    seq_len_++;
}

static float decode_quant(uint8_t packed, size_t idx) {
    size_t bit_off = (idx % 4) * 2;
    uint8_t val = (packed >> bit_off) & 0x03;
    return (val == 0) ? -1.0f : ((val == 2) ? 1.0f : 0.0f);
}

Tensor CompressedKVCache::get_k(int layer, int64_t pos) const {
    if (pos >= seq_len_ || layer >= n_layers_) return Tensor();
    size_t n = k_blocks_[layer][pos].k_data.size() * 4;
    Tensor out({(int64_t)n});
    float* od = out.data<float>();
    for (size_t i = 0; i < n; i++) {
        od[i] = decode_quant(k_blocks_[layer][pos].k_data[i / 4], i);
    }
    return out;
}

Tensor CompressedKVCache::get_v(int layer, int64_t pos) const {
    if (pos >= seq_len_ || layer >= n_layers_) return Tensor();
    size_t n = v_blocks_[layer][pos].v_data.size() * 4;
    Tensor out({(int64_t)n});
    float* od = out.data<float>();
    for (size_t i = 0; i < n; i++) {
        od[i] = decode_quant(v_blocks_[layer][pos].v_data[i / 4], i);
    }
    return out;
}

void CompressedKVCache::clear() {
    seq_len_ = 0;
}

// ===========================================================================
// D5: Prefix caching — share KV cache across requests with common prefix
// ===========================================================================
PrefixCache::PrefixCache(int64_t layer_count) : layers_(layer_count) {}

int64_t PrefixCache::match_prefix(const std::vector<int>& tokens) {
    int64_t best_len = -1;
    for (size_t i = 0; i < entries_.size(); i++) {
        auto& entry = entries_[i];
        int64_t match = 0;
        size_t min_len = std::min(tokens.size(), entry.prefix.size());
        for (size_t j = 0; j < min_len; j++) {
            if (tokens[j] == entry.prefix[j]) match++;
            else break;
        }
        if (match > best_len) best_len = match;
    }
    return best_len;
}

void PrefixCache::store(const std::vector<int>& tokens, int64_t cache_id,
                        int64_t max_seq_len, int64_t num_heads, int64_t head_dim) {
    KVCache cache((int)layers_, max_seq_len, num_heads, head_dim);
    if (cache_id >= 0 && cache_id < (int64_t)entries_.size()) {
        PrefixEntry& e = entries_[(size_t)cache_id];
        e.prefix = tokens;
        e.cache = std::make_unique<KVCache>(std::move(cache));
    } else {
        entries_.push_back(PrefixEntry{tokens, std::make_unique<KVCache>(std::move(cache))});
    }
}

KVCache* PrefixCache::get_cache(int64_t id, int64_t layer) {
    if (id >= 0 && id < (int64_t)entries_.size()) {
        return entries_[(size_t)id].cache.get();
    }
    return nullptr;
}

// ===========================================================================
// D7: Flash decoding — tiled online softmax reduction for multi-turn
// ===========================================================================
Tensor flash_decoding(const Tensor& Q, const Tensor& K, const Tensor& V, int64_t block_size) {
    int64_t B = Q.dim(0), H = Q.dim(1), N = Q.dim(2), D = Q.dim(3);
    Tensor out({B, H, N, D});
    out.zero_();
    const float* q = Q.data<float>();
    const float* k = K.data<float>();
    const float* v = V.data<float>();
    float* o = out.data<float>();
    float scale = 1.0f / std::sqrt((float)D);

    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            int64_t bh_offset = (b * H + h) * N * D;
            const float* q_bh = q + bh_offset;
            const float* k_bh = k + bh_offset;
            const float* v_bh = v + bh_offset;
            float* o_bh = o + bh_offset;

            std::vector<float> row_max((size_t)N, -INFINITY);
            std::vector<float> row_sum((size_t)N, 0.0f);

            for (int64_t blk_start = 0; blk_start < N; blk_start += block_size) {
                int64_t blk_end = std::min(blk_start + block_size, N);

                for (int64_t i = 0; i < N; i++) {
                    float m_prev = row_max[(size_t)i];
                    float m_new = m_prev;

                    for (int64_t j = blk_start; j < blk_end; j++) {
                        float dot = 0;
                        for (int64_t d = 0; d < D; d++)
                            dot += q_bh[i * D + d] * k_bh[j * D + d];
                        float score = dot * scale;
                        m_new = std::max(m_new, score);
                    }

                    float rescale = (m_prev != -INFINITY) ? std::exp(m_prev - m_new) : 0.0f;
                    for (int64_t d = 0; d < D; d++)
                        o_bh[i * D + d] *= rescale;
                    row_sum[(size_t)i] *= rescale;

                    for (int64_t j = blk_start; j < blk_end; j++) {
                        float dot = 0;
                        for (int64_t d = 0; d < D; d++)
                            dot += q_bh[i * D + d] * k_bh[j * D + d];
                        float score = dot * scale;
                        float e = std::exp(score - m_new);
                        row_sum[(size_t)i] += e;
                        for (int64_t d = 0; d < D; d++)
                            o_bh[i * D + d] += e * v_bh[j * D + d];
                    }

                    row_max[(size_t)i] = m_new;
                }
            }

            for (int64_t i = 0; i < N; i++) {
                float inv = 1.0f / (row_sum[(size_t)i] + 1e-10f);
                for (int64_t d = 0; d < D; d++)
                    o_bh[i * D + d] *= inv;
            }
        }
    }
    return out;
}

} // namespace quant
