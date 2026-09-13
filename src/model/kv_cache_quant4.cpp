#include "quant/kv_cache_quant4.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>

namespace quant {

// ===========================================================================
// QUANT4KVCache — QUANT4-quantized KV cache implementation
// ===========================================================================

QUANT4KVCache::QUANT4KVCache(int num_layers, int64_t max_seq_len, int64_t num_heads,
                           int64_t head_dim, int64_t block_size) {
    init(num_layers, max_seq_len, num_heads, head_dim, block_size);
}

void QUANT4KVCache::init(int num_layers, int64_t max_seq_len, int64_t num_heads,
                         int64_t head_dim, int64_t block_size) {
    num_layers_ = num_layers;
    max_seq_len_ = max_seq_len;
    num_heads_ = num_heads;
    head_dim_ = head_dim;
    block_size_ = block_size;

    if (block_size_ != 32 && block_size_ != 64 && block_size_ != 128)
        block_size_ = 64;

    caches_.resize(num_layers);
    for (int i = 0; i < num_layers; i++) {
        auto& c = caches_[i];
        int64_t total = total_elements();
        int64_t nblocks = total_blocks();
        c.k_indices.resize((size_t)((total + 1) / 2), 0);
        c.v_indices.resize((size_t)((total + 1) / 2), 0);
        c.k_codebooks.resize((size_t)(nblocks * QUANT4_CODEBOOK_SIZE), 0);
        c.v_codebooks.resize((size_t)(nblocks * QUANT4_CODEBOOK_SIZE), 0);
        c.current_pos = 0;
    }
}

int64_t QUANT4KVCache::elements_per_block() const {
    return block_size_;
}

int64_t QUANT4KVCache::indices_per_token() const {
    return num_heads_ * head_dim_;
}

int64_t QUANT4KVCache::total_elements() const {
    return max_seq_len_ * num_heads_ * head_dim_;
}

int64_t QUANT4KVCache::total_blocks() const {
    int64_t total = total_elements();
    return (total + elements_per_block() - 1) / elements_per_block();
}

int64_t QUANT4KVCache::codebook_bytes_per_block() const {
    return QUANT4_CODEBOOK_SIZE * 2; // 16 × FP16
}

int64_t QUANT4KVCache::index_bytes_per_block() const {
    return (elements_per_block() + 1) / 2;
}

int64_t QUANT4KVCache::bytes_per_block() const {
    return index_bytes_per_block() + codebook_bytes_per_block();
}

// ===========================================================================
// Lloyd-Max quantizer for 16 centroids
// ===========================================================================

namespace {

void lloyd_max_16(const float* data, int64_t n, uint16_t* codebook_fp16,
                   uint8_t* indices) {
    if (n == 0) return;

    float min_val = data[0], max_val = data[0];
    for (int64_t i = 1; i < n; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    if (max_val - min_val < 1e-10f) {
        for (int64_t i = 0; i < n; i++) indices[i] = 8;
        for (int i = 0; i < 16; i++)
            codebook_fp16[i] = CodebookQUANT4::float_to_half(min_val);
        return;
    }

    // Initialize centroids uniformly
    float centroids[16];
    for (int i = 0; i < 16; i++) {
        centroids[i] = min_val + (max_val - min_val) * (i + 0.5f) / 16.0f;
    }

    // Lloyd-Max iterations (up to 20)
    float sorted_data[4096];
    int64_t actual_n = (n > 4096) ? 4096 : n;
    std::memcpy(sorted_data, data, (size_t)actual_n * sizeof(float));
    std::sort(sorted_data, sorted_data + actual_n);

    for (int iter = 0; iter < 20; iter++) {
        // Assignment: assign each data point to nearest centroid
        for (int64_t i = 0; i < actual_n; i++) {
            float best_dist = std::fabs(sorted_data[i] - centroids[0]);
            int best_idx = 0;
            for (int c = 1; c < 16; c++) {
                float dist = std::fabs(sorted_data[i] - centroids[c]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = c;
                }
            }
            indices[i] = (uint8_t)best_idx;
        }

        // Update: recompute centroids as cluster means
        float sums[16] = {};
        int counts[16] = {};
        for (int64_t i = 0; i < actual_n; i++) {
            int idx = indices[i];
            sums[idx] += sorted_data[i];
            counts[idx]++;
        }
        for (int c = 0; c < 16; c++) {
            if (counts[c] > 0)
                centroids[c] = sums[c] / counts[c];
        }
    }

    // Now assign all n elements
    for (int64_t i = 0; i < n; i++) {
        float best_dist = std::fabs(data[i] - centroids[0]);
        int best_idx = 0;
        for (int c = 1; c < 16; c++) {
            float dist = std::fabs(data[i] - centroids[c]);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = c;
            }
        }
        indices[i] = (uint8_t)best_idx;
    }

    // Store as FP16
    for (int i = 0; i < 16; i++)
        codebook_fp16[i] = CodebookQUANT4::float_to_half(centroids[i]);
}

void dequantize_16(const uint8_t* indices, const uint16_t* codebook_fp16,
                    float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        uint8_t idx = indices[i] & 0x0F;
        dst[i] = CodebookQUANT4::half_to_float(codebook_fp16[idx]);
    }
}

} // anonymous namespace

void QUANT4KVCache::quantize_block_quant4(const float* src, uint8_t* indices,
                                        uint16_t* codebook_fp16, int64_t n) {
    lloyd_max_16(src, n, codebook_fp16, indices);
}

void QUANT4KVCache::dequantize_block_quant4(const uint8_t* indices,
                                          const uint16_t* codebook_fp16,
                                          float* dst, int64_t n) {
    dequantize_16(indices, codebook_fp16, dst, n);
}

// ===========================================================================
// Pack/unpack 4-bit indices (2 per byte, low nibble first)
// ===========================================================================

namespace {

void pack_indices(const uint8_t* raw, uint8_t* packed, int64_t n) {
    int64_t bytes = (n + 1) / 2;
    for (int64_t i = 0; i < bytes; i++) {
        int64_t idx = i * 2;
        uint8_t lo = (idx < n) ? (raw[idx] & 0x0F) : 0;
        uint8_t hi = ((idx | 1) < n) ? (raw[idx + 1] & 0x0F) : 0;
        packed[i] = lo | (hi << 4);
    }
}

void unpack_indices(const uint8_t* packed, uint8_t* raw, int64_t n) {
    int64_t bytes = (n + 1) / 2;
    for (int64_t i = 0; i < bytes; i++) {
        uint8_t b = packed[i];
        int64_t idx = i * 2;
        if (idx < n)     raw[idx]     = b & 0x0F;
        if (idx + 1 < n) raw[idx + 1] = (b >> 4) & 0x0F;
    }
}

} // anonymous namespace

// ===========================================================================
// Core operations
// ===========================================================================

void QUANT4KVCache::append(int layer, const Tensor& k, const Tensor& v) {
    if (layer >= num_layers_) return;
    auto& c = caches_[layer];

    int64_t seq_len = k.shape().dims[2];
    int64_t h = num_heads_;
    int64_t d = head_dim_;
    int64_t tokens_per_block = elements_per_block();
    int64_t per_token = h * d;

    const float* ksrc = k.data<float>();
    const float* vsrc = v.data<float>();

    int64_t pos = c.current_pos;

    for (int64_t s = 0; s < seq_len && pos + s < max_seq_len_; s++) {
        int64_t flat_start = (pos + s) * per_token;
        int64_t block_idx = flat_start / tokens_per_block;
        int64_t offset_in_block = flat_start % tokens_per_block;

        // For each token, we process all heads concatenated.
        // We quantize in chunks that align with block boundaries.
        int64_t src_offset = s * per_token;

        // Compute how many elements in the current block for this token's data
        int64_t elems_remaining = per_token;
        int64_t src_pos = src_offset;
        int64_t flat_pos = flat_start;

        while (elems_remaining > 0) {
            int64_t cur_block_idx = flat_pos / tokens_per_block;
            int64_t cur_offset = flat_pos % tokens_per_block;
            int64_t chunk = std::min(elems_remaining, tokens_per_block - cur_offset);

            // Temporary buffers for this chunk's quantization
            std::vector<uint8_t> raw_indices((size_t)chunk);
            std::vector<float> chunk_data((size_t)chunk);
            for (int64_t i = 0; i < chunk; i++)
                chunk_data[i] = ksrc[src_pos + i];

            std::vector<uint16_t> cb(QUANT4_CODEBOOK_SIZE);
            lloyd_max_16(chunk_data.data(), chunk, cb.data(), raw_indices.data());

            // Pack indices into 4-bit
            std::vector<uint8_t> packed((size_t)((chunk + 1) / 2));
            pack_indices(raw_indices.data(), packed.data(), chunk);

            // Copy packed indices
            int64_t idx_bytes_offset = cur_block_idx * index_bytes_per_block() + cur_offset / 2;
            int64_t packed_size = (int64_t)packed.size();
            if (idx_bytes_offset + packed_size > (int64_t)c.k_indices.size())
                packed_size = (int64_t)c.k_indices.size() - idx_bytes_offset;
            if (packed_size > 0)
                std::memcpy(c.k_indices.data() + idx_bytes_offset, packed.data(), (size_t)packed_size);

            // Store codebook (only for full blocks or if it's a new block)
            if (cur_offset == 0 || c.current_pos == 0) {
                int64_t cb_offset = cur_block_idx * QUANT4_CODEBOOK_SIZE;
                std::memcpy(c.k_codebooks.data() + cb_offset, cb.data(),
                            QUANT4_CODEBOOK_SIZE * sizeof(uint16_t));
            }

            // V cache: same quantization
            for (int64_t i = 0; i < chunk; i++)
                chunk_data[i] = vsrc[src_pos + i];
            lloyd_max_16(chunk_data.data(), chunk, cb.data(), raw_indices.data());
            pack_indices(raw_indices.data(), packed.data(), chunk);

            if (idx_bytes_offset + packed_size > (int64_t)c.v_indices.size())
                packed_size = (int64_t)c.v_indices.size() - idx_bytes_offset;
            if (packed_size > 0)
                std::memcpy(c.v_indices.data() + idx_bytes_offset, packed.data(), (size_t)packed_size);

            int64_t cb_offset = cur_block_idx * QUANT4_CODEBOOK_SIZE;
            std::memcpy(c.v_codebooks.data() + cb_offset, cb.data(),
                        QUANT4_CODEBOOK_SIZE * sizeof(uint16_t));

            elems_remaining -= chunk;
            src_pos += chunk;
            flat_pos += chunk;
        }
    }

    c.current_pos += (int)seq_len;
}

std::pair<Tensor, Tensor> QUANT4KVCache::get_range(int layer, int64_t start,
                                                    int64_t end) const {
    if (layer >= num_layers_) return {};
    const auto& c = caches_[layer];

    int64_t len = end - start;
    if (len <= 0) return {};
    if (start + len > c.current_pos) len = c.current_pos - start;
    if (len <= 0) return {};

    int64_t h = num_heads_;
    int64_t d = head_dim_;
    int64_t per_token = h * d;
    int64_t tokens_per_block = elements_per_block();

    Tensor k_out(Shape{1, h, len, d});
    Tensor v_out(Shape{1, h, len, d});
    float* kdst = k_out.data<float>();
    float* vdst = v_out.data<float>();

    for (int64_t s = 0; s < len; s++) {
        int64_t flat_start = (start + s) * per_token;
        int64_t dst_offset = s * per_token;

        int64_t elems_remaining = per_token;
        int64_t flat_pos = flat_start;
        int64_t dst_pos = dst_offset;

        while (elems_remaining > 0) {
            int64_t cur_block_idx = flat_pos / tokens_per_block;
            int64_t cur_offset = flat_pos % tokens_per_block;
            int64_t chunk = std::min(elems_remaining, tokens_per_block - cur_offset);

            // Unpack 4-bit indices
            std::vector<uint8_t> raw_indices((size_t)chunk);
            int64_t idx_bytes_offset = cur_block_idx * index_bytes_per_block() + cur_offset / 2;
            int64_t packed_bytes = (chunk + 1) / 2;
            std::vector<uint8_t> packed((size_t)packed_bytes);
            int64_t avail = (int64_t)c.k_indices.size() - idx_bytes_offset;
            int64_t to_copy = std::min(packed_bytes, avail);
            if (to_copy > 0)
                std::memcpy(packed.data(), c.k_indices.data() + idx_bytes_offset, (size_t)to_copy);
            unpack_indices(packed.data(), raw_indices.data(), chunk);

            // Get codebook
            int64_t cb_offset = cur_block_idx * QUANT4_CODEBOOK_SIZE;
            std::vector<float> deq((size_t)chunk);
            dequantize_16(raw_indices.data(), c.k_codebooks.data() + cb_offset,
                          deq.data(), chunk);
            std::memcpy(kdst + dst_pos, deq.data(), (size_t)chunk * sizeof(float));

            // V
            std::fill(packed.begin(), packed.end(), 0);
            avail = (int64_t)c.v_indices.size() - idx_bytes_offset;
            to_copy = std::min(packed_bytes, avail);
            if (to_copy > 0)
                std::memcpy(packed.data(), c.v_indices.data() + idx_bytes_offset, (size_t)to_copy);
            unpack_indices(packed.data(), raw_indices.data(), chunk);
            dequantize_16(raw_indices.data(), c.v_codebooks.data() + cb_offset,
                          deq.data(), chunk);
            std::memcpy(vdst + dst_pos, deq.data(), (size_t)chunk * sizeof(float));

            elems_remaining -= chunk;
            flat_pos += chunk;
            dst_pos += chunk;
        }
    }

    return {k_out, v_out};
}

std::pair<Tensor, Tensor> QUANT4KVCache::get_all(int layer) const {
    return get_range(layer, 0, caches_[layer].current_pos);
}

int QUANT4KVCache::context_len() const {
    return caches_.empty() ? 0 : caches_[0].current_pos;
}

size_t QUANT4KVCache::size_bytes() const {
    size_t total = 0;
    for (auto& c : caches_) {
        total += c.k_indices.size() + c.v_indices.size();
        total += c.k_codebooks.size() * sizeof(uint16_t);
        total += c.v_codebooks.size() * sizeof(uint16_t);
    }
    return total;
}

size_t QUANT4KVCache::size_bytes_fp32_equivalent() const {
    int64_t per_token = num_heads_ * head_dim_;
    int64_t tokens = caches_.empty() ? 0 : caches_[0].current_pos;
    return (size_t)(tokens * per_token * 2 * sizeof(float));
}

double QUANT4KVCache::compression_ratio() const {
    size_t fp32_eq = size_bytes_fp32_equivalent();
    size_t quant4 = size_bytes();
    if (quant4 == 0) return 0.0;
    return (double)fp32_eq / (double)quant4;
}

void QUANT4KVCache::clear() {
    for (auto& c : caches_) {
        std::fill(c.k_indices.begin(), c.k_indices.end(), 0);
        std::fill(c.v_indices.begin(), c.v_indices.end(), 0);
        std::fill(c.k_codebooks.begin(), c.k_codebooks.end(), 0);
        std::fill(c.v_codebooks.begin(), c.v_codebooks.end(), 0);
        c.current_pos = 0;
    }
}

void QUANT4KVCache::resize(int64_t new_max_seq_len) {
    max_seq_len_ = new_max_seq_len;
    for (int i = 0; i < num_layers_; i++) {
        auto& c = caches_[i];
        int64_t total = total_elements();
        int64_t nblocks = total_blocks();
        c.k_indices.resize((size_t)((total + 1) / 2), 0);
        c.v_indices.resize((size_t)((total + 1) / 2), 0);
        c.k_codebooks.resize((size_t)(nblocks * QUANT4_CODEBOOK_SIZE), 0);
        c.v_codebooks.resize((size_t)(nblocks * QUANT4_CODEBOOK_SIZE), 0);
        c.current_pos = 0;
    }
}

} // namespace quant
