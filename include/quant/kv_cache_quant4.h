#pragma once
#include "quant/types.h"
#include "quant/tensor.h"
#include "quant/codebook.h"
#include <vector>
#include <cstdint>

namespace quant {

// ===========================================================================
// Q4KVCache — Q4-quantized KV cache (4-bit indices + per-block FP16 codebook)
//
// Memory layout per block:
//   - indices:  (block_elements + 1) / 2 bytes  (4-bit packed, 2 per byte)
//   - codebook: 16 × FP16 (32 bytes)
//
// Total per-block: ceil(block_elements/2) + 32 bytes
// At block_size=64, head_dim=64: per token = (64+32)/64 ≈ 1.5 bytes
// Compared to FP32: 64*4 = 256 bytes per token → ~170× compression per head
//
// Block sizes: 32, 64, or 128 elements per codebook partition.
// Larger blocks → better compression, slightly worse quality.
// ===========================================================================

class Q4KVCache {
public:
    Q4KVCache() = default;
    Q4KVCache(int num_layers, int64_t max_seq_len, int64_t num_heads,
                int64_t head_dim, int64_t block_size = 64);

    void init(int num_layers, int64_t max_seq_len, int64_t num_heads,
              int64_t head_dim, int64_t block_size = 64);

    void append(int layer, const Tensor& k, const Tensor& v);
    std::pair<Tensor, Tensor> get_range(int layer, int64_t start, int64_t end) const;
    std::pair<Tensor, Tensor> get_all(int layer) const;

    int context_len() const;
    int64_t max_seq_len() const { return max_seq_len_; }
    int64_t block_size() const { return block_size_; }
    int num_layers() const { return num_layers_; }
    int64_t num_heads() const { return num_heads_; }
    int64_t head_dim() const { return head_dim_; }

    size_t size_bytes() const;
    size_t size_bytes_fp32_equivalent() const;
    double compression_ratio() const;

    void clear();
    void resize(int64_t new_max_seq_len);

    static void quantize_block_quant4(const float* src, uint8_t* indices,
                                     uint16_t* codebook_fp16, int64_t n);
    static void dequantize_block_quant4(const uint8_t* indices,
                                       const uint16_t* codebook_fp16,
                                       float* dst, int64_t n);

    static constexpr int Q4_CODEBOOK_SIZE = 16;

private:
    struct LayerCache {
        std::vector<uint8_t> k_indices;
        std::vector<uint8_t> v_indices;
        std::vector<uint16_t> k_codebooks;
        std::vector<uint16_t> v_codebooks;
        int current_pos = 0;
    };

    std::vector<LayerCache> caches_;
    int num_layers_ = 0;
    int64_t max_seq_len_ = 0;
    int64_t num_heads_ = 0;
    int64_t head_dim_ = 0;
    int64_t block_size_ = 64;

    int64_t elements_per_block() const;
    int64_t indices_per_token() const;
    int64_t total_elements() const;
    int64_t total_blocks() const;
    int64_t codebook_bytes_per_block() const;
    int64_t index_bytes_per_block() const;
    int64_t bytes_per_block() const;
};

} // namespace quant
