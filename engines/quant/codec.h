#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include "quant/tensor.h"
#include "quant/codebook.h"

namespace quant {
namespace quant8 {

struct CodecConfig {
    int block_size = 256;
    int codebook_size = 256;
    int train_iters = 20;
    bool parallel = true;
};

struct EncodedBlock {
    uint32_t block_id;
    std::string name;
    CodebookQ8 codebook;
    std::vector<uint8_t> indices;
    uint32_t num_weights;
};

class Q8Encoder {
public:
    explicit Q8Encoder(const CodecConfig& cfg = CodecConfig());
    EncodedBlock encode(const float* weights, uint32_t num_weights, uint32_t block_id);
    std::vector<EncodedBlock> encode_tensor(const Tensor& t, const std::string& name);
    CodecConfig config() const;
private:
    CodecConfig config_;
};

class Q8Decoder {
public:
    Q8Decoder();
    Tensor decode(const EncodedBlock& block);
    Tensor decode_blocks(const std::vector<EncodedBlock>& blocks, const Shape& original_shape);
};

} // namespace quant8
} // namespace quant
